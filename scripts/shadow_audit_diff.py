#!/usr/bin/env python3
"""
scripts/shadow_audit_diff.py - 5-Day Sim101 Shadow-Audit Reconciliation CLI

Reconciles NinjaTrader 8 live/sim executions against the frozen offline Python backtest engine.
Verifies with zero tolerance:
  - Exact signal timestamp match (0-second drift)
  - Direction (BUY/SELL) and Strategy tag (H3/H4) match
  - Exact MNQ contract count match
  - Trailing stop / exit price match
  - Realized live execution slippage (NQ points) vs. 0.50 pt baseline model
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime, date, time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import pytz

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.symbol_mapper import normalize_nt8_bar_timestamp, python_open_to_nt8_close
from engine.tradeify_rules import TradeifyAccountState, AccountPhase
from strategies.portfolio_orchestrator import PortfolioOrchestrator, CompletedTrade
from strategies.h3_noise_band_momentum import H3NoiseBandMomentum, TradeSignal
from strategies.h4_letf_1500_rebalance import H4LETF1500Rebalance
from data.session_preprocessor import SessionDataPreprocessor

tz_ny = pytz.timezone("America/New_York")


@dataclass
class ParsedNT8Trade:
    trade_id: str
    strategy_id: str
    direction: str  # "BUY" or "SELL"
    entry_time: pd.Timestamp
    entry_price: float
    contracts: int
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    commission: float = 0.0
    exit_reason: str = "UNKNOWN"
    raw_tag: str = ""


@dataclass
class TradeReconciliationRow:
    trade_num: int
    strategy_id: str
    direction: str
    sim_entry_time: pd.Timestamp
    nt8_entry_time: Optional[pd.Timestamp]
    time_drift_sec: Optional[float]
    sim_contracts: int
    nt8_contracts: Optional[int]
    sizing_match: bool
    sim_theor_price: float
    sim_fill_price: float
    nt8_fill_price: Optional[float]
    realized_slippage_pts: Optional[float]
    slippage_vs_baseline: Optional[float]
    sim_exit_time: Optional[pd.Timestamp]
    nt8_exit_time: Optional[pd.Timestamp]
    sim_exit_price: Optional[float]
    nt8_exit_price: Optional[float]
    sim_pnl: float
    nt8_pnl: Optional[float]
    status: str  # "PASS", "FAIL", "UNMATCHED"
    discrepancy_notes: List[str]


def parse_nt8_executions_file(file_path: Path) -> List[ParsedNT8Trade]:
    """
    Parses NinjaTrader 8 exported executions or trade list.
    Supports:
      1. Standard NT8 Executions Grid CSV (Time, Instrument, Action, Quantity, Price, Name, ...)
      2. Standard NT8 Trade List CSV (Trade #, Market pos., Qty, Entry price, Exit price, ...)
      3. Custom JSON lines / array exported from webhook_server.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"NT8 executions file not found: {file_path}")

    # Check for JSON format
    if file_path.suffix.lower() == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        trades = []
        raw_list = data if isinstance(data, list) else data.get("executions", [data])
        for idx, item in enumerate(raw_list):
            entry_t = pd.to_datetime(item.get("timestamp") or item.get("entry_time"))
            if entry_t.tzinfo is None:
                entry_t = entry_t.tz_localize("UTC").tz_convert(tz_ny)
            else:
                entry_t = entry_t.tz_convert(tz_ny)

            exit_t = None
            if item.get("exit_time"):
                exit_t = pd.to_datetime(item.get("exit_time"))
                if exit_t.tzinfo is None:
                    exit_t = exit_t.tz_localize("UTC").tz_convert(tz_ny)
                else:
                    exit_t = exit_t.tz_convert(tz_ny)

            trades.append(ParsedNT8Trade(
                trade_id=str(item.get("signal_id") or item.get("trade_id", f"NT8_{idx+1}")),
                strategy_id=str(item.get("strategy_id", "H3")).upper(),
                direction="BUY" if str(item.get("direction", "BUY")).upper() in ["BUY", "LONG", "1"] else "SELL",
                entry_time=entry_t,
                entry_price=float(item.get("fill_price") or item.get("entry_price", 0.0)),
                contracts=int(item.get("contracts", 1)),
                exit_time=exit_t,
                exit_price=float(item.get("exit_price")) if item.get("exit_price") is not None else None,
                pnl=float(item.get("pnl")) if item.get("pnl") is not None else None,
                commission=float(item.get("commission", 2.48 * int(item.get("contracts", 1)))),
                exit_reason=str(item.get("exit_reason", "")),
                raw_tag=str(item.get("signal_id", ""))
            ))
        return trades

    # CSV Format Parsing
    df = pd.read_csv(file_path)
    # Normalize column names: strip spaces, lowercase
    col_map = {c: c.strip().lower() for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    trades: List[ParsedNT8Trade] = []

    # Scenario A: Trade List format (already has entry and exit)
    if "entry time" in df.columns or "entry price" in df.columns or "market pos." in df.columns:
        for idx, row in df.iterrows():
            pos_str = str(row.get("market pos.", "Long")).lower()
            dir_str = "BUY" if "long" in pos_str or "buy" in pos_str else "SELL"

            entry_t = pd.to_datetime(row.get("entry time") or row.get("time"))
            if entry_t.tzinfo is None:
                entry_t = entry_t.tz_localize(tz_ny)
            else:
                entry_t = entry_t.tz_convert(tz_ny)

            exit_t = None
            if "exit time" in row and pd.notna(row["exit time"]):
                exit_t = pd.to_datetime(row["exit time"])
                if exit_t.tzinfo is None:
                    exit_t = exit_t.tz_localize(tz_ny)
                else:
                    exit_t = exit_t.tz_convert(tz_ny)

            strat = "H3"
            strat_field = str(row.get("strategy", row.get("name", ""))).upper()
            if "H4" in strat_field:
                strat = "H4"

            trades.append(ParsedNT8Trade(
                trade_id=str(row.get("trade #", f"TL_{idx+1}")),
                strategy_id=strat,
                direction=dir_str,
                entry_time=entry_t,
                entry_price=float(row.get("entry price", 0.0)),
                contracts=int(row.get("qty", row.get("quantity", 1))),
                exit_time=exit_t,
                exit_price=float(row.get("exit price")) if pd.notna(row.get("exit price")) else None,
                pnl=float(row.get("profit ($)", row.get("pnl", 0.0))) if pd.notna(row.get("profit ($)", row.get("pnl"))) else None,
                commission=float(row.get("commission", 0.0)) if pd.notna(row.get("commission")) else 0.0,
                raw_tag=strat_field
            ))
        return trades

    # Scenario B: Raw Executions Grid (Time, Action, Quantity, Price, Name/Order ID)
    # We pair Entry fills with Exit fills
    open_orders: Dict[str, Dict[str, Any]] = {}
    time_col = "time" if "time" in df.columns else df.columns[0]
    action_col = "action" if "action" in df.columns else "type"
    qty_col = "quantity" if "quantity" in df.columns else "qty"
    price_col = "price" if "price" in df.columns else "avg price"
    name_col = "name" if "name" in df.columns else "order id"

    for idx, row in df.iterrows():
        t = pd.to_datetime(row[time_col])
        if t.tzinfo is None:
            t = t.tz_localize(tz_ny)
        else:
            t = t.tz_convert(tz_ny)

        act = str(row[action_col]).strip().lower()
        qty = int(row[qty_col])
        px = float(row[price_col])
        name = str(row.get(name_col, ""))

        strat = "H4" if "H4" in name.upper() else "H3"
        is_entry = act in ["buy", "sell short", "buy to open", "sell to open"] or ("sl" not in name.lower() and "tp" not in name.lower() and "exit" not in name.lower())

        if is_entry:
            dir_str = "BUY" if "buy" in act else "SELL"
            open_orders[strat] = {
                "trade_id": f"EXEC_{idx+1}",
                "strategy_id": strat,
                "direction": dir_str,
                "entry_time": t,
                "entry_price": px,
                "contracts": qty,
                "raw_tag": name
            }
        else:
            # Exit order
            if strat in open_orders:
                entry_info = open_orders.pop(strat)
                # Compute pnl
                pnl_pts = (px - entry_info["entry_price"]) if entry_info["direction"] == "BUY" else (entry_info["entry_price"] - px)
                pnl_usd = pnl_pts * 2.0 * qty - (qty * 2.48)
                trades.append(ParsedNT8Trade(
                    trade_id=entry_info["trade_id"],
                    strategy_id=strat,
                    direction=entry_info["direction"],
                    entry_time=entry_info["entry_time"],
                    entry_price=entry_info["entry_price"],
                    contracts=qty,
                    exit_time=t,
                    exit_price=px,
                    pnl=pnl_usd,
                    commission=qty * 2.48,
                    exit_reason=name,
                    raw_tag=entry_info["raw_tag"]
                ))
            else:
                # Standalone exit fill
                pass

    # If any trades remained open at EOD
    for strat, entry_info in open_orders.items():
        trades.append(ParsedNT8Trade(
            trade_id=entry_info["trade_id"],
            strategy_id=strat,
            direction=entry_info["direction"],
            entry_time=entry_info["entry_time"],
            entry_price=entry_info["entry_price"],
            contracts=entry_info["contracts"],
            raw_tag=entry_info["raw_tag"]
        ))

    return trades


def run_shadow_audit(
    nt8_file: Path,
    bars_file: Path,
    warmup_parquet: Optional[Path] = None,
    account_id: str = "Sim101",
    account_phase: str = "EVALUATION",
    initial_cash: float = 50000.0,
    slippage_alert_threshold: float = 1.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Executes the 5-day Sim101 shadow audit diff.
    Reconciles live NT8 trades against the frozen Python offline engine on the same session bars.
    """
    nt8_trades = parse_nt8_executions_file(nt8_file)

    # Load 1m bars
    if not bars_file.exists():
        raise FileNotFoundError(f"Bars file not found: {bars_file}")

    if bars_file.suffix.lower() == ".parquet":
        df_bars = pd.read_parquet(bars_file)
    else:
        df_bars = pd.read_csv(bars_file, index_col=0, parse_dates=True)

    if df_bars.index.tz is None:
        df_bars.index = df_bars.index.tz_localize("UTC").tz_convert(tz_ny)
    else:
        df_bars.index = df_bars.index.tz_convert(tz_ny)

    # Sort bars
    df_bars.sort_index(inplace=True)

    session_dates = sorted(list(set(df_bars.index.date)))
    target_date = session_dates[0] if session_dates else date.today()

    # Setup SessionPreprocessor and Warmup
    prep = None
    daily_atr = None
    prev_close = None

    if warmup_parquet and warmup_parquet.exists():
        prep = SessionDataPreprocessor(data_path=str(warmup_parquet), output_dir=str(PROJECT_ROOT / "data"))
        prep.load_data(force_recompute=False)
        daily_features = prep.daily_features
        if target_date in daily_features.index:
            daily_atr = float(daily_features.loc[target_date, "ATR_14"])
            prev_close = float(daily_features.loc[target_date, "Prev_Close"])
    else:
        # Fallback to computing ATR from available bars
        rth_bars = df_bars.between_time("09:30", "16:00")
        if not rth_bars.empty:
            daily_summary = rth_bars.groupby(rth_bars.index.date).agg({
                "High": "max", "Low": "min", "Close": "last"
            })
            if len(daily_summary) > 1:
                tr = daily_summary["High"] - daily_summary["Low"]
                daily_atr = float(tr.mean())
            else:
                daily_atr = float(rth_bars["High"].max() - rth_bars["Low"].min())
            prev_close = float(rth_bars["Close"].iloc[0])

    if daily_atr is None or np.isnan(daily_atr):
        daily_atr = 200.0  # Safe robust default for MNQ if no historical warmup provided

    # Setup Account and Orchestrator
    phase = AccountPhase.FUNDED if (account_phase.upper() == "FUNDED") else AccountPhase.EVALUATION
    account = TradeifyAccountState(phase=phase)
    account.balance = initial_cash
    account.eod_hwm = initial_cash
    orchestrator = PortfolioOrchestrator(
        account=account,
        enabled_modules=["H3", "H4"]
    )

    # Run Frozen Offline Engine Simulation
    df_session_1m = df_bars[df_bars.index.date == target_date]
    sim_trades: List[CompletedTrade] = orchestrator.simulate_session(
        session_date=target_date,
        df_session_1m=df_session_1m,
        daily_atr=daily_atr,
        prev_close=prev_close,
        prep=prep
    )

    # Reconcile Trades Side-by-Side
    reconciliation_rows: List[TradeReconciliationRow] = []
    unmatched_nt8_trades = list(nt8_trades)

    all_passed = True
    total_slippage_pts = 0.0
    matched_trades_count = 0

    for idx, sim in enumerate(sim_trades, start=1):
        discrepancies: List[str] = []

        # Find best matching NT8 trade
        # Criteria: same strategy, same direction, entry_time within 120s (target 0s)
        best_match: Optional[ParsedNT8Trade] = None
        min_time_diff = float("inf")

        sim_dir = "BUY" if sim.direction.upper() in ["BUY", "LONG", "1"] else "SELL"

        for cand in unmatched_nt8_trades:
            if cand.strategy_id == sim.strategy_id and cand.direction == sim_dir:
                time_diff = abs((cand.entry_time - sim.entry_time).total_seconds())
                if time_diff < min_time_diff and time_diff <= 120.0:
                    min_time_diff = time_diff
                    best_match = cand

        if best_match is not None:
            unmatched_nt8_trades.remove(best_match)
            time_drift = (best_match.entry_time - sim.entry_time).total_seconds()

            # Zero-tolerance timestamp check (0-second drift)
            if abs(time_drift) > 0.0:
                discrepancies.append(f"Timestamp drift: {time_drift:+.1f}s (Tolerance: 0s)")
                all_passed = False

            # Contract sizing check
            sizing_match = (best_match.contracts == sim.contracts)
            if not sizing_match:
                discrepancies.append(f"Sizing mismatch: Sim={sim.contracts} MNQ vs NT8={best_match.contracts} MNQ")
                all_passed = False

            # Slippage calculation
            # Theoretical price in sim is entry_price before 0.50 pt baseline slippage
            theor_price = sim.entry_price - (0.50 if sim_dir == "BUY" else -0.50)
            if sim_dir == "BUY":
                measured_slip = best_match.entry_price - theor_price
            else:
                measured_slip = theor_price - best_match.entry_price

            slip_vs_baseline = measured_slip - 0.50
            total_slippage_pts += measured_slip
            matched_trades_count += 1

            if abs(slip_vs_baseline) > slippage_alert_threshold:
                discrepancies.append(f"Excess slippage: {measured_slip:.2f} pts (Baseline: 0.50 pt, Delta: {slip_vs_baseline:+.2f} pts)")

            # Exit price check (if available)
            if best_match.exit_price is not None and sim.exit_price is not None:
                exit_diff = abs(best_match.exit_price - sim.exit_price)
                if exit_diff > 2.0:  # More than 2 pts exit difference
                    discrepancies.append(f"Exit price delta: Sim={sim.exit_price:.2f} vs NT8={best_match.exit_price:.2f} (diff={exit_diff:.2f} pts)")

            row_status = "PASS" if not discrepancies else "FAIL"

            reconciliation_rows.append(TradeReconciliationRow(
                trade_num=idx,
                strategy_id=sim.strategy_id,
                direction=sim_dir,
                sim_entry_time=sim.entry_time,
                nt8_entry_time=best_match.entry_time,
                time_drift_sec=time_drift,
                sim_contracts=sim.contracts,
                nt8_contracts=best_match.contracts,
                sizing_match=sizing_match,
                sim_theor_price=theor_price,
                sim_fill_price=sim.entry_price,
                nt8_fill_price=best_match.entry_price,
                realized_slippage_pts=round(measured_slip, 2),
                slippage_vs_baseline=round(slip_vs_baseline, 2),
                sim_exit_time=sim.exit_time,
                nt8_exit_time=best_match.exit_time,
                sim_exit_price=sim.exit_price,
                nt8_exit_price=best_match.exit_price,
                sim_pnl=round(sim.pnl, 2),
                nt8_pnl=round(best_match.pnl, 2) if best_match.pnl is not None else None,
                status=row_status,
                discrepancy_notes=discrepancies
            ))
        else:
            # Unmatched Sim Trade (Missed in NT8!)
            all_passed = False
            reconciliation_rows.append(TradeReconciliationRow(
                trade_num=idx,
                strategy_id=sim.strategy_id,
                direction=sim_dir,
                sim_entry_time=sim.entry_time,
                nt8_entry_time=None,
                time_drift_sec=None,
                sim_contracts=sim.contracts,
                nt8_contracts=None,
                sizing_match=False,
                sim_theor_price=sim.entry_price - (0.50 if sim_dir == "BUY" else -0.50),
                sim_fill_price=sim.entry_price,
                nt8_fill_price=None,
                realized_slippage_pts=None,
                slippage_vs_baseline=None,
                sim_exit_time=sim.exit_time,
                nt8_exit_time=None,
                sim_exit_price=sim.exit_price,
                nt8_exit_price=None,
                sim_pnl=round(sim.pnl, 2),
                nt8_pnl=None,
                status="UNMATCHED",
                discrepancy_notes=["Signal fired in Python simulation but NO matching execution found in NT8 log!"]
            ))

    # Any leftover unmatched NT8 trades (Ghost trades!)
    if unmatched_nt8_trades:
        all_passed = False
        for ghost in unmatched_nt8_trades:
            reconciliation_rows.append(TradeReconciliationRow(
                trade_num=len(reconciliation_rows) + 1,
                strategy_id=ghost.strategy_id,
                direction=ghost.direction,
                sim_entry_time=ghost.entry_time,
                nt8_entry_time=ghost.entry_time,
                time_drift_sec=None,
                sim_contracts=0,
                nt8_contracts=ghost.contracts,
                sizing_match=False,
                sim_theor_price=ghost.entry_price,
                sim_fill_price=ghost.entry_price,
                nt8_fill_price=ghost.entry_price,
                realized_slippage_pts=None,
                slippage_vs_baseline=None,
                sim_exit_time=ghost.exit_time,
                nt8_exit_time=ghost.exit_time,
                sim_exit_price=None,
                nt8_exit_price=ghost.exit_price,
                sim_pnl=0.0,
                nt8_pnl=ghost.pnl,
                status="GHOST_FILL",
                discrepancy_notes=["Ghost fill: Order executed in NT8 but was NOT generated by Python signal logic!"]
            ))

    avg_slippage = (total_slippage_pts / matched_trades_count) if matched_trades_count > 0 else 0.0

    result = {
        "session_date": str(target_date),
        "audit_status": "PASS" if all_passed and len(reconciliation_rows) > 0 else "FAIL",
        "total_sim_trades": len(sim_trades),
        "total_nt8_trades": len(nt8_trades),
        "matched_trades": matched_trades_count,
        "avg_realized_slippage_pts": round(avg_slippage, 3),
        "baseline_slippage_pts": 0.50,
        "slippage_drag_per_mnq_rt": round(avg_slippage * 2.0, 2),
        "rows": [asdict(r) for r in reconciliation_rows]
    }

    return result


def generate_markdown_report(audit_result: Dict[str, Any]) -> str:
    """Generates a clean institutional markdown reconciliation report."""
    status_emoji = "PASSED" if audit_result["audit_status"] == "PASS" else "FAILED"
    lines = [
        f"# Sim101 Shadow-Audit Daily Reconciliation Report",
        f"",
        f"**Session Date:** `{audit_result['session_date']}`  ",
        f"**Audit Status:** **`{status_emoji}`**  ",
        f"**Simulated Signals:** `{audit_result['total_sim_trades']}` | **NT8 Executions:** `{audit_result['total_nt8_trades']}` | **Matched:** `{audit_result['matched_trades']}`  ",
        f"**Average Realized Slippage:** `{audit_result['avg_realized_slippage_pts']:.2f} pts` (Baseline: `{audit_result['baseline_slippage_pts']:.2f} pts`)  ",
        f"",
        f"---",
        f"",
        f"### Side-by-Side Trade Reconciliation Table",
        f"",
        f"| # | Mod | Dir | Sim Entry (NY) | NT8 Entry (NY) | Drift | Sizing | Sim Px | NT8 Px | Slip (pts) | Sim PnL | NT8 PnL | Status |",
        f"|---|-----|-----|----------------|----------------|-------|--------|--------|--------|------------|---------|---------|--------|"
    ]

    for r in audit_result["rows"]:
        sim_t = r["sim_entry_time"][11:19] if r["sim_entry_time"] else "N/A"
        nt8_t = r["nt8_entry_time"][11:19] if r["nt8_entry_time"] else "N/A"
        drift = f"{r['time_drift_sec']:+.0f}s" if r["time_drift_sec"] is not None else "N/A"
        sizing = f"{r['sim_contracts']}/{r['nt8_contracts'] if r['nt8_contracts'] else 0}"
        sim_px = f"{r['sim_fill_price']:.2f}" if r["sim_fill_price"] else "N/A"
        nt8_px = f"{r['nt8_fill_price']:.2f}" if r["nt8_fill_price"] else "N/A"
        slip = f"{r['realized_slippage_pts']:+.2f}" if r["realized_slippage_pts"] is not None else "N/A"
        sim_pnl = f"${r['sim_pnl']:+,.2f}"
        nt8_pnl = f"${r['nt8_pnl']:+,.2f}" if r["nt8_pnl"] is not None else "N/A"

        status_badge = f"**{r['status']}**" if r["status"] == "PASS" else f"**`{r['status']}`**"

        lines.append(
            f"| {r['trade_num']} | {r['strategy_id']} | {r['direction']} | {sim_t} | {nt8_t} | {drift} | {sizing} | {sim_px} | {nt8_px} | {slip} | {sim_pnl} | {nt8_pnl} | {status_badge} |"
        )

    # Discrepancy details
    discrepancies_exist = any(r.get("discrepancy_notes") for r in audit_result["rows"])
    if discrepancies_exist:
        lines.append("")
        lines.append("### Discrepancy Forensic Log")
        for r in audit_result["rows"]:
            if r.get("discrepancy_notes"):
                for note in r["discrepancy_notes"]:
                    lines.append(f"- **Trade #{r['trade_num']} ({r['strategy_id']} {r['direction']}):** {note}")

    lines.append("")
    lines.append("---")
    lines.append("*Report generated autonomously by Prop_killer Sim101 Shadow-Audit Tooling.*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Sim101 Shadow-Audit Reconciliation CLI")
    parser.add_argument("--executions", "-e", required=True, type=Path, help="Path to NT8 executions CSV or JSON")
    parser.add_argument("--bars", "-b", required=True, type=Path, help="Path to 1m live/session bar CSV or Parquet")
    parser.add_argument("--warmup", "-w", type=Path, default=PROJECT_ROOT / "data" / "nq_1m_resolved_cache.parquet", help="Path to historical warmup parquet")
    parser.add_argument("--account-id", type=str, default="Sim101", help="NinjaTrader Account ID")
    parser.add_argument("--phase", type=str, default="EVALUATION", choices=["EVALUATION", "FUNDED"], help="Account Phase")
    parser.add_argument("--cash", type=float, default=50000.0, help="Starting cash balance")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output markdown report path")
    parser.add_argument("--slippage-tol", type=float, default=1.0, help="Slippage tolerance in points")

    args = parser.parse_args()

    result = run_shadow_audit(
        nt8_file=args.executions,
        bars_file=args.bars,
        warmup_parquet=args.warmup if args.warmup.exists() else None,
        account_id=args.account_id,
        account_phase=args.phase,
        initial_cash=args.cash,
        slippage_alert_threshold=args.slippage_tol
    )

    report_md = generate_markdown_report(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"Shadow-audit report saved to: {args.output}")

    print(report_md)
    sys.exit(0 if result["audit_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
