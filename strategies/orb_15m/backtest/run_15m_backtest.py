"""
Institutional 15-Minute ORB Backtester — High Capital Velocity Overhaul
Target: CME Micro E-mini Nasdaq-100 (MNQ) Futures
Prop Firm: Tradeify $50K EOD Trailing Drawdown ($2,000 Barrier, $3,000 Target)

Architecture Overhaul:
  Task 1: Dynamic Contract Sizing & Natural Stop
    - Sizing: MNQ = max(2, 2 * floor((Remaining DD * 0.085) / (Stop Loss Points * $2.00 * 2)))
    - Evaluation Risk: 8.5% of Remaining DD (min $140, max $200 per trade).
    - Natural structural stop: OR_Mid, capped at 40.0 pts catastrophic threshold.
  Task 2: Payoff Asymmetry & Primary Breakout Focus
    - Leg 1 (50%): TP1 = Entry ± 1.25R.
    - Ratchet on TP1: Immediate stop ratchet to Entry ± 2 ticks (0.50 pts).
    - Leg 2 (50% runner): TP2 = Entry ± 2.50R.
    - Profit Lock: On +2.0R expansion, ratchet Leg 2 SL to +1.0R.
    - Secondary Retest logic deleted. 100% focus on Primary Breakouts.
  Task 3: Session Filter Rebalance
    - ATR Exhaustion: max_or_atr_ratio = 0.55 (unblocks trend sessions).
    - Noise Floor: min_or_points = 8.0 pts.
    - Surgical News Filter: blackout window ±2 minutes (CPI, PPI, NFP, FOMC).
  Task 4: First-Passage Monte Carlo & Velocity Scorecard
    - 5,000 bootstrap simulations of Tradeify $50K challenge attempts.
    - Terminal States: Target $53,000 (PASS) vs Floor $48,000 EOD (FAIL).
    - Velocity Scorecard: Expectancy, PF, Pass Rate, Days to Pass, EV Velocity ($EV/d).
"""

from __future__ import annotations
import sys
from pathlib import Path
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# -----------------------------------------------------------------------
# Path resolution
# -----------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
STRATEGY_DIR = SCRIPT_DIR.parent
REPO_ROOT = STRATEGY_DIR.parent.parent

for p in [str(REPO_ROOT), str(STRATEGY_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from core.data_loader import load_continuous_nq_parquet, extract_daily_trading_sessions
from core.risk_state_machine import RiskStateMachine, AccountType
from strategies.orb_15m.src.strategy_15m import Institutional15mORBStrategy

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ORB15m_Velocity")


# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
PV = 2.0          # MNQ point value ($2/pt)
COMM = 1.24       # Commission per contract
SLIP = 0.50       # Slippage in points
BE_TICKS = 2
TICK_SIZE = 0.25
BE_OFFSET = BE_TICKS * TICK_SIZE  # 0.50 pts


# -----------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------
class TradeOutcome:
    STOP_LOSS = "STOP_LOSS"
    PESSIMISTIC_STOP = "PESSIMISTIC_STOP"
    SCALE_OUT_BE = "SCALE_OUT_BE"
    SCALE_OUT_PROFIT_LOCK = "SCALE_OUT_PROFIT_LOCK_1R"
    SCALE_OUT_TP = "SCALE_OUT_TP"
    TIME_EXIT = "TIME_EXIT_1545"


@dataclass
class TwoTierTrade:
    trade_id: str
    direction: int            # 1=LONG, -1=SHORT
    entry_price: float
    initial_sl: float         # Natural structural stop: OR_Mid (capped at 40.0 pts)
    tp1: float                # +1.25R
    tp2: float                # +2.50R
    contracts: int = 4        # Dynamic sizing (4 to 6 MNQ standard)
    leg1_contracts: int = 2
    leg2_contracts: int = 2
    initial_risk: float = 0.0
    entry_time: Optional[pd.Timestamp] = None
    trigger_type: str = "PRIMARY_BREAKOUT"

    # Pre-computed execution levels
    be_level: float = 0.0             # Entry ± 2 ticks (0.50 pts)
    profit_lock_trigger: float = 0.0  # Entry ± 2.0R
    profit_lock_level: float = 0.0    # Entry ± 1.0R

    # Dynamic state
    leg1_filled: bool = False
    leg1_exit_price: float = 0.0
    current_sl: float = 0.0
    be_triggered: bool = False
    profit_lock_triggered: bool = False

    # Outcome
    outcome: str = ""
    leg1_outcome: str = ""
    leg2_outcome: str = ""
    exit_time: Optional[pd.Timestamp] = None
    realized_pnl: float = 0.0

    def __post_init__(self):
        self.initial_risk = abs(self.entry_price - self.initial_sl)
        self.current_sl = self.initial_sl


# -----------------------------------------------------------------------
# PnL helper
# -----------------------------------------------------------------------
def _pnl(entry: float, exit_p: float, direction: int, contracts: int) -> float:
    """Net PnL for one leg including commission."""
    gross = (exit_p - entry) * direction * PV * contracts
    return gross - (COMM * contracts)


# -----------------------------------------------------------------------
# Two-Tier Pessimistic Execution Simulator
# -----------------------------------------------------------------------
def simulate_bar(trade: TwoTierTrade, bar: pd.Series) -> Optional[TwoTierTrade]:
    """
    Evaluate a 1-minute OHLC bar against an active two-tier trade.
    Enforces pessimistic execution: SL + TP on same bar -> always stop loss.
    The instant TP1 fills at +1.25R, ratchets Leg 2 SL to Entry ± 2 ticks.
    If Leg 2 reaches +2.0R unrealized, ratchets SL to +1.0R.
    """
    high = float(bar["High"])
    low = float(bar["Low"])
    bt = bar.name
    d = trade.direction
    e = trade.entry_price

    # =================================================================
    # LONG
    # =================================================================
    if d == 1:
        if not trade.leg1_filled:
            sl_hit = low <= trade.current_sl
            tp1_hit = high >= trade.tp1

            # Pessimistic: SL + TP1 on same bar -> full stop with slippage
            if sl_hit and tp1_hit:
                ex = trade.current_sl - SLIP
                trade.outcome = TradeOutcome.PESSIMISTIC_STOP
                trade.leg1_outcome = trade.leg2_outcome = "STOP_LOSS"
                trade.realized_pnl = _pnl(e, ex, 1, trade.contracts)
                trade.exit_time = bt
                return trade

            if sl_hit:
                ex = trade.current_sl - SLIP
                trade.outcome = TradeOutcome.STOP_LOSS
                trade.leg1_outcome = trade.leg2_outcome = "STOP_LOSS"
                trade.realized_pnl = _pnl(e, ex, 1, trade.contracts)
                trade.exit_time = bt
                return trade

            if tp1_hit:
                # Leg 1 fills at TP1 (+1.25R)
                trade.leg1_filled = True
                trade.leg1_exit_price = trade.tp1
                trade.leg1_outcome = "TAKE_PROFIT_1"

                # Ratchet Leg 2 SL to BE (Entry + 2 ticks)
                trade.be_triggered = True
                trade.current_sl = trade.be_level

                # Intra-bar check for Leg 2:
                # Did price reverse to BE on same bar?
                if low <= trade.be_level:
                    p1 = _pnl(e, trade.tp1, 1, trade.leg1_contracts)
                    p2 = _pnl(e, trade.be_level, 1, trade.leg2_contracts)
                    trade.realized_pnl = round(p1 + p2, 2)
                    trade.leg2_outcome = "BREAK_EVEN"
                    trade.outcome = TradeOutcome.SCALE_OUT_BE
                    trade.exit_time = bt
                    return trade

                # Did price reach +2.0R profit-lock trigger on this bar?
                if high >= trade.profit_lock_trigger:
                    trade.profit_lock_triggered = True
                    trade.current_sl = trade.profit_lock_level
                    if low <= trade.profit_lock_level:
                        p1 = _pnl(e, trade.tp1, 1, trade.leg1_contracts)
                        p2 = _pnl(e, trade.profit_lock_level, 1, trade.leg2_contracts)
                        trade.realized_pnl = round(p1 + p2, 2)
                        trade.leg2_outcome = "PROFIT_LOCK_1R"
                        trade.outcome = TradeOutcome.SCALE_OUT_PROFIT_LOCK
                        trade.exit_time = bt
                        return trade
                    if high >= trade.tp2:
                        p1 = _pnl(e, trade.tp1, 1, trade.leg1_contracts)
                        p2 = _pnl(e, trade.tp2, 1, trade.leg2_contracts)
                        trade.realized_pnl = round(p1 + p2, 2)
                        trade.leg2_outcome = "TAKE_PROFIT_2"
                        trade.outcome = TradeOutcome.SCALE_OUT_TP
                        trade.exit_time = bt
                        return trade
                else:
                    if high >= trade.tp2:
                        p1 = _pnl(e, trade.tp1, 1, trade.leg1_contracts)
                        p2 = _pnl(e, trade.tp2, 1, trade.leg2_contracts)
                        trade.realized_pnl = round(p1 + p2, 2)
                        trade.leg2_outcome = "TAKE_PROFIT_2"
                        trade.outcome = TradeOutcome.SCALE_OUT_TP
                        trade.exit_time = bt
                        return trade

        else:
            # Leg 1 already filled, Leg 2 running
            if not trade.profit_lock_triggered and high >= trade.profit_lock_trigger:
                trade.profit_lock_triggered = True
                trade.current_sl = trade.profit_lock_level

            sl_hit = low <= trade.current_sl
            tp2_hit = high >= trade.tp2

            if sl_hit and tp2_hit:
                # Pessimistic: stop out Leg 2
                p1 = _pnl(e, trade.leg1_exit_price, 1, trade.leg1_contracts)
                p2 = _pnl(e, trade.current_sl, 1, trade.leg2_contracts)
                trade.realized_pnl = round(p1 + p2, 2)
                trade.leg2_outcome = "PROFIT_LOCK_1R" if trade.profit_lock_triggered else "BREAK_EVEN"
                trade.outcome = TradeOutcome.SCALE_OUT_PROFIT_LOCK if trade.profit_lock_triggered else TradeOutcome.SCALE_OUT_BE
                trade.exit_time = bt
                return trade

            if sl_hit:
                p1 = _pnl(e, trade.leg1_exit_price, 1, trade.leg1_contracts)
                p2 = _pnl(e, trade.current_sl, 1, trade.leg2_contracts)
                trade.realized_pnl = round(p1 + p2, 2)
                trade.leg2_outcome = "PROFIT_LOCK_1R" if trade.profit_lock_triggered else "BREAK_EVEN"
                trade.outcome = TradeOutcome.SCALE_OUT_PROFIT_LOCK if trade.profit_lock_triggered else TradeOutcome.SCALE_OUT_BE
                trade.exit_time = bt
                return trade

            if tp2_hit:
                p1 = _pnl(e, trade.leg1_exit_price, 1, trade.leg1_contracts)
                p2 = _pnl(e, trade.tp2, 1, trade.leg2_contracts)
                trade.realized_pnl = round(p1 + p2, 2)
                trade.leg2_outcome = "TAKE_PROFIT_2"
                trade.outcome = TradeOutcome.SCALE_OUT_TP
                trade.exit_time = bt
                return trade

    # =================================================================
    # SHORT (mirror)
    # =================================================================
    elif d == -1:
        if not trade.leg1_filled:
            sl_hit = high >= trade.current_sl
            tp1_hit = low <= trade.tp1

            if sl_hit and tp1_hit:
                ex = trade.current_sl + SLIP
                trade.outcome = TradeOutcome.PESSIMISTIC_STOP
                trade.leg1_outcome = trade.leg2_outcome = "STOP_LOSS"
                trade.realized_pnl = _pnl(e, ex, -1, trade.contracts)
                trade.exit_time = bt
                return trade

            if sl_hit:
                ex = trade.current_sl + SLIP
                trade.outcome = TradeOutcome.STOP_LOSS
                trade.leg1_outcome = trade.leg2_outcome = "STOP_LOSS"
                trade.realized_pnl = _pnl(e, ex, -1, trade.contracts)
                trade.exit_time = bt
                return trade

            if tp1_hit:
                trade.leg1_filled = True
                trade.leg1_exit_price = trade.tp1
                trade.leg1_outcome = "TAKE_PROFIT_1"

                trade.be_triggered = True
                trade.current_sl = trade.be_level

                if high >= trade.be_level:
                    p1 = _pnl(e, trade.tp1, -1, trade.leg1_contracts)
                    p2 = _pnl(e, trade.be_level, -1, trade.leg2_contracts)
                    trade.realized_pnl = round(p1 + p2, 2)
                    trade.leg2_outcome = "BREAK_EVEN"
                    trade.outcome = TradeOutcome.SCALE_OUT_BE
                    trade.exit_time = bt
                    return trade

                if low <= trade.profit_lock_trigger:
                    trade.profit_lock_triggered = True
                    trade.current_sl = trade.profit_lock_level
                    if high >= trade.profit_lock_level:
                        p1 = _pnl(e, trade.tp1, -1, trade.leg1_contracts)
                        p2 = _pnl(e, trade.profit_lock_level, -1, trade.leg2_contracts)
                        trade.realized_pnl = round(p1 + p2, 2)
                        trade.leg2_outcome = "PROFIT_LOCK_1R"
                        trade.outcome = TradeOutcome.SCALE_OUT_PROFIT_LOCK
                        trade.exit_time = bt
                        return trade
                    if low <= trade.tp2:
                        p1 = _pnl(e, trade.tp1, -1, trade.leg1_contracts)
                        p2 = _pnl(e, trade.tp2, -1, trade.leg2_contracts)
                        trade.realized_pnl = round(p1 + p2, 2)
                        trade.leg2_outcome = "TAKE_PROFIT_2"
                        trade.outcome = TradeOutcome.SCALE_OUT_TP
                        trade.exit_time = bt
                        return trade
                else:
                    if low <= trade.tp2:
                        p1 = _pnl(e, trade.tp1, -1, trade.leg1_contracts)
                        p2 = _pnl(e, trade.tp2, -1, trade.leg2_contracts)
                        trade.realized_pnl = round(p1 + p2, 2)
                        trade.leg2_outcome = "TAKE_PROFIT_2"
                        trade.outcome = TradeOutcome.SCALE_OUT_TP
                        trade.exit_time = bt
                        return trade

        else:
            if not trade.profit_lock_triggered and low <= trade.profit_lock_trigger:
                trade.profit_lock_triggered = True
                trade.current_sl = trade.profit_lock_level

            sl_hit = high >= trade.current_sl
            tp2_hit = low <= trade.tp2

            if sl_hit and tp2_hit:
                p1 = _pnl(e, trade.leg1_exit_price, -1, trade.leg1_contracts)
                p2 = _pnl(e, trade.current_sl, -1, trade.leg2_contracts)
                trade.realized_pnl = round(p1 + p2, 2)
                trade.leg2_outcome = "PROFIT_LOCK_1R" if trade.profit_lock_triggered else "BREAK_EVEN"
                trade.outcome = TradeOutcome.SCALE_OUT_PROFIT_LOCK if trade.profit_lock_triggered else TradeOutcome.SCALE_OUT_BE
                trade.exit_time = bt
                return trade

            if sl_hit:
                p1 = _pnl(e, trade.leg1_exit_price, -1, trade.leg1_contracts)
                p2 = _pnl(e, trade.current_sl, -1, trade.leg2_contracts)
                trade.realized_pnl = round(p1 + p2, 2)
                trade.leg2_outcome = "PROFIT_LOCK_1R" if trade.profit_lock_triggered else "BREAK_EVEN"
                trade.outcome = TradeOutcome.SCALE_OUT_PROFIT_LOCK if trade.profit_lock_triggered else TradeOutcome.SCALE_OUT_BE
                trade.exit_time = bt
                return trade

            if tp2_hit:
                p1 = _pnl(e, trade.leg1_exit_price, -1, trade.leg1_contracts)
                p2 = _pnl(e, trade.tp2, -1, trade.leg2_contracts)
                trade.realized_pnl = round(p1 + p2, 2)
                trade.leg2_outcome = "TAKE_PROFIT_2"
                trade.outcome = TradeOutcome.SCALE_OUT_TP
                trade.exit_time = bt
                return trade

    return None


def time_exit_trade(trade: TwoTierTrade, close_p: float, bar_time) -> TwoTierTrade:
    """Force-close at 15:45 EST at market price."""
    d = trade.direction
    trade.outcome = TradeOutcome.TIME_EXIT
    trade.exit_time = bar_time
    if not trade.leg1_filled:
        trade.realized_pnl = _pnl(trade.entry_price, close_p, d, trade.contracts)
    else:
        p1 = _pnl(trade.entry_price, trade.leg1_exit_price, d, trade.leg1_contracts)
        p2 = _pnl(trade.entry_price, close_p, d, trade.leg2_contracts)
        trade.realized_pnl = round(p1 + p2, 2)
    return trade


# -----------------------------------------------------------------------
# Backtest Engine
# -----------------------------------------------------------------------
class InstitutionalBacktestEngine:
    def __init__(self, config_path: str = "config/strategy_config_15m.yaml"):
        cfg_file = STRATEGY_DIR / config_path
        with open(cfg_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.strategy = Institutional15mORBStrategy(self.config)
        self.risk_machine = RiskStateMachine(starting_balance=50000.0)

        scfg = self.config.get("strategy", {})
        ecfg = scfg.get("execution", {})
        plcfg = ecfg.get("profit_lock", {})

        self.force_close_time = self.strategy.force_close_time
        self.profit_lock_trigger_r = float(plcfg.get("trigger_r", 2.00))
        self.profit_lock_to_r = float(plcfg.get("lock_to_r", 1.00))

    def run_backtest(self, sessions: list[dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Running High-Velocity 15m ORB Backtest across {len(sessions):,} sessions...")

        # Precompute 14-day Daily ATR
        true_ranges = []
        for i, s in enumerate(sessions):
            h, l = s["daily_high"], s["daily_low"]
            if i > 0:
                prev_c = sessions[i - 1]["daily_close"]
                tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            else:
                tr = h - l
            true_ranges.append(tr)

        daily_atrs = []
        for i in range(len(sessions)):
            if i < 14:
                atr = np.mean(true_ranges[:max(1, i)]) if i > 0 else 150.0
            else:
                atr = np.mean(true_ranges[i - 14:i])
            daily_atrs.append(float(atr))

        trades_list = []
        daily_records = []
        stats = {"aborted": 0, "pessimistic": 0, "news_blocks": 0, "primary": 0}

        # Sequential Tradeify evaluation challenge tracking
        account_balance = 50000.0
        eod_dd_barrier = 48000.0
        eod_hwm = 50000.0
        challenge_passes = 0
        challenge_fails = 0

        for i, s in enumerate(sessions):
            date_ = s["date"]
            rth = s["rth_bars"]
            if rth.empty or len(rth) < 30:
                daily_records.append({"date": date_, "net_pnl": 0.0, "traded": False})
                continue

            self.strategy.reset_session()
            atr = daily_atrs[i]

            # 1. Opening Range: 09:30–09:44
            or_bars = rth.between_time("09:30", "09:44")
            valid = self.strategy.calculate_opening_range(or_bars, atr)
            if not valid or self.strategy.session_aborted:
                stats["aborted"] += 1
                daily_records.append({"date": date_, "net_pnl": 0.0, "traded": False})
                continue

            # 2. Cumulative Session VWAP from 09:30
            tp_v = (rth["High"] + rth["Low"] + rth["Close"]) / 3.0
            cv = rth["Volume"].cumsum()
            cvp = (tp_v * rth["Volume"]).cumsum()
            vwap_series = cvp / np.maximum(cv, 1)

            # 3. Bar-by-bar evaluation (09:45–11:00 EST)
            eval_bars = rth.loc[
                (rth.index.time >= self.strategy.entry_window_start) &
                (rth.index.time <= self.strategy.primary_cutoff_time)
            ]

            active_trade = None
            for ts, bar in eval_bars.iterrows():
                vwap_now = float(vwap_series.loc[ts])
                ctx = {
                    "session_vwap": vwap_now,
                    "daily_20_ema": s.get("daily_20_ema", 0.0),
                    "exchange_open_price": s.get("exchange_open_price", 0.0)
                }
                sig = self.strategy.evaluate_bar(ts, bar, ctx)

                if "Blackout" in str(sig.get("reasons", [])):
                    stats["news_blocks"] += 1

                if sig.get("action") in ("LONG", "SHORT"):
                    d = 1 if sig["action"] == "LONG" else -1
                    sz = sig["sizing"]
                    risk = sz["stop_points"]
                    ttype = sig.get("trigger_type", "PRIMARY_BREAKOUT")
                    stats["primary"] += 1

                    # Task 1: Aggressive Dynamic Contract Sizing via RiskStateMachine
                    try:
                        sizing_res = self.risk_machine.calculate_sizing(
                            account_type=AccountType.EVALUATION,
                            current_balance=account_balance,
                            current_eod_drawdown_threshold=eod_dd_barrier,
                            stop_loss_points=risk,
                            consecutive_losses=0
                        )
                        contracts = sizing_res.contracts
                    except Exception:
                        contracts = 4

                    leg1_c = contracts // 2
                    leg2_c = contracts - contracts // 2

                    active_trade = TwoTierTrade(
                        trade_id=f"TRD15v-{date_}-{ts.strftime('%H%M')}",
                        direction=d,
                        entry_price=sig["entry_price"],
                        initial_sl=sz["stop_loss"],
                        tp1=sz["tp1"],
                        tp2=sz["tp2"],
                        contracts=contracts,
                        leg1_contracts=leg1_c,
                        leg2_contracts=leg2_c,
                        entry_time=ts,
                        trigger_type=ttype,
                        be_level=round(sig["entry_price"] + d * BE_OFFSET, 2),
                        profit_lock_trigger=round(sig["entry_price"] + d * self.profit_lock_trigger_r * risk, 2),
                        profit_lock_level=round(sig["entry_price"] + d * self.profit_lock_to_r * risk, 2),
                    )
                    break

            # 4. Forward execution
            session_pnl = 0.0
            if active_trade is not None:
                subsequent = rth.loc[rth.index > active_trade.entry_time]
                resolved = False

                for b_time, b in subsequent.iterrows():
                    result = simulate_bar(active_trade, b)
                    if result is not None:
                        active_trade = result
                        resolved = True
                        if "PESSIMISTIC" in active_trade.outcome:
                            stats["pessimistic"] += 1
                        break
                    if b_time.time() >= self.force_close_time:
                        active_trade = time_exit_trade(active_trade, float(b["Close"]), b_time)
                        resolved = True
                        break

                if not resolved and not subsequent.empty:
                    last = subsequent.iloc[-1]
                    active_trade = time_exit_trade(active_trade, float(last["Close"]), subsequent.index[-1])

                session_pnl = active_trade.realized_pnl
                trades_list.append({
                    "trade_id": active_trade.trade_id,
                    "date": str(date_),
                    "entry_time": str(active_trade.entry_time),
                    "exit_time": str(active_trade.exit_time),
                    "direction": "LONG" if active_trade.direction == 1 else "SHORT",
                    "entry_price": active_trade.entry_price,
                    "stop_loss": active_trade.initial_sl,
                    "tp1": active_trade.tp1,
                    "tp2": active_trade.tp2,
                    "contracts": active_trade.contracts,
                    "trigger_type": active_trade.trigger_type,
                    "leg1_outcome": active_trade.leg1_outcome,
                    "leg2_outcome": active_trade.leg2_outcome,
                    "outcome": active_trade.outcome,
                    "net_pnl": round(session_pnl, 2)
                })

                # Update challenge lifecycle
                account_balance += session_pnl
                if account_balance <= eod_dd_barrier:
                    challenge_fails += 1
                    account_balance = 50000.0
                    eod_dd_barrier = 48000.0
                    eod_hwm = 50000.0
                elif account_balance >= 53000.0:
                    challenge_passes += 1
                    account_balance = 50000.0
                    eod_dd_barrier = 48000.0
                    eod_hwm = 50000.0
                elif account_balance > eod_hwm:
                    eod_hwm = account_balance
                    nb = eod_hwm - 2000.0
                    if nb > 50000.0:
                        nb = 50000.0
                    if nb > eod_dd_barrier:
                        eod_dd_barrier = nb

                daily_records.append({"date": date_, "net_pnl": session_pnl, "traded": True})
            else:
                daily_records.append({"date": date_, "net_pnl": 0.0, "traded": False})

        trades_df = pd.DataFrame(trades_list)
        daily_df = pd.DataFrame(daily_records)
        logger.info(
            f"Backtest complete: {len(trades_df):,} trades "
            f"(Primary: {stats['primary']}), "
            f"{stats['aborted']} aborted, {stats['pessimistic']} pessimistic, "
            f"{stats['news_blocks']} news blocks. "
            f"Direct Challenge Cycles: {challenge_passes} Passed, {challenge_fails} Failed."
        )
        return trades_df, daily_df


# -----------------------------------------------------------------------
# First-Passage Monte Carlo Simulation
# -----------------------------------------------------------------------
def run_monte_carlo(
    trades_df: pd.DataFrame,
    num_sims: int = 5000,
    start_bal: float = 50000.0,
    target: float = 3000.0,
    max_dd: float = 2000.0,
    fee: float = 82.50,
    payout: float = 4000.0
) -> dict:
    """
    Simulates 5,000 bootstrap paths of independent Tradeify $50K challenge attempts.
    Evaluates first-passage terminal states:
      - Target reached ($53,000) -> PASS.
      - EOD trailing barrier breached ($48,000 locking at $50,000) -> FAIL.
    Evaluated over active trading days.
    """
    daily_pnls = trades_df.groupby("date")["net_pnl"].sum().values
    n = len(daily_pnls)
    np.random.seed(42)
    max_days = 250

    passed = 0
    failed = 0
    active = 0
    days_to_pass_list = []

    for _ in range(num_sims):
        sim_pnls = np.random.choice(daily_pnls, size=max_days, replace=True)
        bal = start_bal
        hwm = start_bal
        barrier = start_bal - max_dd
        status = "ACTIVE"
        days = 0

        for pnl in sim_pnls:
            days += 1
            bal += pnl
            if bal <= barrier:
                status = "FAILED"
                break
            if bal >= start_bal + target:
                status = "PASSED"
                break
            if bal > hwm:
                hwm = bal
                nb = hwm - max_dd
                if nb > start_bal:
                    nb = start_bal
                if nb > barrier:
                    barrier = nb

        if status == "PASSED":
            passed += 1
            days_to_pass_list.append(days)
        elif status == "FAILED":
            failed += 1
        else:
            active += 1

    pr = (passed / num_sims) * 100.0
    fr = (failed / num_sims) * 100.0
    avg_days = float(np.mean(days_to_pass_list)) if days_to_pass_list else 999.0
    cost = fee / (passed / num_sims) if passed > 0 else float("inf")
    ev = ((passed / num_sims) * payout) - ((failed / num_sims) * fee)
    ev_vel = ev / avg_days if avg_days > 0 else 0.0
    days_payout = avg_days + 5.0
    utility = payout / (cost * days_payout) if cost > 0 and days_payout > 0 else 0.0

    return {
        "pass_rate": round(pr, 2),
        "fail_rate": round(fr, 2),
        "active_rate": round((active / num_sims) * 100.0, 2),
        "avg_days": round(avg_days, 1),
        "cost_per_pass": round(cost, 2),
        "ev": round(ev, 2),
        "ev_vel": round(ev_vel, 2),
        "utility": round(utility, 4),
        "sims": num_sims
    }


# -----------------------------------------------------------------------
# Institutional Tear Sheet & Scorecard
# -----------------------------------------------------------------------
def generate_tearsheet(trades_df: pd.DataFrame, mc: dict):
    total = len(trades_df)
    if total == 0:
        print("No trades executed.")
        return

    wins = trades_df[trades_df["net_pnl"] > 0]
    losses = trades_df[trades_df["net_pnl"] <= 0]
    wr = (len(wins) / total) * 100.0
    gw = wins["net_pnl"].sum()
    gl = abs(losses["net_pnl"].sum())
    pf = gw / gl if gl > 0 else float("inf")
    net = trades_df["net_pnl"].sum()
    exp = trades_df["net_pnl"].mean()

    eq = trades_df["net_pnl"].cumsum()
    pk = np.maximum.accumulate(eq)
    dd = pk - eq
    maxdd = dd.max()

    outcomes = trades_df["outcome"].value_counts()
    contracts_dist = trades_df["contracts"].value_counts().sort_index()

    years = 8.9

    lines = [
        "",
        "=" * 96,
        "INSTITUTIONAL TEAR SHEET: HIGH-VELOCITY 15m ORB (PROP TOURNAMENT OVERHAUL)",
        "=" * 96,
        f"Data Window               : 2018-01-02 to 2026-09-16 ({years} Years CME Continuous 1m)",
        f"Execution Model           : Pessimistic ($1.24 Comms + 0.50 pts Slip), Two-Tier 1.25R / 2.50R",
        f"Risk Architecture         : Dynamic 8.5% Remaining DD Sizing (min $140, max $200) | Structural OR_Mid SL",
        f"Target Prop Account       : Tradeify $50K EOD Trailing DD ($2,000 Barrier | $3,000 Target | $82.50 Fee)",
        "-" * 96,
        "VELOCITY SCORECARD (CORE METRICS):",
        f"  Trade Expectancy        : ${exp:.2f} / trade",
        f"  Profit Factor           : {pf:.2f}",
        f"  Monte Carlo Pass Rate   : {mc['pass_rate']:.2f}% (Fail Rate: {mc['fail_rate']:.2f}%)",
        f"  Avg Trading Days to Pass: {mc['avg_days']:.1f} active trading days",
        f"  Daily EV Velocity (EV/d): ${mc['ev_vel']:.2f} / day",
        f"  Expected Value (EV)     : ${mc['ev']:,.2f}",
        f"  Tournament Utility      : {mc['utility']:.4f}",
        "-" * 96,
        "TRADE METRICS:",
        f"  Total Trades Executed   : {total:,} ({total / years:.1f} trades / year)",
        f"  Win Rate                : {wr:.2f}% ({len(wins)} Wins / {len(losses)} Losses)",
        f"  Total Net Realized PnL  : ${net:,.2f}",
        f"  Average Win             : ${wins['net_pnl'].mean():.2f}" if len(wins) else "  Average Win: N/A",
        f"  Average Loss            : ${losses['net_pnl'].mean():.2f}" if len(losses) else "  Average Loss: N/A",
        f"  Win / Loss Dollar Ratio : {(wins['net_pnl'].mean() / abs(losses['net_pnl'].mean())):.2f}" if len(losses) and len(wins) else "  Win / Loss Ratio: N/A",
        f"  Cumulative Max Drawdown : ${maxdd:,.2f}",
        "-" * 96,
        "DYNAMIC CONTRACT SIZING DISTRIBUTION:",
    ]
    for c_count, cnt in contracts_dist.items():
        lines.append(f"  {c_count:>2} MNQ Contracts         : {cnt:>5} trades ({cnt/total*100:.1f}%)")

    lines.append("-" * 96)
    lines.append("TWO-TIER EXECUTION BREAKDOWN:")
    for oc, cnt in outcomes.items():
        lines.append(f"  {oc:<35} : {cnt:>5} trades ({cnt/total*100:.1f}%)")

    lines += [
        "-" * 96,
        "MONTE CARLO TOURNAMENT ATTEMPTS (5,000 BOOTSTRAP PATHS):",
        f"  Pass Rate               : {mc['pass_rate']:.2f}%",
        f"  Fail Rate               : {mc['fail_rate']:.2f}%",
        f"  Avg Trading Days to Pass: {mc['avg_days']:.1f} active trading days",
        f"  Cost per Successful Pass: ${mc['cost_per_pass']:.2f}",
        f"  Expected Value (EV)     : ${mc['ev']:,.2f}",
        f"  Daily EV Velocity       : ${mc['ev_vel']:.2f} / day",
        "=" * 96,
        ""
    ]

    report = "\n".join(lines)
    print(report)

    out = STRATEGY_DIR / "backtest" / "tearsheet_15m.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Tear sheet written to: {out}")


# -----------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------
def main():
    logger.info("Initializing High-Velocity 15m ORB Backtest Pipeline...")

    parquet = REPO_ROOT / "data" / "nq_1m_continuous.parquet"
    raw_df = load_continuous_nq_parquet(filepath=parquet, filter_rth=False)
    sessions = extract_daily_trading_sessions(raw_df)
    logger.info(f"Loaded {len(sessions):,} daily sessions.")

    engine = InstitutionalBacktestEngine()
    trades_df, daily_df = engine.run_backtest(sessions)

    # Save trade log
    log_path = STRATEGY_DIR / "backtest" / "historical_trade_log_15m.csv"
    trades_df.to_csv(log_path, index=False)
    logger.info(f"Historical trade log saved to: {log_path}")

    # First-Passage Monte Carlo
    mc = run_monte_carlo(trades_df, num_sims=5000)

    # Generate Institutional Tear Sheet
    generate_tearsheet(trades_df, mc)


if __name__ == "__main__":
    main()
