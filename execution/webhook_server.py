"""
Production Signal Dispatcher & Risk Webhook Server (Phase 9 Production Hardened)
Author: Senior Quantitative Systems Engineer

Coordinates live market data ingestion, 35-day causal warmup, per-account state isolation,
CME holiday & rollover guards, idempotency deduplication, and alerting:
  - Strict localhost network binding (127.0.0.1) & X-API-KEY security.
  - Live Bar Ingestion Endpoint (`POST /bar_close`) with NT8 close-timestamp normalization.
  - 35-Trading-Day Historical Warmup Bootstrap for causal ATR_14, SMA(ATR_14, 20), and noise envelopes.
  - Strict RTH-only VWAP calculation resetting at 09:30:00 America/New_York.
  - Bidirectional CME <-> NinjaTrader 8 symbol mapping & Rollover Gap Protection.
  - Per-account atomic JSON state persistence (`state/tradeify_<ACCOUNT_ID>.json`).
  - Signal idempotency deduplication via unique signal_id caching.
  - Evaluation-to-Funded clean phase reset (`POST /admin/reset_to_funded`).
  - Asynchronous operational alerting via Telegram / Discord (`execution/alerts.py`).
"""

from __future__ import annotations
import os
import json
import logging
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
import pytz
import urllib.request
import urllib.error
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Internal imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from engine.state_persistence import TradeifyStatePersistence
from data.session_preprocessor import (
    is_cme_early_close,
    get_cme_flatten_time,
    check_cme_rollover_gap_guard
)
from execution.symbol_mapper import (
    cme_to_nt8_symbol,
    nt8_to_cme_symbol,
    get_active_mnq_contract,
    get_active_nt8_contract,
    validate_nt8_instrument,
    normalize_nt8_bar_timestamp,
    python_open_to_nt8_close
)
from execution.alerts import alert_dispatcher

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [WebhookServer]: %(message)s"
)
logger = logging.getLogger("WebhookServer")

# Configuration
BIND_HOST = os.getenv("HOST", "127.0.0.1")
BIND_PORT = int(os.getenv("PORT", "8765"))
API_KEY = os.getenv("TRADEIFY_API_KEY", os.getenv("API_KEY", "prop_killer_secret_key")).strip()

tz_ny = pytz.timezone("America/New_York")

app = FastAPI(
    title="Prop Killer Live Execution & Risk Dispatcher",
    version="9.1.0",
    description="Production-hardened signal bridge, multi-account isolation, and risk sentinel for Tradeify"
)

# Multi-Account State Persistence Registry
state_managers: Dict[str, TradeifyStatePersistence] = {}


def get_state_manager(account_id: str = "Tradeify-50K-Growth-001") -> TradeifyStatePersistence:
    account_id_clean = account_id.strip()
    if account_id_clean not in state_managers:
        state_managers[account_id_clean] = TradeifyStatePersistence(account_id=account_id_clean)
    return state_managers[account_id_clean]


# Security Dependency
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if not API_KEY or API_KEY == "prop_killer_secret_key":
        # Allow open localhost if default key not configured
        return True
    if x_api_key != API_KEY:
        logger.warning(f"Unauthorized API request: invalid X-API-KEY header ({x_api_key})")
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid X-API-KEY header.")
    return True


# =====================================================================
# Live Bar Ingestion, 35-Day Warmup & RTH VWAP Manager
# =====================================================================

class LiveBarManager:
    def __init__(self):
        self.session_bars: List[Dict[str, Any]] = []
        self.today_date: Optional[date] = None
        self.warmup_sessions_count: int = 0
        self.warmup_complete: bool = False
        self.historical_atr_map: Dict[date, float] = {}
        self.historical_atr_sma20_map: Dict[date, float] = {}
        self.cum_vol: float = 0.0
        self.cum_tp_vol: float = 0.0
        self.live_vwap: Optional[float] = None
        self.bootstrap_historical_warmup()

    def bootstrap_historical_warmup(self) -> None:
        """
        Loads at least 35 historical RTH sessions from cache or continuous parquet
        to pre-populate ATR_14, SMA(ATR_14, 20), and noise parameters.
        """
        cache_path = PROJECT_ROOT / "data" / "nq_1m_resolved_cache.parquet"
        raw_path = PROJECT_ROOT / "data" / "nq_1m_continuous.parquet"

        target_file = cache_path if cache_path.exists() else raw_path
        if not target_file.exists():
            logger.warning(f"Warmup bootstrap: No parquet cache found at {target_file}. Warmup sessions = 0.")
            self.warmup_sessions_count = 0
            self.warmup_complete = False
            return

        try:
            logger.info(f"Bootstrapping historical warmup sessions from {target_file.name}...")
            df = pd.read_parquet(target_file)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert(tz_ny)
            else:
                df.index = df.index.tz_convert(tz_ny)

            # RTH sessions 09:30 - 16:00
            rth = df.between_time("09:30", "16:00")
            daily_bars = rth.groupby(rth.index.date).agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last"
            })

            prev_close = daily_bars["Close"].shift(1)
            tr1 = daily_bars["High"] - daily_bars["Low"]
            tr2 = (daily_bars["High"] - prev_close).abs()
            tr3 = (daily_bars["Low"] - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            daily_atr = tr.rolling(window=14).mean().shift(1)
            daily_atr_sma20 = daily_atr.rolling(window=20).mean().shift(1)

            valid_dates = daily_atr_sma20.dropna().index.tolist()
            self.warmup_sessions_count = len(valid_dates)
            self.historical_atr_map = daily_atr.to_dict()
            self.historical_atr_sma20_map = daily_atr_sma20.to_dict()

            if self.warmup_sessions_count >= 35:
                self.warmup_complete = True
                logger.info(f"Warmup bootstrap SUCCESSFUL: {self.warmup_sessions_count} valid historical sessions loaded (>= 35 required).")
            else:
                self.warmup_complete = False
                logger.warning(f"Warmup bootstrap INCOMPLETE: Only {self.warmup_sessions_count} valid sessions loaded (< 35 required).")
        except Exception as e:
            logger.error(f"Warmup bootstrap failed: {e}")
            self.warmup_sessions_count = 0
            self.warmup_complete = False

    def ingest_bar(self, bar_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingests a 1-minute closed bar from NinjaTrader 8:
        1. Normalizes NT8 close-timestamp to Python open-timestamp.
        2. Enforces RTH-only VWAP starting at 09:30:00 EST.
        3. Updates running session statistics.
        """
        raw_ts = bar_data.get("timestamp")
        if isinstance(raw_ts, str):
            try:
                dt_close = datetime.fromisoformat(raw_ts)
            except ValueError:
                dt_close = datetime.strptime(raw_ts, "%Y-%m-%d %H:%M:%S")
        elif isinstance(raw_ts, datetime):
            dt_close = raw_ts
        else:
            raise ValueError(f"Invalid timestamp format: {raw_ts}")

        bar_minutes = int(bar_data.get("bar_minutes", 1))
        dt_open = normalize_nt8_bar_timestamp(dt_close, bar_minutes=bar_minutes)
        bar_date = dt_open.date()

        # Session Reset on New Day
        if self.today_date != bar_date:
            self.today_date = bar_date
            self.session_bars = []
            self.cum_vol = 0.0
            self.cum_tp_vol = 0.0
            self.live_vwap = None
            logger.info(f"LiveBarManager: New trading session initialized for date {bar_date}.")

        op = float(bar_data["open"])
        hi = float(bar_data["high"])
        lo = float(bar_data["low"])
        cl = float(bar_data["close"])
        vol = float(bar_data["volume"])

        # Strict RTH-Only VWAP (09:30:00 to 16:00:00 NY)
        t_time = dt_open.time()
        rth_start = datetime.strptime("09:30:00", "%H:%M:%S").time()
        rth_end = datetime.strptime("16:00:00", "%H:%M:%S").time()

        if rth_start <= t_time <= rth_end:
            tp = (hi + lo + cl) / 3.0
            self.cum_vol += vol
            self.cum_tp_vol += tp * vol
            if self.cum_vol > 0:
                self.live_vwap = self.cum_tp_vol / self.cum_vol

        bar_record = {
            "timestamp_open": dt_open.isoformat(),
            "timestamp_close": dt_close.isoformat(),
            "open": op,
            "high": hi,
            "low": lo,
            "close": cl,
            "volume": vol,
            "vwap_rth": self.live_vwap
        }
        self.session_bars.append(bar_record)
        return bar_record


bar_manager = LiveBarManager()


def get_current_est_time() -> datetime:
    return datetime.now(tz_ny)


# =====================================================================
# API Endpoints
# =====================================================================

@app.get("/health")
async def health_check():
    today_dt = get_current_est_time().date()
    is_early, close_t = is_cme_early_close(today_dt)
    active_cme = get_active_mnq_contract(today_dt)
    active_nt8 = get_active_nt8_contract(today_dt)
    sp_default = get_state_manager("Tradeify-50K-Growth-001")
    st = sp_default.get_state()

    return {
        "status": "healthy",
        "timestamp_est": get_current_est_time().isoformat(),
        "account_id": st.get("account_id"),
        "account_phase": st.get("account_phase"),
        "current_equity": st.get("current_equity"),
        "eod_hwm": st.get("eod_hwm"),
        "liquidation_floor": st.get("liquidation_floor"),
        "is_floor_locked": st.get("is_floor_locked"),
        "active_contract_cme": active_cme,
        "active_contract_nt8": active_nt8,
        "is_cme_early_close": is_early,
        "eod_flatten_time": get_cme_flatten_time(today_dt),
        "warmup_sessions_count": bar_manager.warmup_sessions_count,
        "warmup_complete": bar_manager.warmup_complete,
        "live_vwap_rth": bar_manager.live_vwap,
        "session_bars_received": len(bar_manager.session_bars)
    }


@app.get("/account_state/{account_id}")
async def get_account_state(account_id: str):
    sp = get_state_manager(account_id)
    return sp.get_state()


@app.post("/admin/reset_to_funded", dependencies=[Depends(verify_api_key)])
async def reset_to_funded_endpoint(request: Request):
    """
    Safely archives current Evaluation state and resets account to clean FUNDED phase.
    Includes zero HWM leak assertions.
    """
    data = await request.json()
    account_id = data.get("account_id", "Tradeify-50K-Growth-001").strip()
    sp = get_state_manager(account_id)
    new_state = sp.reset_to_funded()

    alert_dispatcher.dispatch(
        event_type="EOD_RECONCILIATION_SAVED",
        message=f"Account {account_id} successfully initialized to clean FUNDED phase ($50,000 base, $48,000 floor).",
        details=new_state
    )
    return {"status": "success", "message": "Account reset to clean FUNDED phase", "state": new_state}


@app.post("/bar_close", dependencies=[Depends(verify_api_key)])
async def ingest_bar_close(request: Request):
    """
    Receives closed 1-minute OHLCV bar pushed directly from NT8's OnBarUpdate().
    Normalizes timestamp and updates live RTH VWAP.
    """
    data = await request.json()
    try:
        recorded = bar_manager.ingest_bar(data)
        return {
            "status": "ingested",
            "open_timestamp": recorded["timestamp_open"],
            "vwap_rth": recorded["vwap_rth"],
            "warmup_complete": bar_manager.warmup_complete
        }
    except Exception as e:
        logger.error(f"Error ingesting bar: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/warmup_bars", dependencies=[Depends(verify_api_key)])
async def load_warmup_bars(request: Request):
    """
    Explicitly adds warmup session records or re-triggers warmup bootstrap.
    """
    data = await request.json()
    if data.get("rebootstrap", False):
        bar_manager.bootstrap_historical_warmup()
    return {
        "status": "success",
        "warmup_sessions_count": bar_manager.warmup_sessions_count,
        "warmup_complete": bar_manager.warmup_complete
    }


@app.post("/webhook/signal", dependencies=[Depends(verify_api_key)])
async def receive_alpha_signal(request: Request):
    """
    Receives quantitative signal payload from H3 or H4 alpha modules.
    Enforces Warmup Guard, CME Rollover & Holiday Guards, Idempotency, and Risk Boundaries.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # 1. Warmup Completeness Guard (35 sessions minimum)
    if not bar_manager.warmup_complete:
        msg = f"SIGNAL REJECTED: Warmup incomplete ({bar_manager.warmup_sessions_count} < 35 sessions). 14d ATR/Noise and 20d SMA uncalibrated."
        logger.critical(msg)
        alert_dispatcher.dispatch("ORDER_REJECTED", msg, payload)
        return {"status": "rejected", "reason": msg}

    account_id = payload.get("account_id", "Tradeify-50K-Growth-001").strip()
    sp = get_state_manager(account_id)
    acc_state = sp.get_state()

    strategy_id = payload.get("strategy_id", "UNKNOWN")
    direction = payload.get("direction", "BUY").upper()
    est_now = get_current_est_time()
    today_dt = est_now.date()
    now_time = est_now.time()

    # 2. Idempotency Deduplication Guard
    signal_id = payload.get("signal_id")
    if not signal_id:
        signal_id = f"{today_dt.strftime('%Y%m%d')}_{strategy_id}_{direction}_{account_id}"
        payload["signal_id"] = signal_id

    if sp.is_signal_processed(signal_id):
        msg = f"Duplicate signal {signal_id} rejected by idempotency cache."
        logger.warning(msg)
        return {"status": "duplicate_rejected", "signal_id": signal_id}

    # 3. CME Early-Close Holiday Guard
    is_early, close_str = is_cme_early_close(today_dt)
    if is_early:
        if strategy_id == "H4":
            msg = f"Rejected signal H4: CME Early-Close Day ({today_dt}). 15:00:00 H4 session is cancelled."
            logger.warning(msg)
            alert_dispatcher.dispatch("ORDER_REJECTED", msg, payload)
            return {"status": "rejected", "reason": msg}
        if now_time >= datetime.strptime("12:45:00", "%H:%M:%S").time():
            msg = f"Rejected signal {strategy_id}: CME Early-Close Liquidation Window (after 12:45 EST)."
            logger.warning(msg)
            alert_dispatcher.dispatch("ORDER_REJECTED", msg, payload)
            return {"status": "rejected", "reason": msg}

    # 4. CME Rollover Price-Gap Guard
    rollover_ok, rollover_msg = check_cme_rollover_gap_guard(today_dt, new_contract_has_history=True)
    if not rollover_ok:
        logger.warning(rollover_msg)
        alert_dispatcher.dispatch("ORDER_REJECTED", rollover_msg, payload)
        return {"status": "rejected", "reason": rollover_msg}

    # 5. Symbol Format Validation & Bidirectional Mapping
    symbol_raw = payload.get("symbol", "MNQ")
    if " " in symbol_raw:
        # Incoming NT8 format (e.g. 'MNQ 12-26')
        is_valid, msg = validate_nt8_instrument(symbol_raw, today_dt)
        if not is_valid:
            logger.warning(msg)
            alert_dispatcher.dispatch("ORDER_REJECTED", msg, payload)
            return {"status": "rejected", "reason": msg}
        cme_sym = nt8_to_cme_symbol(symbol_raw)
        nt8_sym = symbol_raw
    else:
        # Incoming CME format or bare 'MNQ'
        active_cme = get_active_mnq_contract(today_dt)
        cme_sym = active_cme
        nt8_sym = cme_to_nt8_symbol(active_cme)

    payload["symbol_cme"] = cme_sym
    payload["symbol_nt8"] = nt8_sym

    # 6. Persistent Account State Validation
    is_funded = (acc_state.get("account_phase") == "FUNDED")
    current_equity = acc_state.get("current_equity", 50000.0)
    today_pnl = acc_state.get("today_realized_pnl", 0.0)
    eod_hwm = acc_state.get("eod_hwm", 50000.0)
    is_floor_locked = acc_state.get("is_floor_locked", False)
    qual_days = acc_state.get("qualifying_days_count", 0)

    # 7. Absorbing Upper-Barrier Checks
    if not is_funded and current_equity >= 53015.00:
        msg = f"Signal {strategy_id} skipped: Evaluation target ($53,015.00) cleared. Account PASSED."
        logger.info(msg)
        return {"status": "skipped", "reason": msg}

    if is_funded and current_equity >= 52660.00 and qual_days >= 4 and today_pnl >= 160.00:
        msg = f"Signal {strategy_id} skipped: Funded payout target ($52,660.00) cleared. Payout LOCKED."
        logger.info(msg)
        return {"status": "skipped", "reason": msg}

    # 8. Funded Pre-Lock House-Money Floor Guard for H4
    contracts = payload.get("contracts", 2)
    if is_funded and not is_floor_locked and eod_hwm < 52100.0 and strategy_id == "H4":
        if today_pnl >= 165.0:
            house_money_budget = today_pnl - 160.0
            sl_pts = abs(float(payload.get("entry_price", 0.0)) - float(payload.get("sl_price", 0.0)))
            per_c_risk = (sl_pts * 2.00) + 4.48 if sl_pts > 0 else 50.0
            max_h4_c = int(house_money_budget // per_c_risk)
            if max_h4_c <= 0:
                msg = f"Signal H4 skipped: House-Money budget (${house_money_budget:.2f}) insufficient to risk 1 MNQ without jeopardizing +$160 floor."
                logger.info(msg)
                return {"status": "skipped", "reason": msg}
            contracts = min(contracts, max_h4_c)
            payload["contracts"] = contracts
        else:
            msg = "Signal H4 skipped: Morning PnL < $165.00 in Funded Pre-Lock zone. Preserving floor."
            logger.info(msg)
            return {"status": "skipped", "reason": msg}

    # 9. Record Processed Signal ID (Idempotency)
    sp.record_signal_processed(signal_id)

    # 10. Write Pending Signal File for NT8 File-Based IPC
    pending_signal_path = PROJECT_ROOT / "state" / f"tradeify_{account_id}_pending_signal.json"
    signal_out = {
        "signal_id": signal_id,
        "account_id": account_id,
        "strategy_id": strategy_id,
        "direction": direction,
        "symbol_cme": cme_sym,
        "symbol_nt8": nt8_sym,
        "contracts": contracts,
        "entry_price": payload.get("entry_price"),
        "sl_price": payload.get("sl_price"),
        "tp1_price": payload.get("tp1_price"),
        "timestamp": get_current_est_time().isoformat()
    }
    with open(pending_signal_path, "w", encoding="utf-8") as f:
        json.dump(signal_out, f, indent=2)

    # Optional local socket/HTTP forward to NT8 listener on 127.0.0.1:8766
    try:
        nt8_url = "http://127.0.0.1:8766/signal"
        req = urllib.request.Request(
            nt8_url,
            data=json.dumps(signal_out).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=0.5) as resp:
            pass
    except Exception:
        # NT8 may be operating in file-watcher mode or not yet listening; normal
        pass

    # 11. Dispatch Operational Alert
    alert_dispatcher.dispatch(
        event_type="ORDER_FILLED",
        message=f"Tradeify Signal Dispatched: {direction} {contracts}x {nt8_sym} ({strategy_id}) for {account_id}.",
        details=signal_out
    )

    return {
        "status": "dispatched",
        "signal_id": signal_id,
        "account_id": account_id,
        "strategy": strategy_id,
        "symbol_nt8": nt8_sym,
        "symbol_cme": cme_sym,
        "contracts": contracts
    }


@app.post("/execution_fill", dependencies=[Depends(verify_api_key)])
async def record_execution_fill(request: Request):
    """
    Called upon trade fill or completion by NinjaTrader 8 or simulation bridge.
    Persists updated state atomically to per-account disk file.
    """
    data = await request.json()
    account_id = data.get("account_id", "Tradeify-50K-Growth-001").strip()
    sp = get_state_manager(account_id)
    net_pnl = float(data.get("net_pnl", 0.0))
    trade_id = data.get("trade_id", "LIVE_TR")
    new_state = sp.record_trade_fill(net_pnl, trade_id)

    alert_dispatcher.dispatch(
        event_type="ORDER_FILLED",
        message=f"Trade Fill Recorded for {account_id}: Net PnL = ${net_pnl:+.2f} (Trade ID: {trade_id}).",
        details={"account_id": account_id, "current_equity": new_state["current_equity"], "eod_hwm": new_state["eod_hwm"]}
    )
    return {"status": "recorded", "state": new_state}


@app.post("/eod_reconciliation", dependencies=[Depends(verify_api_key)])
async def reconcile_eod(request: Request):
    """
    Executes end-of-day reconciliation and persists state to per-account disk file.
    """
    data = await request.json()
    account_id = data.get("account_id", "Tradeify-50K-Growth-001").strip()
    sp = get_state_manager(account_id)
    session_d = data.get("session_date", datetime.now().strftime("%Y-%m-%d"))
    res = sp.reconcile_eod(session_d)

    alert_dispatcher.dispatch(
        event_type="EOD_RECONCILIATION_SAVED",
        message=f"EOD Reconciliation Saved for {account_id}: Equity = ${res['current_equity']:.2f}, Floor = ${res['liquidation_floor']:.2f}.",
        details=res
    )
    return {"status": "reconciled", "state": res}


if __name__ == "__main__":
    logger.info(f"Starting Tradeify Live Execution Dispatcher on {BIND_HOST}:{BIND_PORT}...")
    uvicorn.run(app, host=BIND_HOST, port=BIND_PORT)
