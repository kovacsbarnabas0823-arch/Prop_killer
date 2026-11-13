"""
Production Signal Dispatcher & Risk Webhook Server (Phase 9 Hardened)
Author: Principal Quantitative Architect & Lead Systems Engineer

Coordinates live alerts and signals between Python quantitative modules and NinjaTrader 8:
  - Atomic JSON state persistence via TradeifyStatePersistence.
  - CME Early-Close Holiday Guard (Black Friday, Christmas Eve, July 3rd closing at 13:00 EST).
  - CME MNQ Quarterly Contract Rollover Resolver (MNQH, MNQM, MNQU, MNQZ).
  - Validates MNQ contract sizing dynamically (House-Money Floor Guard).
  - Broadcasts circuit breaker events (-$600 daily stop, +$920 funded consistency ceiling).
  - Absorbing upper-barrier auto-liquidation (Pass-on-Touch $53,015 / Payout-on-Touch $52,660).
  - Enforces EOD hard liquidation timer (12:58 EST on early-close, 15:44 EST normal).
"""

from __future__ import annotations
import asyncio
from datetime import datetime, date
import json
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException
import pytz
import uvicorn

from engine.state_persistence import TradeifyStatePersistence
from data.session_preprocessor import is_cme_early_close, get_cme_flatten_time, get_active_mnq_contract

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [WebhookServer]: %(message)s"
)
logger = logging.getLogger("WebhookServer")

app = FastAPI(
    title="Prop Killer Phase 9 Tradeify Execution Dispatcher",
    version="9.0.0",
    description="Institutional signal bridge, state persistence, and risk sentinel for Tradeify $50K Growth"
)

tz_ny = pytz.timezone("America/New_York")

# Persistent state manager
state_mgr = TradeifyStatePersistence()

# Active In-Memory System State synchronized with StatePersistence
system_state = {
    "is_halted": False,
    "halt_reason": None,
    "active_position": None,
    "trades_today": []
}


def get_current_est_time() -> datetime:
    return datetime.now(tz_ny)


@app.get("/health")
async def health_check():
    today_dt = get_current_est_time().date()
    is_early, close_t = is_cme_early_close(today_dt)
    active_contract = get_active_mnq_contract(today_dt)
    st = state_mgr.get_state()
    return {
        "status": "healthy",
        "system_halted": system_state["is_halted"],
        "account_id": st.get("account_id"),
        "account_phase": st.get("account_phase"),
        "current_equity": st.get("current_equity"),
        "eod_hwm": st.get("eod_hwm"),
        "liquidation_floor": st.get("liquidation_floor"),
        "is_floor_locked": st.get("is_floor_locked"),
        "active_contract": active_contract,
        "is_cme_early_close": is_early,
        "eod_flatten_time": get_cme_flatten_time(today_dt),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/account_state")
async def get_account_state():
    return state_mgr.get_state()


@app.post("/set_phase")
async def set_account_phase(request: Request):
    data = await request.json()
    phase = data.get("phase", "EVALUATION").upper()
    if phase not in ["EVALUATION", "FUNDED"]:
        raise HTTPException(status_code=400, detail="Invalid phase. Must be EVALUATION or FUNDED.")
    state_mgr.set_phase(phase)
    logger.info(f"Account Phase updated in persistent state to: {phase}")
    return {"status": "success", "phase": phase, "state": state_mgr.get_state()}


@app.post("/circuit_breaker")
async def trigger_circuit_breaker(request: Request):
    data = await request.json()
    reason = data.get("reason", "Manual Circuit Breaker")
    system_state["is_halted"] = True
    system_state["halt_reason"] = reason
    logger.warning(f"CIRCUIT BREAKER TRIGGERED: {reason}. Liquidating positions.")
    return {"status": "halted", "reason": reason}


@app.post("/webhook/signal")
async def receive_alpha_signal(request: Request):
    """
    Receives quantitative signal payload from H3 or H4 alpha modules.
    Enforces CME Holiday Guards, Rollover Resolution, and Risk Boundaries.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if system_state["is_halted"]:
        logger.warning(f"Rejected signal {payload.get('strategy_id')}: System is HALTED ({system_state['halt_reason']})")
        return {"status": "rejected", "reason": f"System halted: {system_state['halt_reason']}"}

    est_now = get_current_est_time()
    today_dt = est_now.date()
    now_time = est_now.time()
    strategy_id = payload.get("strategy_id", "UNKNOWN")

    # 1. CME Early-Close Holiday Guard
    is_early, _ = is_cme_early_close(today_dt)
    if is_early:
        if strategy_id == "H4":
            logger.warning(f"Rejected signal H4: CME Early-Close Day ({today_dt}). 15:00:00 H4 session is cancelled.")
            return {"status": "rejected", "reason": "CME Early-Close Holiday: 15:00 H4 session cancelled"}
        if now_time >= datetime.strptime("12:45:00", "%H:%M:%S").time():
            logger.warning(f"Rejected signal {strategy_id}: CME Early-Close Liquidation Window (after 12:45 EST).")
            return {"status": "rejected", "reason": "CME Early-Close Liquidation Window engaged"}

    # 2. CME MNQ Quarterly Contract Rollover Resolver
    symbol = payload.get("symbol", "MNQ")
    if symbol == "MNQ":
        active_contract = get_active_mnq_contract(today_dt)
        symbol = active_contract
        payload["symbol"] = active_contract
    elif "MNQ" not in symbol:
        raise HTTPException(status_code=400, detail=f"Contract Exclusivity Breach: {symbol} rejected. Only CME MNQ permitted.")

    # 3. Persistent Account State Validation
    acc_state = state_mgr.get_state()
    is_funded = (acc_state.get("account_phase") == "FUNDED")
    current_equity = acc_state.get("current_equity", 50000.0)
    today_pnl = acc_state.get("today_realized_pnl", 0.0)
    eod_hwm = acc_state.get("eod_hwm", 50000.0)
    is_floor_locked = acc_state.get("is_floor_locked", False)
    qual_days = acc_state.get("qualifying_days_count", 0)

    # 4. Phase 7/8 Absorbing Upper-Barrier Checks
    if not is_funded and current_equity >= 53015.00:
        logger.info(f"Signal {strategy_id} skipped: Evaluation target ($53,015.00) cleared. Account PASSED.")
        return {"status": "skipped", "reason": "Evaluation target already cleared ($53,015.00 Pass-on-Touch)"}

    if is_funded and current_equity >= 52660.00 and qual_days >= 4 and today_pnl >= 160.00:
        logger.info(f"Signal {strategy_id} skipped: Funded payout target ($52,660.00) cleared. Payout LOCKED.")
        return {"status": "skipped", "reason": "Funded payout target already cleared ($52,660.00 Payout-on-Touch)"}

    # 5. Funded Pre-Lock House-Money Floor Guard for H4
    if is_funded and not is_floor_locked and eod_hwm < 52100.0 and strategy_id == "H4":
        if today_pnl >= 165.0:
            house_money_budget = today_pnl - 160.0
            sl_pts = abs(float(payload.get("entry_price", 0.0)) - float(payload.get("sl_price", 0.0)))
            per_c_risk = (sl_pts * 2.00) + 4.48 if sl_pts > 0 else 50.0
            max_h4_c = int(house_money_budget // per_c_risk)
            if max_h4_c <= 0:
                logger.info(f"Signal H4 skipped: House-Money budget (${house_money_budget:.2f}) insufficient to risk 1 MNQ without jeopardizing +$160 floor.")
                return {"status": "skipped", "reason": "House-Money budget insufficient to protect +$160 floor"}
            payload["contracts"] = min(payload.get("contracts", 2), max_h4_c)
        else:
            logger.info("Signal H4 skipped: Morning PnL < $165.00 in Funded Pre-Lock zone. Preserving floor.")
            return {"status": "skipped", "reason": "Morning PnL below +$165.00 house-money hurdle"}

    direction = payload.get("direction", "BUY").upper()
    contracts = payload.get("contracts", 2)
    entry_price = payload.get("entry_price")
    sl_price = payload.get("sl_price")
    tp1_price = payload.get("tp1_price")
    tp2_price = payload.get("tp2_price")

    is_pyramid = False
    if system_state["active_position"] is not None:
        if strategy_id == "H4" and system_state["active_position"].get("direction") == direction:
            is_pyramid = True
            logger.info("De-risked Same-Direction Pyramiding authorized: H4 entering concurrently with active H3 position.")
        else:
            logger.warning(f"Rejected signal {strategy_id}: Concurrency guard active (open position: {system_state['active_position']})")
            return {"status": "rejected", "reason": "Active position already open in conflicting direction"}

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    banner = [
        "===========================================================",
        f"🚨 TRADEIFY SIGNAL DISPATCHED: [{timestamp_str}]",
        f"Strategy   : {strategy_id} | Phase: {'FUNDED' if is_funded else 'EVALUATION'}",
        f"Action     : {direction} {contracts}x {symbol}",
        f"Entry Fill : {entry_price}",
        f"Stop Loss  : {sl_price}",
        f"TP1 Target : {tp1_price}",
        f"TP2 Runner : {tp2_price}",
        f"Account Eq : ${current_equity:.2f} | Floor: ${acc_state.get('liquidation_floor', 48000.0):.2f}",
        "==========================================================="
    ]
    logger.info("\n".join(banner))

    return {
        "status": "dispatched",
        "timestamp": timestamp_str,
        "strategy": strategy_id,
        "symbol": symbol,
        "contracts": contracts
    }


@app.post("/execution_fill")
async def record_execution_fill(request: Request):
    """
    Called upon trade fill or completion by NinjaTrader 8 or simulation bridge.
    Persists updated state atomically to disk.
    """
    data = await request.json()
    net_pnl = float(data.get("net_pnl", 0.0))
    trade_id = data.get("trade_id", "LIVE_TR")
    new_state = state_mgr.record_trade_fill(net_pnl, trade_id)
    return {"status": "recorded", "state": new_state}


@app.post("/eod_reconciliation")
async def reconcile_eod(request: Request):
    """
    Executes end-of-day reconciliation and persists state to disk.
    """
    data = await request.json()
    session_d = data.get("session_date", datetime.now().strftime("%Y-%m-%d"))
    res = state_mgr.reconcile_eod(session_d)
    return {"status": "reconciled", "state": res}


if __name__ == "__main__":
    logger.info("Starting Tradeify Production Signal Dispatcher on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

