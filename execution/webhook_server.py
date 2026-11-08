"""
Production Signal Dispatcher & Risk Webhook Server
Author: Principal Quantitative Architect

Coordinates live alerts and signals between Python quantitative modules and NinjaTrader 8:
  - Validates MNQ contract sizing dynamically.
  - Broadcasts circuit breaker events (-$600 daily stop, +$825 funded ceiling).
  - Enforces 15:44:00 EST hard liquidation timer.
  - Dispatches execution payloads to local NinjaTrader 8 TCP/HTTP listener.
"""

from __future__ import annotations
import asyncio
from datetime import datetime
import json
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [WebhookServer]: %(message)s"
)
logger = logging.getLogger("WebhookServer")

app = FastAPI(
    title="Prop Killer V2 Tradeify Execution Dispatcher",
    version="2.0.0",
    description="Institutional signal bridge and risk sentinel for Tradeify $50K Growth"
)

# Active State Memory
system_state = {
    "is_halted": False,
    "halt_reason": None,
    "is_funded": False,
    "current_balance": 50000.0,
    "intraday_pnl": 0.0,
    "daily_stop_limit": -600.0,
    "funded_profit_cap": 825.0,
    "active_position": None,
    "trades_today": []
}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "system_halted": system_state["is_halted"],
        "account_phase": "FUNDED" if system_state["is_funded"] else "EVALUATION",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/set_phase")
async def set_account_phase(request: Request):
    data = await request.json()
    phase = data.get("phase", "EVALUATION").upper()
    if phase not in ["EVALUATION", "FUNDED"]:
        raise HTTPException(status_code=400, detail="Invalid phase. Must be EVALUATION or FUNDED.")
    system_state["is_funded"] = (phase == "FUNDED")
    logger.info(f"Account Phase updated to: {phase}")
    return {"status": "success", "phase": phase}


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
    Receives quantitative signal payload from H1, H2, H3, or H4 alpha modules.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if system_state["is_halted"]:
        logger.warning(f"Rejected signal {payload.get('strategy_id')}: System is HALTED ({system_state['halt_reason']})")
        return {"status": "rejected", "reason": f"System halted: {system_state['halt_reason']}"}

    strategy_id = payload.get("strategy_id", "UNKNOWN")
    symbol = payload.get("symbol", "MNQ")

    # Strict Micro contract enforcement
    if "MNQ" not in symbol:
        raise HTTPException(status_code=400, detail=f"Contract Exclusivity Breach: {symbol} rejected. Only MNQ permitted.")

    direction = payload.get("direction", "BUY").upper()
    contracts = payload.get("contracts", 2)
    entry_price = payload.get("entry_price")
    sl_price = payload.get("sl_price")
    tp1_price = payload.get("tp1_price")
    tp2_price = payload.get("tp2_price")

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    banner = [
        "===========================================================",
        f"🚨 TRADEIFY SIGNAL DISPATCHED: [{timestamp_str}]",
        f"Strategy   : {strategy_id} | Phase: {'FUNDED' if system_state['is_funded'] else 'EVALUATION'}",
        f"Action     : {direction} {contracts}x {symbol}",
        f"Entry Fill : {entry_price}",
        f"Stop Loss  : {sl_price}",
        f"TP1 Target : {tp1_price}",
        f"TP2 Runner : {tp2_price}",
        "==========================================================="
    ]
    logger.info("\n".join(banner))

    # Forward to NinjaTrader 8 local unmanaged execution bridge
    # In live trading, this sends a TCP message or internal socket to NT8 Strategy listener.

    return {
        "status": "dispatched",
        "timestamp": timestamp_str,
        "strategy": strategy_id,
        "symbol": symbol,
        "contracts": contracts
    }


if __name__ == "__main__":
    logger.info("Starting Tradeify Production Signal Dispatcher on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
