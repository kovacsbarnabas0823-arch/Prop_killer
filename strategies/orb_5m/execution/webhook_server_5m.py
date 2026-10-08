"""
Dedicated Execution Webhook & Alert Bridge for 5-Minute ORB Champion Strategy
Target: CME Micro Nasdaq-100 (MNQ) Futures
Provides REST endpoint for incoming execution signals, TradingView alerts, and notification dispatches.
"""

from fastapi import FastAPI, Request, HTTPException
import uvicorn
import datetime
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("WebhookServer5M")

app = FastAPI(title="5-Minute ORB Execution Bridge", version="1.0.0")

@app.post("/webhook/orb_5m")
async def receive_5m_signal(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    symbol = data.get("symbol", "MNQ")
    action = data.get("action", "BUY").upper()
    contracts = data.get("contracts", 8)
    trigger_price = data.get("trigger_price")
    initial_sl = data.get("initial_sl")
    trail_step = data.get("trail_step_05r")
    candle_high = data.get("candle_high")
    candle_low = data.get("candle_low")
    ema_12 = data.get("ema_12")

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    alert_lines = [
        "============================================================",
        f"🚨 5-MINUTE ORB + 12 EMA TRADE ALERT [{timestamp}] 🚨",
        f"Instrument    : CME {symbol} Futures",
        f"Execution     : {action} {contracts} Contracts (Resting Stop-Market)",
        f"Trigger Price : {trigger_price}",
        f"Initial SL    : {initial_sl} (Opposite 5m candle extreme)",
        f"0.5R Trail    : Ratchets by {trail_step} pts on favorable ticks",
        f"5m Range      : {candle_low} - {candle_high} | 12 EMA: {ema_12}",
        "Safety Rule   : Max 1 trade/day | EOD Flatten at 15:45 EST",
        "============================================================"
    ]

    alert_msg = "\n".join(alert_lines)
    print(alert_msg)
    logger.info(f"Dispatched 5m ORB signal: {action} {contracts}x {symbol} @ {trigger_price}")

    return {
        "status": "success",
        "strategy": "NQ_Institutional_5m_ORB_Velocity_Locked",
        "action": action,
        "contracts": contracts,
        "timestamp": timestamp
    }

if __name__ == "__main__":
    print("Starting 5-Minute ORB Webhook Server on port 8005...")
    uvicorn.run(app, host="0.0.0.0", port=8005)
