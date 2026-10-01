from fastapi import FastAPI, Request, HTTPException
import uvicorn
import datetime

app = FastAPI(title="Execution Engine Webhook", version="1.0.0")

@app.post("/webhook")
async def receive_signal(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
        
    symbol = data.get("symbol", "MNQ")
    action = data.get("action", "BUY").upper()
    contracts = data.get("contracts", 1)
    entry = data.get("entry", "MARKET")
    sl = data.get("sl")
    tp1 = data.get("tp1")
    tp2 = data.get("tp2")
    
    # Format high-visibility execution alert
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    alert_lines = [
        "========================================",
        f"🚨 TRADE ALERT [{timestamp}] 🚨",
        f"Symbol    : {symbol}",
        f"Action    : {action} {contracts} Contracts",
        f"Entry     : {entry}"
    ]
    
    if sl: alert_lines.append(f"Stop Loss : {sl}")
    if tp1: alert_lines.append(f"TP1       : {tp1}")
    if tp2: alert_lines.append(f"TP2       : {tp2}")
    
    alert_lines.append("========================================")
    
    alert_msg = "\n".join(alert_lines)
    
    # 1. Output to console
    print(alert_msg)
    
    # 2. Add API integrations here for Tradovate / NinjaTrader
    # Example: tradovate_client.place_order(symbol, action, contracts, order_type="Market")
    
    # 3. Add Discord / Telegram Webhook forwards here
    # Example: requests.post(DISCORD_WEBHOOK_URL, json={"content": f"```\n{alert_msg}\n```"})
    
    return {"status": "success", "message": "Signal received and processed"}

if __name__ == "__main__":
    print("Starting Webhook Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
