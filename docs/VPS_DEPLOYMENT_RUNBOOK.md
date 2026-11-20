# Chicago VPS Production Deployment & Operations Runbook

**Target System:** `Prop_killer` Institutional High-Convexity Prop Firm Execution Engine  
**Broker / Prop Firm:** Tradeify $50K Growth Account (CME MNQ, $2.00/point)  
**Target Hardware / Host:** Dedicated Windows Server VPS co-located in Chicago (Equinix CH1/CH2/CH4, < 2ms CME latency)  
**Execution Stack:** NinjaTrader 8 Unmanaged C# Strategy (`TradeifyExecutionManager.cs`) + Local FastAPI Webhook Server (`webhook_server.py`)  

---

## 1. System Architecture & Topology

```
+--------------------------------------------------------------------------------------------------+
|                                    CHICAGO EQUINIX VPS (WINDOWS SERVER)                         |
|                                                                                                  |
|   +------------------------------------+             +---------------------------------------+   |
|   |          NINJATRADER 8             |             |       PYTHON DISPATCH ENGINE          |   |
|   |                                    |             |                                       |   |
|   |  - Tradovate / Rithmic Feed        |  POST /bar  |  - FastAPI (127.0.0.1:8765)           |   |
|   |  - TradeifyExecutionManager.cs     |------------>|  - 35-Day Historical Warmup Cache     |   |
|   |  - OnBarUpdate() (1m OHLCV)        |             |  - H3 & H4 Frozen Signal Generators   |   |
|   |                                    |  HTTP :8766 |  - Tradeify Trailing Floor Engine     |   |
|   |  - Dual IPC Signal Consumer:       |<------------|                                       |   |
|   |     * HttpListener (:8766)         |     or      |  - Alert Dispatcher:                  |   |
|   |     * File Watcher (pending.json)  | File Queue  |     * Telegram / Discord Webhook      |   |
|   |                                    |             |                                       |   |
|   |  - Unmanaged Broker Execution:     |             |  - Per-Account Atomic State:          |   |
|   |     * Independent H3/H4 SL orders  |             |     * state/tradeify_<ACC_ID>.json    |   |
|   |     * Server-side StopMarket fills |             |                                       |   |
|   |     * Consistency Governor Buffer  |             |                                       |   |
|   +------------------------------------+             +---------------------------------------+   |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Server Provisioning & OS Hardening

### 2.1 Clock Synchronization (CME 0-Second Parity)
Microsecond timestamp accuracy is vital to eliminate signal-to-execution drift. Configure Windows Time Service (`w32tm`) to poll stratum-1 NTP servers every 15 minutes.

Open PowerShell as **Administrator**:
```powershell
# Stop Windows Time service
net stop w32time

# Configure Google NTP and NIST servers
w32tm /config /syncfromflags:manual /manualpeerlist:"time.google.com,0x1 time.nist.gov,0x1 time.cloudflare.com,0x1" /reliable:yes /update

# Set polling interval to 512-1024s (MinPollInterval 9, MaxPollInterval 10)
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\W32Time\Config" -Name "MinPollInterval" -Value 9
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\W32Time\Config" -Name "MaxPollInterval" -Value 10
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\W32Time\Config" -Name "UpdateInterval" -Value 100

# Start Windows Time service and verify sync
net start w32time
w32tm /resync /force
w32tm /query /status
```
*Verification target:* `Phase Offset: < 0.0500000s` (< 50ms drift).

### 2.2 Timezone Setting
Ensure the Windows VPS system timezone is strictly set to **Eastern Time (US & Canada)**:
```powershell
Set-TimeZone -Id "Eastern Standard Time"
Get-TimeZone
```

### 2.3 Prevent Windows Update Auto-Reboots during RTH
Windows Update reboots during active trading sessions (08:00–17:00 EST) can orphan broker positions.

Run the following registry commands to lock active hours:
```powershell
# Configure Active Hours (08:00 to 18:00)
New-Item -Path "HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate" -Force
Set-ItemProperty -Path "HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate" -Name "SetActiveHours" -Value 1
Set-ItemProperty -Path "HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate" -Name "ActiveHoursStart" -Value 8
Set-ItemProperty -Path "HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate" -Name "ActiveHoursEnd" -Value 18
Set-ItemProperty -Path "HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate" -Name "NoAutoRebootWithLoggedOnUsers" -Value 1
```

---

## 3. Environment & Python Service Deployment (NSSM)

### 3.1 Python Virtual Environment & Dependencies
In the repository directory (`C:\Users\User\Documents\automated prop`):
```powershell
# 1. Verify virtualenv
.\.venv\Scripts\python.exe --version

# 2. Configure environment file (.env)
Copy-Item .env.example .env -ErrorAction SilentlyContinue
```

Ensure `.env` contains:
```ini
API_KEY=tradeify_live_production_secret_key_2026
WEBHOOK_HOST=127.0.0.1
WEBHOOK_PORT=8765
NT8_PORT=8766
ALERT_TELEGRAM_TOKEN=your_telegram_bot_token
ALERT_TELEGRAM_CHAT_ID=your_telegram_chat_id
ALERT_DISCORD_WEBHOOK_URL=your_discord_webhook_url
```

### 3.2 Install Webhook Server as a Windows Service via NSSM
Download [NSSM (Non-Sucking Service Manager)](https://nssm.cc/) to `C:\tools\nssm.exe` and execute:

```powershell
$nssm = "C:\tools\nssm.exe"
$serviceName = "PropKillerWebhook"
$pythonPath = "C:\Users\User\Documents\automated prop\.venv\Scripts\python.exe"
$scriptPath = "C:\Users\User\Documents\automated prop\execution\webhook_server.py"
$appDir = "C:\Users\User\Documents\automated prop"

# Install service
& $nssm install $serviceName $pythonPath "$scriptPath"
& $nssm set $serviceName AppDirectory "$appDir"
& $nssm set $serviceName AppStdout "$appDir\logs\webhook_stdout.log"
& $nssm set $serviceName AppStderr "$appDir\logs\webhook_stderr.log"
& $nssm set $serviceName AppRotateFiles 1
& $nssm set $serviceName AppRotateBytes 52428800 # 50 MB
& $nssm set $serviceName AppRestartDelay 5000

# Start service
Start-Service $serviceName
Get-Service $serviceName
```

### 3.3 Verify Webhook Health & Warmup Status
```powershell
curl.exe -X GET http://127.0.0.1:8765/health
```
Expected response:
```json
{
  "status": "healthy",
  "phase": "EVALUATION",
  "active_contract": "MNQZ26",
  "warmup_sessions": 2246,
  "warmup_ready": true,
  "is_early_close": false,
  "rollover_today": false
}
```

---

## 4. NinjaTrader 8 Configuration

1. **Install Strategy:**
   - Copy `execution/TradeifyExecutionManager.cs` into your NinjaTrader 8 Custom Strategies folder:
     `Documents\NinjaTrader 8\bin\Custom\Strategies\TradeifyExecutionManager.cs`
   - Open NinjaTrader 8 -> Tools -> New NinjaScript Editor -> Hit **F5 (Compile)**. Verify 0 compilation errors.

2. **Configure Tradovate / Rithmic Broker Connection:**
   - NinjaTrader 8 -> Connections -> Configure -> Add `Tradovate` or `Rithmic for NinjaTrader Brokerage`.
   - Credentials: Enter your Tradeify evaluation or funded account credentials.
   - Connect and verify green connection status light in the bottom-left corner.

3. **Deploy Strategy on Chart:**
   - Open a **1-Minute Chart** of CME MNQ front-month (e.g., `MNQ 12-26` or active quarterly contract).
   - Right-click chart -> **Strategies** -> Select `TradeifyExecutionManager`.
   - Configure strategy parameters:
     - `Account ID`: `Tradeify-50K-Growth-001` (or your Tradeify account name).
     - `Is Funded Account`: `False` for Evaluation ($53K target), `True` for Funded.
     - `Starting Balance`: `50000.0`.
     - `Internal Daily Stop Loss`: `600.0`.
     - `Funded Profit Trigger`: `880.0` (Consistency Governor trigger).
     - `Funded Profit Ceiling`: `920.0` (Hard cap protecting 35% rule).
     - `RT Commission Per MNQ`: `2.48`.
     - `Python Dispatcher URL`: `http://127.0.0.1:8765/webhook/signal`.
   - Set **Enabled = True** and click **OK**.

---

## 5. Account Lifecycle & Phase Transition Protocol

### 5.1 Evaluation Pass-on-Touch
When equity reaches **`$53,015.00`** intraday:
- `TradeifyExecutionManager.cs` automatically liquidates open legs immediately.
- Sets internal daily halt.
- Tradeify dashboard will show Evaluation target reached.

### 5.2 Transitioning from Evaluation to Funded
**CRITICAL:** Never allow an Evaluation `$53,015.00` HWM to leak into a newly funded account's Pre-Lock trailing floor state.

Run the automated transition script:
```powershell
.\.venv\Scripts\python.exe scripts/reset_account_phase.py --account-id Tradeify-50K-Growth-001
```
Or trigger via authenticated API call:
```powershell
curl.exe -X POST http://127.0.0.1:8765/admin/reset_to_funded `
  -H "X-API-KEY: tradeify_live_production_secret_key_2026" `
  -H "Content-Type: application/json" `
  -d '{"account_id": "Tradeify-50K-Growth-001"}'
```
This archives the Evaluation state and safely seeds:
```json
{
  "account_phase": "FUNDED",
  "current_equity": 50000.0,
  "eod_hwm": 50000.0,
  "liquidation_floor": 48000.0,
  "is_floor_locked": false,
  "qualifying_days_count": 0,
  "max_historical_daily_pnl": 0.0
}
```

In NinjaTrader 8:
- Open Strategy parameters on chart.
- Toggle `Is Funded Account = True`.
- Re-enable strategy.

---

## 6. Weekly Operations & Manual Reconciliation Protocol

Every Friday after CME market close (17:00 EST):

1. **Tradeify Dashboard Check:**
   - Log into Tradeify web portal.
   - Record exact ending Account Cash Balance and Trailing Drawdown Floor.
2. **Local State File Check:**
   - Open `state/tradeify_<ACCOUNT_ID>.json`.
   - Verify `current_equity` matches Tradeify cash balance within ±$5.00 (commission differential).
   - Verify `liquidation_floor` matches Tradeify displayed trailing floor.
3. **Rollover Check:**
   - Check if the upcoming week contains CME quarterly rollover (2nd Thursday of March, June, September, December).
   - If rollover occurs, verify `execution/symbol_mapper.py` displays the new active contract.

---

## 7. Mandatory 5-Day `Sim101` Burn-in Acceptance Checklist

Before enabling live capital on any Tradeify funded account, the system **MUST** pass 5 consecutive RTH trading sessions on NinjaTrader's `Sim101` simulator.

At the end of each session, run the shadow audit CLI:
```powershell
.\.venv\Scripts\python.exe scripts/shadow_audit_diff.py `
  --executions "C:\Users\User\Documents\NinjaTrader 8\outgoing\Sim101_Executions.csv" `
  --bars "C:\Users\User\Documents\automated prop\data\session_20261120_1m.csv" `
  --account-id "Sim101" `
  --output "reports/SHADOW_AUDIT_DAY1.md"
```

### Acceptance Criteria Checklist:
- [ ] **Zero Timestamp Drift:** 100% of signals matched between NT8 and Python with `0s` drift.
- [ ] **100% Direction & Strategy Tag Match:** Zero inverted trades (no BUY executed when Python said SELL).
- [ ] **100% Contract Sizing Match:** Exact MNQ contract count matches risk tiers (2–6 MNQ).
- [ ] **Slippage within Tolerance:** Average live execution slippage `< 0.75 pt` (baseline model is `0.50 pt`).
- [ ] **Server-Side Stop Confirmed:** Native broker `StopMarket` order visible in NT8 Orders grid on every fill.
- [ ] **Consistency Governor Functioning:** Simulated funded gains halted cleanly at `+$880.00` net daily PnL.
- [ ] **EOD Flatten Loop:** Account 100% flat at `15:44:50 EST` with zero working orders.
- [ ] **No Unhandled Exceptions:** Python stdout/stderr and NT8 Output window show 0 errors or memory leaks.

*Only when all 5 days display `Audit Status: PASS` may the strategy be attached to a live Tradeify account.*
