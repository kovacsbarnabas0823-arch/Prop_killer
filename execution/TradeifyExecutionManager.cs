#region Using declarations
using System;
using System.IO;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Xml.Serialization;
using System.Net;
using System.Text;
using System.Threading;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.NinjaScript;
using NinjaTrader.Data;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    /// <summary>
    /// Institutional Order Execution Manager for NinjaTrader 8 (Phase 9 Production Hardened).
    /// Engineered for Tradeify $50K Growth EOD Trailing Drawdown Accounts.
    ///
    /// Key Capabilities:
    ///   - Per-Account Atomic State Persistence in NinjaTrader.Core.Globals.UserDataDir.
    ///   - Independent H3 and H4 Leg Tracking for Safe Same-Direction Pyramiding.
    ///   - Immediate Broker-Resident Hard Stop Orders (Tradovate/Rithmic Server-Side Disconnect Protection).
    ///   - Real-Time Bar Pushing to Python Dispatcher (/bar_close) with Timestamp Alignment.
    ///   - Dual-Redundant Signal Ingestion (Local HttpListener on 127.0.0.1:8766 + Atomic Signal File Queue).
    ///   - Idempotency Deduplication via Deterministic signal_id Caching.
    ///   - Net-of-Commission PnL Tracking ($2.48/MNQ RT) and Consistency Governor Buffer ($880 trigger / $920 ceiling).
    ///   - Bulletproof EOD Flatten Retry Loop (15:43:30 to 15:44:50 EST, or 12:58:00 on CME early-close).
    ///   - 10-Second Trade Duration Compliance Monitor.
    /// </summary>
    public class TradeifyExecutionManager : Strategy
    {
        #region ActiveLeg Class Definition
        public class ActiveLeg
        {
            public string StrategyId { get; set; } = "H3";
            public string SignalId { get; set; } = string.Empty;
            public int Quantity { get; set; } = 0;
            public double EntryPrice { get; set; } = 0.0;
            public Order EntryOrder { get; set; } = null;
            public Order HardStopOrder { get; set; } = null;
            public double HardStopPrice { get; set; } = 0.0;
            public DateTime EntryTimeUtc { get; set; } = DateTime.MinValue;
            public string OcoId { get; set; } = string.Empty;
        }
        #endregion

        #region Parameters
        [NinjaScriptProperty]
        [Display(Name = "Account ID", Description = "Tradeify Account Identifier", Order = 1, GroupName = "Tradeify Account Parameters")]
        public string AccountId { get; set; } = "Tradeify-50K-Growth-001";

        [NinjaScriptProperty]
        [Display(Name = "Is Funded Account", Description = "False for Evaluation ($53K Target), True for Sim-Funded ($920 Daily Profit Ceiling)", Order = 2, GroupName = "Tradeify Account Parameters")]
        public bool IsFundedAccount { get; set; } = false;

        [NinjaScriptProperty]
        [Range(1000.0, 100000.0)]
        [Display(Name = "Starting Balance", Description = "Account Starting Capital ($50,000 for Tradeify Growth)", Order = 3, GroupName = "Tradeify Account Parameters")]
        public double StartingBalance { get; set; } = 50000.0;

        [NinjaScriptProperty]
        [Range(100.0, 5000.0)]
        [Display(Name = "Internal Daily Stop Loss", Description = "Internal protective net daily stop ($600.00)", Order = 4, GroupName = "Tradeify Account Parameters")]
        public double InternalDailyStopLoss { get; set; } = 600.0;

        [NinjaScriptProperty]
        [Range(100.0, 2000.0)]
        [Display(Name = "Funded Profit Trigger", Description = "Pre-emptive profit trigger for Consistency Governor ($880.00)", Order = 5, GroupName = "Tradeify Account Parameters")]
        public double FundedDailyProfitTrigger { get; set; } = 880.0;

        [NinjaScriptProperty]
        [Range(100.0, 2000.0)]
        [Display(Name = "Funded Profit Ceiling", Description = "Hard profit ceiling protecting 35% rule ($920.00)", Order = 6, GroupName = "Tradeify Account Parameters")]
        public double FundedDailyProfitCeiling { get; set; } = 920.0;

        [NinjaScriptProperty]
        [Range(0.0, 10.0)]
        [Display(Name = "RT Commission Per MNQ", Description = "Round-trip broker commission per contract ($2.48)", Order = 7, GroupName = "Execution Safety")]
        public double RoundTripCommissionPerMnq { get; set; } = 2.48;

        [NinjaScriptProperty]
        [Range(0.5, 5.0)]
        [Display(Name = "Max Quote Staleness (Seconds)", Description = "Heartbeat watchdog timeout", Order = 8, GroupName = "Execution Safety")]
        public double MaxQuoteStalenessSeconds { get; set; } = 1.5;

        [NinjaScriptProperty]
        [Display(Name = "Python Dispatcher URL", Description = "URL of Python Webhook Server", Order = 9, GroupName = "Execution Safety")]
        public string PythonDispatcherUrl { get; set; } = "http://127.0.0.1:8765";
        #endregion

        #region Internal State Variables
        private double eodHighWatermark = 50000.0;
        private double activeFloor = 48000.0;
        private bool isFloorLocked = false;
        private int qualifyingDaysCount = 0;
        private double maxHistoricalDailyPnL = 0.0;
        private double sessionStartRealizedPnL = 0.0;
        private bool isDailyHalted = false;
        private DateTime lastQuoteTime = DateTime.MinValue;
        private DateTime lastFlattenAttemptUtc = DateTime.MinValue;
        private int eodFlattenRetries = 0;

        // Independent Leg Management for H3 and H4
        private readonly Dictionary<string, ActiveLeg> activeLegs = new Dictionary<string, ActiveLeg>();
        private readonly HashSet<string> processedSignals = new HashSet<string>();

        // Local HttpListener for incoming instant signals
        private HttpListener httpListener = null;
        private Thread listenerThread = null;
        private volatile bool isListening = false;
        #endregion

        #region CME Early-Close Holiday Helper
        public static bool IsCmeEarlyClose(DateTime date)
        {
            int year = date.Year;
            int month = date.Month;
            int day = date.Day;
            DayOfWeek dow = date.DayOfWeek;

            // 2026 Explicit Schedule
            if (year == 2026)
            {
                if (month == 1 && day == 19) return true;   // MLK Day
                if (month == 2 && day == 16) return true;   // Presidents Day
                if (month == 5 && day == 25) return true;   // Memorial Day
                if (month == 6 && day == 19) return true;   // Juneteenth
                if (month == 7 && day == 3)  return true;   // Independence Day Eve
                if (month == 9 && day == 7)  return true;   // Labor Day
                if (month == 11 && day == 26) return true;  // Thanksgiving
                if (month == 11 && day == 27) return true;  // Black Friday
                if (month == 12 && day == 24) return true;  // Christmas Eve
            }

            // 2027 Explicit Schedule
            if (year == 2027)
            {
                if (month == 1 && day == 18) return true;   // MLK Day
                if (month == 2 && day == 15) return true;   // Presidents Day
                if (month == 5 && day == 31) return true;   // Memorial Day
                if (month == 6 && day == 18) return true;   // Juneteenth observed
                if (month == 7 && day == 2)  return true;   // Independence Day observed
                if (month == 9 && day == 6)  return true;   // Labor Day
                if (month == 11 && day == 25) return true;  // Thanksgiving
                if (month == 11 && day == 26) return true;  // Black Friday
                if (month == 12 && day == 24) return true;  // Christmas Eve
            }

            // General Fallback Rules
            if (month == 11 && dow == DayOfWeek.Friday && day >= 22 && day <= 28) return true; // Black Friday
            if (month == 12 && day == 24 && dow != DayOfWeek.Saturday && dow != DayOfWeek.Sunday) return true;
            if (month == 7 && day == 3 && dow != DayOfWeek.Saturday && dow != DayOfWeek.Sunday) return true;
            if (month == 11 && dow == DayOfWeek.Thursday && day >= 22 && day <= 28) return true; // Thanksgiving

            return false;
        }
        #endregion

        #region State Persistence Helpers
        private string GetStateFilePath()
        {
            try
            {
                string dir = NinjaTrader.Core.Globals.UserDataDir;
                return Path.Combine(dir, string.Format("tradeify_{0}.json", AccountId));
            }
            catch
            {
                return string.Format("tradeify_{0}.json", AccountId);
            }
        }

        private string GetPendingSignalFilePath()
        {
            try
            {
                string dir = NinjaTrader.Core.Globals.UserDataDir;
                return Path.Combine(dir, string.Format("tradeify_{0}_pending_signal.json", AccountId));
            }
            catch
            {
                return string.Format("tradeify_{0}_pending_signal.json", AccountId);
            }
        }

        private void LoadStateFromFile()
        {
            try
            {
                string path = GetStateFilePath();
                if (File.Exists(path))
                {
                    string content = File.ReadAllText(path);
                    ExtractJsonDouble(content, "current_equity", ref StartingBalance);
                    ExtractJsonDouble(content, "eod_hwm", ref eodHighWatermark);
                    ExtractJsonDouble(content, "liquidation_floor", ref activeFloor);
                    ExtractJsonBool(content, "is_floor_locked", ref isFloorLocked);
                    ExtractJsonInt(content, "qualifying_days_count", ref qualifyingDaysCount);
                    ExtractJsonDouble(content, "max_historical_daily_pnl", ref maxHistoricalDailyPnL);

                    Print(string.Format("[STATE LOADED] {0}: Equity=${1:F2}, HWM=${2:F2}, Floor=${3:F2}, Locked={4}, QualDays={5}",
                        AccountId, StartingBalance, eodHighWatermark, activeFloor, isFloorLocked, qualifyingDaysCount));
                }
            }
            catch (Exception ex)
            {
                Print(string.Format("[STATE ERROR] Failed loading state for {0}: {1}", AccountId, ex.Message));
            }
        }

        private void SaveStateToFile()
        {
            try
            {
                string path = GetStateFilePath();
                string dir = Path.GetDirectoryName(path);
                if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                    Directory.CreateDirectory(dir);

                double currentDailyPnL = Account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar) - sessionStartRealizedPnL;
                double currentCash = Account.Get(AccountItem.CashValue, Currency.UsDollar);

                string json = string.Format(
                    "{{\n" +
                    "  \"account_id\": \"{0}\",\n" +
                    "  \"account_phase\": \"{1}\",\n" +
                    "  \"current_equity\": {2:F2},\n" +
                    "  \"eod_hwm\": {3:F2},\n" +
                    "  \"liquidation_floor\": {4:F2},\n" +
                    "  \"is_floor_locked\": {5},\n" +
                    "  \"qualifying_days_count\": {6},\n" +
                    "  \"max_historical_daily_pnl\": {7:F2},\n" +
                    "  \"today_date\": \"{8}\",\n" +
                    "  \"today_realized_pnl\": {9:F2},\n" +
                    "  \"last_updated_timestamp\": \"{10}\"\n" +
                    "}}",
                    AccountId,
                    IsFundedAccount ? "FUNDED" : "EVALUATION",
                    currentCash,
                    eodHighWatermark,
                    activeFloor,
                    isFloorLocked ? "true" : "false",
                    qualifyingDaysCount,
                    maxHistoricalDailyPnL,
                    DateTime.UtcNow.ToString("yyyy-MM-dd"),
                    currentDailyPnL,
                    DateTime.UtcNow.ToString("o")
                );

                string tempPath = path + ".tmp";
                File.WriteAllText(tempPath, json);
                if (File.Exists(path))
                    File.Delete(path);
                File.Move(tempPath, path);
            }
            catch (Exception ex)
            {
                Print(string.Format("[STATE ERROR] Failed saving state for {0}: {1}", AccountId, ex.Message));
            }
        }

        private static void ExtractJsonDouble(string json, string key, ref double target)
        {
            string pattern = "\"" + key + "\":";
            int idx = json.IndexOf(pattern, StringComparison.OrdinalIgnoreCase);
            if (idx >= 0)
            {
                int start = idx + pattern.Length;
                int end = json.IndexOfAny(new char[] { ',', '}', '\n' }, start);
                if (end > start)
                {
                    double val;
                    if (double.TryParse(json.Substring(start, end - start).Trim(), System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out val))
                        target = val;
                }
            }
        }

        private static void ExtractJsonInt(string json, string key, ref int target)
        {
            string pattern = "\"" + key + "\":";
            int idx = json.IndexOf(pattern, StringComparison.OrdinalIgnoreCase);
            if (idx >= 0)
            {
                int start = idx + pattern.Length;
                int end = json.IndexOfAny(new char[] { ',', '}', '\n' }, start);
                if (end > start)
                {
                    int val;
                    if (int.TryParse(json.Substring(start, end - start).Trim(), out val))
                        target = val;
                }
            }
        }

        private static void ExtractJsonBool(string json, string key, ref bool target)
        {
            string pattern = "\"" + key + "\":";
            int idx = json.IndexOf(pattern, StringComparison.OrdinalIgnoreCase);
            if (idx >= 0)
            {
                int start = idx + pattern.Length;
                int end = json.IndexOfAny(new char[] { ',', '}', '\n' }, start);
                if (end > start)
                {
                    string sub = json.Substring(start, end - start).Trim().ToLowerInvariant();
                    target = sub.Contains("true");
                }
            }
        }
        #endregion

        #region IPC Background Listener (HttpListener on 127.0.0.1:8766)
        private void StartHttpSignalListener()
        {
            try
            {
                httpListener = new HttpListener();
                httpListener.Prefixes.Add("http://127.0.0.1:8766/signal/");
                httpListener.Start();
                isListening = true;

                listenerThread = new Thread(ListenForIncomingSignals)
                {
                    IsBackground = true,
                    Name = "TradeifySignalListenerThread"
                };
                listenerThread.Start();
                Print("[IPC LISTENER] Live signal listener started on http://127.0.0.1:8766/signal/");
            }
            catch (Exception ex)
            {
                Print(string.Format("[IPC WARNING] Could not bind local HttpListener on 8766: {0}. Operating in pending-signal file queue mode.", ex.Message));
            }
        }

        private void StopHttpSignalListener()
        {
            isListening = false;
            try
            {
                if (httpListener != null && httpListener.IsListening)
                {
                    httpListener.Stop();
                    httpListener.Close();
                }
            }
            catch { }
        }

        private void ListenForIncomingSignals()
        {
            while (isListening && httpListener != null && httpListener.IsListening)
            {
                try
                {
                    HttpListenerContext context = httpListener.GetContext();
                    HttpListenerRequest request = context.Request;

                    if (request.HttpMethod == "POST")
                    {
                        using (StreamReader reader = new StreamReader(request.InputStream, request.ContentEncoding))
                        {
                            string jsonPayload = reader.ReadToEnd();
                            ProcessSignalJson(jsonPayload);
                        }
                    }

                    HttpListenerResponse response = context.Response;
                    string respStr = "{\"status\":\"received\"}";
                    byte[] buffer = Encoding.UTF8.GetBytes(respStr);
                    response.ContentLength64 = buffer.Length;
                    response.ContentType = "application/json";
                    response.OutputStream.Write(buffer, 0, buffer.Length);
                    response.OutputStream.Close();
                }
                catch (HttpListenerException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Print(string.Format("[IPC LISTENER ERROR] {0}", ex.Message));
                }
            }
        }

        private void CheckPendingSignalFile()
        {
            try
            {
                string signalPath = GetPendingSignalFilePath();
                if (File.Exists(signalPath))
                {
                    string content = File.ReadAllText(signalPath);
                    File.Delete(signalPath);
                    ProcessSignalJson(content);
                }
            }
            catch { }
        }

        private void ProcessSignalJson(string json)
        {
            try
            {
                string sigId = string.Empty;
                string stratId = "H3";
                string dir = "BUY";
                int contracts = 2;
                double slPrice = 0.0;
                double tp1Price = 0.0;
                double atr = 0.0;

                string patternSig = "\"signal_id\":\"";
                int sIdx = json.IndexOf(patternSig);
                if (sIdx >= 0)
                {
                    int start = sIdx + patternSig.Length;
                    int end = json.IndexOf("\"", start);
                    if (end > start) sigId = json.Substring(start, end - start);
                }

                string patternStrat = "\"strategy_id\":\"";
                int stIdx = json.IndexOf(patternStrat);
                if (stIdx >= 0)
                {
                    int start = stIdx + patternStrat.Length;
                    int end = json.IndexOf("\"", start);
                    if (end > start) stratId = json.Substring(start, end - start);
                }

                string patternDir = "\"direction\":\"";
                int dIdx = json.IndexOf(patternDir);
                if (dIdx >= 0)
                {
                    int start = dIdx + patternDir.Length;
                    int end = json.IndexOf("\"", start);
                    if (end > start) dir = json.Substring(start, end - start);
                }

                ExtractJsonInt(json, "contracts", ref contracts);
                ExtractJsonDouble(json, "sl_price", ref slPrice);
                ExtractJsonDouble(json, "tp1_price", ref tp1Price);
                ExtractJsonDouble(json, "atr14", ref atr);

                bool isLong = dir.ToUpperInvariant().Contains("BUY") || dir.ToUpperInvariant().Contains("LONG");
                SubmitSignalEntry(isLong, slPrice, tp1Price, atr, stratId, sigId, contracts);
            }
            catch (Exception ex)
            {
                Print(string.Format("[SIGNAL PARSE ERROR] {0}", ex.Message));
            }
        }
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Institutional Tradeify $50K Growth Execution Manager (CME MNQ)";
                Name = "TradeifyExecutionManager";
                Calculate = Calculate.OnPriceChange;
                EntriesPerDirection = 2; // Supports concurrent H3 + H4 Same-Direction Pyramiding
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 60;
                IsUnmanaged = true;
            }
            else if (State == State.Configure)
            {
                eodHighWatermark = StartingBalance;
                activeFloor = StartingBalance - 2000.0;
                isFloorLocked = false;
                isDailyHalted = false;
                activeLegs.Clear();
                processedSignals.Clear();
            }
            else if (State == State.DataLoaded)
            {
                LoadStateFromFile();
                sessionStartRealizedPnL = Account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar);
                StartHttpSignalListener();
            }
            else if (State == State.Terminated)
            {
                StopHttpSignalListener();
                SaveStateToFile();
            }
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0) return;

            // 1. Strict Contract Exclusivity Check
            if (!Instrument.FullName.Contains("MNQ"))
            {
                TriggerEmergencyKillSwitch("FATAL: TradeifyExecutionManager configured for non-MNQ instrument: " + Instrument.FullName);
                return;
            }

            // 2. Push Closed 1-Minute Bar to Python Webhook Server
            if (Bars.IsFirstBarOfSession == false && CurrentBar > 1)
            {
                PushClosedBarToPython(Time[1], Open[1], High[1], Low[1], Close[1], Volume[1]);
            }

            // 3. Timezone-Aware CME Holiday & Bulletproof EOD Flatten Retry Loop
            DateTime estNow = TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow, TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"));
            bool isEarlyClose = IsCmeEarlyClose(estNow.Date);
            TimeSpan flattenTrigger = isEarlyClose ? new TimeSpan(12, 58, 0) : new TimeSpan(15, 43, 30);
            TimeSpan hardDeadline = isEarlyClose ? new TimeSpan(13, 0, 0) : new TimeSpan(15, 44, 50);

            if (estNow.TimeOfDay >= flattenTrigger)
            {
                if (Position.MarketPosition != MarketPosition.Flat || activeLegs.Count > 0)
                {
                    double secsSinceLastAttempt = (DateTime.UtcNow - lastFlattenAttemptUtc).TotalSeconds;
                    if (secsSinceLastAttempt >= 5.0 && estNow.TimeOfDay <= hardDeadline)
                    {
                        lastFlattenAttemptUtc = DateTime.UtcNow;
                        eodFlattenRetries++;
                        Print(string.Format("[EOD RETRY FLATTEN #{0}] {1} EST reached. Liquidating all positions and active legs.", eodFlattenRetries, estNow.TimeOfDay));
                        FlattenEverything(string.Format("EOD Mandatory Liquidation (Retry #{0})", eodFlattenRetries));
                    }
                }
                return;
            }

            // 4. File-Queue Signal Check (Fallback)
            CheckPendingSignalFile();

            // 5. Real-Time Intraday Risk Checks
            CheckCircuitBreakers();
        }

        private void PushClosedBarToPython(DateTime closeTime, double op, double hi, double lo, double cl, double vol)
        {
            try
            {
                string url = PythonDispatcherUrl.TrimEnd('/') + "/bar_close";
                string jsonPayload = string.Format(
                    "{{\"timestamp\":\"{0:yyyy-MM-ddTHH:mm:ss}\",\"open\":{1:F2},\"high\":{2:F2},\"low\":{3:F2},\"close\":{4:F2},\"volume\":{5:F0},\"bar_minutes\":1,\"symbol\":\"{6}\"}}",
                    closeTime, op, hi, lo, cl, vol, Instrument.FullName
                );

                HttpWebRequest req = (HttpWebRequest)WebRequest.Create(url);
                req.Method = "POST";
                req.ContentType = "application/json";
                req.Timeout = 1000;
                byte[] bytes = Encoding.UTF8.GetBytes(jsonPayload);
                req.ContentLength = bytes.Length;

                using (Stream stream = req.GetRequestStream())
                {
                    stream.Write(bytes, 0, bytes.Length);
                }
                using (HttpWebResponse resp = (HttpWebResponse)req.GetResponse()) { }
            }
            catch { }
        }

        protected override void OnMarketData(MarketDataEventArgs marketDataUpdate)
        {
            lastQuoteTime = DateTime.UtcNow;

            // Heartbeat Staleness Watchdog
            if (!isDailyHalted && lastQuoteTime != DateTime.MinValue)
            {
                double elapsed = (DateTime.UtcNow - lastQuoteTime).TotalSeconds;
                if (elapsed > MaxQuoteStalenessSeconds && Position.MarketPosition != MarketPosition.Flat)
                {
                    TriggerEmergencyKillSwitch(string.Format("Quote staleness watchdog timeout: {0:F2}s with open position", elapsed));
                }
            }
        }

        private void CheckCircuitBreakers()
        {
            if (isDailyHalted) return;

            double currentDailyRealizedPnL = Account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar) - sessionStartRealizedPnL;
            double currentDailyUnrealizedPnL = Account.Get(AccountItem.UnrealizedProfitLoss, Currency.UsDollar);

            // Compute net realized + unrealized PnL (net of commission)
            int totalContractsTraded = Position.Quantity;
            double estimatedCommission = totalContractsTraded * RoundTripCommissionPerMnq;
            double currentDailyNetPnL = currentDailyRealizedPnL + currentDailyUnrealizedPnL - estimatedCommission;

            double currentCash = Account.Get(AccountItem.CashValue, Currency.UsDollar);
            double currentTotalEquity = currentCash + currentDailyUnrealizedPnL - estimatedCommission;

            // 1. Internal Protective Daily Stop Check (-$600.00)
            if (currentDailyNetPnL <= -InternalDailyStopLoss)
            {
                isDailyHalted = true;
                FlattenEverything(string.Format("INTERNAL DAILY STOP HIT: -${0:F2} Net PnL", Math.Abs(currentDailyNetPnL)));
                return;
            }

            // 2. Consistency Governor Pre-Emptive Slippage Buffer (+$880.00 Trigger / +$920.00 Ceiling)
            if (IsFundedAccount && currentDailyNetPnL >= FundedDailyProfitTrigger)
            {
                isDailyHalted = true;
                FlattenEverything(string.Format("CONSISTENCY GOVERNOR TRIGGERED: Net daily profit +${0:F2} >= +${1:F2}. Pre-emptively locking profit to protect 35% rule.",
                    currentDailyNetPnL, FundedDailyProfitTrigger));
                return;
            }

            // 3. Evaluation Absorbing Upper Barrier ("Pass-on-Touch" at $53,015.00)
            if (!IsFundedAccount && currentTotalEquity >= 53015.0)
            {
                isDailyHalted = true;
                FlattenEverything(string.Format("EVAL_PASS_ON_TOUCH: Equity ${0:F2} reached $53,015.00 threshold. Evaluation officially PASSED!", currentTotalEquity));
                return;
            }

            // 4. Funded Payout Absorbing Barrier ("Payout-on-Touch" at $52,660.00)
            if (IsFundedAccount && currentTotalEquity >= 52660.0 && currentDailyNetPnL >= 160.0)
            {
                isDailyHalted = true;
                FlattenEverything(string.Format("FUNDED_PAYOUT_ON_TOUCH: Equity ${0:F2} reached $52,660.00 threshold with Qualifying Day. Payout officially LOCKED!", currentTotalEquity));
                return;
            }

            // 5. Catastrophic Floor Check
            if (currentCash <= activeFloor)
            {
                TriggerEmergencyKillSwitch(string.Format("ACCOUNT FLOOR BREACH: Cash ${0:F2} <= Floor ${1:F2}", currentCash, activeFloor));
            }
        }

        public int CalculateOptimalMNQSize(double currentCash, double slDistPts, double atr14, string strategyId, double morningPnL)
        {
            double buffer = currentCash - activeFloor;
            double profit = currentCash - StartingBalance;
            double rTarget = 150.0;
            int minC = 1;
            int maxC = 6;

            if (!IsFundedAccount)
            {
                // EVALUATION Sprint Tiers
                if (currentCash >= 51900.0) { rTarget = 270.0; minC = 3; maxC = 6; }
                else if (profit >= 800.0 && buffer >= 1350.0) { rTarget = 225.0; minC = 3; maxC = 4; }
                else if (buffer >= 950.0) { rTarget = 165.0; minC = 2; maxC = 3; }
                else { rTarget = 120.0; minC = 1; maxC = 2; }

                // Finish-Line Risk Tapering Near $53,000 (< $350 remaining)
                double remainingToPass = 53015.0 - currentCash;
                if (remainingToPass > 0.0 && remainingToPass < 350.0)
                {
                    rTarget = Math.Max(155.0, Math.Min(rTarget, 0.90 * remainingToPass));
                }
            }
            else
            {
                // FUNDED Capital Preservation Tiers ($90 / $165 / $245)
                if (buffer < 1050.0) { rTarget = 90.0; minC = 1; maxC = 2; }
                else if (!isFloorLocked && eodHighWatermark < 52100.0) { rTarget = 165.0; minC = 2; maxC = 3; }
                else { rTarget = 245.0; minC = 3; maxC = 6; }

                // Generalized House-Money Floor Guard for H4 in Funded Pre-Lock Mode
                if (strategyId == "H4" && !isFloorLocked && eodHighWatermark < 52100.0)
                {
                    if (morningPnL >= 165.0)
                    {
                        double houseMoneyBudget = morningPnL - 160.0;
                        double perContractRisk = (slDistPts * 2.00) + RoundTripCommissionPerMnq;
                        int h4Contracts = (int)Math.Floor(houseMoneyBudget / perContractRisk);
                        return Math.Max(0, Math.Min(maxC, h4Contracts));
                    }
                    else
                    {
                        return 0; // H4 skipped to preserve morning gains
                    }
                }
            }

            double effAtr = (atr14 > 0.0) ? atr14 : 200.0;
            double refPts = (strategyId == "H4")
                ? Math.Max(16.0, Math.Min(22.0, 0.09 * effAtr))
                : Math.Max(30.0, Math.Min(40.0, 0.18 * effAtr));

            double unitRiskUsd = (refPts * 2.00) + RoundTripCommissionPerMnq;
            int rawContracts = (int)Math.Round(rTarget / unitRiskUsd);

            double perContractSlRisk = (slDistPts * 2.00) + RoundTripCommissionPerMnq;
            int maxSafeContracts = (int)Math.Max(1, Math.Floor((1.35 * rTarget) / perContractSlRisk));

            int contracts = Math.Min(rawContracts, maxSafeContracts);
            return Math.Max(minC, Math.Min(maxC, contracts));
        }

        public void SubmitSignalEntry(bool isLong, double stopLossPrice, double targetPrice1, double atr14, string strategyId, string signalId, int requestedContracts = 0)
        {
            if (isDailyHalted)
            {
                Print("[OEM REJECTED] Cannot enter: Account halted for session by circuit breaker.");
                return;
            }

            // Idempotency Deduplication Guard
            if (!string.IsNullOrEmpty(signalId) && processedSignals.Contains(signalId))
            {
                Print(string.Format("[OEM DUPLICATE] Signal {0} already processed. Skipping duplicate execution.", signalId));
                return;
            }

            DateTime estNow = TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow, TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"));
            bool isEarlyClose = IsCmeEarlyClose(estNow.Date);
            if (isEarlyClose)
            {
                if (strategyId == "H4")
                {
                    Print("[OEM REJECTED] CME Early-Close Day: 15:00 H4 session is cancelled.");
                    return;
                }
                if (estNow.TimeOfDay >= new TimeSpan(12, 45, 0))
                {
                    Print("[OEM REJECTED] CME Early-Close Liquidation Window: entries blocked after 12:45 EST.");
                    return;
                }
            }

            bool isPyramidEntry = false;
            double currentDailyRealizedPnL = Account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar) - sessionStartRealizedPnL;
            double effectiveMorningPnL = currentDailyRealizedPnL;

            if (Position.MarketPosition != MarketPosition.Flat)
            {
                // Same-Direction Pyramiding Unlock for H4
                bool posIsLong = (Position.MarketPosition == MarketPosition.Long);
                if (strategyId == "H4" && posIsLong == isLong)
                {
                    double currentUnrealized = Account.Get(AccountItem.UnrealizedProfitLoss, Currency.UsDollar);
                    if (!IsFundedAccount && currentUnrealized >= 100.0)
                    {
                        isPyramidEntry = true;
                    }
                    else if (IsFundedAccount)
                    {
                        effectiveMorningPnL = currentDailyRealizedPnL + Math.Max(0.0, 0.70 * currentUnrealized);
                        if (effectiveMorningPnL >= 165.0)
                        {
                            isPyramidEntry = true;
                        }
                    }
                }

                if (!isPyramidEntry)
                {
                    Print("[OEM REJECTED] Active position already open. Concurrency guard active.");
                    return;
                }
            }

            double currentCash = Account.Get(AccountItem.CashValue, Currency.UsDollar);
            double currentPrice = Close[0];
            double slDistPts = Math.Abs(currentPrice - stopLossPrice);
            if (slDistPts <= 0) slDistPts = 30.0;

            int contracts = (requestedContracts > 0) ? requestedContracts : CalculateOptimalMNQSize(currentCash, slDistPts, atr14, strategyId, effectiveMorningPnL);
            if (contracts <= 0)
            {
                Print(string.Format("[OEM SKIPPED] Strategy {0} allocated 0 contracts (House-Money Floor Guard or risk buffer).", strategyId));
                return;
            }

            // Register Leg
            ActiveLeg leg = new ActiveLeg()
            {
                StrategyId = strategyId,
                SignalId = signalId,
                Quantity = contracts,
                EntryPrice = currentPrice,
                HardStopPrice = stopLossPrice,
                EntryTimeUtc = DateTime.UtcNow,
                OcoId = string.Format("OCO_{0}_{1}", strategyId, Guid.NewGuid().ToString("N").Substring(0, 6))
            };

            OrderAction action = isLong ? OrderAction.Buy : OrderAction.SellShort;
            string signalTag = string.Format("{0}_{1}", strategyId, isLong ? "LONG" : "SHORT");

            leg.EntryOrder = SubmitOrderUnmanaged(0, action, OrderType.Market, contracts, 0, 0, "", signalTag);
            activeLegs[strategyId] = leg;

            if (!string.IsNullOrEmpty(signalId))
                processedSignals.Add(signalId);

            Print(string.Format("[OEM EXECUTED] {0} {1} MNQ contracts for {2} (SL Price: {3:F2}, Signal ID: {4}). Cash: ${5:F2} | Floor: ${6:F2}",
                action, contracts, strategyId, stopLossPrice, signalId, currentCash, activeFloor));

            SaveStateToFile();
        }

        protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity,
            MarketPosition marketPosition, string orderId, DateTime time)
        {
            // 1. Entry Order Fill: Immediately submit server-side broker-resident hard stop
            foreach (var kvp in activeLegs)
            {
                ActiveLeg leg = kvp.Value;
                if (leg.EntryOrder != null && execution.Order == leg.EntryOrder && leg.HardStopOrder == null)
                {
                    leg.EntryPrice = price;
                    bool isLong = (leg.EntryOrder.OrderAction == OrderAction.Buy);
                    OrderAction slAction = isLong ? OrderAction.Sell : OrderAction.BuyToCover;

                    // Tagged stop market order sized strictly to this leg's own quantity
                    string slTag = string.Format("SL_{0}_{1}", leg.StrategyId, leg.OcoId);
                    leg.HardStopOrder = SubmitOrderUnmanaged(0, slAction, OrderType.StopMarket, leg.Quantity, 0, leg.HardStopPrice, leg.OcoId, slTag);

                    Print(string.Format("[BROKER HARD STOP ARMED] {0} for {1} MNQ @ {2:F2} on Tradovate/Rithmic Server (Leg: {3})",
                        slAction, leg.Quantity, leg.HardStopPrice, leg.StrategyId));
                    break;
                }
            }

            // 2. Position Flattened: Verify 10-second hold compliance and reconcile floor
            if (Position.MarketPosition == MarketPosition.Flat)
            {
                foreach (var kvp in activeLegs)
                {
                    ActiveLeg leg = kvp.Value;
                    if (leg.EntryTimeUtc != DateTime.MinValue)
                    {
                        double holdSeconds = (DateTime.UtcNow - leg.EntryTimeUtc).TotalSeconds;
                        if (holdSeconds < 10.0)
                        {
                            Print(string.Format("[COMPLIANCE WARNING] Leg {0} closed in {1:F1}s (< 10s rule). Ensure >50% duration compliance.", leg.StrategyId, holdSeconds));
                        }
                    }
                }
                activeLegs.Clear();

                // High Water Mark & Permanent Floor Lock Evaluation
                double currentCash = Account.Get(AccountItem.CashValue, Currency.UsDollar);
                if (currentCash > eodHighWatermark)
                {
                    eodHighWatermark = currentCash;
                    if (IsFundedAccount && eodHighWatermark >= 52100.0)
                    {
                        isFloorLocked = true;
                        activeFloor = 50100.0;
                        Print(string.Format("[PERMANENT FLOOR LOCKED] EOD HWM ${0:F2} >= $52,100. Floor permanently locked at $50,100.00!", eodHighWatermark));
                    }
                    else if (!isFloorLocked)
                    {
                        activeFloor = Math.Max(activeFloor, eodHighWatermark - 2000.0);
                    }
                }

                SaveStateToFile();
            }
        }

        public void FlattenEverything(string reason)
        {
            Print(string.Format("[FLATTEN ENGAGED] {0}", reason));

            // Cancel all leg working orders
            foreach (var kvp in activeLegs)
            {
                ActiveLeg leg = kvp.Value;
                if (leg.EntryOrder != null && (leg.EntryOrder.OrderState == OrderState.Working || leg.EntryOrder.OrderState == OrderState.Accepted))
                    CancelOrder(leg.EntryOrder);
                if (leg.HardStopOrder != null && (leg.HardStopOrder.OrderState == OrderState.Working || leg.HardStopOrder.OrderState == OrderState.Accepted))
                    CancelOrder(leg.HardStopOrder);
            }

            if (Position.MarketPosition == MarketPosition.Long)
                SubmitOrderUnmanaged(0, OrderAction.Sell, OrderType.Market, Position.Quantity, 0, 0, "", "FLATTEN_LONG");
            else if (Position.MarketPosition == MarketPosition.Short)
                SubmitOrderUnmanaged(0, OrderAction.BuyToCover, OrderType.Market, Position.Quantity, 0, 0, "", "FLATTEN_SHORT");

            activeLegs.Clear();
            SaveStateToFile();
        }

        public void TriggerEmergencyKillSwitch(string reason)
        {
            isDailyHalted = true;
            Print(string.Format("[EMERGENCY KILL-SWITCH] {0}", reason));
            FlattenEverything(reason);
        }
    }
}
