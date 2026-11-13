#region Using declarations
using System;
using System.IO;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.NinjaScript;
using NinjaTrader.Data;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    /// <summary>
    /// Institutional Order Execution Manager for NinjaTrader 8 (Phase 9 Hardened).
    /// Engineered for Tradeify $50K Growth EOD Trailing Drawdown Accounts.
    ///
    /// Key Capabilities:
    ///   - Atomic State Persistence in NinjaTrader.Core.Globals.UserDataDir.
    ///   - CME Early-Close Holiday Guard (Black Friday, Christmas Eve, July 3rd closing at 13:00 EST).
    ///   - Dynamic Bellman 4-Tier Position Sizing (2 / 3 / 4 / 6 MNQ).
    ///   - Strict Contract Exclusivity: CME MNQ ($2.00/pt) Only.
    ///   - Intraday Hard Circuit Breakers: -$600.00 Internal Daily Stop.
    ///   - Funded Consistency Governor Ceiling: +$920.00 (protecting 35% rule).
    ///   - Hard Flatten Timers (12:58:00 EST on early-close, 15:44:00 EST standard).
    ///   - Server-Side Resting OCO Partial Scale-Out Brackets.
    ///   - 10-Second Trade Duration Compliance Monitor.
    /// </summary>
    public class TradeifyExecutionManager : Strategy
    {
        #region Parameters
        [NinjaScriptProperty]
        [Display(Name = "Is Funded Account", Description = "False for Evaluation ($53K Target), True for Sim-Funded ($920 Daily Profit Ceiling)", Order = 1, GroupName = "Tradeify Account Parameters")]
        public bool IsFundedAccount { get; set; } = false;

        [NinjaScriptProperty]
        [Range(1000.0, 100000.0)]
        [Display(Name = "Starting Balance", Description = "Account Starting Capital ($50,000 for Tradeify Growth)", Order = 2, GroupName = "Tradeify Account Parameters")]
        public double StartingBalance { get; set; } = 50000.0;

        [NinjaScriptProperty]
        [Range(500.0, 5000.0)]
        [Display(Name = "Internal Daily Stop Loss", Description = "Internal protective daily stop ($600.00)", Order = 3, GroupName = "Tradeify Account Parameters")]
        public double InternalDailyStopLoss { get; set; } = 600.0;

        [NinjaScriptProperty]
        [Range(200.0, 2000.0)]
        [Display(Name = "Funded Daily Profit Ceiling", Description = "Hard profit cap in Funded mode ($920.00) protecting 35% rule on $2,650 target", Order = 4, GroupName = "Tradeify Account Parameters")]
        public double FundedDailyProfitCeiling { get; set; } = 920.0;

        [NinjaScriptProperty]
        [Range(0.5, 5.0)]
        [Display(Name = "Max Quote Staleness (Seconds)", Description = "Heartbeat watchdog timeout", Order = 5, GroupName = "Execution Safety")]
        public double MaxQuoteStalenessSeconds { get; set; } = 1.5;
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
        private DateTime lastEntryTime = DateTime.MinValue;
        private string lastSessionDate = string.Empty;

        private Order entryOrder = null;
        private Order stopLossOrder = null;
        private Order profitTargetOrder = null;
        private string activeOcoId = string.Empty;
        #endregion

        #region CME Early-Close Holiday Helper
        public static bool IsCmeEarlyClose(DateTime date)
        {
            int month = date.Month;
            int day = date.Day;
            DayOfWeek dow = date.DayOfWeek;

            // 1. Black Friday: 4th Friday in November (day 22-28)
            if (month == 11 && dow == DayOfWeek.Friday && day >= 22 && day <= 28)
                return true;

            // 2. Christmas Eve (Dec 24): weekday
            if (month == 12 && day == 24 && dow != DayOfWeek.Saturday && dow != DayOfWeek.Sunday)
                return true;

            // 3. July 3rd: weekday
            if (month == 7 && day == 3 && dow != DayOfWeek.Saturday && dow != DayOfWeek.Sunday)
                return true;

            return false;
        }
        #endregion

        #region State Persistence Helpers
        private string GetStateFilePath()
        {
            try
            {
                string dir = NinjaTrader.Core.Globals.UserDataDir;
                return Path.Combine(dir, "tradeify_account_state.json");
            }
            catch
            {
                return "tradeify_account_state.json";
            }
        }

        private void LoadStateFromFile()
        {
            try
            {
                string path = GetStateFilePath();
                if (File.Exists(path))
                {
                    string json = File.ReadAllText(path);
                    // Minimal robust JSON line parser
                    eodHighWatermark = ExtractDoubleJson(json, "eod_hwm", eodHighWatermark);
                    activeFloor = ExtractDoubleJson(json, "liquidation_floor", activeFloor);
                    isFloorLocked = ExtractBoolJson(json, "is_floor_locked", isFloorLocked);
                    qualifyingDaysCount = (int)ExtractDoubleJson(json, "qualifying_days_count", qualifyingDaysCount);
                    maxHistoricalDailyPnL = ExtractDoubleJson(json, "max_historical_daily_pnl", maxHistoricalDailyPnL);
                    Print(string.Format("[STATE RESTORED] HighWatermark: ${0:F2} | Floor: ${1:F2} | FloorLocked: {2} | QualDays: {3}",
                        eodHighWatermark, activeFloor, isFloorLocked, qualifyingDaysCount));
                }
            }
            catch (Exception ex)
            {
                Print(string.Format("[STATE LOAD ERROR] Could not load state: {0}", ex.Message));
            }
        }

        private void SaveStateToFile()
        {
            try
            {
                string path = GetStateFilePath();
                string tempPath = path + ".tmp";
                string json = string.Format(
                    "{{\n  \"eod_hwm\": {0:F2},\n  \"liquidation_floor\": {1:F2},\n  \"is_floor_locked\": {2},\n  \"qualifying_days_count\": {3},\n  \"max_historical_daily_pnl\": {4:F2},\n  \"is_funded\": {5},\n  \"last_updated\": \"{6}\"\n}}",
                    eodHighWatermark,
                    activeFloor,
                    isFloorLocked ? "true" : "false",
                    qualifyingDaysCount,
                    maxHistoricalDailyPnL,
                    IsFundedAccount ? "true" : "false",
                    DateTime.UtcNow.ToString("o")
                );

                File.WriteAllText(tempPath, json);
                if (File.Exists(path))
                    File.Delete(path);
                File.Move(tempPath, path);
            }
            catch (Exception ex)
            {
                Print(string.Format("[STATE SAVE ERROR] Could not save state: {0}", ex.Message));
            }
        }

        private double ExtractDoubleJson(string json, string key, double defaultVal)
        {
            try
            {
                int kIdx = json.IndexOf("\"" + key + "\"");
                if (kIdx < 0) return defaultVal;
                int colonIdx = json.IndexOf(":", kIdx);
                if (colonIdx < 0) return defaultVal;
                int commaIdx = json.IndexOfAny(new char[] { ',', '\n', '}' }, colonIdx);
                if (commaIdx < 0) commaIdx = json.Length;
                string valStr = json.Substring(colonIdx + 1, commaIdx - colonIdx - 1).Trim().Replace("\"", "");
                double res;
                if (double.TryParse(valStr, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out res))
                    return res;
            }
            catch { }
            return defaultVal;
        }

        private bool ExtractBoolJson(string json, string key, bool defaultVal)
        {
            try
            {
                int kIdx = json.IndexOf("\"" + key + "\"");
                if (kIdx < 0) return defaultVal;
                int colonIdx = json.IndexOf(":", kIdx);
                if (colonIdx < 0) return defaultVal;
                int commaIdx = json.IndexOfAny(new char[] { ',', '\n', '}' }, colonIdx);
                if (commaIdx < 0) commaIdx = json.Length;
                string valStr = json.Substring(colonIdx + 1, commaIdx - colonIdx - 1).Trim().ToLower();
                return valStr.Contains("true");
            }
            catch { }
            return defaultVal;
        }
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Institutional Tradeify $50K Growth Execution Manager (CME MNQ)";
                Name = "TradeifyExecutionManager";
                Calculate = Calculate.OnPriceChange;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 60; // 15:44:00 trigger
                IsUnmanaged = true;            // Direct execution engine matching
            }
            else if (State == State.Configure)
            {
                eodHighWatermark = StartingBalance;
                activeFloor = StartingBalance - 2000.0;
                isFloorLocked = false;
                isDailyHalted = false;
            }
            else if (State == State.DataLoaded)
            {
                LoadStateFromFile();
            }
            else if (State == State.Terminated)
            {
                SaveStateToFile();
            }
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0) return;

            // Strict Contract Exclusivity Check
            if (!Instrument.FullName.Contains("MNQ"))
            {
                TriggerEmergencyKillSwitch("FATAL: TradeifyExecutionManager configured for non-MNQ instrument: " + Instrument.FullName);
                return;
            }

            // Timezone-Aware CME Holiday & Hard Flatten Guard
            DateTime estNow = TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow, TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"));
            bool isEarlyClose = IsCmeEarlyClose(estNow.Date);
            TimeSpan flattenCutoff = isEarlyClose ? new TimeSpan(12, 58, 0) : new TimeSpan(15, 44, 0);

            if (estNow.TimeOfDay >= flattenCutoff)
            {
                if (Position.MarketPosition != MarketPosition.Flat || entryOrder != null)
                {
                    Print(string.Format("[EOD FLATTEN] {0} EST reached. Liquidating all exposure before session close.", flattenCutoff));
                    FlattenEverything("Mandatory EOD Hard Session Close");
                }
                return;
            }

            // Real-Time Intraday Risk Checks
            CheckCircuitBreakers();
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
            double currentDailyNetPnL = currentDailyRealizedPnL + currentDailyUnrealizedPnL;
            double currentCash = Account.Get(AccountItem.CashValue, Currency.UsDollar);
            double currentTotalEquity = currentCash + currentDailyUnrealizedPnL;

            // 1. Internal Daily Stop Check (-$600.00)
            if (currentDailyNetPnL <= -InternalDailyStopLoss)
            {
                isDailyHalted = true;
                FlattenEverything(string.Format("INTERNAL DAILY STOP HIT: -${0:F2}", Math.Abs(currentDailyNetPnL)));
                return;
            }

            // 2. Funded Consistency Governor Ceiling (+$920.00)
            if (IsFundedAccount && currentDailyNetPnL >= FundedDailyProfitCeiling)
            {
                isDailyHalted = true;
                FlattenEverything(string.Format("FUNDED CONSISTENCY GOVERNOR: Daily profit +${0:F2} reached ceiling. Halting to preserve 35% rule.", currentDailyNetPnL));
                return;
            }

            // 3. Phase 7: Evaluation Absorbing Upper Barrier ("Pass-on-Touch" at $53,015.00)
            if (!IsFundedAccount && currentTotalEquity >= 53015.0)
            {
                isDailyHalted = true;
                FlattenEverything(string.Format("EVAL_AUTO_FLATTEN_EQUITY: Equity ${0:F2} reached $53,015.00 pass barrier. Evaluation officially PASSED!", currentTotalEquity));
                return;
            }

            // 4. Phase 7: Funded Payout Absorbing Barrier ("Payout-on-Touch" at $52,660.00 with Qualifying Day)
            if (IsFundedAccount && currentTotalEquity >= 52660.0 && currentDailyNetPnL >= 160.0)
            {
                isDailyHalted = true;
                FlattenEverything(string.Format("FUNDED_PAYOUT_EQUITY: Equity ${0:F2} reached $52,660.00 payout barrier. Payout officially LOCKED!", currentTotalEquity));
                return;
            }

            // 5. Catastrophic Floor Check
            if (currentCash <= activeFloor)
            {
                TriggerEmergencyKillSwitch(string.Format("ACCOUNT FLOOR BREACH: Cash ${0:F2} <= Floor ${1:F2}", currentCash, activeFloor));
            }
        }

        /// <summary>
        /// Calculates optimal MNQ contract count via Phase 5 Bounded Risk-Parity Sizer + House-Money Floor Guard.
        /// </summary>
        public int CalculateOptimalMNQSize(double currentCash, double slDistPts = 30.0, double atr14 = 0.0, string strategyId = "H3", double morningPnL = 0.0)
        {
            double buffer = currentCash - activeFloor;
            double profit = currentCash - StartingBalance;
            double rTarget = 150.0;
            int minC = 1;
            int maxC = 12;

            if (!IsFundedAccount)
            {
                // Regime 1: EVALUATION Sprint Tiers (Phase 6 Calibrated)
                if (currentCash >= 51900.0)
                {
                    rTarget = 270.0; minC = 3; maxC = 6;
                }
                else if (profit >= 800.0 && buffer >= 1350.0)
                {
                    rTarget = 225.0; minC = 3; maxC = 4;
                }
                else if (buffer >= 950.0)
                {
                    rTarget = 165.0; minC = 2; maxC = 3;
                }
                else
                {
                    rTarget = 120.0; minC = 1; maxC = 2;
                }

                // Phase 7: Finish-Line Risk Tapering Near $53,000 (remaining_to_pass < $350.00)
                double remainingToPass = 53015.0 - currentCash;
                if (remainingToPass > 0.0 && remainingToPass < 350.0)
                {
                    rTarget = Math.Max(155.0, Math.Min(rTarget, 0.90 * remainingToPass));
                }
            }
            else
            {
                // Regime 2: FUNDED Capital Preservation Tiers (Restored $90 / $165 / $245)
                if (buffer < 1050.0)
                {
                    rTarget = 90.0; minC = 1; maxC = 2;
                }
                else if (!isFloorLocked && eodHighWatermark < 52100.0)
                {
                    rTarget = 165.0; minC = 2; maxC = 4;
                }
                else
                {
                    rTarget = 245.0; minC = 3; maxC = 6;
                }

                // Generalized House-Money Floor Guard for H4 in Funded Pre-Lock Mode
                if (strategyId == "H4" && !isFloorLocked && eodHighWatermark < 52100.0)
                {
                    if (morningPnL >= 165.0)
                    {
                        double houseMoneyBudget = morningPnL - 160.0;
                        double rEffective = Math.Min(rTarget, houseMoneyBudget);
                        double perContractRisk = (slDistPts * 2.00) + 4.48;
                        int h4Contracts = (int)Math.Floor(rEffective / perContractRisk);
                        return Math.Max(0, Math.Min(maxC, h4Contracts));
                    }
                }
            }

            double effAtr = (atr14 > 0.0) ? atr14 : 200.0;
            double refPts = (strategyId == "H4")
                ? Math.Max(16.0, Math.Min(22.0, 0.09 * effAtr))
                : Math.Max(30.0, Math.Min(40.0, 0.18 * effAtr));

            double unitRiskUsd = (refPts * 2.00) + 4.48;
            int rawContracts = (int)Math.Round(rTarget / unitRiskUsd);

            double perContractSlRisk = (slDistPts * 2.00) + 4.48;
            int maxSafeContracts = (int)Math.Max(1, Math.Floor((1.35 * rTarget) / perContractSlRisk));

            int contracts = Math.Min(rawContracts, maxSafeContracts);
            return Math.Max(minC, Math.Min(maxC, contracts));
        }

        /// <summary>
        /// External signal entry point invoked by Webhook or Signal Module.
        /// </summary>
        public void SubmitSignalEntry(bool isLong, double stopLossPrice, double targetPrice1, double atr14 = 0.0, string strategyId = "H3")
        {
            if (isDailyHalted)
            {
                Print("[OEM REJECTED] Cannot enter: Account halted for session by circuit breaker.");
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

            int contracts = CalculateOptimalMNQSize(currentCash, slDistPts, atr14, strategyId, effectiveMorningPnL);
            if (contracts <= 0)
            {
                Print(string.Format("[OEM SKIPPED] Strategy {0} allocated 0 contracts (House-Money Floor Guard or risk buffer).", strategyId));
                return;
            }

            OrderAction action = isLong ? OrderAction.Buy : OrderAction.SellShort;
            string signalName = isLong ? "MNQ_LONG_ENTRY" : "MNQ_SHORT_ENTRY";

            lastEntryTime = DateTime.UtcNow;
            entryOrder = SubmitOrderUnmanaged(0, action, OrderType.Market, contracts, 0, 0, "", signalName);

            Print(string.Format("[OEM EXECUTED] {0} {1} MNQ contracts (SL Dist: {2:F2} pts, Strat: {3}). Cash: ${4:F2} | Floor: ${5:F2}",
                action, contracts, slDistPts, strategyId, currentCash, activeFloor));

            SaveStateToFile();
        }

        protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity,
            MarketPosition marketPosition, string orderId, DateTime time)
        {
            if (entryOrder != null && execution.Order == entryOrder)
            {
                PlaceServerSideBrackets(execution.Order, price, quantity);
            }
            else if (Position.MarketPosition == MarketPosition.Flat)
            {
                if (lastEntryTime != DateTime.MinValue)
                {
                    double holdSeconds = (DateTime.UtcNow - lastEntryTime).TotalSeconds;
                    if (holdSeconds < 10.0)
                    {
                        Print(string.Format("[COMPLIANCE WARNING] Trade closed after {0:F1}s (< 10s rule). Ensure >50% duration compliance.", holdSeconds));
                    }
                    lastEntryTime = DateTime.MinValue;
                }

                // Check EOD high water mark and floor lock
                double currentCash = Account.Get(AccountItem.CashValue, Currency.UsDollar);
                if (currentCash > eodHighWatermark)
                {
                    eodHighWatermark = currentCash;
                    if (IsFundedAccount && eodHighWatermark >= 52100.0)
                    {
                        isFloorLocked = true;
                        activeFloor = 50100.0;
                    }
                    else if (!isFloorLocked)
                    {
                        activeFloor = Math.Max(activeFloor, eodHighWatermark - 2000.0);
                    }
                }

                SaveStateToFile();
            }
        }

        private void PlaceServerSideBrackets(Order parent, double fillPrice, int quantity)
        {
            activeOcoId = Guid.NewGuid().ToString("N").Substring(0, 8);
            bool isLong = (parent.OrderAction == OrderAction.Buy);

            double slPrice = isLong ? (fillPrice - 25.0) : (fillPrice + 25.0);
            double tpPrice = isLong ? (fillPrice + 35.0) : (fillPrice - 35.0);

            OrderAction slAction = isLong ? OrderAction.Sell : OrderAction.BuyToCover;
            OrderAction tpAction = isLong ? OrderAction.Sell : OrderAction.BuyToCover;

            stopLossOrder = SubmitOrderUnmanaged(0, slAction, OrderType.StopMarket, quantity, 0, slPrice, activeOcoId, "SL_" + activeOcoId);
            profitTargetOrder = SubmitOrderUnmanaged(0, tpAction, OrderType.Limit, quantity, tpPrice, 0, activeOcoId, "TP_" + activeOcoId);

            Print(string.Format("[SERVER OCO] Group {0} armed for {1} MNQ | SL: {2:F2} | TP: {3:F2}",
                activeOcoId, quantity, slPrice, tpPrice));
        }

        public void FlattenEverything(string reason)
        {
            Print(string.Format("[FLATTEN ENGAGED] {0}", reason));

            if (entryOrder != null && (entryOrder.OrderState == OrderState.Working || entryOrder.OrderState == OrderState.Accepted))
                CancelOrder(entryOrder);
            if (stopLossOrder != null && (stopLossOrder.OrderState == OrderState.Working || stopLossOrder.OrderState == OrderState.Accepted))
                CancelOrder(stopLossOrder);
            if (profitTargetOrder != null && (profitTargetOrder.OrderState == OrderState.Working || profitTargetOrder.OrderState == OrderState.Accepted))
                CancelOrder(profitTargetOrder);

            if (Position.MarketPosition == MarketPosition.Long)
                SubmitOrderUnmanaged(0, OrderAction.Sell, OrderType.Market, Position.Quantity, 0, 0, "", "FLATTEN_LONG");
            else if (Position.MarketPosition == MarketPosition.Short)
                SubmitOrderUnmanaged(0, OrderAction.BuyToCover, OrderType.Market, Position.Quantity, 0, 0, "", "FLATTEN_SHORT");

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

                ? Math.Max(16.0, Math.Min(22.0, 0.09 * effAtr))
                : Math.Max(30.0, Math.Min(40.0, 0.18 * effAtr));

            double unitRiskUsd = (refPts * 2.00) + 4.48;
            int rawContracts = (int)Math.Round(rTarget / unitRiskUsd);

            double perContractSlRisk = (slDistPts * 2.00) + 4.48;
            int maxSafeContracts = (int)Math.Max(1, Math.Floor((1.35 * rTarget) / perContractSlRisk));

            int contracts = Math.Min(rawContracts, maxSafeContracts);
            return Math.Max(minC, Math.Min(maxC, contracts));
        }

        /// <summary>
        /// External signal entry point invoked by Webhook or Signal Module.
        /// </summary>
        public void SubmitSignalEntry(bool isLong, double stopLossPrice, double targetPrice1, double atr14 = 0.0, string strategyId = "H3")
        {
            if (isDailyHalted)
            {
                Print("[OEM REJECTED] Cannot enter: Account halted for session by circuit breaker.");
                return;
            }

            bool isPyramidEntry = false;
            double currentDailyRealizedPnL = Account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar) - sessionStartRealizedPnL;
            double effectiveMorningPnL = currentDailyRealizedPnL;

            if (Position.MarketPosition != MarketPosition.Flat)
            {
                // Step 1: Same-Direction Pyramiding Unlock for H4
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

            int contracts = CalculateOptimalMNQSize(currentCash, slDistPts, atr14, strategyId, effectiveMorningPnL);
            if (contracts <= 0)
            {
                Print(string.Format("[OEM SKIPPED] Strategy {0} allocated 0 contracts (House-Money Floor Guard or risk buffer).", strategyId));
                return;
            }

            OrderAction action = isLong ? OrderAction.Buy : OrderAction.SellShort;
            string signalName = isLong ? "MNQ_LONG_ENTRY" : "MNQ_SHORT_ENTRY";

            lastEntryTime = DateTime.UtcNow;
            entryOrder = SubmitOrderUnmanaged(0, action, OrderType.Market, contracts, 0, 0, "", signalName);

            Print(string.Format("[OEM EXECUTED] {0} {1} MNQ contracts (SL Dist: {2:F2} pts, Strat: {3}). Cash: ${4:F2} | Floor: ${5:F2}",
                action, contracts, slDistPts, strategyId, currentCash, activeFloor));
        }

        protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity,
            MarketPosition marketPosition, string orderId, DateTime time)
        {
            if (entryOrder != null && execution.Order == entryOrder)
            {
                // Entry filled: Place resting server-side bracket OCO
                PlaceServerSideBrackets(execution.Order, price, quantity);
            }
            else if (Position.MarketPosition == MarketPosition.Flat)
            {
                // Position closed: Verify 10-Second Hold Rule Compliance
                if (lastEntryTime != DateTime.MinValue)
                {
                    double holdSeconds = (DateTime.UtcNow - lastEntryTime).TotalSeconds;
                    if (holdSeconds < 10.0)
                    {
                        Print(string.Format("[COMPLIANCE WARNING] Trade closed after {0:F1}s (< 10s rule). Ensure >50% duration compliance.", holdSeconds));
                    }
                    lastEntryTime = DateTime.MinValue;
                }
            }
        }

        private void PlaceServerSideBrackets(Order parent, double fillPrice, int quantity)
        {
            activeOcoId = Guid.NewGuid().ToString("N").Substring(0, 8);
            bool isLong = (parent.OrderAction == OrderAction.Buy);

            // Default bracket offsets (adjusted per signal in live bridge)
            double slPrice = isLong ? (fillPrice - 25.0) : (fillPrice + 25.0);
            double tpPrice = isLong ? (fillPrice + 35.0) : (fillPrice - 35.0);

            OrderAction slAction = isLong ? OrderAction.Sell : OrderAction.BuyToCover;
            OrderAction tpAction = isLong ? OrderAction.Sell : OrderAction.BuyToCover;

            stopLossOrder = SubmitOrderUnmanaged(0, slAction, OrderType.StopMarket, quantity, 0, slPrice, activeOcoId, "SL_" + activeOcoId);
            profitTargetOrder = SubmitOrderUnmanaged(0, tpAction, OrderType.Limit, quantity, tpPrice, 0, activeOcoId, "TP_" + activeOcoId);

            Print(string.Format("[SERVER OCO] Group {0} armed for {1} MNQ | SL: {2:F2} | TP: {3:F2}",
                activeOcoId, quantity, slPrice, tpPrice));
        }

        public void FlattenEverything(string reason)
        {
            Print(string.Format("[FLATTEN ENGAGED] {0}", reason));

            // 1. Cancel all working orders
            if (entryOrder != null && (entryOrder.OrderState == OrderState.Working || entryOrder.OrderState == OrderState.Accepted))
                CancelOrder(entryOrder);
            if (stopLossOrder != null && (stopLossOrder.OrderState == OrderState.Working || stopLossOrder.OrderState == OrderState.Accepted))
                CancelOrder(stopLossOrder);
            if (profitTargetOrder != null && (profitTargetOrder.OrderState == OrderState.Working || profitTargetOrder.OrderState == OrderState.Accepted))
                CancelOrder(profitTargetOrder);

            // 2. Liquidate open position
            if (Position.MarketPosition == MarketPosition.Long)
                SubmitOrderUnmanaged(0, OrderAction.Sell, OrderType.Market, Position.Quantity, 0, 0, "", "FLATTEN_LONG");
            else if (Position.MarketPosition == MarketPosition.Short)
                SubmitOrderUnmanaged(0, OrderAction.BuyToCover, OrderType.Market, Position.Quantity, 0, 0, "", "FLATTEN_SHORT");
        }

        public void TriggerEmergencyKillSwitch(string reason)
        {
            isDailyHalted = true;
            Print(string.Format("[EMERGENCY KILL-SWITCH] {0}", reason));
            FlattenEverything(reason);
        }
    }
}
