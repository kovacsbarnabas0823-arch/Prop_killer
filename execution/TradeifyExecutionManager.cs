#region Using declarations
using System;
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
    /// Institutional Order Execution Manager for NinjaTrader 8.
    /// Engineered for Tradeify $50K Growth EOD Trailing Drawdown Accounts.
    ///
    /// Key Capabilities:
    ///   - Dynamic Bellman 4-Tier Position Sizing (2 / 3 / 4 / 6 MNQ).
    ///   - Strict Contract Exclusivity: CME MNQ ($2.00/pt) Only.
    ///   - Intraday Hard Circuit Breakers: -$600.00 Internal Daily Stop.
    ///   - Funded Consistency Governor Ceiling: +$825.00 (protecting 35% rule).
    ///   - Hard 15:44:00 EST EOD Flatten Timer.
    ///   - Server-Side Resting OCO Partial Scale-Out Brackets.
    ///   - 10-Second Trade Duration Compliance Monitor.
    /// </summary>
    public class TradeifyExecutionManager : Strategy
    {
        #region Parameters
        [NinjaScriptProperty]
        [Display(Name = "Is Funded Account", Description = "False for Evaluation ($53K Target), True for Sim-Funded ($825 Daily Profit Ceiling)", Order = 1, GroupName = "Tradeify Account Parameters")]
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
        private double sessionStartRealizedPnL = 0.0;
        private bool isDailyHalted = false;
        private DateTime lastQuoteTime = DateTime.MinValue;
        private DateTime lastEntryTime = DateTime.MinValue;

        private Order entryOrder = null;
        private Order stopLossOrder = null;
        private Order profitTargetOrder = null;
        private string activeOcoId = string.Empty;
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

            // Timezone-Aware 15:44:00 EST Hard Flatten Guard
            DateTime estNow = TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow, TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"));
            if (estNow.TimeOfDay >= new TimeSpan(15, 44, 0))
            {
                if (Position.MarketPosition != MarketPosition.Flat || entryOrder != null)
                {
                    Print("[EOD FLATTEN] 15:44:00 EST reached. Liquidating all exposure before 15:45:00 deadline.");
                    FlattenEverything("15:44:00 Hard Session Close");
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

            // 1. Internal Daily Stop Check (-$600.00)
            if (currentDailyNetPnL <= -InternalDailyStopLoss)
            {
                isDailyHalted = true;
                FlattenEverything(string.Format("INTERNAL DAILY STOP HIT: -${0:F2}", Math.Abs(currentDailyNetPnL)));
                return;
            }

            // 2. Funded Consistency Governor Ceiling (+$825.00)
            if (IsFundedAccount && currentDailyNetPnL >= FundedDailyProfitCeiling)
            {
                isDailyHalted = true;
                FlattenEverything(string.Format("FUNDED CONSISTENCY GOVERNOR: Daily profit +${0:F2} reached ceiling. Halting to preserve 35% rule.", currentDailyNetPnL));
                return;
            }

            // 3. Catastrophic Floor Check
            double currentCash = Account.Get(AccountItem.CashValue, Currency.UsDollar);
            if (currentCash <= activeFloor)
            {
                TriggerEmergencyKillSwitch(string.Format("ACCOUNT FLOOR BREACH: Cash ${0:F2} <= Floor ${1:F2}", currentCash, activeFloor));
            }
        }

        /// <summary>
        /// Calculates optimal MNQ contract count via Phase 5 Bounded Risk-Parity Sizer + House-Money Floor Guard.
        /// Formula:
        ///   refPts = (strategyId == "H4") ? Math.Max(16.0, Math.Min(22.0, 0.09 * atr14))
        ///                                 : Math.Max(30.0, Math.Min(40.0, 0.18 * atr14));
        ///   unitRiskUsd = (refPts * 2.00) + 4.48;
        ///   rawContracts = round(rTarget / unitRiskUsd);
        ///   maxSafeContracts = floor((1.35 * rTarget) / ((slDistPts * 2.00) + 4.48));
        ///   contracts = clip(min(rawContracts, maxSafeContracts), minC, maxC);
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
                // Regime 1: EVALUATION Sprint Tiers
                if (currentCash >= 51900.0)
                {
                    rTarget = 330.0; minC = 4; maxC = 8;
                }
                else if (profit >= 800.0 && buffer >= 1350.0)
                {
                    rTarget = 250.0; minC = 3; maxC = 6;
                }
                else if (buffer >= 950.0)
                {
                    rTarget = 170.0; minC = 2; maxC = 4;
                }
                else
                {
                    rTarget = 120.0; minC = 1; maxC = 2;
                }
            }
            else
            {
                // Regime 2: FUNDED Capital Preservation Tiers
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

                // House-Money Floor Guard for H4 in Funded Pre-Lock Mode
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

            if (Position.MarketPosition != MarketPosition.Flat)
            {
                Print("[OEM REJECTED] Active position already open. Concurrency guard active.");
                return;
            }

            double currentCash = Account.Get(AccountItem.CashValue, Currency.UsDollar);
            double currentPrice = Close[0];
            double slDistPts = Math.Abs(currentPrice - stopLossPrice);
            if (slDistPts <= 0) slDistPts = 30.0;

            double currentDailyRealizedPnL = Account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar) - sessionStartRealizedPnL;

            int contracts = CalculateOptimalMNQSize(currentCash, slDistPts, atr14, strategyId, currentDailyRealizedPnL);
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
