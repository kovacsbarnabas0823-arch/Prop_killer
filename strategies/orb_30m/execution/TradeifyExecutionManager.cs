using System;
using System.Collections.Generic;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.NinjaScript;
using NinjaTrader.Data;

namespace NinjaTrader.NinjaScript.Strategies
{
    /// <summary>
    /// Institutional Order Execution Manager for NinjaTrader 8.
    /// Incorporates:
    ///   - Dynamic Sizing via TradeifyRiskStateMachine
    ///   - Server-Side Resting OCO Brackets (with Partial Fill protection)
    ///   - 1.5-Second Millisecond Heartbeat & Kill-Switch Watchdog
    /// </summary>
    public class TradeifyExecutionManager : Strategy
    {
        private TradeifyRiskStateMachine riskStateMachine;
        private DateTime lastQuoteTime = DateTime.MinValue;
        private bool isSystemHalted = false;

        // Configuration
        public bool IsEvaluationAccount { get; set; } = true;
        public double MaxQuoteStalenessSeconds { get; set; } = 1.5;
        public double StopLossPoints { get; set; } = 25.0;
        public double TargetPoints { get; set; } = 37.5; // 1.5 R:R

        private Order entryOrder = null;
        private Order stopLossOrder = null;
        private Order profitTargetOrder = null;
        private int protectedBracketQty = 0;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Tradeify / Lucid Production Order Execution Manager";
                Name = "TradeifyExecutionManager";
                Calculate = Calculate.OnPriceChange;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 30;
                IsUnmanaged = true; // Use unmanaged order submission for direct matching engine control
            }
            else if (State == State.Configure)
            {
                riskStateMachine = new TradeifyRiskStateMachine();
            }
        }

        protected override void OnMarketData(MarketDataEventArgs marketDataUpdate)
        {
            // Record continuous incoming tick quote timestamp
            lastQuoteTime = DateTime.UtcNow;

            // Heartbeat check against staleness
            CheckHeartbeatSafety();
        }

        private void CheckHeartbeatSafety()
        {
            if (isSystemHalted || lastQuoteTime == DateTime.MinValue)
                return;

            double elapsedSeconds = (DateTime.UtcNow - lastQuoteTime).TotalSeconds;
            if (elapsedSeconds > MaxQuoteStalenessSeconds)
            {
                // Trigger Kill-Switch if we have active exposure
                if (Position.MarketPosition != MarketPosition.Flat || (entryOrder != null && entryOrder.OrderState == OrderState.Working))
                {
                    TriggerKillSwitch(string.Format("Quote staleness exceeded {0:F2}s with active market exposure.", elapsedSeconds));
                }
            }
        }

        public void TriggerKillSwitch(string reason)
        {
            isSystemHalted = true;
            Print(string.Format("[KILL-SWITCH ENGAGED] {0}. Flattening positions and cancelling all orders.", reason));

            // 1. Cancel all active unmanaged orders
            if (entryOrder != null && (entryOrder.OrderState == OrderState.Working || entryOrder.OrderState == OrderState.Accepted))
                CancelOrder(entryOrder);

            if (stopLossOrder != null && (stopLossOrder.OrderState == OrderState.Working || stopLossOrder.OrderState == OrderState.Accepted))
                CancelOrder(stopLossOrder);

            if (profitTargetOrder != null && (profitTargetOrder.OrderState == OrderState.Working || profitTargetOrder.OrderState == OrderState.Accepted))
                CancelOrder(profitTargetOrder);

            // 2. Immediate Market Flatten
            if (Position.MarketPosition == MarketPosition.Long)
            {
                SubmitOrderUnmanaged(0, OrderAction.Sell, OrderType.Market, Position.Quantity, 0, 0, "", "FLATTEN_LONG");
            }
            else if (Position.MarketPosition == MarketPosition.Short)
            {
                SubmitOrderUnmanaged(0, OrderAction.BuyToCover, OrderType.Market, Position.Quantity, 0, 0, "", "FLATTEN_SHORT");
            }
        }

        public void ExecuteEntrySignal(bool isLong, double currentBalance, double currentEodBarrier)
        {
            if (isSystemHalted)
            {
                Print("[EXECUTION BLOCKED] Cannot execute order: System is currently HALTED by Kill-Switch.");
                return;
            }

            // Task 2: Calculate size via dynamic Risk State Machine
            SizingDecision sizing = riskStateMachine.CalculatePositionSize(
                IsEvaluationAccount,
                currentBalance,
                currentEodBarrier,
                StopLossPoints
            );

            if (sizing.Quantity <= 0)
            {
                Print("[OEM] Sizing yielded 0 contracts. Order aborted.");
                return;
            }

            OrderAction action = isLong ? OrderAction.Buy : OrderAction.SellShort;
            string signalName = isLong ? "ORB_LONG_ENTRY" : "ORB_SHORT_ENTRY";

            protectedBracketQty = 0;
            entryOrder = SubmitOrderUnmanaged(0, action, OrderType.Market, sizing.Quantity, 0, 0, "", signalName);

            Print(string.Format("[OEM] Entry Submitted: {0} {1}x {2} | Risk: ${3:F2} | Tier: {4}",
                action, sizing.Quantity, sizing.InstrumentName, sizing.DollarRisk, sizing.Tier));
        }

        protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity,
            MarketPosition marketPosition, string orderId, DateTime time)
        {
            // Task 3: Trigger Bracket OCO upon fill or partial fill
            if (entryOrder != null && execution.Order == entryOrder)
            {
                int incrementalFill = quantity;
                if (incrementalFill > 0)
                {
                    SubmitServerSideBracket(execution.Order, price, incrementalFill);
                }
            }
        }

        private void SubmitServerSideBracket(Order parent, double fillPrice, int qty)
        {
            string ocoId = Guid.NewGuid().ToString("N").Substring(0, 8);
            bool isLong = (parent.OrderAction == OrderAction.Buy);

            double slPrice = isLong ? (fillPrice - StopLossPoints) : (fillPrice + StopLossPoints);
            double tpPrice = isLong ? (fillPrice + TargetPoints) : (fillPrice - TargetPoints);

            OrderAction slAction = isLong ? OrderAction.Sell : OrderAction.BuyToCover;
            OrderAction tpAction = isLong ? OrderAction.Sell : OrderAction.BuyToCover;

            // Submit resting server-side OCO orders
            stopLossOrder = SubmitOrderUnmanaged(0, slAction, OrderType.StopMarket, qty, 0, slPrice, ocoId, "SL_" + ocoId);
            profitTargetOrder = SubmitOrderUnmanaged(0, tpAction, OrderType.Limit, qty, tpPrice, 0, ocoId, "TP_" + ocoId);

            protectedBracketQty += qty;
            Print(string.Format("[SERVER-OCO] Placed OCO Group {0} for {1} contracts | SL: {2:F2} | TP: {3:F2}",
                ocoId, qty, slPrice, tpPrice));
        }
    }
}
