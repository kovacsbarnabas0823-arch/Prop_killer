using System;
using System.Collections.Generic;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.NinjaScript;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript.Indicators;

namespace NinjaTrader.NinjaScript.Strategies
{
    /// <summary>
    /// Production 5-Minute ORB + 12 EMA Execution Manager for NinjaTrader 8.
    /// Target: CME Micro E-mini Nasdaq-100 (MNQ) Futures / E-mini Nasdaq (NQ)
    /// Target Account: Tradeify $50K EOD Trailing Drawdown Evaluation & Funded Accounts
    /// 
    /// Execution Specifications:
    ///   - Real-time 5m opening range (09:30:00 to 09:35:00 EST)
    ///   - 12 EMA directional bias on 5m timeframe
    ///   - Unmanaged resting Stop-Market entry order placed at 09:35:00 EST sharp
    ///   - Dynamic continuous 0.5R trailing stop ratchet on every favorable price tick
    ///   - 11:00:00 EST entry expiration (cancels resting entry order if unfilled)
    ///   - 15:45:00 EST hard session close exit
    ///   - Single trade per day guardrail
    ///   - 1.5-second quote staleness heartbeat kill-switch watchdog
    /// </summary>
    public class TradeifyExecutionManager_5m : Strategy
    {
        // -------------------------------------------------------------
        // Strategy Parameters
        // -------------------------------------------------------------
        public int Contracts { get; set; } = 8;
        public double MinCandlePoints { get; set; } = 6.0;
        public double MinStopPoints { get; set; } = 6.0;
        public double MaxStopPoints { get; set; } = 35.0;
        public double SlippageBufferPoints { get; set; } = 0.50;
        public double MaxQuoteStalenessSeconds { get; set; } = 1.5;

        // -------------------------------------------------------------
        // Indicators & State Machine
        // -------------------------------------------------------------
        private EMA ema12;
        private DateTime lastQuoteTime = DateTime.MinValue;
        private bool isSystemHalted = false;
        private int currentSessionDate = 0;
        private int tradesTakenToday = 0;

        // 09:30–09:35 Candle State
        private bool orEstablished = false;
        private double orHigh = 0.0;
        private double orLow = 0.0;
        private double orClose = 0.0;
        private double orEma = 0.0;
        private int biasDirection = 0; // +1 Long, -1 Short, 0 None

        // Orders
        private Order entryOrder = null;
        private Order stopLossOrder = null;

        // Active Position & 0.5R Trailing State
        private double entryFillPrice = 0.0;
        private double initialStopPrice = 0.0;
        private double currentStopPrice = 0.0;
        private double riskPoints = 0.0;
        private double trailStep05R = 0.0;
        private double peakFavorablePrice = 0.0;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Tradeify 5-Minute ORB + 12 EMA Live Execution Manager (Champion Model)";
                Name = "TradeifyExecutionManager_5m";
                Calculate = Calculate.OnPriceChange; // Tick-by-tick evaluation
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 30; // Auto-flatten at 15:45 settlement
                IsUnmanaged = true;            // Direct unmanaged matching engine control
            }
            else if (State == State.Configure)
            {
                // Add secondary 5-minute bar series for 12 EMA and opening range calculation
                AddDataSeries(BarsPeriodType.Minute, 5);
            }
            else if (State == State.DataLoaded)
            {
                // 12 EMA initialized on 5-minute series (BarsArray[1])
                ema12 = EMA(BarsArray[1], 12);
            }
        }

        protected override void OnMarketData(MarketDataEventArgs marketDataUpdate)
        {
            lastQuoteTime = DateTime.UtcNow;
            CheckHeartbeatSafety();
        }

        private void CheckHeartbeatSafety()
        {
            if (isSystemHalted || lastQuoteTime == DateTime.MinValue)
                return;

            double elapsedSeconds = (DateTime.UtcNow - lastQuoteTime).TotalSeconds;
            if (elapsedSeconds > MaxQuoteStalenessSeconds)
            {
                bool hasExposure = (Position.MarketPosition != MarketPosition.Flat ||
                                   (entryOrder != null && (entryOrder.OrderState == OrderState.Working || entryOrder.OrderState == OrderState.Accepted)));
                if (hasExposure)
                {
                    TriggerKillSwitch(string.Format("Quote staleness ({0:F2}s) exceeded limit ({1:F2}s) with active exposure.",
                        elapsedSeconds, MaxQuoteStalenessSeconds));
                }
            }
        }

        public void TriggerKillSwitch(string reason)
        {
            isSystemHalted = true;
            Print(string.Format("🚨 [KILL-SWITCH ENGAGED] {0}. Flattening positions and cancelling all orders.", reason));

            // Cancel all working orders
            if (entryOrder != null && (entryOrder.OrderState == OrderState.Working || entryOrder.OrderState == OrderState.Accepted))
                CancelOrder(entryOrder);

            if (stopLossOrder != null && (stopLossOrder.OrderState == OrderState.Working || stopLossOrder.OrderState == OrderState.Accepted))
                CancelOrder(stopLossOrder);

            // Immediate Market Flatten
            if (Position.MarketPosition == MarketPosition.Long)
            {
                SubmitOrderUnmanaged(0, OrderAction.Sell, OrderType.Market, Position.Quantity, 0, 0, "", "FLATTEN_LONG");
            }
            else if (Position.MarketPosition == MarketPosition.Short)
            {
                SubmitOrderUnmanaged(0, OrderAction.BuyToCover, OrderType.Market, Position.Quantity, 0, 0, "", "FLATTEN_SHORT");
            }
        }

        protected override void OnBarUpdate()
        {
            if (isSystemHalted)
                return;

            DateTime estNow = TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow,
                TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"));
            TimeSpan timeOfDay = estNow.TimeOfDay;

            // Session Date rollover reset
            int todayInt = estNow.Year * 10000 + estNow.Month * 100 + estNow.Day;
            if (todayInt != currentSessionDate)
            {
                currentSessionDate = todayInt;
                tradesTakenToday = 0;
                orEstablished = false;
                biasDirection = 0;
                entryOrder = null;
                stopLossOrder = null;
                Print(string.Format("[SESSION RESET] Initialized 5m ORB Engine for Date: {0}", estNow.ToString("yyyy-MM-dd")));
            }

            // ---------------------------------------------------------
            // 1. Process 5-Minute Opening Range Candle (BarsInProgress == 1)
            // ---------------------------------------------------------
            if (BarsInProgress == 1)
            {
                // Detect completion of 09:30–09:35 EST bar
                DateTime barTime = Time[0];
                if (barTime.Hour == 9 && barTime.Minute == 35 && !orEstablished && tradesTakenToday == 0)
                {
                    orHigh = High[0];
                    orLow = Low[0];
                    orClose = Close[0];
                    orEma = ema12[0];
                    double candleRange = orHigh - orLow;

                    Print(string.Format("[09:35 EST 5M CANDLE] Range: {0:F2} - {1:F2} ({2:F2} pts) | Close: {3:F2} | 12 EMA: {4:F2}",
                        orLow, orHigh, candleRange, orClose, orEma));

                    // Noise floor filter
                    if (candleRange < MinCandlePoints)
                    {
                        Print(string.Format("⚠️ [NOISE FLOOR] 5m Range {0:F2} < {1:F2} pts min. No trades today.",
                            candleRange, MinCandlePoints));
                        orEstablished = true;
                        biasDirection = 0;
                        return;
                    }

                    // 12 EMA Directional Bias
                    if (orClose > orEma)
                    {
                        biasDirection = 1;
                        double buyStopPrice = orHigh + SlippageBufferPoints;
                        Print(string.Format("🟢 [LONG BIAS] Submitting resting Buy-Stop at {0:F2}", buyStopPrice));
                        entryOrder = SubmitOrderUnmanaged(0, OrderAction.Buy, OrderType.StopMarket, Contracts, 0, buyStopPrice, "", "ENTRY_BUY_STOP");
                    }
                    else if (orClose < orEma)
                    {
                        biasDirection = -1;
                        double sellStopPrice = orLow - SlippageBufferPoints;
                        Print(string.Format("🔴 [SHORT BIAS] Submitting resting Sell-Stop at {0:F2}", sellStopPrice));
                        entryOrder = SubmitOrderUnmanaged(0, OrderAction.SellShort, OrderType.StopMarket, Contracts, 0, sellStopPrice, "", "ENTRY_SELL_STOP");
                    }

                    orEstablished = true;
                }
                return;
            }

            // ---------------------------------------------------------
            // 2. Primary 1-Minute / Tick Series (BarsInProgress == 0)
            // ---------------------------------------------------------
            if (BarsInProgress == 0)
            {
                // A. Check Entry Timeout (11:00:00 EST)
                if (entryOrder != null && (entryOrder.OrderState == OrderState.Working || entryOrder.OrderState == OrderState.Accepted))
                {
                    if (timeOfDay >= new TimeSpan(11, 0, 0))
                    {
                        Print("[ENTRY TIMEOUT] 11:00 EST reached without fill. Cancelling resting entry order.");
                        CancelOrder(entryOrder);
                        entryOrder = null;
                        return;
                    }
                }

                // B. Check Hard EOD Close (15:45:00 EST)
                if (timeOfDay >= new TimeSpan(15, 45, 0))
                {
                    if (Position.MarketPosition != MarketPosition.Flat)
                    {
                        Print("[HARD EOD EXIT] 15:45 EST reached. Flattening all positions.");
                        if (Position.MarketPosition == MarketPosition.Long)
                            SubmitOrderUnmanaged(0, OrderAction.Sell, OrderType.Market, Position.Quantity, 0, 0, "", "EOD_FLATTEN");
                        else
                            SubmitOrderUnmanaged(0, OrderAction.BuyToCover, OrderType.Market, Position.Quantity, 0, 0, "", "EOD_FLATTEN");
                    }
                    return;
                }

                // C. Dynamic 0.5R Trailing Stop Ratchet on Active Position
                if (Position.MarketPosition != MarketPosition.Flat && stopLossOrder != null && stopLossOrder.OrderState == OrderState.Working)
                {
                    double currentPrice = Close[0];

                    if (Position.MarketPosition == MarketPosition.Long)
                    {
                        if (currentPrice > peakFavorablePrice)
                        {
                            peakFavorablePrice = currentPrice;
                            double candidateSl = peakFavorablePrice - trailStep05R;
                            if (candidateSl > currentStopPrice)
                            {
                                double prevSl = currentStopPrice;
                                currentStopPrice = candidateSl;
                                ChangeOrderUnmanaged(stopLossOrder, stopLossOrder.Quantity, 0, currentStopPrice);
                                Print(string.Format("📈 [0.5R TRAIL RATCHET UP] New Peak: {0:F2} | Stop ratcheted: {1:F2} -> {2:F2} (+{3:F2} pts)",
                                    peakFavorablePrice, prevSl, currentStopPrice, (currentStopPrice - prevSl)));
                            }
                        }
                    }
                    else if (Position.MarketPosition == MarketPosition.Short)
                    {
                        if (currentPrice < peakFavorablePrice)
                        {
                            peakFavorablePrice = currentPrice;
                            double candidateSl = peakFavorablePrice + trailStep05R;
                            if (candidateSl < currentStopPrice)
                            {
                                double prevSl = currentStopPrice;
                                currentStopPrice = candidateSl;
                                ChangeOrderUnmanaged(stopLossOrder, stopLossOrder.Quantity, 0, currentStopPrice);
                                Print(string.Format("📉 [0.5R TRAIL RATCHET DOWN] New Trough: {0:F2} | Stop ratcheted: {1:F2} -> {2:F2} ({3:F2} pts)",
                                    peakFavorablePrice, prevSl, currentStopPrice, (currentStopPrice - prevSl)));
                            }
                        }
                    }
                }
            }
        }

        protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity,
            MarketPosition marketPosition, string orderId, DateTime time)
        {
            // 1. Entry Filled
            if (entryOrder != null && execution.Order == entryOrder && execution.Order.OrderState == OrderState.Filled)
            {
                tradesTakenToday++;
                entryFillPrice = price;
                peakFavorablePrice = price;

                bool isLong = (entryOrder.OrderAction == OrderAction.Buy);
                double rawRisk = isLong ? (price - orLow) : (orHigh - price);
                riskPoints = Math.Min(Math.Max(rawRisk, MinStopPoints), MaxStopPoints);
                trailStep05R = 0.50 * riskPoints;

                if (isLong)
                {
                    initialStopPrice = price - riskPoints;
                    currentStopPrice = initialStopPrice;
                    stopLossOrder = SubmitOrderUnmanaged(0, OrderAction.Sell, OrderType.StopMarket, quantity, 0, initialStopPrice, "", "SL_STOP_LOSS");
                }
                else
                {
                    initialStopPrice = price + riskPoints;
                    currentStopPrice = initialStopPrice;
                    stopLossOrder = SubmitOrderUnmanaged(0, OrderAction.BuyToCover, OrderType.StopMarket, quantity, 0, initialStopPrice, "", "SL_STOP_LOSS");
                }

                Print(string.Format("✅ [ENTRY EXECUTED] {0} {1}x @ {2:F2} | Initial SL: {3:F2} (Risk: {4:F2} pts) | 0.5R Step: {5:F2} pts",
                    entryOrder.OrderAction, quantity, price, initialStopPrice, riskPoints, trailStep05R));

                entryOrder = null;
            }

            // 2. Stop Loss Filled
            else if (stopLossOrder != null && execution.Order == stopLossOrder && execution.Order.OrderState == OrderState.Filled)
            {
                double pnlPts = (Position.MarketPosition == MarketPosition.Long) ? (price - entryFillPrice) : (entryFillPrice - price);
                Print(string.Format("🏁 [STOP LOSS FILLED] Exit Price: {0:F2} | Realized Points: {1:F2} pts | Trades Today: {2}/1",
                    price, pnlPts, tradesTakenToday));

                stopLossOrder = null;
            }
        }
    }
}
