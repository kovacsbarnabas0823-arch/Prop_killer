using System;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.NinjaScript;
using NinjaTrader.Data;

namespace NinjaTrader.NinjaScript.Strategies
{
    public enum AccountLifecycleTier
    {
        Tier1_Evaluation,      // High Velocity Mode (8.5% of remaining DD)
        Tier2_BufferBuilding,  // Funded Buffer Mode (5.0% of remaining DD)
        Tier3_Annuity          // Capital Preservation Mode (2.5% of remaining DD)
    }

    public class SizingDecision
    {
        public string InstrumentName { get; set; } // "MNQ" or "NQ"
        public int Quantity { get; set; }          // Whole contracts
        public double DollarRisk { get; set; }     // Allocated cash risk
        public AccountLifecycleTier Tier { get; set; }
        public double RemainingDrawdown { get; set; }
    }

    public class TradeifyRiskStateMachine
    {
        public double StartingBalance { get; set; } = 50000.0;
        public double MnqPointValue { get; set; } = 2.0;
        public double NqPointValue { get; set; } = 20.0;
        public int MnqToNqThreshold { get; set; } = 10;
        public double FundedBufferThreshold { get; set; } = 5000.0;

        public AccountLifecycleTier EvaluateTier(bool isEvaluation, double currentBalance)
        {
            if (isEvaluation)
                return AccountLifecycleTier.Tier1_Evaluation;

            double buffer = currentBalance - StartingBalance;
            if (buffer < FundedBufferThreshold)
                return AccountLifecycleTier.Tier2_BufferBuilding;
            else
                return AccountLifecycleTier.Tier3_Annuity;
        }

        public SizingDecision CalculatePositionSize(
            bool isEvaluation,
            double currentBalance,
            double currentEodDrawdownThreshold,
            double stopLossPoints)
        {
            if (stopLossPoints <= 0)
                throw new ArgumentException("Stop loss points must be strictly positive.");

            double remainingDd = currentBalance - currentEodDrawdownThreshold;
            if (remainingDd <= 0)
                throw new InvalidOperationException("Account breached! Remaining drawdown <= 0.");

            AccountLifecycleTier tier = EvaluateTier(isEvaluation, currentBalance);

            double riskPct = 0.085; // Default Tier 1: 8.5%
            if (tier == AccountLifecycleTier.Tier2_BufferBuilding)
                riskPct = 0.050;    // Tier 2: 5.0%
            else if (tier == AccountLifecycleTier.Tier3_Annuity)
                riskPct = 0.025;    // Tier 3: 2.5%

            double dollarRiskAllocated = remainingDd * riskPct;
            double riskPerMnq = stopLossPoints * MnqPointValue;

            // Round down to whole MNQ contracts
            int mnqContracts = (int)Math.Floor(dollarRiskAllocated / riskPerMnq);

            string instrument = "MNQ";
            int finalQuantity = mnqContracts;
            double actualRisk = mnqContracts * riskPerMnq;

            // Transition to standard NQ if MNQ contracts exceed threshold (> 10)
            if (mnqContracts > MnqToNqThreshold)
            {
                int nqContracts = (int)Math.Floor(mnqContracts / 10.0);
                if (nqContracts >= 1)
                {
                    instrument = "NQ";
                    finalQuantity = nqContracts;
                    actualRisk = nqContracts * (stopLossPoints * NqPointValue);
                }
            }

            return new SizingDecision
            {
                InstrumentName = instrument,
                Quantity = finalQuantity,
                DollarRisk = actualRisk,
                Tier = tier,
                RemainingDrawdown = remainingDd
            };
        }
    }
}
