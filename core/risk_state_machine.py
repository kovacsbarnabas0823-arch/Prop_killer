from dataclasses import dataclass
from enum import Enum
import math
from typing import Dict, Any, Tuple

class AccountType(Enum):
    EVALUATION = "EVALUATION"
    FUNDED = "FUNDED"

class RiskTier(Enum):
    TIER_1_EVALUATION = "TIER_1_EVALUATION"       # High Velocity Mode (8.5% of remaining DD)
    TIER_2_BUFFER_BUILDING = "TIER_2_BUFFER_BUILDING" # Funded Buffer Mode (5.0% of remaining DD)
    TIER_3_ANNUITY = "TIER_3_ANNUITY"             # Capital Preservation Mode (2.5% of remaining DD)

@dataclass
class SizingResult:
    instrument: str          # 'MNQ' or 'NQ'
    contracts: int           # Whole number of contracts
    dollar_risk: float       # Exact cash risk allocated
    stop_distance_points: float # Stop distance in NQ/MNQ index points
    effective_risk_pct: float   # Risk percentage of remaining drawdown
    tier: RiskTier
    buffer_above_start: float
    remaining_drawdown: float

class RiskStateMachine:
    """
    Dynamic Prop-Firm Risk & Position Sizing State Machine for Tradeify EOD Trailing Drawdown.
    
    Parameters:
        starting_balance: Initial account balance (default: ,000)
        mnq_point_value: Cash value per point for MNQ (default: .00)
        nq_point_value: Cash value per point for NQ (default: .00)
        mnq_to_nq_threshold: Number of MNQ contracts before converting to standard NQ (default: 10)
    """
    def __init__(
        self,
        starting_balance: float = 50000.0,
        mnq_point_value: float = 2.0,
        nq_point_value: float = 20.0,
        mnq_to_nq_threshold: int = 10,
        funded_buffer_threshold: float = 5000.0
    ):
        self.starting_balance = starting_balance
        self.mnq_point_value = mnq_point_value
        self.nq_point_value = nq_point_value
        self.mnq_to_nq_threshold = mnq_to_nq_threshold
        self.funded_buffer_threshold = funded_buffer_threshold

    def evaluate_tier(self, account_type: AccountType, current_balance: float) -> RiskTier:
        """Determines the active risk tier based on account type and profit buffer."""
        if account_type == AccountType.EVALUATION:
            return RiskTier.TIER_1_EVALUATION
        
        profit_buffer = current_balance - self.starting_balance
        if profit_buffer < self.funded_buffer_threshold:
            return RiskTier.TIER_2_BUFFER_BUILDING
        else:
            return RiskTier.TIER_3_ANNUITY

    def calculate_sizing(
        self,
        account_type: AccountType,
        current_balance: float,
        current_eod_drawdown_threshold: float,
        stop_loss_points: float,
        consecutive_losses: int = 0
    ) -> SizingResult:
        """
        Calculates exact contract size and instrument (MNQ vs NQ).
        
        Formula:
            Remaining DD = Current Balance - Current EOD DD Threshold
            Dollar Risk = Remaining DD * Tier Risk Percentage (* 0.70 if consecutive_losses >= 2)
            Raw MNQ Contracts = Dollar Risk / (Stop Loss Points * $2.00)
            Sizing rounds DOWN to the nearest integer.
            Enforces minimum 2 MNQ contracts and even number of contracts.
            If Raw MNQ >= 10, converts to standard NQ contracts (1 NQ = 10 MNQ).
        """
        if stop_loss_points <= 0:
            raise ValueError("Stop loss points must be strictly positive.")
            
        remaining_drawdown = current_balance - current_eod_drawdown_threshold
        if remaining_drawdown <= 0:
            raise ValueError(f"Account breach! Balance (${current_balance:.2f}) is at or below threshold (${current_eod_drawdown_threshold:.2f}).")

        tier = self.evaluate_tier(account_type, current_balance)
        buffer_above_start = current_balance - self.starting_balance

        # Base risk percentages per tier
        if tier == RiskTier.TIER_1_EVALUATION:
            risk_pct = 0.085  # 8.5% High Velocity
        elif tier == RiskTier.TIER_2_BUFFER_BUILDING:
            risk_pct = 0.050  # 5.0% Buffer Building
        elif tier == RiskTier.TIER_3_ANNUITY:
            risk_pct = 0.025  # 2.5% Annuity Preservation
        else:
            raise ValueError(f"Unknown tier: {tier}")

        # Streak-Adaptive Drawdown Shield (Anti-clustering risk modifier)
        # If strategy registers >= 2 consecutive losing sessions, scale down trade risk by 30%
        if consecutive_losses >= 2:
            risk_pct *= 0.70  # e.g., from 8.5% down to 5.95% (~6.0%)

        dollar_risk_allocated = remaining_drawdown * risk_pct
        risk_per_mnq = stop_loss_points * self.mnq_point_value

        # Calculate raw MNQ contracts (rounded down)
        mnq_contracts = math.floor(dollar_risk_allocated / risk_per_mnq)

        # Enforce minimum 2 micro contracts and even contract count for clean 50/50 two-tier splitting
        if mnq_contracts < 2:
            mnq_contracts = 2
        elif mnq_contracts % 2 != 0:
            mnq_contracts -= 1

        # Task 4: Round down to whole MNQ, convert to NQ if contracts > 10
        if mnq_contracts > self.mnq_to_nq_threshold:
            # Convert to standard NQ contracts (1 NQ = 10 MNQ = $20/pt)
            nq_contracts = math.floor(mnq_contracts / 10)
            if nq_contracts < 2:
                # Keep as MNQ if converting to NQ drops contract count below 2
                selected_instrument = "MNQ"
                final_contracts = mnq_contracts
                actual_dollar_risk = mnq_contracts * risk_per_mnq
            else:
                if nq_contracts % 2 != 0:
                    nq_contracts -= 1
                selected_instrument = "NQ"
                final_contracts = nq_contracts
                actual_dollar_risk = nq_contracts * (stop_loss_points * self.nq_point_value)
        else:
            selected_instrument = "MNQ"
            final_contracts = mnq_contracts
            actual_dollar_risk = mnq_contracts * risk_per_mnq

        return SizingResult(
            instrument=selected_instrument,
            contracts=final_contracts,
            dollar_risk=round(actual_dollar_risk, 2),
            stop_distance_points=stop_loss_points,
            effective_risk_pct=round((actual_dollar_risk / remaining_drawdown) * 100, 2),
            tier=tier,
            buffer_above_start=round(buffer_above_start, 2),
            remaining_drawdown=round(remaining_drawdown, 2)
        )

# ==============================================================================
# Comprehensive Unit Verification
# ==============================================================================
if __name__ == "__main__":
    rsm = RiskStateMachine(starting_balance=50000.0)
    stop_pts = 25.0  # 25-point stop on Nasdaq (/MNQ, /NQ)

    print("=" * 80)
    print("DYNAMIC RISK STATE MACHINE VERIFICATION (TRADEIFY SPECIFICATION)")
    print("=" * 80)

    test_cases = [
        # 1. Tier 1: Fresh Evaluation ( balance,  EOD DD barrier)
        (AccountType.EVALUATION, 50000.0, 48000.0, "Fresh Eval Account ( DD remaining)"),
        
        # 2. Tier 1: Evaluation In-Profit (,500 balance, ,500 EOD DD barrier)
        (AccountType.EVALUATION, 51500.0, 49500.0, "Eval Account In-Profit ( DD remaining)"),

        # 3. Tier 2: Fresh Funded (,000 balance, ,000 EOD DD barrier, buffer < )
        (AccountType.FUNDED, 50000.0, 48000.0, "Funded Account Day 1 ( profit buffer)"),

        # 4. Tier 2: Funded Account with  profit (,000 balance, ,000 locked barrier)
        (AccountType.FUNDED, 53000.0, 50000.0, "Funded Account with  Buffer ( DD remaining)"),

        # 5. Tier 3: Funded Account in Annuity Mode (,000 balance, ,000 locked barrier, buffer > )
        (AccountType.FUNDED, 56000.0, 50000.0, "Funded Account Annuity Mode ( buffer,  DD remaining)"),

        # 6. Tier 3: High Capital Buffer transitioning to standard NQ (e.g.  balance,  DD remaining)
        (AccountType.FUNDED, 70000.0, 50000.0, "Funded Account Large Buffer (>10 MNQ -> NQ transition)"),
    ]

    for acct_type, bal, dd_thresh, desc in test_cases:
        res = rsm.calculate_sizing(acct_type, bal, dd_thresh, stop_pts)
        print(f"Scenario: {desc}")
        print(f"  Balance:  | EOD Barrier:  | Remaining DD: ")
        print(f"  Tier: {res.tier.value}")
        print(f"  Stop: {stop_pts} pts | Target Instrument: {res.instrument} | Contracts: {res.contracts}")
        print(f"  Actual Dollar Risk:  ({res.effective_risk_pct}% of remaining DD)")
        print("-" * 80)
