"""
CLI Tool for Clean Tradeify Evaluation-to-Funded Phase Transition
Author: Senior Quantitative Systems Engineer

Safely archives passed Evaluation account JSON state and initializes a clean FUNDED state:
  - Account Phase: FUNDED
  - Current Equity: $50,000.00
  - High-Water Mark: $50,000.00
  - Liquidation Floor: $48,000.00 (Trailing Drawdown active)
  - Floor Locked: False
  - Qualifying Days: 0
  - Max Historical Daily PnL: $0.00
  - Payout Cycle: 1

Usage:
  python scripts/reset_account_phase.py --account-id Tradeify-50K-Growth-001
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.state_persistence import TradeifyStatePersistence


def main():
    parser = argparse.ArgumentParser(description="Clean Tradeify Evaluation-to-Funded State Initializer")
    parser.add_argument("--account-id", type=str, default="Tradeify-50K-Growth-001", help="Target Tradeify Account ID")
    parser.add_argument("--force", action="store_true", help="Force reset without confirmation prompt")
    args = parser.parse_args()

    account_id = args.account_id.strip()
    print("=" * 70)
    print("TRADEIFY ACCOUNT PHASE RESET UTILITY")
    print(f"Target Account ID : {account_id}")
    print("Target State      : FUNDED ($50,000.00 base, $48,000.00 floor)")
    print("=" * 70)

    sp = TradeifyStatePersistence(account_id=account_id)
    curr = sp.get_state()
    print(f"Current Phase     : {curr.get('account_phase')}")
    print(f"Current Equity    : ${curr.get('current_equity', 0.0):.2f}")
    print(f"Current HWM       : ${curr.get('eod_hwm', 0.0):.2f}")
    print(f"Current Floor     : ${curr.get('liquidation_floor', 0.0):.2f}")
    print(f"Floor Locked      : {curr.get('is_floor_locked')}")

    if not args.force:
        confirm = input("\nProceed with archiving current state and initializing clean FUNDED phase? (yes/no): ")
        if confirm.lower() not in ["yes", "y"]:
            print("Operation aborted by user.")
            sys.exit(0)

    new_state = sp.reset_to_funded()
    print("\n[SUCCESS] Account successfully initialized to clean FUNDED phase:")
    print(f"  Account ID        : {new_state['account_id']}")
    print(f"  Phase             : {new_state['account_phase']}")
    print(f"  Equity            : ${new_state['current_equity']:.2f}")
    print(f"  EOD HWM           : ${new_state['eod_hwm']:.2f}")
    print(f"  Liquidation Floor : ${new_state['liquidation_floor']:.2f}")
    print(f"  Floor Locked      : {new_state['is_floor_locked']}")
    print(f"  Qualifying Days   : {new_state['qualifying_days_count']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
