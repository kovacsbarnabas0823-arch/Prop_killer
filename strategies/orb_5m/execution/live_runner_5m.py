"""
Production Live Runner CLI: 5-Minute ORB Strategy
Entry Point for executing the 5m Champion Strategy in SIM, Dry-Run, or Live Broker modes.

Usage:
  python strategies/orb_5m/execution/live_runner_5m.py --mode dry_run
  python strategies/orb_5m/execution/live_runner_5m.py --mode sim --symbol MNQ --contracts 8
"""

from __future__ import annotations
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
import yaml

# Support UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.orb_5m.execution.order_execution_manager_5m import (
    OrderExecutionManager5m,
    run_verification_scenario
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [LIVE-RUNNER-5M]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("LiveRunner5M")


def load_config() -> dict:
    cfg_path = PROJECT_ROOT / "strategies" / "orb_5m" / "config" / "strategy_config_5m.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="5-Minute ORB Production Execution Engine")
    parser.add_argument("--mode", choices=["dry_run", "sim", "live"], default="dry_run",
                        help="Execution mode (default: dry_run)")
    parser.add_argument("--symbol", default="MNQ", help="Target futures symbol (default: MNQ)")
    parser.add_argument("--contracts", type=int, default=8, help="Contract sizing (default: 8)")
    args = parser.parse_args()

    cfg = load_config()
    strat_cfg = cfg.get("strategy", {})

    print("=" * 85)
    print("🚀 AUTOMATED PROP TRADING: 5-MINUTE ORB LIVE EXECUTION SYSTEM")
    print(f"Strategy: {strat_cfg.get('name')}")
    print(f"Symbol  : {args.symbol} | Contracts: {args.contracts} | Mode: {args.mode.upper()}")
    print("Rules   : 09:30-09:35 Candle + 12 EMA | 09:35 Resting Stop | Dynamic 0.5R Trail | Max 1 Trade/Day")
    print("=" * 85)

    if args.mode == "dry_run":
        logger.info("Executing comprehensive end-to-end dry-run verification scenario...")
        asyncio.run(run_verification_scenario())
        logger.info("✅ Dry-run successfully verified.")
    elif args.mode in ("sim", "live"):
        logger.info(f"Starting OrderExecutionManager5m in {args.mode.upper()} mode...")
        oem = OrderExecutionManager5m(
            symbol=args.symbol,
            contracts=args.contracts,
            min_stop_points=float(strat_cfg.get("risk_management", {}).get("min_stop_points", 6.0)),
            max_stop_points=float(strat_cfg.get("risk_management", {}).get("max_stop_points", 35.0)),
            slippage_buffer_points=float(strat_cfg.get("risk_management", {}).get("slippage_points", 0.50))
        )
        try:
            asyncio.run(oem.start())
            print("\n[READY] OEM-5M is actively listening for market quotes and 09:35:00 EST candle triggers.")
            print("Press Ctrl+C to stop.")
            asyncio.get_event_loop().run_forever()
        except KeyboardInterrupt:
            print("\nShutting down OEM-5M...")
            asyncio.run(oem.stop())


if __name__ == "__main__":
    main()
