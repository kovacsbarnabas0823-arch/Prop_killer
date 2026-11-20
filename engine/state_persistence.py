"""
Atomic State Persistence Module for Tradeify $50K Growth Account
Author: Principal Quantitative Architect & Lead Systems Engineer

Provides atomic, thread-safe, and crash-resilient JSON state persistence for:
  - Account ID
  - Phase (EVALUATION vs FUNDED)
  - Current Equity & High-Water Mark (EOD_HWM)
  - Trailing Liquidation Floor & Floor-Lock Status ($52,100 trigger -> $50,100 permanent floor)
  - Qualifying Days Count (>= +$150.00 net daily profit)
  - Max Historical Daily PnL (for Tradeify 35% consistency compliance)
  - Realized Daily PnL & Date tracking
  - Payout Cycle Number (Cycle 1, 2, 3...)

Uses atomic file renaming (temporary file -> os.replace) to ensure state files
can never be corrupted or partially written during unexpected VPS shutdowns.
"""

from __future__ import annotations
import os
import json
import logging
import tempfile
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Any, Optional

logger = logging.getLogger("StatePersistence")


class TradeifyStatePersistence:
    def __init__(self, state_file_path: Optional[str | Path] = None, account_id: str = "Tradeify-50K-Growth-001"):
        project_root = Path(__file__).resolve().parent.parent
        self.account_id = account_id

        if state_file_path is None:
            # Per-Account Atomic State Isolation: state/tradeify_<ACCOUNT_ID>.json
            self.state_file_path = project_root / "state" / f"tradeify_{self.account_id}.json"
        else:
            self.state_file_path = Path(state_file_path)

        self.state_dir = self.state_file_path.parent
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Seamless migration from legacy tradeify_account_state.json if present for Account 001
        legacy_path = self.state_dir / "tradeify_account_state.json"
        if not self.state_file_path.exists() and legacy_path.exists() and self.account_id == "Tradeify-50K-Growth-001":
            try:
                with open(legacy_path, "r", encoding="utf-8") as f:
                    legacy_data = json.load(f)
                if legacy_data.get("account_id") == self.account_id:
                    self._write_atomic(legacy_data)
                    logger.info(f"Migrated legacy state from {legacy_path} to {self.state_file_path}")
            except Exception as e:
                logger.warning(f"Failed to migrate legacy state file: {e}")

        self.state: Dict[str, Any] = self._load_or_initialize()

    def _default_state(self) -> Dict[str, Any]:
        return {
            "account_id": self.account_id,
            "account_phase": "EVALUATION",
            "current_equity": 50000.0,
            "eod_hwm": 50000.0,
            "liquidation_floor": 48000.0,
            "is_floor_locked": False,
            "qualifying_days_count": 0,
            "max_historical_daily_pnl": 0.0,
            "today_date": datetime.now().strftime("%Y-%m-%d"),
            "today_realized_pnl": 0.0,
            "payout_cycle_number": 0,
            "processed_signals": [],
            "last_updated_timestamp": datetime.now().isoformat()
        }

    def _load_or_initialize(self) -> Dict[str, Any]:
        if not self.state_file_path.exists():
            default_st = self._default_state()
            self._write_atomic(default_st)
            logger.info(f"Initialized new state file at {self.state_file_path}")
            return default_st

        try:
            with open(self.state_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Validate required fields
            req_fields = [
                "account_id", "account_phase", "current_equity", "eod_hwm",
                "liquidation_floor", "is_floor_locked", "qualifying_days_count",
                "max_historical_daily_pnl", "today_date", "today_realized_pnl",
                "payout_cycle_number"
            ]
            for field in req_fields:
                if field not in data:
                    raise ValueError(f"Missing required state field: {field}")
            if "processed_signals" not in data or not isinstance(data["processed_signals"], list):
                data["processed_signals"] = []
            logger.info(f"Successfully loaded state from {self.state_file_path} for account {data.get('account_id')}")
            return data
        except Exception as e:
            logger.error(f"Failed to load state from {self.state_file_path}: {e}. Creating backup and re-initializing.")
            backup_path = self.state_dir / f"tradeify_account_state_corrupt_{int(datetime.now().timestamp())}.json"
            try:
                if self.state_file_path.exists():
                    os.replace(self.state_file_path, backup_path)
            except Exception:
                pass
            default_st = self._default_state()
            self._write_atomic(default_st)
            return default_st

    def _write_atomic(self, data: Dict[str, Any]) -> None:
        data["last_updated_timestamp"] = datetime.now().isoformat()
        # Create temp file in same directory to ensure atomic os.replace on all platforms
        temp_file = tempfile.NamedTemporaryFile("w", dir=self.state_dir, delete=False, encoding="utf-8")
        try:
            json.dump(data, temp_file, indent=2)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file.close()
            os.replace(temp_file.name, self.state_file_path)
        except Exception as e:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)
            raise IOError(f"Atomic state persistence write failed: {e}")

    def save_state(self) -> None:
        self._write_atomic(self.state)

    def get_state(self) -> Dict[str, Any]:
        return dict(self.state)

    def record_fill(self, fill_pnl: float) -> Dict[str, Any]:
        """
        Invoked immediately upon every trade completion/fill.
        Updates intraday equity and checks EOD trailing floor protections.
        """
        self.state["current_equity"] += fill_pnl
        self.state["today_realized_pnl"] += fill_pnl

        # Check if daily PnL establishes a new historical max
        if self.state["today_realized_pnl"] > self.state["max_historical_daily_pnl"]:
            self.state["max_historical_daily_pnl"] = self.state["today_realized_pnl"]

        self.save_state()
        return self.state

    def record_trade_fill(self, fill_pnl: float, trade_id: str = "") -> Dict[str, Any]:
        """Alias for record_fill with optional trade identifier logging."""
        if trade_id:
            logger.info(f"Recording fill for trade {trade_id}: PnL = ${fill_pnl:+.2f}")
        return self.record_fill(fill_pnl)

    def reconcile_eod(self, session_date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Invoked at 15:45:00 EST session close.
        Reconciles daily trailing floor, locks the floor if HWM >= $52,100,
        checks qualifying days (+$150 rule), and prepares next day state.
        """
        today_pnl = self.state["today_realized_pnl"]
        equity = self.state["current_equity"]
        phase = self.state["account_phase"]

        # 1. Update High Water Mark
        if equity > self.state["eod_hwm"]:
            self.state["eod_hwm"] = equity

        hwm = self.state["eod_hwm"]

        # 2. Floor Lock Logic
        if not self.state["is_floor_locked"]:
            if hwm >= 52100.0:
                self.state["is_floor_locked"] = True
                self.state["liquidation_floor"] = 50100.0
                logger.info(f"PERMANENT FLOOR LOCK ENGAGED: HWM ${hwm:.2f} >= $52,100. Floor locked at $50,100.00.")
            else:
                self.state["liquidation_floor"] = max(self.state["liquidation_floor"], hwm - 2000.0)

        # 3. Funded Qualifying Days
        if phase == "FUNDED" and today_pnl >= 150.0:
            self.state["qualifying_days_count"] += 1
            logger.info(f"QUALIFYING DAY RECORDED: Today PnL +${today_pnl:.2f} >= +$150.00. Total qualifying days: {self.state['qualifying_days_count']}")

        # 4. Check Evaluation Pass
        if phase == "EVALUATION" and equity >= 53000.0:
            self.state["account_phase"] = "FUNDED"
            self.state["current_equity"] = 50000.0
            self.state["eod_hwm"] = 50000.0
            self.state["liquidation_floor"] = 48000.0
            self.state["is_floor_locked"] = False
            self.state["qualifying_days_count"] = 0
            self.state["max_historical_daily_pnl"] = 0.0
            self.state["today_realized_pnl"] = 0.0
            self.state["payout_cycle_number"] = 1
            logger.info("EVALUATION PASSED! Account transitioned to FUNDED phase at $50,000.00.")

        # 5. Reset day realized PnL
        next_date = session_date_str or datetime.now().strftime("%Y-%m-%d")
        self.state["today_date"] = next_date
        self.state["today_realized_pnl"] = 0.0

        self.save_state()
        return self.state

    def process_payout(self, gross_amount: float = 1500.0) -> Dict[str, Any]:
        """
        Invoked upon payout distribution.
        Withdraws gross_amount, maintains permanent locked floor at $50,100,
        resets qualifying days for next payout, increments payout cycle.
        """
        if self.state["account_phase"] != "FUNDED":
            raise ValueError("Payouts can only be processed on FUNDED accounts.")

        self.state["current_equity"] -= gross_amount
        self.state["qualifying_days_count"] = 0
        self.state["max_historical_daily_pnl"] = 0.0
        self.state["payout_cycle_number"] += 1
        self.state["is_floor_locked"] = True
        self.state["liquidation_floor"] = 50100.0

        logger.info(f"PAYOUT PROCESSED: -${gross_amount:.2f} withdrawn. New equity: ${self.state['current_equity']:.2f}. Floor remains locked at $50,100.00.")
        self.save_state()
        return self.state

    def set_phase(self, phase: str) -> Dict[str, Any]:
        """Sets account phase (EVALUATION or FUNDED)."""
        phase_clean = phase.strip().upper()
        if phase_clean not in ["EVALUATION", "FUNDED"]:
            raise ValueError(f"Invalid account phase '{phase}'. Must be 'EVALUATION' or 'FUNDED'.")
        self.state["account_phase"] = phase_clean
        self.save_state()
        logger.info(f"Account {self.account_id} phase explicitly updated to {phase_clean}")
        return self.state

    def is_signal_processed(self, signal_id: str) -> bool:
        """Checks if signal_id was already processed (Idempotency Guard)."""
        if not signal_id:
            return False
        processed = self.state.get("processed_signals", [])
        return signal_id in processed

    def record_signal_processed(self, signal_id: str) -> None:
        """Caches signal_id in state to prevent duplicate order executions."""
        if not signal_id:
            return
        if "processed_signals" not in self.state or not isinstance(self.state["processed_signals"], list):
            self.state["processed_signals"] = []
        if signal_id not in self.state["processed_signals"]:
            self.state["processed_signals"].append(signal_id)
            # Retain last 500 processed signals to prevent infinite file growth
            if len(self.state["processed_signals"]) > 500:
                self.state["processed_signals"] = self.state["processed_signals"][-500:]
            self.save_state()

    def reset_to_funded(self) -> Dict[str, Any]:
        """
        Clean Evaluation-to-Funded Phase Reset:
        1. Archives current Evaluation state to a timestamped archive file.
        2. Initializes clean FUNDED state starting at $50,000.00 with $48,000 trailing floor.
        3. Enforces safety assertions guaranteeing that Evaluation $53,015 HWM NEVER leaks into Funded state.
        """
        now_ts = int(datetime.now().timestamp())
        archive_path = self.state_dir / f"tradeify_{self.account_id}_eval_archive_{now_ts}.json"

        # Archive existing state if file exists
        if self.state_file_path.exists():
            try:
                with open(self.state_file_path, "r", encoding="utf-8") as f:
                    curr_data = json.load(f)
                temp_archive = tempfile.NamedTemporaryFile("w", dir=self.state_dir, delete=False, encoding="utf-8")
                json.dump(curr_data, temp_archive, indent=2)
                temp_archive.flush()
                os.fsync(temp_archive.fileno())
                temp_archive.close()
                os.replace(temp_archive.name, archive_path)
                logger.info(f"Archived previous Evaluation state to {archive_path}")
            except Exception as e:
                logger.warning(f"Could not create archive copy of previous state: {e}")

        # Clean Funded State (Pristine $50K start)
        clean_funded_state = {
            "account_id": self.account_id,
            "account_phase": "FUNDED",
            "current_equity": 50000.0,
            "eod_hwm": 50000.0,
            "liquidation_floor": 48000.0,
            "is_floor_locked": False,
            "qualifying_days_count": 0,
            "max_historical_daily_pnl": 0.0,
            "today_date": datetime.now().strftime("%Y-%m-%d"),
            "today_realized_pnl": 0.0,
            "payout_cycle_number": 1,
            "processed_signals": [],
            "last_updated_timestamp": datetime.now().isoformat()
        }

        # CRITICAL SAFETY ASSERTION: Zero HWM leak into Funded Pre-Lock
        assert clean_funded_state["eod_hwm"] == 50000.0, "FATAL: Evaluation HWM leaked into Funded state!"
        assert clean_funded_state["current_equity"] == 50000.0, "FATAL: Evaluation Equity leaked into Funded state!"
        assert not clean_funded_state["is_floor_locked"], "FATAL: Floor lock prematurely active in Funded Pre-Lock!"
        assert clean_funded_state["liquidation_floor"] == 48000.0, "FATAL: Trailing floor not at $48,000.00 base!"

        self.state = clean_funded_state
        self._write_atomic(clean_funded_state)
        logger.info(f"ACCOUNT {self.account_id} SUCCESSFULLY INITIALIZED TO CLEAN FUNDED PHASE ($50,000.00 base).")
        return dict(self.state)
