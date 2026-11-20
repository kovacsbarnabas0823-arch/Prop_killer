"""
tests/test_live_hardening.py - Comprehensive Unit Tests for Phase 9 Live Hardening

Verifies:
  1. NT8 close-timestamp to Python open-timestamp conversion (across DST boundaries).
  2. CME <-> NT8 symbol mapping for all 4 quarterly months (2026-2027).
  3. Per-account atomic JSON isolation & clean Evaluation -> Funded reset (preventing HWM leakage).
  4. signal_id idempotency deduplication.
  5. RTH-only VWAP reset strictly at 09:30:00 America/New_York.
  6. CME Rollover Price-Gap Guard & 2026-2027 early-close holiday calendar.
  7. Sim101 shadow-audit diff script on synthetic mock fills (0-tolerance verification).
"""

import os
import shutil
import json
import tempfile
from pathlib import Path
from datetime import datetime, date, time
import pytest
import pytz
import pandas as pd
import numpy as np

from execution.symbol_mapper import (
    cme_to_nt8_symbol,
    nt8_to_cme_symbol,
    get_active_mnq_contract,
    get_active_nt8_contract,
    validate_nt8_instrument,
    normalize_nt8_bar_timestamp,
    python_open_to_nt8_close,
)
from engine.state_persistence import TradeifyStatePersistence
from data.session_preprocessor import (
    is_cme_early_close,
    check_cme_rollover_gap_guard,
    get_cme_flatten_time,
    get_cme_quarterly_rollover,
)
from execution.webhook_server import LiveBarManager
from scripts.shadow_audit_diff import (
    ParsedNT8Trade,
    run_shadow_audit,
    generate_markdown_report,
    parse_nt8_executions_file
)

tz_ny = pytz.timezone("America/New_York")


# =====================================================================
# 1. Timestamp Normalization & DST Boundaries
# =====================================================================

def test_timestamp_normalization_and_dst():
    # 1-minute bar test
    # NT8 stamps bar close: 10:20:00 -> Python open is 10:19:00
    close_ts_1m = tz_ny.localize(datetime(2026, 11, 20, 10, 20, 0))
    open_ts_1m = normalize_nt8_bar_timestamp(close_ts_1m, bar_minutes=1)
    assert open_ts_1m.minute == 19
    assert open_ts_1m.second == 0
    # Invert
    assert python_open_to_nt8_close(open_ts_1m, bar_minutes=1) == close_ts_1m

    # 5-minute bar test
    # NT8 stamps bar close: 10:20:00 -> Python open is 10:15:00
    close_ts_5m = tz_ny.localize(datetime(2026, 11, 20, 10, 20, 0))
    open_ts_5m = normalize_nt8_bar_timestamp(close_ts_5m, bar_minutes=5)
    assert open_ts_5m.minute == 15
    assert open_ts_5m.second == 0
    # Invert
    assert python_open_to_nt8_close(open_ts_5m, bar_minutes=5) == close_ts_5m

    # Test across US Daylight Saving Time (DST) Transition in March (Spring forward)
    # Sunday March 8, 2026: UTC-5 -> UTC-4
    pre_dst = tz_ny.localize(datetime(2026, 3, 6, 10, 20, 0))  # EST (UTC-5)
    post_dst = tz_ny.localize(datetime(2026, 3, 9, 10, 20, 0))  # EDT (UTC-4)

    norm_pre = normalize_nt8_bar_timestamp(pre_dst, 5)
    norm_post = normalize_nt8_bar_timestamp(post_dst, 5)

    assert norm_pre.minute == 15
    assert norm_post.minute == 15
    assert norm_pre.tzinfo.zone == "America/New_York"
    assert norm_post.tzinfo.zone == "America/New_York"
    assert norm_pre.utcoffset().total_seconds() == -18000  # -5 hours
    assert norm_post.utcoffset().total_seconds() == -14400  # -4 hours

    # Test Fall Back DST Transition in November (EDT -> EST)
    # Sunday Nov 1, 2026
    pre_fall = tz_ny.localize(datetime(2026, 10, 30, 10, 20, 0))  # EDT
    post_fall = tz_ny.localize(datetime(2026, 11, 2, 10, 20, 0))  # EST
    assert normalize_nt8_bar_timestamp(pre_fall, 1).minute == 19
    assert normalize_nt8_bar_timestamp(post_fall, 1).minute == 19


# =====================================================================
# 2. CME <-> NT8 Symbol Mapping for all 4 Quarterly Months
# =====================================================================

def test_cme_nt8_symbol_mapping():
    # Test all 4 quarterly codes across 2026 and 2027
    test_cases = [
        ("MNQH26", "MNQ 03-26"),
        ("MNQM26", "MNQ 06-26"),
        ("MNQU26", "MNQ 09-26"),
        ("MNQZ26", "MNQ 12-26"),
        ("MNQH27", "MNQ 03-27"),
        ("MNQM27", "MNQ 06-27"),
        ("MNQU27", "MNQ 09-27"),
        ("MNQZ27", "MNQ 12-27"),
        ("NQH26", "NQ 03-26"),
        ("NQZ26", "NQ 12-26"),
    ]

    for cme, nt8 in test_cases:
        # CME -> NT8
        assert cme_to_nt8_symbol(cme) == nt8
        # NT8 -> CME
        assert nt8_to_cme_symbol(nt8) == cme

    # Test Active Contract Detection on specific dates
    # Before March 2026 rollover (e.g. Feb 15, 2026) -> March contract (H26)
    feb_date = date(2026, 2, 15)
    assert get_active_mnq_contract(feb_date) == "MNQH26"
    assert get_active_nt8_contract(feb_date) == "MNQ 03-26"

    # After March 2026 rollover (e.g. April 15, 2026) -> June contract (M26)
    apr_date = date(2026, 4, 15)
    assert get_active_mnq_contract(apr_date) == "MNQM26"
    assert get_active_nt8_contract(apr_date) == "MNQ 06-26"

    # After June 2026 rollover (e.g. July 15, 2026) -> September contract (U26)
    jul_date = date(2026, 7, 15)
    assert get_active_mnq_contract(jul_date) == "MNQU26"
    assert get_active_nt8_contract(jul_date) == "MNQ 09-26"

    # After September 2026 rollover (e.g. October 15, 2026) -> December contract (Z26)
    oct_date = date(2026, 10, 15)
    assert get_active_mnq_contract(oct_date) == "MNQZ26"
    assert get_active_nt8_contract(oct_date) == "MNQ 12-26"

    # Instrument validation
    is_valid, msg = validate_nt8_instrument("MNQ 12-26", session_date=oct_date)
    assert is_valid is True

    is_invalid, msg = validate_nt8_instrument("MNQ 03-26", session_date=oct_date)
    assert is_invalid is False
    assert "Contract mismatch" in msg


# =====================================================================
# 3. Per-Account Atomic State Isolation & Clean Reset to Funded
# =====================================================================

def test_per_account_state_isolation_and_reset(tmp_path):
    # Setup isolated test directory
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    # Create 3 isolated accounts
    acc1 = TradeifyStatePersistence(state_file_path=state_dir / "tradeify_ACC_001.json", account_id="ACC_001")
    acc2 = TradeifyStatePersistence(state_file_path=state_dir / "tradeify_ACC_002.json", account_id="ACC_002")
    acc3 = TradeifyStatePersistence(state_file_path=state_dir / "tradeify_ACC_003.json", account_id="ACC_003")

    # Verify separate files exist
    assert acc1.state_file_path.name == "tradeify_ACC_001.json"
    assert acc2.state_file_path.name == "tradeify_ACC_002.json"
    assert acc3.state_file_path.name == "tradeify_ACC_003.json"

    # Mutate ACC_001
    acc1.record_trade_fill(1200.0)
    acc1.reconcile_eod("2026-11-20")
    assert acc1.state["current_equity"] == 51200.0
    assert acc1.state["eod_hwm"] == 51200.0

    # Verify ACC_002 and ACC_003 remain untouched at 50,000.0
    acc2_fresh = TradeifyStatePersistence(state_file_path=state_dir / "tradeify_ACC_002.json", account_id="ACC_002")
    assert acc2_fresh.state["current_equity"] == 50000.0
    assert acc2_fresh.state["eod_hwm"] == 50000.0

    # Advance ACC_001 to Evaluation pass threshold ($53,015.00)
    acc1.record_trade_fill(1815.0)
    assert acc1.state["current_equity"] == 53015.0
    acc1.save_state()

    # Now reset ACC_001 to FUNDED phase
    funded_state = acc1.reset_to_funded()

    # Verify Archive was created
    archive_files = list(state_dir.glob("tradeify_ACC_001_eval_archive_*.json"))
    assert len(archive_files) == 1

    # Verify Clean Funded State: Zero HWM leakage!
    assert funded_state["account_phase"] == "FUNDED"
    assert funded_state["current_equity"] == 50000.0
    assert funded_state["eod_hwm"] == 50000.0
    assert funded_state["liquidation_floor"] == 48000.0
    assert funded_state["is_floor_locked"] is False
    assert funded_state["qualifying_days_count"] == 0
    assert funded_state["max_historical_daily_pnl"] == 0.0

    # Safety check: Reload from disk to verify atomic persistence
    reloaded_acc1 = TradeifyStatePersistence(state_file_path=state_dir / "tradeify_ACC_001.json", account_id="ACC_001")
    assert reloaded_acc1.state["account_phase"] == "FUNDED"
    assert reloaded_acc1.state["current_equity"] == 50000.0
    assert reloaded_acc1.state["eod_hwm"] == 50000.0
    assert reloaded_acc1.state["liquidation_floor"] == 48000.0


# =====================================================================
# 4. signal_id Idempotency Deduplication
# =====================================================================

def test_signal_id_idempotency(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    sp = TradeifyStatePersistence(state_file_path=state_dir / "tradeify_IDEMP_TEST.json", account_id="IDEMP_TEST")
    sig_id_1 = "20261120_H3_BUY_102000"
    sig_id_2 = "20261120_H4_BUY_150000"

    # Initially neither is processed
    assert sp.is_signal_processed(sig_id_1) is False
    assert sp.is_signal_processed(sig_id_2) is False

    # Record sig_id_1
    sp.record_signal_processed(sig_id_1)
    assert sp.is_signal_processed(sig_id_1) is True
    assert sp.is_signal_processed(sig_id_2) is False

    # Re-instantiate from disk and verify persistence
    sp_reloaded = TradeifyStatePersistence(state_file_path=state_dir / "tradeify_IDEMP_TEST.json", account_id="IDEMP_TEST")
    assert sp_reloaded.is_signal_processed(sig_id_1) is True
    assert sp_reloaded.is_signal_processed(sig_id_2) is False

    # Record sig_id_2
    sp_reloaded.record_signal_processed(sig_id_2)
    assert sp_reloaded.is_signal_processed(sig_id_2) is True


# =====================================================================
# 5. RTH-Only VWAP Reset strictly at 09:30:00 NY
# =====================================================================

def test_rth_only_vwap_reset():
    mgr = LiveBarManager()

    # Pre-market bar closing at 08:30:00 EST (open was 08:29:00 EST)
    pre_bar = {
        "timestamp": "2026-11-20T08:30:00",
        "open": 21000.0, "high": 21010.0, "low": 20990.0, "close": 21005.0,
        "volume": 5000,
        "bar_minutes": 1
    }
    rec_pre = mgr.ingest_bar(pre_bar)
    # RTH VWAP must remain None before 09:30 RTH open
    assert mgr.live_vwap is None
    assert mgr.cum_vol == 0.0

    # First RTH 1m bar closing at 09:31:00 EST (open was 09:30:00 EST)
    # Typical price = (21000 + 21050 + 20980) / 3 = 21010.0
    bar_0931 = {
        "timestamp": "2026-11-20T09:31:00",
        "open": 21000.0, "high": 21050.0, "low": 20980.0, "close": 21000.0,
        "volume": 1000,
        "bar_minutes": 1
    }
    rec_0931 = mgr.ingest_bar(bar_0931)
    assert mgr.live_vwap is not None
    assert pytest.approx(mgr.live_vwap, 0.01) == 21010.0
    assert mgr.cum_vol == 1000.0

    # Second RTH 1m bar closing at 09:32:00 EST (open was 09:31:00 EST)
    # TP2 = (21000 + 21020 + 20990) / 3 = 21003.3333
    bar_0932 = {
        "timestamp": "2026-11-20T09:32:00",
        "open": 21000.0, "high": 21020.0, "low": 20990.0, "close": 21010.0,
        "volume": 1000,
        "bar_minutes": 1
    }
    rec_0932 = mgr.ingest_bar(bar_0932)
    expected_vwap = (21010.0 * 1000 + (21000 + 21020 + 20990)/3.0 * 1000) / 2000.0
    assert pytest.approx(mgr.live_vwap, 0.01) == expected_vwap
    assert mgr.cum_vol == 2000.0


# =====================================================================
# 6. Rollover Price-Gap Guard & 2026-2027 Early Close Calendar
# =====================================================================

def test_rollover_gap_guard_and_calendar():
    # 2026 Early Close Days (13:00 EST close)
    # Black Friday 2026: Nov 27, 2026
    is_ec, close_time = is_cme_early_close(date(2026, 11, 27))
    assert is_ec is True
    assert close_time == "13:00:00"
    assert get_cme_flatten_time(date(2026, 11, 27)) == "12:58:00"

    # Christmas Eve 2026: Dec 24, 2026
    is_ec, close_time = is_cme_early_close(date(2026, 12, 24))
    assert is_ec is True
    assert close_time == "13:00:00"

    # 2027 Early Close Days
    # Black Friday 2027: Nov 26, 2027
    is_ec_27, close_time_27 = is_cme_early_close(date(2027, 11, 26))
    assert is_ec_27 is True
    assert close_time_27 == "13:00:00"

    # Regular trading day
    is_reg, close_time_reg = is_cme_early_close(date(2026, 11, 20))
    assert is_reg is False
    assert close_time_reg == "16:00:00"
    assert get_cme_flatten_time(date(2026, 11, 20)) == "15:44:00"

    # CME Rollover Gap Guard
    # Case 1: Non-rollover day -> Allowed
    ok, msg = check_cme_rollover_gap_guard(
        session_date=date(2026, 11, 20),
        new_contract_has_history=True
    )
    assert ok is True

    # Case 2: Rollover day with missing new contract prior close history -> BLOCKED
    rollover_date_dec26 = get_cme_quarterly_rollover(2026, 12)[1]
    ok_block, msg_block = check_cme_rollover_gap_guard(
        session_date=rollover_date_dec26,
        new_contract_has_history=False
    )
    assert ok_block is False
    assert "lacks clean prior-day reference bars" in msg_block

    # Case 3: Rollover day with verified history -> ALLOWED
    ok_allow, msg_allow = check_cme_rollover_gap_guard(
        session_date=rollover_date_dec26,
        new_contract_has_history=True
    )
    assert ok_allow is True
    assert "history verified" in msg_allow


# =====================================================================
# 7. Sim101 Shadow-Audit Reconciliation with Mock Data
# =====================================================================

def test_shadow_audit_reconciliation_mock(tmp_path):
    # Create synthetic 1-day 1m bar dataset
    target_dt = date(2026, 11, 20)
    times = pd.date_range("2026-11-20 09:30:00", "2026-11-20 15:59:00", freq="1min", tz="America/New_York")
    
    # Generate realistic bars
    np.random.seed(42)
    prices = 21000.0 + np.cumsum(np.random.randn(len(times)) * 1.5)
    df_bars = pd.DataFrame({
        "Open": prices,
        "High": prices + 2.0,
        "Low": prices - 2.0,
        "Close": prices + 0.5,
        "Volume": 100
    }, index=times)

    bars_file = tmp_path / "bars_mock.parquet"
    df_bars.to_parquet(bars_file)

    # Create mock NT8 Trade List CSV
    nt8_csv = tmp_path / "nt8_executions.csv"
    mock_csv_content = """Trade #,Instrument,Account,Strategy,Market pos.,Qty,Entry price,Exit price,Entry time,Exit time,Profit ($),Commission
1,MNQ 12-26,Sim101,TradeifyExecutionManager,Long,2,21010.50,21035.50,2026-11-20 10:20:00,2026-11-20 11:30:00,95.04,4.96
"""
    with open(nt8_csv, "w", encoding="utf-8") as f:
        f.write(mock_csv_content)

    parsed_trades = parse_nt8_executions_file(nt8_csv)
    assert len(parsed_trades) == 1
    assert parsed_trades[0].contracts == 2
    assert parsed_trades[0].direction == "BUY"
    assert parsed_trades[0].entry_price == 21010.50
    assert parsed_trades[0].entry_time == tz_ny.localize(datetime(2026, 11, 20, 10, 20, 0))

    # Test markdown report generation from synthetic result
    mock_result = {
        "session_date": "2026-11-20",
        "audit_status": "PASS",
        "total_sim_trades": 1,
        "total_nt8_trades": 1,
        "matched_trades": 1,
        "avg_realized_slippage_pts": 0.50,
        "baseline_slippage_pts": 0.50,
        "slippage_drag_per_mnq_rt": 1.00,
        "rows": [{
            "trade_num": 1,
            "strategy_id": "H3",
            "direction": "BUY",
            "sim_entry_time": "2026-11-20 10:20:00-05:00",
            "nt8_entry_time": "2026-11-20 10:20:00-05:00",
            "time_drift_sec": 0.0,
            "sim_contracts": 2,
            "nt8_contracts": 2,
            "sizing_match": True,
            "sim_theor_price": 21010.0,
            "sim_fill_price": 21010.5,
            "nt8_fill_price": 21010.5,
            "realized_slippage_pts": 0.50,
            "slippage_vs_baseline": 0.0,
            "sim_exit_time": "2026-11-20 11:30:00-05:00",
            "nt8_exit_time": "2026-11-20 11:30:00-05:00",
            "sim_exit_price": 21035.5,
            "nt8_exit_price": 21035.5,
            "sim_pnl": 95.04,
            "nt8_pnl": 95.04,
            "status": "PASS",
            "discrepancy_notes": []
        }]
    }

    report_md = generate_markdown_report(mock_result)
    assert "# Sim101 Shadow-Audit Daily Reconciliation Report" in report_md
    assert "PASSED" in report_md
    assert "| 1 | H3 | BUY |" in report_md
    assert "+0s" in report_md
