"""
Symbol Format Mapping & Timestamp Normalization Module
Author: Senior Quantitative Systems Engineer

Provides:
  1. Bidirectional CME <-> NinjaTrader 8 symbol mapping:
       MNQZ26 / MNQZ6 <-> MNQ 12-26
       MNQH27 / MNQH7 <-> MNQ 03-27
       MNQM27 / MNQM7 <-> MNQ 06-27
       MNQU27 / MNQU7 <-> MNQ 09-27
  2. Front-month validation against expected active contract on trading date.
  3. Bar Timestamp Normalization:
       NinjaTrader 8 stamps bars by CLOSE time (e.g. 10:20:00 for 10:15-10:20).
       Python historical bars use OPEN time (10:15:00 for 10:15-10:20).
       Provides exact DST-safe bidirectional reconciliation.
"""

from __future__ import annotations
import re
from datetime import datetime, date, timedelta
from typing import Tuple, Optional
import pytz

TZ_NY = pytz.timezone("America/New_York")

MONTH_CODE_TO_NUM = {
    'H': '03',  # March
    'M': '06',  # June
    'U': '09',  # September
    'Z': '12',  # December
}

NUM_TO_MONTH_CODE = {
    '03': 'H',
    '3': 'H',
    '06': 'M',
    '6': 'M',
    '09': 'U',
    '9': 'U',
    '12': 'Z',
}


def cme_to_nt8_symbol(cme_symbol: str) -> str:
    """
    Converts CME Globex contract format to NinjaTrader 8 format.
    Examples:
      'MNQZ26' -> 'MNQ 12-26'
      'MNQZ6'  -> 'MNQ 12-26'
      'MNQH27' -> 'MNQ 03-27'
      'MNQH7'  -> 'MNQ 03-27'
      'NQZ26'  -> 'NQ 12-26'
    """
    clean_sym = cme_symbol.strip().upper()
    # Match pattern: (MNQ|NQ)([HMUZ])(\d{1,2})
    match = re.match(r"^([A-Z]+)([HMUZ])(\d{1,2})$", clean_sym)
    if not match:
        raise ValueError(f"Unrecognized CME symbol format: '{cme_symbol}'. Expected format like 'MNQZ26' or 'MNQZ6'.")

    root, m_code, yr = match.groups()
    month_str = MONTH_CODE_TO_NUM[m_code]
    
    # Normalize 1-digit year to 2-digit year (e.g. '6' -> '26', '7' -> '27')
    if len(yr) == 1:
        # 2020s decade
        yr_str = f"2{yr}"
    else:
        yr_str = yr

    return f"{root} {month_str}-{yr_str}"


def nt8_to_cme_symbol(nt8_symbol: str, two_digit_year: bool = True) -> str:
    """
    Converts NinjaTrader 8 contract format to CME Globex format.
    Examples:
      'MNQ 12-26' -> 'MNQZ26' (or 'MNQZ6' if two_digit_year=False)
      'MNQ 03-27' -> 'MNQH27'
      'NQ 12-26'  -> 'NQZ26'
    """
    clean_sym = nt8_symbol.strip().upper()
    # Match pattern: (MNQ|NQ)\s*(\d{1,2})-(\d{2})
    match = re.match(r"^([A-Z]+)\s*(\d{1,2})-(\d{2})$", clean_sym)
    if not match:
        raise ValueError(f"Unrecognized NT8 symbol format: '{nt8_symbol}'. Expected format like 'MNQ 12-26'.")

    root, month_num, yr_2d = match.groups()
    padded_month = month_num.zfill(2)
    if padded_month not in NUM_TO_MONTH_CODE:
        raise ValueError(f"Invalid expiry month '{month_num}' in symbol '{nt8_symbol}'. Expected 03, 06, 09, or 12.")

    m_code = NUM_TO_MONTH_CODE[padded_month]
    yr_out = yr_2d if two_digit_year else yr_2d[-1]
    return f"{root}{m_code}{yr_out}"


def get_cme_quarterly_rollover(year: int, month: int) -> Tuple[date, date]:
    """
    Returns (third_friday, rollover_thursday) for CME equity index contracts in month (3, 6, 9, 12).
    CME Equity index rolls on the 2nd Thursday before the 3rd Friday of the expiry month.
    """
    first_day = date(year, month, 1)
    first_friday_day = 1 + (4 - first_day.weekday()) % 7
    third_friday = date(year, month, first_friday_day + 14)
    rollover_thursday = third_friday - timedelta(days=8)
    return third_friday, rollover_thursday


def get_active_mnq_contract(dt: Optional[date] = None, two_digit_year: bool = True) -> str:
    """
    Resolves the active front-month CME Micro E-mini Nasdaq (MNQ) contract symbol.
    Quarterly cycle:
      H = March, M = June, U = September, Z = December.
    Rollover occurs on the 2nd Thursday before the 3rd Friday of each expiry month.
    Returns: e.g. 'MNQZ26' (or 'MNQZ6' if two_digit_year=False).
    """
    if dt is None:
        dt = date.today()

    year = dt.year
    month_codes = {3: 'H', 6: 'M', 9: 'U', 12: 'Z'}
    quarters = [
        (year - 1, 12),
        (year, 3),
        (year, 6),
        (year, 9),
        (year, 12),
        (year + 1, 3)
    ]

    for i in range(len(quarters) - 1):
        y_curr, m_curr = quarters[i]
        y_next, m_next = quarters[i + 1]
        _, roll_curr = get_cme_quarterly_rollover(y_curr, m_curr)
        _, roll_next = get_cme_quarterly_rollover(y_next, m_next)
        if roll_curr <= dt < roll_next:
            yr_str = str(y_next)[-2:] if two_digit_year else str(y_next)[-1]
            return f"MNQ{month_codes[m_next]}{yr_str}"

    return "MNQ"


def get_active_nt8_contract(dt: Optional[date] = None) -> str:
    """
    Resolves the expected active front-month contract in NinjaTrader 8 format.
    Example: 'MNQ 12-26' on November 20, 2026.
    """
    cme_sym = get_active_mnq_contract(dt=dt, two_digit_year=True)
    return cme_to_nt8_symbol(cme_sym)


def validate_nt8_instrument(instrument_name: str, session_date: Optional[date] = None) -> Tuple[bool, str]:
    """
    Validates whether an incoming NT8 instrument name matches the expected front-month contract.
    Returns: (is_valid, message)
    """
    if session_date is None:
        session_date = date.today()

    expected_nt8 = get_active_nt8_contract(session_date)
    clean_input = instrument_name.strip().upper()

    # Exact or sub-match check
    if clean_input == expected_nt8.upper() or expected_nt8.upper() in clean_input:
        return True, f"Instrument '{instrument_name}' matches active front-month '{expected_nt8}'."
    
    # Also support if NT8 sends CME code like 'MNQZ26'
    try:
        nt8_converted = cme_to_nt8_symbol(clean_input)
        if nt8_converted == expected_nt8:
            return True, f"CME Instrument '{instrument_name}' mapped to valid active '{expected_nt8}'."
    except Exception:
        pass

    return False, f"Contract mismatch: received '{instrument_name}', expected active front-month '{expected_nt8}'."


def normalize_nt8_bar_timestamp(close_dt: datetime, bar_minutes: int = 1) -> datetime:
    """
    Converts NinjaTrader 8 bar-CLOSE timestamp to Python bar-OPEN timestamp.
    Example:
      NT8 1m bar closing at 10:20:00 -> Python 1m bar opening at 10:19:00.
      NT8 5m bar closing at 10:20:00 -> Python 5m bar opening at 10:15:00.
    Preserves timezone information in America/New_York.
    """
    if close_dt.tzinfo is None:
        localized_close = TZ_NY.localize(close_dt)
    else:
        localized_close = close_dt.astimezone(TZ_NY)

    open_dt = localized_close - timedelta(minutes=bar_minutes)
    return open_dt


def python_open_to_nt8_close(open_dt: datetime, bar_minutes: int = 1) -> datetime:
    """
    Converts Python bar-OPEN timestamp to NinjaTrader 8 bar-CLOSE timestamp.
    Example:
      Python 1m bar opening at 10:19:00 -> NT8 1m bar closing at 10:20:00.
      Python 5m bar opening at 10:15:00 -> NT8 5m bar closing at 10:20:00.
    """
    if open_dt.tzinfo is None:
        localized_open = TZ_NY.localize(open_dt)
    else:
        localized_open = open_dt.astimezone(TZ_NY)

    close_dt = localized_open + timedelta(minutes=bar_minutes)
    return close_dt
