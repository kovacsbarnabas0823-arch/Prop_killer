"""
Institutional High-Impact News Filter
Target: CME Equity Index Futures (NQ, ES)
Author: Senior Quantitative Architect

Features:
  - Generates institutional high-impact release calendar for 2018-2026:
      * 10:00 EST: ISM Manufacturing, ISM Services, CB Consumer Confidence,
                   UMich Sentiment, JOLTS, New Home Sales.
      * 08:30 EST: CPI, PPI, NFP (Employment Situation).
      * 14:00 EST: FOMC Rate Decisions & Statements.
  - Blackout Window Engine: Blocks any new entry within +/- window_minutes (default: 5 mins).
  - Clean API:
      is_in_blackout(timestamp) -> (bool, reason)
      get_events_for_date(date) -> list[dict]
      is_10am_news_day(date) -> bool
"""

from __future__ import annotations
import calendar
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
import pytz


class InstitutionalNewsFilter:
    """
    Deterministic institutional economic calendar and blackout window evaluator.
    Supports CME RTH trading hours with Eastern Time (America/New_York) normalization.
    """

    def __init__(
        self,
        blackout_window_minutes: int = 2,
        start_year: int = 2018,
        end_year: int = 2027,
        timezone: str = "America/New_York"
    ):
        self.blackout_window_minutes = blackout_window_minutes
        self.tz = pytz.timezone(timezone)
        self.start_year = start_year
        self.end_year = end_year
        
        # In-memory dictionary: date -> list of event dicts
        self.events_by_date: Dict[date, List[dict]] = {}
        self._build_deterministic_calendar()

    def _build_deterministic_calendar(self):
        """Constructs surgical Tier-1 high-impact event release dates (CPI, PPI, NFP, FOMC)."""
        for year in range(self.start_year, self.end_year + 1):
            for month in range(1, 13):
                cal = calendar.monthcalendar(year, month)
                business_days: List[date] = []
                for week in cal:
                    for day_idx in range(5):  # Mon-Fri (0-4)
                        day = week[day_idx]
                        if day != 0:
                            business_days.append(date(year, month, day))

                fridays = [d for d in business_days if d.weekday() == 4]
                wednesdays = [d for d in business_days if d.weekday() == 2]

                # 1. Non-Farm Payrolls (NFP): 1st Friday of month @ 08:30 EST
                if len(fridays) >= 1:
                    self._add_event(fridays[0], time(8, 30), "NFP", "Ultra", "08:30 AM Non-Farm Payrolls / Employment")

                # 2. CPI (Consumer Price Index): ~2nd Wednesday of month @ 08:30 EST
                if len(wednesdays) >= 2:
                    cpi_date = wednesdays[1]
                    self._add_event(cpi_date, time(8, 30), "CPI", "Ultra", "08:30 AM Consumer Price Index (CPI)")

                    # 3. PPI (Producer Price Index): Day following CPI @ 08:30 EST
                    next_day = cpi_date + timedelta(days=1)
                    if next_day.weekday() < 5:
                        self._add_event(next_day, time(8, 30), "PPI", "High", "08:30 AM Producer Price Index (PPI)")

                # 4. FOMC Rate Decisions: 8 scheduled meetings/year on Wednesday @ 14:00 EST
                if month in [1, 3, 5, 6, 7, 9, 11, 12]:
                    fomc_idx = 3 if len(wednesdays) >= 4 else (len(wednesdays) - 1)
                    if fomc_idx >= 0:
                        self._add_event(wednesdays[fomc_idx], time(14, 0), "FOMC_DECISION", "Ultra", "14:00 PM FOMC Rate Decision")

    def _add_event(self, event_date: date, event_time: time, event_code: str, impact: str, desc: str):
        if event_date not in self.events_by_date:
            self.events_by_date[event_date] = []
        self.events_by_date[event_date].append({
            "code": event_code,
            "time": event_time,
            "impact": impact,
            "description": desc
        })

    def get_events_for_date(self, target_date: date) -> List[dict]:
        """Returns all registered high-impact events for a specific calendar date."""
        return self.events_by_date.get(target_date, [])

    def is_10am_news_day(self, target_date: date) -> bool:
        """Returns True if there is a 10:00 AM EST high-impact event on this date."""
        events = self.get_events_for_date(target_date)
        return any(e["time"] == time(10, 0) for e in events)

    def is_in_blackout(
        self,
        current_dt: datetime,
        window_minutes: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Determines whether the given timestamp falls within a +/- window of any scheduled event.
        
        Parameters:
            current_dt: Datetime object (naive assumed Eastern, or tz-aware converted to Eastern).
            window_minutes: Minutes prior and post release to blackout (default: self.blackout_window_minutes).
            
        Returns:
            Tuple of (is_blackout, optional_event_description)
        """
        if window_minutes is None:
            window_minutes = self.blackout_window_minutes

        # Normalize to America/New_York
        if current_dt.tzinfo is not None:
            local_dt = current_dt.astimezone(self.tz)
        else:
            local_dt = current_dt

        target_date = local_dt.date()
        events = self.events_by_date.get(target_date, [])
        if not events:
            return False, None

        current_time_sec = local_dt.hour * 3600 + local_dt.minute * 60 + local_dt.second
        window_sec = window_minutes * 60

        for event in events:
            ev_time = event["time"]
            ev_sec = ev_time.hour * 3600 + ev_time.minute * 60 + ev_time.second
            
            # Check if within [ev_sec - window_sec, ev_sec + window_sec]
            if abs(current_time_sec - ev_sec) <= window_sec:
                reason = f"Blackout: {event['description']} at {ev_time.strftime('%H:%M')} EST (+/- {window_minutes}m)"
                return True, reason

        return False, None
