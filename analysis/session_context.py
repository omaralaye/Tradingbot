"""
analysis/session_context.py
----------------------------
Determines which trading sessions are active at a given UTC datetime
and provides a volatility profile accordingly.

Session windows are loaded from config/sessions.yaml and are configurable.
Crypto instruments trade 24/7 — session logic affects volatility
expectations only, not whether the market is "open."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from typing import Optional

from loguru import logger


@dataclass
class SessionInfo:
    """Snapshot of the current session state.

    Attributes:
        active_sessions:   Names of currently active sessions (e.g. ['london', 'new_york']).
        is_overlap:        True if two major sessions overlap (higher volatility).
        overlap_name:      Name of the active overlap window, if any.
        volatility_profile: 'low' | 'medium' | 'high' | 'very_high'.
    """
    active_sessions:   list[str]           = field(default_factory=list)
    is_overlap:        bool                = False
    overlap_name:      Optional[str]       = None
    volatility_profile: str               = "low"


class SessionContext:
    """Evaluates trading session state from a sessions.yaml config dict.

    Example usage:
        import yaml
        with open('config/sessions.yaml') as f:
            cfg = yaml.safe_load(f)
        ctx = SessionContext(cfg)
        info = ctx.get_current_session()
    """

    _VOLATILITY_RANK = {"low": 0, "medium": 1, "high": 2, "very_high": 3}
    _VOLATILITY_MULTIPLIER = {"low": 0.7, "medium": 1.0, "high": 1.3, "very_high": 1.6}

    def __init__(self, sessions_config: dict):
        """
        Args:
            sessions_config: Parsed sessions.yaml dict with 'sessions' and 'overlaps' keys.
        """
        self._sessions  = sessions_config.get("sessions", {})
        self._overlaps  = sessions_config.get("overlaps", {})
        self._avoid_windows = sessions_config.get("avoid_trading_windows", [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_session(self, dt: Optional[datetime] = None) -> SessionInfo:
        """Return session information for the given UTC datetime.

        Args:
            dt: UTC datetime; defaults to now() if not provided.

        Returns:
            SessionInfo describing active sessions and volatility.
        """
        dt = dt or datetime.now(tz=timezone.utc)
        current_time = dt.time()

        active = []
        for name, config in self._sessions.items():
            if self._is_time_in_window(current_time, config["open"], config["close"]):
                active.append(name)

        # Check overlaps
        overlap_name = None
        for name, config in self._overlaps.items():
            if self._is_time_in_window(current_time, config["open"], config["close"]):
                overlap_name = name
                break

        # Determine volatility profile
        if overlap_name:
            vol = self._overlaps[overlap_name].get("volatility", "high")
        elif active:
            vols = [self._sessions[s].get("volatility", "low") for s in active]
            vol  = max(vols, key=lambda v: self._VOLATILITY_RANK.get(v, 0))
        else:
            vol = "low"

        return SessionInfo(
            active_sessions=active,
            is_overlap=overlap_name is not None,
            overlap_name=overlap_name,
            volatility_profile=vol,
        )

    def is_high_volatility(self, dt: Optional[datetime] = None) -> bool:
        """Return True if currently in a high or very_high volatility window."""
        info = self.get_current_session(dt)
        return self._VOLATILITY_RANK.get(info.volatility_profile, 0) >= 2

    def get_volatility_multiplier(self, dt: Optional[datetime] = None) -> float:
        """Return a volatility multiplier to scale ATR-based stop distances.

        Returns a float typically in [0.7, 1.6]:
          0.7 = quiet period (tighter stops may work)
          1.6 = London/NY overlap (wider stops needed to avoid noise)
        """
        info = self.get_current_session(dt)
        return self._VOLATILITY_MULTIPLIER.get(info.volatility_profile, 1.0)

    def is_avoided_window(self, dt: Optional[datetime] = None) -> bool:
        """Return True if the current time falls in a configured avoid_trading_window."""
        dt = dt or datetime.now(tz=timezone.utc)
        current_time = dt.time()
        for window in self._avoid_windows:
            start_str, end_str = window.split("-")
            if self._is_time_in_window(current_time, start_str.strip(), end_str.strip()):
                logger.debug("In avoided trading window: {}", window)
                return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_time(t_str: str) -> time:
        """Parse 'HH:MM' string to time object."""
        h, m = map(int, t_str.split(":"))
        return time(h, m)

    def _is_time_in_window(self, current: time, open_str: str, close_str: str) -> bool:
        """Check if current time falls in [open, close), handling midnight crossover."""
        open_t  = self._parse_time(open_str)
        close_t = self._parse_time(close_str)

        if open_t < close_t:
            # Normal window (e.g. 08:00-17:00)
            return open_t <= current < close_t
        else:
            # Crosses midnight (e.g. Sydney 22:00-07:00)
            return current >= open_t or current < close_t
