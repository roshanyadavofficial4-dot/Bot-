"""
Session Filter — Phase 2
==========================
Restricts trading to UTC windows with historically better liquidity and trend quality.

Crypto trades 24/7 but liquidity, spread, and trend quality vary drastically by hour.
The worst hours (dead zones) produce: wider spreads, false OFI signals, poor fills.

Session definitions (UTC):
  ASIA_OPEN     02:00–08:00  (moderate — decent for XRP/ADA/DOGE)
  LONDON_OPEN   08:00–12:00  (best — high volume, strong trends)
  NY_OVERLAP    12:00–17:00  (best — highest volume globally)
  NY_CLOSE      17:00–20:00  (acceptable)
  DEAD_ZONE     20:00–02:00  (skip — low volume, manipulated wicks)

Config override: TRADE_START_HOUR_UTC and TRADE_END_HOUR_UTC in config.py
take precedence if set. This filter only adds the dead-zone block on top.
"""

import logging
from datetime import datetime, timezone
from typing import Tuple

from config import TRADE_START_HOUR_UTC, TRADE_END_HOUR_UTC

logger = logging.getLogger("SessionFilter")

# Dead zone: skip trading during these UTC hours (low liquidity)
DEAD_ZONE_START = 20   # 8 PM UTC
DEAD_ZONE_END   = 2    # 2 AM UTC  (wraps midnight)


class SessionFilter:
    """
    Stateless session gate.

    Usage:
        sf = SessionFilter()
        ok, session_name, reason = sf.is_active_session()
    """

    def is_active_session(
        self,
        utc_hour: int = None,
    ) -> Tuple[bool, str, str]:
        """
        Returns (is_active: bool, session_name: str, reason: str).

        utc_hour: override for testing (0–23). Defaults to current UTC hour.
        """
        if utc_hour is None:
            utc_hour = datetime.now(timezone.utc).hour

        # ── Config window check ───────────────────────────────────────────────
        # TRADE_START/END_HOUR_UTC are the existing config-level gates
        in_config_window = (
            TRADE_START_HOUR_UTC <= utc_hour < TRADE_END_HOUR_UTC
        )
        if not in_config_window:
            return (
                False,
                "OUT_OF_HOURS",
                f"hour={utc_hour} outside config window "
                f"[{TRADE_START_HOUR_UTC},{TRADE_END_HOUR_UTC})"
            )

        # ── Dead zone check ───────────────────────────────────────────────────
        in_dead_zone = (
            utc_hour >= DEAD_ZONE_START or utc_hour < DEAD_ZONE_END
        )
        if in_dead_zone:
            return (
                False,
                "DEAD_ZONE",
                f"hour={utc_hour} in dead zone "
                f"[{DEAD_ZONE_START}:00–{DEAD_ZONE_END:02d}:00 UTC] "
                f"(low liquidity, wide spreads)"
            )

        # ── Classify active session ───────────────────────────────────────────
        if 2 <= utc_hour < 8:
            session = "ASIA_OPEN"
        elif 8 <= utc_hour < 12:
            session = "LONDON_OPEN"
        elif 12 <= utc_hour < 17:
            session = "NY_OVERLAP"
        else:
            session = "NY_CLOSE"

        return True, session, f"active (hour={utc_hour} UTC)"
