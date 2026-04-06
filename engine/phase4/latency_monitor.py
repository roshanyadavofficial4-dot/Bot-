"""
Latency Monitor — Phase 4 (Upgrade)
======================================
Monitors execution latency and API response times.
Adjusts trading behavior when delays are detected.

Tracked metrics:
  - API response time (scanner → exchange)
  - Order placement latency
  - Fill confirmation latency
  - Total round-trip time (signal → fill)

Latency impact on trading:
  - HIGH latency → widen slippage buffer, skip fast signals
  - EXTREME latency → halt new entries (can't execute cleanly)
  - PERSISTENT latency → trigger alert via Telegram

Integration:
  - Called by executor.py to log each order round-trip
  - Called by scanner.py to log API call timing
  - Orchestrator checks get_trading_permission() before each trade
"""

import logging
import time
from collections import deque
from typing import Dict, Optional, Tuple

logger = logging.getLogger("LatencyMonitor")

# Latency thresholds (seconds)
LATENCY_NORMAL    = 0.5    # < 500ms = normal
LATENCY_ELEVATED  = 1.0    # 500ms–1s = elevated, adjust slippage
LATENCY__HIGH     = 2.0    # 1–2s = high, skip fast signals
LATENCY_EXTREME   = 5.0    # >5s = extreme, block new entries

# Rolling window for latency averaging
WINDOW_SIZE = 20

# Slippage adjustments per latency tier (multiplier on base SLIPPAGE_BUFFER)
SLIPPAGE_MULT = {
    'NORMAL':   1.0,
    'ELEVATED': 1.5,
    'HIGH':     2.5,
    'EXTREME':  5.0,   # effectively blocks trades (cost too high)
}


class LatencyMonitor:
    """
    Usage:
        monitor = LatencyMonitor()

        # Around each API call:
        token = monitor.start_timer('api_scan')
        result = exchange.fetch_ohlcv(...)
        monitor.stop_timer(token)

        # Before each trade:
        allowed, meta = monitor.get_trading_permission()
    """

    def __init__(self):
        self._api_latencies:   deque = deque(maxlen=WINDOW_SIZE)
        self._order_latencies: deque = deque(maxlen=WINDOW_SIZE)
        self._fill_latencies:  deque = deque(maxlen=WINDOW_SIZE)
        self._active_timers:   Dict[str, float] = {}
        self._tier            = 'NORMAL'
        self._last_update     = time.time()
        self._alerts_sent     = 0

    # ── Timing API ─────────────────────────────────────────────────────────────

    def start_timer(self, label: str) -> str:
        """Start a latency measurement. Returns token for stop_timer."""
        token = f"{label}_{time.time()}"
        self._active_timers[token] = time.time()
        return token

    def stop_timer(self, token: str, category: str = 'api') -> Optional[float]:
        """
        Stop a timer and record the measurement.

        category: 'api' | 'order' | 'fill'
        Returns elapsed seconds.
        """
        start = self._active_timers.pop(token, None)
        if start is None:
            return None

        elapsed = time.time() - start

        if category == 'api':
            self._api_latencies.append(elapsed)
        elif category == 'order':
            self._order_latencies.append(elapsed)
        elif category == 'fill':
            self._fill_latencies.append(elapsed)

        # Update tier on new data
        self._update_tier()

        if elapsed > LATENCY_EXTREME:
            logger.warning(f"LatencyMonitor: EXTREME {category} latency: {elapsed:.2f}s")

        return round(elapsed, 4)

    def record_roundtrip(self, total_seconds: float) -> None:
        """Convenience: record a full signal→fill round-trip time."""
        self._order_latencies.append(total_seconds)
        self._update_tier()
        logger.debug(f"LatencyMonitor: round-trip {total_seconds*1000:.0f}ms | tier={self._tier}")

    # ── Trading permission ─────────────────────────────────────────────────────

    def get_trading_permission(self) -> Tuple[bool, Dict]:
        """
        Check if latency conditions allow trading.

        Returns:
            allowed        bool
            meta           dict
        """
        self._update_tier()
        meta = self.get_summary()
        meta['slippage_multiplier'] = SLIPPAGE_MULT[self._tier]

        if self._tier == 'EXTREME':
            logger.warning(
                f"LatencyMonitor: EXTREME latency → blocking new entries | "
                f"avg_api={meta['avg_api_ms']:.0f}ms"
            )
            return False, meta

        if self._tier == 'HIGH':
            logger.debug(
                f"LatencyMonitor: HIGH latency — skipping fast signal types | "
                f"avg={meta['avg_api_ms']:.0f}ms"
            )
            meta['skip_fast_signals'] = True
            return True, meta

        return True, meta

    def get_summary(self) -> Dict:
        """Current latency statistics."""
        avg_api   = self._avg(self._api_latencies)
        avg_order = self._avg(self._order_latencies)
        avg_fill  = self._avg(self._fill_latencies)
        p95_api   = self._p95(self._api_latencies)

        return {
            'tier':              self._tier,
            'avg_api_ms':        round(avg_api * 1000, 1),
            'avg_order_ms':      round(avg_order * 1000, 1),
            'avg_fill_ms':       round(avg_fill * 1000, 1),
            'p95_api_ms':        round(p95_api * 1000, 1),
            'slippage_multiplier': SLIPPAGE_MULT[self._tier],
            'samples':           len(self._api_latencies),
            'skip_fast_signals': self._tier == 'HIGH',
        }

    def get_adjusted_slippage(self, base_slippage: float) -> float:
        """Return slippage buffer adjusted for current latency conditions."""
        return round(base_slippage * SLIPPAGE_MULT[self._tier], 6)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _update_tier(self) -> None:
        """Recompute latency tier from rolling average."""
        avg = self._avg(self._api_latencies)

        if avg >= LATENCY_EXTREME:
            new_tier = 'EXTREME'
        elif avg >= LATENCY_HIGH:
            new_tier = 'HIGH'
        elif avg >= LATENCY_ELEVATED:
            new_tier = 'ELEVATED'
        else:
            new_tier = 'NORMAL'

        if new_tier != self._tier:
            logger.info(
                f"LatencyMonitor: tier {self._tier} → {new_tier} "
                f"(avg={avg*1000:.0f}ms)"
            )
            self._tier = new_tier

    @staticmethod
    def _avg(data: deque) -> float:
        if not data:
            return 0.0
        return sum(data) / len(data)

    @staticmethod
    def _p95(data: deque) -> float:
        if len(data) < 5:
            return 0.0
        import numpy as np
        return float(np.percentile(list(data), 95))
