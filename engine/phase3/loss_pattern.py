"""
Loss Pattern Detector — Phase 3
==================================
Identifies repeating conditions present when losses occur.
Used to build a blocklist of market states that statistically cause losses.

How it works:
  1. After each losing trade, log the market conditions (regime, session, indicators)
  2. Maintain a frequency map of losing patterns
  3. Before a new trade, check if current conditions match a high-loss pattern
  4. If pattern frequency exceeds threshold → veto the trade

Patterns tracked:
  - Regime + session combination
  - ADX range bucket + OFI direction
  - Volatility band (low/med/high) + trend direction
  - Hour-of-day buckets (UTC)

This is NOT curve-fitting — it's empirical blacklisting based on repeated evidence.
Patterns expire after N trades so the system adapts as market structure changes.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("LossPatternDetector")

# Pattern veto thresholds
MIN_OCCURRENCES_TO_TRACK = 3      # need 3+ losses in a pattern before flagging
LOSS_RATE_VETO_THRESHOLD = 0.75   # if pattern has >=75% loss rate → veto
PATTERN_EXPIRY_TRADES    = 50     # patterns age out after 50 total trades

# Bucket definitions for discretizing continuous values
ADX_BUCKETS   = [(0, 20, "weak"), (20, 28, "moderate"), (28, 999, "strong")]
VOL_BUCKETS   = [(0, 0.005, "low"), (0.005, 0.012, "med"), (0.012, 999, "high")]
HOUR_BUCKETS  = [(0, 6, "asia"), (6, 12, "london"), (12, 18, "ny"), (18, 24, "off")]


class LossPatternDetector:
    """
    Usage:
        detector = LossPatternDetector()
        # After each trade close:
        detector.record_trade(conditions, won=False)
        # Before a new trade:
        vetoed, reason = detector.check(current_conditions)
    """

    def __init__(self):
        # pattern_key → {'wins': int, 'losses': int, 'last_seen': int}
        self._patterns: Dict[str, Dict] = defaultdict(lambda: {'wins': 0, 'losses': 0, 'last_seen': 0})
        self._trade_counter = 0
        self._veto_log: List[str] = []   # last 20 vetoed patterns for diagnostics

    # ── Public API ─────────────────────────────────────────────────────────────

    def record_trade(self, conditions: Dict, won: bool) -> None:
        """
        Record a completed trade with its market conditions.

        conditions should include:
          regime, session, adx, atr_pct, ofi_direction, hour_utc, signal_direction
        """
        self._trade_counter += 1
        keys = self._extract_pattern_keys(conditions)

        for key in keys:
            entry = self._patterns[key]
            if won:
                entry['wins']   += 1
            else:
                entry['losses'] += 1
            entry['last_seen'] = self._trade_counter

        # Purge expired patterns
        if self._trade_counter % 20 == 0:
            self._purge_expired()

        if not won:
            logger.debug(f"LossPattern: recorded loss patterns: {keys[:3]}")

    def check(self, conditions: Dict) -> Tuple[bool, str]:
        """
        Check if current conditions match a known high-loss pattern.

        Returns:
            vetoed  bool   — True if this trade setup should be skipped
            reason  str    — description of matched pattern
        """
        keys = self._extract_pattern_keys(conditions)

        worst_loss_rate  = 0.0
        worst_key        = ""
        worst_count      = 0

        for key in keys:
            if key not in self._patterns:
                continue
            entry  = self._patterns[key]
            total  = entry['wins'] + entry['losses']
            if total < MIN_OCCURRENCES_TO_TRACK:
                continue
            loss_rate = entry['losses'] / total
            if loss_rate > worst_loss_rate:
                worst_loss_rate = loss_rate
                worst_key       = key
                worst_count     = total

        if worst_loss_rate >= LOSS_RATE_VETO_THRESHOLD:
            reason = (
                f"LossPattern veto: '{worst_key}' "
                f"loss_rate={worst_loss_rate:.1%} "
                f"over {worst_count} trades"
            )
            logger.warning(reason)
            self._veto_log.append(reason)
            if len(self._veto_log) > 20:
                self._veto_log.pop(0)
            return True, reason

        return False, ""

    def get_pattern_summary(self) -> List[Dict]:
        """Returns top 10 loss patterns for dashboard/logging."""
        results = []
        for key, entry in self._patterns.items():
            total = entry['wins'] + entry['losses']
            if total < MIN_OCCURRENCES_TO_TRACK:
                continue
            loss_rate = entry['losses'] / total
            results.append({
                'pattern':   key,
                'loss_rate': round(loss_rate, 3),
                'total':     total,
                'losses':    entry['losses'],
                'wins':      entry['wins'],
            })
        return sorted(results, key=lambda x: x['loss_rate'], reverse=True)[:10]

    def get_veto_log(self) -> List[str]:
        return list(self._veto_log)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _extract_pattern_keys(self, conditions: Dict) -> List[str]:
        """
        Convert a conditions dict to a set of pattern keys.
        Each key represents one combination of discretized conditions.
        We generate multiple keys (single-factor + combinations) to catch
        both isolated and compound patterns.
        """
        keys = []

        regime    = conditions.get('regime', 'UNKNOWN')
        session   = conditions.get('session', 'UNKNOWN')
        adx       = conditions.get('adx', 0.0)
        atr_pct   = conditions.get('atr_pct', 0.0)
        ofi_dir   = "pos" if conditions.get('ofi', 0) > 0 else "neg"
        sig_dir   = conditions.get('signal_direction', 'UNKNOWN')
        hour      = conditions.get('hour_utc', 0)

        adx_bucket  = self._bucket(adx, ADX_BUCKETS)
        vol_bucket  = self._bucket(atr_pct, VOL_BUCKETS)
        hour_bucket = self._bucket(hour, HOUR_BUCKETS)

        # Single-factor keys
        keys.append(f"regime:{regime}")
        keys.append(f"session:{session}")
        keys.append(f"adx:{adx_bucket}")
        keys.append(f"vol:{vol_bucket}")
        keys.append(f"hour:{hour_bucket}")

        # Two-factor combinations (most useful)
        keys.append(f"regime:{regime}|session:{session}")
        keys.append(f"regime:{regime}|adx:{adx_bucket}")
        keys.append(f"vol:{vol_bucket}|sig:{sig_dir}")
        keys.append(f"hour:{hour_bucket}|regime:{regime}")
        keys.append(f"adx:{adx_bucket}|ofi:{ofi_dir}")

        # Three-factor (more specific, needs more data)
        keys.append(f"regime:{regime}|vol:{vol_bucket}|sig:{sig_dir}")
        keys.append(f"session:{session}|adx:{adx_bucket}|ofi:{ofi_dir}")

        return keys

    @staticmethod
    def _bucket(value: float, buckets: List[Tuple]) -> str:
        for low, high, label in buckets:
            if low <= value < high:
                return label
        return "unknown"

    def _purge_expired(self) -> None:
        """Remove patterns not seen in the last PATTERN_EXPIRY_TRADES trades."""
        cutoff = self._trade_counter - PATTERN_EXPIRY_TRADES
        to_delete = [
            k for k, v in self._patterns.items()
            if v['last_seen'] < cutoff
        ]
        for k in to_delete:
            del self._patterns[k]
        if to_delete:
            logger.debug(f"LossPattern: purged {len(to_delete)} expired patterns")
