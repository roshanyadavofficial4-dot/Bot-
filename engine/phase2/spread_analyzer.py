"""
Spread Analyzer — Phase 2
===========================
Detects abnormal spread conditions. Extends (does NOT replace) the
existing spread_spike filter in strategy.py.

The strategy.py _is_spread_spike() checks: current > 1.5× rolling average.
This module adds:
  1. Absolute spread limit (in bps) per symbol tier
  2. Spread trend: is the spread widening vs narrowing over last N obs?
  3. Spread quality score: 0–10, used by orchestrator for optional gating

Symbol tiers (by typical liquidity):
  TIER_A  BTCUSDT, ETHUSDT  → max 3bps
  TIER_B  XRPUSDT, DOGEUSDT, ADAUSDT → max 8bps  (our primary symbols)
  TIER_C  everything else → max 15bps

The orchestrator uses SpreadAnalyzer.analyze() to get a quality score
and optionally block trades when spread quality is POOR.
"""

import logging
from collections import deque
from typing import Dict, Tuple

from config import MAX_SPREAD_PCT

logger = logging.getLogger("SpreadAnalyzer")

# Absolute spread limits by symbol tier (in bps = 0.01%)
SPREAD_LIMITS_BPS = {
    'TIER_A': 3.0,    # BTC, ETH
    'TIER_B': 8.0,    # XRP, DOGE, ADA — our targets
    'TIER_C': 15.0,   # other
}

TIER_A_SYMBOLS = {'BTC/USDT:USDT', 'ETH/USDT:USDT'}
TIER_B_SYMBOLS = {'XRP/USDT:USDT', 'DOGE/USDT:USDT', 'ADA/USDT:USDT'}

SPREAD_HISTORY_LEN = 20   # rolling window for trend detection


class SpreadAnalyzer:
    """
    Per-symbol spread tracking and quality scoring.

    One instance per bot run (holds rolling spread history per symbol).
    """

    def __init__(self):
        self._history: Dict[str, deque] = {}

    def _get_tier(self, symbol: str) -> str:
        if symbol in TIER_A_SYMBOLS:
            return 'TIER_A'
        if symbol in TIER_B_SYMBOLS:
            return 'TIER_B'
        return 'TIER_C'

    def analyze(
        self,
        symbol: str,
        spread_pct: float,
    ) -> Tuple[bool, float, str]:
        """
        Returns (trade_allowed: bool, quality_score: float, reason: str).

        quality_score: 0.0 (terrible) – 10.0 (perfect)
        trade_allowed: False if spread is above tier limit OR widening sharply
        """
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=SPREAD_HISTORY_LEN)

        hist = self._history[symbol]
        hist.append(spread_pct)

        spread_bps = spread_pct * 10_000
        tier       = self._get_tier(symbol)
        max_bps    = SPREAD_LIMITS_BPS[tier]

        # ── Absolute limit ────────────────────────────────────────────────────
        if spread_bps > max_bps:
            reason = (
                f"spread={spread_bps:.1f}bps > tier_limit={max_bps:.1f}bps ({tier})"
            )
            logger.debug(f"SpreadAnalyzer BLOCK [{symbol}]: {reason}")
            return False, 0.0, reason

        # ── Also enforce the config MAX_SPREAD_PCT ────────────────────────────
        if spread_pct > MAX_SPREAD_PCT:
            reason = f"spread={spread_bps:.1f}bps > config_max={MAX_SPREAD_PCT*10000:.1f}bps"
            logger.debug(f"SpreadAnalyzer BLOCK [{symbol}]: {reason}")
            return False, 0.0, reason

        # ── Spread trend: is it widening? ─────────────────────────────────────
        widening = False
        if len(hist) >= 5:
            recent_avg = sum(list(hist)[-5:]) / 5
            older_avg  = sum(list(hist)[:-5]) / max(len(hist) - 5, 1)
            if older_avg > 0 and recent_avg > older_avg * 1.5:
                widening = True
                logger.debug(
                    f"SpreadAnalyzer [{symbol}]: spread widening "
                    f"recent={recent_avg*10000:.1f}bps older={older_avg*10000:.1f}bps"
                )

        if widening:
            return False, 2.0, f"spread_widening (recent={recent_avg*10000:.1f}bps)"

        # ── Quality score ─────────────────────────────────────────────────────
        # 10.0 at 0 spread → 0.0 at max_bps
        quality = max(0.0, 10.0 * (1.0 - spread_bps / max_bps))
        quality = round(quality, 1)

        return True, quality, f"ok ({spread_bps:.1f}bps tier={tier} score={quality})"
