"""
Regime Confidence Engine — Phase 2
=====================================
Thin wrapper around the existing RegimeDetector that:
  1. Normalises confidence to 0.0–1.0
  2. Exposes a single gate method for the orchestrator
  3. Adds a minimum confidence threshold before trading is permitted

The existing RegimeDetector does the heavy lifting (ADX + BB + EMA votes).
This module just adds a clean interface and the minimum confidence gate.

Minimum confidence by regime:
  TRENDING  → 0.45  (a weakly trending market is still tradeable)
  UNKNOWN   → 0.60  (need high confidence before trading uncertain conditions)
  RANGING   → N/A   (never trade ranging — blocked regardless of confidence)
  VOLATILE  → N/A   (never trade volatile — blocked regardless)
"""

import logging
from typing import Tuple

from engine.regime_detector import RegimeDetector

logger = logging.getLogger("RegimeConfidence")

# Minimum confidence required to trade each regime
REGIME_MIN_CONFIDENCE = {
    'TRENDING': 0.45,
    'UNKNOWN':  0.60,
    'RANGING':  1.01,   # effectively infinite — never trade
    'VOLATILE': 1.01,
}


class RegimeConfidenceEngine:
    """
    Facade around RegimeDetector with orchestrator-friendly interface.

    Usage:
        engine = RegimeConfidenceEngine()
        ok, regime, conf, desc = engine.evaluate(df, adx, h_volatility)
    """

    def __init__(self):
        self._detector = RegimeDetector()

    def evaluate(
        self,
        df,
        adx: float,
        h_volatility: float = 0.0,
    ) -> Tuple[bool, str, float, str]:
        """
        Returns:
            trade_allowed  bool   — whether regime + confidence permit a trade
            regime         str    — 'TRENDING' | 'RANGING' | 'VOLATILE' | 'UNKNOWN'
            confidence     float  — 0.0–1.0 from RegimeDetector vote tallying
            description    str    — human-readable regime summary

        The existing RegimeDetector.classify() is called unchanged.
        Only the gate logic is added here.
        """
        regime, confidence, description = self._detector.classify(df, adx, h_volatility)

        min_conf = REGIME_MIN_CONFIDENCE.get(regime, 1.01)
        trade_allowed = confidence >= min_conf

        if not trade_allowed:
            logger.debug(
                f"RegimeConfidence GATE: regime={regime} conf={confidence:.2f} "
                f"min={min_conf:.2f} → BLOCKED"
            )
        else:
            logger.debug(
                f"RegimeConfidence GATE: regime={regime} conf={confidence:.2f} "
                f"min={min_conf:.2f} → ALLOWED"
            )

        return trade_allowed, regime, confidence, description

    def get_multipliers(self, regime: str) -> dict:
        """Passthrough to RegimeDetector.get_regime_multipliers()."""
        return self._detector.get_regime_multipliers(regime)
