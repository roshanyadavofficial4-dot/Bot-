"""
Regime Detector — v1.0
=======================
AUDIT: Original bot had NO proper regime detection.
ADX check was a single number — not regime classification.

This module provides:
1. Market regime: TRENDING / RANGING / VOLATILE / UNKNOWN
2. Regime confidence score (0-1)
3. Recommended strategy adjustment for each regime

Why this matters for small capital:
- RANGING market + OFI signal = ~50% win rate after fees = NET NEGATIVE
- Only TRENDING + strong OFI = true positive expectancy
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple

logger = logging.getLogger("RegimeDetector")


class RegimeDetector:
    """
    Multi-factor regime classification.

    Inputs: OHLCV DataFrame (5m), ADX value
    Output: (regime_label, confidence, description)
    """

    # ADX thresholds
    ADX_STRONG_TREND  = 30
    ADX_WEAK_TREND    = 22
    ADX_RANGING       = 18

    # Bollinger Band width thresholds (relative to price)
    BB_TIGHT          = 0.004   # < 0.4% = extremely tight = range
    BB_WIDE           = 0.015   # > 1.5% = volatile

    def classify(
        self,
        df: pd.DataFrame,
        adx: float,
        h_volatility: float = 0.0
    ) -> Tuple[str, float, str]:
        """
        Returns (regime, confidence, description)

        Regimes:
            TRENDING   → trend trade, full position size
            RANGING    → skip or very tight scalp (not recommended)
            VOLATILE   → spike/panic mode — skip entirely
            UNKNOWN    → insufficient data
        """
        if df is None or len(df) < 30:
            return 'UNKNOWN', 0.0, 'Insufficient data'

        confidence_signals = []
        regime_votes       = {'TRENDING': 0, 'RANGING': 0, 'VOLATILE': 0}

        # ── ADX vote ──────────────────────────────────────────────────────────
        if adx >= self.ADX_STRONG_TREND:
            regime_votes['TRENDING'] += 2
            confidence_signals.append(f'ADX={adx:.1f} strong')
        elif adx >= self.ADX_WEAK_TREND:
            regime_votes['TRENDING'] += 1
            confidence_signals.append(f'ADX={adx:.1f} weak trend')
        elif adx <= self.ADX_RANGING:
            regime_votes['RANGING'] += 2
            confidence_signals.append(f'ADX={adx:.1f} ranging')
        else:
            regime_votes['RANGING'] += 1

        # ── Bollinger Band width ───────────────────────────────────────────────
        try:
            close    = df['close']
            sma20    = close.rolling(20).mean()
            std20    = close.rolling(20).std()
            bb_width = (std20.iloc[-1] / sma20.iloc[-1])

            if bb_width < self.BB_TIGHT:
                regime_votes['RANGING'] += 2
                confidence_signals.append(f'BB_width={bb_width*100:.2f}% tight')
            elif bb_width > self.BB_WIDE:
                regime_votes['VOLATILE'] += 2
                confidence_signals.append(f'BB_width={bb_width*100:.2f}% wide')
            else:
                regime_votes['TRENDING'] += 1
        except Exception:
            pass

        # ── Price vs EMA ──────────────────────────────────────────────────────
        try:
            ema21  = df['close'].ewm(span=21).mean()
            ema55  = df['close'].ewm(span=55).mean()
            last_c = df['close'].iloc[-1]
            # EMA21 above EMA55 AND price above EMA21 = trending
            if ema21.iloc[-1] > ema55.iloc[-1] and last_c > ema21.iloc[-1]:
                regime_votes['TRENDING'] += 1
                confidence_signals.append('EMA21>EMA55 bullish')
            elif ema21.iloc[-1] < ema55.iloc[-1] and last_c < ema21.iloc[-1]:
                regime_votes['TRENDING'] += 1
                confidence_signals.append('EMA21<EMA55 bearish')
            else:
                regime_votes['RANGING'] += 1
        except Exception:
            pass

        # ── Annualised volatility vote ─────────────────────────────────────────
        if h_volatility > 0.50:   # > 50% annualised vol = panic mode
            regime_votes['VOLATILE'] += 2
            confidence_signals.append(f'h_vol={h_volatility:.2f} high')

        # ── Final classification ───────────────────────────────────────────────
        total_votes   = sum(regime_votes.values()) or 1
        winner        = max(regime_votes, key=regime_votes.get)
        max_votes     = regime_votes[winner]
        confidence    = max_votes / total_votes

        description = f"{winner} ({confidence*100:.0f}%) | " + ", ".join(confidence_signals)
        logger.debug(f"Regime: {description}")

        return winner, round(confidence, 3), description

    def get_regime_multipliers(self, regime: str) -> dict:
        """
        Returns recommended parameter adjustments per regime.
        """
        multipliers = {
            'TRENDING': {
                'position_size': 1.0,
                'tp_multiplier': 1.0,
                'sl_multiplier': 1.0,
                'trade_allowed': True,
            },
            'RANGING': {
                'position_size': 0.0,   # Don't trade ranging markets with OFI
                'tp_multiplier': 0.7,
                'sl_multiplier': 0.8,
                'trade_allowed': False,
            },
            'VOLATILE': {
                'position_size': 0.0,
                'tp_multiplier': 0.0,
                'sl_multiplier': 0.0,
                'trade_allowed': False,
            },
            'UNKNOWN': {
                'position_size': 0.5,
                'tp_multiplier': 0.8,
                'sl_multiplier': 1.0,
                'trade_allowed': False,   # Don't trade without regime clarity
            },
        }
        return multipliers.get(regime, multipliers['UNKNOWN'])
