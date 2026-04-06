"""
Market Efficiency Engine — Phase 3
=====================================
Detects whether the current market is "clean" (tradeable) or "noisy" (random).

A clean market has:
  - Price moving with trending structure (higher highs, lower lows)
  - Volume confirming price moves
  - Low randomness in candle pattern
  - OFI and CVD agreeing

A noisy market has:
  - Choppy, back-and-forth price action
  - High proportion of doji / spinning top candles
  - OFI and CVD contradicting each other
  - ADX falling while price oscillates

Outputs:
  - efficiency_score: 0.0 (pure noise) → 1.0 (clean trend)
  - is_tradeable: bool — True if score >= threshold
  - market_type: 'clean_trend' | 'noisy' | 'consolidating' | 'chaotic'

Used by: AdaptiveParamEngine, Orchestrator (pre-trade gate)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple

logger = logging.getLogger("MarketEfficiencyEngine")

# Score threshold to permit trading
TRADEABLE_THRESHOLD = 0.40   # below → skip (noisy/chaotic market)

# Weights for efficiency components (sum = 1.0)
WEIGHTS = {
    'directional_consistency': 0.30,
    'candle_quality':          0.25,
    'volume_confirmation':     0.20,
    'indicator_agreement':     0.15,
    'price_range_efficiency':  0.10,
}


class MarketEfficiencyEngine:
    """
    Stateless per-call scoring.

    Usage:
        engine = MarketEfficiencyEngine()
        score, meta = engine.evaluate(df, candidate)
        if not meta['is_tradeable']:
            skip_trade()
    """

    def evaluate(
        self,
        df: pd.DataFrame,
        candidate: Dict,
    ) -> Tuple[float, Dict]:
        """
        Evaluate market efficiency from recent OHLCV data + scanner candidate.

        df:        OHLCV DataFrame (5m), recent ~30 candles
        candidate: scanner candidate dict with adx, ofi, cvd, vol_ratio, etc.

        Returns:
            efficiency_score  float  0.0–1.0
            meta              dict
        """
        if df is None or len(df) < 20:
            return 0.5, {'is_tradeable': True, 'market_type': 'unknown', 'reason': 'insufficient_data'}

        scores = {}

        # ── 1. Directional consistency (30%) ──────────────────────────────────
        scores['directional_consistency'] = self._directional_consistency(df)

        # ── 2. Candle quality (25%) ───────────────────────────────────────────
        scores['candle_quality'] = self._candle_quality(df)

        # ── 3. Volume confirmation (20%) ──────────────────────────────────────
        scores['volume_confirmation'] = self._volume_confirmation(df, candidate)

        # ── 4. Indicator agreement (15%) ──────────────────────────────────────
        scores['indicator_agreement'] = self._indicator_agreement(candidate)

        # ── 5. Price range efficiency (10%) ───────────────────────────────────
        scores['price_range_efficiency'] = self._price_range_efficiency(df)

        # ── Weighted total ────────────────────────────────────────────────────
        total = sum(scores[k] * WEIGHTS[k] for k in scores)
        total = round(max(0.0, min(1.0, total)), 4)

        market_type = self._classify(total, candidate)
        is_tradeable = total >= TRADEABLE_THRESHOLD

        if not is_tradeable:
            logger.debug(
                f"MarketEfficiency: score={total:.3f} → NOT tradeable "
                f"({market_type}) | {scores}"
            )

        return total, {
            'efficiency_score': total,
            'is_tradeable':     is_tradeable,
            'market_type':      market_type,
            'component_scores': {k: round(v, 3) for k, v in scores.items()},
            'threshold':        TRADEABLE_THRESHOLD,
        }

    # ── Component scorers (each returns 0.0–1.0) ────────────────────────────

    def _directional_consistency(self, df: pd.DataFrame) -> float:
        """Fraction of candles moving in dominant direction over last 15 bars."""
        closes  = df['close'].values[-15:]
        if len(closes) < 5:
            return 0.5

        diffs   = np.diff(closes)
        up_bars = np.sum(diffs > 0)
        dn_bars = np.sum(diffs < 0)
        total   = len(diffs)

        # Score based on dominance — 50/50 = 0.0, 100% one direction = 1.0
        dominance = abs(up_bars - dn_bars) / max(total, 1)
        return float(dominance)

    def _candle_quality(self, df: pd.DataFrame) -> float:
        """
        Penalises doji/spinning top candles (body < 20% of range).
        High-quality candles have a clear body direction.
        """
        recent = df.tail(15)
        if len(recent) < 5:
            return 0.5

        body_ratios = abs(recent['close'] - recent['open']) / (
            (recent['high'] - recent['low']).replace(0, np.nan)
        )
        body_ratios = body_ratios.dropna()

        if len(body_ratios) == 0:
            return 0.5

        # Fraction of candles with body > 30% of range
        strong_candles = (body_ratios > 0.30).sum() / len(body_ratios)
        return float(strong_candles)

    def _volume_confirmation(self, df: pd.DataFrame, candidate: Dict) -> float:
        """Volume should rise when price makes larger moves."""
        recent = df.tail(10)
        if len(recent) < 5:
            return 0.5

        vol_ratio = candidate.get('vol_ratio', 1.0)

        # Vol ratio > 1.5 in a trending market is confirmation
        if vol_ratio >= 2.0:
            return 1.0
        elif vol_ratio >= 1.5:
            return 0.75
        elif vol_ratio >= 1.2:
            return 0.50
        else:
            return 0.20

    def _indicator_agreement(self, candidate: Dict) -> float:
        """
        OFI and CVD should agree directionally.
        ADX should be trending (>=20).
        """
        ofi  = candidate.get('ofi', 0.0)
        cvd  = candidate.get('cvd', 0.0)
        adx  = candidate.get('adx', 0.0)

        score = 0.0

        # OFI and CVD direction agreement
        if (ofi > 0 and cvd > 0) or (ofi < 0 and cvd < 0):
            score += 0.60   # both aligned
        elif ofi == 0 or cvd == 0:
            score += 0.30   # one neutral
        else:
            score += 0.00   # contradicting

        # ADX trending
        if adx >= 28:
            score += 0.40
        elif adx >= 22:
            score += 0.25
        elif adx >= 18:
            score += 0.10
        else:
            score += 0.00

        return min(1.0, score)

    def _price_range_efficiency(self, df: pd.DataFrame) -> float:
        """
        Measure how 'efficiently' price has moved.
        Efficient market: net displacement / total path length is high.
        Choppy market: lots of back and forth, ratio near 0.
        """
        closes = df['close'].values[-20:]
        if len(closes) < 5:
            return 0.5

        net_move   = abs(closes[-1] - closes[0])
        total_path = np.sum(np.abs(np.diff(closes)))

        if total_path < 1e-9:
            return 0.5

        efficiency = net_move / total_path
        return float(min(1.0, efficiency))

    def _classify(self, score: float, candidate: Dict) -> str:
        adx = candidate.get('adx', 0.0)

        if score >= 0.65 and adx >= 25:
            return 'clean_trend'
        elif score >= 0.45:
            return 'consolidating'
        elif score >= 0.25:
            return 'noisy'
        else:
            return 'chaotic'
