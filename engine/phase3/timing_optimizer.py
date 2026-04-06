"""
Trade Timing Optimizer — Phase 3
===================================
Improves entry timing by detecting micro-pullbacks within a valid trend.

Problem: Entering on a breakout candle often means chasing price.
Solution: Wait for a 1-2 candle micro-pullback toward a key level,
          then enter when momentum resumes.

Strategies:
  1. Micro-pullback wait: RSI > 70 or extreme candle → hold for 1 candle
  2. Momentum confirmation: require a follow-through candle before entry
  3. Spread window: prefer entry when spread is tightest (typically
     after a high-volume flush, not mid-momentum)
  4. Time-of-day quality: weight entry quality by historical intraday
     performance windows

Outputs:
  - timing_quality: 0.0 (worst) → 1.0 (ideal entry timing)
  - should_wait:    True if better entry expected within 1–2 candles
  - wait_reason:    str
  - urgency:        'immediate' | 'wait_1' | 'wait_2' | 'skip'
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple

logger = logging.getLogger("TradeTimingOptimizer")

# Quality threshold — below this, suggest waiting
TIMING_QUALITY_WAIT = 0.40

# RSI extreme levels (candle-level chasing indicators)
RSI_OVERBOUGHT  = 70
RSI_OVERSOLD    = 30
RSI_EXTENDED    = 65   # elevated but not extreme → caution zone

# Momentum extension: if current candle body > N× avg body → chasing
MOMENTUM_EXTENSION_MULT = 2.5


class TradeTimingOptimizer:
    """
    Usage:
        optimizer = TradeTimingOptimizer()
        quality, meta = optimizer.evaluate(df, candidate, signal_direction)
        if meta['should_wait']:
            # defer entry
    """

    def evaluate(
        self,
        df: pd.DataFrame,
        candidate: Dict,
        signal_direction: str,   # 'BUY' | 'SELL'
    ) -> Tuple[float, Dict]:
        """
        Returns:
            timing_quality  float   0.0–1.0
            meta            dict
        """
        if df is None or len(df) < 10:
            return 0.7, {'should_wait': False, 'urgency': 'immediate', 'reason': 'insufficient_data'}

        scores  = {}
        reasons = []

        # ── 1. Momentum extension check ───────────────────────────────────────
        momentum_ok, mom_reason = self._check_momentum_extension(df, signal_direction)
        scores['momentum'] = 1.0 if momentum_ok else 0.2
        if not momentum_ok:
            reasons.append(mom_reason)

        # ── 2. RSI positioning ────────────────────────────────────────────────
        rsi_score, rsi_reason = self._check_rsi(candidate, signal_direction)
        scores['rsi'] = rsi_score
        if rsi_score < 0.5:
            reasons.append(rsi_reason)

        # ── 3. Micro-pullback quality ─────────────────────────────────────────
        pullback_score = self._check_pullback(df, signal_direction)
        scores['pullback'] = pullback_score

        # ── 4. Spread timing ──────────────────────────────────────────────────
        spread_score = self._check_spread_timing(candidate)
        scores['spread'] = spread_score
        if spread_score < 0.5:
            reasons.append("spread_elevated_for_entry")

        # ── 5. Candle close proximity ─────────────────────────────────────────
        # Prefer entering near candle open of a new candle, not mid-candle
        close_score = self._check_candle_position(df)
        scores['candle_pos'] = close_score

        # ── Weighted quality ──────────────────────────────────────────────────
        weights = {
            'momentum': 0.35,
            'rsi':      0.25,
            'pullback': 0.20,
            'spread':   0.10,
            'candle_pos': 0.10,
        }
        quality = sum(scores[k] * weights[k] for k in weights)
        quality = round(max(0.0, min(1.0, quality)), 4)

        should_wait = quality < TIMING_QUALITY_WAIT
        urgency = self._determine_urgency(quality, scores)

        if should_wait:
            logger.debug(
                f"TimingOptimizer: quality={quality:.3f} → wait | "
                f"reasons={reasons}"
            )

        return quality, {
            'timing_quality': quality,
            'should_wait':    should_wait,
            'urgency':        urgency,
            'wait_reason':    " | ".join(reasons) if reasons else "none",
            'component_scores': {k: round(v, 3) for k, v in scores.items()},
        }

    # ── Component checks ────────────────────────────────────────────────────

    def _check_momentum_extension(
        self, df: pd.DataFrame, direction: str
    ) -> Tuple[bool, str]:
        """
        True if current candle body is NOT extended (i.e., not chasing).
        Extended = current candle body > 2.5× avg body of last 10 candles.
        """
        recent    = df.tail(11)
        bodies    = abs(recent['close'] - recent['open'])
        avg_body  = bodies.iloc[:-1].mean()
        curr_body = bodies.iloc[-1]

        if avg_body < 1e-9:
            return True, ""

        ratio = curr_body / avg_body
        if ratio > MOMENTUM_EXTENSION_MULT:
            return False, f"momentum_extended({ratio:.1f}x_avg)"
        return True, ""

    def _check_rsi(
        self, candidate: Dict, direction: str
    ) -> Tuple[float, str]:
        """Score RSI positioning. Returns (score, reason)."""
        rsi = candidate.get('rsi', 50.0)

        if direction == 'BUY':
            if rsi >= RSI_OVERBOUGHT:
                return 0.10, f"RSI_overbought({rsi:.0f})"
            elif rsi >= RSI_EXTENDED:
                return 0.45, f"RSI_elevated({rsi:.0f})"
            elif 45 <= rsi < 65:
                return 1.0, ""   # ideal zone
            else:
                return 0.70, ""  # low RSI is fine for BUY
        else:  # SELL
            if rsi <= RSI_OVERSOLD:
                return 0.10, f"RSI_oversold({rsi:.0f})"
            elif rsi <= 35:
                return 0.45, f"RSI_depressed({rsi:.0f})"
            elif 35 < rsi <= 55:
                return 1.0, ""
            else:
                return 0.70, ""

    def _check_pullback(self, df: pd.DataFrame, direction: str) -> float:
        """
        Score 1.0 if last 1–2 candles show a shallow pullback (better entry).
        Score 0.5 if price is continuing momentum (OK but slightly chasing).
        Score 0.2 if price has pulled back too far (signal may be weakening).
        """
        if len(df) < 4:
            return 0.5

        c = df['close'].values
        recent_3 = c[-3:]

        if direction == 'BUY':
            # Ideal: c[-2] < c[-3] (pullback) then c[-1] > c[-2] (resuming)
            had_pullback = recent_3[1] < recent_3[0]
            resuming     = recent_3[2] > recent_3[1]
            if had_pullback and resuming:
                return 1.0
            elif not had_pullback:
                return 0.5   # straight up — slight chase
            else:
                return 0.3   # still pulling back

        else:  # SELL
            had_pullback = recent_3[1] > recent_3[0]
            resuming     = recent_3[2] < recent_3[1]
            if had_pullback and resuming:
                return 1.0
            elif not had_pullback:
                return 0.5
            else:
                return 0.3

    def _check_spread_timing(self, candidate: Dict) -> float:
        """Lower spread at entry = better timing."""
        from config import MAX_SPREAD_PCT
        spread = candidate.get('spread_pct', 0.0)
        if spread <= MAX_SPREAD_PCT * 0.5:
            return 1.0
        elif spread <= MAX_SPREAD_PCT * 0.8:
            return 0.65
        elif spread <= MAX_SPREAD_PCT:
            return 0.40
        else:
            return 0.10

    def _check_candle_position(self, df: pd.DataFrame) -> float:
        """
        Score based on where current close is within the current candle range.
        BUY near low of candle range = good. BUY at top = chasing.
        (Approximation using last closed candle position)
        """
        if len(df) < 2:
            return 0.5
        last = df.iloc[-1]
        rng  = last['high'] - last['low']
        if rng < 1e-9:
            return 0.5
        close_pos = (last['close'] - last['low']) / rng
        # For a long trade, lower close within candle range = better (entry near low)
        # Invert for scoring: 0.0 (top of candle) → 0.2, 0.5 (mid) → 0.7, 1.0 (bottom) → 1.0
        # We don't know direction here so score neutrally (middle is best — confirmed close)
        score = 1.0 - abs(close_pos - 0.5) * 0.6
        return round(score, 3)

    def _determine_urgency(self, quality: float, scores: Dict) -> str:
        if quality >= 0.70:
            return 'immediate'
        elif quality >= 0.50:
            return 'wait_1'    # wait 1 candle for better setup
        elif quality >= 0.35:
            return 'wait_2'    # wait up to 2 candles
        else:
            return 'skip'
