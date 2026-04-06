"""
Chop Filter — Phase 2
========================
Detects sideways/choppy market conditions that destroy OFI-based strategies.

Chop = price moves a lot intrabar but goes nowhere directionally.
Classic example: alternating up/down 5m candles with no net displacement.

Three signals used (all must agree to call it choppy):
  1. ATR ratio:    current ATR / avg ATR over N bars < CHOP_ATR_RATIO
     → if current volatility is below historical average, market is sleeping
  2. Price range:  (high - low over window) / (sum of abs bar moves) < CHOP_EFFICIENCY
     → if bars are noisy but price hasn't moved net, it's a chop zone
  3. Close clustering: std(close, N) / close[-1] < CHOP_STD_THRESHOLD
     → if closes are tightly clustered, no real trend is present

Any 2 of 3 signals fires CHOP_DETECTED.
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple

logger = logging.getLogger("ChopFilter")

# Tuning constants
CHOP_WINDOW       = 14   # bars for rolling calculations
CHOP_ATR_RATIO    = 0.75  # current ATR < 75% of rolling avg → potentially choppy
CHOP_EFFICIENCY   = 0.30  # net price displacement / sum(|bar moves|) < 30% → choppy
CHOP_STD_THRESHOLD = 0.003  # std(close) / close < 0.3% → tightly clustered


class ChopFilter:
    """
    Stateless chop detection filter.

    Usage:
        chop = ChopFilter()
        is_choppy, reason = chop.is_choppy(df)
    """

    def is_choppy(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Returns (is_choppy: bool, reason: str).

        df must have columns: open, high, low, close, volume
        Requires at least CHOP_WINDOW + 5 rows.
        """
        if df is None or len(df) < CHOP_WINDOW + 5:
            return False, "insufficient_data"

        try:
            close  = df['close']
            high   = df['high']
            low    = df['low']

            signals_fired = 0
            reasons       = []

            # ── Signal 1: ATR ratio ──────────────────────────────────────────
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low  - close.shift()).abs(),
            ], axis=1).max(axis=1)
            atr_current = tr.iloc[-CHOP_WINDOW:].mean()
            atr_avg     = tr.mean()

            if atr_avg > 0 and (atr_current / atr_avg) < CHOP_ATR_RATIO:
                signals_fired += 1
                reasons.append(
                    f"atr_ratio={atr_current/atr_avg:.2f}<{CHOP_ATR_RATIO}"
                )

            # ── Signal 2: Price displacement efficiency ──────────────────────
            window_slice    = df.iloc[-CHOP_WINDOW:]
            net_displacement = abs(window_slice['close'].iloc[-1] - window_slice['close'].iloc[0])
            bar_moves        = window_slice['close'].diff().abs().sum()

            if bar_moves > 0:
                efficiency = net_displacement / bar_moves
                if efficiency < CHOP_EFFICIENCY:
                    signals_fired += 1
                    reasons.append(
                        f"efficiency={efficiency:.2f}<{CHOP_EFFICIENCY}"
                    )

            # ── Signal 3: Close clustering ───────────────────────────────────
            recent_close = close.iloc[-CHOP_WINDOW:]
            std_ratio    = recent_close.std() / (recent_close.mean() + 1e-9)

            if std_ratio < CHOP_STD_THRESHOLD:
                signals_fired += 1
                reasons.append(
                    f"std_ratio={std_ratio*100:.3f}%<{CHOP_STD_THRESHOLD*100:.1f}%"
                )

            # ── Decision: 2 of 3 signals = chop ─────────────────────────────
            if signals_fired >= 2:
                reason_str = " | ".join(reasons)
                logger.debug(f"CHOP DETECTED ({signals_fired}/3): {reason_str}")
                return True, reason_str

            return False, "clean"

        except Exception as e:
            logger.error(f"ChopFilter error: {e}")
            return False, "error"  # fail open — don't block on errors
