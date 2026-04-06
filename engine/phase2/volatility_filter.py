"""
Volatility Filter — Phase 2
=============================
Blocks trades when market volatility is outside the tradeable band.

Two failure modes this prevents:
  1. Too LOW volatility  → spread + fee costs eat all edge; dead market
  2. Too HIGH volatility → stop-loss gets hit by noise; slippage explodes

Uses ATR% (ATR as fraction of price) as the primary volatility measure
since it's already computed by the scanner and available in the candidate dict.

Secondary: annualised historical volatility (h_volatility) for regime context.

Thresholds (configurable at top of module):
  ATR_PCT_MIN   = 0.0008   (0.08%)  — below this = dead market, skip
  ATR_PCT_MAX   = 0.015    (1.5%)   — above this = too wild, skip
  H_VOL_MAX     = 0.60     (60% annualised) — extreme panic, skip
"""

import logging
from typing import Tuple

from config import MAX_ATR_VOLATILITY

logger = logging.getLogger("VolatilityFilter")

ATR_PCT_MIN = 0.0008           # 0.08% — minimum meaningful move per bar
ATR_PCT_MAX = MAX_ATR_VOLATILITY  # 1.5% — from config, already validated
H_VOL_MAX   = 0.60             # 60% annualised — panic threshold


class VolatilityFilter:
    """
    Stateless volatility gate.

    Usage:
        vf = VolatilityFilter()
        ok, reason = vf.is_tradeable(atr, last_price, h_volatility)
    """

    def is_tradeable(
        self,
        atr: float,
        last_price: float,
        h_volatility: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        Returns (is_tradeable: bool, reason: str).

        atr:          ATR value in price units
        last_price:   current asset price
        h_volatility: annualised historical volatility (0-1 scale; optional)
        """
        if last_price <= 0:
            return False, "invalid_price"

        atr_pct = atr / last_price

        # ── Too low ──────────────────────────────────────────────────────────
        if atr_pct < ATR_PCT_MIN:
            reason = f"atr_pct={atr_pct*100:.4f}%<{ATR_PCT_MIN*100:.2f}% (dead_market)"
            logger.debug(f"VolatilityFilter BLOCK: {reason}")
            return False, reason

        # ── Too high ─────────────────────────────────────────────────────────
        if atr_pct > ATR_PCT_MAX:
            reason = f"atr_pct={atr_pct*100:.3f}%>{ATR_PCT_MAX*100:.2f}% (too_wild)"
            logger.debug(f"VolatilityFilter BLOCK: {reason}")
            return False, reason

        # ── Panic check via h_volatility ─────────────────────────────────────
        if h_volatility > H_VOL_MAX:
            reason = f"h_vol={h_volatility:.2f}>{H_VOL_MAX:.2f} (panic_mode)"
            logger.debug(f"VolatilityFilter BLOCK: {reason}")
            return False, reason

        return True, f"ok (atr={atr_pct*100:.3f}%)"
