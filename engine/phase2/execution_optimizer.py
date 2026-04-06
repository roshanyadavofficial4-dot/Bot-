"""
Execution Optimizer — Phase 2
================================
Decides whether to use a LIMIT (post-only/GTX) or MARKET order for entry.

Logic:
  LIMIT (post-only) is preferred when:
    - spread is tight (< LIMIT_SPREAD_THRESHOLD)
    - market is not in a fast-moving breakout (ADX < BREAKOUT_ADX_THRESHOLD)
    - session has sufficient liquidity (not dead zone)

  MARKET is used when:
    - spread is wide (large fill risk on limit)
    - breakout detected (price moving fast, limit may not fill)
    - dead zone detected (limit orders may sit unfilled)

Also provides:
  - limit price offset calculation (how far inside the spread to place limit)
  - order timeout: how long to wait before cancelling a limit and going market

This module does NOT place orders. It returns a recommendation dict
consumed by executor.py / orchestrator.py.
"""

import logging
from datetime import datetime, timezone
from typing import Dict

from config import MAX_SPREAD_PCT, FEE_RATE

logger = logging.getLogger("ExecutionOptimizer")

# Thresholds
LIMIT_SPREAD_THRESHOLD  = MAX_SPREAD_PCT * 0.6   # only use limit if spread < 60% of max
BREAKOUT_ADX_THRESHOLD  = 28.0                    # if ADX > 28, market is moving fast → market order
LIMIT_OFFSET_BPS        = 0.5                     # place limit 0.5bps inside spread
LIMIT_TIMEOUT_SECONDS   = 8                       # cancel limit after 8s if unfilled
DEAD_ZONE_HOURS         = {20, 21, 22, 23, 0, 1}


class ExecutionOptimizer:
    """
    Usage:
        opt = ExecutionOptimizer()
        rec = opt.recommend(spread_pct, adx, entry_price, side)
        # rec['order_type'] → 'LIMIT' | 'MARKET'
    """

    def recommend(
        self,
        spread_pct: float,
        adx: float,
        entry_price: float,
        side: str,           # 'BUY' | 'SELL'
        utc_hour: int = None,
    ) -> Dict:
        """
        Returns recommendation dict:
            order_type     str    — 'LIMIT' | 'MARKET'
            limit_price    float  — if LIMIT, the suggested price
            timeout_sec    int    — if LIMIT, cancel after this many seconds
            rationale      str    — human-readable reason
        """
        if utc_hour is None:
            utc_hour = datetime.now(timezone.utc).hour

        use_market = False
        reasons    = []

        # ── Spread check ─────────────────────────────────────────────────────
        if spread_pct > LIMIT_SPREAD_THRESHOLD:
            use_market = True
            reasons.append(
                f"wide_spread={spread_pct*10000:.1f}bps>{LIMIT_SPREAD_THRESHOLD*10000:.1f}bps"
            )

        # ── Breakout check ────────────────────────────────────────────────────
        if adx > BREAKOUT_ADX_THRESHOLD:
            use_market = True
            reasons.append(f"breakout_adx={adx:.1f}>{BREAKOUT_ADX_THRESHOLD}")

        # ── Dead zone check ───────────────────────────────────────────────────
        if utc_hour in DEAD_ZONE_HOURS:
            use_market = True
            reasons.append(f"dead_zone_hour={utc_hour}")

        order_type = "MARKET" if use_market else "LIMIT"

        # ── Limit price calculation ───────────────────────────────────────────
        limit_price = entry_price
        if order_type == "LIMIT":
            offset = entry_price * (LIMIT_OFFSET_BPS / 10_000)
            if side.upper() == "BUY":
                # Bid slightly below current ask to save half-spread
                limit_price = round(entry_price - offset, 8)
            else:
                limit_price = round(entry_price + offset, 8)

        rationale = f"{order_type}: " + (
            " | ".join(reasons) if reasons else "optimal_conditions"
        )

        logger.debug(f"ExecutionOptimizer: {rationale}")

        return {
            'order_type':   order_type,
            'limit_price':  limit_price,
            'timeout_sec':  LIMIT_TIMEOUT_SECONDS if order_type == "LIMIT" else 0,
            'rationale':    rationale,
            'fee_rate':     FEE_RATE if order_type == "LIMIT" else FEE_RATE * 2.5,
        }
