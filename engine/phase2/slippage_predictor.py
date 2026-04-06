"""
Slippage Predictor — Phase 2
==============================
Pre-trade slippage estimation. This is a thin enhancement layer on top
of execution_model.estimate_slippage() that adds:

  1. Side-specific asymmetry: buy slippage != sell slippage (ask/bid depth differ)
  2. Session adjustment: dead-zone sessions have 50% more slippage
  3. A simple confidence interval: best/worst case range
  4. Go/no-go gate: block if worst-case slippage makes trade unprofitable

The existing execution_model.py estimate_slippage() is called as the
base calculation — this module does NOT duplicate that logic.
"""

import logging
from datetime import datetime, timezone
from typing import Dict

from engine.execution_model import estimate_slippage
from config import FEE_RATE, SLIPPAGE_BUFFER

logger = logging.getLogger("SlippagePredictor")

# Hours where liquidity is thin → widen slippage estimate
DEAD_ZONE_HOURS   = {20, 21, 22, 23, 0, 1}
DEAD_ZONE_MULT    = 1.5   # 50% extra slippage during dead zone

# If worst-case slippage > this fraction of expected profit, skip
MAX_SLIPPAGE_PROFIT_RATIO = 0.45   # slippage must not consume >45% of expected profit


class SlippagePredictor:
    """
    Usage:
        predictor = SlippagePredictor()
        result = predictor.predict(notional, atr_pct, liquidity_depth,
                                   side, expected_profit_pct)
    """

    def predict(
        self,
        notional: float,
        atr_pct: float,
        liquidity_depth: float,
        side: str,                     # 'buy' | 'sell'
        expected_profit_pct: float,    # from RR calculation
        utc_hour: int = None,
    ) -> Dict:
        """
        Returns dict:
            base_slippage    float — base estimate from execution_model
            adjusted         float — session-adjusted estimate
            worst_case       float — conservative upper bound
            trade_viable     bool  — False if worst_case makes trade unprofitable
            reason           str
        """
        if utc_hour is None:
            utc_hour = datetime.now(timezone.utc).hour

        # ── Base estimate from existing execution model ───────────────────────
        base = estimate_slippage(notional, atr_pct, liquidity_depth, side)

        # ── Session adjustment ────────────────────────────────────────────────
        session_mult = DEAD_ZONE_MULT if utc_hour in DEAD_ZONE_HOURS else 1.0
        adjusted     = base * session_mult

        # ── Confidence interval: worst case = 2× adjusted ────────────────────
        worst_case   = adjusted * 2.0

        # ── Viability gate ────────────────────────────────────────────────────
        # Round-trip slippage cost
        rt_slippage  = adjusted * 2   # entry + exit
        total_cost   = rt_slippage + (FEE_RATE * 2)

        trade_viable = True
        reason       = "ok"

        if expected_profit_pct > 0:
            cost_ratio = total_cost / expected_profit_pct
            if cost_ratio > MAX_SLIPPAGE_PROFIT_RATIO:
                trade_viable = False
                reason = (
                    f"slippage_cost_ratio={cost_ratio:.2f} > "
                    f"max={MAX_SLIPPAGE_PROFIT_RATIO} "
                    f"(total_cost={total_cost*100:.3f}% vs profit={expected_profit_pct*100:.3f}%)"
                )
                logger.debug(f"SlippagePredictor BLOCK: {reason}")

        logger.debug(
            f"SlippagePredictor: base={base*100:.4f}% adj={adjusted*100:.4f}% "
            f"worst={worst_case*100:.4f}% viable={trade_viable}"
        )

        return {
            'base_slippage':    round(base, 6),
            'adjusted':         round(adjusted, 6),
            'worst_case':       round(worst_case, 6),
            'total_cost':       round(total_cost, 6),
            'trade_viable':     trade_viable,
            'reason':           reason,
        }
