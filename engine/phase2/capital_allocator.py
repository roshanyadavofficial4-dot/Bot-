"""
Capital Allocator — Phase 2
==============================
Divides available capital across concurrent trade slots.

Principles:
  - Never allocate more than MAX_RISK_PER_TRADE per slot
  - Never let total allocation exceed the adaptive risk ceiling
  - Slots with higher edge_score get proportionally more capital
  - Always leave a cash buffer (MIN_CASH_RESERVE_PCT) unallocated

When only 1 slot: allocates up to adaptive risk_per_trade (existing behavior).
When 2 slots:     splits using edge_score weighting (higher score = more capital).

This module does NOT size positions — that's still risk_manager.py.
It returns a risk_pct per slot that risk_manager uses as input.
"""

import logging
from typing import Dict, List

from config import MAX_RISK_PER_TRADE
from engine.capital_adaptive import get_adaptive_params

logger = logging.getLogger("CapitalAllocator")

MIN_CASH_RESERVE_PCT = 0.05   # always keep 5% of balance unallocated


class CapitalAllocator:
    """
    Usage:
        allocator = CapitalAllocator()
        allocations = allocator.allocate(balance, slots)
        # slots: [{'symbol': ..., 'edge_score': float}, ...]
        # returns: [{'symbol': ..., 'risk_pct': float}, ...]
    """

    def allocate(
        self,
        balance: float,
        slots: List[Dict],
    ) -> List[Dict]:
        """
        Returns list of {'symbol': str, 'risk_pct': float} for each slot.

        risk_pct is the fraction of balance to risk on that trade.
        Feeds directly into risk_manager.calculate_position_size(risk=...).
        """
        if not slots:
            return []

        ap          = get_adaptive_params(balance)
        base_risk   = ap.risk_per_trade          # adaptive ceiling for this balance
        max_total   = base_risk * len(slots)     # cap total risk
        max_total   = min(max_total, MAX_RISK_PER_TRADE * 1.5)  # never exceed 3%

        # Single slot — simple case
        if len(slots) == 1:
            risk = min(base_risk, MAX_RISK_PER_TRADE)
            logger.debug(
                f"CapitalAllocator: 1 slot {slots[0].get('symbol')} "
                f"risk={risk*100:.3f}%"
            )
            return [{'symbol': slots[0].get('symbol', '?'), 'risk_pct': risk}]

        # Multi-slot — weight by edge_score
        edge_scores = [max(s.get('edge_score', 1.0), 1.0) for s in slots]
        total_score = sum(edge_scores)
        weights     = [e / total_score for e in edge_scores]

        # Apply cash reserve: only risk (1 - reserve) of available pool
        usable_risk = base_risk * len(slots) * (1 - MIN_CASH_RESERVE_PCT)
        usable_risk = min(usable_risk, max_total)

        allocations = []
        for slot, weight in zip(slots, weights):
            risk = usable_risk * weight
            risk = min(risk, MAX_RISK_PER_TRADE)   # per-slot hard cap
            allocations.append({
                'symbol':   slot.get('symbol', '?'),
                'risk_pct': round(risk, 5),
            })
            logger.debug(
                f"CapitalAllocator: {slot.get('symbol')} weight={weight:.2f} "
                f"edge={slot.get('edge_score', 0):.1f} risk={risk*100:.3f}%"
            )

        return allocations
