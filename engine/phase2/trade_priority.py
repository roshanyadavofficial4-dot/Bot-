"""
Trade Priority Engine — Phase 2
==================================
When multiple symbols pass all filters, this engine ranks them and selects
the best N candidates for execution (where N = available trade slots).

Ranking formula:
    priority_score = edge_score * regime_conf_weight * liquidity_weight

Where:
    edge_score        — from EdgeScorer (0–100)
    regime_conf_weight — 1.0 + (regime_confidence - 0.5) * 0.4
                         → high-confidence trending = up to 1.2x boost
    liquidity_weight  — 1.0 + min(liquidity_depth / 10000, 0.2)
                         → deep order book = small bonus (max 1.2x)

Tiebreaker: higher ADX wins.
"""

import logging
from typing import Dict, List, Tuple

from engine.phase2.edge_scorer import EdgeScorer

logger = logging.getLogger("TradePriority")

scorer = EdgeScorer()


class TradePriorityEngine:
    """
    Usage:
        engine = TradePriorityEngine()
        ranked = engine.rank(candidates_with_scores, max_slots)
        # candidates_with_scores: list of dicts, each has:
        #   'candidate'        — raw scanner candidate
        #   'signal'           — 'BUY' | 'SELL'
        #   'win_prob'         — float
        #   'regime_confidence'— float
    """

    def rank(
        self,
        candidates: List[Dict],
        max_slots: int = 1,
    ) -> List[Dict]:
        """
        Returns up to max_slots candidates, sorted by priority_score descending.

        Each returned item has an added 'priority_score' and 'edge_score' field.
        """
        if not candidates:
            return []

        scored = []
        for item in candidates:
            candidate       = item['candidate']
            signal          = item.get('signal', 'BUY')
            win_prob        = item.get('win_prob', 0.5)
            regime_conf     = item.get('regime_confidence', 0.5)

            edge_score, breakdown = scorer.score(candidate, signal, win_prob)

            # Regime confidence weight (0.9 at low conf → 1.2 at high conf)
            regime_weight = 1.0 + (regime_conf - 0.5) * 0.4
            regime_weight = max(0.8, min(1.2, regime_weight))

            # Liquidity weight (1.0 → 1.2)
            depth         = candidate.get('liquidity_depth', 0.0)
            liq_weight    = 1.0 + min(depth / 10_000.0, 0.2)

            priority_score = edge_score * regime_weight * liq_weight

            scored.append({
                **item,
                'edge_score':     round(edge_score, 2),
                'edge_breakdown': breakdown,
                'priority_score': round(priority_score, 3),
            })

            logger.debug(
                f"TradePriority: {candidate.get('symbol', '?')} "
                f"edge={edge_score:.1f} regime_w={regime_weight:.2f} "
                f"liq_w={liq_weight:.2f} → priority={priority_score:.2f}"
            )

        # Sort descending by priority_score, tiebreak by ADX
        scored.sort(
            key=lambda x: (x['priority_score'], x['candidate'].get('adx', 0.0)),
            reverse=True,
        )

        selected = scored[:max_slots]

        if selected:
            top = selected[0]
            logger.info(
                f"TradePriority: selected {top['candidate'].get('symbol')} "
                f"priority={top['priority_score']:.2f} edge={top['edge_score']:.1f} "
                f"from {len(candidates)} candidates"
            )

        return selected
