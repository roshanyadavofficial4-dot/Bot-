"""
Edge Scorer — Phase 2 Engine v5.1
====================================
Assigns a numeric score (0-100) to any candidate signal.

v5.1 UPGRADE: Tiered scoring model for PRIMARY and SECONDARY setups.

SCORING TIERS:
  PRIMARY   ≥ 80 → full position size (size_multiplier=1.0)
  SECONDARY 65-79 → reduced position size (size_multiplier=0.6)
  REJECTED  < 65 → blocked regardless of signal

SCORING COMPONENTS v5.1 (same structure as v5.0, thresholds unchanged):
  Component                Max pts  Logic
  ──────────────────────   ───────  ─────────────────────────────────────────
  ADX strength                15    ADX >= 30 -> 15; >= 25 -> 10; >= 20 -> 5
  Liquidity sweep quality     20    wick_ratio >= 0.6 -> 20; >= 0.45 -> 12; else 0
  Absorption score            20    absorption_score linear scale 0.35 -> 1.0 (was 0.40)
  Forced move                 10    forced_move=True -> 10; else 0
  Volume spike                15    vol_ratio >= 2.0 -> 15; >= 1.5 -> 10; >= 1.3 -> 5
  Structure reclaim/break     10    structure confirmed -> 10; else 0
  Win probability             10    ML win_prob linear scale 0.50 -> 1.0
  ──────────────────────   ───────
  TOTAL                      100   (+ RR bonus up to 5)

KEY CHANGE v5.1:
  - MINIMUM_EDGE_SCORE lowered to 40 (was 50) to allow secondary setups to pass gate
  - SECONDARY_TIER_SCORE (65) is the threshold above which secondary setups are eligible
  - PRIMARY_TIER_SCORE (80) is the threshold for primary classification
  - Orchestrator reads 'setup_tier' from signal to apply correct size multiplier
  - Legacy scoring unchanged

Backward compatibility: all original score fields remain valid.
"""

import logging
from typing import Dict, Tuple

from config import MAX_SPREAD_PCT, MAX_ATR_VOLATILITY

logger = logging.getLogger("EdgeScorer")

# Gate threshold — minimum to pass the edge gate at all
MINIMUM_EDGE_SCORE = 40

# Tier thresholds (used by orchestrator for size decisions)
PRIMARY_TIER_SCORE   = 80.0
SECONDARY_TIER_SCORE = 65.0


def classify_score_tier(score: float) -> str:
    """
    Returns 'PRIMARY', 'SECONDARY', or 'REJECTED' based on edge score.
    Used by orchestrator to determine position sizing.
    """
    if score >= PRIMARY_TIER_SCORE:
        return 'PRIMARY'
    elif score >= SECONDARY_TIER_SCORE:
        return 'SECONDARY'
    return 'REJECTED'


class EdgeScorer:
    """
    Stateless signal scoring engine — v5.1 Tiered Liquidity-Driven.

    Usage:
        scorer = EdgeScorer()
        score, breakdown = scorer.score(candidate, signal_direction, win_prob)
        tier = classify_score_tier(score)
        if score >= MINIMUM_EDGE_SCORE and tier != 'REJECTED':
            # proceed — orchestrator handles size modulation by tier
    """

    def score(
        self,
        candidate: Dict,
        signal_direction: str,
        win_prob: float = 0.5,
    ) -> Tuple[float, Dict]:
        """
        Returns (total_score, breakdown_dict).

        Automatically detects if new liquidity fields are present.
        v5.1: Also includes setup_tier in breakdown for orchestrator.
        """
        has_liquidity_fields = (
            'sweep_detected'   in candidate or
            'absorption_score' in candidate or
            'forced_move'      in candidate
        )

        if has_liquidity_fields:
            return self._score_liquidity_edge(candidate, signal_direction, win_prob)
        else:
            return self._score_legacy(candidate, signal_direction, win_prob)

    # ── v5.1 Liquidity-Driven Scoring ─────────────────────────────────────────

    def _score_liquidity_edge(
        self,
        candidate: Dict,
        signal_direction: str,
        win_prob: float,
    ) -> Tuple[float, Dict]:
        """
        Full v5.1 scoring. Supports both primary and secondary signal scoring.

        Key v5.1 change: absorption lower bound extended to 0.35 (was 0.40)
        to award partial credit for secondary-tier signals that still show
        absorption, just at a slightly weaker level.
        """
        adx              = candidate.get('adx', 0.0)
        vol_ratio        = candidate.get('vol_ratio', 1.0)
        sweep_detected   = candidate.get('sweep_detected', False)
        absorption_score = candidate.get('absorption_score', 0.0)
        forced_move      = candidate.get('forced_move', False)
        vol_spike_ratio  = candidate.get('volume_spike_ratio', vol_ratio)
        structure_ok     = (
            candidate.get('structure_reclaim', False) or
            candidate.get('structure_break', False) or
            candidate.get('setup_type', '') in ('TRAP_REVERSAL', 'CONTINUATION')
        )
        setup_type       = candidate.get('setup_type', '')
        setup_tier       = candidate.get('setup_tier', 'PRIMARY')   # v5.1

        breakdown = {}

        # ── 1. ADX strength (15 pts) ──────────────────────────────────────────
        if adx >= 30:         breakdown['adx'] = 15.0
        elif adx >= 25:       breakdown['adx'] = 10.0
        elif adx >= 20:       breakdown['adx'] = 5.0
        else:                 breakdown['adx'] = 0.0

        # ── 2. Liquidity sweep quality (20 pts) ───────────────────────────────
        if sweep_detected:
            wr = candidate.get('wick_ratio', 0.0)
            if wr >= 0.65:    breakdown['sweep'] = 20.0
            elif wr >= 0.50:  breakdown['sweep'] = 14.0
            elif wr >= 0.45:  breakdown['sweep'] = 8.0
            else:             breakdown['sweep'] = 5.0
        else:
            if setup_type == 'CONTINUATION' and absorption_score >= 0.35:
                breakdown['sweep'] = 10.0   # absorption zone as sweep proxy
            else:
                breakdown['sweep'] = 0.0

        # ── 3. Absorption strength (20 pts) ───────────────────────────────────
        # v5.1: scale from 0.35 (secondary floor) up to 1.0
        if absorption_score >= 0.35:
            normalized = (absorption_score - 0.35) / 0.65
            breakdown['absorption'] = round(normalized * 20.0, 1)
        else:
            breakdown['absorption'] = 0.0

        # ── 4. Forced move (10 pts) ───────────────────────────────────────────
        breakdown['forced_move'] = 10.0 if forced_move else 0.0

        # ── 5. Volume spike (15 pts) ──────────────────────────────────────────
        # v5.1: 1.3x earns 5 pts (secondary tier), 1.5x earns 10, 2.0x earns 15
        eff_vol = max(vol_spike_ratio, vol_ratio)
        if eff_vol >= 2.0:         breakdown['volume'] = 15.0
        elif eff_vol >= 1.5:       breakdown['volume'] = 10.0
        elif eff_vol >= 1.3:       breakdown['volume'] = 5.0
        else:                      breakdown['volume'] = 0.0

        # ── 6. Structure reclaim / break (10 pts) ─────────────────────────────
        breakdown['structure'] = 10.0 if structure_ok else 0.0

        # ── 7. Win probability (10 pts) ───────────────────────────────────────
        if win_prob >= 0.50:
            breakdown['win_prob'] = round((win_prob - 0.50) / 0.50 * 10.0, 1)
        else:
            breakdown['win_prob'] = 0.0

        # ── Bonus: RR quality ─────────────────────────────────────────────────
        rr_ratio = candidate.get('rr_ratio', 1.0)
        if rr_ratio >= 2.5:        breakdown['rr_bonus'] = 5.0
        elif rr_ratio >= 1.5:      breakdown['rr_bonus'] = 2.0
        else:                      breakdown['rr_bonus'] = 0.0

        total_score = min(sum(breakdown.values()), 100.0)
        breakdown['total'] = round(total_score, 2)
        breakdown['model'] = 'liquidity_v5.1'

        # v5.1: include tier classification in breakdown for orchestrator
        breakdown['score_tier'] = classify_score_tier(total_score)

        logger.debug(
            f"EdgeScorer v5.1: {signal_direction} tier={breakdown['score_tier']} "
            f"adx={breakdown['adx']} sweep={breakdown['sweep']} "
            f"abs={breakdown['absorption']} forced={breakdown['forced_move']} "
            f"vol={breakdown['volume']} struct={breakdown['structure']} "
            f"winP={breakdown['win_prob']} RR={breakdown.get('rr_bonus',0)} "
            f"-> TOTAL={total_score:.1f}"
        )

        return round(total_score, 2), breakdown

    # ── Legacy Scoring (backward compatibility) ────────────────────────────────

    def _score_legacy(
        self,
        candidate: Dict,
        signal_direction: str,
        win_prob: float,
    ) -> Tuple[float, Dict]:
        """Original v4.0 model. Used when liquidity fields are absent."""
        adx        = candidate.get('adx', 0.0)
        ofi        = candidate.get('ofi', 0.0)
        vol_ratio  = candidate.get('vol_ratio', 1.0)
        cvd        = candidate.get('cvd', 0.0)
        spread_pct = candidate.get('spread_pct', 0.0)
        atr        = candidate.get('atr', 0.0)
        last_price = candidate.get('last_price', 1.0)

        breakdown = {}

        if adx >= 30:   breakdown['adx'] = 25.0
        elif adx >= 25: breakdown['adx'] = 18.0
        elif adx >= 20: breakdown['adx'] = 10.0
        else:           breakdown['adx'] = 0.0

        ofi_abs = abs(ofi)
        if ofi_abs >= 0.30:   breakdown['ofi'] = 20.0
        elif ofi_abs >= 0.15: breakdown['ofi'] = 12.0
        elif ofi_abs >= 0.08: breakdown['ofi'] = 5.0
        else:                 breakdown['ofi'] = 0.0

        if vol_ratio >= 2.0:   breakdown['volume'] = 15.0
        elif vol_ratio >= 1.5: breakdown['volume'] = 10.0
        elif vol_ratio >= 1.3: breakdown['volume'] = 5.0
        else:                  breakdown['volume'] = 0.0

        if signal_direction == 'BUY'  and cvd > 0:  breakdown['cvd'] = 15.0
        elif signal_direction == 'SELL' and cvd < 0: breakdown['cvd'] = 15.0
        elif abs(cvd) < 0.05:                        breakdown['cvd'] = 5.0
        else:                                        breakdown['cvd'] = 0.0

        if win_prob >= 0.52:
            breakdown['win_prob'] = round((win_prob - 0.52) / 0.48 * 15.0, 1)
        else:
            breakdown['win_prob'] = 0.0

        if spread_pct < MAX_SPREAD_PCT * 0.5:  breakdown['spread'] = 5.0
        elif spread_pct < MAX_SPREAD_PCT:      breakdown['spread'] = 3.0
        else:                                  breakdown['spread'] = 0.0

        if atr > 0 and last_price > 0:
            ap = atr / last_price
            if 0.001 <= ap <= 0.008: breakdown['atr_quality'] = 5.0
            elif ap <= MAX_ATR_VOLATILITY: breakdown['atr_quality'] = 2.0
            else:                    breakdown['atr_quality'] = 0.0
        else:
            breakdown['atr_quality'] = 0.0

        total_score = min(sum(breakdown.values()), 100.0)
        breakdown['total'] = round(total_score, 2)
        breakdown['model'] = 'legacy_v4'
        breakdown['score_tier'] = classify_score_tier(total_score)

        return round(total_score, 2), breakdown
