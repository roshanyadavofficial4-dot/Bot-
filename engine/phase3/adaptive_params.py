"""
Adaptive Parameter Engine — Phase 3
======================================
Dynamically adjusts system thresholds based on market conditions and
recent performance. Acts as a living configuration layer that tightens
or relaxes filters as the environment changes.

Principle: parameters shift incrementally — never abrupt jumps.
All adjustments are bounded within safe operating ranges.

Adjustable Parameters:
  - min_edge_score      (base: 40)
  - min_win_prob        (base: 0.52)
  - min_adx             (base: 20)
  - min_ofi             (base: 0.08)
  - risk_pct_multiplier (base: 1.0)
  - max_spread_mult     (base: 1.0)

Triggers:
  - Volatile market   → tighten all filters
  - Clean trend       → relax filters slightly (more opportunities)
  - Poor performance  → tighten filters
  - Strong performance → allow marginal relaxation (anti-overfit floor)
  - Ranging market    → tighten OR skip entirely
"""

import logging
from typing import Dict

logger = logging.getLogger("AdaptiveParamEngine")

# ── Absolute bounds — parameters never go outside these ─────────────────────
BOUNDS = {
    'min_edge_score':      (35, 75),
    'min_win_prob':        (0.50, 0.70),
    'min_adx':             (18, 35),
    'min_ofi':             (0.05, 0.25),
    'risk_pct_multiplier': (0.25, 1.25),
    'max_spread_mult':     (0.5, 2.0),
}

# ── Base values (from config.py) ─────────────────────────────────────────────
DEFAULTS = {
    'min_edge_score':      40.0,
    'min_win_prob':        0.52,
    'min_adx':             20.0,
    'min_ofi':             0.08,
    'risk_pct_multiplier': 1.0,
    'max_spread_mult':     1.0,
}

# Step sizes for incremental adjustments
STEP = {
    'min_edge_score':      2.0,
    'min_win_prob':        0.01,
    'min_adx':             1.0,
    'min_ofi':             0.01,
    'risk_pct_multiplier': 0.05,
    'max_spread_mult':     0.1,
}


class AdaptiveParamEngine:
    """
    Usage:
        engine = AdaptiveParamEngine()
        # Each scan cycle:
        params = engine.get_params(market_context, perf_summary)
        # params is a dict of adjusted thresholds
    """

    def __init__(self):
        self._params = dict(DEFAULTS)
        self._adjustment_log: list = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_params(self, market_context: Dict, perf_summary: Dict) -> Dict:
        """
        Returns the current adjusted parameter set.

        market_context: {regime, volatility_level, adx, efficiency_score, ...}
        perf_summary:   PerformanceTracker.get_summary() output
        """
        adjustments = []

        # ── Market structure adjustments ──────────────────────────────────────
        regime = market_context.get('regime', 'UNKNOWN')
        vol_level = market_context.get('volatility_level', 'med')   # low/med/high
        efficiency = market_context.get('efficiency_score', 0.5)    # 0-1

        if regime == 'VOLATILE':
            adjustments.append(('tighten', 'volatile_market', {
                'min_edge_score':      +4,
                'min_win_prob':        +0.03,
                'min_adx':             +3,
                'risk_pct_multiplier': -0.20,
            }))

        elif regime == 'RANGING':
            adjustments.append(('tighten', 'ranging_market', {
                'min_edge_score':      +6,
                'min_win_prob':        +0.04,
                'min_ofi':             +0.05,
                'risk_pct_multiplier': -0.25,
            }))

        elif regime == 'TRENDING':
            # Only relax in a clean, efficient trend
            if efficiency > 0.65:
                adjustments.append(('relax', 'clean_trend', {
                    'min_edge_score':      -2,
                    'min_win_prob':        -0.01,
                    'risk_pct_multiplier': +0.10,
                }))

        if vol_level == 'high':
            adjustments.append(('tighten', 'high_vol', {
                'max_spread_mult':  -0.2,
                'min_ofi':          +0.03,
            }))
        elif vol_level == 'low':
            adjustments.append(('tighten', 'low_vol_low_opportunity', {
                'min_edge_score':  +2,
                'min_adx':         +2,
            }))

        # ── Performance-based adjustments ─────────────────────────────────────
        win_rate     = perf_summary.get('win_rate', 0.5)
        consec_loss  = perf_summary.get('consecutive_losses', 0)
        total_trades = perf_summary.get('total_trades', 0)
        expectancy   = perf_summary.get('expectancy', 0.0)

        if total_trades >= 10:
            if win_rate < 0.35 or expectancy < -0.002:
                adjustments.append(('tighten', 'poor_performance', {
                    'min_edge_score':      +4,
                    'min_win_prob':        +0.03,
                    'risk_pct_multiplier': -0.15,
                }))
            elif win_rate > 0.60 and expectancy > 0.01:
                # Strong performance — marginal relaxation (capped at defaults)
                adjustments.append(('relax', 'strong_performance', {
                    'min_edge_score':      -2,
                    'risk_pct_multiplier': +0.05,
                }))

        if consec_loss >= 3:
            adjustments.append(('tighten', 'consecutive_losses', {
                'min_edge_score':      +3,
                'min_win_prob':        +0.02,
                'risk_pct_multiplier': -0.10,
            }))

        # ── Apply all adjustments and clamp ──────────────────────────────────
        target = dict(DEFAULTS)
        for direction, label, deltas in adjustments:
            for key, delta in deltas.items():
                if key in target:
                    target[key] += delta
                    logger.debug(f"AdaptiveParam: [{label}] {key} {'+' if delta>0 else ''}{delta}")

        # Clamp to bounds
        for key, (lo, hi) in BOUNDS.items():
            target[key] = round(max(lo, min(hi, target[key])), 4)

        # Smooth transitions — blend current → target by 30% each cycle
        for key in self._params:
            if key in target:
                self._params[key] = round(
                    self._params[key] * 0.70 + target[key] * 0.30, 4
                )

        summary = {
            'adjustments_applied': [a[1] for a in adjustments],
            **self._params,
        }

        logger.debug(
            f"AdaptiveParams: edge={self._params['min_edge_score']:.1f} "
            f"win_prob={self._params['min_win_prob']:.3f} "
            f"risk_mult={self._params['risk_pct_multiplier']:.2f}"
        )

        return summary

    def reset_to_defaults(self) -> None:
        """Hard reset — use after entering a new session or after restart."""
        self._params = dict(DEFAULTS)
        logger.info("AdaptiveParams: reset to defaults")

    def get_current(self) -> Dict:
        """Get current params without triggering an update."""
        return dict(self._params)
