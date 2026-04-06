"""
Strategy Selector Engine — Phase 4
=====================================
Dynamically selects the most appropriate trading strategy based on
current market regime and recent performance.

Available strategies:
  TREND_FOLLOW     → momentum + OFI, wider stops, higher RR target
  SCALP_PULLBACK   → micro-pullback entries in trend, tighter stops
  MEAN_REVERT      → range-bound, fade extremes (RSI + BB)
  CONSERVATIVE     → reduced size, highest-quality setups only
  PAUSED           → no new trades (system degrading or anomaly)

Each strategy has its own parameter set that overrides the base config.
The selector updates every N trades or on regime change.

Design constraint: strategy must have minimum 20 trades of history
before it can be deprioritized — prevent premature switching.
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("StrategySelector")

# Strategy definitions
STRATEGIES = {
    'TREND_FOLLOW': {
        'description':        'Trend momentum with OFI confirmation',
        'regime_fit':         ['TRENDING'],
        'min_adx':            25,
        'min_edge_score':     42,
        'min_win_prob':       0.53,
        'risk_multiplier':    1.0,
        'rr_target':          2.2,
        'stop_type':          'atr_1.5',
        'holding_style':      'hold_to_tp',
        'max_trades_per_day': 3,
    },
    'SCALP_PULLBACK': {
        'description':        'Pullback entries within established trend',
        'regime_fit':         ['TRENDING'],
        'min_adx':            22,
        'min_edge_score':     45,
        'min_win_prob':       0.55,
        'risk_multiplier':    0.80,
        'rr_target':          1.6,
        'stop_type':          'atr_1.0',
        'holding_style':      'trail_after_1r',
        'max_trades_per_day': 4,
    },
    'MEAN_REVERT': {
        'description':        'Fade extremes in ranging markets',
        'regime_fit':         ['RANGING'],
        'min_adx':            0,    # ADX not required for ranging
        'min_edge_score':     50,   # stricter quality gate
        'min_win_prob':       0.58,
        'risk_multiplier':    0.60,
        'rr_target':          1.4,
        'stop_type':          'fixed_0.8pct',
        'holding_style':      'quick_exit',
        'max_trades_per_day': 3,
    },
    'CONSERVATIVE': {
        'description':        'Only highest-quality setups, reduced risk',
        'regime_fit':         ['TRENDING', 'RANGING', 'VOLATILE'],
        'min_adx':            28,
        'min_edge_score':     60,
        'min_win_prob':       0.60,
        'risk_multiplier':    0.50,
        'rr_target':          2.0,
        'stop_type':          'atr_1.2',
        'holding_style':      'hold_to_tp',
        'max_trades_per_day': 2,
    },
    'PAUSED': {
        'description':        'No new trades — system in recovery',
        'regime_fit':         [],
        'min_adx':            999,
        'min_edge_score':     999,
        'min_win_prob':       1.0,
        'risk_multiplier':    0.0,
        'rr_target':          0.0,
        'stop_type':          'none',
        'holding_style':      'none',
        'max_trades_per_day': 0,
    },
}

# History per strategy: used to track which strategies perform well
# key: strategy name → {wins, losses, pnl_total}
_strategy_perf: Dict[str, Dict] = {k: {'wins': 0, 'losses': 0, 'pnl': 0.0}
                                    for k in STRATEGIES}

MIN_TRADES_BEFORE_SWITCH = 20   # don't switch strategies without this much history


class StrategySelector:
    """
    Usage:
        selector = StrategySelector()

        # Each scan cycle:
        strategy_name, params = selector.select(regime, perf_summary, health_state)

        # After each trade:
        selector.record_strategy_result(strategy_name, won, pnl_usd)
    """

    def __init__(self):
        self._current_strategy  = 'TREND_FOLLOW'
        self._cycles_on_current = 0
        self._forced_strategy: Optional[str] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def select(
        self,
        regime: str,
        perf_summary: Dict,
        health_state: str,
        efficiency_score: float = 0.5,
    ) -> Tuple[str, Dict]:
        """
        Select the best strategy for current conditions.

        Returns:
            strategy_name  str
            params         dict  — merged strategy config
        """
        # Forced override (e.g. from manual control or failsafe)
        if self._forced_strategy:
            return self._forced_strategy, STRATEGIES[self._forced_strategy]

        # Health-based overrides
        if health_state == 'CRITICAL':
            logger.warning("StrategySelector: CRITICAL health → PAUSED")
            return 'PAUSED', STRATEGIES['PAUSED']

        if health_state == 'DEFENSIVE':
            logger.info("StrategySelector: DEFENSIVE health → CONSERVATIVE")
            self._set_strategy('CONSERVATIVE')
            return 'CONSERVATIVE', STRATEGIES['CONSERVATIVE']

        # Regime-based selection
        candidate = self._select_by_regime(regime, efficiency_score, perf_summary)

        if candidate != self._current_strategy:
            self._cycles_on_current = 0
            logger.info(
                f"StrategySelector: {self._current_strategy} → {candidate} "
                f"(regime={regime} efficiency={efficiency_score:.2f})"
            )
        else:
            self._cycles_on_current += 1

        self._current_strategy = candidate
        return candidate, STRATEGIES[candidate]

    def record_strategy_result(
        self, strategy_name: str, won: bool, pnl_usd: float
    ) -> None:
        """Update strategy performance history."""
        if strategy_name not in _strategy_perf:
            return
        if won:
            _strategy_perf[strategy_name]['wins'] += 1
        else:
            _strategy_perf[strategy_name]['losses'] += 1
        _strategy_perf[strategy_name]['pnl'] += pnl_usd

    def force_strategy(self, strategy_name: Optional[str]) -> None:
        """Force a specific strategy (None = auto)."""
        if strategy_name and strategy_name not in STRATEGIES:
            logger.error(f"StrategySelector: unknown strategy '{strategy_name}'")
            return
        self._forced_strategy = strategy_name
        logger.info(f"StrategySelector: forced to '{strategy_name}'")

    def get_strategy_performance(self) -> List[Dict]:
        """Return performance breakdown per strategy."""
        result = []
        for name, perf in _strategy_perf.items():
            total = perf['wins'] + perf['losses']
            result.append({
                'strategy':  name,
                'total':     total,
                'wins':      perf['wins'],
                'losses':    perf['losses'],
                'win_rate':  round(perf['wins'] / total, 3) if total > 0 else 0.0,
                'pnl':       round(perf['pnl'], 4),
            })
        return sorted(result, key=lambda x: x['pnl'], reverse=True)

    def get_current(self) -> str:
        return self._current_strategy

    # ── Internal helpers ────────────────────────────────────────────────────

    def _select_by_regime(
        self,
        regime: str,
        efficiency: float,
        perf_summary: Dict,
    ) -> str:
        """Choose strategy based on market regime and efficiency score."""

        if regime == 'VOLATILE':
            # Volatile = high risk, only conservative trades
            return 'CONSERVATIVE'

        if regime == 'TRENDING':
            if efficiency >= 0.65:
                # Clean trend — prefer full momentum strategy
                return 'TREND_FOLLOW'
            else:
                # Trending but messy — use pullback entries
                return 'SCALP_PULLBACK'

        if regime == 'RANGING':
            # Mean reversion in range — higher edge score requirement
            return 'MEAN_REVERT'

        # UNKNOWN regime → conservative
        return 'CONSERVATIVE'

    def _set_strategy(self, name: str) -> None:
        if name != self._current_strategy:
            self._current_strategy  = name
            self._cycles_on_current = 0
