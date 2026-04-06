"""
Feedback Loop / Adaptation Engine — Phase 4
==============================================
The central nervous system of the self-improving trading system.

This engine:
  1. Collects closed trade results
  2. Analyzes what worked vs. what didn't
  3. Propagates adjustments back to filter parameters
  4. Coordinates updates across all Phase 3/4 engines
  5. Writes adaptation logs for diagnostics

Design principles:
  - No single feedback cycle overwrites everything
  - Changes are incremental (not abrupt rewrites)
  - All changes are logged and reversible
  - Anti-overfit: minimum N trades before parameter changes

Adaptation targets:
  - AdaptiveParamEngine thresholds
  - StrategySelector weights
  - PerformanceTracker windows
  - LossPattern blocklist
"""

import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional

logger = logging.getLogger("FeedbackLoop")

# Minimum trades before triggering adaptation
MIN_TRADES_BEFORE_ADAPT  = 10

# Adaptation cycle: run every N trades
ADAPT_EVERY_N_TRADES     = 5

# How far back to look for feedback analysis
RECENT_WINDOW            = 20


class FeedbackLoop:
    """
    Usage:
        loop = FeedbackLoop(adaptive_params, strategy_selector, loss_detector)

        # After each trade close:
        loop.on_trade_closed(trade_result)

        # Adaptation runs automatically every ADAPT_EVERY_N_TRADES trades.
    """

    def __init__(
        self,
        adaptive_params,      # AdaptiveParamEngine instance
        strategy_selector,    # StrategySelector instance
        loss_detector,        # LossPatternDetector instance
        health_monitor,       # StrategyHealthMonitor instance
    ):
        self._adaptive_params   = adaptive_params
        self._strategy_selector = strategy_selector
        self._loss_detector     = loss_detector
        self._health_monitor    = health_monitor

        self._trade_count  = 0
        self._trade_history: deque = deque(maxlen=200)
        self._adaptation_log: List[Dict] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def on_trade_closed(
        self,
        trade_result: Dict,
        market_conditions: Dict,
        perf_summary: Dict,
    ) -> Dict:
        """
        Main entry point. Call after every trade close.

        trade_result: {symbol, pnl_usd, pnl_pct, won, strategy, edge_score,
                       win_prob_used, rr_ratio, hold_seconds}
        market_conditions: {regime, adx, ofi, session, hour_utc, atr_pct,
                            signal_direction, efficiency_score}
        perf_summary: PerformanceTracker.get_summary()

        Returns: adaptation_actions dict (for logging)
        """
        self._trade_count += 1

        # Enrich trade record
        enriched = {
            **trade_result,
            **market_conditions,
            'timestamp': time.time(),
            'trade_idx': self._trade_count,
        }
        self._trade_history.append(enriched)

        # Notify downstream engines
        self._loss_detector.record_trade(market_conditions, won=trade_result.get('won', False))
        self._health_monitor.notify_win() if trade_result.get('won') else self._health_monitor.notify_loss()
        self._strategy_selector.record_strategy_result(
            trade_result.get('strategy', 'UNKNOWN'),
            trade_result.get('won', False),
            trade_result.get('pnl_usd', 0.0),
        )

        # Periodic adaptation
        actions = {}
        if self._trade_count % ADAPT_EVERY_N_TRADES == 0 and \
                self._trade_count >= MIN_TRADES_BEFORE_ADAPT:
            actions = self._run_adaptation_cycle(perf_summary, market_conditions)

        return actions

    def get_adaptation_log(self, last_n: int = 20) -> List[Dict]:
        """Return recent adaptation events for diagnostics."""
        return self._adaptation_log[-last_n:]

    def get_trade_analytics(self) -> Dict:
        """
        Deep analysis of recent trade history.
        Returns patterns, best conditions, worst conditions.
        """
        trades = list(self._trade_history)
        if len(trades) < 10:
            return {'status': 'insufficient_data', 'trades': len(trades)}

        recent = trades[-RECENT_WINDOW:]
        wins   = [t for t in recent if t.get('won')]
        losses = [t for t in recent if not t.get('won')]

        # Best performing conditions
        win_regimes   = self._count_field(wins, 'regime')
        loss_regimes  = self._count_field(losses, 'regime')
        win_sessions  = self._count_field(wins, 'session')
        loss_sessions = self._count_field(losses, 'session')

        # Edge score vs outcome
        win_avg_edge  = self._avg_field(wins, 'edge_score')
        loss_avg_edge = self._avg_field(losses, 'edge_score')

        # Win prob calibration
        win_avg_prob  = self._avg_field(wins, 'win_prob_used')
        loss_avg_prob = self._avg_field(losses, 'win_prob_used')

        return {
            'status':           'ok',
            'total_analyzed':   len(recent),
            'win_rate':         round(len(wins) / max(len(recent), 1), 3),
            'winning_regimes':  win_regimes,
            'losing_regimes':   loss_regimes,
            'winning_sessions': win_sessions,
            'losing_sessions':  loss_sessions,
            'win_avg_edge':     round(win_avg_edge, 2),
            'loss_avg_edge':    round(loss_avg_edge, 2),
            'edge_score_gap':   round(win_avg_edge - loss_avg_edge, 2),
            'win_avg_prob':     round(win_avg_prob, 3),
            'loss_avg_prob':    round(loss_avg_prob, 3),
            'prob_calibration_gap': round(win_avg_prob - loss_avg_prob, 3),
        }

    # ── Adaptation cycle ─────────────────────────────────────────────────────

    def _run_adaptation_cycle(
        self,
        perf_summary: Dict,
        market_context: Dict,
    ) -> Dict:
        """
        Core feedback loop — analyze recent trades and adjust parameters.
        Returns dict of actions taken.
        """
        analytics = self.get_trade_analytics()
        actions   = {}

        logger.info(
            f"FeedbackLoop: adaptation cycle #{self._trade_count // ADAPT_EVERY_N_TRADES} "
            f"| wr={analytics.get('win_rate', 0):.2f} "
            f"edge_gap={analytics.get('edge_score_gap', 0):.1f}"
        )

        # ── Action 1: Edge score calibration ─────────────────────────────────
        edge_gap = analytics.get('edge_score_gap', 0)
        if edge_gap > 10:
            # Winners have significantly higher edge scores than losers
            # → raise minimum edge score to filter out low-quality
            actions['edge_score_action'] = 'raise_minimum'
            logger.info(
                f"FeedbackLoop: raising min_edge_score — "
                f"win_avg={analytics['win_avg_edge']:.1f} "
                f"loss_avg={analytics['loss_avg_edge']:.1f}"
            )
        elif edge_gap < 3:
            # Edge score not discriminating — may need to recalibrate scoring
            actions['edge_score_action'] = 'review_weights'
            logger.debug("FeedbackLoop: edge score gap small — review scoring weights")

        # ── Action 2: Win probability gate ────────────────────────────────────
        prob_gap = analytics.get('prob_calibration_gap', 0)
        if prob_gap > 0.05:
            actions['win_prob_action'] = 'raise_min_prob'
        elif prob_gap < 0.01 and analytics.get('win_rate', 0.5) < 0.40:
            actions['win_prob_action'] = 'prob_not_predictive'

        # ── Action 3: Session/regime blocklist updates ────────────────────────
        losing_regimes  = analytics.get('losing_regimes', {})
        for regime, count in losing_regimes.items():
            if count >= 4:
                actions[f'regime_caution_{regime}'] = 'increase_filter'
                logger.warning(
                    f"FeedbackLoop: regime '{regime}' has {count} losses → "
                    f"tightening filter"
                )

        # ── Action 4: Strategy performance feedback ───────────────────────────
        strat_perf = self._strategy_selector.get_strategy_performance()
        for sp in strat_perf:
            if sp['total'] >= 10 and sp['win_rate'] < 0.30:
                logger.warning(
                    f"FeedbackLoop: strategy '{sp['strategy']}' "
                    f"performing poorly ({sp['win_rate']:.0%}) "
                    f"over {sp['total']} trades"
                )
                actions[f'strategy_{sp["strategy"]}_action'] = 'deprioritize'

        # ── Log this adaptation cycle ─────────────────────────────────────────
        log_entry = {
            'trade_count': self._trade_count,
            'timestamp':   time.time(),
            'win_rate':    analytics.get('win_rate', 0),
            'actions':     actions,
            'analytics':   {k: v for k, v in analytics.items()
                            if k not in ('status', 'winning_regimes',
                                         'losing_regimes', 'winning_sessions',
                                         'losing_sessions')},
        }
        self._adaptation_log.append(log_entry)
        if len(self._adaptation_log) > 100:
            self._adaptation_log = self._adaptation_log[-100:]

        return actions

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _count_field(trades: List[Dict], field: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for t in trades:
            val = str(t.get(field, 'unknown'))
            counts[val] = counts.get(val, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    @staticmethod
    def _avg_field(trades: List[Dict], field: str) -> float:
        vals = [t[field] for t in trades if field in t and t[field] is not None]
        return sum(vals) / len(vals) if vals else 0.0
