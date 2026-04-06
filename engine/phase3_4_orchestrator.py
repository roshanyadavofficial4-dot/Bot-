"""
Orchestrator Phase 3+4 Integration Patch
==========================================
Extends the existing Phase 2 Orchestrator with all Phase 3 and Phase 4
intelligence engines WITHOUT rewriting the core gate logic.

Usage pattern:
  Replace direct Orchestrator instantiation in main.py with:

      from engine.phase3_4_orchestrator import IntelligentOrchestrator
      orchestrator = IntelligentOrchestrator(balance=INITIAL_CAPITAL)

  The IntelligentOrchestrator is a drop-in wrapper — it preserves the
  existing 10-gate decision pipeline and adds intelligence gates on top.

Extended gate sequence (additions in brackets):
  [0] Failsafe gate          → Phase 4 hard-stop (abort before any processing)
  [1] Session Filter         → (existing Phase 2)
  [2] Anomaly Detector       → Phase 4 flash crash / news spike detection
  [3] Chop Filter            → (existing Phase 2)
  [4] Volatility Filter      → (existing Phase 2)
  [5] Spread Analyzer        → (existing Phase 2)
  [6] Market Efficiency      → Phase 3 clean vs noisy market gate
  [6] Regime Confidence      → (existing Phase 2)
  [7] Trade Slot Check       → (existing Phase 2)
  [8] Loss Pattern Detector  → Phase 3 veto on known bad patterns
  [9] Trade Clustering       → Phase 3 cluster overexposure gate
  [10] Edge Score Gate       → (existing Phase 2, threshold from AdaptiveParams)
  [11] Timing Optimizer      → Phase 3 timing quality gate
  [12] Expectancy Gate       → (existing Phase 2)
  [13] Risk Gate             → (existing Phase 2 + ProfitProtection multiplier)
  [14] Slippage Gate         → (existing Phase 2 + LatencyMonitor adjustment)
  [15] Strategy Health Gate  → Phase 3 defensive mode check
  → Execute

After execution: FeedbackLoop.on_trade_closed() updates all engines.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

# ── Phase 2 (existing) ────────────────────────────────────────────────────────
from engine.phase2.orchestrator        import OrchestratorDecision
from engine.phase2.regime_confidence   import RegimeConfidenceEngine
from engine.phase2.chop_filter         import ChopFilter
from engine.phase2.volatility_filter   import VolatilityFilter
from engine.phase2.session_filter      import SessionFilter
from engine.phase2.spread_analyzer     import SpreadAnalyzer
from engine.phase2.slippage_predictor  import SlippagePredictor
from engine.phase2.execution_optimizer import ExecutionOptimizer
from engine.phase2.edge_scorer         import EdgeScorer, MINIMUM_EDGE_SCORE
from engine.phase2.trade_priority      import TradePriorityEngine
from engine.phase2.dynamic_trade_slot  import DynamicTradeSlotEngine
from engine.phase2.capital_allocator   import CapitalAllocator
from engine.phase2.performance_tracker import PerformanceTracker
from engine.phase2.expectancy_engine   import ExpectancyEngine

# ── Phase 3 (new) ─────────────────────────────────────────────────────────────
from engine.phase3.strategy_health    import StrategyHealthMonitor
from engine.phase3.loss_pattern       import LossPatternDetector
from engine.phase3.adaptive_params    import AdaptiveParamEngine
from engine.phase3.market_efficiency  import MarketEfficiencyEngine
from engine.phase3.timing_optimizer   import TradeTimingOptimizer
from engine.phase3.trade_clustering   import TradeClusteringEngine
from engine.phase3.profit_protection  import ProfitProtectionEngine

# ── Phase 4 (new) ─────────────────────────────────────────────────────────────
from engine.phase4.strategy_selector  import StrategySelector
from engine.phase4.feedback_loop      import FeedbackLoop
from engine.phase4.capital_growth     import CapitalGrowthManager
from engine.phase4.strategy_failsafe  import StrategyFailsafe
from engine.phase4.anomaly_detector   import AnomalyDetector
from engine.phase4.latency_monitor    import LatencyMonitor

# ── Existing engines ──────────────────────────────────────────────────────────
from engine.risk_manager   import RiskManager
from engine.ml_brain       import MLBrain
from engine.ai_predictor   import AIPredictor
from config import INITIAL_CAPITAL, MAX_SPREAD_PCT

logger = logging.getLogger("IntelligentOrchestrator")

# Timing quality gate threshold — below this, suggest waiting
MIN_TIMING_QUALITY = 0.35


class IntelligentOrchestrator:
    """
    Drop-in replacement for the Phase 2 Orchestrator.
    Wraps all Phase 3 and Phase 4 engines into the gate pipeline.

    Main method: evaluate(candidate, df, balance) → OrchestratorDecision
    Post-trade:  on_trade_closed(result, conditions)
    """

    def __init__(self, balance: float = INITIAL_CAPITAL):
        # ── Phase 2 engines (existing) ────────────────────────────────────────
        self.session_filter   = SessionFilter()
        self.chop_filter      = ChopFilter()
        self.vol_filter       = VolatilityFilter()
        self.spread_analyzer  = SpreadAnalyzer()
        self.slippage_pred    = SlippagePredictor()
        self.exec_optimizer   = ExecutionOptimizer()
        self.edge_scorer      = EdgeScorer()
        self.trade_priority   = TradePriorityEngine()
        self.slot_engine      = DynamicTradeSlotEngine()
        self.capital_alloc    = CapitalAllocator()
        self.perf_tracker     = PerformanceTracker()
        self.expectancy_eng   = ExpectancyEngine()
        self.regime_conf      = RegimeConfidenceEngine()
        self.risk_manager     = RiskManager()
        self.ml_brain         = MLBrain()
        self.ai_predictor     = AIPredictor()

        # ── Phase 3 engines ────────────────────────────────────────────────────
        self.health_monitor   = StrategyHealthMonitor()
        self.loss_detector    = LossPatternDetector()
        self.adaptive_params  = AdaptiveParamEngine()
        self.mkt_efficiency   = MarketEfficiencyEngine()
        self.timing_optimizer = TradeTimingOptimizer()
        self.trade_clustering = TradeClusteringEngine()
        self.profit_protect   = ProfitProtectionEngine()
        self.profit_protect.set_session_start(balance)

        # ── Phase 4 engines ────────────────────────────────────────────────────
        self.strategy_sel     = StrategySelector()
        self.anomaly_detect   = AnomalyDetector()
        self.latency_monitor  = LatencyMonitor()
        self.capital_growth   = CapitalGrowthManager(initial_balance=balance)
        self.failsafe         = StrategyFailsafe(initial_balance=balance)
        self.feedback_loop    = FeedbackLoop(
            adaptive_params   = self.adaptive_params,
            strategy_selector = self.strategy_sel,
            loss_detector     = self.loss_detector,
            health_monitor    = self.health_monitor,
        )

        self._current_balance = balance
        logger.info(
            f"IntelligentOrchestrator: initialized at ${balance:.2f} "
            f"| Phase 3+4 engines active"
        )

    # ── Main evaluation pipeline ───────────────────────────────────────────────

    def evaluate(
        self,
        candidate: Dict,
        df=None,                     # pandas DataFrame from scanner (optional)
        balance: float = 0.0,
        active_trades: List = None,
        funding_rate: float = 0.0,
        signal_result=None,          # v5: SignalResult dataclass from strategy.generate_signal_full()
        # ── Phase-2 compatibility kwargs (mapped below) ─────────────────────
        signal: str = None,
        win_prob: float = None,
        current_balance: float = None,
        trades_today: int = None,
        open_trades: int = None,
        daily_pnl_pct: float = None,
        rr_ratio: float = None,
        risk_manager=None,
        **extra_kwargs,
    ) -> OrchestratorDecision:
        """
        Full Phase 3+4 extended gate pipeline.
        Returns OrchestratorDecision (compatible with existing executor).

        signal_result (optional): v5 SignalResult — when provided the candidate
        dict is enriched with liquidity fields so EdgeScorer runs in v5 mode,
        and the resulting decision is populated with structure-based entry/SL/TP.

        Phase-2 compatibility: accepts legacy kwargs (signal, win_prob, current_balance,
        trades_today, open_trades, daily_pnl_pct, rr_ratio, risk_manager) so that
        existing main.py call sites work without change.
        """
        # ── Always work on a copy so we never mutate the caller's dict ──────────
        candidate = dict(candidate)

        # ── Phase-2 compatibility: map legacy kwargs onto Phase-3/4 params ────
        if current_balance is not None and balance == 0.0:
            balance = current_balance
        if trades_today is not None and open_trades is not None:
            if active_trades is None:
                active_trades = [None] * int(open_trades)
        if signal is not None:
            candidate.setdefault('signal', signal)
        if win_prob is not None:
            candidate.setdefault('win_prob', win_prob)

        # ── v5: enrich candidate with liquidity fields from SignalResult ───────
        if signal_result is not None:
            candidate.setdefault('sweep_detected',     signal_result.sweep_detected)
            candidate.setdefault('absorption_score',   signal_result.absorption_score)
            candidate.setdefault('forced_move',        signal_result.forced_move_detected)
            candidate.setdefault('volume_spike_ratio', signal_result.volume_spike_ratio)
            candidate.setdefault('structure_reclaim',  signal_result.structure_reclaim)
            candidate.setdefault('win_prob',           signal_result.win_prob)
            candidate.setdefault('setup_type',         signal_result.setup_type)

        self._current_balance = balance
        decision = OrchestratorDecision()
        decision.symbol = candidate.get('symbol', '')

        perf_summary  = self.perf_tracker.get_summary()
        active_trades = active_trades or []

        # ── Get adaptive parameters for this cycle ────────────────────────────
        market_ctx = self._build_market_context(candidate, df)
        ap = self.adaptive_params.get_params(market_ctx, perf_summary)

        # Dynamic gate thresholds from adaptive params
        min_edge   = ap.get('min_edge_score', MINIMUM_EDGE_SCORE)
        min_prob   = ap.get('min_win_prob', 0.52)

        # ── Get current strategy ──────────────────────────────────────────────
        health_state = self.health_monitor.get_state()
        strategy_name, strategy_params = self.strategy_sel.select(
            regime           = market_ctx.get('regime', 'UNKNOWN'),
            perf_summary     = perf_summary,
            health_state     = health_state,
            efficiency_score = market_ctx.get('efficiency_score', 0.5),
        )
        # Override edge/prob thresholds from strategy
        min_edge = max(min_edge, strategy_params.get('min_edge_score', min_edge))
        min_prob = max(min_prob, strategy_params.get('min_win_prob', min_prob))

        # ═══════════════════════════════════════════════════════════════
        # GATE 0 — FAILSAFE (hard stop, first check)
        # ═══════════════════════════════════════════════════════════════
        fs_ok, fs_reason = self.failsafe.check(perf_summary, balance)
        if not fs_ok:
            return self._reject(decision, f"[FAILSAFE] {fs_reason}")

        # ═══════════════════════════════════════════════════════════════
        # GATE 1 — SESSION FILTER (existing)
        # ═══════════════════════════════════════════════════════════════
        if not self.session_filter.is_tradeable():
            return self._reject(decision, "[SESSION] Outside trading hours")

        # ═══════════════════════════════════════════════════════════════
        # GATE 2 — ANOMALY DETECTOR (Phase 4)
        # ═══════════════════════════════════════════════════════════════
        anomaly, anomaly_meta = self.anomaly_detect.check(df, candidate, funding_rate)
        if anomaly and anomaly_meta.get('action') in ('skip_trade', 'pause_3_candles', 'abort_all'):
            return self._reject(decision, f"[ANOMALY] {anomaly_meta.get('description', 'anomaly detected')}")

        # ═══════════════════════════════════════════════════════════════
        # GATE 3 — CHOP FILTER (existing)
        # ═══════════════════════════════════════════════════════════════
        chop_ok, chop_reason = self.chop_filter.check(candidate, df)
        if not chop_ok:
            return self._reject(decision, f"[CHOP] {chop_reason}")

        # ═══════════════════════════════════════════════════════════════
        # GATE 4 — VOLATILITY FILTER (existing)
        # ═══════════════════════════════════════════════════════════════
        vol_ok, vol_reason = self.vol_filter.check(candidate)
        if not vol_ok:
            return self._reject(decision, f"[VOL] {vol_reason}")

        # ═══════════════════════════════════════════════════════════════
        # GATE 5 — SPREAD ANALYZER (existing)
        # ═══════════════════════════════════════════════════════════════
        spread_ok, spread_reason = self.spread_analyzer.check(candidate)
        if not spread_ok:
            return self._reject(decision, f"[SPREAD] {spread_reason}")

        # ═══════════════════════════════════════════════════════════════
        # GATE 6 — MARKET EFFICIENCY (Phase 3)
        # ═══════════════════════════════════════════════════════════════
        eff_score, eff_meta = self.mkt_efficiency.evaluate(df, candidate)
        market_ctx['efficiency_score'] = eff_score
        if not eff_meta.get('is_tradeable', True):
            return self._reject(
                decision,
                f"[EFFICIENCY] Market noisy (score={eff_score:.2f} type={eff_meta.get('market_type')})"
            )

        # ═══════════════════════════════════════════════════════════════
        # GATE 7 — REGIME CONFIDENCE (existing)
        # ═══════════════════════════════════════════════════════════════
        regime, rc_conf, rc_desc = self.regime_conf.evaluate(candidate, df)
        decision.regime            = regime
        decision.regime_confidence = rc_conf
        if regime not in ('TRENDING',) or rc_conf < 0.45:
            # Allow RANGING only if strategy is MEAN_REVERT
            if not (regime == 'RANGING' and strategy_name == 'MEAN_REVERT'):
                return self._reject(decision, f"[REGIME] {regime} conf={rc_conf:.2f}")

        # ═══════════════════════════════════════════════════════════════
        # GATE 8 — TRADE SLOTS (existing)
        # ═══════════════════════════════════════════════════════════════
        slots_available = self.slot_engine.get_available_slots(len(active_trades))
        if slots_available <= 0:
            return self._reject(decision, f"[SLOTS] No slots available ({len(active_trades)} active)")

        # ═══════════════════════════════════════════════════════════════
        # GATE 9 — LOSS PATTERN DETECTOR (Phase 3)
        # ═══════════════════════════════════════════════════════════════
        pattern_conditions = {**candidate, **market_ctx,
                               'signal_direction': candidate.get('signal', 'UNKNOWN')}
        lp_blocked, lp_reason = self.loss_detector.check(pattern_conditions)
        if lp_blocked:
            return self._reject(decision, f"[LOSS_PATTERN] {lp_reason}")

        # ═══════════════════════════════════════════════════════════════
        # GATE 10 — TRADE CLUSTERING (Phase 3)
        # ═══════════════════════════════════════════════════════════════
        cluster_signal = {**candidate,
                          'regime':           regime,
                          'signal_direction': candidate.get('signal', 'UNKNOWN')}
        cl_blocked, cl_reason = self.trade_clustering.check(cluster_signal)
        if cl_blocked:
            return self._reject(decision, f"[CLUSTER] {cl_reason}")

        # ═══════════════════════════════════════════════════════════════
        # GATE 11 — ML WIN PROBABILITY
        # ═══════════════════════════════════════════════════════════════
        win_prob, top_drivers = self.ml_brain.predict_win_probability(candidate)
        if win_prob < min_prob:
            return self._reject(decision, f"[ML] win_prob={win_prob:.3f} < {min_prob:.3f}")

        # ═══════════════════════════════════════════════════════════════
        # GATE 12 — EDGE SCORE (existing, adaptive threshold)
        # ═══════════════════════════════════════════════════════════════
        edge_score, breakdown = self.edge_scorer.score(
            candidate,
            candidate.get('signal', 'BUY'),
            win_prob,
        )
        decision.edge_score = edge_score
        if edge_score < min_edge:
            return self._reject(decision, f"[EDGE] score={edge_score:.1f} < {min_edge:.1f}")

        # ═══════════════════════════════════════════════════════════════
        # GATE 13 — TRADE TIMING (Phase 3)
        # ═══════════════════════════════════════════════════════════════
        timing_q, timing_meta = self.timing_optimizer.evaluate(
            df, candidate, candidate.get('signal', 'BUY')
        )
        if timing_meta.get('urgency') == 'skip':
            return self._reject(decision, f"[TIMING] skip — {timing_meta.get('wait_reason')}")
        # 'wait_1' or 'wait_2' → soft advisory, does not block (logged only)
        if timing_meta.get('should_wait'):
            logger.debug(f"TimingAdvisory: {timing_meta.get('wait_reason')} — proceeding")

        # ═══════════════════════════════════════════════════════════════
        # GATE 14 — EXPECTANCY (existing)
        # ═══════════════════════════════════════════════════════════════
        rr_ratio = strategy_params.get('rr_target', 2.0)
        risk_usd = balance * 0.015   # rough estimate for EV check
        ev, ev_viable, ev_reason = self.expectancy_eng.evaluate_proposed(
            rr_ratio, win_prob, risk_usd
        )
        if not ev_viable:
            return self._reject(decision, f"[EV] {ev_reason}")

        # ═══════════════════════════════════════════════════════════════
        # GATE 15 — RISK MANAGER (existing + profit protection)
        # ═══════════════════════════════════════════════════════════════
        session_pnl_pct = perf_summary.get('session_pnl', 0) / max(balance, 1e-9)
        pp_params       = self.profit_protect.get_protection_params(session_pnl_pct)

        if pp_params.get('in_cooldown'):
            return self._reject(
                decision,
                f"[PROFIT_PROTECT] Big-win cooldown — "
                f"{pp_params['cooldown_remaining']:.0f}s remaining"
            )

        risk_ok, risk_reason = self.risk_manager.can_trade(balance)
        if not risk_ok:
            return self._reject(decision, f"[RISK] {risk_reason}")

        # ═══════════════════════════════════════════════════════════════
        # GATE 16 — STRATEGY HEALTH (Phase 3)
        # ═══════════════════════════════════════════════════════════════
        health_state_now, health_meta = self.health_monitor.evaluate(perf_summary)
        if health_meta.get('halt_trading'):
            return self._reject(decision, f"[HEALTH] {health_state_now} — trading halted")

        # ═══════════════════════════════════════════════════════════════
        # GATE 17 — SLIPPAGE + LATENCY (existing + Phase 4 adjustment)
        # ═══════════════════════════════════════════════════════════════
        lat_ok, lat_meta = self.latency_monitor.get_trading_permission()
        slip_mult = lat_meta.get('slippage_multiplier', 1.0)
        slip_ok, slip_reason = self.slippage_pred.check(candidate, slippage_multiplier=slip_mult)
        if not slip_ok:
            return self._reject(decision, f"[SLIPPAGE] {slip_reason}")

        # ═══════════════════════════════════════════════════════════════
        # ALL GATES PASSED — build execute decision
        # ═══════════════════════════════════════════════════════════════

        # Capital allocation with health + profit protection multipliers
        combined_risk_mult = (
            health_meta.get('risk_multiplier', 1.0) *
            pp_params.get('risk_multiplier', 1.0) *
            strategy_params.get('risk_multiplier', 1.0)
        )

        slots = self.capital_alloc.allocate(balance, [
            {'symbol': decision.symbol, 'edge_score': edge_score}
        ])
        base_risk_pct = slots[0]['risk_pct'] if slots else 0.015
        final_risk_pct = round(base_risk_pct * combined_risk_mult, 5)

        decision.execute              = True
        decision.signal               = candidate.get('signal', 'BUY')
        decision.risk_pct             = final_risk_pct
        decision.order_recommendation = self.exec_optimizer.recommend(candidate, df)

        # ── v5: populate structure-based entry / SL / TP from SignalResult ────
        if signal_result is not None and signal_result.entry_price > 0:
            decision.entry_price = signal_result.entry_price
            decision.stop_loss   = signal_result.stop_loss
            decision.take_profit = signal_result.take_profit
            decision.rr_ratio    = signal_result.rr_ratio
            decision.setup_type  = signal_result.setup_type
            decision.confidence  = signal_result.confidence

        decision.engine_outputs       = {
            'strategy':          strategy_name,
            'setup_type':        getattr(decision, 'setup_type', None),
            'edge_score':        edge_score,
            'win_prob':          win_prob,
            'top_drivers':       top_drivers,
            'regime':            regime,
            'regime_conf':       rc_conf,
            'efficiency_score':  eff_score,
            'market_type':       eff_meta.get('market_type'),
            'timing_quality':    timing_q,
            'timing_urgency':    timing_meta.get('urgency'),
            'adaptive_params':   {k: ap[k] for k in
                                  ('min_edge_score', 'min_win_prob', 'risk_pct_multiplier')},
            'health_state':      health_state_now,
            'risk_multiplier':   round(combined_risk_mult, 3),
            'latency_tier':      lat_meta.get('tier'),
            'ev_usd':            ev,
            'breakdown':         breakdown,
        }

        logger.info(
            f"✅ EXECUTE {decision.symbol} {decision.signal} | "
            f"edge={edge_score:.1f} prob={win_prob:.3f} "
            f"risk={final_risk_pct*100:.2f}% "
            f"setup={getattr(decision, 'setup_type', 'legacy')} "
            f"entry={getattr(decision, 'entry_price', 0):.4f} "
            f"sl={getattr(decision, 'stop_loss', 0):.4f} "
            f"tp={getattr(decision, 'take_profit', 0):.4f} "
            f"strategy={strategy_name} regime={regime} "
            f"timing={timing_meta.get('urgency')}"
        )

        return decision

    # ── Post-trade update ──────────────────────────────────────────────────────

    def on_trade_closed(
        self,
        trade_result: Dict,
        market_conditions: Dict,
        current_balance: float,
    ) -> None:
        """
        Call after every trade close. Updates all intelligence engines.

        trade_result: {symbol, pnl_usd, pnl_pct, won, strategy,
                       edge_score, win_prob_used, rr_ratio, hold_seconds}
        market_conditions: {regime, adx, ofi, session, hour_utc, atr_pct,
                            signal_direction, efficiency_score}
        """
        self._current_balance = current_balance

        # ── Performance tracker ───────────────────────────────────────────────
        self.perf_tracker.record_trade(
            symbol       = trade_result.get('symbol', ''),
            pnl_usd      = trade_result.get('pnl_usd', 0.0),
            pnl_pct      = trade_result.get('pnl_pct', 0.0),
            side         = trade_result.get('side', 'LONG'),
            hold_seconds = trade_result.get('hold_seconds', 0),
        )

        # ── Expectancy engine ─────────────────────────────────────────────────
        self.expectancy_eng.record(
            pnl_usd  = trade_result.get('pnl_usd', 0.0),
            risk_usd = current_balance * 0.015,
        )

        # ── Profit protection ─────────────────────────────────────────────────
        self.profit_protect.record_trade(
            pnl_pct         = trade_result.get('pnl_pct', 0.0),
            current_balance = current_balance,
        )

        # ── Trade cluster cleanup ─────────────────────────────────────────────
        trade_id = trade_result.get('trade_id', trade_result.get('symbol', ''))
        self.trade_clustering.remove_trade(trade_id)

        # ── ML brain retraining trigger ───────────────────────────────────────
        # Retrain every 50 trades
        perf = self.perf_tracker.get_summary()
        if perf.get('total_trades', 0) % 50 == 0 and perf.get('total_trades', 0) > 0:
            logger.info("IntelligentOrchestrator: triggering ML retrain")
            self.ml_brain.train_model()

        # ── Feedback loop (central intelligence update) ───────────────────────
        self.feedback_loop.on_trade_closed(
            trade_result     = trade_result,
            market_conditions= market_conditions,
            perf_summary     = perf,
        )

        # ── Capital growth check ──────────────────────────────────────────────
        milestone, suggested_wd = self.capital_growth.check_withdrawal_milestone(current_balance)
        if milestone:
            logger.info(
                f"💰 Capital milestone! Suggested withdrawal: ${suggested_wd:.2f}"
            )

        logger.debug(
            f"on_trade_closed: {trade_result.get('symbol')} "
            f"pnl=${trade_result.get('pnl_usd', 0):.4f} "
            f"won={trade_result.get('won')} | "
            f"session_trades={perf.get('total_trades')} "
            f"wr={perf.get('win_rate', 0):.2f}"
        )

    # ── Helpers ─────────────────────────────────────────────────────────────

    def register_active_trade(self, trade_id: str, signal: Dict) -> None:
        """Register a newly entered trade in the cluster engine."""
        self.trade_clustering.register_trade(trade_id, signal)

    def daily_reset(self, new_balance: float) -> None:
        """Call at daily session start."""
        self.profit_protect.reset_session(new_balance)
        self.failsafe.daily_reset(new_balance)
        self.adaptive_params.reset_to_defaults()
        logger.info(f"IntelligentOrchestrator: daily reset at ${new_balance:.2f}")

    def get_intelligence_summary(self) -> Dict:
        """Full system state for dashboard / Telegram reporting."""
        perf    = self.perf_tracker.get_summary()
        growth  = self.capital_growth.get_growth_summary(self._current_balance)
        health  = self.health_monitor.get_state()
        fs      = self.failsafe.get_status(self._current_balance)
        cluster = self.trade_clustering.get_cluster_summary()
        lat     = self.latency_monitor.get_summary()

        return {
            'performance':    perf,
            'growth':         growth,
            'health_state':   health,
            'strategy':       self.strategy_sel.get_current(),
            'failsafe':       fs,
            'cluster':        cluster,
            'latency':        lat,
            'adaptive_params': self.adaptive_params.get_current(),
            'loss_patterns':  self.loss_detector.get_pattern_summary()[:5],
            'feedback_log':   self.feedback_loop.get_adaptation_log(5),
        }

    def _build_market_context(self, candidate: Dict, df) -> Dict:
        """Extract market context dict from candidate for use by adaptive engines."""
        import datetime
        hour_utc = datetime.datetime.utcnow().hour

        atr     = candidate.get('atr', 0.0)
        price   = candidate.get('last_price', 1.0)
        atr_pct = atr / max(price, 1e-9)

        vol_level = 'low'
        if atr_pct > 0.012:
            vol_level = 'high'
        elif atr_pct > 0.005:
            vol_level = 'med'

        return {
            'regime':           candidate.get('regime', 'UNKNOWN'),
            'volatility_level': vol_level,
            'adx':              candidate.get('adx', 0.0),
            'atr_pct':          atr_pct,
            'ofi':              candidate.get('ofi', 0.0),
            'session':          candidate.get('session', 'UNKNOWN'),
            'hour_utc':         hour_utc,
            'signal_direction': candidate.get('signal', 'UNKNOWN'),
            'efficiency_score': 0.5,  # updated after mkt_efficiency gate
        }

    @staticmethod
    def _reject(decision: OrchestratorDecision, reason: str) -> OrchestratorDecision:
        decision.execute     = False
        decision.skip_reason = reason
        logger.debug(f"SKIP: {reason}")
        return decision
