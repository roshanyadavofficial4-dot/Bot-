"""
Orchestrator — Phase 2 Core v5.1
==============================
The single decision authority for the entire trading system.

v5.1 UPDATE: Integrates with Strategy v5.1 tiered setup system.
  - Reads setup_tier (PRIMARY/SECONDARY) from SignalResult
  - Applies size_multiplier for dynamic position sizing
  - SECONDARY tier: risk_pct * 0.6, otherwise all gates identical
  - Re-entry signals: always SECONDARY sizing
  - Edge score gate: uses MINIMUM_EDGE_SCORE (40) to allow secondary through
  - New gate [11]: setup_tier gate rejects REJECTED tier explicitly

Gate sequence (17-gate system, gates 1-10 unchanged):

    [1]  Session Filter        -> are we in tradeable hours?
    [2]  Chop Filter           -> is the market moving directionally?
    [3]  Volatility Filter     -> is ATR in the tradeable band?
    [4]  Spread Analyzer       -> is spread normal?
    [5]  Regime Confidence     -> trending + sufficient confidence?
    [6]  Trade Slot Check      -> do we have capacity?
    [7]  Edge Score Gate       -> signal quality >= threshold?
    [8]  Expectancy Gate       -> positive EV after costs?
    [9]  Risk Gate             -> risk manager permits trade?
    [10] Slippage Gate         -> execution cost viable?
    [11] Setup Tier Gate       -> score >= SECONDARY threshold (65)? [v5.1]

ALL gates must pass. Any single failure = skip trade.

v5.1 MULTI-SYMBOL PARALLELIZATION:
  evaluate_multi() accepts a list of pre-filtered candidates from multiple
  symbols, scores all via TradePriorityEngine, and selects top N by slot
  availability. Correlation and cluster checks applied before selection.

  This is the preferred entry point when scanning >1 symbol simultaneously.
  evaluate() remains available for single-symbol compatibility.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

from engine.phase2.regime_confidence  import RegimeConfidenceEngine
from engine.phase2.chop_filter        import ChopFilter
from engine.phase2.volatility_filter  import VolatilityFilter
from engine.phase2.session_filter     import SessionFilter
from engine.phase2.spread_analyzer    import SpreadAnalyzer
from engine.phase2.slippage_predictor import SlippagePredictor
from engine.phase2.execution_optimizer import ExecutionOptimizer
from engine.phase2.edge_scorer        import EdgeScorer, MINIMUM_EDGE_SCORE, SECONDARY_TIER_SCORE, classify_score_tier
from engine.phase2.trade_priority     import TradePriorityEngine
from engine.phase2.dynamic_trade_slot import DynamicTradeSlotEngine
from engine.phase2.capital_allocator  import CapitalAllocator
from engine.phase2.performance_tracker import PerformanceTracker
from engine.phase2.expectancy_engine  import ExpectancyEngine
from engine.phase3.trade_clustering   import TradeClusteringEngine

logger = logging.getLogger("Orchestrator")

MIN_REGIME_CONFIDENCE = 0.45
MIN_EDGE_SCORE        = MINIMUM_EDGE_SCORE      # 40 (v5.1: lowered to pass secondary)
MIN_EDGE_TIER_SCORE   = SECONDARY_TIER_SCORE    # 65 (gate 11: reject REJECTED tier)
MIN_WIN_PROB          = 0.50                    # v5.1: aligned with secondary floor


class OrchestratorDecision:
    """
    Structured result returned by Orchestrator.evaluate().

    v5.1 additions:
      setup_tier        — PRIMARY | SECONDARY | REJECTED
      size_multiplier   — 1.0 or 0.6 (applied to risk_pct)
      is_reentry        — True if signal is a micro re-entry
      effective_risk_pct — risk_pct * size_multiplier (actual risk after tier adjustment)
    """
    __slots__ = (
        'execute', 'skip_reason', 'signal', 'symbol',
        'edge_score', 'regime', 'regime_confidence',
        'risk_pct', 'order_recommendation',
        'engine_outputs',
        'entry_price', 'stop_loss', 'take_profit', 'rr_ratio',
        'setup_type', 'setup_description', 'confidence',
        # v5.1
        'setup_tier', 'size_multiplier', 'is_reentry', 'effective_risk_pct',
    )

    def __init__(self):
        self.execute              = False
        self.skip_reason          = ""
        self.signal               = "HOLD"
        self.symbol               = ""
        self.edge_score           = 0.0
        self.regime               = "UNKNOWN"
        self.regime_confidence    = 0.0
        self.risk_pct             = 0.0
        self.order_recommendation = {}
        self.engine_outputs       = {}
        self.entry_price          = 0.0
        self.stop_loss            = 0.0
        self.take_profit          = 0.0
        self.rr_ratio             = 1.5
        self.setup_type           = ""
        self.setup_description    = ""
        self.confidence           = 0.0
        # v5.1
        self.setup_tier           = "PRIMARY"
        self.size_multiplier      = 1.0
        self.is_reentry           = False
        self.effective_risk_pct   = 0.0

    def __repr__(self):
        if self.execute:
            return (
                f"OrchestratorDecision(EXECUTE {self.signal} {self.symbol} "
                f"tier={self.setup_tier} setup={self.setup_type} "
                f"edge={self.edge_score:.1f} RR={self.rr_ratio:.2f} "
                f"regime={self.regime}@{self.regime_confidence:.2f} "
                f"risk={self.effective_risk_pct*100:.3f}% entry={self.entry_price:.6f})"
            )
        return f"OrchestratorDecision(SKIP: {self.skip_reason})"


class Orchestrator:
    """
    Central decision engine. One instance per bot run.

    Usage (single symbol):
        decision = orchestrator.evaluate(candidate, signal, win_prob, ...)

    Usage (multi-symbol, preferred for frequency):
        decisions = orchestrator.evaluate_multi(candidates_with_signals, ...)
        for d in decisions:
            if d.execute: ...
    """

    def __init__(self):
        self.regime_engine    = RegimeConfidenceEngine()
        self.chop_filter      = ChopFilter()
        self.vol_filter       = VolatilityFilter()
        self.session_filter   = SessionFilter()
        self.spread_analyzer  = SpreadAnalyzer()
        self.slippage_pred    = SlippagePredictor()
        self.exec_optimizer   = ExecutionOptimizer()
        self.edge_scorer      = EdgeScorer()
        self.trade_priority   = TradePriorityEngine()
        self.slot_engine      = DynamicTradeSlotEngine()
        self.capital_alloc    = CapitalAllocator()
        self.perf_tracker     = PerformanceTracker()
        self.expectancy_eng   = ExpectancyEngine()
        # v5.1: clustering engine for multi-symbol correlation protection
        self.cluster_engine   = TradeClusteringEngine()

        self._decision_log: List[Dict] = []

    # ── Single-symbol evaluation (backward compatible) ────────────────────────

    def evaluate(
        self,
        candidate: Dict,
        signal: str,
        win_prob: float,
        current_balance: float,
        trades_today: int,
        open_trades: int,
        daily_pnl_pct: float,
        rr_ratio: float = 2.0,
        risk_manager=None,
        signal_result=None,
    ) -> OrchestratorDecision:
        """
        Run all gate sequence. Returns OrchestratorDecision.

        v5.1 integration:
          - Reads setup_tier and size_multiplier from signal_result
          - Applies size_multiplier to risk_pct before returning
          - Gate [11]: rejects signals with score_tier == REJECTED
        """
        decision        = OrchestratorDecision()
        decision.symbol = candidate.get('symbol', 'UNKNOWN')
        outputs         = {}

        # ── v5.1: Enrich candidate with all fields from SignalResult ──────────
        if signal_result is not None:
            rr_ratio                            = signal_result.rr_ratio
            win_prob                            = signal_result.win_prob
            decision.entry_price                = signal_result.entry_price
            decision.stop_loss                  = signal_result.stop_loss
            decision.take_profit                = signal_result.take_profit
            decision.rr_ratio                   = signal_result.rr_ratio
            decision.setup_type                 = signal_result.setup_type
            decision.setup_description          = signal_result.setup_description
            decision.confidence                 = signal_result.confidence
            decision.setup_tier                 = signal_result.setup_tier
            decision.size_multiplier            = signal_result.size_multiplier
            decision.is_reentry                 = signal_result.is_reentry

            candidate = dict(candidate)
            candidate['sweep_detected']         = signal_result.sweep_detected
            candidate['absorption_score']       = signal_result.absorption_score
            candidate['forced_move']            = signal_result.forced_move_detected
            candidate['volume_spike_ratio']     = signal_result.volume_spike_ratio
            candidate['structure_reclaim']      = signal_result.structure_reclaim
            candidate['rr_ratio']               = signal_result.rr_ratio
            candidate['setup_type']             = signal_result.setup_type
            candidate['wick_ratio']             = getattr(signal_result, 'wick_ratio', 0.0)
            candidate['setup_tier']             = signal_result.setup_tier   # v5.1

        if signal not in ('BUY', 'SELL'):
            decision.skip_reason = f"no_signal (signal={signal})"
            self._log_decision(decision, outputs)
            return decision

        decision.signal = signal
        t0 = time.time()

        # ─────────────────────────────────────────────────────────────────────
        # [1] SESSION FILTER
        # ─────────────────────────────────────────────────────────────────────
        session_ok, session_name, session_reason = self.session_filter.is_active_session()
        outputs['session'] = {'ok': session_ok, 'name': session_name, 'reason': session_reason}
        if not session_ok:
            decision.skip_reason = f"session_filter: {session_reason}"
            self._log_decision(decision, outputs)
            return decision

        # ─────────────────────────────────────────────────────────────────────
        # [2] CHOP FILTER
        # ─────────────────────────────────────────────────────────────────────
        df = candidate.get('df')
        is_choppy, chop_reason = self.chop_filter.is_choppy(df)
        outputs['chop'] = {'is_choppy': is_choppy, 'reason': chop_reason}
        if is_choppy:
            decision.skip_reason = f"chop_filter: {chop_reason}"
            self._log_decision(decision, outputs)
            return decision

        # ─────────────────────────────────────────────────────────────────────
        # [3] VOLATILITY FILTER
        # ─────────────────────────────────────────────────────────────────────
        atr        = candidate.get('atr', 0.0)
        last_price = candidate.get('last_price', 1.0)
        h_vol      = candidate.get('h_volatility', 0.0)
        vol_ok, vol_reason = self.vol_filter.is_tradeable(atr, last_price, h_vol)
        outputs['volatility'] = {'ok': vol_ok, 'reason': vol_reason}
        if not vol_ok:
            decision.skip_reason = f"volatility_filter: {vol_reason}"
            self._log_decision(decision, outputs)
            return decision

        # ─────────────────────────────────────────────────────────────────────
        # [4] SPREAD ANALYZER
        # ─────────────────────────────────────────────────────────────────────
        spread_pct = candidate.get('spread_pct', 0.0)
        spread_ok, spread_quality, spread_reason = self.spread_analyzer.analyze(
            decision.symbol, spread_pct
        )
        outputs['spread'] = {'ok': spread_ok, 'quality': spread_quality, 'reason': spread_reason}
        if not spread_ok:
            decision.skip_reason = f"spread_analyzer: {spread_reason}"
            self._log_decision(decision, outputs)
            return decision

        # ─────────────────────────────────────────────────────────────────────
        # [5] REGIME CONFIDENCE
        # ─────────────────────────────────────────────────────────────────────
        adx = candidate.get('adx', 0.0)
        regime_ok, regime, regime_conf, regime_desc = self.regime_engine.evaluate(
            df, adx, h_vol
        )
        outputs['regime'] = {
            'ok': regime_ok, 'regime': regime,
            'confidence': regime_conf, 'description': regime_desc
        }
        if not regime_ok:
            decision.skip_reason = f"regime_confidence: {regime} conf={regime_conf:.2f}"
            self._log_decision(decision, outputs)
            return decision

        decision.regime            = regime
        decision.regime_confidence = regime_conf

        # ─────────────────────────────────────────────────────────────────────
        # [6] TRADE SLOT CHECK
        # ─────────────────────────────────────────────────────────────────────
        consec_losses  = risk_manager.consecutive_losses if risk_manager else 0
        concurrent_rem, daily_rem, slot_reason = self.slot_engine.get_available_slots(
            current_balance, consec_losses, daily_pnl_pct, open_trades, trades_today
        )
        outputs['slots'] = {
            'concurrent_remaining': concurrent_rem,
            'daily_remaining': daily_rem,
            'reason': slot_reason,
        }
        if concurrent_rem <= 0 or daily_rem <= 0:
            decision.skip_reason = f"no_slots: {slot_reason}"
            self._log_decision(decision, outputs)
            return decision

        # ─────────────────────────────────────────────────────────────────────
        # [7] EDGE SCORE GATE
        # ─────────────────────────────────────────────────────────────────────
        edge_score, edge_breakdown = self.edge_scorer.score(candidate, signal, win_prob)
        outputs['edge'] = {'score': edge_score, 'breakdown': edge_breakdown}
        if edge_score < MIN_EDGE_SCORE:
            decision.skip_reason = f"edge_score={edge_score:.1f} < min={MIN_EDGE_SCORE}"
            self._log_decision(decision, outputs)
            return decision

        decision.edge_score = edge_score

        # ─────────────────────────────────────────────────────────────────────
        # [8] EXPECTANCY GATE
        # ─────────────────────────────────────────────────────────────────────
        risk_usd    = current_balance * 0.015
        ev_usd, ev_ok, ev_reason = self.expectancy_eng.evaluate_proposed(
            rr_ratio, win_prob, risk_usd
        )
        outputs['expectancy'] = {'ev_usd': ev_usd, 'viable': ev_ok, 'reason': ev_reason}
        if not ev_ok:
            decision.skip_reason = f"expectancy_gate: {ev_reason}"
            self._log_decision(decision, outputs)
            return decision

        # ─────────────────────────────────────────────────────────────────────
        # [9] RISK MANAGER GATE
        # ─────────────────────────────────────────────────────────────────────
        if risk_manager is not None:
            if risk_manager.is_in_cooldown():
                decision.skip_reason = "risk_cooldown"
                self._log_decision(decision, outputs)
                return decision

        # ─────────────────────────────────────────────────────────────────────
        # [10] SLIPPAGE GATE
        # ─────────────────────────────────────────────────────────────────────
        atr_pct         = (atr / last_price) if last_price > 0 else 0.0
        depth           = candidate.get('liquidity_depth', 1000.0)
        expected_profit = atr_pct * rr_ratio
        slip_result     = self.slippage_pred.predict(
            risk_usd, atr_pct, depth, signal.lower(), expected_profit
        )
        outputs['slippage'] = slip_result
        if not slip_result['trade_viable']:
            decision.skip_reason = f"slippage_gate: {slip_result['reason']}"
            self._log_decision(decision, outputs)
            return decision

        # ─────────────────────────────────────────────────────────────────────
        # [11] SETUP TIER GATE (v5.1)
        # Rejects signals whose score is below the secondary threshold (65).
        # This gate works in tandem with gate [7]:
        #   Gate [7] blocks score < 40 (MINIMUM_EDGE_SCORE)
        #   Gate [11] blocks score 40-64 (below SECONDARY_TIER_SCORE)
        # Scores 65-79 pass as SECONDARY, 80+ pass as PRIMARY.
        # ─────────────────────────────────────────────────────────────────────
        score_tier = edge_breakdown.get('score_tier', classify_score_tier(edge_score))
        if score_tier == 'REJECTED':
            decision.skip_reason = (
                f"setup_tier_gate: score={edge_score:.1f} below secondary threshold ({MIN_EDGE_TIER_SCORE})"
            )
            self._log_decision(decision, outputs)
            return decision

        # Override setup_tier from edge_breakdown if signal_result had a different one
        # (signal_result tier is authoritative — edge scorer confirms/overrides)
        if score_tier != decision.setup_tier:
            logger.debug(
                f"[{decision.symbol}] Tier override: signal={decision.setup_tier} "
                f"scorer={score_tier} — using scorer result"
            )
            decision.setup_tier    = score_tier
            decision.size_multiplier = 1.0 if score_tier == 'PRIMARY' else 0.6

        outputs['tier'] = {'score_tier': score_tier, 'size_multiplier': decision.size_multiplier}

        # ─────────────────────────────────────────────────────────────────────
        # ALL GATES PASSED — build execution recommendation
        # ─────────────────────────────────────────────────────────────────────

        slot_info  = [{'symbol': decision.symbol, 'edge_score': edge_score}]
        allocation = self.capital_alloc.allocate(current_balance, slot_info)
        base_risk_pct = allocation[0]['risk_pct'] if allocation else 0.015

        # v5.1: Apply size multiplier for secondary/re-entry setups
        effective_risk_pct         = base_risk_pct * decision.size_multiplier
        decision.risk_pct          = base_risk_pct
        decision.effective_risk_pct = effective_risk_pct

        order_rec = self.exec_optimizer.recommend(spread_pct, adx, last_price, signal)

        if decision.entry_price > 0:
            order_rec['order_type']    = 'LIMIT'
            order_rec['limit_price']   = decision.entry_price
            order_rec['stop_loss']     = decision.stop_loss
            order_rec['take_profit']   = decision.take_profit
            order_rec['rr_ratio']      = decision.rr_ratio
            order_rec['setup_type']    = decision.setup_type
            order_rec['setup_tier']    = decision.setup_tier       # v5.1
            order_rec['size_multiplier'] = decision.size_multiplier  # v5.1
            order_rec['effective_risk_pct'] = effective_risk_pct   # v5.1

        decision.execute              = True
        decision.order_recommendation = order_rec
        decision.engine_outputs       = outputs

        elapsed = (time.time() - t0) * 1000
        logger.info(
            f"Orchestrator: EXECUTE {signal} {decision.symbol} | "
            f"tier={decision.setup_tier} setup={decision.setup_type} "
            f"edge={edge_score:.1f} RR={decision.rr_ratio:.2f} "
            f"regime={regime}@{regime_conf:.2f} "
            f"base_risk={base_risk_pct*100:.3f}% "
            f"eff_risk={effective_risk_pct*100:.3f}% (x{decision.size_multiplier:.1f}) "
            f"order={order_rec['order_type']} entry={decision.entry_price:.6f} "
            f"sl={decision.stop_loss:.6f} tp={decision.take_profit:.6f} "
            f"ev=${ev_usd:.4f} eval={elapsed:.1f}ms"
        )

        self._log_decision(decision, outputs)
        return decision

    # ── Multi-symbol evaluation (v5.1: preferred for frequency expansion) ─────

    def evaluate_multi(
        self,
        candidates_with_signals: List[Dict],
        current_balance: float,
        trades_today: int,
        open_trades: int,
        daily_pnl_pct: float,
        risk_manager=None,
    ) -> List[OrchestratorDecision]:
        """
        v5.1: Evaluate multiple symbols simultaneously and return up to N decisions.

        Each item in candidates_with_signals must have:
          {
            'candidate':     dict (from scanner),
            'signal':        'BUY' | 'SELL',
            'win_prob':      float,
            'signal_result': SignalResult (optional but preferred),
            'regime_confidence': float,
          }

        Process:
          1. Run all 11 gates per candidate (same as evaluate())
          2. Rank passing candidates by trade_priority
          3. Apply cluster/correlation check (TradeClusteringEngine)
          4. Return top N by available slots

        Correlation protection:
          - Max 1 trade per correlation cluster (BTC_CORRELATED group)
          - Max 2 same-direction trades simultaneously
          - No duplicate symbols

        Returns list of OrchestratorDecision (execute=True only for approved trades).
        """
        if not candidates_with_signals:
            return []

        # Determine available slots first (no point evaluating if full)
        consec_losses = risk_manager.consecutive_losses if risk_manager else 0
        conc_rem, daily_rem, _ = self.slot_engine.get_available_slots(
            current_balance, consec_losses, daily_pnl_pct, open_trades, trades_today
        )
        max_new_trades = min(conc_rem, daily_rem)
        if max_new_trades <= 0:
            logger.debug("Orchestrator.evaluate_multi: no slots available")
            return []

        # Evaluate all candidates through single-symbol path
        evaluated = []
        for item in candidates_with_signals:
            d = self.evaluate(
                candidate       = item['candidate'],
                signal          = item.get('signal', 'HOLD'),
                win_prob        = item.get('win_prob', 0.5),
                current_balance = current_balance,
                trades_today    = trades_today,
                open_trades     = open_trades,
                daily_pnl_pct   = daily_pnl_pct,
                risk_manager    = risk_manager,
                signal_result   = item.get('signal_result'),
            )
            if d.execute:
                evaluated.append({
                    'decision':   d,
                    'candidate':  item['candidate'],
                    'signal':     item.get('signal', 'BUY'),
                    'edge_score': d.edge_score,
                    'priority':   d.edge_score * (1.0 + (d.regime_confidence - 0.5) * 0.4),
                })

        if not evaluated:
            return []

        # Sort by priority descending
        evaluated.sort(key=lambda x: x['priority'], reverse=True)

        # Apply cluster/correlation filter and select top N
        selected: List[OrchestratorDecision] = []
        for item in evaluated:
            if len(selected) >= max_new_trades:
                break

            d   = item['decision']
            sym = d.symbol

            cluster_signal = {
                'symbol':           sym,
                'regime':           d.regime,
                'signal_direction': d.signal,
                'adx':              item['candidate'].get('adx', 0.0),
            }
            blocked, block_reason = self.cluster_engine.check_cluster(cluster_signal)
            if blocked:
                logger.debug(
                    f"Orchestrator.multi: {sym} blocked by cluster — {block_reason}"
                )
                d.execute     = False
                d.skip_reason = f"cluster_gate: {block_reason}"
                continue

            # Register the trade in cluster engine
            trade_id = f"{sym}_{int(time.time())}"
            self.cluster_engine.register_trade(trade_id, cluster_signal)
            selected.append(d)

            logger.info(
                f"Orchestrator.multi: selected {sym} "
                f"tier={d.setup_tier} edge={d.edge_score:.1f} "
                f"priority={item['priority']:.2f}"
            )

        return selected

    # ── Post-trade feedback ───────────────────────────────────────────────────

    def record_trade_result(
        self,
        symbol: str,
        pnl_usd: float,
        pnl_pct: float,
        side: str,
        risk_usd: float,
        hold_seconds: float = 0,
        trade_id: str = None,
    ) -> None:
        """Feed results back into trackers. Call after every trade closes."""
        self.perf_tracker.record_trade(symbol, pnl_usd, pnl_pct, side, hold_seconds)
        self.expectancy_eng.record(pnl_usd, risk_usd)
        if trade_id:
            self.cluster_engine.remove_trade(trade_id)

    def get_performance(self) -> Dict:
        return self.perf_tracker.get_summary()

    def is_in_review_mode(self) -> bool:
        return self.perf_tracker.is_in_review_mode()

    # ── Decision audit log ────────────────────────────────────────────────────

    def _log_decision(self, decision: OrchestratorDecision, outputs: Dict) -> None:
        entry = {
            'timestamp':         time.time(),
            'symbol':            decision.symbol,
            'signal':            decision.signal,
            'execute':           decision.execute,
            'reason':            decision.skip_reason if not decision.execute else 'EXECUTE',
            'edge':              decision.edge_score,
            'regime':            decision.regime,
            'setup_type':        decision.setup_type,
            'rr_ratio':          decision.rr_ratio,
            'entry_price':       decision.entry_price,
            'stop_loss':         decision.stop_loss,
            'take_profit':       decision.take_profit,
            'confidence':        decision.confidence,
            # v5.1
            'setup_tier':        decision.setup_tier,
            'size_multiplier':   decision.size_multiplier,
            'is_reentry':        decision.is_reentry,
            'eff_risk_pct':      decision.effective_risk_pct,
        }
        self._decision_log.append(entry)
        if len(self._decision_log) > 50:
            self._decision_log.pop(0)

    def get_decision_log(self) -> List[Dict]:
        return list(self._decision_log)
