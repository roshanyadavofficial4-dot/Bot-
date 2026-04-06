"""
Strategy Health Monitor — Phase 3
====================================
Detects strategy degradation in real-time and triggers defensive/review modes.

Inputs:  PerformanceTracker summary (rolling metrics)
Outputs: health_status, defensive_mode flag, severity level

Health States:
  HEALTHY     → normal trading, no restrictions
  CAUTION     → reduce risk slightly, tighten filters
  DEFENSIVE   → cut position size by 50%, require higher edge scores
  CRITICAL    → halt new trades, await manual review or recovery

Degradation Signals:
  - Win rate drop below rolling baseline
  - 3+ consecutive losses
  - Expectancy negative over last 15 trades
  - Drawdown exceeding session threshold
  - Rapid equity decline (velocity check)
"""

import logging
import time
from typing import Dict, Tuple
from collections import deque

logger = logging.getLogger("StrategyHealthMonitor")

# Health thresholds
WIN_RATE_CAUTION     = 0.40   # below → CAUTION
WIN_RATE_DEFENSIVE   = 0.30   # below → DEFENSIVE
WIN_RATE_CRITICAL    = 0.20   # below → CRITICAL (over last 10 trades)

CONSEC_LOSS_CAUTION  = 3
CONSEC_LOSS_DEFENSIVE = 5
CONSEC_LOSS_CRITICAL  = 7

DD_CAUTION           = 0.02   # 2% drawdown → CAUTION
DD_DEFENSIVE         = 0.04   # 4% drawdown → DEFENSIVE
DD_CRITICAL          = 0.06   # 6% drawdown → CRITICAL

# Recovery thresholds — must exceed these to exit defensive mode
RECOVERY_WIN_RATE    = 0.50
RECOVERY_CONSEC_WINS = 3
RECOVERY_MIN_TRADES  = 5


class StrategyHealthMonitor:
    """
    Single instance. Called after every trade close with current perf summary.

    Usage:
        monitor = StrategyHealthMonitor()
        # After each trade:
        status, meta = monitor.evaluate(perf_summary)
        if meta['defensive_mode']:
            # apply restrictions
    """

    def __init__(self):
        self._current_state   = "HEALTHY"
        self._state_since     = time.time()
        self._consecutive_wins = 0
        self._equity_snapshots = deque(maxlen=20)  # for velocity check
        self._last_equity      = 0.0
        self._defensive_since  = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(self, perf_summary: Dict) -> Tuple[str, Dict]:
        """
        Evaluate current health given a PerformanceTracker summary dict.

        Returns:
            state   str   — 'HEALTHY' | 'CAUTION' | 'DEFENSIVE' | 'CRITICAL'
            meta    dict  — flags and recommended adjustments
        """
        win_rate     = perf_summary.get('win_rate', 0.5)
        consec_loss  = perf_summary.get('consecutive_losses', 0)
        max_dd       = perf_summary.get('max_dd_usd', 0.0)
        expectancy   = perf_summary.get('expectancy', 0.0)
        total_trades = perf_summary.get('total_trades', 0)
        session_pnl  = perf_summary.get('session_pnl', 0.0)

        # Track equity velocity
        self._equity_snapshots.append(session_pnl)
        equity_velocity = self._compute_velocity()

        # ── Determine raw severity ──────────────────────────────────────────
        severity = self._compute_severity(
            win_rate, consec_loss, max_dd, expectancy, total_trades, equity_velocity
        )

        # ── Apply recovery logic (prevent flip-flopping) ────────────────────
        if self._current_state in ("DEFENSIVE", "CRITICAL"):
            if not self._has_recovered(win_rate, perf_summary):
                # Stay in current state — don't downgrade prematurely
                severity = max(severity, 2)  # CAUTION at minimum

        # ── Update state ────────────────────────────────────────────────────
        new_state = self._severity_to_state(severity)
        if new_state != self._current_state:
            logger.warning(
                f"StrategyHealth: {self._current_state} → {new_state} "
                f"(wr={win_rate:.2f} cl={consec_loss} dd={max_dd:.4f} ev={expectancy:.4f})"
            )
            self._current_state = new_state
            self._state_since   = time.time()
            if new_state in ("DEFENSIVE", "CRITICAL"):
                self._defensive_since = time.time()

        meta = self._build_meta(severity, win_rate, consec_loss, max_dd, equity_velocity)

        logger.debug(
            f"StrategyHealth: {self._current_state} | sev={severity} "
            f"wr={win_rate:.2f} cl={consec_loss}"
        )
        return self._current_state, meta

    def get_state(self) -> str:
        return self._current_state

    def is_defensive(self) -> bool:
        return self._current_state in ("DEFENSIVE", "CRITICAL")

    def time_in_current_state(self) -> float:
        """Seconds since last state transition."""
        return time.time() - self._state_since

    def notify_win(self):
        """Call when a trade wins — tracks consecutive wins for recovery."""
        self._consecutive_wins += 1

    def notify_loss(self):
        """Call when a trade loses — resets consecutive win counter."""
        self._consecutive_wins = 0

    # ── Internal helpers ────────────────────────────────────────────────────

    def _compute_severity(
        self,
        win_rate: float,
        consec_loss: int,
        max_dd: float,
        expectancy: float,
        total_trades: int,
        velocity: float,
    ) -> int:
        """
        Returns severity: 0=HEALTHY, 1=CAUTION, 2=DEFENSIVE, 3=CRITICAL
        Uses max-severity logic: any single critical signal → severity 3.
        """
        if total_trades < 5:
            return 0  # insufficient data to judge

        sev = 0

        # Win rate degradation
        if win_rate < WIN_RATE_CRITICAL and total_trades >= 10:
            sev = max(sev, 3)
        elif win_rate < WIN_RATE_DEFENSIVE:
            sev = max(sev, 2)
        elif win_rate < WIN_RATE_CAUTION:
            sev = max(sev, 1)

        # Consecutive losses
        if consec_loss >= CONSEC_LOSS_CRITICAL:
            sev = max(sev, 3)
        elif consec_loss >= CONSEC_LOSS_DEFENSIVE:
            sev = max(sev, 2)
        elif consec_loss >= CONSEC_LOSS_CAUTION:
            sev = max(sev, 1)

        # Drawdown
        if max_dd > DD_CRITICAL:
            sev = max(sev, 3)
        elif max_dd > DD_DEFENSIVE:
            sev = max(sev, 2)
        elif max_dd > DD_CAUTION:
            sev = max(sev, 1)

        # Negative expectancy
        if expectancy < -0.005 and total_trades >= 15:
            sev = max(sev, 2)
        elif expectancy < 0 and total_trades >= 10:
            sev = max(sev, 1)

        # Rapid equity decline (velocity)
        if velocity < -0.003:   # losing >0.3% of session equity per trade rapidly
            sev = max(sev, 2)

        return sev

    def _compute_velocity(self) -> float:
        """
        Rate of equity change over recent snapshots.
        Negative = equity declining rapidly.
        """
        snaps = list(self._equity_snapshots)
        if len(snaps) < 4:
            return 0.0
        # Linear slope approximation over last 4 snapshots
        recent   = snaps[-4:]
        slope    = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
        return slope

    def _has_recovered(self, win_rate: float, perf: Dict) -> bool:
        """True if recovery conditions are met to exit defensive mode."""
        if win_rate < RECOVERY_WIN_RATE:
            return False
        if self._consecutive_wins < RECOVERY_CONSEC_WINS:
            return False
        if perf.get('total_trades', 0) < RECOVERY_MIN_TRADES:
            return False
        return True

    def _severity_to_state(self, severity: int) -> str:
        return {0: "HEALTHY", 1: "CAUTION", 2: "DEFENSIVE", 3: "CRITICAL"}.get(severity, "HEALTHY")

    def _build_meta(
        self,
        severity: int,
        win_rate: float,
        consec_loss: int,
        max_dd: float,
        velocity: float,
    ) -> Dict:
        """Build adjustment recommendations for orchestrator."""
        meta = {
            'severity':          severity,
            'defensive_mode':    severity >= 2,
            'halt_trading':      severity >= 3,
            'risk_multiplier':   1.0,
            'min_edge_score':    40,  # base minimum
            'description':       self._current_state,
            'equity_velocity':   round(velocity, 6),
            'consecutive_wins':  self._consecutive_wins,
        }

        if severity == 1:  # CAUTION
            meta['risk_multiplier'] = 0.75
            meta['min_edge_score']  = 50

        elif severity == 2:  # DEFENSIVE
            meta['risk_multiplier'] = 0.50
            meta['min_edge_score']  = 60

        elif severity == 3:  # CRITICAL
            meta['risk_multiplier'] = 0.25
            meta['min_edge_score']  = 70

        return meta
