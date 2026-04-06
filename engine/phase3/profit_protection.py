"""
Profit Protection Engine — Phase 3
=====================================
Locks in profits and tightens risk management after profitable periods.

Principle: money earned is harder to earn back than it is to lose.
Once profit is banked, protect it aggressively.

Mechanisms:
  1. Daily profit lock  — once X% daily profit is made, reduce risk
  2. Session peak lock  — tighten stops as session PnL grows
  3. Trailing stop adjustment — tighter trail after significant moves
  4. Position reduction — partial exit after reaching profit milestone
  5. Cooling period     — after a big win, wait N minutes before next trade

Integration:
  - Called by orchestrator and risk_manager
  - Outputs: risk_multiplier, min_rr_required, recommended_stop_tightening
"""

import logging
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger("ProfitProtectionEngine")

# Protection thresholds (% of session start balance)
PROTECTION_LEVELS = [
    # (profit_pct_threshold, risk_multiplier, description)
    (0.01, 0.90, "1%+ profit → slight risk reduction"),
    (0.02, 0.75, "2%+ profit → moderate protection"),
    (0.03, 0.60, "3%+ profit → strong protection"),
    (0.05, 0.40, "5%+ profit → aggressive lock"),
    (0.08, 0.25, "8%+ profit → maximum protection"),
]

# After a single big win, wait this many seconds before next trade
BIG_WIN_COOLDOWN_SEC    = 300    # 5 min cooldown after a 1%+ single trade win
BIG_WIN_THRESHOLD_PCT   = 0.01   # 1% single-trade win triggers cooldown

# Trailing stop tightening multipliers
TRAIL_TIGHTEN_NORMAL  = 1.0
TRAIL_TIGHTEN_PROTECT = 0.75   # tighten by 25% when in profit protection mode
TRAIL_TIGHTEN_MAX     = 0.60   # maximum tightening

# Minimum RR required when in protection mode (raise the bar)
MIN_RR_PROTECTION     = 1.8    # require at least 1.8:1 when protecting profits


class ProfitProtectionEngine:
    """
    Usage:
        engine = ProfitProtectionEngine()

        # After each trade:
        engine.record_trade(pnl_pct, session_start_balance, current_balance)

        # Before each trade:
        params = engine.get_protection_params(session_pnl_pct)
        apply params['risk_multiplier'] to position sizing
    """

    def __init__(self):
        self._session_start_balance = None
        self._peak_session_pnl_pct  = 0.0
        self._last_big_win_time     = 0.0
        self._trades_since_peak     = 0
        self._protection_level      = 0    # 0 = none, higher = more protection
        self._session_pnl_history   = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_session_start(self, balance: float) -> None:
        """Call at the start of each trading session."""
        self._session_start_balance = balance
        self._peak_session_pnl_pct  = 0.0
        self._trades_since_peak     = 0
        self._protection_level      = 0
        self._session_pnl_history   = []
        logger.info(f"ProfitProtection: session started at ${balance:.2f}")

    def record_trade(self, pnl_pct: float, current_balance: float) -> None:
        """Record a completed trade result."""
        self._session_pnl_history.append(pnl_pct)

        if self._session_start_balance and self._session_start_balance > 0:
            session_return = (current_balance - self._session_start_balance) / \
                             self._session_start_balance
            if session_return > self._peak_session_pnl_pct:
                self._peak_session_pnl_pct = session_return

        # Big win cooldown trigger
        if abs(pnl_pct) >= BIG_WIN_THRESHOLD_PCT and pnl_pct > 0:
            self._last_big_win_time = time.time()
            logger.info(
                f"ProfitProtection: big win {pnl_pct*100:.2f}% → "
                f"{BIG_WIN_COOLDOWN_SEC}s cooldown"
            )

    def get_protection_params(
        self,
        session_pnl_pct: float,
        current_rr: Optional[float] = None,
    ) -> Dict:
        """
        Returns protection parameters for the current trade decision.

        session_pnl_pct: current session return as fraction (0.03 = 3%)
        current_rr:      proposed trade R:R ratio

        Returns dict with:
            risk_multiplier       float
            trail_tighten         float
            min_rr_required       float
            in_cooldown           bool
            cooldown_remaining    float
            protection_level      str
        """
        # Determine protection tier
        risk_mult     = 1.0
        level_label   = "none"

        for threshold, mult, label in reversed(PROTECTION_LEVELS):
            if session_pnl_pct >= threshold:
                risk_mult   = mult
                level_label = label
                break

        # Drawdown from peak (gives back profits → trigger extra protection)
        drawdown_from_peak = self._peak_session_pnl_pct - session_pnl_pct
        if drawdown_from_peak > 0.01 and session_pnl_pct > 0:
            # Giving back >1% from peak → tighten further
            risk_mult = max(risk_mult * 0.85, 0.25)
            level_label += " +drawback_protection"

        # Trailing stop tighten
        if session_pnl_pct >= 0.03:
            trail_tighten = TRAIL_TIGHTEN_MAX
        elif session_pnl_pct >= 0.015:
            trail_tighten = TRAIL_TIGHTEN_PROTECT
        else:
            trail_tighten = TRAIL_TIGHTEN_NORMAL

        # Minimum RR requirement
        min_rr = 1.5   # base minimum
        if session_pnl_pct >= 0.02:
            min_rr = MIN_RR_PROTECTION

        # Big win cooldown
        elapsed        = time.time() - self._last_big_win_time
        in_cooldown    = elapsed < BIG_WIN_COOLDOWN_SEC and self._last_big_win_time > 0
        cooldown_rem   = max(0.0, BIG_WIN_COOLDOWN_SEC - elapsed) if in_cooldown else 0.0

        if in_cooldown:
            logger.debug(
                f"ProfitProtection: in cooldown — {cooldown_rem:.0f}s remaining"
            )

        logger.debug(
            f"ProfitProtection: pnl={session_pnl_pct*100:.2f}% "
            f"risk_mult={risk_mult:.2f} trail={trail_tighten:.2f} "
            f"cooldown={in_cooldown}"
        )

        return {
            'risk_multiplier':    round(risk_mult, 4),
            'trail_tighten':      round(trail_tighten, 4),
            'min_rr_required':    round(min_rr, 2),
            'in_cooldown':        in_cooldown,
            'cooldown_remaining': round(cooldown_rem, 1),
            'protection_level':   level_label,
            'peak_session_pnl':   round(self._peak_session_pnl_pct, 4),
        }

    def get_adjusted_stop(self, original_stop_pct: float, session_pnl_pct: float) -> float:
        """
        Returns a tightened stop loss percentage based on current protection level.
        Never wider than original_stop_pct.
        """
        params = self.get_protection_params(session_pnl_pct)
        tighten = params['trail_tighten']
        return round(original_stop_pct * tighten, 5)

    def should_partial_exit(self, session_pnl_pct: float, trade_pnl_pct: float) -> Tuple[bool, float]:
        """
        Suggest partial exit when both session and trade profits are significant.

        Returns:
            should_exit  bool
            exit_fraction  float — fraction of position to close (e.g. 0.5)
        """
        if session_pnl_pct >= 0.03 and trade_pnl_pct >= 0.015:
            return True, 0.50   # take off half at 3% session / 1.5% trade gain
        elif session_pnl_pct >= 0.02 and trade_pnl_pct >= 0.020:
            return True, 0.50
        return False, 0.0

    def reset_session(self, new_balance: float) -> None:
        """Call at session end / daily reset."""
        self.set_session_start(new_balance)
