"""
Strategy Health Failsafe — Phase 4
=====================================
Hard-stop mechanism that halts trading when the system is degrading
beyond recoverable thresholds, or when anomalous conditions are detected.

This is the LAST LINE OF DEFENSE. It operates independently of
StrategyHealthMonitor and can shut everything down unconditionally.

Failsafe triggers (ANY one → halt):
  - Consecutive losses >= HARD_STOP_CONSEC_LOSSES
  - Daily drawdown >= HARD_DAILY_DD_PCT
  - Session drawdown >= HARD_SESSION_DD_PCT
  - Win rate < HARD_MIN_WIN_RATE over last 20 trades (after 20+ trades)
  - Negative expectancy sustained for 25+ trades
  - Manual halt flag set

Recovery:
  - Automatic: after COOLDOWN_HOURS, system re-enables with caution flags
  - Manual: requires explicit reset() call

All halts are logged with full context for post-mortem analysis.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("StrategyFailsafe")

# Hard stop thresholds
HARD_STOP_CONSEC_LOSSES  = 8       # 8 consecutive losses → halt
HARD_DAILY_DD_PCT        = 0.05    # 5% daily drawdown → halt
HARD_SESSION_DD_PCT      = 0.08    # 8% session drawdown → halt
HARD_MIN_WIN_RATE        = 0.18    # below 18% win rate over 20 trades → halt
HARD_NEG_EV_TRADES       = 25      # 25 consecutive negative EV trades → halt

# Auto-recovery
COOLDOWN_HOURS           = 2.0     # auto-resume after 2 hours
MIN_BALANCE_TO_TRADE     = 3.0     # never trade below $3


class StrategyFailsafe:
    """
    Usage:
        failsafe = StrategyFailsafe(initial_balance=10.0)

        # Before EVERY trade:
        allowed, reason = failsafe.check(perf_summary, current_balance)
        if not allowed:
            return  # abort trade

        # After daily reset:
        failsafe.daily_reset(current_balance)
    """

    def __init__(self, initial_balance: float):
        self._initial_balance   = initial_balance
        self._session_start_bal = initial_balance
        self._day_start_bal     = initial_balance

        self._halted            = False
        self._halt_reason       = ""
        self._halt_time         = 0.0
        self._manual_halt       = False
        self._halt_log: List[Dict] = []

        self._neg_ev_streak     = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def check(
        self,
        perf_summary: Dict,
        current_balance: float,
    ) -> Tuple[bool, str]:
        """
        Returns (trading_allowed, reason).
        Call before every new trade entry.
        """
        # Manual halt always blocks
        if self._manual_halt:
            return False, "Manual halt active — call failsafe.resume() to re-enable"

        # Auto-recovery check
        if self._halted:
            recovered = self._check_auto_recovery()
            if not recovered:
                elapsed = (time.time() - self._halt_time) / 3600
                remaining = max(0, COOLDOWN_HOURS - elapsed)
                return False, (
                    f"Failsafe halt: {self._halt_reason} | "
                    f"auto-resume in {remaining:.1f}h"
                )

        # Minimum balance check
        if current_balance < MIN_BALANCE_TO_TRADE:
            self._trigger_halt(
                f"Balance ${current_balance:.2f} < minimum ${MIN_BALANCE_TO_TRADE}",
                perf_summary, current_balance
            )
            return False, self._halt_reason

        # ── Hard-stop checks ──────────────────────────────────────────────────
        total_trades = perf_summary.get('total_trades', 0)
        consec_loss  = perf_summary.get('consecutive_losses', 0)
        win_rate     = perf_summary.get('win_rate', 0.5)
        expectancy   = perf_summary.get('expectancy', 0.0)

        # 1. Consecutive losses
        if consec_loss >= HARD_STOP_CONSEC_LOSSES:
            self._trigger_halt(
                f"{consec_loss} consecutive losses (limit={HARD_STOP_CONSEC_LOSSES})",
                perf_summary, current_balance
            )
            return False, self._halt_reason

        # 2. Daily drawdown
        if self._day_start_bal > 0:
            daily_dd = (self._day_start_bal - current_balance) / self._day_start_bal
            if daily_dd >= HARD_DAILY_DD_PCT:
                self._trigger_halt(
                    f"Daily drawdown {daily_dd*100:.2f}% >= {HARD_DAILY_DD_PCT*100}%",
                    perf_summary, current_balance
                )
                return False, self._halt_reason

        # 3. Session drawdown
        if self._session_start_bal > 0:
            session_dd = (self._session_start_bal - current_balance) / self._session_start_bal
            if session_dd >= HARD_SESSION_DD_PCT:
                self._trigger_halt(
                    f"Session drawdown {session_dd*100:.2f}% >= {HARD_SESSION_DD_PCT*100}%",
                    perf_summary, current_balance
                )
                return False, self._halt_reason

        # 4. Win rate collapse (only after sufficient trades)
        if total_trades >= 20 and win_rate < HARD_MIN_WIN_RATE:
            self._trigger_halt(
                f"Win rate {win_rate*100:.1f}% < {HARD_MIN_WIN_RATE*100}% over {total_trades} trades",
                perf_summary, current_balance
            )
            return False, self._halt_reason

        # 5. Negative EV streak
        if expectancy < 0:
            self._neg_ev_streak += 1
        else:
            self._neg_ev_streak = 0

        if self._neg_ev_streak >= HARD_NEG_EV_TRADES:
            self._trigger_halt(
                f"Negative expectancy for {self._neg_ev_streak} consecutive trades",
                perf_summary, current_balance
            )
            return False, self._halt_reason

        return True, ""

    def manual_halt(self, reason: str = "Manual halt") -> None:
        """Immediately halt all trading (call from Telegram command or admin)."""
        self._manual_halt  = True
        self._halt_reason  = reason
        self._halt_time    = time.time()
        logger.critical(f"Failsafe: MANUAL HALT — {reason}")
        self._halt_log.append({
            'type': 'manual', 'reason': reason,
            'timestamp': time.time(),
        })

    def resume(self) -> None:
        """Re-enable trading after manual halt or forced recovery."""
        self._manual_halt = False
        self._halted      = False
        self._halt_reason = ""
        self._neg_ev_streak = 0
        logger.warning("Failsafe: trading RESUMED")

    def daily_reset(self, current_balance: float) -> None:
        """Call at session/daily reset to update reference balance."""
        self._day_start_bal     = current_balance
        self._session_start_bal = current_balance
        self._neg_ev_streak     = 0
        # Do NOT auto-clear halts on daily reset — requires explicit resume()
        logger.info(f"Failsafe: daily reset at ${current_balance:.2f}")

    def is_halted(self) -> bool:
        return self._halted or self._manual_halt

    def get_halt_log(self) -> List[Dict]:
        return list(self._halt_log)

    def get_status(self, current_balance: float) -> Dict:
        elapsed = (time.time() - self._halt_time) / 3600 if self._halt_time else 0
        remaining = max(0, COOLDOWN_HOURS - elapsed) if self._halted else 0
        daily_dd = 0.0
        if self._day_start_bal > 0:
            daily_dd = (self._day_start_bal - current_balance) / self._day_start_bal

        return {
            'halted':            self._halted or self._manual_halt,
            'manual_halt':       self._manual_halt,
            'halt_reason':       self._halt_reason,
            'cooldown_remaining_h': round(remaining, 2),
            'neg_ev_streak':     self._neg_ev_streak,
            'daily_dd_pct':      round(daily_dd * 100, 2),
            'halt_count':        len(self._halt_log),
        }

    # ── Internal helpers ────────────────────────────────────────────────────

    def _trigger_halt(
        self,
        reason: str,
        perf_summary: Dict,
        balance: float,
    ) -> None:
        self._halted      = True
        self._halt_reason = reason
        self._halt_time   = time.time()

        entry = {
            'type':      'auto',
            'reason':    reason,
            'timestamp': time.time(),
            'balance':   round(balance, 4),
            'perf':      {k: perf_summary.get(k) for k in
                          ('win_rate', 'consecutive_losses', 'expectancy', 'total_trades')},
        }
        self._halt_log.append(entry)
        if len(self._halt_log) > 50:
            self._halt_log = self._halt_log[-50:]

        logger.critical(
            f"Failsafe: HALT TRIGGERED — {reason} | "
            f"balance=${balance:.2f} | "
            f"auto-resume in {COOLDOWN_HOURS}h"
        )

    def _check_auto_recovery(self) -> bool:
        """Returns True if cooldown period has elapsed and auto-recovery applies."""
        if not self._halted or self._manual_halt:
            return not self._halted

        elapsed = (time.time() - self._halt_time) / 3600
        if elapsed >= COOLDOWN_HOURS:
            logger.warning(
                f"Failsafe: auto-recovery after {elapsed:.1f}h — "
                f"resuming with caution"
            )
            self._halted      = False
            self._halt_reason = ""
            return True
        return False
