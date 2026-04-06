"""
Performance Tracker — Phase 2
================================
In-memory performance tracking for the current session.

This complements (does NOT replace) trade_logger.py which handles
database persistence. This module is fast, in-memory, and used by
the orchestrator for real-time decision gating (e.g. "if expectancy < 0, pause").

Tracked metrics:
  - Win rate (rolling 20 trades)
  - Expectancy (EV per trade in $)
  - Sharpe ratio (annualised, rolling)
  - Max drawdown (session)
  - Consecutive losses (feeds back to risk manager)
  - Per-symbol breakdown

The orchestrator reads these metrics to apply adaptive gates:
  - win_rate < 0.35 for last 10 trades → reduce slot count
  - expectancy < 0 for last 10 trades  → force review mode
"""

import logging
import time
import numpy as np
from collections import deque
from typing import Dict, List, Optional

logger = logging.getLogger("PerformanceTracker")

ROLLING_WINDOW = 20   # trades for rolling metrics


class PerformanceTracker:
    """
    Single instance per bot run. Thread-safe via deque.
    """

    def __init__(self):
        self._trades: deque = deque(maxlen=ROLLING_WINDOW)
        self._all_pnl: List[float] = []       # full session history
        self._equity_curve: List[float] = []  # for max DD calculation
        self._start_time  = time.time()

    def record_trade(
        self,
        symbol: str,
        pnl_usd: float,
        pnl_pct: float,
        side: str,
        hold_seconds: float = 0,
    ) -> None:
        """Call after every trade close."""
        trade = {
            'symbol':       symbol,
            'pnl_usd':      pnl_usd,
            'pnl_pct':      pnl_pct,
            'side':         side,
            'hold_seconds': hold_seconds,
            'timestamp':    time.time(),
            'won':          pnl_usd > 0,
        }
        self._trades.append(trade)
        self._all_pnl.append(pnl_usd)

        # Running equity for drawdown
        equity = sum(self._all_pnl)
        self._equity_curve.append(equity)

        logger.debug(
            f"PerformanceTracker: {symbol} pnl=${pnl_usd:.4f} "
            f"({pnl_pct*100:.2f}%) {'WIN' if pnl_usd > 0 else 'LOSS'}"
        )

    def get_summary(self, window: int = None) -> Dict:
        """
        Returns rolling performance summary.
        window: override rolling window size (default: ROLLING_WINDOW)
        """
        trades = list(self._trades) if window is None else list(self._trades)[-window:]

        if not trades:
            return self._empty_summary()

        pnls    = [t['pnl_usd'] for t in trades]
        pcts    = [t['pnl_pct'] for t in trades]
        wins    = [p for p in pnls if p > 0]
        losses  = [p for p in pnls if p <= 0]

        win_rate   = len(wins) / len(pnls)
        avg_win    = np.mean(wins) if wins else 0.0
        avg_loss   = np.mean(losses) if losses else 0.0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Sharpe (annualised from 5m candle trades)
        sharpe = 0.0
        if len(pcts) > 2:
            mean_r = np.mean(pcts)
            std_r  = np.std(pcts) + 1e-9
            # Annualisation: ~288 5m candles/day, 365 days
            sharpe = float((mean_r / std_r) * np.sqrt(288 * 365))

        # Session max drawdown
        max_dd = 0.0
        if len(self._equity_curve) > 1:
            curve   = np.array(self._equity_curve)
            peak    = np.maximum.accumulate(curve)
            max_dd  = float(np.max(peak - curve))

        # Consecutive losses (tail of all trades)
        consec_losses = 0
        for t in reversed(list(self._trades)):
            if not t['won']:
                consec_losses += 1
            else:
                break

        return {
            'total_trades':     len(self._all_pnl),
            'rolling_trades':   len(trades),
            'win_rate':         round(win_rate, 4),
            'avg_win':          round(avg_win, 4),
            'avg_loss':         round(avg_loss, 4),
            'expectancy':       round(expectancy, 4),
            'sharpe':           round(sharpe, 3),
            'max_dd_usd':       round(max_dd, 4),
            'consecutive_losses': consec_losses,
            'session_pnl':      round(sum(self._all_pnl), 4),
            'uptime_hours':     round((time.time() - self._start_time) / 3600, 2),
        }

    def is_in_review_mode(self) -> bool:
        """
        Returns True if recent performance warrants pausing new trades.
        Used by orchestrator as an optional soft gate.
        """
        summary = self.get_summary(window=10)
        if summary['total_trades'] < 5:
            return False
        # Negative expectancy over last 10 trades = review
        return summary['expectancy'] < -0.001 and summary['win_rate'] < 0.30

    @staticmethod
    def _empty_summary() -> Dict:
        return {
            'total_trades': 0, 'rolling_trades': 0, 'win_rate': 0.0,
            'avg_win': 0.0, 'avg_loss': 0.0, 'expectancy': 0.0,
            'sharpe': 0.0, 'max_dd_usd': 0.0, 'consecutive_losses': 0,
            'session_pnl': 0.0, 'uptime_hours': 0.0,
        }
