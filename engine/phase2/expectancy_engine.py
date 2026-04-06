"""
Expectancy Engine — Phase 2
==============================
Calculates and tracks the statistical expectancy of the trading strategy.

Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)

A positive expectancy means the strategy makes money on average per trade.
This engine:
  1. Tracks live expectancy from closed trades (rolling window)
  2. Computes expected value for a PROPOSED trade given current parameters
  3. Provides a go/no-go gate: skip if expected value < minimum threshold

Used by orchestrator BEFORE submitting a trade to confirm EV is positive.
"""

import logging
from typing import Dict, List, Tuple

from config import FEE_RATE, SLIPPAGE_BUFFER

logger = logging.getLogger("ExpectancyEngine")

# Minimum expected value as a fraction of risked amount to take a trade
MIN_EV_RATIO = 0.10    # expected value must be >= 10% of amount risked

# Rolling window for live strategy expectancy
EXPECTANCY_WINDOW = 30


class ExpectancyEngine:
    """
    Usage:
        engine = ExpectancyEngine()
        # After each trade:
        engine.record(pnl_usd, risk_usd)
        # Before each trade:
        ev, viable = engine.evaluate_proposed(rr_ratio, win_prob, risk_usd)
    """

    def __init__(self):
        self._history: List[Dict] = []   # list of {pnl, risk}

    def record(self, pnl_usd: float, risk_usd: float) -> None:
        """Record a completed trade for rolling expectancy tracking."""
        self._history.append({'pnl': pnl_usd, 'risk': max(risk_usd, 1e-9)})
        if len(self._history) > EXPECTANCY_WINDOW:
            self._history.pop(0)

    def get_strategy_expectancy(self) -> Dict:
        """
        Returns current rolling strategy expectancy.

        expectancy_ratio: EV / avg_risk (dimensionless — compares strategies)
        expectancy_usd:   average $ EV per trade
        """
        if len(self._history) < 5:
            return {
                'status': 'insufficient_data',
                'expectancy_usd': 0.0,
                'expectancy_ratio': 0.0,
                'win_rate': 0.0,
                'trades': len(self._history),
            }

        pnls   = [h['pnl'] for h in self._history]
        risks  = [h['risk'] for h in self._history]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate      = len(wins) / len(pnls)
        avg_win       = sum(wins) / len(wins) if wins else 0.0
        avg_loss      = sum(losses) / len(losses) if losses else 0.0
        expectancy    = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        avg_risk      = sum(risks) / len(risks)
        ev_ratio      = expectancy / avg_risk if avg_risk > 0 else 0.0

        return {
            'status':           'ok',
            'expectancy_usd':   round(expectancy, 4),
            'expectancy_ratio': round(ev_ratio, 4),
            'win_rate':         round(win_rate, 4),
            'avg_win_usd':      round(avg_win, 4),
            'avg_loss_usd':     round(avg_loss, 4),
            'trades':           len(pnls),
        }

    def evaluate_proposed(
        self,
        rr_ratio: float,
        win_prob: float,
        risk_usd: float,
    ) -> Tuple[float, bool, str]:
        """
        Evaluate the EV of a proposed trade BEFORE execution.

        rr_ratio:  reward-to-risk ratio (e.g. 2.0 = TP is 2× the SL distance)
        win_prob:  ML-estimated probability of win (0.0–1.0)
        risk_usd:  dollar amount being risked

        Returns:
            ev_usd     float  — expected value of this trade in $
            viable     bool   — True if EV meets minimum threshold
            reason     str
        """
        # Round-trip cost in $ terms
        rt_cost = (FEE_RATE * 2 + SLIPPAGE_BUFFER) * risk_usd * rr_ratio

        gross_ev = (win_prob * risk_usd * rr_ratio) - ((1 - win_prob) * risk_usd)
        net_ev   = gross_ev - rt_cost

        min_ev   = risk_usd * MIN_EV_RATIO
        viable   = net_ev >= min_ev

        reason = (
            f"ev=${net_ev:.4f} (gross={gross_ev:.4f} cost={rt_cost:.4f}) "
            f"min=${min_ev:.4f} {'✓' if viable else '✗'}"
        )

        logger.debug(
            f"ExpectancyEngine: rr={rr_ratio:.2f} win_prob={win_prob:.3f} "
            f"risk=${risk_usd:.4f} → {reason}"
        )

        return round(net_ev, 6), viable, reason
