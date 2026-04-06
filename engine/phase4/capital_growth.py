"""
Capital Growth Manager — Phase 4
===================================
Manages capital allocation between reinvestment and withdrawal,
tracking the overall health and growth trajectory of the account.

Purpose:
  - Track running capital growth milestones
  - Apply compounding reinvestment rules
  - Recommend withdrawal amounts at defined thresholds
  - Adjust risk tiers as capital grows (micro → small → mid capital)

Capital Tiers:
  MICRO      < $50      — survival mode, min risk, max protection
  SEED       $50–$200   — careful growth, 1% risk max
  GROWING    $200–$1000 — normal operation, 1.5% risk
  STABLE     $1000+     — full operation, compound properly

Reinvestment Policy (configurable):
  Default: 70% reinvest, 30% withdraw when milestone hit.
  Milestones: every +25% gain from last withdrawal point.
"""

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger("CapitalGrowthManager")

# Capital tier definitions
TIERS = [
    {'name': 'MICRO',   'min': 0,     'max': 50,   'risk_pct': 0.010, 'max_risk': 0.015},
    {'name': 'SEED',    'min': 50,    'max': 200,  'risk_pct': 0.012, 'max_risk': 0.018},
    {'name': 'GROWING', 'min': 200,   'max': 1000, 'risk_pct': 0.015, 'max_risk': 0.020},
    {'name': 'STABLE',  'min': 1000,  'max': 1e9,  'risk_pct': 0.018, 'max_risk': 0.022},
]

# Withdrawal policy
REINVEST_FRACTION    = 0.70   # 70% stays in account
WITHDRAW_FRACTION    = 0.30   # 30% extracted as income
MILESTONE_GAIN_PCT   = 0.25   # trigger withdrawal at every +25% gain from anchor

# Minimum withdrawal (not worth extracting less than this)
MIN_WITHDRAWAL_USD   = 5.0


class CapitalGrowthManager:
    """
    Usage:
        manager = CapitalGrowthManager(initial_balance=10.0)

        # Each scan cycle:
        tier_params = manager.get_tier_params(current_balance)

        # Check if withdrawal milestone reached:
        should, amount = manager.check_withdrawal_milestone(current_balance)
        if should:
            manager.record_withdrawal(amount)
    """

    def __init__(self, initial_balance: float):
        self._initial_balance    = initial_balance
        self._anchor_balance     = initial_balance   # last withdrawal point
        self._peak_balance       = initial_balance
        self._total_withdrawn    = 0.0
        self._withdrawal_history = []
        self._growth_log         = []

        logger.info(
            f"CapitalGrowthManager: initialized at ${initial_balance:.2f}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_tier_params(self, current_balance: float) -> Dict:
        """
        Returns risk parameters appropriate for current capital tier.

        Returns:
            tier_name        str
            risk_pct         float   — recommended risk per trade
            max_risk_pct     float   — absolute ceiling
            leverage_max     int
            compounding_ok   bool    — True if large enough to compound properly
            growth_pct       float   — total % gain from initial
        """
        tier = self._get_tier(current_balance)

        # Update peak
        if current_balance > self._peak_balance:
            self._peak_balance = current_balance

        growth_pct = (current_balance - self._initial_balance) / \
                     max(self._initial_balance, 1e-9)

        # Dynamic leverage: reduce at extremes
        leverage = 3
        if tier['name'] == 'MICRO':
            leverage = 2   # extra cautious with micro capital

        params = {
            'tier_name':       tier['name'],
            'risk_pct':        tier['risk_pct'],
            'max_risk_pct':    tier['max_risk'],
            'leverage_max':    leverage,
            'compounding_ok':  current_balance >= 50.0,
            'growth_pct':      round(growth_pct, 4),
            'peak_balance':    round(self._peak_balance, 4),
            'total_withdrawn': round(self._total_withdrawn, 4),
            'net_growth':      round(
                current_balance + self._total_withdrawn - self._initial_balance, 4
            ),
        }

        logger.debug(
            f"CapitalGrowth: tier={tier['name']} "
            f"balance=${current_balance:.2f} "
            f"growth={growth_pct*100:.1f}%"
        )
        return params

    def check_withdrawal_milestone(
        self, current_balance: float
    ) -> Tuple[bool, float]:
        """
        Check if current balance has exceeded withdrawal milestone threshold.

        Returns:
            milestone_hit  bool
            suggested_withdrawal  float  — amount to withdraw
        """
        gain_from_anchor = current_balance - self._anchor_balance

        if gain_from_anchor <= 0:
            return False, 0.0

        gain_pct = gain_from_anchor / max(self._anchor_balance, 1e-9)

        if gain_pct < MILESTONE_GAIN_PCT:
            return False, 0.0

        withdraw_amount = gain_from_anchor * WITHDRAW_FRACTION

        if withdraw_amount < MIN_WITHDRAWAL_USD:
            logger.debug(
                f"CapitalGrowth: milestone hit but withdrawal "
                f"${withdraw_amount:.2f} < min ${MIN_WITHDRAWAL_USD}"
            )
            return False, 0.0

        logger.info(
            f"CapitalGrowth: withdrawal milestone! "
            f"gain={gain_pct*100:.1f}% from ${self._anchor_balance:.2f} → "
            f"suggest withdraw ${withdraw_amount:.2f}"
        )
        return True, round(withdraw_amount, 2)

    def record_withdrawal(self, amount: float, current_balance: float) -> Dict:
        """
        Record a withdrawal event and reset the anchor balance.

        Returns new anchor and reinvested amount.
        """
        self._total_withdrawn += amount
        new_anchor = current_balance - amount
        self._anchor_balance  = new_anchor

        event = {
            'amount':      round(amount, 4),
            'balance_before': round(current_balance, 4),
            'balance_after':  round(new_anchor, 4),
            'total_withdrawn': round(self._total_withdrawn, 4),
            'anchor_reset':    round(new_anchor, 4),
        }
        self._withdrawal_history.append(event)

        logger.info(
            f"CapitalGrowth: withdrew ${amount:.2f} | "
            f"new balance=${new_anchor:.2f} | "
            f"total withdrawn=${self._total_withdrawn:.2f}"
        )
        return event

    def get_growth_summary(self, current_balance: float) -> Dict:
        """Full growth summary for dashboard / Telegram reporting."""
        tier    = self._get_tier(current_balance)
        net_pnl = current_balance + self._total_withdrawn - self._initial_balance

        milestone_hit, suggested_wd = self.check_withdrawal_milestone(current_balance)

        return {
            'initial_balance':  round(self._initial_balance, 4),
            'current_balance':  round(current_balance, 4),
            'peak_balance':     round(self._peak_balance, 4),
            'total_withdrawn':  round(self._total_withdrawn, 4),
            'net_profit':       round(net_pnl, 4),
            'net_return_pct':   round(net_pnl / max(self._initial_balance, 1e-9) * 100, 2),
            'tier':             tier['name'],
            'withdrawal_count': len(self._withdrawal_history),
            'milestone_due':    milestone_hit,
            'suggested_withdrawal': round(suggested_wd, 2) if milestone_hit else 0.0,
        }

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _get_tier(balance: float) -> Dict:
        for tier in reversed(TIERS):
            if balance >= tier['min']:
                return tier
        return TIERS[0]
