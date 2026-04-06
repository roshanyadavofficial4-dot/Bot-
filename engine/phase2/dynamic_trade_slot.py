"""
Dynamic Trade Slot Engine — Phase 2
======================================
Decides how many concurrent trades are permitted at any moment.

This consolidates the logic that was previously scattered across:
  - config.py MAX_TRADES_PER_DAY
  - capital_adaptive.py trade_limit
  - main.py min(adaptive_limit, config_cap)

Single source of truth for slot decisions. Three factors:
  1. Capital tier (from capital_adaptive)
  2. Risk manager state (consecutive losses = fewer slots)
  3. Current daily PnL (losing day = reduce slots as protection)

Slot map by capital tier:
  TINY   ($0–$6)    → max 1 concurrent, 1 per day
  SMALL  ($6–$12)   → max 1 concurrent, 2 per day
  MICRO  ($12–$24)  → max 1 concurrent, 2 per day
  BASE   ($24–$60)  → max 2 concurrent, 2 per day
  SCALE  ($60+)     → max 2 concurrent, 2 per day (config cap)

Note: config.py MAX_TRADES_PER_DAY=2 is the hard daily cap.
This engine may REDUCE but never INCREASE beyond that cap.
"""

import logging
import math
from typing import Tuple

from config import MAX_TRADES_PER_DAY
from engine.capital_adaptive import get_adaptive_params

logger = logging.getLogger("DynamicTradeSlot")


class DynamicTradeSlotEngine:
    """
    Usage:
        slots = DynamicTradeSlotEngine()
        concurrent, daily = slots.get_available_slots(
            current_balance, consecutive_losses, daily_pnl_pct,
            current_open_trades, trades_today
        )
    """

    def get_available_slots(
        self,
        current_balance: float,
        consecutive_losses: int,
        daily_pnl_pct: float,
        current_open_trades: int,
        trades_today: int,
    ) -> Tuple[int, int, str]:
        """
        Returns:
            concurrent_slots_remaining  int  — how many MORE trades can open now
            daily_slots_remaining       int  — how many more trades allowed today
            reason                      str  — explanation for logging

        concurrent_slots_remaining = 0 means: don't open any new trade.
        """
        ap          = get_adaptive_params(current_balance)
        tier        = ap.capital_tier

        # ── Base limits from adaptive system ─────────────────────────────────
        adaptive_daily = math.floor(ap.trade_limit)
        daily_cap      = min(adaptive_daily, MAX_TRADES_PER_DAY)

        # ── Max concurrent by tier ────────────────────────────────────────────
        tier_concurrent = {
            'TINY':  1,  # v5.1: unchanged — micro capital
            'SMALL': 2,  # v5.1: raised from 1 to allow 1 primary + 1 secondary
            'MICRO': 2,  # v5.1: raised from 1
            'BASE':  3,  # v5.1: raised from 2 to enable multi-symbol parallelization
            'SCALE': 3,  # v5.1: raised from 2
        }.get(tier, 1)

        # ── Reduce slots on loss streak ───────────────────────────────────────
        if consecutive_losses >= 2:
            tier_concurrent = 1   # only 1 trade when on losing streak
            daily_cap       = max(1, daily_cap - 1)

        # ── Reduce slots on losing day ────────────────────────────────────────
        if daily_pnl_pct < -0.01:   # down >1% on the day
            tier_concurrent = 1
            daily_cap       = max(1, daily_cap - 1)

        # ── Remaining ────────────────────────────────────────────────────────
        concurrent_remaining = max(0, tier_concurrent - current_open_trades)
        daily_remaining      = max(0, daily_cap - trades_today)

        reason = (
            f"tier={tier} concurrent={concurrent_remaining}/{tier_concurrent} "
            f"daily={daily_remaining}/{daily_cap} "
            f"losses={consecutive_losses} day_pnl={daily_pnl_pct*100:.2f}%"
        )

        logger.debug(f"TradeSlots: {reason}")
        return concurrent_remaining, daily_remaining, reason
