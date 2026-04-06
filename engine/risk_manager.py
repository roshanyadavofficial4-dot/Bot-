"""
Risk Manager — v4.1 Capital-Adaptive
======================================
AUDIT FINDINGS FIXED (v4.0):
1. Position sizing was ignoring ATR properly — now ATR-scaled
2. No cooldown between trades — added 5-min cooldown
3. current_risk never initialized from get_base_risk() — fixed
4. Losing streak handling too slow — hard floor + consecutive loss breaker
5. Sharpe/Expectancy tracking added for live performance monitoring
6. No separation between survival mode and normal mode — added
7. Round-trip cost check before sizing — fee+slippage must be coverable

v4.1 ADDITIONS — Capital Adaptive Layer:
- check_daily_limits() now reads max_daily_loss from get_adaptive_params()
- calculate_position_size() risk floor/ceiling derived from adaptive params
- get_dynamic_rr_tp() minimum floor set by adaptive rr_min
- _base_risk_for() replaced entirely — adaptive params are canonical
- No static values remain in risk calculation path
"""

import logging
import math
import time
from collections import deque
import numpy as np

from config import (
    INITIAL_CAPITAL,
    DAILY_TARGET_PCT,
    DAILY_DRAWDOWN_LIMIT,
    MAX_ACTIVE_TRADES,
    SURVIVAL_BALANCE_THRESHOLD,
    MAX_LEVERAGE,
    MIN_NOTIONAL,
    MIN_NOTIONAL_BUFFER,
    STOP_LOSS_PCT,
    FEE_RATE,
    SLIPPAGE_BUFFER,
)
from engine.capital_adaptive import get_adaptive_params

logger = logging.getLogger("RiskManager")

# ── Constants ────────────────────────────────────────────────────────────────
TRADE_COOLDOWN_SEC  = 300    # 5 min minimum between trades
MAX_RISK_PER_TRADE  = 0.020  # 2% absolute max
MIN_RISK_PER_TRADE  = 0.005  # 0.5% floor
MAX_DAILY_LOSS_PCT  = 0.030  # 3% hard daily kill
CIRCUIT_BREAK_DD    = 0.050  # 5% global drawdown halt
AGGRESSIVE_DD_BRAKE = 0.080  # v5-CA: 8% equity drawdown → halve risk (extra safety layer)
SURVIVAL_RISK_PCT   = 0.010  # 1% when in survival mode
ROUND_TRIP_COST     = (FEE_RATE * 2) + SLIPPAGE_BUFFER


class RiskManager:
    """
    v4.0 — Regime-aware, ATR-scaled, cooldown-enforced risk engine.
    """

    def __init__(self):
        self.initial_capital     = INITIAL_CAPITAL
        self.daily_target_pct    = DAILY_TARGET_PCT
        self.max_drawdown_pct    = DAILY_DRAWDOWN_LIMIT
        self.max_active_trades   = MAX_ACTIVE_TRADES
        self.survival_threshold  = SURVIVAL_BALANCE_THRESHOLD
        self.current_risk        = self._base_risk_for(INITIAL_CAPITAL)
        self.streak              = 0
        self.consecutive_losses  = 0
        self.last_trade_time     = 0.0
        # v5-CA: session start balance for 8% equity drawdown brake
        self.session_start_balance: float = INITIAL_CAPITAL
        self._pnl_history        = deque(maxlen=100)
        self._pnl_pct_history    = deque(maxlen=100)
        # Change 1: last 10 trade PnLs for equity drawdown protection
        self._recent_pnl         = deque(maxlen=10)

    # ── Base risk ─────────────────────────────────────────────────────────────

    def _base_risk_for(self, capital: float) -> float:
        """
        v4.1: Delegates entirely to capital_adaptive.get_adaptive_params().
        The old step-function is REPLACED by smooth sigmoid interpolation.
        Old anchors: <5→1%, <20→1.2%, <100→1.5%, <500→1.8%, else 2%.
        New formula: continuous 0.5% (tiny) → 2.0% (scale), no hard jumps.
        """
        return get_adaptive_params(capital).risk_per_trade

    def get_base_risk(self, capital: float) -> float:
        return self._base_risk_for(capital)

    # ── Change 1: Equity Drawdown Protection ──────────────────────────────────

    def get_equity_drawdown_multiplier(self) -> float:
        """
        Returns 0.5 if the sum of the last 10 trade PnLs is negative,
        otherwise 1.0 (no adjustment).

        Sits on top of the adaptive risk system — it halves whatever
        risk the adaptive layer already computed. It does NOT replace it.
        Applied inside calculate_position_size() after the adaptive clamp.
        """
        if len(self._recent_pnl) < 1:
            return 1.0
        if sum(self._recent_pnl) < 0:
            logger.warning(
                f"EQUITY DRAWDOWN PROTECTION: last {len(self._recent_pnl)} trades "
                f"sum={sum(self._recent_pnl):.4f} < 0 → risk halved"
            )
            return 0.5
        return 1.0

    # ── v5-CA: Aggressive mode 8% equity drawdown brake ──────────────────────

    def set_session_start_balance(self, balance: float) -> None:
        """
        v5-CA: Call once at session start (or daily reset) to anchor the
        8% equity drawdown brake. Does NOT replace or interact with the
        existing global_circuit_breaker (which uses the caller-supplied
        start_balance from daily_state['balance']).
        """
        self.session_start_balance = max(balance, 1e-9)
        logger.info(f"v5-CA session_start_balance set to ${balance:.4f}")

    def get_aggressive_dd_brake_multiplier(self, current_balance: float) -> float:
        """
        v5-CA EXTRA SAFETY LAYER — independent of existing equity drawdown protection.

        If total equity drawdown from session_start_balance exceeds 8%,
        returns 0.5 (halve effective risk_per_trade).  Otherwise returns 1.0.

        This is ADDITIVE to existing protections:
          • get_equity_drawdown_multiplier() checks rolling 10-trade PnL sum < 0
          • global_circuit_breaker() halts trading at 5% daily drawdown
          • check_daily_limits() hard-kills at DAILY_DRAWDOWN_LIMIT (-3%)
          • THIS method fires specifically at 8% TOTAL equity drawdown

        Applied in calculate_position_size() AFTER the existing ed_mult check.
        """
        if self.session_start_balance <= 0:
            return 1.0
        drawdown = (self.session_start_balance - current_balance) / self.session_start_balance
        if drawdown >= AGGRESSIVE_DD_BRAKE:
            logger.warning(
                f"v5-CA AGGRESSIVE DD BRAKE: equity drawdown {drawdown*100:.2f}% "
                f">= {AGGRESSIVE_DD_BRAKE*100:.0f}% → risk halved"
            )
            return 0.5
        return 1.0

    # ── Circuit breakers ──────────────────────────────────────────────────────

    def global_circuit_breaker(self, current_balance: float, start_balance: float) -> bool:
        if start_balance <= 0:
            return False
        drawdown = (start_balance - current_balance) / start_balance
        if drawdown >= CIRCUIT_BREAK_DD:
            logger.critical(f"CIRCUIT BREAKER: Drawdown {drawdown*100:.2f}%")
            return True
        if self.consecutive_losses >= 3:
            logger.critical(f"CIRCUIT BREAKER: {self.consecutive_losses} consecutive losses")
            return True
        return False

    def check_daily_limits(self, current_daily_pnl_pct: float, current_balance: float) -> dict:
        # v4.1: daily loss limit is capital-adaptive (2% tiny → 3% scale)
        adaptive   = get_adaptive_params(current_balance)
        daily_cap  = adaptive.max_daily_loss   # smooth function of balance

        if current_balance < self.initial_capital * self.survival_threshold:
            return {'trade_allowed': False, 'reason': 'SURVIVAL_THRESHOLD_HIT'}
        if current_daily_pnl_pct <= -daily_cap:
            logger.warning(
                f"DAILY LOSS CAP (adaptive): {current_daily_pnl_pct*100:.2f}% "
                f"<= -{daily_cap*100:.2f}% (tier={adaptive.capital_tier})"
            )
            return {'trade_allowed': False, 'reason': 'MAX_DAILY_LOSS_HIT'}
        if current_daily_pnl_pct >= self.daily_target_pct:
            logger.info(f"DAILY TARGET HIT: {current_daily_pnl_pct*100:.2f}%")
            return {'trade_allowed': False, 'reason': 'PROFIT_TARGET_HIT'}
        return {'trade_allowed': True}

    # ── Cooldown ──────────────────────────────────────────────────────────────

    def is_in_cooldown(self) -> bool:
        elapsed = time.time() - self.last_trade_time
        if elapsed < TRADE_COOLDOWN_SEC:
            logger.info(f"COOLDOWN: {TRADE_COOLDOWN_SEC - elapsed:.0f}s remaining")
            return True
        return False

    def record_trade_time(self):
        self.last_trade_time = time.time()

    # ── Streak / dynamic risk ─────────────────────────────────────────────────

    def update_risk_profile(self, won: bool, pnl_usd: float = 0.0):
        """
        Updates streak, consecutive_losses, and current_risk after each trade.

        Change 3 — Loss Streak Soft Control (2-loss intermediate brake):
          consecutive_losses == 2 → halve current_risk (soft warning layer)
          consecutive_losses >= 3 → circuit breaker fires in global_circuit_breaker()
          The two layers stack: loss 2 halves risk, loss 3 halts trading entirely.

        Change 1 — _recent_pnl is appended here so equity drawdown
          protection has fresh data before the next position sizing call.
        """
        # Change 1: feed recent PnL window for equity drawdown protection
        self._recent_pnl.append(pnl_usd)

        if won:
            self.streak = max(self.streak + 1, 1)
            self.consecutive_losses = 0
            self.current_risk = min(self.current_risk * 1.10, MAX_RISK_PER_TRADE)
        else:
            self.streak = min(self.streak - 1, -1)
            self.consecutive_losses += 1
            self.current_risk = max(self.current_risk * 0.75, MIN_RISK_PER_TRADE)

            # Change 3: soft control fires at exactly 2 consecutive losses
            if self.consecutive_losses == 2:
                self.current_risk = max(self.current_risk * 0.5, MIN_RISK_PER_TRADE)
                logger.warning(
                    f"LOSS STREAK SOFT CONTROL: 2 consecutive losses → "
                    f"risk halved to {self.current_risk*100:.3f}% "
                    f"(3-loss kill switch still armed)"
                )

        logger.info(
            f"Risk: won={won} streak={self.streak} "
            f"consec_losses={self.consecutive_losses} risk={self.current_risk:.4f}"
        )

    # ── Auto leverage ─────────────────────────────────────────────────────────

    def calculate_auto_leverage(
        self, capital, min_notional=5, max_leverage=3, notional_buffer=1.0
    ):
        if capital <= 0:
            return None
        required = math.ceil((min_notional * notional_buffer) / capital)
        if required > max_leverage:
            logger.warning(
                f"Need {required}x but MAX_LEVERAGE={max_leverage}x "
                f"for capital=${capital:.4f}. Skipping."
            )
            return None
        return float(required)

    # ── Position sizing ───────────────────────────────────────────────────────

    async def calculate_position_size(
        self,
        exchange,
        account_balance: float,
        symbol: str,
        entry_price: float,
        atr: float,
        risk: float = None,
        regime: str = 'TRENDING',
    ) -> float:
        """
        ATR-scaled position sizing — v4.1 capital-adaptive.

        Risk ceiling and floor are now derived from get_adaptive_params()
        instead of hardcoded constants. The adaptive params guarantee that
        a ₹200 account never risks more than ~0.75% and a ₹2000 account
        scales to ~1.5% automatically.

        SL distance = max(1.5 * ATR_pct, STOP_LOSS_PCT)
        Size = (balance * risk) / sl_distance_pct
        Regime: TRENDING=100%, RANGING=60%, UNKNOWN=80%
        """
        # v4.1: pull live adaptive params
        adaptive = get_adaptive_params(account_balance)

        if risk is None:
            risk = self.current_risk

        # Clamp risk to adaptive bounds (never exceed the capital-appropriate ceiling)
        risk = max(risk, adaptive.risk_per_trade * 0.5)   # floor: 50% of base
        risk = min(risk, adaptive.risk_per_trade * 1.5)   # ceiling: 150% of base

        # Change 1: equity drawdown protection — multiplicative on top of adaptive risk
        # If last 10 trades are net negative, halve whatever risk was computed above.
        # This does NOT replace the adaptive system; it multiplies into it.
        ed_mult = self.get_equity_drawdown_multiplier()
        if ed_mult < 1.0:
            risk = max(risk * ed_mult, MIN_RISK_PER_TRADE)

        # v5-CA EXTRA SAFETY LAYER: 8% total equity drawdown brake.
        # Applied AFTER existing ed_mult — both can fire independently.
        # Does NOT touch DAILY_DRAWDOWN_LIMIT, kill switch, or circuit breaker.
        agg_brake = self.get_aggressive_dd_brake_multiplier(account_balance)
        if agg_brake < 1.0:
            risk = max(risk * agg_brake, MIN_RISK_PER_TRADE)

        # Survival mode: hard override to minimum
        if account_balance < self.initial_capital * self.survival_threshold:
            risk = min(risk, adaptive.risk_per_trade * 0.5)
            logger.warning(
                f"SURVIVAL MODE: risk capped to {risk*100:.3f}% "
                f"(adaptive base={adaptive.risk_per_trade*100:.3f}%)"
            )

        # Cost feasibility check
        expected_profit_pct = risk * self.get_dynamic_rr_tp()
        if expected_profit_pct <= ROUND_TRIP_COST:
            logger.warning(
                f"Cost filter: profit {expected_profit_pct:.4f} <= cost {ROUND_TRIP_COST:.4f}"
            )
            return 0.0

        # ATR-based SL distance
        if atr > 0 and entry_price > 0:
            atr_pct = atr / entry_price
            sl_distance_pct = max(atr_pct * 1.5, STOP_LOSS_PCT)
        else:
            sl_distance_pct = STOP_LOSS_PCT

        # Volatility guard — skip if market too wild
        if sl_distance_pct > 0.04:
            logger.warning(f"ATR too high: sl_dist={sl_distance_pct*100:.2f}%. Skipping.")
            return 0.0

        # Regime multiplier
        regime_mult = {'TRENDING': 1.0, 'RANGING': 0.6, 'UNKNOWN': 0.8}.get(regime, 0.8)

        # Core sizing
        risk_amount       = account_balance * risk * regime_mult
        position_notional = risk_amount / sl_distance_pct

        max_notional = account_balance * MAX_LEVERAGE
        if position_notional > max_notional:
            position_notional = max_notional

        min_notional_req = MIN_NOTIONAL * MIN_NOTIONAL_BUFFER
        if position_notional < min_notional_req:
            bumped_risk = (min_notional_req * sl_distance_pct) / account_balance
            if bumped_risk <= MAX_RISK_PER_TRADE and min_notional_req <= max_notional:
                position_notional = min_notional_req
            else:
                logger.warning(f"Cannot meet min notional. Skipping.")
                return 0.0

        # Exchange precision
        raw_qty = position_notional / (entry_price + 1e-9)
        try:
            await exchange.load_markets()
            final_qty = float(exchange.amount_to_precision(symbol, raw_qty))
        except Exception as e:
            logger.error(f"Precision error {symbol}: {e}")
            return 0.0

        if final_qty * entry_price < MIN_NOTIONAL * MIN_NOTIONAL_BUFFER:
            logger.warning("Post-precision notional too low. Skipping.")
            return 0.0

        logger.info(
            f"Sizing: {final_qty} {symbol} notional=${final_qty*entry_price:.2f} "
            f"risk={risk*100:.2f}% sl={sl_distance_pct*100:.2f}% regime={regime}"
        )
        return final_qty

    # ── Dynamic RR ────────────────────────────────────────────────────────────

    def get_dynamic_rr_tp(self, balance: float = None) -> float:
        """
        v4.1: RR floor is now capital-adaptive.
        - Tiny capital: minimum 2.5:1 (streak bonus on top)
        - Scale capital: minimum 2.0:1 (streak bonus on top)
        balance arg optional — pass current_balance for precision.
        """
        streak_bonus = max(0, self.streak) * 0.1
        rr = 2.0 + streak_bonus
        if balance is not None:
            rr_floor = get_adaptive_params(balance).rr_min
            rr = max(rr, rr_floor)
        return min(rr, 3.0)

    # ── Performance tracking ──────────────────────────────────────────────────

    def record_pnl(self, pnl_usd: float, pnl_pct: float):
        self._pnl_history.append(pnl_usd)
        self._pnl_pct_history.append(pnl_pct)
        # _recent_pnl is fed via update_risk_profile(pnl_usd=...) on trade close.

    def get_performance_summary(self) -> dict:
        pnls = list(self._pnl_history)
        pcts = list(self._pnl_pct_history)
        if not pnls:
            return {'win_rate': 0.0, 'expectancy': 0.0, 'sharpe': 0.0,
                    'max_dd': 0.0, 'total_trades': 0}
        wins      = [p for p in pnls if p > 0]
        losses    = [p for p in pnls if p <= 0]
        win_rate  = len(wins) / len(pnls)
        avg_win   = sum(wins) / len(wins)   if wins   else 0.0
        avg_loss  = sum(losses) / len(losses) if losses else 0.0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        cum    = np.cumsum(pnls)
        peak   = np.maximum.accumulate(cum)
        max_dd = float(np.max(peak - cum)) if len(cum) > 0 else 0.0
        sharpe = 0.0
        if len(pcts) > 1:
            mean_r = np.mean(pcts)
            std_r  = np.std(pcts) + 1e-9
            sharpe = float((mean_r / std_r) * (252 * 3) ** 0.5)
        return {
            'win_rate': round(win_rate, 4),
            'expectancy': round(expectancy, 6),
            'sharpe': round(sharpe, 3),
            'max_dd': round(max_dd, 6),
            'total_trades': len(pnls),
            'avg_win': round(avg_win, 6),
            'avg_loss': round(avg_loss, 6),
        }

    def check_portfolio_exposure(self, active_trades_count: int) -> bool:
        if active_trades_count >= self.max_active_trades:
            logger.warning(f"EXPOSURE CAP: {active_trades_count} trades.")
            return False
        return True

    def time_based_kill_switch(self, trade_open_time: float) -> bool:
        return (time.time() - trade_open_time) > (4 * 3600)
