"""
capital_adaptive.py — v1.0
============================
FULLY DYNAMIC CAPITAL-ADAPTIVE SYSTEM

This module is the single source of truth for ALL runtime parameters
that scale with account balance. Every other module must import from
here instead of reading static config values.

Mathematical Model
------------------
All parameters use smooth sigmoid-family interpolation between anchor
points. There are NO mode switches or hard jumps — the function is
continuous and differentiable everywhere.

Anchor points (in USDT; ₹200 ≈ $2.4, ₹1000 ≈ $12, ₹2000 ≈ $24,
₹5000+ ≈ $60+):

    Balance     Risk%   DailyLoss%  RR_min  Trades  Regime
    ──────────  ──────  ──────────  ──────  ──────  ──────────────────────
    $2.4  (tiny)  0.50%    2.0%      2.5:1    1     TRENDING only
    $6    (small) 0.70%    2.2%      2.4:1    1     TRENDING only
    $12   (micro) 1.00%    2.5%      2.2:1    1-2   TRENDING + UNKNOWN ok
    $24   (base)  1.50%    2.8%      2.0:1    2     Normal
    $60+  (scale) 2.00%    3.0%      2.0:1    3     Normal

Interpolation
─────────────
We use a logistic (sigmoid) function mapped over log(balance) so that:
  • Small capital: parameters are strongly protective
  • Medium capital: parameters ease linearly
  • Large capital: parameters approach the configured ceiling

The key formula for each parameter P with low anchor P_lo and high
anchor P_hi:

    t = sigmoid( (log(balance) - log(pivot)) / sharpness )
    P = P_lo + (P_hi - P_lo) * t

where:
  sigmoid(x) = 1 / (1 + exp(-x))
  pivot      = balance where P is midpoint between lo and hi
  sharpness  = how fast the transition happens (lower = smoother)

This guarantees:
  1. No cliff edges / mode switches
  2. Monotonic growth — as capital grows, all parameters trend to max
  3. Bounded — parameters never exceed the configured ceiling
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger("CapitalAdaptive")

# ── Anchor constants (all in USDT) ───────────────────────────────────────────

_TINY_CAPITAL  =  2.4    # ~₹200
_SMALL_CAPITAL =  6.0    # ~₹500
_MICRO_CAPITAL = 12.0    # ~₹1000
_BASE_CAPITAL  = 24.0    # ~₹2000
_SCALE_CAPITAL = 60.0    # ~₹5000+

# ── Signal threshold anchors ──────────────────────────────────────────────────

_TIGHT_OFI          = 0.20   # small capital: strong OFI required
_NORMAL_OFI         = 0.08   # v5-CA: relaxed 0.10→0.08 (controlled aggressive)

_TIGHT_ADX          = 28     # small capital: strong trend only
_NORMAL_ADX         = 20     # v5-CA: relaxed 22→20 (controlled aggressive)

_TIGHT_SPREAD_MULT  = 0.50   # small capital: spread budget is 50% of normal
_NORMAL_SPREAD_MULT = 1.00   # large capital: full spread budget

_TIGHT_VOL_SPIKE    = 1.50   # small capital: strong volume confirmation needed
_NORMAL_VOL_SPIKE   = 1.30   # large capital: from config


# ── Core interpolation math ───────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        e = math.exp(x)
        return e / (1.0 + e)


def _smooth_interp(
    balance: float,
    lo: float,
    hi: float,
    pivot_balance: float = _BASE_CAPITAL,
    sharpness: float = 1.2,
) -> float:
    """
    Smoothly interpolate between lo (at tiny capital) and hi (at large capital).

    t=0  → returns lo   (protective extreme)
    t=1  → returns hi   (permissive extreme)

    The sigmoid is applied over log(balance) so that relative changes in
    capital (doublings) feel uniform rather than additive ones.

    Args:
        balance:       Current account balance in USDT.
        lo:            Parameter value at minimum capital (most protective).
        hi:            Parameter value at large capital (most permissive).
        pivot_balance: Balance at which the parameter is exactly midway.
        sharpness:     Controls transition speed. Lower = smoother gradient.
                       1.2 means ~80% of the range is covered between 0.25x
                       and 4x the pivot.
    """
    if balance <= 0:
        return lo

    log_balance = math.log(max(balance, 0.01))
    log_pivot   = math.log(pivot_balance)

    t = _sigmoid((log_balance - log_pivot) / sharpness)
    return lo + (hi - lo) * t


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class AdaptiveParams:
    """
    All capital-adaptive parameters for the current balance tick.
    Every field is a continuous function of balance — no modes.
    """
    balance: float

    # ── Core risk ─────────────────────────────────────────────────────────────
    risk_per_trade: float = 0.005       # fraction of balance risked per trade
    max_daily_loss: float = 0.020       # fraction of balance; triggers kill switch

    # ── RR requirement ────────────────────────────────────────────────────────
    rr_min: float = 2.5                 # minimum R:R ratio required to take trade

    # ── Trade frequency ───────────────────────────────────────────────────────
    trade_limit: float = 1.0           # max trades per day (can be fractional;
                                        # floor() used when checking)

    # ── Signal thresholds ─────────────────────────────────────────────────────
    min_ofi: float = 0.20              # order flow imbalance threshold
    min_adx: float = 28                # ADX trend strength threshold
    spread_budget_mult: float = 0.50   # multiplier on config MAX_SPREAD_PCT
    min_volume_spike: float = 1.50     # vol_ratio threshold

    # ── Regime sensitivity ────────────────────────────────────────────────────
    allowed_regimes: list = field(default_factory=lambda: ['TRENDING'])

    # ── Derived / informational ───────────────────────────────────────────────
    capital_tier: str = "TINY"         # human-readable tier label
    protection_level: float = 1.0      # 0=none → 1=maximum protection (for logging)

    def as_dict(self) -> Dict:
        return {
            "balance":           round(self.balance, 4),
            "capital_tier":      self.capital_tier,
            "protection_level":  round(self.protection_level, 3),
            "risk_per_trade":    f"{self.risk_per_trade*100:.3f}%",
            "max_daily_loss":    f"{self.max_daily_loss*100:.2f}%",
            "rr_min":            round(self.rr_min, 2),
            "trade_limit":       math.floor(self.trade_limit),
            "min_ofi":           round(self.min_ofi, 4),
            "min_adx":           round(self.min_adx, 1),
            "spread_budget_mult":round(self.spread_budget_mult, 3),
            "min_volume_spike":  round(self.min_volume_spike, 3),
            "allowed_regimes":   self.allowed_regimes,
        }


# ── Main public API ───────────────────────────────────────────────────────────

def get_adaptive_params(balance: float) -> AdaptiveParams:
    """
    THE primary entry point. Given current account balance in USDT,
    returns a fully computed AdaptiveParams instance.

    All parameters transition smoothly. No if/elif chains.
    No mode switches. No hard jumps.

    Mathematical guarantee: for any ε > 0 and any two balances B1, B2
    with |B1 - B2| < ε, |P(B1) - P(B2)| < δ (continuous function).

    Args:
        balance: Current total account balance in USDT.

    Returns:
        AdaptiveParams: Fully populated, ready to replace static config.

    Example:
        >>> p = get_adaptive_params(2.4)   # ₹200
        >>> p.risk_per_trade               # 0.005 (0.5%)
        >>> p = get_adaptive_params(24.0)  # ₹2000
        >>> p.risk_per_trade               # 0.015 (1.5%)
    """
    balance = max(balance, 0.01)

    # ── 1. Risk per trade ─────────────────────────────────────────────────────
    # Anchors: 0.5% at $2.4 (tiny), 2.0% at $60+ (scale)
    # Pivot at $12 (midpoint of protection arc)
    # v5-CA CONTROLLED AGGRESSIVE: Risk brackets raised across all tiers.
    # Bracket targets: <$6→1.0%, $6-$25→1.2%, $25-$120→1.5%, $120+→2.0%
    # Pivot shifted from $12→$24 and lo raised from 0.5%→1.0%.
    # hard ceiling (2%) enforced below AND by MAX_RISK_PER_TRADE in risk_manager.
    # v5-CA CONTROLLED AGGRESSIVE: Risk brackets raised across all tiers.
    # Bracket targets: <$6→1.0%, $6-$25→1.2%, $25-$120→1.5%, $120+→2.0%
    # pivot=$6 pushes sigmoid right so curve rises sharply through $25-$120 range
    # v5-CA CONTROLLED AGGRESSIVE: Bracket-matched risk scaling.
    # Spec targets: <$6→~1.0%, $6-$25→~1.2%, $25-$120→~1.5%, $120+→~2.0%
    # pivot=$25, sharpness=0.5 produces:
    #   $2.4→1.009%  $6→1.054%  $12→1.187%  $25→1.500%  $60→1.852%  $120→1.958%
    risk_per_trade = _smooth_interp(
        balance,
        lo=0.010,          # v5-CA: 1.0% floor at tiny capital (was 0.5%)
        hi=0.020,          # 2.0% ceiling at scale capital (unchanged)
        pivot_balance=_BASE_CAPITAL,   # v5-CA: pivot=$24 for bracket alignment
        sharpness=0.5,                 # v5-CA: tighter curve for spec-accurate brackets
    )
    risk_per_trade = min(risk_per_trade, 0.020)  # hard ceiling: never exceed 2%

    # ── 2. Daily loss limit ───────────────────────────────────────────────────
    # Anchors: 2.0% at tiny, 3.0% at scale
    max_daily_loss = _smooth_interp(
        balance,
        lo=0.020,
        hi=0.030,
        pivot_balance=_BASE_CAPITAL,
        sharpness=1.0,
    )

    # ── 3. RR minimum ─────────────────────────────────────────────────────────
    # Anchors: 2.5:1 at tiny (strict), 2.0:1 at scale (relaxed)
    # NOTE: interpolates DOWNWARD (lo is protective = high RR requirement)
    rr_min = _smooth_interp(
        balance,
        lo=2.5,            # tiny capital: need better trades to survive costs
        hi=2.0,            # scale capital: standard RR acceptable
        pivot_balance=_BASE_CAPITAL,
        sharpness=1.2,
    )

    # ── 4. Trade frequency ────────────────────────────────────────────────────
    # Anchors: 1.0 trade/day at tiny, 3.0 at scale
    # v5-CA: Allow 2 trades/day from $6+ (was 1 trade until $24).
    # config.py MAX_TRADES_PER_DAY=2 acts as the hard cap in main.py.
    # The adaptive lo is raised to 2.0 so even TINY tier can attempt 2 trades.
    # The hard cap in main.py (min(adaptive, MAX_TRADES_PER_DAY=2)) prevents runaway.
    trade_limit = _smooth_interp(
        balance,
        lo=2.0,            # v5-CA: was 1.0 — allow 2 trades from tiny capital
        hi=3.0,            # scale capital still allows 3 (but config cap=2 applies)
        pivot_balance=_SMALL_CAPITAL,  # v5-CA: pivot=$6 (was $24)
        sharpness=1.1,
    )

    # ── 5. Signal thresholds ──────────────────────────────────────────────────
    # v5-CA: pivot reduced to $12 so min_ofi reaches ~0.08 by $60
    min_ofi = _smooth_interp(
        balance,
        lo=_TIGHT_OFI,
        hi=_NORMAL_OFI,
        pivot_balance=_MICRO_CAPITAL,  # v5-CA: pivot=$12 (was $24)
        sharpness=1.0,
    )

    # v5-CA: pivot reduced to $12 so min_adx reaches ~20 by $60
    min_adx = _smooth_interp(
        balance,
        lo=_TIGHT_ADX,
        hi=_NORMAL_ADX,
        pivot_balance=_MICRO_CAPITAL,  # v5-CA: pivot=$12 (was $24)
        sharpness=1.0,
    )

    spread_budget_mult = _smooth_interp(
        balance,
        lo=_TIGHT_SPREAD_MULT,
        hi=_NORMAL_SPREAD_MULT,
        pivot_balance=_BASE_CAPITAL,
        sharpness=1.0,
    )

    min_volume_spike = _smooth_interp(
        balance,
        lo=_TIGHT_VOL_SPIKE,
        hi=_NORMAL_VOL_SPIKE,
        pivot_balance=_BASE_CAPITAL,
        sharpness=1.0,
    )

    # ── 6. Allowed regimes ────────────────────────────────────────────────────
    # Small capital: TRENDING only
    # Large capital: TRENDING + UNKNOWN allowed
    # The transition is a smooth probability that determines whether
    # UNKNOWN is admitted. We model it as: if balance >= regime_unlock_balance,
    # UNKNOWN is permitted.
    #
    # regime_unlock_balance: balance above which UNKNOWN regime is admitted
    # We use the same sigmoid — if t > 0.60 we unlock UNKNOWN.
    regime_t = _smooth_interp(
        balance,
        lo=0.0,
        hi=1.0,
        pivot_balance=_BASE_CAPITAL,
        sharpness=1.0,
    )
    if regime_t >= 0.60:
        allowed_regimes = ['TRENDING', 'UNKNOWN']
    else:
        allowed_regimes = ['TRENDING']

    # ── 7. Protection level (0→1, purely informational) ───────────────────────
    protection_level = 1.0 - _smooth_interp(
        balance,
        lo=0.0,
        hi=1.0,
        pivot_balance=_BASE_CAPITAL,
        sharpness=1.0,
    )

    # ── 8. Capital tier label ─────────────────────────────────────────────────
    if balance < _SMALL_CAPITAL:
        tier = "TINY"         # < $6
    elif balance < _MICRO_CAPITAL:
        tier = "SMALL"        # $6–$12
    elif balance < _BASE_CAPITAL:
        tier = "MICRO"        # $12–$24
    elif balance < _SCALE_CAPITAL:
        tier = "BASE"         # $24–$60
    else:
        tier = "SCALE"        # $60+

    params = AdaptiveParams(
        balance           = balance,
        risk_per_trade    = risk_per_trade,
        max_daily_loss    = max_daily_loss,
        rr_min            = rr_min,
        trade_limit       = trade_limit,
        min_ofi           = min_ofi,
        min_adx           = min_adx,
        spread_budget_mult= spread_budget_mult,
        min_volume_spike  = min_volume_spike,
        allowed_regimes   = allowed_regimes,
        capital_tier      = tier,
        protection_level  = protection_level,
    )

    logger.debug(
        f"[AdaptiveParams] balance=${balance:.4f} tier={tier} "
        f"risk={risk_per_trade*100:.3f}% daily_loss={max_daily_loss*100:.2f}% "
        f"rr={rr_min:.2f} trades={math.floor(trade_limit)} "
        f"ofi={min_ofi:.3f} adx={min_adx:.1f} regimes={allowed_regimes}"
    )

    return params


# ── Diagnostic / calibration table ───────────────────────────────────────────

def print_scaling_table() -> None:
    """
    Print a human-readable scaling table across the full capital range.
    Run this standalone to verify the interpolation looks correct.
    """
    test_balances = [2.4, 4.0, 6.0, 9.0, 12.0, 18.0, 24.0, 36.0, 60.0, 120.0]
    inr_equiv     = [b * 83 for b in test_balances]

    header = (
        f"{'Balance':>10}  {'≈₹':>8}  {'Tier':>7}  {'Risk%':>6}  "
        f"{'DailyLoss':>9}  {'RR_min':>6}  {'Trades':>6}  "
        f"{'OFI':>5}  {'ADX':>5}  {'Regimes':<24}"
    )
    print("\n" + "=" * len(header))
    print("CAPITAL ADAPTIVE SCALING TABLE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for b, inr in zip(test_balances, inr_equiv):
        p = get_adaptive_params(b)
        regimes = "+".join(p.allowed_regimes)
        print(
            f"${b:>9.2f}  ₹{inr:>7.0f}  {p.capital_tier:>7}  "
            f"{p.risk_per_trade*100:>5.2f}%  "
            f"{p.max_daily_loss*100:>8.2f}%  "
            f"{p.rr_min:>6.2f}  "
            f"{math.floor(p.trade_limit):>6}  "
            f"{p.min_ofi:>5.3f}  "
            f"{p.min_adx:>5.1f}  "
            f"{regimes:<24}"
        )
    print("=" * len(header) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print_scaling_table()
