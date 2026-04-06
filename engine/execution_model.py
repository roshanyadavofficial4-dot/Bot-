"""
Execution Model — v1.0
=======================
Realistic slippage + fee simulation for pre-trade cost estimation.

AUDIT: The original bot placed trades without modeling:
- Market impact slippage on small-cap futures
- Funding rate cost (can be negative PnL over time)
- Partial fill probability
- Time-in-force GTX rejection rate

This module gives realistic pre-trade cost estimates
so the strategy can decide if edge > cost.
"""

import logging
from config import FEE_RATE, SLIPPAGE_BUFFER

logger = logging.getLogger("ExecutionModel")

# Slippage model constants (empirical for Binance USDT-M futures, micro-cap)
SLIPPAGE_BASE_BPS   = 2.0    # 2bps minimum market impact
SLIPPAGE_VOL_SCALAR = 0.15   # scales with ATR%: high ATR = more slippage
SLIPPAGE_SIZE_SCALAR = 0.05  # scales with notional size relative to depth


def estimate_slippage(
    notional: float,
    atr_pct: float,
    liquidity_depth: float,
    side: str
) -> float:
    """
    Returns estimated one-way slippage as a fraction of price.

    Model:
        slippage = base + vol_component + size_component
        vol_component  = ATR% * 0.15  (volatile market → wider fills)
        size_component = (notional / depth) * 0.05
    """
    if liquidity_depth <= 0:
        liquidity_depth = 1000.0   # conservative fallback

    vol_component  = atr_pct * SLIPPAGE_VOL_SCALAR
    size_component = (notional / liquidity_depth) * SLIPPAGE_SIZE_SCALAR
    total_slippage = (SLIPPAGE_BASE_BPS / 10000) + vol_component + size_component

    # Cap at 0.5% per side — if higher, market is too illiquid
    total_slippage = min(total_slippage, 0.005)

    logger.debug(
        f"Slippage model: notional={notional:.2f} atr%={atr_pct*100:.3f} "
        f"depth={liquidity_depth:.0f} → {total_slippage*100:.3f}%"
    )
    return total_slippage


def estimate_round_trip_cost(
    entry_price: float,
    notional: float,
    atr_pct: float,
    liquidity_depth: float,
    hold_periods: int = 4,   # expected candles held
    funding_rate: float = 0.0001
) -> dict:
    """
    Full round-trip cost estimate:
        - Entry fee (maker: 0.02%, taker: 0.04%)
        - Exit fee
        - Entry slippage
        - Exit slippage
        - Funding cost (for futures, charged every 8h)

    Returns cost as a fraction of notional AND minimum required
    move to break even.
    """
    # Use maker rate (GTX orders) but assume 20% taker fills
    avg_fee_rate    = FEE_RATE * 0.8 + (FEE_RATE * 2.5) * 0.2   # blended
    fee_cost        = avg_fee_rate * 2                             # entry + exit

    slip_entry      = estimate_slippage(notional, atr_pct, liquidity_depth, 'buy')
    slip_exit       = estimate_slippage(notional, atr_pct, liquidity_depth, 'sell')
    slip_total      = slip_entry + slip_exit

    # Funding: fraction of 8h period based on expected hold
    funding_periods = hold_periods / (8 * 12)    # 5m candles in 8h = 96
    funding_cost    = abs(funding_rate) * funding_periods

    total_cost_pct  = fee_cost + slip_total + funding_cost
    breakeven_move  = total_cost_pct   # price must move at least this much

    return {
        'fee_cost':      round(fee_cost, 6),
        'slippage':      round(slip_total, 6),
        'funding':       round(funding_cost, 6),
        'total_cost_pct': round(total_cost_pct, 6),
        'breakeven_pct': round(breakeven_move, 6),
        'breakeven_usd': round(breakeven_move * notional, 6),
    }


def adjust_price_for_slippage(price: float, side: str, slippage_pct: float) -> float:
    """
    Returns realistic fill price after slippage.
    BUY  → pays slightly more  (slippage against you)
    SELL → receives slightly less
    """
    if side.lower() == 'buy':
        return price * (1 + slippage_pct)
    else:
        return price * (1 - slippage_pct)
