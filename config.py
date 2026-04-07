"""
config.py — v5.1 Micro-Cap Optimized
=====================================
Updated for 1 trade/day + \~1% daily target with $2.4 capital
"""

import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger("Configuration")

if not load_dotenv():
    logger.warning("CRITICAL: .env file not found. Set environment variables manually.")

# =============================================================================
# === MICRO-CAP + STARVATION MODE (NEW - Patch 1) ===
# =============================================================================
MICRO_SCALP_ENABLED       = True
TRADE_STARVATION_HOURS    = 22          # 22 ghante bina trade → auto relax
STARVATION_RELAX_FACTOR   = 0.82        # 18% looser thresholds

MICRO_RISK_PCT            = 0.008       # 0.8% risk per micro-scalp trade
MICRO_TARGET_RR           = 2.0
MICRO_MAX_HOLD_MINUTES    = 30

# Relaxed defaults for low-capital / starvation mode
MIN_ADX_MICRO             = 18
MIN_VOLUME_SPIKE_MICRO    = 1.10
MIN_ABSORPTION_MICRO      = 0.25
MAX_SPREAD_PCT_MICRO      = 0.0008      # 0.08% (realistic for DOGE/XRP/ADA)

# =============================================================================
# === EXCHANGE & CAPITAL ===
# =============================================================================
USE_SANDBOX        = os.getenv("USE_SANDBOX", "True").lower() == "true"
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET     = os.getenv("BINANCE_SECRET")

if not USE_SANDBOX and (not BINANCE_API_KEY or not BINANCE_SECRET):
    raise RuntimeError("Production mode requires valid Binance API keys.")

BASE_ASSET         = os.getenv("BASE_ASSET", "USDT")
INITIAL_CAPITAL    = float(os.getenv("INITIAL_CAPITAL", "2.4"))

# =============================================================================
# === TRADING MODE & RISK ENGINE ===
# =============================================================================
TRADING_MODE                = "FUTURES"
MODE                        = "SNIPER"

GEAR_1_CAPITAL              = 24.0
FIXED_RISK_PCT              = 0.012
MAX_RISK_CAP                = 0.020
VOLATILITY_THRESHOLD        = 0.020
MICRO_CAPITAL_MODE          = True
DAILY_TARGET_PCT            = 0.050
DAILY_DRAWDOWN_LIMIT        = -0.030          # -3% hard kill
MAX_ACTIVE_TRADES           = 1
DEFAULT_LEVERAGE            = 3
MAX_LEVERAGE                = 3
MIN_NOTIONAL                = 5.0
MIN_NOTIONAL_BUFFER         = 1.20
STOP_LOSS_PCT               = 0.010
TAKE_PROFIT_TARGET          = 0.020
MAX_TRADES_PER_DAY          = 4
SURVIVAL_BALANCE_THRESHOLD  = 0.78

# =============================================================================
# === FEES & EXECUTION ===
# =============================================================================
FEE_RATE           = 0.0002
SLIPPAGE_BUFFER    = 0.0008

# =============================================================================
# === TRADING HOURS (UTC) ===
# =============================================================================
TRADE_START_HOUR_UTC = 8
TRADE_END_HOUR_UTC   = 20

# =============================================================================
# === SIGNAL FILTERS (v5.1 relaxed for frequency) ===
# =============================================================================
MIN_IMBALANCE       = 0.30
MIN_VOLUME_SPIKE    = 1.30
MIN_ADX             = 20
MICRO_EDGE_MIN      = 0.005
RSI_MIN             = 45
RSI_MAX             = 68
MAX_ATR_VOLATILITY  = 0.015
MIN_OFI             = 0.08
MIN_ARB_DELTA       = 0.0001
MAX_SPREAD_PCT      = 0.0004          # main mode (micro mode uses MAX_SPREAD_PCT_MICRO)

# =============================================================================
# === SYMBOLS ===
# =============================================================================
ALLOWED_SYMBOLS = [
    'DOGE/USDT:USDT',
    'XRP/USDT:USDT',
    'ADA/USDT:USDT',
    'SOL/USDT:USDT',
    'BNB/USDT:USDT',
]
MAX_SCAN_SYMBOLS = 5
MIN_PRIORITY_GAP = 5.0

# =============================================================================
# === SCANNER ===
# =============================================================================
SCAN_INTERVAL   = 30
OHLCV_TIMEFRAME = '5m'

# =============================================================================
# === GEMINI API KEYS ===
# =============================================================================
GEMINI_API_KEYS = [
    os.getenv(f"GEMINI_API_KEY_{i}", f"gemini_key_{i:02d}_placeholder")
    for i in range(1, 11)
]

# =============================================================================
# === TELEGRAM ===
# =============================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_ADMIN_ID  = os.getenv("TELEGRAM_ADMIN_ID")
TRADING_ENABLED    = True
