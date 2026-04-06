"""
config.py — v4.0
=================
AUDIT CHANGES:
- MAX_LEVERAGE reduced: 3x max for micro-capital safety
- MIN_ADX raised: 22 (was 18) — stricter trend filter
- SCAN_INTERVAL raised: 30s (was 15s) — reduce API hammering
- STOP_LOSS_PCT now ATR-driven in risk_manager but kept as floor here
- Added SLIPPAGE_BUFFER and FEE_RATE to config (were hardcoded or missing)
- DAILY_DRAWDOWN_LIMIT: now -3% hard (was -4%, too generous)
- MAX_TRADES_PER_DAY: 3 (was 1 — too restrictive if edge is real)
- TRADE_COOLDOWN_SEC: 300s enforced in risk manager
"""

import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger("Configuration")

if not load_dotenv():
    logger.warning("CRITICAL: .env file not found. Set environment variables manually.")

# ── Exchange ───────────────────────────────────────────────────────────────────
USE_SANDBOX        = os.getenv("USE_SANDBOX", "True").lower() == "true"
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET     = os.getenv("BINANCE_SECRET")

if not USE_SANDBOX and (not BINANCE_API_KEY or not BINANCE_SECRET):
    raise RuntimeError("Production mode requires valid Binance API keys.")

# ── Capital ────────────────────────────────────────────────────────────────────
BASE_ASSET         = os.getenv("BASE_ASSET", "USDT")
INITIAL_CAPITAL    = float(os.getenv("INITIAL_CAPITAL", "2.4"))   # ~₹200

# ── Futures mode ───────────────────────────────────────────────────────────────
TRADING_MODE       = "FUTURES"
MODE               = "SNIPER"

# ── Risk engine ────────────────────────────────────────────────────────────────
GEAR_1_CAPITAL              = 24.0
FIXED_RISK_PCT              = 0.012
MAX_RISK_CAP                = 0.020
VOLATILITY_THRESHOLD        = 0.020
MICRO_CAPITAL_MODE          = True
DAILY_TARGET_PCT            = 0.050           # 5% daily target
DAILY_DRAWDOWN_LIMIT        = -0.030          # -3% hard kill (v4: was -4%)
MAX_ACTIVE_TRADES           = 1
DEFAULT_LEVERAGE            = 3
MAX_RISK_PER_TRADE          = 0.020
MAX_LEVERAGE                = 3               # HARD CAP — micro capital
MIN_NOTIONAL                = 5.0            # Binance USDT-M minimum $5
MIN_NOTIONAL_BUFFER         = 1.20
STOP_LOSS_PCT               = 0.010          # 1% floor SL (ATR overrides in risk_manager)
TAKE_PROFIT_TARGET          = 0.020          # 2% TP1
MAX_TRADES_PER_DAY          = 4              # v5.1: raised to 4 to allow ~1/day across primary + secondary setups
                                             # DynamicTradeSlotEngine still enforces per-tier and per-balance limits
SURVIVAL_BALANCE_THRESHOLD  = 0.78

# ── Fees & execution realism (v4: were missing) ────────────────────────────────
FEE_RATE           = 0.0002    # Binance maker rate (with BNB discount)
SLIPPAGE_BUFFER    = 0.0008    # 0.08% estimated slippage per side

# ── Trading hours (UTC) ────────────────────────────────────────────────────────
TRADE_START_HOUR_UTC = 8
TRADE_END_HOUR_UTC   = 20

# ── Signal filters (v4: tightened) ────────────────────────────────────────────
MIN_IMBALANCE       = 0.30
MIN_VOLUME_SPIKE    = 1.30     # v4: raised from 1.25 (stricter)
MIN_ADX             = 20       # v5-CA: relaxed 22→20 (controlled aggressive)
MICRO_EDGE_MIN      = 0.005
RSI_MIN             = 45
RSI_MAX             = 68
MAX_ATR_VOLATILITY  = 0.015    # ATR% ceiling — skip above this
MIN_OFI             = 0.08       # v5-CA: relaxed 0.10→0.08 (controlled aggressive)
MIN_ARB_DELTA       = 0.0001
MAX_SPREAD_PCT      = 0.0004   # 0.04% max spread (v4: tighter than 0.05%)

# ── Allowed symbols — low notional focus ──────────────────────────────────────
# v5.1: expanded for multi-symbol parallelization
ALLOWED_SYMBOLS = [
    'DOGE/USDT:USDT',
    'XRP/USDT:USDT',
    'ADA/USDT:USDT',
    'SOL/USDT:USDT',
    'BNB/USDT:USDT',
]
MAX_SCAN_SYMBOLS = 5
MIN_PRIORITY_GAP = 5.0

# ── Scanner ────────────────────────────────────────────────────────────────────
SCAN_INTERVAL  = 30            # v4: was 15s — reduce API pressure
OHLCV_TIMEFRAME = '5m'

# ── API Keys ───────────────────────────────────────────────────────────────────
GEMINI_API_KEYS = [
    os.getenv(f"GEMINI_API_KEY_{i}", f"gemini_key_{i:02d}_placeholder")
    for i in range(1, 11)
]

# ── Telegram ───────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_ADMIN_ID  = os.getenv("TELEGRAM_ADMIN_ID")
TRADING_ENABLED    = True
