"""
Strategy Engine — v5.1 FREQUENCY-OPTIMIZED LIQUIDITY EDGE
==========================================================
HIGH-EXPECTANCY SYSTEM: Liquidity Sweep + Absorption + Forced Move

v5.1 UPGRADE — Controlled Frequency Expansion (0.38 → 0.8-1.2 trades/day)
+ NEW: MICRO_SCALP setup for $2.4 micro-capital + starvation mode
"""

import logging
import time
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field

from config import (
    MIN_VOLUME_SPIKE, MAX_ATR_VOLATILITY, MIN_OFI,
    MAX_SPREAD_PCT, MIN_ADX, FEE_RATE, SLIPPAGE_BUFFER,
    MICRO_SCALP_ENABLED   # ← NEW IMPORT for micro-scalp
)
from engine.capital_adaptive import get_adaptive_params
from engine.ml_brain import MLBrain

logger = logging.getLogger("Strategy")

MIN_NET_EDGE = (FEE_RATE * 2) + SLIPPAGE_BUFFER + 0.001

SETUP_TRAP_REVERSAL   = "TRAP_REVERSAL"
SETUP_CONTINUATION    = "CONTINUATION"
SETUP_MICRO_SCALP     = "MICRO_SCALP"          # ← NEW

# ── v5.1: Setup tier constants ────────────────────────────────────────────────
TIER_PRIMARY   = "PRIMARY"
TIER_SECONDARY = "SECONDARY"
TIER_REJECTED  = "REJECTED"

PRIMARY_SCORE_THRESHOLD   = 80.0
SECONDARY_SCORE_THRESHOLD = 65.0

# Primary thresholds (unchanged)
PRIMARY_VOLUME_SPIKE      = 1.5
PRIMARY_ABSORPTION_A      = 0.45
PRIMARY_ABSORPTION_B      = 0.40
PRIMARY_WIN_PROB          = 0.52

# Secondary thresholds (relaxed)
SECONDARY_VOLUME_SPIKE    = 1.3
SECONDARY_ABSORPTION      = 0.35
SECONDARY_WIN_PROB        = 0.50
SECONDARY_SIZE_MULTIPLIER = 0.60

# Sweep / structure constants (unchanged)
SWEEP_LOOKBACK        = 10
SWEEP_MIN_WICK_RATIO  = 0.45
SWEEP_RECLAIM_BARS    = 3

MIN_CVD_ABSORPTION    = 0.10
MIN_VOLUME_SPIKE_ABS  = 1.5

FORCED_MOVE_ATR_MULT  = 1.2

SETUP_A_RR_MIN        = 1.2
SETUP_A_RR_MAX        = 1.8
SETUP_B_RR_MIN        = 2.0
SETUP_B_RR_MAX        = 3.0

# v5.1: Micro re-entry constants
REENTRY_FIBO_LOW       = 0.382
REENTRY_FIBO_HIGH      = 0.618
REENTRY_SIGNAL_EXPIRY  = 300
REENTRY_MIN_ABSORPTION = 0.30


@dataclass
class SignalResult:
    signal: str
    confidence: float
    setup_type: str
    entry_price: float
    stop_loss: float
    take_profit: float
    rr_ratio: float
    win_prob: float
    sweep_detected: bool
    absorption_score: float
    forced_move_detected: bool
    volume_spike_ratio: float
    structure_reclaim: bool
    symbol: str
    timestamp: float
    setup_description: str
    setup_tier: str = field(default=TIER_PRIMARY)
    size_multiplier: float = field(default=1.0)
    is_reentry: bool = field(default=False)
    parent_signal_ts: float = field(default=0.0)


class LiquiditySweepDetector:
    def detect_bull_sweep(self, df, lookback=SWEEP_LOOKBACK):
        if df is None or len(df) < lookback + 3:
            return {'detected': False}
        try:
            closes = df['close'].values
            lows   = df['low'].values
            highs  = df['high'].values
            opens  = df['open'].values
            swing_low = float(np.min(lows[-(lookback + 3):-3]))
            for i in range(-3, 0):
                cl, cc, ch, co = lows[i], closes[i], highs[i], opens[i]
                cr = ch - cl
                if cr < 1e-10:
                    continue
                if cl < swing_low:
                    lw = min(co, cc) - cl
                    wr = lw / cr
                    if cc > swing_low and wr >= SWEEP_MIN_WICK_RATIO:
                        atr = self._calc_atr(df, 14)
                        return {
                            'detected': True, 'direction': 'BULL',
                            'sweep_low': cl, 'reclaim_level': swing_low,
                            'sweep_candle_idx': len(df) + i,
                            'wick_ratio': round(wr, 3),
                            'candles_to_reclaim': abs(i),
                            'forced_move': cr >= FORCED_MOVE_ATR_MULT * atr if atr > 0 else True,
                            'candle_range': cr, 'atr': atr,
                        }
            return {'detected': False}
        except Exception as e:
            logger.debug(f"Bull sweep error: {e}")
            return {'detected': False}

    def detect_bear_sweep(self, df, lookback=SWEEP_LOOKBACK):
        if df is None or len(df) < lookback + 3:
            return {'detected': False}
        try:
            closes = df['close'].values
            lows   = df['low'].values
            highs  = df['high'].values
            opens  = df['open'].values
            swing_high = float(np.max(highs[-(lookback + 3):-3]))
            for i in range(-3, 0):
                ch, cc, cl, co = highs[i], closes[i], lows[i], opens[i]
                cr = ch - cl
                if cr < 1e-10:
                    continue
                if ch > swing_high:
                    uw = ch - max(co, cc)
                    wr = uw / cr
                    if cc < swing_high and wr >= SWEEP_MIN_WICK_RATIO:
                        atr = self._calc_atr(df, 14)
                        return {
                            'detected': True, 'direction': 'BEAR',
                            'sweep_high': ch, 'reclaim_level': swing_high,
                            'sweep_candle_idx': len(df) + i,
                            'wick_ratio': round(wr, 3),
                            'candles_to_reclaim': abs(i),
                            'forced_move': cr >= FORCED_MOVE_ATR_MULT * atr if atr > 0 else True,
                            'candle_range': cr, 'atr': atr,
                        }
            return {'detected': False}
        except Exception as e:
            logger.debug(f"Bear sweep error: {e}")
            return {'detected': False}

    def _calc_atr(self, df, period=14):
        try:
            if len(df) < period:
                return 0.0
            h, l, c = df['high'], df['low'], df['close']
            tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
            return float(tr.rolling(period).mean().iloc[-1])
        except Exception:
            return 0.0


class AbsorptionDetector:
    def score_absorption(self, df, cvd, ofi, signal_side):
        score = 0.0
        if signal_side == 'BUY':
            if cvd > MIN_CVD_ABSORPTION:   score += 0.40
            elif cvd > 0:                  score += 0.20
            if ofi > MIN_OFI * 2:          score += 0.35
            elif ofi > MIN_OFI:            score += 0.20
        elif signal_side == 'SELL':
            if cvd < -MIN_CVD_ABSORPTION:  score += 0.40
            elif cvd < 0:                  score += 0.20
            if ofi < -MIN_OFI * 2:         score += 0.35
            elif ofi < -MIN_OFI:           score += 0.20
        try:
            if df is not None and len(df) >= 6:
                rcc = df['close'].iloc[-1] - df['close'].iloc[-4]
                rv  = df['volume'].iloc[-3:].sum()
                pv  = df['volume'].iloc[-6:-3].sum()
                if pv > 0:
                    vr = rv / pv
                    if signal_side == 'BUY'  and rcc < 0 and cvd > 0:
                        score += 0.25 * min(vr / 1.5, 1.0)
                    elif signal_side == 'SELL' and rcc > 0 and cvd < 0:
                        score += 0.25 * min(vr / 1.5, 1.0)
        except Exception:
            pass
        return min(score, 1.0)

    def detect_volume_spike(self, df, min_ratio=MIN_VOLUME_SPIKE_ABS):
        try:
            if df is None or len(df) < 21:
                return False, 1.0
            avg = df['volume'].iloc[-21:-1].mean()
            last = df['volume'].iloc[-1]
            ratio = last / avg if avg > 0 else 1.0
            return ratio >= min_ratio, round(ratio, 3)
        except Exception:
            return False, 1.0


class StructureEngine:
    def check_structure_reclaim(self, df, sweep_result, signal_side):
        if not sweep_result.get('detected'):
            return False
        try:
            rl = sweep_result.get('reclaim_level', 0)
            lc = df['close'].iloc[-1]
            if signal_side == 'BUY':  return lc > rl
            elif signal_side == 'SELL': return lc < rl
        except Exception:
            pass
        return False

    def check_structure_break(self, df, lookback=8):
        if df is None or len(df) < lookback + 2:
            return {'detected': False}
        try:
            closes, highs, lows = df['close'].values, df['high'].values, df['low'].values
            sh = float(np.max(highs[-(lookback+2):-2]))
            sl = float(np.min(lows[-(lookback+2):-2]))
            lc, pc = closes[-1], closes[-2]
            if lc > sh and pc <= sh:
                return {'detected': True, 'direction': 'BULL', 'break_level': sh,
                        'breakout_strength': (lc - sh) / sh}
            if lc < sl and pc >= sl:
                return {'detected': True, 'direction': 'BEAR', 'break_level': sl,
                        'breakout_strength': (sl - lc) / sl}
        except Exception as e:
            logger.debug(f"Structure break error: {e}")
        return {'detected': False}

    def find_vwap_entry(self, df, signal_side, **kwargs):
        try:
            dt = df.tail(20)
            tp = (dt['high'] + dt['low'] + dt['close']) / 3
            vwap = (tp * dt['volume']).sum() / dt['volume'].sum()
            lc   = float(df['close'].iloc[-1])
            lo   = float(df['open'].iloc[-1])
            body = abs(lc - lo)
            if signal_side == 'BUY':
                e = max(vwap, lc - body * 0.50)
                return round(max(e, lc * 0.997), 8)
            else:
                e = min(vwap, lc + body * 0.50)
                return round(min(e, lc * 1.003), 8)
        except Exception as e:
            logger.debug(f"VWAP entry error: {e}")
        return float(df['close'].iloc[-1]) if df is not None and len(df) > 0 else 0.0

    def check_structure_still_valid(self, df, original_stop, signal_side):
        try:
            ll = float(df['low'].iloc[-1])
            lh = float(df['high'].iloc[-1])
            if signal_side == 'BUY':  return ll > original_stop
            else:                      return lh < original_stop
        except Exception:
            return False


class Strategy:
    def __init__(self):
        self.ml_brain        = MLBrain()
        self.last_decisions  = []
        self._spread_history = deque(maxlen=20)
        self.sweep_detector  = LiquiditySweepDetector()
        self.absorption      = AbsorptionDetector()
        self.structure       = StructureEngine()
        self._recent_signals: Dict[str, Dict] = {}

    # ── ALL EXISTING HELPER METHODS (detect_regime, calculate_atr, etc.) ──
    # (tere original file ke sab methods yahin hain — maine unko touch nahi kiya)

    def detect_regime(self, df, adx):
        # ... original code ...
        pass   # (tere file ka pura code yahin rahega)

    def calculate_atr(self, df, period=14):
        # ... original ...
        pass

    def _calculate_rsi(self, df, period=14):
        # ... original ...
        pass

    def _detect_trend(self, df_1h, last_price):
        # ... original ...
        pass

    def _spread_acceptable(self, spread_pct):
        # ... original ...
        pass

    def _is_spread_spike(self, spread_pct):
        # ... original ...
        pass

    def _build_features(self, candidate, last_price, atr):
        # ... original ...
        pass

    def _log_decision(self, symbol, win_prob, drivers):
        # ... original ...
        pass

    def _calc_sl_tp_trap(self, df, signal_side, sweep_result, atr):
        # ... original ...
        pass

    def _calc_sl_tp_cont(self, df, signal_side, structure_break, atr):
        # ... original ...
        pass

    def _register_signal_for_reentry(self, symbol, result):
        # ... original ...
        pass

    def _get_recent_signal(self, symbol):
        # ... original ...
        pass

    # ── EXISTING _eval_a_primary, _eval_a_secondary, _eval_b_primary, _eval_b_secondary, _eval_reentry ──
    # (sab tere original file mein hain — maine unko yahin chhod diya hai)

    # ... (tere original file ke sab _eval_* methods yahin paste kar dena — space bachane ke liye maine yahan skip kiya hai lekin tujhe full original code already hai)

    # ── NEW: MICRO_SCALP SETUP ─────────────────────────────────────────────────
    def _eval_micro_scalp(self, candidate, df, last_price, ofi, cvd, vol_ratio, win_prob):
        """$2.4 friendly MICRO_SCALP — tight 2:1 RR, low filters"""
        symbol = candidate.get('symbol', '?')

        sweep = self.sweep_detector.detect_bull_sweep(df) or self.sweep_detector.detect_bear_sweep(df)
        if not sweep or not sweep.get('detected', False):
            return None

        abs_score = self.absorption.score_absorption(df, cvd, ofi, sweep['direction'])
        if abs_score < 0.25:
            return None
        if vol_ratio < 1.10:
            return None
        if win_prob < 0.47:
            return None

        entry = last_price
        if sweep['direction'] == 'BULL':
            tp = entry * 1.008
            sl = entry * 0.996
        else:
            tp = entry * 0.992
            sl = entry * 1.004

        rr = 2.0
        conf = round(min(abs_score * 0.6 + (win_prob - 0.5) * 2 * 0.4, 1.0), 3)

        logger.info(f"[MICRO_SCALP] {symbol} {sweep['direction']} | abs={abs_score:.2f} vol={vol_ratio:.2f}x winP={win_prob:.3f} E={entry:.6f} RR={rr:.1f}")

        return SignalResult(
            signal=sweep['direction'],
            confidence=conf,
            setup_type=SETUP_MICRO_SCALP,
            entry_price=round(entry, 8),
            stop_loss=round(sl, 8),
            take_profit=round(tp, 8),
            rr_ratio=rr,
            win_prob=win_prob,
            sweep_detected=True,
            absorption_score=abs_score,
            forced_move_detected=sweep.get('forced_move', False),
            volume_spike_ratio=vol_ratio,
            structure_reclaim=True,
            symbol=symbol,
            timestamp=time.time(),
            setup_description=f"MICRO_SCALP: abs={abs_score:.2f} vol={vol_ratio:.2f}x",
            setup_tier=TIER_SECONDARY,
            size_multiplier=0.65,
        )

    # ── MAIN SIGNAL GENERATOR (updated) ───────────────────────────────────────
    def generate_signal_full(self, candidate, total_trades=0, account_balance=None):
        """Full v5.1 + MICRO_SCALP"""
        # ... (tere original file ka pura pre-flight gates, win_prob, regime calculation wala code yahin rahega — maine sirf end modify kiya)

        # ── Priority evaluation (existing 1 to 5) ─────────────────────────────
        # 1. Primary A
        r = self._eval_a_primary(...)   # tera original code
        if r:
            self._register_signal_for_reentry(symbol, r)
            return r

        # 2. Primary B
        # ... tera original ...

        # 3. Secondary A
        # ... tera original ...

        # 4. Secondary B
        # ... tera original ...

        # 5. Micro re-entry
        r = self._eval_reentry(symbol, df, last_price, ofi, cvd, atr, vol_ratio, win_prob)
        if r:
            return r

        # === NEW MICRO_SCALP (starvation / low capital mode) ===
        if MICRO_SCALP_ENABLED:
            r = self._eval_micro_scalp(candidate, df, last_price, ofi, cvd, vol_ratio, win_prob)
            if r:
                logger.info(f"MICRO_SCALP triggered for {symbol} | reason: {r.setup_description}")
                return r

        logger.debug(f"[{symbol}] v5.1: no setup qualified — HOLD")
        return None

    def build_orchestrator_payload(self, result):
        """v5.1: includes setup_tier and size_multiplier"""
        return {
            'signal': result.signal, 'confidence': result.confidence,
            'setup_type': result.setup_type, 'entry_price': result.entry_price,
            'stop_loss': result.stop_loss, 'take_profit': result.take_profit,
            'rr_ratio': result.rr_ratio, 'win_prob': result.win_prob,
            'sweep_detected': result.sweep_detected, 'absorption_score': result.absorption_score,
            'forced_move': result.forced_move_detected,
            'volume_spike_ratio': result.volume_spike_ratio,
            'structure_reclaim': result.structure_reclaim,
            'setup_description': result.setup_description,
            'timestamp': result.timestamp,
            'setup_tier': result.setup_tier,
            'size_multiplier': result.size_multiplier,
            'is_reentry': result.is_reentry,
            'parent_signal_ts': result.parent_signal_ts,
          }
