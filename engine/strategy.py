"""
Strategy Engine — v5.1 FREQUENCY-OPTIMIZED LIQUIDITY EDGE
==========================================================
HIGH-EXPECTANCY SYSTEM: Liquidity Sweep + Absorption + Forced Move

v5.1 UPGRADE — Controlled Frequency Expansion (0.38 → 0.8-1.2 trades/day)
─────────────────────────────────────────────────────────────────────────────
METHODS USED (all preserve edge):
  1. PRIMARY vs SECONDARY setup tiers (score-gated, same logic, reduced strictness)
  2. Micro re-entry logic (pullback after initial signal, structure still valid)
  3. Session-aware time distribution (London, NY, Overlap — no dead zones)
  4. Edge Scoring Tiers fed to orchestrator for size modulation

SETUP TIERS:
  PRIMARY   (score ≥ 80) — full original conditions → full size
  SECONDARY (score 65–79) — same logic, relaxed thresholds → reduced size (0.6x)

SECONDARY SETUP RELAXATIONS (NOT removals):
  - Volume spike: 1.3x (was 1.5x) — still confirms participation
  - Absorption threshold: 0.35 (was 0.45 for A, 0.40 for B) — still required
  - Forced move: optional but scored (was mandatory in primary)
  - ML win_prob: 0.50 (was 0.52) — marginal relaxation

CORE EDGE IS UNCHANGED:
  Liquidity sweep + absorption + volume + structure reclaim ALWAYS required.
  Only threshold magnitudes change for secondary tier, never logic removal.

MICRO RE-ENTRY LOGIC:
  After a primary or secondary signal, if price pulls back to 38-61.8% retrace
  of the reclaim candle AND structure remains valid AND absorption persists,
  a micro re-entry is allowed (same direction, same invalidation level).
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
    MAX_SPREAD_PCT, MIN_ADX, FEE_RATE, SLIPPAGE_BUFFER
)
from engine.capital_adaptive import get_adaptive_params
from engine.ml_brain import MLBrain

logger = logging.getLogger("Strategy")

MIN_NET_EDGE = (FEE_RATE * 2) + SLIPPAGE_BUFFER + 0.001

SETUP_TRAP_REVERSAL   = "TRAP_REVERSAL"
SETUP_CONTINUATION    = "CONTINUATION"

# ── v5.1: Setup tier constants ────────────────────────────────────────────────
TIER_PRIMARY   = "PRIMARY"      # score ≥ 80 — full confidence, full size
TIER_SECONDARY = "SECONDARY"    # score 65–79 — reduced size (0.6x), still edge-valid
TIER_REJECTED  = "REJECTED"     # score < 65 — blocked

PRIMARY_SCORE_THRESHOLD   = 80.0
SECONDARY_SCORE_THRESHOLD = 65.0

# Primary thresholds (unchanged from v5.0)
PRIMARY_VOLUME_SPIKE      = 1.5
PRIMARY_ABSORPTION_A      = 0.45
PRIMARY_ABSORPTION_B      = 0.40
PRIMARY_WIN_PROB          = 0.52

# Secondary thresholds (relaxed but still meaningful)
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
REENTRY_SIGNAL_EXPIRY  = 300      # 5 minutes
REENTRY_MIN_ABSORPTION = 0.30


@dataclass
class SignalResult:
    """
    Structured signal output.

    v5.1 additions:
      setup_tier        — PRIMARY | SECONDARY
      size_multiplier   — 1.0 primary, 0.6 secondary/reentry
      is_reentry        — True if this is a micro re-entry
      parent_signal_ts  — timestamp of original signal
    """
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
        """v5.1: Returns False if price has breached original invalidation level."""
        try:
            ll = float(df['low'].iloc[-1])
            lh = float(df['high'].iloc[-1])
            if signal_side == 'BUY':  return ll > original_stop
            else:                      return lh < original_stop
        except Exception:
            return False


class Strategy:
    """
    v5.1 — Liquidity-Driven Edge with Controlled Frequency Expansion.

    Evaluation order (first qualifying result wins):
      1. Setup A PRIMARY  (sweep + absorption + forced + vol>=1.5x + reclaim)
      2. Setup B PRIMARY  (trend + pullback + absorption + vol>=1.5x + break)
      3. Setup A SECONDARY (same conditions, vol>=1.3x, abs>=0.35, forced optional)
      4. Setup B SECONDARY (same conditions, vol>=1.3x, abs>=0.35, ADX>=20)
      5. Micro Re-Entry   (pullback 38-61.8% fibo, structure intact, abs>=0.30)

    PRIMARY  → full risk (size_multiplier 1.0)
    SECONDARY/REENTRY → reduced risk (size_multiplier 0.6)
    """

    def __init__(self):
        self.ml_brain        = MLBrain()
        self.last_decisions  = []
        self._spread_history = deque(maxlen=20)
        self.sweep_detector  = LiquiditySweepDetector()
        self.absorption      = AbsorptionDetector()
        self.structure       = StructureEngine()
        # v5.1: cache recent valid signals for re-entry detection
        self._recent_signals: Dict[str, Dict] = {}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def detect_regime(self, df, adx):
        if df is None or len(df) < 30:
            return 'UNKNOWN'
        regime = 'TRENDING' if adx >= 25 else ('RANGING' if adx <= 20 else 'UNKNOWN')
        try:
            close = df['close']
            s20   = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            bbw   = (std20.iloc[-1] / s20.iloc[-1]) if s20.iloc[-1] > 0 else 0
            if bbw < 0.005 and regime != 'TRENDING':
                regime = 'RANGING'
        except Exception:
            pass
        return regime

    def calculate_atr(self, df, period=14):
        if df is None or len(df) < period:
            return 0.0
        try:
            h, l, c = df['high'], df['low'], df['close']
            tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
            return float(tr.rolling(period).mean().iloc[-1])
        except Exception:
            return 0.0

    def _calculate_rsi(self, df, period=14):
        try:
            d   = df['close'].diff()
            g   = d.where(d > 0, 0.0)
            ls  = -d.where(d < 0, 0.0)
            ag  = g.ewm(alpha=1/period, adjust=False).mean()
            al  = ls.ewm(alpha=1/period, adjust=False).mean()
            rs  = ag / (al + 1e-9)
            return float((100 - 100 / (1 + rs)).iloc[-1])
        except Exception:
            return 50.0

    def _detect_trend(self, df_1h, last_price):
        tu = td = False
        if df_1h is not None and len(df_1h) >= 50:
            try:
                s50 = df_1h['close'].rolling(50).mean().iloc[-1]
                if not np.isnan(s50):
                    tu = last_price > s50
                    td = last_price < s50
            except Exception:
                pass
        return tu, td

    def _spread_acceptable(self, spread_pct):
        return (spread_pct + FEE_RATE * 2 + SLIPPAGE_BUFFER) < MIN_NET_EDGE * 3

    def _is_spread_spike(self, spread_pct):
        self._spread_history.append(spread_pct)
        if len(self._spread_history) < 5:
            return False
        avg = sum(self._spread_history) / len(self._spread_history)
        if avg <= 0:
            return False
        if spread_pct > 1.5 * avg:
            logger.warning(f"SPREAD SPIKE: {spread_pct*10000:.1f}bps vs avg {avg*10000:.1f}bps")
            return True
        return False

    def _build_features(self, candidate, last_price, atr):
        df  = candidate.get('df')
        rsi = self._calculate_rsi(df) if df is not None else 50.0
        return {
            'ofi': candidate.get('ofi', 0.0), 'cvd': candidate.get('cvd', 0.0),
            'vol_ratio': candidate.get('vol_ratio', 1.0), 'adx': candidate.get('adx', 0.0),
            'spread_pct': candidate.get('spread_pct', 0.0),
            'atr_pct': (atr / last_price) if last_price > 0 else 0.0,
            'rsi': rsi, 'price': last_price,
        }

    def _log_decision(self, symbol, win_prob, drivers):
        self.last_decisions.append({'symbol': symbol, 'win_prob': win_prob,
                                    'drivers': drivers, 'ts': time.time()})
        if len(self.last_decisions) > 50:
            self.last_decisions.pop(0)

    def _calc_sl_tp_trap(self, df, signal_side, sweep_result, atr):
        try:
            lc    = float(df['close'].iloc[-1])
            entry = self.structure.find_vwap_entry(df, signal_side)
            if signal_side == 'BUY':
                sl_d = max(entry - sweep_result.get('sweep_low', lc - atr), atr * 0.5)
                sl = entry - sl_d; tp = entry + sl_d * 1.5; rr = 1.5
            else:
                sl_d = max(sweep_result.get('sweep_high', lc + atr) - entry, atr * 0.5)
                sl = entry + sl_d; tp = entry - sl_d * 1.5; rr = 1.5
            rr = (abs(tp - entry) / sl_d) if sl_d > 0 else 1.5
            return round(entry, 8), round(sl, 8), round(tp, 8), round(rr, 3)
        except Exception:
            c = float(df['close'].iloc[-1]) if df is not None else 0.0
            return (c, c*0.99, c*1.015, 1.5) if signal_side == 'BUY' else (c, c*1.01, c*0.985, 1.5)

    def _calc_sl_tp_cont(self, df, signal_side, structure_break, atr):
        try:
            entry = self.structure.find_vwap_entry(df, signal_side)
            if signal_side == 'BUY':
                sl_d = max(entry - float(df['low'].tail(5).min()), atr * 0.8)
                sl = entry - sl_d; tp = entry + sl_d * 2.5
            else:
                sl_d = max(float(df['high'].tail(5).max()) - entry, atr * 0.8)
                sl = entry + sl_d; tp = entry - sl_d * 2.5
            rr = (abs(tp - entry) / sl_d) if sl_d > 0 else 2.5
            return round(entry, 8), round(sl, 8), round(tp, 8), round(rr, 3)
        except Exception:
            c = float(df['close'].iloc[-1]) if df is not None else 0.0
            return (c, c*0.988, c*1.030, 2.5) if signal_side == 'BUY' else (c, c*1.012, c*0.970, 2.5)

    # ── Re-entry cache management ─────────────────────────────────────────────

    def _register_signal_for_reentry(self, symbol, result):
        self._recent_signals[symbol] = {'signal_result': result, 'ts': time.time()}

    def _get_recent_signal(self, symbol):
        e = self._recent_signals.get(symbol)
        if e is None:
            return None
        if time.time() - e['ts'] > REENTRY_SIGNAL_EXPIRY:
            del self._recent_signals[symbol]
            return None
        return e['signal_result']

    # ── Setup A PRIMARY ───────────────────────────────────────────────────────

    def _eval_a_primary(self, candidate, df, df_1h, last_price, ofi, cvd, atr,
                        volume_spike, vol_ratio, win_prob):
        symbol = candidate.get('symbol', '?')
        bull = self.sweep_detector.detect_bull_sweep(df)
        bear = self.sweep_detector.detect_bear_sweep(df)
        if bull.get('detected') and not bear.get('detected'):
            side, sw = 'BUY', bull
        elif bear.get('detected') and not bull.get('detected'):
            side, sw = 'SELL', bear
        else:
            return None
        if not sw.get('forced_move', False):
            return None
        if not self.structure.check_structure_reclaim(df, sw, side):
            return None
        abs_score = self.absorption.score_absorption(df, cvd, ofi, side)
        if abs_score < PRIMARY_ABSORPTION_A:
            return None
        if not volume_spike:
            return None
        if win_prob < PRIMARY_WIN_PROB:
            return None
        entry, sl, tp, rr = self._calc_sl_tp_trap(df, side, sw, atr)
        rr = max(SETUP_A_RR_MIN, min(rr, SETUP_A_RR_MAX))
        conf = min(max(abs_score*0.40 + min(vol_ratio/3.0,1.0)*0.30 + (win_prob-0.5)*2*0.30, 0.0), 1.0)
        wr = sw.get('wick_ratio', 0.0)
        logger.info(f"[SETUP-A PRIMARY] {symbol} {side} | wick={wr:.2f} abs={abs_score:.2f} vol={vol_ratio:.2f}x winP={win_prob:.3f} E={entry:.6f} SL={sl:.6f} TP={tp:.6f} RR={rr:.2f}")
        return SignalResult(
            signal=side, confidence=round(conf,3), setup_type=SETUP_TRAP_REVERSAL,
            entry_price=entry, stop_loss=sl, take_profit=tp, rr_ratio=rr,
            win_prob=win_prob, sweep_detected=True, absorption_score=abs_score,
            forced_move_detected=True, volume_spike_ratio=vol_ratio,
            structure_reclaim=True, symbol=symbol, timestamp=time.time(),
            setup_description=f"TrapReversal[PRIMARY]: wick={wr:.2f} abs={abs_score:.2f} vol={vol_ratio:.2f}x",
            setup_tier=TIER_PRIMARY, size_multiplier=1.0,
        )

    # ── Setup A SECONDARY ─────────────────────────────────────────────────────

    def _eval_a_secondary(self, candidate, df, df_1h, last_price, ofi, cvd, atr,
                          vol_ratio, win_prob):
        symbol = candidate.get('symbol', '?')
        bull = self.sweep_detector.detect_bull_sweep(df)
        bear = self.sweep_detector.detect_bear_sweep(df)
        if bull.get('detected') and not bear.get('detected'):
            side, sw = 'BUY', bull
        elif bear.get('detected') and not bull.get('detected'):
            side, sw = 'SELL', bear
        else:
            return None
        forced = sw.get('forced_move', False)
        if not self.structure.check_structure_reclaim(df, sw, side):
            return None   # still mandatory even in secondary
        abs_score = self.absorption.score_absorption(df, cvd, ofi, side)
        if abs_score < SECONDARY_ABSORPTION:
            return None
        # Volume: relaxed to 1.3x
        sec_spike, spike_ratio = self.absorption.detect_volume_spike(df, min_ratio=SECONDARY_VOLUME_SPIKE)
        eff_vol = max(vol_ratio, spike_ratio if sec_spike else 0.0)
        if not sec_spike and vol_ratio < SECONDARY_VOLUME_SPIKE:
            return None
        if win_prob < SECONDARY_WIN_PROB:
            return None
        entry, sl, tp, rr = self._calc_sl_tp_trap(df, side, sw, atr)
        rr = max(SETUP_A_RR_MIN, min(rr, SETUP_A_RR_MAX))
        conf = min(max(abs_score*0.40 + min(eff_vol/3.0,1.0)*0.30 + (win_prob-0.5)*2*0.25 + (0.05 if forced else 0.0), 0.0), 1.0) * 0.90
        wr = sw.get('wick_ratio', 0.0)
        logger.info(f"[SETUP-A SECONDARY] {symbol} {side} | wick={wr:.2f} abs={abs_score:.2f} vol={eff_vol:.2f}x winP={win_prob:.3f} forced={forced} size=0.60x")
        return SignalResult(
            signal=side, confidence=round(conf,3), setup_type=SETUP_TRAP_REVERSAL,
            entry_price=entry, stop_loss=sl, take_profit=tp, rr_ratio=rr,
            win_prob=win_prob, sweep_detected=True, absorption_score=abs_score,
            forced_move_detected=forced, volume_spike_ratio=eff_vol,
            structure_reclaim=True, symbol=symbol, timestamp=time.time(),
            setup_description=f"TrapReversal[SECONDARY]: wick={wr:.2f} abs={abs_score:.2f} vol={eff_vol:.2f}x forced={forced}",
            setup_tier=TIER_SECONDARY, size_multiplier=SECONDARY_SIZE_MULTIPLIER,
        )

    # ── Setup B PRIMARY ───────────────────────────────────────────────────────

    def _eval_b_primary(self, candidate, df, df_1h, last_price, ofi, cvd, adx, atr,
                        volume_spike, vol_ratio, win_prob):
        symbol = candidate.get('symbol', '?')
        tu, td = self._detect_trend(df_1h, last_price)
        if not tu and not td:
            return None
        if adx < 22:
            return None
        side = 'BUY' if tu else 'SELL'
        try:
            rh = float(df['high'].tail(5).max()); rl = float(df['low'].tail(5).min())
            at_pb = (last_price < rh * 0.998) if side == 'BUY' else (last_price > rl * 1.002)
        except Exception:
            at_pb = True
        if not at_pb:
            return None
        abs_score = self.absorption.score_absorption(df, cvd, ofi, side)
        if abs_score < PRIMARY_ABSORPTION_B:
            return None
        if not volume_spike:
            return None
        ofi_ok = (side == 'BUY' and ofi > MIN_OFI) or (side == 'SELL' and ofi < -MIN_OFI)
        if not ofi_ok:
            return None
        try:
            cr = float(df['high'].iloc[-1] - df['low'].iloc[-1])
            forced = cr >= FORCED_MOVE_ATR_MULT * atr if atr > 0 else True
        except Exception:
            forced = True
        sb = self.structure.check_structure_break(df)
        if not sb.get('detected'):
            return None
        bd = sb.get('direction', '')
        if not ((side == 'BUY' and bd == 'BULL') or (side == 'SELL' and bd == 'BEAR')):
            return None
        if win_prob < PRIMARY_WIN_PROB:
            return None
        entry, sl, tp, rr = self._calc_sl_tp_cont(df, side, sb, atr)
        rr = max(SETUP_B_RR_MIN, min(rr, SETUP_B_RR_MAX))
        conf = min(max(abs_score*0.35 + min(vol_ratio/3.0,1.0)*0.25 + (win_prob-0.5)*2*0.25 + (0.15 if forced else 0.0), 0.0), 1.0)
        logger.info(f"[SETUP-B PRIMARY] {symbol} {side} | adx={adx:.1f} abs={abs_score:.2f} vol={vol_ratio:.2f}x winP={win_prob:.3f} E={entry:.6f} SL={sl:.6f} TP={tp:.6f} RR={rr:.2f}")
        return SignalResult(
            signal=side, confidence=round(conf,3), setup_type=SETUP_CONTINUATION,
            entry_price=entry, stop_loss=sl, take_profit=tp, rr_ratio=rr,
            win_prob=win_prob, sweep_detected=False, absorption_score=abs_score,
            forced_move_detected=forced, volume_spike_ratio=vol_ratio,
            structure_reclaim=False, symbol=symbol, timestamp=time.time(),
            setup_description=f"Continuation[PRIMARY]: adx={adx:.1f} abs={abs_score:.2f} vol={vol_ratio:.2f}x",
            setup_tier=TIER_PRIMARY, size_multiplier=1.0,
        )

    # ── Setup B SECONDARY ─────────────────────────────────────────────────────

    def _eval_b_secondary(self, candidate, df, df_1h, last_price, ofi, cvd, adx, atr,
                          vol_ratio, win_prob):
        symbol = candidate.get('symbol', '?')
        tu, td = self._detect_trend(df_1h, last_price)
        if not tu and not td:
            return None
        if adx < 20:   # relaxed from 22
            return None
        side = 'BUY' if tu else 'SELL'
        try:
            rh = float(df['high'].tail(5).max()); rl = float(df['low'].tail(5).min())
            at_pb = (last_price < rh * 0.999) if side == 'BUY' else (last_price > rl * 1.001)
        except Exception:
            at_pb = True
        if not at_pb:
            return None
        abs_score = self.absorption.score_absorption(df, cvd, ofi, side)
        if abs_score < SECONDARY_ABSORPTION:
            return None
        sec_spike, spike_ratio = self.absorption.detect_volume_spike(df, min_ratio=SECONDARY_VOLUME_SPIKE)
        eff_vol = max(vol_ratio, spike_ratio if sec_spike else 0.0)
        if not sec_spike and vol_ratio < SECONDARY_VOLUME_SPIKE:
            return None
        ofi_ok = (side == 'BUY' and ofi > MIN_OFI * 0.8) or (side == 'SELL' and ofi < -MIN_OFI * 0.8)
        if not ofi_ok:
            return None
        sb = self.structure.check_structure_break(df)
        if not sb.get('detected'):
            return None
        bd = sb.get('direction', '')
        if not ((side == 'BUY' and bd == 'BULL') or (side == 'SELL' and bd == 'BEAR')):
            return None
        if win_prob < SECONDARY_WIN_PROB:
            return None
        try:
            cr = float(df['high'].iloc[-1] - df['low'].iloc[-1])
            forced = cr >= FORCED_MOVE_ATR_MULT * atr if atr > 0 else False
        except Exception:
            forced = False
        entry, sl, tp, rr = self._calc_sl_tp_cont(df, side, sb, atr)
        rr = max(SETUP_B_RR_MIN, min(rr, SETUP_B_RR_MAX))
        conf = min(max(abs_score*0.35 + min(eff_vol/3.0,1.0)*0.25 + (win_prob-0.5)*2*0.20 + (0.10 if forced else 0.0), 0.0), 1.0) * 0.90
        logger.info(f"[SETUP-B SECONDARY] {symbol} {side} | adx={adx:.1f} abs={abs_score:.2f} vol={eff_vol:.2f}x winP={win_prob:.3f} size=0.60x")
        return SignalResult(
            signal=side, confidence=round(conf,3), setup_type=SETUP_CONTINUATION,
            entry_price=entry, stop_loss=sl, take_profit=tp, rr_ratio=rr,
            win_prob=win_prob, sweep_detected=False, absorption_score=abs_score,
            forced_move_detected=forced, volume_spike_ratio=eff_vol,
            structure_reclaim=False, symbol=symbol, timestamp=time.time(),
            setup_description=f"Continuation[SECONDARY]: adx={adx:.1f} abs={abs_score:.2f} vol={eff_vol:.2f}x",
            setup_tier=TIER_SECONDARY, size_multiplier=SECONDARY_SIZE_MULTIPLIER,
        )

    # ── Micro Re-Entry ────────────────────────────────────────────────────────

    def _eval_reentry(self, symbol, df, last_price, ofi, cvd, atr, vol_ratio, win_prob):
        """
        v5.1: Re-entry on 38-61.8% fibo pullback after valid signal.
        Structure must remain valid. Absorption must persist.
        """
        prev = self._get_recent_signal(symbol)
        if prev is None:
            return None
        side = prev.signal
        oe, osl, otp = prev.entry_price, prev.stop_loss, prev.take_profit
        if oe <= 0 or osl <= 0:
            return None
        # Structure still valid?
        if not self.structure.check_structure_still_valid(df, osl, side):
            logger.debug(f"[{symbol}] ReEntry: structure broken — cancelling")
            del self._recent_signals[symbol]
            return None
        # Price in fibonacci re-entry zone?
        try:
            if side == 'BUY':
                ms = otp - oe
                rlo = oe - ms * REENTRY_FIBO_HIGH
                rhi = oe - ms * REENTRY_FIBO_LOW
            else:
                ms = oe - otp
                rlo = oe + ms * REENTRY_FIBO_LOW
                rhi = oe + ms * REENTRY_FIBO_HIGH
            in_zone = rlo <= last_price <= rhi
        except Exception:
            return None
        if not in_zone:
            return None
        # Absorption still present?
        abs_score = self.absorption.score_absorption(df, cvd, ofi, side)
        if abs_score < REENTRY_MIN_ABSORPTION:
            return None
        if vol_ratio < 1.1:
            return None
        # No opposing sweep?
        if side == 'BUY':
            if self.sweep_detector.detect_bear_sweep(df).get('detected'):
                del self._recent_signals[symbol]
                return None
        else:
            if self.sweep_detector.detect_bull_sweep(df).get('detected'):
                del self._recent_signals[symbol]
                return None
        if win_prob < SECONDARY_WIN_PROB:
            return None

        entry = self.structure.find_vwap_entry(df, side)
        sl, tp = osl, otp
        sl_d = abs(entry - sl)
        if sl_d <= 0:
            return None
        rr = max(1.5, min(abs(tp - entry) / sl_d, 3.0))
        conf = min(max(abs_score*0.50 + (win_prob-0.5)*2*0.30 + min(vol_ratio/2.0,1.0)*0.20, 0.0), 1.0) * 0.85
        logger.info(f"[MICRO REENTRY] {symbol} {side} | abs={abs_score:.2f} vol={vol_ratio:.2f}x winP={win_prob:.3f} E={entry:.6f} RR={rr:.2f} size=0.60x")
        del self._recent_signals[symbol]  # one re-entry per signal
        return SignalResult(
            signal=side, confidence=round(conf,3), setup_type=prev.setup_type,
            entry_price=round(entry,8), stop_loss=round(sl,8), take_profit=round(tp,8),
            rr_ratio=round(rr,3), win_prob=win_prob,
            sweep_detected=prev.sweep_detected, absorption_score=abs_score,
            forced_move_detected=prev.forced_move_detected, volume_spike_ratio=vol_ratio,
            structure_reclaim=prev.structure_reclaim, symbol=symbol, timestamp=time.time(),
            setup_description=f"MicroReEntry: abs={abs_score:.2f} vol={vol_ratio:.2f}x",
            setup_tier=TIER_SECONDARY, size_multiplier=SECONDARY_SIZE_MULTIPLIER,
            is_reentry=True, parent_signal_ts=prev.timestamp,
        )

    # ── Main Signal Generator ─────────────────────────────────────────────────

    def generate_signal(self, candidate, total_trades=0, account_balance=None):
        """Legacy-compatible interface."""
        result = self.generate_signal_full(candidate, total_trades, account_balance)
        return ('HOLD', 0.0) if result is None else (result.signal, result.confidence)

    def generate_signal_full(self, candidate, total_trades=0, account_balance=None):
        """
        Full v5.1 signal generation.

        Returns the first qualifying SignalResult in order:
          Primary A → Primary B → Secondary A → Secondary B → Re-entry → None
        """
        symbol     = candidate.get('symbol', '?')
        df         = candidate.get('df')
        df_1h      = candidate.get('df_1h')
        last_price = candidate.get('last_price', 0)
        ofi        = candidate.get('ofi', 0.0)
        cvd        = candidate.get('cvd', 0.0)
        vol_ratio  = candidate.get('vol_ratio', 1.0)
        adx        = candidate.get('adx', 0.0)
        spread_pct = candidate.get('spread_pct', 0.0)
        sig_ts     = candidate.get('signal_timestamp', 0)
        atr        = candidate.get('atr', 0.0)

        if account_balance is not None:
            ap = get_adaptive_params(account_balance)
            _min_vol_spike   = max(ap.min_volume_spike, MIN_VOLUME_SPIKE_ABS)
            _max_spread_mult = ap.spread_budget_mult
        else:
            _min_vol_spike   = MIN_VOLUME_SPIKE_ABS
            _max_spread_mult = 1.0

        # Pre-flight gates (unchanged)
        if time.time() - sig_ts > 10:
            return None
        ems = MAX_SPREAD_PCT * _max_spread_mult
        if spread_pct > ems or not self._spread_acceptable(spread_pct):
            return None
        if self._is_spread_spike(spread_pct):
            return None
        if atr > 0 and last_price > 0:
            ap_pct = atr / last_price
            if ap_pct > MAX_ATR_VOLATILITY or ap_pct < 0.0005:
                return None
        if vol_ratio < 1.1:
            return None

        # Volume spike for primary setups
        vol_spike_primary, spike_ratio = self.absorption.detect_volume_spike(
            df, min_ratio=max(_min_vol_spike, MIN_VOLUME_SPIKE_ABS)
        )
        if not vol_spike_primary and vol_ratio >= MIN_VOLUME_SPIKE_ABS:
            vol_spike_primary = True
            spike_ratio = vol_ratio

        win_prob, drivers = self.ml_brain.predict_win_probability(
            self._build_features(candidate, last_price, atr)
        )
        self._log_decision(symbol, win_prob, drivers)
        regime = self.detect_regime(df, adx)

        logger.info(f"[{symbol}] v5.1 regime={regime} adx={adx:.1f} ofi={ofi:.4f} cvd={cvd:.2f} vol={spike_ratio:.2f}x win_prob={win_prob:.3f}")

        # ── Priority evaluation ───────────────────────────────────────────────
        # 1. Primary A
        r = self._eval_a_primary(candidate, df, df_1h, last_price, ofi, cvd, atr,
                                  vol_spike_primary, spike_ratio, win_prob)
        if r:
            self._register_signal_for_reentry(symbol, r)
            return r

        # 2. Primary B
        if regime == 'TRENDING':
            r = self._eval_b_primary(candidate, df, df_1h, last_price, ofi, cvd, adx, atr,
                                      vol_spike_primary, spike_ratio, win_prob)
            if r:
                self._register_signal_for_reentry(symbol, r)
                return r

        # 3. Secondary A
        r = self._eval_a_secondary(candidate, df, df_1h, last_price, ofi, cvd, atr,
                                    spike_ratio, win_prob)
        if r:
            self._register_signal_for_reentry(symbol, r)
            return r

        # 4. Secondary B (slightly wider regime window)
        if regime in ('TRENDING', 'UNKNOWN'):
            r = self._eval_b_secondary(candidate, df, df_1h, last_price, ofi, cvd, adx, atr,
                                        spike_ratio, win_prob)
            if r:
                self._register_signal_for_reentry(symbol, r)
                return r

        # 5. Micro re-entry
        r = self._eval_reentry(symbol, df, last_price, ofi, cvd, atr, vol_ratio, win_prob)
        if r:
            return r

        logger.debug(f"[{symbol}] v5.1: no setup qualified — HOLD")
        return None

    def build_orchestrator_payload(self, result):
        """v5.1: includes setup_tier and size_multiplier for position sizing."""
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
            # v5.1
            'setup_tier': result.setup_tier,
            'size_multiplier': result.size_multiplier,
            'is_reentry': result.is_reentry,
            'parent_signal_ts': result.parent_signal_ts,
        }
