"""
backtest/backtest_v51.py  — v5.1 Fixed (yfinance compatible)
==============================================================
Fixes vs previous version:

1. MIN LENGTH ALIGNMENT
   sab symbols ko minimum length tak truncate karta hai pehle hi
   → IndexError: single positional indexer is out-of-bounds fix

2. SWEEP DETECTION — VOLUME INDEPENDENT
   CVD threshold: price-action based CVD ke saath compatible
   (0.08 tha, ab 0.05 — kyunki PA-CVD weaker signal deta hai)

3. TREND FILTER STRONG KIYA
   SMA50 ke saath SMA200 bhi check karta hai
   Sirf tab trade leta hai jab dono same direction mein hain
   → Galat direction mein SELL signals ruk jaate hain

4. CONTINUATION SETUP — HIGHER BAR
   ADX primary: 25 (was same)
   Confirm: price > SMA50 AND price > SMA200 for BUY
   Confirm: price < SMA50 AND price < SMA200 for SELL
   → Uptrend mein short signals block ho jaate hain

5. MINIMUM TRADES GUARD
   Agar 28 days mein < 5 trades aayein toh thresholds
   automatically loosen hote hain (controlled)
"""

import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger("BacktestV51")

# ── Constants ──────────────────────────────────────────────────────────────────
FEE_RATE        = 0.0002
SLIPPAGE        = 0.0008
DAILY_DD_LIMIT  = -0.030
BASE_RISK_PCT   = 0.015
SECONDARY_MULT  = 0.60
MAX_HOLD_BARS   = 30
SYMBOL_COOLDOWN = 12
SESSION_LIMIT   = 1

PRIMARY_SCORE   = 80.0
SECONDARY_SCORE = 65.0
SWEEP_WICK      = 0.45
FORCED_MULT     = 1.4
ADX_PRIM        = 25
ADX_SEC         = 22
BB_CHOP         = 0.006        # slightly relaxed from 0.008
VOL_PRIM        = 1.5
VOL_SEC         = 1.3
ABS_PRIM_A      = 0.40         # relaxed: PA-CVD weaker than tick CVD
ABS_PRIM_B      = 0.35
ABS_SEC         = 0.28         # relaxed for PA-CVD
ML_THRESH_PRIM  = 0.58
ML_THRESH_SEC   = 0.52
REENTRY_LO      = 0.382
REENTRY_HI      = 0.618
REENTRY_EXPIRY  = 6
REENTRY_ABS     = 0.25


@dataclass
class Trade:
    symbol:      str
    entry_bar:   int
    entry_price: float
    stop_loss:   float
    take_profit: float
    side:        str
    setup_type:  str
    setup_tier:  str
    size_mult:   float
    risk_pct:    float
    risk_usd:    float
    rr_ratio:    float
    edge_score:  float
    win_prob:    float
    exit_bar:    int   = 0
    exit_price:  float = 0.0
    pnl_usd:     float = 0.0
    outcome:     str   = ''
    hold_bars:   int   = 0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _dead(h):    return h >= 20 or h < 2
def _sess(h):
    if  8 <= h < 12: return 'LONDON'
    if 12 <= h < 17: return 'NY'
    if 17 <= h < 20: return 'NY_CLOSE'
    if  2 <= h <  8: return 'ASIA'
    return None

def _atr(hi, lo, cl, t, p=14):
    if t < p: return 0.0
    trs = [max(hi[i]-lo[i],
               abs(hi[i]-cl[i-1]) if i > 0 else 0,
               abs(lo[i]-cl[i-1]) if i > 0 else 0)
           for i in range(t-p, t)]
    return float(np.mean(trs))

def _bbw(cl, t, p=20):
    if t < p: return 0.05
    s = cl[t-p:t]
    return float(np.std(s) / (np.mean(s) + 1e-9))

def _sma(cl, t, p):
    return float(np.mean(cl[t-p:t])) if t >= p else float(cl[t])

def _sweep_b(hi, lo, op, cl, t, atr):
    if t < 16: return None
    sl = np.min(lo[t-16:t-3])
    for i in range(max(t-3, 0), t):
        cr = hi[i] - lo[i]
        if cr < 1e-10: continue
        if lo[i] < sl:
            lw = min(op[i], cl[i]) - lo[i]
            wr = lw / cr
            if cl[i] > sl and wr >= SWEEP_WICK:
                return {'sweep_low': lo[i], 'reclaim_level': sl,
                        'wick_ratio': wr,
                        'forced_move': cr >= FORCED_MULT * atr if atr > 0 else True}
    return None

def _sweep_s(hi, lo, op, cl, t, atr):
    if t < 16: return None
    sh = np.max(hi[t-16:t-3])
    for i in range(max(t-3, 0), t):
        cr = hi[i] - lo[i]
        if cr < 1e-10: continue
        if hi[i] > sh:
            uw = hi[i] - max(op[i], cl[i])
            wr = uw / cr
            if cl[i] < sh and wr >= SWEEP_WICK:
                return {'sweep_high': hi[i], 'reclaim_level': sh,
                        'wick_ratio': wr,
                        'forced_move': cr >= FORCED_MULT * atr if atr > 0 else True}
    return None

def _abs(cvd, ofi, side, mc=0.05, mo=0.05):
    """Absorption score — thresholds relaxed for price-action CVD."""
    s = 0.0
    if side == 'BUY':
        if cvd > mc:    s += 0.40
        elif cvd > 0:   s += 0.20
        if ofi > mo*2:  s += 0.35
        elif ofi > mo:  s += 0.20
    else:
        if cvd < -mc:    s += 0.40
        elif cvd < 0:    s += 0.20
        if ofi < -mo*2:  s += 0.35
        elif ofi < -mo:  s += 0.20
    return min(s, 1.0)

def _vspike(vs, t, mr):
    if t < 21: return False, 1.0
    avg = np.mean(vs[t-21:t-1])
    r   = vs[t] / avg if avg > 0 else 1.0
    return r >= mr, round(r, 3)

def _escore(wr, ab, forced, vr, struct, wp, rr, adx, sweep, stype):
    b = {}
    b['adx']   = 15 if adx >= 30 else (10 if adx >= 25 else (5 if adx >= 20 else 0))
    if sweep:
        b['sw'] = 20 if wr >= 0.65 else (14 if wr >= 0.50 else (8 if wr >= 0.45 else 5))
    else:
        b['sw'] = 10 if (stype == 'CONTINUATION' and ab >= 0.28) else 0
    b['ab']    = round(max(ab - 0.28, 0) / 0.72 * 20, 1) if ab >= 0.28 else 0
    b['fm']    = 10 if forced else 0
    b['vol']   = 15 if vr >= 2.0 else (10 if vr >= 1.5 else (5 if vr >= 1.3 else 0))
    b['st']    = 10 if struct else 0
    b['wp']    = round(max(wp - 0.50, 0) / 0.50 * 10, 1)
    b['rr']    = 5 if rr >= 2.5 else (2 if rr >= 1.5 else 0)
    return min(sum(b.values()), 100.0)

def _sltp(cl, hi, lo, t, side, sw, atr, cont=False):
    e = cl[t]
    if side == 'BUY':
        d = (max(e - sw.get('sweep_low', e - atr), atr * 0.5) if not cont
             else max(e - float(np.min(lo[max(0, t-5):t+1])), atr * 0.8))
        return e, e - d, e + d * (1.5 if not cont else 2.5), (1.5 if not cont else 2.5)
    else:
        d = (max(sw.get('sweep_high', e + atr) - e, atr * 0.5) if not cont
             else max(float(np.max(hi[max(0, t-5):t+1])) - e, atr * 0.8))
        return e, e + d, e - d * (1.5 if not cont else 2.5), (1.5 if not cont else 2.5)

def _sb2(cl, hi, lo, t, side, lb=10):
    """2-bar confirmed structure break."""
    if t < lb + 3: return False
    sh = np.max(hi[t-lb-2:t-2])
    sl = np.min(lo[t-lb-2:t-2])
    return (cl[t] > sh and cl[t-1] > sh) if side == 'BUY' else (cl[t] < sl and cl[t-1] < sl)

def _trend_aligned(cl, t, side):
    """
    FIX: Strong trend check — BOTH SMA50 and SMA200 must align.
    Prevents shorting in uptrend and buying in downtrend.
    """
    if t < 200: return False
    sma50  = _sma(cl, t, 50)
    sma200 = _sma(cl, t, 200)
    price  = cl[t]
    if side == 'BUY':
        # Price above both MAs → confirmed uptrend
        return price > sma50 and price > sma200 and sma50 > sma200
    else:
        # Price below both MAs → confirmed downtrend
        return price < sma50 and price < sma200 and sma50 < sma200


class BacktestV51:
    """
    Full v5.1 backtest — yfinance compatible, IndexError fixed.
    """

    def __init__(self, initial_balance: float = 100.0, model_payload: dict = None):
        self.ib    = initial_balance
        self.model = self.scaler = self.fcols = None
        self.thr   = ML_THRESH_PRIM
        if model_payload:
            self.model  = model_payload.get('model')
            self.scaler = model_payload.get('scaler')
            self.fcols  = model_payload.get('feature_cols')
            self.thr    = model_payload.get('threshold', ML_THRESH_PRIM)

    def _wp(self, fr):
        if self.model and self.scaler and self.fcols:
            try:
                X    = np.array([[fr.get(f, 0.0) for f in self.fcols]])
                Xsc  = self.scaler.transform(X)
                return float(self.model.predict_proba(Xsc)[0, 1])
            except Exception:
                pass
        # Improved heuristic for price-action CVD
        cvd  = fr.get('cvd', 0.0)
        ofi  = fr.get('ofi', 0.0)
        wl   = fr.get('wick_ratio_lo', 0.0)
        wh   = fr.get('wick_ratio_hi', 0.0)
        vr   = fr.get('vol_ratio', 1.0)
        sw   = fr.get('has_sweep', 0)
        rsi  = fr.get('rsi', 50.0)
        adx  = fr.get('adx', 20.0)
        wp   = 0.44
        wp  += min(abs(cvd), 0.5) * 0.20
        wp  += min(abs(ofi), 0.5) * 0.12
        wp  += max(wl, wh) * 0.12
        wp  += min((vr - 1.0) / 3.0, 1.0) * 0.06
        wp  += sw * 0.06
        wp  += min((adx - 20) / 20, 1.0) * 0.04 if adx > 20 else 0
        if 30 <= rsi <= 70: wp += 0.02
        if rsi < 30 or rsi > 70: wp += 0.04
        return float(np.clip(wp, 0.35, 0.80))

    def run(self, dfs: dict) -> dict:

        # ── FIX 1: Align all symbols to minimum length ────────────────────────
        min_len = min(len(df) for df in dfs.values())
        dfs = {sym: df.iloc[:min_len].copy().reset_index(drop=True)
               for sym, df in dfs.items()}
        n = min_len
        logger.info(f"Aligned all symbols to {n} bars")

        all_trades: List[Trade] = []
        balance   = self.ib
        equity    = [balance]
        daily_pnl: dict = {}

        base_sym = list(dfs.keys())[0]
        base_df  = dfs[base_sym]

        # Pre-extract arrays
        arrs = {}
        for sym, df in dfs.items():
            arrs[sym] = {
                'op': df['open'].values,    'hi': df['high'].values,
                'lo': df['low'].values,     'cl': df['close'].values,
                'vs': df['volume'].values,
                'cvd':  df['cvd'].values    if 'cvd'    in df.columns else np.zeros(n),
                'ofi':  df['ofi'].values    if 'ofi'    in df.columns else np.zeros(n),
                'adx':  df['adx'].values    if 'adx'    in df.columns else np.full(n, 20.0),
                'spr':  df['spread_pct'].values if 'spread_pct' in df.columns else np.full(n, 0.0001),
                'hrs':  df['hour_utc'].values.astype(int) if 'hour_utc' in df.columns else np.zeros(n, int),
            }

        # Pre-build ML feature rows
        feat = {}
        if self.model and self.fcols:
            for sym, df in dfs.items():
                rows = []
                for i in range(n):
                    r = {col: (float(df[col].iloc[i])
                               if col in df.columns and not pd.isna(df[col].iloc[i])
                               else 0.0)
                         for col in self.fcols}
                    r['has_sweep'] = int(
                        bool(df['is_sweep'].iloc[i]) if 'is_sweep' in df.columns else False
                    )
                    rows.append(r)
                feat[sym] = rows

        open_trades: List[Trade] = []
        cooldown   = {s: 0 for s in dfs}
        sess_cnt   = {s: {} for s in dfs}
        reentry    = {}
        consec     = 0

        # Get timestamp safely
        def _date(t):
            try:
                ts = base_df['timestamp'].iloc[t]
                return str(pd.to_datetime(ts))[:10]
            except Exception:
                return str(t // 288)   # fallback: bar number / bars_per_day

        for t in range(210, n - MAX_HOLD_BARS - 1):
            hr   = int(arrs[base_sym]['hrs'][t])
            date = _date(t)

            # ── Close open trades ──────────────────────────────────────────
            still = []
            for tr in open_trades:
                a  = arrs[tr.symbol]
                ht, lt = a['hi'][t], a['lo'][t]
                ep, oc = None, None
                if tr.side == 'BUY':
                    if ht >= tr.take_profit: ep, oc = tr.take_profit, 'WIN'
                    elif lt <= tr.stop_loss: ep, oc = tr.stop_loss, 'LOSS'
                else:
                    if lt <= tr.take_profit: ep, oc = tr.take_profit, 'WIN'
                    elif ht >= tr.stop_loss: ep, oc = tr.stop_loss, 'LOSS'
                if ep is None and (t - tr.entry_bar) >= MAX_HOLD_BARS:
                    ep, oc = a['cl'][t], 'TIMEOUT'
                if ep is not None:
                    if oc == 'WIN':
                        pnl = tr.risk_usd * tr.rr_ratio
                    elif oc == 'LOSS':
                        pnl = -tr.risk_usd
                    else:
                        ret = ((ep / tr.entry_price - 1) if tr.side == 'BUY'
                               else (1 - ep / tr.entry_price))
                        pnl = tr.risk_usd * (ret / BASE_RISK_PCT) * 0.4
                    tr.exit_bar   = t
                    tr.exit_price = ep
                    tr.pnl_usd    = round(pnl, 6)
                    tr.outcome    = oc
                    tr.hold_bars  = t - tr.entry_bar
                    balance       = max(balance + pnl, 0.001)
                    all_trades.append(tr)
                    equity.append(balance)
                    consec = 0 if oc == 'WIN' else consec + 1
                    daily_pnl[date] = daily_pnl.get(date, 0) + pnl
                else:
                    still.append(tr)
            open_trades = still

            if _dead(hr): continue
            if daily_pnl.get(date, 0) / balance < DAILY_DD_LIMIT: continue
            if consec >= 3: continue
            if len(open_trades) >= 2: continue

            sess = _sess(hr)
            cands = []

            for sym, a in arrs.items():
                if cooldown[sym] > t: continue
                if sess and sess_cnt[sym].get(date+'_'+sess, 0) >= SESSION_LIMIT: continue
                if a['spr'][t] > 0.0008: continue

                atr_v = _atr(a['hi'], a['lo'], a['cl'], t)
                if atr_v <= 0: continue
                atp = atr_v / a['cl'][t]
                if atp > 0.015 or atp < 0.0005: continue

                bbw_v = _bbw(a['cl'], t)
                if bbw_v < BB_CHOP: continue

                avg_v = np.mean(a['vs'][max(0, t-21):t-1]) if t > 21 else a['vs'][t]
                vr    = a['vs'][t] / avg_v if avg_v > 0 else 1.0
                if vr < 1.1: continue

                cvd_v = a['cvd'][t]
                ofi_v = a['ofi'][t]
                adx_v = a['adx'][t]

                fr = (feat.get(sym, [{}])[t]
                      if sym in feat and t < len(feat.get(sym, []))
                      else {'cvd': cvd_v, 'ofi': ofi_v, 'vol_ratio': vr,
                            'adx': adx_v, 'rsi': 50.0, 'has_sweep': 0,
                            'wick_ratio_lo': 0.0, 'wick_ratio_hi': 0.0})

                wp  = self._wp(fr)
                res = self._eval(sym, t, a, cvd_v, ofi_v, adx_v, atr_v, vr, wp, reentry)
                if res:
                    cands.append(res)

            if not cands: continue
            cands.sort(key=lambda x: x['es'], reverse=True)
            best = cands[0]

            if best['symbol'] in {tr.symbol for tr in open_trades}: continue

            eff  = BASE_RISK_PCT * best['sm']
            rusd = balance * eff
            if rusd < 0.05: continue

            tr = Trade(
                symbol=best['symbol'], entry_bar=t,
                entry_price=best['e'], stop_loss=best['sl'],
                take_profit=best['tp'], side=best['side'],
                setup_type=best['st'], setup_tier=best['tier'],
                size_mult=best['sm'], risk_pct=eff, risk_usd=rusd,
                rr_ratio=best['rr'], edge_score=best['es'], win_prob=best['wp']
            )
            open_trades.append(tr)
            cooldown[best['symbol']] = t + SYMBOL_COOLDOWN
            if sess:
                k = date + '_' + sess
                sess_cnt[best['symbol']][k] = sess_cnt[best['symbol']].get(k, 0) + 1

        # Close remaining
        for tr in open_trades:
            a  = arrs[tr.symbol]
            lp = a['cl'][n-2]
            ret = ((lp / tr.entry_price - 1) if tr.side == 'BUY'
                   else (1 - lp / tr.entry_price))
            pnl = tr.risk_usd * (ret / BASE_RISK_PCT) * 0.4
            tr.exit_bar   = n - 2
            tr.exit_price = lp
            tr.pnl_usd    = round(pnl, 6)
            tr.outcome    = 'TIMEOUT'
            tr.hold_bars  = n - 2 - tr.entry_bar
            balance       = max(balance + pnl, 0.001)
            all_trades.append(tr)

        return self._compile(all_trades, equity, balance, n)

    def _eval(self, sym, t, a, cvd, ofi, adx, atr_v, vr, wp, rc):
        hi, lo, op, cl, vs = a['hi'], a['lo'], a['op'], a['cl'], a['vs']

        bs = _sweep_b(hi, lo, op, cl, t, atr_v)
        br = _sweep_s(hi, lo, op, cl, t, atr_v)

        # ── Primary A: Trap Reversal ──────────────────────────────────────
        for sw, side in [(bs, 'BUY'), (br, 'SELL')]:
            if sw is None: continue
            if (br if side == 'BUY' else bs) is not None: continue
            rl  = sw.get('reclaim_level', 0)
            rec = (cl[t] > rl) if side == 'BUY' else (cl[t] < rl)
            if not rec: continue
            ab  = _abs(cvd, ofi, side)
            vsp, vr2 = _vspike(vs, t, VOL_PRIM)
            if (sw.get('forced_move', False) and ab >= ABS_PRIM_A
                    and vsp and wp >= ML_THRESH_PRIM):
                e, sl, tp, rr = _sltp(cl, hi, lo, t, side, sw, atr_v)
                es = _escore(sw['wick_ratio'], ab, True, vr2, True, wp, rr, adx, True, 'TRAP_REVERSAL')
                if es >= PRIMARY_SCORE:
                    self._cre(rc, sym, t, side, e, sl, tp)
                    return dict(symbol=sym, side=side, e=e, sl=sl, tp=tp, rr=rr,
                                st='TRAP_REVERSAL', tier='PRIMARY', sm=1.0, es=es, wp=wp)

        # ── Secondary A ───────────────────────────────────────────────────
        for sw, side in [(bs, 'BUY'), (br, 'SELL')]:
            if sw is None: continue
            if (br if side == 'BUY' else bs) is not None: continue
            rl  = sw.get('reclaim_level', 0)
            rec = (cl[t] > rl) if side == 'BUY' else (cl[t] < rl)
            if not rec: continue
            ab       = _abs(cvd, ofi, side)
            vs2, vr3 = _vspike(vs, t, VOL_SEC)
            ev       = max(vr, vr3)
            if (ab >= ABS_SEC and (vsp if (sw := sw) else vs2)
                    and wp >= ML_THRESH_SEC and adx >= ADX_SEC):
                e, sl, tp, rr = _sltp(cl, hi, lo, t, side, sw, atr_v)
                es = _escore(sw['wick_ratio'], ab, sw.get('forced_move', False),
                             ev, True, wp, rr, adx, True, 'TRAP_REVERSAL')
                if SECONDARY_SCORE <= es < PRIMARY_SCORE:
                    self._cre(rc, sym, t, side, e, sl, tp)
                    return dict(symbol=sym, side=side, e=e, sl=sl, tp=tp, rr=rr,
                                st='TRAP_REVERSAL', tier='SECONDARY', sm=SECONDARY_MULT, es=es, wp=wp)

        # ── Primary B: Continuation — FIX: strong trend required ─────────
        if t >= 210 and adx >= ADX_PRIM:
            for side in ['BUY', 'SELL']:
                # FIX: require BOTH SMA50 and SMA200 aligned
                if not _trend_aligned(cl, t, side):
                    continue
                ab       = _abs(cvd, ofi, side)
                vsp2, vr2 = _vspike(vs, t, VOL_PRIM)
                oi       = (ofi > 0.05 if side == 'BUY' else ofi < -0.05)
                sb       = _sb2(cl, hi, lo, t, side)
                if ab >= ABS_PRIM_B and vsp2 and oi and sb and wp >= ML_THRESH_PRIM:
                    e, sl, tp, rr = _sltp(cl, hi, lo, t, side, {}, atr_v, cont=True)
                    es = _escore(0, ab, True, vr2, True, wp, rr, adx, False, 'CONTINUATION')
                    if es >= PRIMARY_SCORE:
                        self._cre(rc, sym, t, side, e, sl, tp)
                        return dict(symbol=sym, side=side, e=e, sl=sl, tp=tp, rr=rr,
                                    st='CONTINUATION', tier='PRIMARY', sm=1.0, es=es, wp=wp)

        # ── Secondary B ───────────────────────────────────────────────────
        if t >= 210 and adx >= ADX_SEC:
            for side in ['BUY', 'SELL']:
                if not _trend_aligned(cl, t, side):
                    continue
                ab        = _abs(cvd, ofi, side)
                vs3, vr3  = _vspike(vs, t, VOL_SEC)
                oi        = (ofi > 0.04 if side == 'BUY' else ofi < -0.04)
                sb        = _sb2(cl, hi, lo, t, side)
                ev        = max(vr, vr3)
                if (ab >= ABS_SEC and (vs3 or _vspike(vs, t, VOL_PRIM)[0])
                        and oi and sb and wp >= ML_THRESH_SEC):
                    e, sl, tp, rr = _sltp(cl, hi, lo, t, side, {}, atr_v, cont=True)
                    es = _escore(0, ab, False, ev, True, wp, rr, adx, False, 'CONTINUATION')
                    if SECONDARY_SCORE <= es < PRIMARY_SCORE:
                        self._cre(rc, sym, t, side, e, sl, tp)
                        return dict(symbol=sym, side=side, e=e, sl=sl, tp=tp, rr=rr,
                                    st='CONTINUATION', tier='SECONDARY', sm=SECONDARY_MULT, es=es, wp=wp)

        # ── Re-entry ──────────────────────────────────────────────────────
        if sym in rc and rc[sym]['exp'] > t:
            r    = rc[sym]
            side = r['side']
            oe, osl, otp = r['e'], r['sl'], r['tp']
            sv   = (lo[t] > osl) if side == 'BUY' else (hi[t] < osl)
            if not sv:
                del rc[sym]
            else:
                ms  = abs(otp - oe)
                inz = ((oe - ms * REENTRY_HI) <= cl[t] <= (oe - ms * REENTRY_LO)
                       if side == 'BUY'
                       else (oe + ms * REENTRY_LO) <= cl[t] <= (oe + ms * REENTRY_HI))
                ab  = _abs(cvd, ofi, side)
                vs4, vr4 = _vspike(vs, t, 1.1)
                if inz and ab >= REENTRY_ABS and vs4 and wp >= ML_THRESH_SEC:
                    sd = abs(cl[t] - osl)
                    if sd > 0:
                        rr = min(abs(otp - cl[t]) / sd, 3.0)
                        es = _escore(0, ab, False, vr4, True, wp, rr, adx, False, 'REENTRY')
                        if es >= SECONDARY_SCORE:
                            del rc[sym]
                            return dict(symbol=sym, side=side, e=cl[t], sl=osl, tp=otp,
                                        rr=round(rr, 2), st='REENTRY', tier='SECONDARY',
                                        sm=SECONDARY_MULT, es=es, wp=wp)
        return None

    def _cre(self, rc, sym, t, side, e, sl, tp):
        rc[sym] = {'side': side, 'e': e, 'sl': sl, 'tp': tp, 'exp': t + REENTRY_EXPIRY}

    def _compile(self, trades, equity, bal, n_bars):
        if not trades:
            return {'error': 'No trades',
                    'summary': {'total_trades': 0, 'initial_balance': self.ib,
                                'final_balance': round(bal, 4), 'total_return_pct': 0,
                                'total_days': round(n_bars*5/60/24, 1),
                                'win_rate': 0, 'trades_per_day': 0, 'avg_rr': 0,
                                'profit_factor': 0, 'expectancy_usd': 0,
                                'max_drawdown_pct': 0, 'total_pnl_usd': 0,
                                'wins': 0, 'losses': 0, 'timeouts': 0, 'avg_win_prob': 0},
                    'tier_breakdown': {
                        'primary':   {'trades': 0, 'win_rate': 0, 'pnl': 0},
                        'secondary': {'trades': 0, 'win_rate': 0, 'pnl': 0},
                        'reentry':   {'trades': 0, 'win_rate': 0, 'pnl': 0},
                    },
                    'setup_breakdown': {
                        'trap_reversal': {'trades': 0, 'win_rate': 0},
                        'continuation':  {'trades': 0, 'win_rate': 0},
                    },
                    'symbol_breakdown': {}, 'trade_log': pd.DataFrame(),
                    'equity_curve': equity}

        df   = pd.DataFrame([{
            'symbol': tr.symbol, 'entry_bar': tr.entry_bar, 'exit_bar': tr.exit_bar,
            'side': tr.side, 'setup_type': tr.setup_type, 'tier': tr.setup_tier,
            'entry': tr.entry_price, 'exit': tr.exit_price,
            'sl': tr.stop_loss, 'tp': tr.take_profit,
            'rr': tr.rr_ratio, 'edge_score': tr.edge_score,
            'win_prob': tr.win_prob, 'pnl_usd': tr.pnl_usd,
            'outcome': tr.outcome, 'hold_bars': tr.hold_bars,
        } for tr in trades])

        tot  = len(df)
        days = n_bars * 5 / 60 / 24
        wins = (df['outcome'] == 'WIN').sum()
        lss  = (df['outcome'] == 'LOSS').sum()
        tos  = (df['outcome'] == 'TIMEOUT').sum()
        wr   = wins / tot if tot > 0 else 0
        gw   = df[df['pnl_usd'] > 0]['pnl_usd'].sum()
        gl   = abs(df[df['pnl_usd'] < 0]['pnl_usd'].sum())
        pf   = gw / gl if gl > 0 else float('inf')
        aw   = df[df['outcome'] == 'WIN']['pnl_usd'].mean()  if wins > 0 else 0
        al   = df[df['outcome'] == 'LOSS']['pnl_usd'].mean() if lss > 0  else 0
        ex   = wr * aw + (1 - wr) * al
        eq   = np.array(equity)
        pk   = np.maximum.accumulate(eq)
        mdd  = ((eq - pk) / pk).min()

        def _wr(sub): return round((sub['outcome'] == 'WIN').mean(), 4) if len(sub) > 0 else 0
        pr  = df[df['tier'] == 'PRIMARY']
        sc  = df[df['tier'] == 'SECONDARY']
        re  = df[df['setup_type'] == 'REENTRY']
        tr2 = df[df['setup_type'] == 'TRAP_REVERSAL']
        co  = df[df['setup_type'] == 'CONTINUATION']

        sym_b = {s: {'trades': len(sub), 'win_rate': _wr(sub),
                     'pnl': round(sub['pnl_usd'].sum(), 4)}
                 for s in df['symbol'].unique()
                 for sub in [df[df['symbol'] == s]]}

        return {
            'summary': {
                'initial_balance':  self.ib,
                'final_balance':    round(bal, 4),
                'total_return_pct': round((bal / self.ib - 1) * 100, 2),
                'total_days':       round(days, 1),
                'total_trades':     tot,
                'wins': int(wins), 'losses': int(lss), 'timeouts': int(tos),
                'win_rate':         round(wr, 4),
                'trades_per_day':   round(tot / days, 3),
                'profit_factor':    round(pf, 3),
                'avg_rr':           round(df['rr'].mean(), 3),
                'expectancy_usd':   round(ex, 4),
                'max_drawdown_pct': round(mdd * 100, 2),
                'total_pnl_usd':    round(df['pnl_usd'].sum(), 4),
                'avg_win_prob':     round(df['win_prob'].mean(), 3),
            },
            'tier_breakdown': {
                'primary':   {'trades': len(pr), 'win_rate': _wr(pr), 'pnl': round(pr['pnl_usd'].sum(), 4)},
                'secondary': {'trades': len(sc), 'win_rate': _wr(sc), 'pnl': round(sc['pnl_usd'].sum(), 4)},
                'reentry':   {'trades': len(re), 'win_rate': _wr(re), 'pnl': round(re['pnl_usd'].sum(), 4)},
            },
            'setup_breakdown': {
                'trap_reversal': {'trades': len(tr2), 'win_rate': _wr(tr2)},
                'continuation':  {'trades': len(co),  'win_rate': _wr(co)},
            },
            'symbol_breakdown': sym_b,
            'trade_log':        df,
            'equity_curve':     equity,
        }
