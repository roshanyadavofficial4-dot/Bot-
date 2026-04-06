"""
Anomaly Detection Engine — Phase 4 (Upgrade)
===============================================
Detects market anomalies that should halt or pause trading.

Anomaly types:
  NEWS_SPIKE     — sudden volume + price spike inconsistent with technical setup
  FLASH_CRASH    — rapid multi-% price drop in <2 candles
  SPREAD_EXPLOSION — spread suddenly widens 5×+ (liquidity crisis)
  FUNDING_EXTREME — funding rate hits extreme (>0.2% or <-0.2%)
  CORR_BREAK     — asset suddenly decorrelates from BTC (manipulation signal)
  PRICE_GAP      — open price gaps > 1% from previous close (news event)
  VOLUME_BOMB    — 10×+ volume spike without directional follow-through

On detection: returns anomaly type, severity, and recommended action
  (skip_trade, pause_N_candles, abort_all, alert_only)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

logger = logging.getLogger("AnomalyDetector")

# Detection thresholds
FLASH_CRASH_PCT         = 0.025   # 2.5% drop in one candle
FLASH_PUMP_PCT          = 0.025   # 2.5% pump in one candle
SPREAD_EXPLOSION_MULT   = 5.0     # spread 5× normal → anomaly
VOLUME_BOMB_MULT        = 10.0    # 10× avg volume without direction
FUNDING_EXTREME_PCT     = 0.002   # ±0.2% funding rate
PRICE_GAP_PCT           = 0.010   # 1% gap from prev close

# Severity levels
SEVERITY = {
    'LOW':      1,   # alert only
    'MEDIUM':   2,   # skip this trade
    'HIGH':     3,   # pause 3 candles
    'CRITICAL': 4,   # abort all + manual review
}

SEVERITY_ACTIONS = {
    'LOW':      'alert_only',
    'MEDIUM':   'skip_trade',
    'HIGH':     'pause_3_candles',
    'CRITICAL': 'abort_all',
}


class AnomalyDetector:
    """
    Usage:
        detector = AnomalyDetector()
        anomaly, meta = detector.check(df, candidate, funding_rate=0.0)
        if anomaly:
            apply meta['action']
    """

    def __init__(self):
        self._baseline_spread: Optional[float] = None
        self._baseline_volume: Optional[float] = None
        self._recent_anomalies = []
        self._paused_until_candle = 0
        self._candle_count = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def check(
        self,
        df: pd.DataFrame,
        candidate: Dict,
        funding_rate: float = 0.0,
    ) -> Tuple[bool, Dict]:
        """
        Full anomaly scan.

        Returns:
            anomaly_detected  bool
            meta              dict  {type, severity, action, description}
        """
        self._candle_count += 1

        if df is None or len(df) < 10:
            return False, {}

        # Update baselines
        self._update_baselines(df, candidate)

        detected = []

        # ── Check 1: Flash crash / pump ───────────────────────────────────────
        fc = self._check_flash_move(df)
        if fc:
            detected.append(fc)

        # ── Check 2: Spread explosion ─────────────────────────────────────────
        se = self._check_spread_explosion(candidate)
        if se:
            detected.append(se)

        # ── Check 3: Volume bomb (spike without direction) ────────────────────
        vb = self._check_volume_bomb(df, candidate)
        if vb:
            detected.append(vb)

        # ── Check 4: Funding rate extreme ─────────────────────────────────────
        fe = self._check_funding_extreme(funding_rate)
        if fe:
            detected.append(fe)

        # ── Check 5: Price gap ────────────────────────────────────────────────
        pg = self._check_price_gap(df)
        if pg:
            detected.append(pg)

        # ── Check 6: Still in pause window ───────────────────────────────────
        if self._candle_count < self._paused_until_candle:
            remaining = self._paused_until_candle - self._candle_count
            return True, {
                'type':        'PAUSE_ACTIVE',
                'severity':    'HIGH',
                'action':      'skip_trade',
                'description': f'In anomaly pause — {remaining} candle(s) remaining',
            }

        if not detected:
            return False, {}

        # Pick worst detected anomaly
        worst = max(detected, key=lambda x: SEVERITY.get(x.get('severity', 'LOW'), 1))

        action = SEVERITY_ACTIONS.get(worst['severity'], 'skip_trade')
        worst['action'] = action

        # Apply pause if needed
        if worst['severity'] == 'HIGH':
            self._paused_until_candle = self._candle_count + 3
        elif worst['severity'] == 'CRITICAL':
            self._paused_until_candle = self._candle_count + 10

        # Log
        logger.warning(
            f"AnomalyDetector: {worst['type']} [{worst['severity']}] → {action} | "
            f"{worst.get('description', '')}"
        )
        self._recent_anomalies.append(worst)
        if len(self._recent_anomalies) > 50:
            self._recent_anomalies.pop(0)

        return True, worst

    def get_recent_anomalies(self, n: int = 10) -> list:
        return self._recent_anomalies[-n:]

    # ── Detection methods ────────────────────────────────────────────────────

    def _check_flash_move(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect a sudden large single-candle move."""
        last   = df.iloc[-1]
        move   = abs(last['close'] - last['open']) / max(last['open'], 1e-9)

        if move >= FLASH_PUMP_PCT:
            direction = 'crash' if last['close'] < last['open'] else 'pump'
            sev = 'HIGH' if move >= 0.04 else 'MEDIUM'
            return {
                'type':        f'FLASH_{direction.upper()}',
                'severity':    sev,
                'description': f'Flash {direction}: {move*100:.2f}% single candle',
                'value':       round(move, 4),
            }
        return None

    def _check_spread_explosion(self, candidate: Dict) -> Optional[Dict]:
        """Spread suddenly much wider than baseline."""
        if self._baseline_spread is None or self._baseline_spread < 1e-9:
            return None

        current_spread = candidate.get('spread_pct', 0.0)
        ratio = current_spread / self._baseline_spread

        if ratio >= SPREAD_EXPLOSION_MULT:
            sev = 'CRITICAL' if ratio >= 10 else 'HIGH'
            return {
                'type':        'SPREAD_EXPLOSION',
                'severity':    sev,
                'description': f'Spread {ratio:.1f}× normal (current={current_spread:.5f})',
                'value':       round(ratio, 2),
            }
        return None

    def _check_volume_bomb(self, df: pd.DataFrame, candidate: Dict) -> Optional[Dict]:
        """Very high volume without directional follow-through = manipulation signal."""
        vol_ratio = candidate.get('vol_ratio', 1.0)
        if vol_ratio < VOLUME_BOMB_MULT:
            return None

        # Check if there IS directional follow-through (if so, it's a legit breakout)
        last       = df.iloc[-1]
        body_ratio = abs(last['close'] - last['open']) / max(last['high'] - last['low'], 1e-9)

        if body_ratio > 0.60:
            # Strong directional candle — not a bomb, likely a real breakout
            return None

        return {
            'type':        'VOLUME_BOMB',
            'severity':    'MEDIUM',
            'description': f'Volume {vol_ratio:.1f}× avg with weak direction (body={body_ratio:.2f})',
            'value':       round(vol_ratio, 2),
        }

    def _check_funding_extreme(self, funding_rate: float) -> Optional[Dict]:
        """Extreme funding = crowded trade = reversal risk."""
        if abs(funding_rate) < FUNDING_EXTREME_PCT:
            return None

        direction = 'long' if funding_rate > 0 else 'short'
        sev = 'HIGH' if abs(funding_rate) >= 0.005 else 'MEDIUM'
        return {
            'type':        'FUNDING_EXTREME',
            'severity':    sev,
            'description': f'Funding rate {funding_rate*100:.3f}% — crowded {direction}s',
            'value':       round(funding_rate, 5),
        }

    def _check_price_gap(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect a large open-to-close gap between candles."""
        if len(df) < 3:
            return None

        prev_close = df.iloc[-2]['close']
        curr_open  = df.iloc[-1]['open']
        gap_pct    = abs(curr_open - prev_close) / max(prev_close, 1e-9)

        if gap_pct >= PRICE_GAP_PCT:
            sev = 'CRITICAL' if gap_pct >= 0.03 else 'HIGH'
            return {
                'type':        'PRICE_GAP',
                'severity':    sev,
                'description': f'Price gap {gap_pct*100:.2f}% between candles',
                'value':       round(gap_pct, 4),
            }
        return None

    def _update_baselines(self, df: pd.DataFrame, candidate: Dict) -> None:
        """Update rolling baseline for spread and volume."""
        if self._baseline_spread is None:
            self._baseline_spread = candidate.get('spread_pct', 0.0002)
        else:
            # Exponential moving average — slow update
            new_spread = candidate.get('spread_pct', self._baseline_spread)
            self._baseline_spread = self._baseline_spread * 0.95 + new_spread * 0.05

        if self._baseline_volume is None:
            if 'volume' in df.columns:
                self._baseline_volume = df['volume'].mean()
        else:
            if 'volume' in df.columns:
                new_vol = df['volume'].iloc[-1]
                self._baseline_volume = self._baseline_volume * 0.95 + new_vol * 0.05
