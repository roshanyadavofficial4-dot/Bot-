"""
Trade Clustering Engine — Phase 3
====================================
Detects when multiple signals share similar characteristics (clustering),
and limits total exposure to any single pattern cluster.

Problem: Multiple signals firing on BTC-correlated alts simultaneously
         creates hidden correlation risk — all will win or lose together.

Solution: Group active/pending trades by pattern similarity.
          If a cluster already has N active trades → block new entries
          until an existing one closes.

Cluster dimensions:
  - Regime similarity (same regime = correlated)
  - Signal direction (same side = directionally correlated)
  - Symbol correlation group (BTC-correlated vs stablecoin-correlated)
  - Time proximity (signals within 1 candle of each other)
  - Indicator pattern similarity (similar ADX/OFI range)

This prevents overexposure to a single market hypothesis.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("TradeClusteringEngine")

# Max trades per cluster before blocking new entries
MAX_PER_CLUSTER      = 1   # conservative: only 1 trade per hypothesis
MAX_SAME_DIRECTION   = 2   # max 2 trades in same direction at once

# Cluster expiry — clusters dissolve after N seconds (prevents stale blocking)
CLUSTER_EXPIRY_SEC   = 900   # 15 minutes

# Symbol correlation groups (update as needed)
CORRELATION_GROUPS = {
    'BTC_CORRELATED':    ['DOGE/USDT:USDT', 'XRP/USDT:USDT', 'ADA/USDT:USDT',
                          'BNB/USDT:USDT', 'SOL/USDT:USDT'],
    'STABLECOIN_PAIRS':  [],
    'DEFI':              ['LINK/USDT:USDT', 'UNI/USDT:USDT'],
}


def _get_correlation_group(symbol: str) -> str:
    for group, symbols in CORRELATION_GROUPS.items():
        if symbol in symbols:
            return group
    return 'UNCATEGORIZED'


class TradeClusteringEngine:
    """
    Usage:
        engine = TradeClusteringEngine()

        # Before entering a trade:
        blocked, reason = engine.check_cluster(signal)
        if not blocked:
            engine.register_trade(trade_id, signal)

        # When trade closes:
        engine.remove_trade(trade_id)
    """

    def __init__(self):
        # trade_id → {cluster_key, direction, timestamp, symbol}
        self._active_trades: Dict[str, Dict] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def check_cluster(self, signal: Dict) -> Tuple[bool, str]:
        """
        Check if the given signal would create a risky cluster.

        signal: {symbol, regime, signal_direction, adx, ofi, ...}

        Returns:
            blocked  bool  — True if new trade should be blocked
            reason   str
        """
        self._purge_expired()

        symbol    = signal.get('symbol', '')
        direction = signal.get('signal_direction', 'UNKNOWN')
        regime    = signal.get('regime', 'UNKNOWN')
        adx       = signal.get('adx', 0.0)
        corr_grp  = _get_correlation_group(symbol)

        cluster_key = self._build_cluster_key(regime, direction, corr_grp, adx)

        # Check 1: Regime + direction cluster limit
        cluster_count = sum(
            1 for t in self._active_trades.values()
            if t['cluster_key'] == cluster_key
        )
        if cluster_count >= MAX_PER_CLUSTER:
            reason = (
                f"Cluster limit: '{cluster_key}' already has "
                f"{cluster_count}/{MAX_PER_CLUSTER} active trade(s)"
            )
            logger.debug(f"TradeCluster: blocked — {reason}")
            return True, reason

        # Check 2: Same-direction limit
        same_dir_count = sum(
            1 for t in self._active_trades.values()
            if t['direction'] == direction
        )
        if same_dir_count >= MAX_SAME_DIRECTION:
            reason = (
                f"Direction limit: already {same_dir_count} "
                f"{direction} trades active"
            )
            logger.debug(f"TradeCluster: blocked — {reason}")
            return True, reason

        # Check 3: Same symbol already active
        same_symbol = [
            tid for tid, t in self._active_trades.items()
            if t['symbol'] == symbol
        ]
        if same_symbol:
            reason = f"Symbol duplicate: {symbol} already has an active trade"
            logger.debug(f"TradeCluster: blocked — {reason}")
            return True, reason

        return False, ""

    def register_trade(self, trade_id: str, signal: Dict) -> None:
        """Register a trade that has been approved and entered."""
        symbol    = signal.get('symbol', '')
        direction = signal.get('signal_direction', 'UNKNOWN')
        regime    = signal.get('regime', 'UNKNOWN')
        adx       = signal.get('adx', 0.0)
        corr_grp  = _get_correlation_group(symbol)

        self._active_trades[trade_id] = {
            'cluster_key': self._build_cluster_key(regime, direction, corr_grp, adx),
            'direction':   direction,
            'symbol':      symbol,
            'regime':      regime,
            'corr_group':  corr_grp,
            'timestamp':   time.time(),
        }
        logger.debug(
            f"TradeCluster: registered {trade_id} | "
            f"cluster={self._active_trades[trade_id]['cluster_key']}"
        )

    def remove_trade(self, trade_id: str) -> None:
        """Remove a trade from the cluster registry (after close)."""
        removed = self._active_trades.pop(trade_id, None)
        if removed:
            logger.debug(f"TradeCluster: removed {trade_id}")

    def get_cluster_summary(self) -> Dict:
        """Diagnostic view of current cluster state."""
        self._purge_expired()
        cluster_counts: Dict[str, int] = defaultdict(int)
        direction_counts: Dict[str, int] = defaultdict(int)

        for t in self._active_trades.values():
            cluster_counts[t['cluster_key']] += 1
            direction_counts[t['direction']] += 1

        return {
            'active_trades':     len(self._active_trades),
            'cluster_breakdown': dict(cluster_counts),
            'direction_breakdown': dict(direction_counts),
        }

    # ── Internal helpers ────────────────────────────────────────────────────

    def _build_cluster_key(
        self,
        regime: str,
        direction: str,
        corr_group: str,
        adx: float,
    ) -> str:
        """
        Build a cluster key from discretized conditions.
        Similar market conditions → same cluster key.
        """
        adx_bucket = "weak" if adx < 22 else ("moderate" if adx < 30 else "strong")
        return f"{regime}|{direction}|{corr_group}|adx:{adx_bucket}"

    def _purge_expired(self) -> None:
        cutoff = time.time() - CLUSTER_EXPIRY_SEC
        to_remove = [
            tid for tid, t in self._active_trades.items()
            if t['timestamp'] < cutoff
        ]
        for tid in to_remove:
            logger.warning(f"TradeCluster: auto-expired stale trade {tid}")
            del self._active_trades[tid]
