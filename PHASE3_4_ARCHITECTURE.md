# Trading Bot v5 — Phase 3 + Phase 4 Architecture

## Overview

Phase 3 and Phase 4 transform the stable Phase 2 system into a **self-improving,
adaptive, income-oriented trading machine** — without rewriting any existing logic.

All new engines are **additive gates and observers**. The core pipeline
(scanner → regime → filters → edge → executor) remains unchanged.

---

## New File Structure

```
engine/
├── phase2/                        ← UNCHANGED (existing)
│   ├── orchestrator.py
│   ├── edge_scorer.py
│   ├── performance_tracker.py
│   ├── expectancy_engine.py
│   ├── capital_allocator.py
│   └── ... (all existing)
│
├── phase3/                        ← NEW: Advanced Intelligence
│   ├── __init__.py
│   ├── strategy_health.py         ← Degradation detection + defensive mode
│   ├── loss_pattern.py            ← Empirical losing-condition blocklist
│   ├── adaptive_params.py         ← Dynamic threshold adjustment
│   ├── market_efficiency.py       ← Clean vs noisy market scoring
│   ├── timing_optimizer.py        ← Micro-pullback entry timing
│   ├── trade_clustering.py        ← Correlated-exposure limiter
│   └── profit_protection.py       ← Profit lock + cooldown system
│
├── phase4/                        ← NEW: Full AI + Income System
│   ├── __init__.py
│   ├── strategy_selector.py       ← Regime-based strategy switching
│   ├── feedback_loop.py           ← Central self-improvement engine
│   ├── capital_growth.py          ← Compounding + withdrawal manager
│   ├── strategy_failsafe.py       ← Hard-stop / auto-recovery system
│   ├── anomaly_detector.py        ← Flash crash / news spike detection
│   └── latency_monitor.py         ← Execution delay tracking + adjustment
│
└── phase3_4_orchestrator.py       ← NEW: Drop-in intelligent wrapper
```

---

## Integration (main.py change — 3 lines)

```python
# BEFORE (Phase 2):
from engine.phase2.orchestrator import Orchestrator
orchestrator = Orchestrator()

# AFTER (Phase 3+4 — drop-in):
from engine.phase3_4_orchestrator import IntelligentOrchestrator
orchestrator = IntelligentOrchestrator(balance=current_balance)
```

After each trade close, add one call:

```python
orchestrator.on_trade_closed(
    trade_result      = trade_result_dict,
    market_conditions = conditions_dict,
    current_balance   = balance,
)
```

---

## Extended Gate Pipeline (17 gates total)

```
[0]  Failsafe          Phase 4  Hard-stop: drawdown / consec losses / balance floor
[1]  Session Filter    Phase 2  Trading hours check (unchanged)
[2]  Anomaly Detect    Phase 4  Flash crash, spread explosion, volume bomb, funding extreme
[3]  Chop Filter       Phase 2  Directional market check (unchanged)
[4]  Vol Filter        Phase 2  ATR band check (unchanged)
[5]  Spread Analyzer   Phase 2  Spread quality check (unchanged)
[6]  Mkt Efficiency    Phase 3  Clean vs noisy market (score >= 0.40 required)
[7]  Regime Conf       Phase 2  Trending regime + confidence (unchanged)
[8]  Trade Slots       Phase 2  Concurrent trade capacity (unchanged)
[9]  Loss Pattern      Phase 3  Known-bad market-state veto (empirical blocklist)
[10] Trade Cluster     Phase 3  Correlated exposure limiter
[11] ML Win Prob       Phase 2  Win probability gate (threshold adaptive)
[12] Edge Score        Phase 2  Signal quality gate (threshold now adaptive)
[13] Trade Timing      Phase 3  Micro-pullback quality (skip if urgency=skip)
[14] Expectancy        Phase 2  EV > minimum threshold (unchanged)
[15] Risk Manager      Phase 2  Position size + daily limits (unchanged)
[16] Strategy Health   Phase 3  Defensive mode gate + risk multiplier
[17] Slippage+Latency  Phase 4  Execution cost viability (latency-adjusted)

→ ALL GATES PASS → Execute
```

---

## Intelligence Loop (how the system improves itself)

```
Trade Executed
     ↓
Trade Closed
     ↓
on_trade_closed()
     ├── PerformanceTracker.record_trade()       ← update rolling metrics
     ├── ExpectancyEngine.record()               ← update EV history
     ├── ProfitProtection.record_trade()         ← update session PnL
     ├── LossPatternDetector.record_trade()      ← add to pattern memory
     ├── StrategyHealthMonitor.notify_win/loss() ← update health state
     ├── StrategySelector.record_strategy_result() ← track per-strategy PnL
     └── FeedbackLoop.on_trade_closed()
              ├── Analyze last N trades
              ├── Detect losing conditions → adjust AdaptiveParams
              ├── Evaluate strategy performance → deprioritize weak strategies
              ├── Log adaptation event
              └── (every 50 trades) → trigger ML retrain
                        ↓
             Next Scan Cycle
                        ↓
             adaptive_params.get_params()
                        ↓
             New thresholds applied to gates [11][12]
                        ↓
             Better filter quality → improved outcomes
```

---

## Phase 3 Engine Details

### StrategyHealthMonitor
- 4 states: HEALTHY → CAUTION → DEFENSIVE → CRITICAL
- Reduces risk by 25–75% and raises edge score requirements in degraded states
- Requires 3 consecutive wins to recover from DEFENSIVE mode
- Tracks equity velocity to detect rapid deterioration

### LossPatternDetector
- Builds an empirical blocklist of market conditions → loss correlations
- Patterns: regime+session, ADX bucket+OFI direction, hour-of-day+volatility
- Vetoes trades when pattern has ≥75% loss rate over ≥3 occurrences
- Patterns expire after 50 trades (anti-overfit mechanism)

### AdaptiveParamEngine
- Adjusts: min_edge_score, min_win_prob, min_adx, min_ofi, risk_multiplier
- Volatile market → tighten all filters
- Clean trend → relax slightly
- Poor performance → tighten filters
- Smooth transitions: blends toward target by 30% each cycle (no abrupt jumps)
- All adjustments bounded within hard min/max limits

### MarketEfficiencyEngine
- Scores: directional consistency, candle quality, volume confirmation,
  indicator agreement, price range efficiency
- Score < 0.40 → market is too noisy → skip trade
- Outputs: efficiency_score (0–1), market_type (clean_trend/noisy/chaotic)

### TradeTimingOptimizer
- Checks: momentum extension (chasing?), RSI positioning, micro-pullback,
  spread timing, candle position
- urgency: immediate / wait_1 / wait_2 / skip
- Only 'skip' is a hard block — others are advisory

### TradeClusteringEngine
- Limits: 1 trade per regime+direction+correlation cluster
- Max 2 trades in same direction simultaneously
- Blocks duplicate symbol entries
- Auto-expires stale cluster entries after 15 minutes

### ProfitProtectionEngine
- Protection tiers at 1%, 2%, 3%, 5%, 8% daily gain
- Risk multiplier: 0.90→0.40 depending on profit level
- Tighter stop losses at 1.5%+ profit (trail reduced by 25–40%)
- Big-win cooldown: 5 minutes after any 1%+ single trade win
- Partial exit recommendation at 3% session + 1.5% trade profit

---

## Phase 4 Engine Details

### StrategySelector
- Strategies: TREND_FOLLOW, SCALP_PULLBACK, MEAN_REVERT, CONSERVATIVE, PAUSED
- Each has its own risk_multiplier, rr_target, min_edge_score
- DEFENSIVE health → CONSERVATIVE; CRITICAL health → PAUSED
- Tracks per-strategy win rate + PnL for diagnostics

### FeedbackLoop
- Runs every 5 trades (after minimum 10 trades)
- Actions: raise min_edge_score, raise min_win_prob, deprioritize strategies,
  flag losing regime/session combinations
- Logs all adaptation events with full trade context (200-trade memory)

### CapitalGrowthManager
- Tier system: MICRO (<$50) → SEED → GROWING → STABLE ($1000+)
- Each tier has appropriate risk_pct and leverage limits
- Withdrawal milestones at every +25% gain from anchor point
- Policy: 70% reinvest, 30% withdraw at milestone

### StrategyFailsafe
- Hard stops: 8 consec losses, 5% daily DD, 8% session DD, <18% win rate
- Also stops if expectancy negative for 25+ consecutive trades
- Auto-recovery after 2-hour cooldown
- Manual halt/resume via admin command (wired to Telegram)

### AnomalyDetector
- Detects: flash crash/pump (>2.5%), spread explosion (5×), volume bomb (10×),
  funding extreme (±0.2%), price gap (>1%)
- Severity: LOW=alert, MEDIUM=skip, HIGH=pause 3 candles, CRITICAL=abort all
- Maintains rolling baseline for spread/volume to detect deviations

### LatencyMonitor
- Tracks: API latency, order placement, fill confirmation
- Tiers: NORMAL/ELEVATED/HIGH/EXTREME
- HIGH → widen slippage buffer 2.5×, skip fast signals
- EXTREME → block new entries entirely
- Provides `start_timer(label)` / `stop_timer(token)` for executor integration

---

## Adaptive Behavior Summary

| Condition               | System Response                                      |
|-------------------------|------------------------------------------------------|
| Volatile market         | Tighten ADX, OFI, edge score; reduce risk            |
| Clean trending market   | Relax thresholds slightly; allow more opportunities  |
| Ranging market          | Switch to MEAN_REVERT strategy; tighter gates        |
| 3+ consecutive losses   | CAUTION: reduce risk 25%, raise edge requirement     |
| 5+ consecutive losses   | DEFENSIVE: halve risk, require edge ≥ 60             |
| 8+ consecutive losses   | FAILSAFE HALT: no new trades for 2 hours             |
| Flash crash detected    | Skip trade, pause 3 candles                          |
| Spread explosion        | CRITICAL: abort all pending entries                  |
| 2% daily profit         | Reduce risk to 75% of base                           |
| 5% daily profit         | Reduce risk to 40% of base                           |
| 8% daily profit         | Maximum protection, 25% risk                         |
| High latency            | Widen slippage buffer, skip fast signals             |
| Known losing pattern    | Veto trade (empirical blocklist, ≥75% loss rate)     |
| Cluster overexposure    | Block new entries in same hypothesis cluster         |
| Strategy underperforming| Deprioritize in StrategySelector                     |
| +25% capital growth     | Suggest 30% withdrawal, reset anchor                 |

---

## Anti-Overfit Safeguards

1. **Minimum trade counts**: No parameter changes until N trades of history
2. **Smooth transitions**: AdaptiveParams blends 30% toward target each cycle
3. **Hard bounds**: Every adjustable parameter has absolute min/max limits
4. **Pattern expiry**: LossPattern blocklist expires entries after 50 trades
5. **Interpretable logic**: No black-box ML in gate decisions — ML used for
   win probability only, all gates use rule-based logic
6. **Strategy minimum**: Strategies need 20 trades before deprioritization
7. **Recovery requirements**: Defensive mode requires 3 consecutive wins to exit

---

## Capital Safety Priority Order

1. StrategyFailsafe — hard halt, no override
2. ProfitProtection — lock gains before they erode
3. StrategyHealthMonitor — early warning system
4. AdaptiveParamEngine — tighten filters proactively
5. LossPatternDetector — avoid known bad setups
6. TradeClusteringEngine — prevent correlated blowup
7. RiskManager — position sizing (existing, unchanged)
