# Trading Bot v5.1 — Quick Start Guide

## Step 1 — Dependencies install karo (apne PC par)

```bash
pip install ccxt pandas numpy scikit-learn joblib requests
```

---

## Step 2 — Real data download karo (~10-15 minutes)

```bash
python scripts/download_data.py
```

Yeh automatically download karega:
- DOGE, XRP, ADA, SOL, BNB — Binance Futures
- Last 6 months, 5-minute candles
- CVD + OFI features engineer karega
- `data/historical/` mein save karega

**API key ki zaroorat NAHI hai — public endpoints use karta hai.**

---

## Step 3 — MLBrain train karo + Backtest chalao

```bash
python scripts/train_realdata.py
```

Yeh karega:
- Real data par MLBrain train karega
- Full v5.1 backtest chalayega
- Results print karega (win rate, RR, trades/day, drawdown)
- Trained model `engine/mlbrain_model.pkl` mein save karega
- Trade log `backtest/results/` mein save karega

**Sirf training chahiye (no backtest):**
```bash
python scripts/train_realdata.py --no-backtest
```

**Last 3 months ka data use karo:**
```bash
python scripts/train_realdata.py --months 3
```

---

## Step 4 — Paper trading (IMPORTANT — real deploy se pehle)

```bash
python main.py
```

`.env` file mein `USE_SANDBOX=True` set karo.
Minimum 30 days paper trading karo.
Agar results backtest se match karte hain → real deploy ka consider karo.

---

## Folder Structure

```
trading_bot_v5.1/
├── scripts/
│   ├── download_data.py      ← STEP 2: real data download
│   └── train_realdata.py     ← STEP 3: train + backtest
├── backtest/
│   ├── backtest_v51.py       ← Full v5.1 backtest engine
│   └── results/              ← Backtest reports saved here
├── data/
│   └── historical/           ← Downloaded CSVs saved here
├── engine/
│   ├── strategy.py           ← v5.1 upgraded
│   ├── mlbrain_model.pkl     ← Trained model (after Step 3)
│   └── phase2/
│       ├── edge_scorer.py    ← v5.1 tiered scoring
│       └── orchestrator.py   ← v5.1 with evaluate_multi()
├── model/
│   └── mlbrain_v51_real.pkl  ← Real data trained model backup
└── QUICKSTART.md             ← Yeh file
```

---

## Important Notes

1. **Synthetic backtest accurate nahi hai** — sirf real Binance data par
   train + backtest karo (Steps 2-3)

2. **Paper trading skip mat karo** — 30 days minimum

3. **₹200 ($2.4) se deploy mat karo** — Binance Futures minimum
   notional $5 hai, leverage ke saath bhi yeh kaafi risky hai.
   Minimum $20-50 se shuru karo agar results achhe hain.

4. **Daily profit guarantee nahi hai** — backtest results real
   performance guarantee nahi karte. Market conditions change hoti hain.
