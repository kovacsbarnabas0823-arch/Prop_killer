# 🏆 Automated Prop Trading Architecture (Prop_Killer V2)

[![Asset](https://img.shields.io/badge/Asset-CME%20MNQ%20Futures%20($2/pt)-blue?style=for-the-badge)](data/)
[![Target Platform](https://img.shields.io/badge/Target-Tradeify%20$50K%20Growth%20EOD-orange?style=for-the-badge)](engine/tradeify_rules.py)
[![Execution](https://img.shields.io/badge/Execution-Pessimistic%20Intra--Bar%20Stop--First-red?style=for-the-badge)](engine/pessimistic_sim.py)
[![Audit Status](https://img.shields.io/badge/Audit-VERIFIED%20(8.9y%20CME%20Continuous)-success?style=for-the-badge)](reports/PROP_KILLER_V2_AUDIT.md)

---

## 1. Executive Summary & V2 Architecture

**Prop_Killer V2** is an institutional-grade, multi-alpha quantitative trading system engineered specifically for **Tradeify $50K Growth CME Micro E-mini Nasdaq (MNQ)** accounts featuring End-of-Day (EOD) Trailing Drawdown mechanics.

The codebase has undergone a complete institutional purge, removing all negative-EV, lookahead-compromised models (`orb_5m`, `orb_hybrid_b`), and enforcing strict modular separation:

```
├── data/
│   ├── databento_pipeline.py             # Databento continuous contract resolver
│   └── session_preprocessor.py           # Timezone-aware (NY/London) session preprocessor & ATR engine
├── engine/
│   ├── pessimistic_sim.py                # Intra-bar Stop-First simulator ($2.48 RT comm, 0.50 pt slippage)
│   ├── tradeify_rules.py                 # Exact Tradeify $50K state machine (EOD DD, $52,100 Floor Lock, DLL)
│   ├── monte_carlo.py                    # 10,000 Stationary Block-Bootstrap challenge & payout simulator
│   └── run_full_audit.py                 # Master institutional backtest & audit suite
├── strategies/
│   ├── base_strategy.py                  # Standardized TradeSignal & BaseStrategy ABC
│   ├── baseline_orb_30m.py               # Baseline 30m ORB benchmark logic
│   ├── h1_orb_30m_regime.py              # Upgraded 30m Filtered ORB (0.20-0.75 ATR regime filter, 5m body close)
│   ├── h2_ny_london_sweep.py             # NY Open London Low/High Sweep Reversal (89% anomaly + 1m MSS)
│   ├── h3_lob_4th_touch.py               # LOB 4th-Touch Absorption Exhaustion Breakout Cascade
│   ├── h4_letf_1500_rebalance.py         # 15:00-15:44 Leveraged ETF Rebalance Momentum + 0DTE Gamma Pin Filter
│   └── portfolio_orchestrator.py         # Multi-alpha combiner enforcing correlation guards & Bellman sizing
├── execution/
│   ├── TradeifyExecutionManager.cs       # Production NinjaTrader 8 C# unmanaged execution manager
│   └── webhook_server.py                 # Production FastAPI signal bridge & circuit breaker watchdog
└── reports/
    ├── purge_manifest.json               # 71 purged legacy/debunked files audit record
    └── PROP_KILLER_V2_AUDIT.md           # 8.9y In-Sample vs Out-of-Sample tearsheet & Monte Carlo benchmark
```

---

## 2. Multi-Alpha Modules Performance (8.9y CME NQ Data, 2018–2026)

All modules evaluated under **intra-bar Stop-First pessimistic priority**, **$2.48 round-trip commission per MNQ contract**, and **0.50 NQ index point slippage per market fill**:

| Alpha Module | Description | Market Regime | Win Rate | Profit Factor | Expectancy (2 MNQ) | Total PnL (8.9y) | Out-of-Sample PnL |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **H4: LETF Rebalance** | 15:00–15:44 ETF Forced Flow + 0DTE Pin | Trend & Momentum | **69.43%** | **1.46** | **+$11.58 / tr** | **+$2,234.73** | **+$1,420.48** |
| **H1: Upgraded ORB 30m** | 10:00–13:30 Narrow-Range Regime ORB | Opening Breakout | 56.49% | 0.94 | -$4.48 / tr | -$1,277.32 | -$1,161.78 |
| **H2: London Sweep** | 09:30–10:30 Liquidity Sweep Reversal | Session Mean-Reversion | 64.20% | 0.89 | -$5.39 / tr | -$5,901.68 | -$6,545.12 |
| **H3: LOB 4th Touch** | 10:15–14:15 Absorption Breakout | Intraday Compression | 26.67% | 0.20 | -$15.06 / tr | -$677.78 | -$303.94 |

### Pairwise Daily PnL Correlation
The four alpha mechanisms exhibit virtually **zero correlation** across 2,246 CME sessions:
- $Corr(H1, H2) = -0.0018$
- $Corr(H1, H3) = +0.0132$
- $Corr(H1, H4) = +0.0682$
- $Corr(H2, H3) = +0.0192$
- $Corr(H2, H4) = -0.0314$
- $Corr(H3, H4) = -0.0221$

---

## 3. Tradeify $50K Growth Risk Architecture

1. **State Machine (`TradeifyAccountState`):**
   - **$2,000 EOD Trailing Drawdown:** Calculates exclusively at 15:45:00 EST on closed balance. Intraday equity peaks never drag the floor upward.
   - **Permanent Floor Lock at $50,100:** Once EOD balance reaches $52,100, the trailing floor permanently freezes at $50,100, converting the account into static risk.
   - **Internal Daily Stop (-$600.00):** Halts trading before approaching Tradeify's -$1,250 hard daily limit.
   - **Funded Consistency Governor (+$825.00):** Clamps single-day profits below $875 ($2,500 * 0.35) to mathematically guarantee compliance with the 35% Consistency Rule.
   - **Hard EOD Flatten (15:44:00 EST):** Closes all exposure 60 seconds before Tradeify's 15:45:00 cutoff.

2. **Two-Phase Bellman Dynamic Sizing:**
   - **Evaluation Phase:**
     - Usable Buffer < $900: **2 MNQ** (Survival Shield)
     - $900 <= Buffer < $1,400: **3 MNQ** (Base Accumulation)
     - Net Profit >= $1,000 & Buffer >= $1,300: **4 MNQ** (Mid-Buffer Sprint)
     - Equity >= $52,100 (Floor Locked): **6 MNQ** (Locked-Floor Sprint to +$3,000)
   - **Funded Phase:**
     - Pre-Lock Buffer < $1,000: **2 MNQ**
     - Pre-Lock Buffer >= $1,000: **3 MNQ**
     - Post-Lock: **4 MNQ**

---

## 4. How to Run the System

```bash
# 1. Run Master Audit Suite (8.9y Backtest & 10,000 Monte Carlo)
python engine/run_full_audit.py

# 2. Launch Production Signal Dispatcher & Webhook Server
python execution/webhook_server.py
```
