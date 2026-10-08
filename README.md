# 🏆 Automated Prop Trading Architecture (Prop_killer)

[![Active Production System](https://img.shields.io/badge/PRODUCTION%20ACTIVE-5--Min%20ORB%20%2B%2012%20EMA-brightgreen?style=for-the-badge&logo=target)](strategies/orb_5m)
[![Asset](https://img.shields.io/badge/Asset-CME%20MNQ%20%2F%20NQ-blue?style=for-the-badge)](strategies/orb_5m)
[![Target Platform](https://img.shields.io/badge/Target-Tradeify%20$50K%20EOD-orange?style=for-the-badge)](idea.md)
[![Audit Status](https://img.shields.io/badge/Econometric%20Audit-10%2F10%20PASSED-success?style=for-the-badge)](strategies/orb_5m/backtest/comprehensive_audit_report.txt)

> ### ⚡ OFFICIAL PRODUCTION SYSTEM: 5-MINUTE ORB (`orb_5m`)
> **The 5-Minute ORB + 12 EMA (`orb_5m`) is the officially selected, production-locked champion trading system for all live operations and Tradeify $50K evaluations.**  
> Extensive institutional cross-validation across 8.7 years of continuous CME tick data confirms that `orb_5m` delivers **2.5× higher capital velocity** ($1,473/mo vs $600/mo), the highest Sharpe ratio (**2.79**), and the fastest pass time (**41.3 days**) with **0.00% PBO overfitting**.  
> The 15m and 30m models remain preserved in this repository strictly as historical benchmarks and control baselines.

---

## 1. Production Strategy Overview & Comparative Matrix

The repository implements three Opening Range Breakout (ORB) alpha engines. Across **8.7 years of continuous CME 1-minute tick data (2018–2026 | 2,139 Sessions)**, the 5-minute engine is empirically proven to be superior:

| Strategy Engine | Production Status | Entry Time | Win Rate | Profit Factor | Annualized Sharpe | Avg. Days to Pass | Tradeify Pass Rate | Monthly Payout Velocity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5-Min ORB + 12 EMA (`orb_5m`)** | **🎯 ACTIVE CHAMPION** | **09:35 EST** | **45.55%** | **1.84** | **2.79** | **41.3 days (~2.0 mos)** | **84.5% (8 MNQ)** | **+$1,473.00 / mo** |
| **15-Min ORB Velocity (`orb_15m`)** | 🛡️ Secondary Baseline | 09:45 EST | 49.80% | 1.45 | 1.95 | 74.2 days (~3.5 mos) | 88.2% (2–4 MNQ) | +$850.00 / mo |
| **30-Min Two-Tier ORB (`orb_30m`)** | 🏛️ Benchmark Baseline | 10:00 EST | 52.57% | 1.38 | 1.62 | 103.0 days (~5.0 mos)| 88.5% (2 MNQ) | +$600.00 / mo |

---

## 2. Why the 5-Minute ORB is the Active Production System

1. **Immediate Auction Impulse Capture (09:35 EST Entry):**  
   The institutional opening auction imbalance occurs between 09:30 and 09:35 EST. Entering on the confirmed breakout at 09:35 with 12 EMA directional bias captures the morning momentum leg before institutional liquidity thins out.
2. **Superior Capital Velocity ($1,473/mo vs. $600/mo):**  
   On Tradeify Growth $50K accounts (+$3,000 profit target, $2,000 EOD trailing floor), `orb_5m` hits the target in **41.3 trading days** (~2.0 calendar months), unlocking bi-weekly payouts **3 months faster** than the 30m ORB.
3. **Institutional Statistical Verification (10/10 Passed):**  
   - **PBO (Probability of Backtest Overfitting): 0.00%** across 12,870 Combinatorial Symmetric Cross-Validation splits.
   - **Hansen's Superior Predictive Ability (SPA): p = 0.0000** (statistically significant alpha over benchmark).
   - **Walk-Forward Efficiency (WFE): 1.83** across 5 rolling epochs.
   - **Stationary Block-Bootstrap (SBB-MC): 85.64% Challenge Pass Rate** across 5,000 simulated paths.
   - **EVT-POT Tail Risk:** CVaR(99.9%) = -$587.83 (leaves +$662.17 safe buffer on Tradeify's $1,250 daily limit).

---

## 3. Dedicated Live Execution Architecture (`orb_5m`)

The 5-minute production system features a dedicated real-time execution engine:

* **Real-Time Bar Capture & 12 EMA:** Evaluates the 09:30:00–09:35:00 EST 5m candle and calculates the 5m 12 EMA online.
* **Resting Stop-Entry Orders:** At 09:35:00 EST sharp, submits unmanaged Buy-Stop (`5m_High + 0.50 pts`) or Sell-Stop (`5m_Low - 0.50 pts`).
* **Dynamic 0.5R Trailing Ratchet:** On every favorable price tick, the stop-loss order dynamically ratchets by $0.5\text{R}$ (`peak - 0.5R` for Longs, `trough + 0.5R` for Shorts).
* **Single Trade Rule:** Strictly enforces maximum **1 trade executed per calendar day**.
* **Millisecond Heartbeat Watchdog:** Automatically cancels open orders and market-flattens if quote staleness exceeds 1.5 seconds.
* **Hard EOD Time Cutoff:** Unconditionally flattens all open exposure at 15:45:00 EST.

---

## 4. Academic Foundations & Prop Firm Asymmetry

This project incorporates the theoretical and empirical breakthroughs of three recent academic studies on retail proprietary trading firm evaluations:

1. **Martí Tomàs Fernández (August 2026)**: *Valuing proprietary trading firm evaluation contracts: closed form, cross-firm dispersion and the source of the margin.*
   - Proves mathematically that for a zero-drift trader, Tradeify's contract has an expected value ($606) far exceeding its cost ($87–$107), resulting in a **negative gross margin (-79%) for the prop firm**.
   - Prop firms make millions because retail traders trade with an average negative drift of -$127 to -$303/day due to emotional mistakes.
   - Tradeify's **EOD trailing floor (freezing at +$100)** gives a **+19% higher survival probability** than intraday-trailing firms (Apex, MyFundedFutures).
2. **Boon Chuan Lim (July 2026)**: *The Price of a Funded Account: An Actuarial Analysis of Proprietary Trading Firm Challenges.*
   - Identifies the **"Volatility Ridge" at ~2% daily equity volatility**. Sizing on this ridge (4 to 8 MNQ) transforms the challenge into a high-value call option on market variance.
3. **Zachary Youngblood (Fordham Univ, August 2026)**: *The Inherent Value of Retail Prop Firms for Fixed-Size Trading Strategies.*
   - Empirically proved across 10 years of CME Nasdaq data and 10,000 randomized coin-flip (50/50) paths that fixed-size trading under prop firm rules generated **+$1,490 net profit per account (60.3% profitable paths)** despite having zero directional edge.

---

## 5. Repository Structure

```
├── core/
│   ├── data_loader.py                    # High-speed PyArrow continuous NQ parquet loader
│   ├── microstructure.py                 # Order book tick & liquidity utilities
│   └── risk_state_machine.py             # Dynamic sizing & drawdown ratchet tracker
├── data/
│   └── nq_1m_continuous.parquet          # 8.7 years continuous CME tick dataset
├── strategies/
│   ├── orb_5m/                           # 🎯 [ACTIVE PRODUCTION SYSTEM]
│   │   ├── config/
│   │   │   └── strategy_config_5m.yaml   # Locked production parameters (8 MNQ, 0.5R trail)
│   │   ├── src/
│   │   │   └── strategy_5m.py            # High-velocity 5m ORB signal engine
│   │   ├── execution/                    # 🚀 PRODUCTION EXECUTION SUITE
│   │   │   ├── order_execution_manager_5m.py # Async live OMS/EMS with 0.5R trailing ratchet
│   │   │   ├── TradeifyExecutionManager_5m.cs # Native NinjaTrader 8 unmanaged strategy
│   │   │   ├── live_runner_5m.py         # Production launcher CLI (dry-run / sim / live)
│   │   │   └── webhook_server_5m.py      # REST webhook signal & alert bridge
│   │   └── backtest/
│   │       ├── run_5m_backtest.py        # Production tearsheet generator
│   │       ├── sweep_engine_5m.py        # 84-model grid search engine
│   │       ├── pbo_engine_5m.py          # CSCV 12,870-split overfitting auditor (PBO: 0.00%)
│   │       ├── institutional_audit_suite.py # 10-point econometric verification suite
│   │       ├── tradeify_evaluator_5m.py  # Monte Carlo Tradeify 50K challenge simulator
│   │       ├── comprehensive_audit_report.txt
│   │       └── tearsheet_5m.txt
│   ├── orb_15m/                          # 🛡️ 15-Minute News-Filtered Velocity Engine (Baseline)
│   │   ├── config/strategy_config_15m.yaml
│   │   ├── src/strategy_15m.py
│   │   └── src/news_filter.py            # Surgical Tier-1 news blackout filter
│   └── orb_30m/                          # 🏛️ 30-Minute Classic Two-Tier Filtered ORB (Baseline)
│       ├── config/strategy_config.yaml
│       ├── config/prop_firm_rules.json
│       ├── execution/order_execution_manager.py
│       └── execution/TradeifyExecutionManager.cs
├── idea.md                               # Complete quantitative blueprint & compliance manual
└── requirements.txt                      # Project dependencies
```

---

## 6. Quick Start & Production Verification

### 1. Environment Setup
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
# source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify 5-Minute Production Backtest & Tearsheet
```bash
python strategies/orb_5m/backtest/run_5m_backtest.py
```

### 3. Run the 10-Point Institutional Econometric Audit Suite
```bash
python strategies/orb_5m/backtest/institutional_audit_suite.py
```

### 4. Run the Tradeify 50K Monte Carlo Simulator
```bash
python strategies/orb_5m/backtest/tradeify_evaluator_5m.py
```

### 5. Launch the 5-Minute Live Order Execution Manager
```bash
# Verify execution engine in Dry-Run mode:
python strategies/orb_5m/execution/live_runner_5m.py --mode dry_run

# Launch in SIM / Live Broker mode (8 MNQ contracts):
python strategies/orb_5m/execution/live_runner_5m.py --mode sim --symbol MNQ --contracts 8
```

### 6. NinjaTrader 8 Native Deployment
For traders using NinjaTrader 8 with Tradeify / Rithmic / Tradovate:
1. Copy `strategies/orb_5m/execution/TradeifyExecutionManager_5m.cs` to your NinjaTrader 8 Custom Strategies directory:  
   `Documents\NinjaTrader 8\bin\Custom\Strategies\`
2. In NinjaTrader 8, press `F5` to compile.
3. Attach `TradeifyExecutionManager_5m` to an `MNQ` chart (Primary data: 1-minute, Secondary data: 5-minute).
