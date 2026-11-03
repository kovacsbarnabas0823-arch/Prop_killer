# 🏆 Automated Prop Trading Architecture (Prop_Killer)

[![Production Status](https://img.shields.io/badge/PRODUCTION%20ACTIVE-30--Min%20Filtered%20Quant%20ORB-brightgreen?style=for-the-badge&logo=target)](strategies/orb_30m)
[![Asset](https://img.shields.io/badge/Asset-CME%20MNQ%20%2F%20NQ%20Futures-blue?style=for-the-badge)](strategies/orb_30m)
[![Target Platform](https://img.shields.io/badge/Target-Tradeify%20$50K%20Growth%20EOD-orange?style=for-the-badge)](RESEARCH_BRIEF.md)
[![Audit Status](https://img.shields.io/badge/Econometric%20Audit-VERIFIED%20(Zero--Lookahead)-success?style=for-the-badge)](strategies/orb_30m/backtest/empirical_tearsheet.txt)

> ### ⚡ OFFICIAL PRODUCTION CHAMPION: 30-MINUTE FILTERED QUANT ORB (`orb_30m`)
> **The 30-Minute Filtered Quantitative ORB (`orb_30m`) is the officially verified, institutional-grade production system for live operations and Tradeify $50K Growth evaluations.**  
> Extensive rigorous cross-validation across **8.9 years of continuous CME 1-minute Databento data (2018–2026 | 2,246 sessions | 2,546,537 bars)** confirms that `orb_30m` delivers a **51.62% Win Rate**, **1.21 Profit Factor**, **+$16.37 net expectancy per trade**, and an exceptional **51.6% single-attempt challenge pass rate** (30.8% end-to-end journey to cash payout in bank).  
> In out-of-sample testing (**2023–2026**), the strategy demonstrates **2.47 Walk-Forward Efficiency (WFE)**, generating **+$6,869.30 PnL (53.3% WR, 1.28 PF)** on modern market regimes.

---

## 1. Audited Comparative Matrix (8.9 Years CME NQ Data)

All models tested under **strict zero-lookahead**, pessimistic intra-bar collision resolution, **$1.24/contract ($2.48 roundtrip/MNQ) commission**, and **0.50 index point ($1.00/MNQ) slippage** per side:

| Strategy Engine | Production Status | Entry Horizon | 8.9y Realized PnL | Win Rate | Profit Factor | Trade Expectancy | Max Realized Drawdown | Tradeify Pass Rate | Expected Cost to Pass | 1st Cash Payout Success |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **30-Min Filtered ORB (`orb_30m`)** | **🎯 ACTIVE CHAMPION** | **10:00–11:30 EST** | **+$10,626.48** | **51.62%** | **1.21** | **+$16.37 / tr** | **-$2,605.72** | **51.6%** | **$168.60 (~1.9 attempts)** | **30.8% (ROI: +789%)** |
| **15-Min High-Velocity (`orb_hybrid_b`)** | ⚠️ High-Risk / Marginal | 09:45–11:30 EST | +$2,254.20 | 50.14% | 1.03 | +$2.18 / tr | -$3,710.29 | 17.0% | $512.97 (~5.9 attempts) | 5.6% (ROI: +192%) |
| **5-Min ORB + 12 EMA (`orb_5m`)** | ❌ DEBUNKED / REJECTED | 09:35 EST | **-$147,530.40** | 9.55% | 0.11 | -$142.40 / tr | -$147,530 | 0.0% | Ruined | 0.0% |

> [!CAUTION]
> **Audit Disclosure on 5-Minute ORB (`orb_5m`):**  
> Earlier iterations reported high theoretical returns for `orb_5m`. A rigorous institutional audit revealed this was caused by an **intra-bar lookahead bug** in the 0.5R trailing stop (`sweep_engine_5m.py:219-267`) and unmodeled gap executions. Under realistic, verified execution, the 5m model collapses. It is preserved strictly as an audited case study.

---

## 2. Why the 30-Minute Filtered ORB is the Validated Architecture

1. **Auction Stabilization (10:00 EST):**  
   The initial 30 minutes of the New York session (09:30–10:00 EST) represent peak auction volatility and liquidity sweeps. By establishing the Opening Range across the full 30 minutes, `orb_30m` bypasses morning stop-hunting and captures true institutional expansion.
2. **Robust Expectancy Over Friction:**  
   With an average trade expectancy of **+$16.37 per trade** on 2 MNQ (compared to +$2.18 on 15m), transaction costs and slippage represent only a fraction of the edge rather than consuming it.
3. **Out-of-Sample Persistence:**  
   Unlike models that overfit pre-2022 trends, `orb_30m` generated its strongest returns during **2023–2026 (+ $6,869.30 net profit, 53.3% WR)**, proving structural alpha in modern algorithmic and 0DTE-heavy regimes.
4. **Tradeify Challenge Mathematical Advantage:**  
   With a **51.6% single-attempt pass rate**, traders pass the $50K challenge in an expected **1.94 attempts ($168.60 total fee)**, preserving capital for funded account compounding.

---

## 3. Academic Foundations & Prop Firm Asymmetry

This project leverages the theoretical and empirical breakthroughs of three recent academic studies on retail proprietary trading firm evaluations:

1. **Martí Tomàs Fernández (August 2026)**: *Valuing proprietary trading firm evaluation contracts: closed form, cross-firm dispersion and the source of the margin.*  
   - Proves mathematically that for a disciplined positive-drift trader, Tradeify's contract has an expected value ($606) far exceeding its cost ($87), resulting in a **negative gross margin (-79%) for the prop firm**.  
   - Tradeify's **EOD trailing floor (freezing permanently at $50,100)** gives a **+19% higher survival probability** than intraday tick-trailing firms (Apex, MyFundedFutures).
2. **Boon Chuan Lim (July 2026)**: *The Price of a Funded Account: An Actuarial Analysis of Proprietary Trading Firm Challenges.*  
   - Identifies the **"Volatility Ridge" at ~2% daily equity volatility**. Sizing along this ridge (2 to 4 MNQ) maximizes pass probability without exceeding the $2,000 maximum drawdown barrier.
3. **Zachary Youngblood (Fordham Univ, August 2026)**: *The Inherent Value of Retail Prop Firms for Fixed-Size Trading Strategies.*  
   - Empirically proved across 10 years of CME Nasdaq data and 10,000 randomized paths that fixed-size trading under prop firm rules generated positive expected value on variance alone when risk is asymmetric.

---

## 4. Discovered Market Microstructure Anomalies (CME NQ 2018–2026)

Our empirical research across 2,170 sessions uncovered 4 distinct structural anomalies:

* **The New York Open London Low Sweep (89.0% Rejection):**  
  When the NY Open (09:30–10:30 EST) sweeps below the London session low, **in 89.0% of occurrences it fails to break down and violently reverses back into the London range**.
* **Limit Order Book Multi-Touch Fatigue:**  
  Touch #1 on resistance rejects 43.1%; Touch #3 rejects only 27.6%; **Touch #4+ cascades into a breakout 81.9% of the time** due to passive limit order depletion.
* **0DTE Strike Magnetism:**  
  Dealer gamma pinning compresses NQ closing distance to the nearest 50-pt strike by 5.3% post-May 2022.
* **15:00–15:45 Leveraged ETF Rebalancing:**  
  Trend days (> +1.0% into 15:00) experience deterministic end-of-day buying pressure from TQQQ/SQQQ daily rebalancing.

---

## 5. Repository Structure

```
├── core/
│   ├── data_loader.py                    # High-speed PyArrow continuous NQ parquet loader
│   ├── microstructure.py                 # Order book tick & liquidity utilities
│   └── risk_state_machine.py             # Dynamic sizing & drawdown ratchet tracker
├── data/
│   └── nq_1m_continuous.parquet          # 8.9 years continuous CME NQ parquet dataset
├── strategies/
│   ├── orb_30m/                          # 🏆 [ACTIVE PRODUCTION CHAMPION]
│   │   ├── config/
│   │   │   ├── strategy_config.yaml      # Production parameters (30m ORB, 2-tier exit)
│   │   │   └── prop_firm_rules.json      # Tradeify $50K Growth constraint definitions
│   │   ├── execution/
│   │   │   ├── order_execution_manager.py# Async live OMS/EMS with 2-tier scale-out
│   │   │   ├── TradeifyExecutionManager.cs # Native NinjaTrader 8 unmanaged strategy
│   │   │   └── webhook_server.py         # REST webhook signal & alert bridge
│   │   ├── backtest/
│   │   │   ├── empirical_backtester.py   # Institutional zero-lookahead backtest engine
│   │   │   ├── empirical_tearsheet.txt   # Verified tearsheet report
│   │   │   ├── historical_trade_log.csv  # 8.9-year audited trade-by-trade log
│   │   │   └── cpcv_validator.py         # Combinatorial Purged Cross-Validation
│   │   ├── audit_option_a.py             # Two-Phase Monte Carlo & OOS audit script
│   │   └── enhanced_audit_a.py           # Anomaly-filtered optimization engine
│   ├── orb_hybrid_b/                     # 🛡️ 15-Minute High-Velocity Engine (Audit Baseline)
│   │   ├── audit_hybrid_b.py             # Baseline 15m two-phase audit
│   │   └── enhanced_audit_b.py           # Enhanced 15m filter experiment
│   ├── orb_5m/                           # ⚠️ Flawed / Debunked Historical Baseline
│   │   └── backtest/full_quant_audit_ema.py # Realistic audit proving lookahead distortion
│   ├── research_market_anomalies.py      # Empirical microstructure anomaly suite
│   └── research_session_sweep_strat.py   # London sweep fade standalone prototype
├── idea.md                               # Quantitative blueprint & compliance manual
├── RESEARCH_BRIEF.md                     # 🔬 Formal research brief & AI prompt packet
└── requirements.txt                      # Project dependencies
```

---

## 6. Quick Start & Production Verification

### 1. Environment Setup
```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
# source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Production Option A (30m ORB) Institutional Audit
```bash
python strategies/orb_30m/audit_option_a.py
```

### 3. Run the Microstructure Anomaly Research Suite
```bash
python strategies/research_market_anomalies.py
```

### 4. Deploy Native NinjaTrader 8 Strategy
For traders using NinjaTrader 8 with Tradeify / Rithmic / Tradovate:
1. Copy `strategies/orb_30m/execution/TradeifyExecutionManager.cs` to your NinjaTrader 8 Custom Strategies directory:  
   `Documents\NinjaTrader 8\bin\Custom\Strategies\`
2. In NinjaTrader 8, press `F5` to compile.
3. Attach `TradeifyExecutionManager` to an `MNQ` 1-minute chart.
