# 🔬 Quantitative Research Brief & Project Dossier (Prop_Killer)
> **Target Audience:** Quantitative Researchers, Algorithmic Trading AI Agents, and Econometric Auditors.  
> **Objective:** Formulate, evaluate, and test novel, falsifiable quantitative hypotheses to maximize capital velocity and payout probability on prop firm accounts.

---

## 1. Executive Summary & Problem Formulation

This project is an institutional quantitative algorithmic trading framework engineered for **CME Micro E-mini Nasdaq-100 (`MNQ`) and E-mini Nasdaq-100 (`NQ`) Futures**.  
Its operational target is to consistently pass and harvest cash withdrawals from **Tradeify $50K Growth EOD Trailing Drawdown** evaluation accounts.

In retail prop trading, algorithmic failure is overwhelmingly caused by:
1. Underestimating market friction (commissions + bid-ask spread + execution slippage).
2. Lookahead bias and overly optimistic intra-bar assumptions in backtesters.
3. Overfitting to noisy short-horizon bars (5-minute and 15-minute whipsaws).
4. Failure to model the **two-phase reality**: passing the evaluation challenge is only Phase 1; surviving the funded account under consistency and withdrawal buffer rules to reach **Phase 2 (Cash Payout)** requires a fundamentally different risk profile.

---

## 2. Prop Firm Economic & Regulatory Constraints (Tradeify $50K Growth)

Every proposed hypothesis and strategy must strictly comply with the following mathematical constraints:

| Parameter | Value | Algorithmic Implementation Rule |
| :--- | :--- | :--- |
| **Starting Balance** | **$50,000.00** | Initial account equity. |
| **Profit Target (Phase 1)** | **+$3,000.00 (+6.0%)** | Challenge passes when balance reaches $\ge \$53,000.00$. |
| **Trailing Max Drawdown** | **$2,000.00** | **EOD Trailing Only:** Calculated at 15:45 EST close. Intraday equity wicks do NOT pull the floor up. Once balance reaches $\$52,100$, the floor **permanently locks at $50,100.00**. |
| **Daily Loss Limit** | **$1,250.00** | Hard intraday circuit breaker. If intraday loss reaches $-\$1,250$, account is closed. |
| **Max Position Size** | **4 NQ / 40 MNQ** | Point value: MNQ = **$2.00 / point**, NQ = **$20.00 / point**. |
| **Evaluation Cost** | **$87.00 per attempt** | One-time purchase fee (no monthly recurring subscription). Reset fee: $95.00. Modeled as independent Bernoulli trials. |
| **Overnight Exposure** | **STRICTLY ZERO** | CME Globex overnight holding is prohibited. **All open positions must unconditionally flatten at 15:45:00 EST.** |
| **Funded Phase Criteria (Phase 2)** | **Min 5 Days, 35% Consistency** | To request the first cash payout ($\$1,500$), the funded account must trade $\ge 5$ separate days, and **no single day's profit can exceed 35% of total accumulated profit**. |
| **Safety Buffer Lock** | **Balance $\ge \$52,500.00$** | Withdrawals cannot dip below the $\$50,100$ floor. Withdrawing $\$1,500$ requires an equity buffer $\ge \$52,500$ so the next trade does not liquidate the account. |

---

## 3. Backtesting Methodology & Execution Ground Rules

All empirical evaluations are conducted under strict **Zero-Lookahead** and **Pessimistic Execution** standards:

1. **Continuous Dataset:** 8.9 years of 1-minute CME NQ continuous data (**2018-01-01 to 2026-09-17 | 2,546,537 bars across 2,246 daily trading sessions**).
2. **Outright Front-Month Resolution:** NQ parent parquet files contain up to 130 simultaneous symbols (quarterly contracts and calendar spreads). The data loader strictly resolves outright front-month volume per day to eliminate overlapping multi-symbol pricing anomalies.
3. **Execution Friction:**
   * Commission: **$1.24 per contract per side ($2.48 roundtrip per MNQ)**.
   * Slippage: **0.50 index points ($1.00 per MNQ) pessimistic entry and exit**.
4. **Intra-Bar Collision Handling (The Pessimistic Rule):** If within the same 1-minute bar both Stop Loss and Take Profit levels are touched (`Low <= SL` and `High >= TP`), the simulator **unconditionally assumes Stop Loss occurred first**.
5. **Gap Sizing:** If a bar opens beyond the stop level, the execution price is filled at the open price (not the theoretical stop price).

---

## 4. Current State: Audited Strategy Baselines

Across multiple institutional audits and grid searches on the 8.9-year dataset, three core Opening Range Breakout (ORB) models were evaluated:

| Strategy Architecture | 8.9y Realized PnL | Win Rate | Profit Factor | Expectancy / Trade | Max Realized Drawdown | Challenge Pass Rate | Expected Cost to Pass | Funded Payout Rate | Institutional Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **5m ORB + 12 EMA (`orb_5m`)** | **-$147,530.40** | 9.55% | 0.11 | -$142.40 | -$147,530 | 0.0% | N/A (Ruined) | 0.0% | ❌ **DEBUNKED:** The original tearsheet (+70k) was an artifact of an intra-bar lookahead bug in the 0.5R trailing stop (`sweep_engine_5m.py:219-267`). Real execution collapses completely. |
| **15m High-Velocity ORB (`orb_hybrid_b`)** | **+$2,254.20** | 50.14% | 1.03 | +$2.18 | -$3,710.29 | 17.0% | $512.97 (~5.9 attempts) | 5.6% | ⚠️ **MARGINAL / HIGH RISK:** NQ 15m wicks trigger stops prematurely. Friction ($4.48/trade) consumes the edge. 83% of challenge attempts blow the $2,000 trailing barrier. |
| **30m Filtered Quant ORB (`orb_30m`)** | **+$10,626.48** | **51.62%** | **1.21** | **+$16.37** | **-$2,605.72** | **51.6%** | **$168.60 (~1.9 attempts)** | **30.8%** | 🏆 **VALIDATED CHAMPION:** Statistically robust, high expectancy, and **out-of-sample persistence increases in modern data (2023–2026: +$6,869 PnL, 53.3% WR, 1.28 PF, WFE: 2.47)**. |

---

## 5. Tested & Failed Hypotheses (Negative Knowledge)

To avoid reinventing failed mechanics, the following quantitative approaches have already been empirically tested and rejected:

1. **Tight Intra-Bar Trailing Ratchet (0.5R on 1m bars):**
   * *Hypothesis:* Ratcheting stops to `Peak - 0.5R` on 1m bars locks profit quickly.
   * *Empirical Result:* Nasdaq noise within 1-minute bars is 5–15 points. A 0.5R stop gets wicked out constantly before trends develop, destroying expectancy.
2. **15-Minute Opening Range on NQ:**
   * *Hypothesis:* 15m allows entering 15 minutes earlier than 30m, increasing trade frequency.
   * *Empirical Result:* The 09:30–09:45 window is dominated by opening auction imbalance clearing. False breakout rate is high; trade expectancy collapses from +$16.37 (on 30m) down to +$2.18 (on 15m).
3. **Strict Overnight Inventory Filter on 30m ORB:**
   * *Hypothesis:* If overnight is 80%+ Net Long, block 30m Long breakouts unless price touches Overnight VWAP.
   * *Empirical Result:* While sound in theory, on 30m ORB it dropped 8.9y PnL from +$10,626 to +$4,505. By 10:00 EST, a breakout in the direction of overnight volume is often an institutional "Open-Drive" trend day. Blocking it cut off the largest +2.5R winning runners.

---

## 6. Empirically Proven Market Microstructure Anomalies (CME NQ 2018–2026)

The following 4 structural anomalies were measured and verified across 2,170 triple-session trading days on the continuous NQ dataset:

```
Session Chronology:
[Asia Session: 18:00 - 02:00 EST] ──> [London Session: 03:00 - 09:29 EST] ──> [NY RTH: 09:30 - 15:45 EST]
```

### Anomaly 1: The New York Open London Low Sweep & Reversal (LLSR)
* **Finding:** During the 09:30–10:30 EST New York opening window, price sweeps below the London Session Low (Low < LL) in **38.2% of all trading days**.
* **Statistical Asymmetry:** When NY sweeps the London Low, **in 89.0% OF OCCURRENCES IT FAILS TO BREAK DOWN AND REVERSES BACK INSIDE THE LONDON RANGE**.
* **Microstructure Driver:** Stop-loss harvesting below the London range by institutional market makers looking for cheap liquidity to establish long inventory.

### Anomaly 2: Limit Order Book (LOB) Resting Liquidity Multi-Touch Fatigue
* **Finding:** Repeated touches of an established resistance level (e.g., Overnight High / Prior Day High):
  * **Touch #1:** 43.1% Rejection | 56.9% Breakout
  * **Touch #2:** 35.2% Rejection | 64.8% Breakout
  * **Touch #3:** 27.6% Rejection | 72.4% Breakout
  * **Touch #4+:** **18.1% Rejection | 81.9% Breakout Cascade**
* **Microstructure Driver:** Retail dogma states "levels get stronger with each touch". Order flow reality is the exact opposite: resting passive limit orders are absorbed and depleted by aggressive market orders. By touch 3–4, the liquidity reservoir is empty, causing rapid slippage/breakout cascades.

### Anomaly 3: 0DTE Options Strike Magnetism & Gamma Pinning (Post-May 2022)
* **Finding:** Average distance of NQ to the nearest 50-point strike at 15:59 EST compressed from **12.73 points (pre-2022)** to **12.09 points (post-2022)**.
* **Microstructure Driver:** Market maker dynamic delta-hedging of zero-DTE QQQ/NDX options creates deterministic mean-reverting pin forces toward high open-interest strikes into the 16:00 close.

### Anomaly 4: 15:00–15:45 Leveraged ETF Rebalancing Imbalance
* **Finding:** On days where NQ is up > +1.0% into 15:00 EST, leveraged ETFs (TQQQ, SQQQ) must mechanically buy index futures into the 16:00 close ($\Delta Q = A \cdot L(L-1)R$), creating a statistically significant positive drift between 15:00 and 15:45 EST.

---

## 7. Current Project Roadmap & Immediate Objectives

1. **Option A Sprint Sizing Deployment:**
   - Implement the Profit-Buffer Stepping protocol (2 MNQ base $\to$ 4 MNQ sprint when buffer $> \$1,500$) to compress evaluation pass time from 237 days down to 65–85 days.
2. **Cross-Asset Breadth Confirmation:**
   - Integrate CME E-mini S&P 500 (`ES`) VWAP alignment to filter false NQ breakouts.
3. **Standalone Microstructure Engine Development:**
   - Develop and validate an independent execution engine exploiting the 89.0% London Low Sweep reversal anomaly to provide an uncorrelated alpha source alongside the 30m ORB.
4. **NinjaTrader 8 & Tradeify Execution Bridge:**
   - Finalize `TradeifyExecutionManager.cs` unmanaged strategy for live broker deployment via Tradovate / Rithmic.

---

## 8. Prompt Template for the Research AI

```markdown
You are an expert quantitative researcher specializing in CME equity index futures (NQ/MNQ) and retail proprietary trading firm economics (specifically Tradeify $50K Growth with EOD trailing drawdown).

Read the project dossier above carefully. Your mission is to propose 3 to 5 novel, falsifiable, and mathematically rigorous quantitative hypotheses designed to either:
1. Significantly increase the trade expectancy (currently +$16.37) and capital velocity of the 30-Minute Filtered Quant ORB (Option A), OR
2. Design a standalone, uncorrelated intraday alpha engine exploiting one of the verified microstructure anomalies (e.g., London Low Sweep Reversal, LOB Multi-Touch Fatigue, or 15:00-15:45 Leveraged ETF Rebalancing).

For each proposed hypothesis, provide:
- Theoretical Rationale (Why does the institutional edge exist? Who is on the losing side?)
- Mathematical / Algorithmic Rules (Exact indicators, session anchors, entry trigger, stop loss placement, multi-tier profit targets, and invalidation criteria)
- Expected Impact on Metrics (Hypothesized effect on Win Rate, Profit Factor, Expectancy, and Max Drawdown)
- Potential Failure Modes / Risks under Tradeify's $2,000 EOD trailing floor and $1,250 daily loss limit
- Specific Backtest Formulation (How to test it on 1-minute continuous NQ data with zero lookahead)
```
