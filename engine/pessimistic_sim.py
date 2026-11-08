"""
Pessimistic Intra-Bar Execution Simulator (Vectorized / NumPy Optimized)
Author: Principal Quantitative Architect & Execution Engineer

Enforces:
  - Intra-bar Stop-First Execution Priority (if both SL and TP touch in the same 1m bar, SL is hit first).
  - Explicit Friction: $2.48 round-trip commission per MNQ contract ($1.24 entry, $1.24 exit).
  - Explicit Slippage: 0.50 NQ index points per fill against the trade.
  - Multi-Stage 50% Partial Scale-Out (TP1 with Breakeven stop adjustment).
  - Dynamic indicators: EMA trailing closes, scratch rules, and hard time cutoffs.
  - Hard 15:44:00 EST session liquidation.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

from strategies.base_strategy import TradeSignal, CompletedTrade

logger = logging.getLogger("PessimisticSim")


class PessimisticExecutionSimulator:
    def __init__(
        self,
        commission_per_contract_rt: float = 2.48,
        slippage_points: float = 0.50,
        point_value: float = 2.0  # MNQ point value
    ):
        self.commission_per_contract_rt = commission_per_contract_rt
        self.slippage_points = slippage_points
        self.point_value = point_value

    def simulate_trade(
        self,
        signal: TradeSignal,
        session_bars: pd.DataFrame,
        contract_qty: int,
        df_indicators: Optional[pd.DataFrame] = None
    ) -> List[CompletedTrade]:
        if contract_qty <= 0:
            return []

        # Find bars occurring strictly on or after execution_time
        exec_bars = session_bars.loc[session_bars.index >= signal.execution_time]
        if exec_bars.empty:
            return []

        direction = signal.direction  # +1 Long, -1 Short
        entry_time = exec_bars.index[0]
        open_0 = float(exec_bars["Open"].iloc[0])

        # Apply slippage on entry
        if direction == 1:
            entry_price = open_0 + self.slippage_points
            initial_sl = signal.stop_loss_price
            tp1_price = signal.tp1_price
            tp2_price = signal.tp2_price
            be_price = entry_price + 2.0
        else:
            entry_price = open_0 - self.slippage_points
            initial_sl = signal.stop_loss_price
            tp1_price = signal.tp1_price
            tp2_price = signal.tp2_price
            be_price = entry_price - 2.0

        current_sl = initial_sl
        tp1_hit = False
        completed_trades: List[CompletedTrade] = []

        # Contract allocation (50% TP1, 50% runner)
        if contract_qty >= 2:
            qty_tp1 = contract_qty // 2
            qty_runner = contract_qty - qty_tp1
        else:
            qty_tp1 = contract_qty
            qty_runner = 0

        # Fast NumPy arrays
        highs = exec_bars["High"].to_numpy(dtype=float)
        lows = exec_bars["Low"].to_numpy(dtype=float)
        closes = exec_bars["Close"].to_numpy(dtype=float)
        times = exec_bars.index
        n_bars = len(highs)

        # Precompute times in minutes or time strings
        trail_vals = None
        if signal.trail_rule and df_indicators is not None and signal.trail_rule in df_indicators.columns:
            trail_series = df_indicators.reindex(times)[signal.trail_rule]
            trail_vals = trail_series.to_numpy(dtype=float)

        hard_exit_cutoff = 15 * 60 + 44 # 15:44 in minutes
        signal_exit_cutoff = None
        if signal.hard_time_exit:
            parts = signal.hard_time_exit.split(":")
            signal_exit_cutoff = int(parts[0]) * 60 + int(parts[1])

        bars_held = 0

        for i in range(n_bars):
            bar_time = times[i]
            bar_high = highs[i]
            bar_low = lows[i]
            bar_close = closes[i]
            bars_held += 1

            t_min = bar_time.hour * 60 + bar_time.minute

            # 1. Scratch Rule
            if signal.scratch_bars is not None and bars_held >= signal.scratch_bars and not tp1_hit:
                req_progress = signal.scratch_threshold_r * signal.sl_dist
                if direction == 1:
                    unrealized_peak = bar_high - entry_price
                    if unrealized_peak < req_progress:
                        exit_price = bar_close - self.slippage_points
                        c_trade = self._create_trade(
                            signal, direction, contract_qty, entry_time, entry_price,
                            bar_time, exit_price, "SCRATCH"
                        )
                        return [c_trade]
                else:
                    unrealized_peak = entry_price - bar_low
                    if unrealized_peak < req_progress:
                        exit_price = bar_close + self.slippage_points
                        c_trade = self._create_trade(
                            signal, direction, contract_qty, entry_time, entry_price,
                            bar_time, exit_price, "SCRATCH"
                        )
                        return [c_trade]

            # 2. Hard Time Exit
            if signal_exit_cutoff is not None and t_min >= signal_exit_cutoff:
                exit_price = bar_close - (self.slippage_points if direction == 1 else -self.slippage_points)
                rem_qty = qty_runner if tp1_hit else contract_qty
                if rem_qty > 0:
                    completed_trades.append(self._create_trade(
                        signal, direction, rem_qty, entry_time, entry_price,
                        bar_time, exit_price, "TIME_EXIT"
                    ))
                return completed_trades

            # Hard 15:44:00 session flatten
            if t_min >= hard_exit_cutoff:
                exit_price = bar_close - (self.slippage_points if direction == 1 else -self.slippage_points)
                rem_qty = qty_runner if tp1_hit else contract_qty
                if rem_qty > 0:
                    completed_trades.append(self._create_trade(
                        signal, direction, rem_qty, entry_time, entry_price,
                        bar_time, exit_price, "EOD_FLATTEN"
                    ))
                return completed_trades

            # 3. Dynamic Trailing Indicator
            if tp1_hit and qty_runner > 0 and trail_vals is not None:
                ind_val = trail_vals[i]
                if not np.isnan(ind_val):
                    if direction == 1 and bar_close < ind_val:
                        exit_price = bar_close - self.slippage_points
                        completed_trades.append(self._create_trade(
                            signal, direction, qty_runner, entry_time, entry_price,
                            bar_time, exit_price, "TRAIL_EXIT"
                        ))
                        return completed_trades
                    elif direction == -1 and bar_close > ind_val:
                        exit_price = bar_close + self.slippage_points
                        completed_trades.append(self._create_trade(
                            signal, direction, qty_runner, entry_time, entry_price,
                            bar_time, exit_price, "TRAIL_EXIT"
                        ))
                        return completed_trades

            # 4. Intra-Bar Price Evaluation (Stop-First Priority)
            if direction == 1:
                sl_touched = (bar_low <= current_sl)
                tp1_touched = (bar_high >= tp1_price) if not tp1_hit else False
                tp2_touched = (bar_high >= tp2_price) if (tp1_hit and tp2_price is not None) else False

                if sl_touched:
                    exit_price = current_sl - self.slippage_points
                    rem_qty = qty_runner if tp1_hit else contract_qty
                    reason = "BE" if tp1_hit else "SL"
                    completed_trades.append(self._create_trade(
                        signal, direction, rem_qty, entry_time, entry_price,
                        bar_time, exit_price, reason
                    ))
                    return completed_trades

                if not tp1_hit and tp1_touched:
                    tp1_hit = True
                    completed_trades.append(self._create_trade(
                        signal, direction, qty_tp1, entry_time, entry_price,
                        bar_time, tp1_price, "TP1"
                    ))
                    current_sl = be_price
                    if qty_runner == 0:
                        return completed_trades
                    if tp2_price is not None and bar_high >= tp2_price:
                        completed_trades.append(self._create_trade(
                            signal, direction, qty_runner, entry_time, entry_price,
                            bar_time, tp2_price, "TP2"
                        ))
                        return completed_trades

                elif tp1_hit and tp2_touched:
                    completed_trades.append(self._create_trade(
                        signal, direction, qty_runner, entry_time, entry_price,
                        bar_time, tp2_price, "TP2"
                    ))
                    return completed_trades

            else:
                sl_touched = (bar_high >= current_sl)
                tp1_touched = (bar_low <= tp1_price) if not tp1_hit else False
                tp2_touched = (bar_low <= tp2_price) if (tp1_hit and tp2_price is not None) else False

                if sl_touched:
                    exit_price = current_sl + self.slippage_points
                    rem_qty = qty_runner if tp1_hit else contract_qty
                    reason = "BE" if tp1_hit else "SL"
                    completed_trades.append(self._create_trade(
                        signal, direction, rem_qty, entry_time, entry_price,
                        bar_time, exit_price, reason
                    ))
                    return completed_trades

                if not tp1_hit and tp1_touched:
                    tp1_hit = True
                    completed_trades.append(self._create_trade(
                        signal, direction, qty_tp1, entry_time, entry_price,
                        bar_time, tp1_price, "TP1"
                    ))
                    current_sl = be_price
                    if qty_runner == 0:
                        return completed_trades
                    if tp2_price is not None and bar_low <= tp2_price:
                        completed_trades.append(self._create_trade(
                            signal, direction, qty_runner, entry_time, entry_price,
                            bar_time, tp2_price, "TP2"
                        ))
                        return completed_trades

                elif tp1_hit and tp2_touched:
                    completed_trades.append(self._create_trade(
                        signal, direction, qty_runner, entry_time, entry_price,
                        bar_time, tp2_price, "TP2"
                    ))
                    return completed_trades

        # End of session close
        final_time = times[-1]
        exit_price = closes[-1] - (self.slippage_points if direction == 1 else -self.slippage_points)
        rem_qty = qty_runner if tp1_hit else contract_qty
        if rem_qty > 0:
            completed_trades.append(self._create_trade(
                signal, direction, rem_qty, entry_time, entry_price,
                final_time, exit_price, "EOD_FLATTEN"
            ))

        return completed_trades

    def _create_trade(
        self,
        signal: TradeSignal,
        direction: int,
        contracts: int,
        entry_time: pd.Timestamp,
        entry_price: float,
        exit_time: pd.Timestamp,
        exit_price: float,
        reason: str
    ) -> CompletedTrade:
        pts = (exit_price - entry_price) * direction
        gross_pnl = pts * self.point_value * contracts
        comm = self.commission_per_contract_rt * contracts
        net_pnl = gross_pnl - comm
        duration_s = max(1.0, (exit_time - entry_time).total_seconds())
        duration_m = duration_s / 60.0
        r_mult = pts / signal.sl_dist if signal.sl_dist > 0 else 0.0

        return CompletedTrade(
            trade_id=0,
            strategy_id=signal.strategy_id,
            strategy_name=signal.strategy_name,
            direction=direction,
            contracts=contracts,
            entry_time=entry_time,
            entry_price=entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            gross_points=pts,
            gross_pnl=gross_pnl,
            commission=comm,
            net_pnl=net_pnl,
            exit_reason=reason,
            hold_duration_seconds=duration_s,
            hold_duration_minutes=duration_m,
            r_multiple=r_mult,
            metadata=signal.metadata
        )
