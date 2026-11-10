"""
Pessimistic Intra-Bar Execution Simulator (Phase 2 Upgraded)
Author: Principal Quantitative Architect & Execution Engineer

Features:
  - Intra-bar Stop-First Execution Priority (if both SL and TP touch in the same bar, SL is hit first).
  - Explicit Friction: $2.48 round-trip commission per MNQ contract ($1.24 entry, $1.24 exit).
  - Explicit Slippage: 0.50 NQ index points per fill against the trade.
  - Full-Position Asymmetric Execution Mode (scale_out_tp1 = False): 100% contracts held to asymmetric target/trail.
  - Legacy 50% Scale-Out Mode (scale_out_tp1 = True).
  - Delayed Breakeven Activation (be_trigger_r): Stop moves to BE+2.0 only after reaching e.g. +1.4R or +1.5R.
  - Dynamic RTH VWAP / EMA Trailing Exits on confirmed bar closes.
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
            tp_target = signal.tp1_price
            be_price = entry_price + 2.0
            if signal.be_trigger_r is not None and signal.be_trigger_r > 0:
                be_trigger_price = entry_price + (signal.be_trigger_r * signal.sl_dist)
            else:
                be_trigger_price = None
        else:
            entry_price = open_0 - self.slippage_points
            initial_sl = signal.stop_loss_price
            tp_target = signal.tp1_price
            be_price = entry_price - 2.0
            if signal.be_trigger_r is not None and signal.be_trigger_r > 0:
                be_trigger_price = entry_price - (signal.be_trigger_r * signal.sl_dist)
            else:
                be_trigger_price = None

        current_sl = initial_sl
        is_be_activated = False
        tp1_hit = False
        completed_trades: List[CompletedTrade] = []

        scale_out_mode = signal.scale_out_tp1
        if scale_out_mode:
            qty_tp1 = contract_qty // 2 if contract_qty >= 2 else contract_qty
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

        # Precompute trailing indicator values if present
        trail_vals = None
        trail_upper_vals = None
        trail_lower_vals = None

        if signal.trail_rule and df_indicators is not None:
            if signal.trail_rule == "VWAP_BAND" and "UpperBand" in df_indicators.columns and "VWAP" in df_indicators.columns:
                trail_upper_vals = df_indicators.reindex(times)["UpperBand"].to_numpy(dtype=float)
                trail_lower_vals = df_indicators.reindex(times)["LowerBand"].to_numpy(dtype=float)
                trail_vals = df_indicators.reindex(times)["VWAP"].to_numpy(dtype=float)
            elif signal.trail_rule in df_indicators.columns:
                trail_vals = df_indicators.reindex(times)[signal.trail_rule].to_numpy(dtype=float)

        hard_exit_cutoff = 15 * 60 + 44  # 15:44 in minutes
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
            if signal.scratch_bars is not None and bars_held >= signal.scratch_bars and not is_be_activated and not tp1_hit:
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

            # 2. Hard Time Exit Check
            if signal_exit_cutoff is not None and t_min >= signal_exit_cutoff:
                exit_price = bar_close - (self.slippage_points if direction == 1 else -self.slippage_points)
                rem_qty = qty_runner if (scale_out_mode and tp1_hit) else contract_qty
                if rem_qty > 0:
                    completed_trades.append(self._create_trade(
                        signal, direction, rem_qty, entry_time, entry_price,
                        bar_time, exit_price, "TIME_EXIT"
                    ))
                return completed_trades

            # Hard 15:44:00 session flatten
            if t_min >= hard_exit_cutoff:
                exit_price = bar_close - (self.slippage_points if direction == 1 else -self.slippage_points)
                rem_qty = qty_runner if (scale_out_mode and tp1_hit) else contract_qty
                if rem_qty > 0:
                    completed_trades.append(self._create_trade(
                        signal, direction, rem_qty, entry_time, entry_price,
                        bar_time, exit_price, "EOD_FLATTEN"
                    ))
                return completed_trades

            # 3. Delayed Breakeven Check (Disabled by default if be_trigger_price is None)
            if be_trigger_price is not None and not is_be_activated:
                if direction == 1 and bar_high >= be_trigger_price:
                    is_be_activated = True
                    current_sl = be_price
                elif direction == -1 and bar_low <= be_trigger_price:
                    is_be_activated = True
                    current_sl = be_price

            # 3b. Maróy (2025, SSRN 5095349) Non-Dilutive High-Water-Mark Ratchet Floor
            if signal.ratchet_tiers:
                for req_r, lock_r in signal.ratchet_tiers:
                    if direction == 1:
                        if bar_high - entry_price >= req_r * signal.sl_dist:
                            ratchet_price = entry_price + (lock_r * signal.sl_dist)
                            if ratchet_price > current_sl:
                                current_sl = ratchet_price
                    else:
                        if entry_price - bar_low >= req_r * signal.sl_dist:
                            ratchet_price = entry_price - (lock_r * signal.sl_dist)
                            if ratchet_price < current_sl:
                                current_sl = ratchet_price

            # 4. Dynamic Trailing Indicator Exit Check (Evaluated on confirmed 5-minute candle close)
            # Engage trailing once activation threshold is reached (e.g. is_be_activated or immediate)
            trail_active = is_be_activated or (signal.trail_activation_r <= 0.0)
            is_5m_close = ((bar_time.minute + 1) % 5 == 0) or (i + 1 < n_bars and (times[i+1].minute // 5 != bar_time.minute // 5)) or (i == n_bars - 1)
            if trail_active and trail_vals is not None and is_5m_close:
                val = trail_vals[i]
                if not np.isnan(val):
                    if signal.trail_rule == "VWAP_BAND" and trail_upper_vals is not None and trail_lower_vals is not None:
                        u_band = trail_upper_vals[i]
                        l_band = trail_lower_vals[i]
                        band_mode = getattr(signal, "trail_band_mode", "WIDE")
                        if direction == 1:
                            if band_mode == "TIGHT":  # Zarattini (2024, p.10) tightened trail: max(UB, VWAP)
                                threshold = max(u_band, val) if not np.isnan(u_band) else val
                            else:  # WIDE trail: min(UB, VWAP)
                                threshold = min(u_band, val) if not np.isnan(u_band) else val
                            if bar_close < threshold:
                                exit_price = bar_close - self.slippage_points
                                rem_qty = qty_runner if (scale_out_mode and tp1_hit) else contract_qty
                                completed_trades.append(self._create_trade(
                                    signal, direction, rem_qty, entry_time, entry_price,
                                    bar_time, exit_price, "TRAIL_EXIT"
                                ))
                                return completed_trades
                        else:
                            if band_mode == "TIGHT":  # Zarattini (2024, p.10) tightened trail: min(LB, VWAP)
                                threshold = min(l_band, val) if not np.isnan(l_band) else val
                            else:  # WIDE trail: max(LB, VWAP)
                                threshold = max(l_band, val) if not np.isnan(l_band) else val
                            if bar_close > threshold:
                                exit_price = bar_close + self.slippage_points
                                rem_qty = qty_runner if (scale_out_mode and tp1_hit) else contract_qty
                                completed_trades.append(self._create_trade(
                                    signal, direction, rem_qty, entry_time, entry_price,
                                    bar_time, exit_price, "TRAIL_EXIT"
                                ))
                                return completed_trades
                    else:
                        # Standard VWAP or EMA confirmed 5m close cross
                        if direction == 1 and bar_close < val:
                            exit_price = bar_close - self.slippage_points
                            rem_qty = qty_runner if (scale_out_mode and tp1_hit) else contract_qty
                            completed_trades.append(self._create_trade(
                                signal, direction, rem_qty, entry_time, entry_price,
                                bar_time, exit_price, "TRAIL_EXIT"
                            ))
                            return completed_trades
                        elif direction == -1 and bar_close > val:
                            exit_price = bar_close + self.slippage_points
                            rem_qty = qty_runner if (scale_out_mode and tp1_hit) else contract_qty
                            completed_trades.append(self._create_trade(
                                signal, direction, rem_qty, entry_time, entry_price,
                                bar_time, exit_price, "TRAIL_EXIT"
                            ))
                            return completed_trades

            # 5. Intra-Bar Price Evaluation (Stop-First Pessimistic Priority)
            if direction == 1:
                sl_touched = (bar_low <= current_sl)
                tp_touched = (tp_target is not None and bar_high >= tp_target)

                # PESSIMISTIC PRIORITY: Stop Loss is executed first
                if sl_touched:
                    exit_price = current_sl - self.slippage_points
                    rem_qty = qty_runner if (scale_out_mode and tp1_hit) else contract_qty
                    reason = "BE" if is_be_activated else "SL"
                    completed_trades.append(self._create_trade(
                        signal, direction, rem_qty, entry_time, entry_price,
                        bar_time, exit_price, reason
                    ))
                    return completed_trades

                # Full-Position Mode Target Hit
                if not scale_out_mode and tp_touched:
                    completed_trades.append(self._create_trade(
                        signal, direction, contract_qty, entry_time, entry_price,
                        bar_time, tp_target, "TP_FULL"
                    ))
                    return completed_trades

                # Scale-Out Mode TP1 Hit
                elif scale_out_mode and not tp1_hit and tp_touched:
                    tp1_hit = True
                    is_be_activated = True
                    current_sl = be_price
                    completed_trades.append(self._create_trade(
                        signal, direction, qty_tp1, entry_time, entry_price,
                        bar_time, tp_target, "TP1"
                    ))
                    if qty_runner == 0:
                        return completed_trades
                    if signal.tp2_price is not None and bar_high >= signal.tp2_price:
                        completed_trades.append(self._create_trade(
                            signal, direction, qty_runner, entry_time, entry_price,
                            bar_time, signal.tp2_price, "TP2"
                        ))
                        return completed_trades

                elif scale_out_mode and tp1_hit and signal.tp2_price is not None and bar_high >= signal.tp2_price:
                    completed_trades.append(self._create_trade(
                        signal, direction, qty_runner, entry_time, entry_price,
                        bar_time, signal.tp2_price, "TP2"
                    ))
                    return completed_trades

            else:
                # Short Position Evaluation
                sl_touched = (bar_high >= current_sl)
                tp_touched = (tp_target is not None and bar_low <= tp_target)

                # Stop-First Pessimistic Priority
                if sl_touched:
                    exit_price = current_sl + self.slippage_points
                    rem_qty = qty_runner if (scale_out_mode and tp1_hit) else contract_qty
                    reason = "BE" if is_be_activated else "SL"
                    completed_trades.append(self._create_trade(
                        signal, direction, rem_qty, entry_time, entry_price,
                        bar_time, exit_price, reason
                    ))
                    return completed_trades

                # Full-Position Mode Target Hit
                if not scale_out_mode and tp_touched:
                    completed_trades.append(self._create_trade(
                        signal, direction, contract_qty, entry_time, entry_price,
                        bar_time, tp_target, "TP_FULL"
                    ))
                    return completed_trades

                # Scale-Out Mode TP1 Hit
                elif scale_out_mode and not tp1_hit and tp_touched:
                    tp1_hit = True
                    is_be_activated = True
                    current_sl = be_price
                    completed_trades.append(self._create_trade(
                        signal, direction, qty_tp1, entry_time, entry_price,
                        bar_time, tp_target, "TP1"
                    ))
                    if qty_runner == 0:
                        return completed_trades
                    if signal.tp2_price is not None and bar_low <= signal.tp2_price:
                        completed_trades.append(self._create_trade(
                            signal, direction, qty_runner, entry_time, entry_price,
                            bar_time, signal.tp2_price, "TP2"
                        ))
                        return completed_trades

                elif scale_out_mode and tp1_hit and signal.tp2_price is not None and bar_low <= signal.tp2_price:
                    completed_trades.append(self._create_trade(
                        signal, direction, qty_runner, entry_time, entry_price,
                        bar_time, signal.tp2_price, "TP2"
                    ))
                    return completed_trades

        # End of session close
        final_time = times[-1]
        exit_price = closes[-1] - (self.slippage_points if direction == 1 else -self.slippage_points)
        rem_qty = qty_runner if (scale_out_mode and tp1_hit) else contract_qty
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
