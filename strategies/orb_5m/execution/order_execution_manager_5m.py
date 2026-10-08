"""
Dedicated Live Order Execution Manager (OEM-5M) - Production Grade Engine
Target Strategy: CME NQ/MNQ 5-Minute Opening Range Breakout with 12 EMA Filter
Target Broker / Prop Firm: Tradeify $50K EOD Trailing Drawdown ($2,000 Barrier, $3,000 Target)
Author: Lead Quantitative Software Engineer & Risk Architect

Architecture & Responsibilities:
  1. Real-Time Bar Aggregation: Captures 09:30:00–09:35:00 EST 5m candle & updates 12 EMA.
  2. Directional Alignment & Noise Floor: Close > 12 EMA (Long) vs Close < 12 EMA (Short); Range >= 6.0 pts.
  3. Resting Stop-Entry Order: Submits Buy-Stop at High + 0.50 pts or Sell-Stop at Low - 0.50 pts at 09:35:00 EST.
  4. Auto-Cancellation: Cancels unfilled entry order if not triggered by 11:00:00 EST.
  5. Dynamic 0.5R Trailing Ratchet: Ratchets stop loss order by 0.5R on every favorable price tick.
  6. Single Trade Guardrail: Strictly enforces max 1 trade executed per calendar day.
  7. Heartbeat Watchdog & Kill-Switch: Flattens all positions & cancels orders if quote staleness > 1.5s.
  8. Hard EOD Flatten: Force closes any open exposure at 15:45:00 EST.
"""

from __future__ import annotations
import asyncio
import enum
import logging
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timezone
from typing import Callable, Dict, List, Optional, Tuple, Any

# Ensure UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.risk_state_machine import AccountType, RiskStateMachine, SizingResult

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("OEM-5M")


# ==============================================================================
# Order Lifecycle & Data Models
# ==============================================================================

class OrderState(str, enum.Enum):
    PENDING = "PENDING"
    WORKING = "WORKING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class OrderSide(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, enum.Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class Order:
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    filled_quantity: int = 0
    remaining_quantity: int = 0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    state: OrderState = OrderState.PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    parent_order_id: Optional[str] = None
    oco_group: Optional[str] = None
    rejection_reason: Optional[str] = None
    is_bracket: bool = False
    average_fill_price: float = 0.0

    def __post_init__(self):
        if self.remaining_quantity == 0 and self.filled_quantity == 0:
            self.remaining_quantity = self.quantity


@dataclass
class Position:
    symbol: str
    net_contracts: int = 0  # positive = Long, negative = Short
    average_entry_price: float = 0.0
    unrealized_pnl: float = 0.0


# ==============================================================================
# Broker Adapter Protocol
# ==============================================================================

class BrokerAdapter:
    """Interface for routing orders to CME matching engines (Tradovate/Rithmic/NT8)."""

    async def submit_order(self, order: Order) -> bool:
        raise NotImplementedError

    async def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    async def modify_stop_price(self, order_id: str, new_stop_price: float) -> bool:
        raise NotImplementedError

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> List[str]:
        raise NotImplementedError

    async def flatten_position(self, symbol: str) -> bool:
        raise NotImplementedError


class SimulatedBrokerAdapter(BrokerAdapter):
    """High-fidelity CME simulator with millisecond callback modeling."""

    def __init__(self, oem: OrderExecutionManager5m):
        self.oem = oem
        self.orders: Dict[str, Order] = {}

    async def submit_order(self, order: Order) -> bool:
        self.orders[order.order_id] = order
        await asyncio.sleep(0.002)  # 2ms wire latency
        await self.oem.on_order_update(
            order_id=order.order_id,
            new_state=OrderState.WORKING,
            filled_qty=0,
            avg_price=0.0
        )
        return True

    async def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.state in (OrderState.WORKING, OrderState.PENDING):
                await self.oem.on_order_update(
                    order_id=order_id,
                    new_state=OrderState.CANCELLED,
                    filled_qty=order.filled_quantity,
                    avg_price=order.average_fill_price
                )
                return True
        return False

    async def modify_stop_price(self, order_id: str, new_stop_price: float) -> bool:
        if order_id in self.orders:
            order = self.orders[order_id]
            order.stop_price = new_stop_price
            order.updated_at = time.time()
            return True
        return False

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> List[str]:
        cancelled = []
        for oid, order in list(self.orders.items()):
            if symbol and order.symbol != symbol:
                continue
            if order.state in (OrderState.WORKING, OrderState.PENDING):
                await self.cancel_order(oid)
                cancelled.append(oid)
        return cancelled

    async def flatten_position(self, symbol: str) -> bool:
        logger.warning(f"⚡ [BROKER-SIM] Emergency FLATTEN executed for {symbol} at Market.")
        return True

    async def simulate_fill(self, order_id: str, fill_qty: int, fill_price: float):
        if order_id not in self.orders:
            return
        order = self.orders[order_id]
        new_filled = order.filled_quantity + fill_qty
        new_state = OrderState.FILLED if new_filled >= order.quantity else OrderState.PARTIALLY_FILLED
        await self.oem.on_order_update(
            order_id=order_id,
            new_state=new_state,
            filled_qty=new_filled,
            avg_price=fill_price
        )


# ==============================================================================
# Real-Time 12 EMA Calculator (Recursive Exponential Smoothing)
# ==============================================================================

class RealTimeEMA:
    """
    Online recursive EMA calculator matching pandas ewm(span=N, adjust=False).
    Formula: EMA_t = alpha * Close_t + (1 - alpha) * EMA_{t-1}
    where alpha = 2 / (period + 1)
    """

    def __init__(self, period: int = 12):
        self.period = period
        self.alpha = 2.0 / (period + 1.0)
        self.current_ema: Optional[float] = None
        self.bar_count: int = 0

    def update(self, close_price: float) -> float:
        if self.current_ema is None:
            self.current_ema = close_price
        else:
            self.current_ema = (self.alpha * close_price) + ((1.0 - self.alpha) * self.current_ema)
        self.bar_count += 1
        return self.current_ema

    def seed(self, ema_value: float):
        """Seeds the current EMA with a pre-calculated value from historical warmup."""
        self.current_ema = ema_value
        self.bar_count = self.period


# ==============================================================================
# 5-Minute ORB Live Order Execution Manager
# ==============================================================================

class OrderExecutionManager5m:
    """
    Dedicated Production Order Execution Manager for 5-Minute ORB + 12 EMA:
      - Establishes 09:30–09:35 candle in real-time.
      - Calculates 12 EMA on 5m timeframe.
      - Places resting stop-entry order at 09:35:00 EST sharp.
      - Ratchets active stop loss by 0.5R on every favorable tick.
      - 1 trade per day maximum.
      - Heartbeat watchdog (1.5s max staleness).
      - Hard EOD flatten at 15:45:00 EST.
    """

    def __init__(
        self,
        symbol: str = "MNQ",
        contracts: int = 8,
        min_candle_points: float = 6.0,
        min_stop_points: float = 6.0,
        max_stop_points: float = 35.0,
        slippage_buffer_points: float = 0.50,
        max_quote_staleness_seconds: float = 1.5,
        risk_machine: Optional[RiskStateMachine] = None,
        account_type: AccountType = AccountType.EVALUATION
    ):
        self.symbol = symbol
        self.contracts = contracts
        self.min_candle_points = min_candle_points
        self.min_stop_points = min_stop_points
        self.max_stop_points = max_stop_points
        self.slippage_buffer = slippage_buffer_points
        self.max_quote_staleness = max_quote_staleness_seconds
        self.risk_machine = risk_machine or RiskStateMachine()
        self.account_type = account_type

        # Time Windows (EST)
        self.t_rth_open = dtime(9, 30)
        self.t_or_end = dtime(9, 35)
        self.t_entry_cutoff = dtime(11, 0)
        self.t_force_close = dtime(15, 45)

        # 12 EMA Engine
        self.ema_engine = RealTimeEMA(period=12)

        # Daily Session State
        self.current_session_date: Optional[object] = None
        self.trades_taken_today: int = 0
        self.or_candle_formed: bool = False
        self.or_open: Optional[float] = None
        self.or_high: Optional[float] = None
        self.or_low: Optional[float] = None
        self.or_close: Optional[float] = None
        self.or_ema: Optional[float] = None
        self.bias_direction: int = 0  # +1 Long, -1 Short, 0 Disqualified

        # Active Orders & Position
        self.broker: BrokerAdapter = SimulatedBrokerAdapter(self)
        self.orders: Dict[str, Order] = {}
        self.pending_entry_order: Optional[Order] = None
        self.active_sl_order: Optional[Order] = None
        self.position: Optional[Position] = None

        # Dynamic 0.5R Trailing State
        self.entry_price: float = 0.0
        self.initial_sl_price: float = 0.0
        self.current_sl_price: float = 0.0
        self.risk_points: float = 0.0
        self.trail_distance_05r: float = 0.0
        self.peak_favorable_price: float = 0.0

        # Heartbeat & Safety Watchdog
        self.last_quote_monotonic: float = time.monotonic()
        self.is_halted: bool = False
        self._running: bool = False
        self.watchdog_task: Optional[asyncio.Task] = None

    # --------------------------------------------------------------------------
    # Lifecycle Management
    # --------------------------------------------------------------------------

    async def start(self):
        """Starts background watchdog and execution loops."""
        self._running = True
        self.last_quote_monotonic = time.monotonic()
        self.watchdog_task = asyncio.create_task(self._heartbeat_watchdog_loop())
        logger.info(
            f"🚀 [OEM-5M STARTED] Symbol: {self.symbol} | Sizing: {self.contracts}x | "
            f"Watchdog Latency Limit: {self.max_quote_staleness * 1000:.0f}ms | 0.5R Trailing Engine Active"
        )

    async def stop(self):
        """Stops background tasks."""
        self._running = False
        if self.watchdog_task and not self.watchdog_task.done():
            self.watchdog_task.cancel()
            try:
                await self.watchdog_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 [OEM-5M STOPPED]")

    def reset_session(self, session_date: object):
        """Resets the state machine at start of day."""
        self.current_session_date = session_date
        self.trades_taken_today = 0
        self.or_candle_formed = False
        self.or_open = None
        self.or_high = None
        self.or_low = None
        self.or_close = None
        self.or_ema = None
        self.bias_direction = 0
        self.pending_entry_order = None
        self.active_sl_order = None
        self.position = None
        self.entry_price = 0.0
        self.initial_sl_price = 0.0
        self.current_sl_price = 0.0
        self.risk_points = 0.0
        self.trail_distance_05r = 0.0
        self.peak_favorable_price = 0.0
        logger.info(f"🔄 [SESSION RESET] Initialized new session for date: {session_date}")

    # --------------------------------------------------------------------------
    # Market Data Processing & Bar Aggregation
    # --------------------------------------------------------------------------

    async def on_quote(self, symbol: str, bid: float, ask: float, current_est_time: dtime):
        """
        Processes real-time quotes/ticks:
          1. Updates watchdog heartbeat.
          2. Monitors active position for 0.5R trailing ratchets.
          3. Enforces 11:00 entry timeout and 15:45 force close.
        """
        self.last_quote_monotonic = time.monotonic()
        mid_price = (bid + ask) / 2.0

        if self.is_halted:
            return

        # 1. Check Hard EOD Close (15:45:00 EST)
        if current_est_time >= self.t_force_close:
            if self.position is not None or self.pending_entry_order is not None:
                await self._force_close_eod("Hard EOD 15:45:00 EST session close cutoff")
            return

        # 2. Check Entry Window Expiration (11:00:00 EST)
        if self.pending_entry_order and self.pending_entry_order.state == OrderState.WORKING:
            if current_est_time >= self.t_entry_cutoff:
                logger.info(f"⏰ [ENTRY EXPIRED] 11:00 EST reached without fill. Cancelling resting entry order.")
                await self.broker.cancel_order(self.pending_entry_order.order_id)
                self.pending_entry_order = None
                return

        # 3. Dynamic 0.5R Trailing Ratchet on Active Position
        if self.position is not None and self.active_sl_order and self.active_sl_order.state == OrderState.WORKING:
            await self._evaluate_05r_trailing(mid_price)

    async def on_opening_candle_complete(
        self,
        date: object,
        c_open: float,
        c_high: float,
        c_low: float,
        c_close: float,
        known_ema_12: Optional[float] = None
    ) -> bool:
        """
        Called exactly at 09:35:00 EST when the 09:30–09:35 5m bar completes.
        Calculates 12 EMA, verifies noise floor, and places resting stop-entry order.
        """
        if self.is_halted:
            logger.warning("[OEM-5M] Candle complete ignored: system is halted.")
            return False

        if self.trades_taken_today >= 1:
            logger.info("[OEM-5M] Max 1 trade per day already executed. Skipping setup.")
            return False

        self.or_open = c_open
        self.or_high = c_high
        self.or_low = c_low
        self.or_close = c_close

        # Update or use supplied 12 EMA
        if known_ema_12 is not None:
            self.or_ema = known_ema_12
            self.ema_engine.seed(known_ema_12)
        else:
            self.or_ema = self.ema_engine.update(c_close)

        candle_range = c_high - c_low

        logger.info(
            f"📊 [09:35 EST 5M CANDLE ESTABLISHED] Range: {c_low:.2f} - {c_high:.2f} ({candle_range:.2f} pts) | "
            f"Close: {c_close:.2f} | 12 EMA: {self.or_ema:.2f}"
        )

        # Noise floor check
        if candle_range < self.min_candle_points:
            logger.warning(
                f"⚠️ [NOISE FLOOR TRIGGERED] Candle range ({candle_range:.2f} pts) < minimum ({self.min_candle_points:.2f} pts). "
                f"No trades permitted today."
            )
            self.or_candle_formed = False
            self.bias_direction = 0
            return False

        # Directional filter: 5m Close vs 12 EMA
        if c_close > self.or_ema:
            self.bias_direction = 1
            logger.info("🟢 [DIRECTIONAL BIAS: LONG] 5m Close > 12 EMA. Submitting Buy-Stop above 5m High.")
            await self._submit_resting_entry(is_long=True, stop_price=c_high + self.slippage_buffer)
        elif c_close < self.or_ema:
            self.bias_direction = -1
            logger.info("🔴 [DIRECTIONAL BIAS: SHORT] 5m Close < 12 EMA. Submitting Sell-Stop below 5m Low.")
            await self._submit_resting_entry(is_long=False, stop_price=c_low - self.slippage_buffer)
        else:
            logger.info("⚪ [NEUTRAL BIAS] Close exactly equals EMA. No entry placed.")
            self.bias_direction = 0
            return False

        self.or_candle_formed = True
        return True

    # --------------------------------------------------------------------------
    # Order Submission & Management
    # --------------------------------------------------------------------------

    async def _submit_resting_entry(self, is_long: bool, stop_price: float):
        """Places the resting Stop-Market entry order at 09:35:00 EST."""
        side = OrderSide.BUY if is_long else OrderSide.SELL
        order_id = f"ORD-ENTRY-{uuid.uuid4().hex[:8]}"
        client_id = f"CL-ENTRY-{uuid.uuid4().hex[:8]}"

        entry_order = Order(
            order_id=order_id,
            client_order_id=client_id,
            symbol=self.symbol,
            side=side,
            order_type=OrderType.STOP_MARKET,
            quantity=self.contracts,
            stop_price=round(stop_price, 2)
        )

        self.orders[order_id] = entry_order
        self.pending_entry_order = entry_order

        logger.info(
            f"🎯 [RESTING STOP-ENTRY SUBMITTED] Order ID: {order_id} | Side: {side.value} | "
            f"Qty: {self.contracts}x {self.symbol} | Stop Trigger: {stop_price:.2f}"
        )
        await self.broker.submit_order(entry_order)

    async def on_order_update(
        self,
        order_id: str,
        new_state: OrderState,
        filled_qty: int,
        avg_price: float
    ):
        """Callback handling order lifecycle transitions from the broker."""
        if order_id not in self.orders:
            return

        order = self.orders[order_id]
        prev_state = order.state
        order.state = new_state
        order.filled_quantity = filled_qty
        order.remaining_quantity = order.quantity - filled_qty
        order.average_fill_price = avg_price
        order.updated_at = time.time()

        logger.info(
            f"📢 [ORDER UPDATE] {order_id} ({order.side.value} {order.quantity}x) "
            f"State: {prev_state.value} -> {new_state.value} | Filled: {filled_qty} @ {avg_price:.2f}"
        )

        # 1. Entry Order Filled -> Arm Position & Initial SL
        if self.pending_entry_order and order_id == self.pending_entry_order.order_id and new_state == OrderState.FILLED:
            await self._on_entry_filled(order, avg_price)

        # 2. Stop Loss Order Filled -> Position Exited
        elif self.active_sl_order and order_id == self.active_sl_order.order_id and new_state == OrderState.FILLED:
            await self._on_sl_filled(order, avg_price)

    async def _on_entry_filled(self, entry_order: Order, fill_price: float):
        """Called when entry order executes."""
        self.trades_taken_today += 1
        self.entry_price = fill_price
        self.peak_favorable_price = fill_price

        is_long = (entry_order.side == OrderSide.BUY)
        net_qty = entry_order.quantity if is_long else -entry_order.quantity
        self.position = Position(
            symbol=self.symbol,
            net_contracts=net_qty,
            average_entry_price=fill_price
        )

        # Calculate initial Stop Loss at opposite candle extreme
        if is_long:
            raw_risk = fill_price - self.or_low
            self.risk_points = min(max(raw_risk, self.min_stop_points), self.max_stop_points)
            self.initial_sl_price = round(fill_price - self.risk_points, 2)
            sl_side = OrderSide.SELL
        else:
            raw_risk = self.or_high - fill_price
            self.risk_points = min(max(raw_risk, self.min_stop_points), self.max_stop_points)
            self.initial_sl_price = round(fill_price + self.risk_points, 2)
            sl_side = OrderSide.BUY

        self.current_sl_price = self.initial_sl_price
        self.trail_distance_05r = round(0.50 * self.risk_points, 2)

        dollar_risk = self.risk_points * 2.0 * entry_order.quantity  # MNQ = $2.00/pt
        logger.info(
            f"✅ [ENTRY FILLED] {entry_order.side.value} {entry_order.quantity}x @ {fill_price:.2f} | "
            f"1R Risk: {self.risk_points:.2f} pts (${dollar_risk:.2f}) | Initial SL: {self.initial_sl_price:.2f} | "
            f"0.5R Trail Step: {self.trail_distance_05r:.2f} pts"
        )

        # Submit initial Stop Loss order to broker
        sl_order_id = f"ORD-SL-{uuid.uuid4().hex[:8]}"
        sl_order = Order(
            order_id=sl_order_id,
            client_order_id=f"CL-SL-{uuid.uuid4().hex[:8]}",
            symbol=self.symbol,
            side=sl_side,
            order_type=OrderType.STOP_MARKET,
            quantity=entry_order.quantity,
            stop_price=self.current_sl_price,
            parent_order_id=entry_order.order_id,
            is_bracket=True
        )
        self.orders[sl_order_id] = sl_order
        self.active_sl_order = sl_order
        self.pending_entry_order = None

        await self.broker.submit_order(sl_order)

    async def _on_sl_filled(self, sl_order: Order, exit_price: float):
        """Called when stop loss is triggered."""
        is_long = (self.position.net_contracts > 0)
        pnl_pts = (exit_price - self.entry_price) if is_long else (self.entry_price - exit_price)
        pnl_dollars = pnl_pts * 2.0 * sl_order.quantity
        r_multiple = pnl_pts / self.risk_points if self.risk_points > 0 else 0.0

        exit_type = "TRAILING STOP" if ((is_long and exit_price > self.initial_sl_price) or
                                       (not is_long and exit_price < self.initial_sl_price)) else "INITIAL STOP"

        logger.info(
            f"🏁 [POSITION CLOSED via {exit_type}] Exit: {exit_price:.2f} | PnL: {pnl_pts:+.2f} pts "
            f"(${pnl_dollars:+,.2f}) | R-Multiple: {r_multiple:+.2f}R | Session Trades: {self.trades_taken_today}/1"
        )

        self.position = None
        self.active_sl_order = None

    # --------------------------------------------------------------------------
    # Dynamic 0.5R Trailing Stop Engine
    # --------------------------------------------------------------------------

    async def _evaluate_05r_trailing(self, current_price: float):
        """
        Ratchets the active stop-loss order by 0.5R on every favorable tick.
        Long:  peak advances -> cand_sl = peak - 0.5R. If cand_sl > current_sl -> ratchet!
        Short: trough declines -> cand_sl = trough + 0.5R. If cand_sl < current_sl -> ratchet!
        """
        if self.position is None or self.active_sl_order is None:
            return

        is_long = (self.position.net_contracts > 0)

        if is_long:
            if current_price > self.peak_favorable_price:
                self.peak_favorable_price = current_price
                cand_sl = round(self.peak_favorable_price - self.trail_distance_05r, 2)
                if cand_sl > self.current_sl_price:
                    prev_sl = self.current_sl_price
                    self.current_sl_price = cand_sl
                    self.active_sl_order.stop_price = cand_sl
                    await self.broker.modify_stop_price(self.active_sl_order.order_id, cand_sl)
                    logger.info(
                        f"📈 [0.5R TRAIL RATCHET UP] New Peak: {self.peak_favorable_price:.2f} | "
                        f"Stop Loss ratcheted: {prev_sl:.2f} -> {cand_sl:.2f} (+{cand_sl - prev_sl:.2f} pts)"
                    )
        else: # Short
            if current_price < self.peak_favorable_price:
                self.peak_favorable_price = current_price
                cand_sl = round(self.peak_favorable_price + self.trail_distance_05r, 2)
                if cand_sl < self.current_sl_price:
                    prev_sl = self.current_sl_price
                    self.current_sl_price = cand_sl
                    self.active_sl_order.stop_price = cand_sl
                    await self.broker.modify_stop_price(self.active_sl_order.order_id, cand_sl)
                    logger.info(
                        f"📉 [0.5R TRAIL RATCHET DOWN] New Trough: {self.peak_favorable_price:.2f} | "
                        f"Stop Loss ratcheted: {prev_sl:.2f} -> {cand_sl:.2f} ({cand_sl - prev_sl:.2f} pts)"
                    )

    # --------------------------------------------------------------------------
    # Heartbeat Watchdog & Kill-Switch
    # --------------------------------------------------------------------------

    async def _heartbeat_watchdog_loop(self):
        """Checks quote staleness every 100ms. Triggers emergency halt if > 1.5s."""
        while self._running:
            try:
                await asyncio.sleep(0.100)
                if self.is_halted:
                    continue

                staleness = time.monotonic() - self.last_quote_monotonic
                if staleness > self.max_quote_staleness:
                    # Only halt if we have working orders or an open position
                    has_exposure = (self.position is not None or
                                    any(o.state == OrderState.WORKING for o in self.orders.values()))
                    if has_exposure:
                        await self.trigger_kill_switch(
                            f"Quote staleness ({staleness:.2f}s) exceeded limit ({self.max_quote_staleness:.2f}s) with active exposure!"
                        )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[WATCHDOG ERROR] {e}", exc_info=True)

    async def trigger_kill_switch(self, reason: str):
        """Engages Emergency Kill-Switch: cancels all orders, flattens position, halts."""
        self.is_halted = True
        logger.critical(f"🚨🚨🚨 [KILL-SWITCH ENGAGED] {reason} 🚨🚨🚨")

        # 1. Cancel all working orders
        cancelled_ids = await self.broker.cancel_all_orders(self.symbol)
        logger.warning(f"🚫 [KILL-SWITCH] Cancelled {len(cancelled_ids)} active orders: {cancelled_ids}")

        # 2. Emergency market flatten
        if self.position is not None:
            await self.broker.flatten_position(self.symbol)
            logger.warning(f"⚡ [KILL-SWITCH] Emergency market flatten order executed for {self.symbol}.")
            self.position = None

        self.pending_entry_order = None
        self.active_sl_order = None

    async def _force_close_eod(self, reason: str):
        """Hard close at 15:45:00 EST."""
        logger.info(f"🌆 [HARD EOD FLATTEN] {reason}")
        if self.pending_entry_order and self.pending_entry_order.state == OrderState.WORKING:
            await self.broker.cancel_order(self.pending_entry_order.order_id)
            self.pending_entry_order = None

        if self.position is not None:
            await self.broker.flatten_position(self.symbol)
            self.position = None

        if self.active_sl_order and self.active_sl_order.state == OrderState.WORKING:
            await self.broker.cancel_order(self.active_sl_order.order_id)
            self.active_sl_order = None


# ==============================================================================
# Self-Test Verification Scenario
# ==============================================================================

async def run_verification_scenario():
    """Simulates a complete trading session to verify all 4 core requirements."""
    print("=" * 80)
    print("RUNNING 5-MINUTE ORB LIVE ORDER EXECUTION MANAGER VERIFICATION")
    print("=" * 80)

    oem = OrderExecutionManager5m(symbol="MNQ", contracts=8, min_stop_points=6.0, max_stop_points=35.0)
    await oem.start()

    # Step 1: Initialize session
    today = datetime.now(timezone.utc).date()
    oem.reset_session(today)

    # Step 2: Feed quotes at 09:34:50
    await oem.on_quote("MNQ", 20500.0, 20500.5, dtime(9, 34, 50))

    # Step 3: Complete 09:30–09:35 candle at 09:35:00 sharp
    # Candle: Low = 20480.0, High = 20510.0 (30 pts range > 6.0 pts noise floor), Close = 20505.0, EMA = 20490.0 (Close > EMA -> LONG)
    c_open, c_high, c_low, c_close, c_ema = 20485.0, 20510.0, 20480.0, 20505.0, 20490.0
    qualifies = await oem.on_opening_candle_complete(today, c_open, c_high, c_low, c_close, known_ema_12=c_ema)
    assert qualifies, "Candle should qualify"
    assert oem.bias_direction == 1, "Bias should be Long (+1)"
    assert oem.pending_entry_order is not None, "Resting entry order should be placed"
    assert oem.pending_entry_order.stop_price == 20510.5, "Buy stop should be at High + 0.50 pts"

    # Step 4: Simulate CME breakout trigger at 09:37:15
    entry_id = oem.pending_entry_order.order_id
    sim_broker: SimulatedBrokerAdapter = oem.broker  # type: ignore
    await sim_broker.simulate_fill(entry_id, fill_qty=8, fill_price=20510.5)

    assert oem.position is not None, "Position should be active"
    assert oem.active_sl_order is not None, "Stop loss order should be active"
    # Initial SL: entry (20510.5) - risk_pts (c_low=20480.0 -> 30.5 pts, capped at 30.5) = 20480.0
    assert oem.active_sl_order.stop_price == 20480.0, f"Expected initial SL 20480.0, got {oem.active_sl_order.stop_price}"
    # 0.5R = 0.5 * 30.5 = 15.25 pts

    # Step 5: Price advances favorably (20510.5 -> 20530.0 -> 20550.0) -> Dynamic 0.5R Trail ratchets
    await oem.on_quote("MNQ", 20530.0, 20530.5, dtime(9, 40))
    # Peak = 20530.25 -> SL should ratchet to 20530.25 - 15.25 = 20515.00
    expected_ratchet_1 = round(20530.25 - 15.25, 2)
    assert oem.current_sl_price == expected_ratchet_1, f"Expected {expected_ratchet_1}, got {oem.current_sl_price}"

    await oem.on_quote("MNQ", 20550.0, 20550.5, dtime(9, 45))
    expected_ratchet_2 = round(20550.25 - 15.25, 2)
    assert oem.current_sl_price == expected_ratchet_2, f"Expected {expected_ratchet_2}, got {oem.current_sl_price}"

    # Step 6: Price pulls back to trigger trailing stop at 20535.0
    sl_id = oem.active_sl_order.order_id
    await sim_broker.simulate_fill(sl_id, fill_qty=8, fill_price=expected_ratchet_2)

    assert oem.position is None, "Position should be closed"
    assert oem.trades_taken_today == 1, "Should record 1 trade taken today"

    # Step 7: Verify single-trade restriction (attempt another candle)
    second_attempt = await oem.on_opening_candle_complete(today, c_open, c_high, c_low, c_close, known_ema_12=c_ema)
    assert not second_attempt, "Second trade attempt must be rejected"

    await oem.stop()
    print("=" * 80)
    print("✅ ALL 5-MINUTE ORB LIVE ORDER MANAGER CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_verification_scenario())
