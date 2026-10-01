"""
Order Execution Manager (OEM) - Institutional Grade Futures Execution Bridge
Author: Senior Execution Engineer
Target Systems: Tradeify / Lucid EOD Trailing Drawdown Accounts (CME NQ / MNQ)

Architecture:
  - Task 1: Modular async architecture interfacing with Risk State Machine
  - Task 2: Order Lifecycle Tracker (PENDING, WORKING, PARTIALLY_FILLED, FILLED, REJECTED, CANCELLED)
  - Task 3: Server-side Resting Bracket Orders (OCO: Stop-Loss + Take-Profit) with Partial Fill handling
  - Task 4: Millisecond Heartbeat & Kill-Switch Watchdog (Flatten All & Halt if quote age > 1.5s)
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
from typing import Callable, Dict, List, Optional, Tuple

# Support UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Interface to the Core Risk State Machine
from core.risk_state_machine import AccountType, RiskStateMachine, SizingResult

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("OrderExecutionManager")


# ==============================================================================
# Task 2: Data Models & Lifecycle States
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
# Broker Adapter Protocol (Interface for Tradovate, Rithmic, NinjaTrader, Simulator)
# ==============================================================================

class BrokerAdapter:
    """Abstract interface for server-side order routing and market data streams."""

    async def submit_order(self, order: Order) -> bool:
        raise NotImplementedError

    async def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> List[str]:
        raise NotImplementedError

    async def flatten_position(self, symbol: str) -> bool:
        raise NotImplementedError

    async def submit_server_oco_bracket(
        self,
        symbol: str,
        quantity: int,
        stop_side: OrderSide,
        stop_price: float,
        target_side: OrderSide,
        target_price: float,
        oco_group_id: str,
        parent_order_id: str
    ) -> Tuple[Order, Order]:
        raise NotImplementedError


# ==============================================================================
# Simulated High-Performance Broker Adapter (For verification & local testing)
# ==============================================================================

class SimulatedBrokerAdapter(BrokerAdapter):
    """
    High-fidelity broker simulator mimicking CME matching engine callbacks,
    server-side resting OCO brackets, and network events.
    """

    def __init__(self, oem: OrderExecutionManager):
        self.oem = oem
        self.orders: Dict[str, Order] = {}
        self.oco_groups: Dict[str, List[str]] = {}

    async def submit_order(self, order: Order) -> bool:
        self.orders[order.order_id] = order
        # Simulate network latency to matching engine (e.g. 5ms)
        await asyncio.sleep(0.005)
        # Transition PENDING -> WORKING
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
            if order.state in (OrderState.WORKING, OrderState.PARTIALLY_FILLED, OrderState.PENDING):
                await self.oem.on_order_update(
                    order_id=order_id,
                    new_state=OrderState.CANCELLED,
                    filled_qty=order.filled_quantity,
                    avg_price=order.average_fill_price
                )
                return True
        return False

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> List[str]:
        cancelled = []
        for oid, order in list(self.orders.items()):
            if symbol and order.symbol != symbol:
                continue
            if order.state in (OrderState.WORKING, OrderState.PARTIALLY_FILLED, OrderState.PENDING):
                await self.cancel_order(oid)
                cancelled.append(oid)
        return cancelled

    async def flatten_position(self, symbol: str) -> bool:
        # Immediate market close execution
        logger.warning(f"⚡ [BROKER-SIMULATOR] Emergency FLATTEN executed for {symbol} at Market.")
        return True

    async def submit_server_oco_bracket(
        self,
        symbol: str,
        quantity: int,
        stop_side: OrderSide,
        stop_price: float,
        target_side: OrderSide,
        target_price: float,
        oco_group_id: str,
        parent_order_id: str
    ) -> Tuple[Order, Order]:
        """Submits server-side resting OCO bracket."""
        sl_order = Order(
            order_id=f"ORD-SL-{uuid.uuid4().hex[:8]}",
            client_order_id=f"CL-SL-{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            side=stop_side,
            order_type=OrderType.STOP_MARKET,
            quantity=quantity,
            stop_price=stop_price,
            parent_order_id=parent_order_id,
            oco_group=oco_group_id,
            is_bracket=True,
            state=OrderState.WORKING
        )

        tp_order = Order(
            order_id=f"ORD-TP-{uuid.uuid4().hex[:8]}",
            client_order_id=f"CL-TP-{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            side=target_side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=target_price,
            parent_order_id=parent_order_id,
            oco_group=oco_group_id,
            is_bracket=True,
            state=OrderState.WORKING
        )

        self.orders[sl_order.order_id] = sl_order
        self.orders[tp_order.order_id] = tp_order
        self.oco_groups[oco_group_id] = [sl_order.order_id, tp_order.order_id]

        logger.info(
            f"🛡️  [SERVER-OCO REGISTERED] Group: {oco_group_id} | Qty: {quantity} | "
            f"SL @ {stop_price:.2f} ({sl_order.order_id}) | TP @ {target_price:.2f} ({tp_order.order_id})"
        )
        return sl_order, tp_order

    async def simulate_fill(self, order_id: str, fill_qty: int, fill_price: float):
        """Simulates CME execution callback."""
        if order_id not in self.orders:
            return
        order = self.orders[order_id]
        new_filled = order.filled_quantity + fill_qty
        new_state = OrderState.FILLED if new_filled >= order.quantity else OrderState.PARTIALLY_FILLED

        # Handle OCO cross-cancellation if a bracket leg filled
        if order.is_bracket and order.oco_group in self.oco_groups:
            partner_ids = [oid for oid in self.oco_groups[order.oco_group] if oid != order_id]
            for partner_id in partner_ids:
                logger.info(f"🔄 [OCO TRIGGER] Leg {order_id} filled; auto-cancelling counterpart {partner_id}.")
                await self.cancel_order(partner_id)

        await self.oem.on_order_update(
            order_id=order_id,
            new_state=new_state,
            filled_qty=new_filled,
            avg_price=fill_price
        )


# ==============================================================================
# Order Execution Manager (Main Engine)
# ==============================================================================

class OrderExecutionManager:
    """
    Production-grade Order Execution Manager bridging ORB signals,
    dynamic risk sizing, server-side OCO brackets, and heartbeat safety.
    """

    def __init__(
        self,
        risk_machine: Optional[RiskStateMachine] = None,
        max_quote_staleness_seconds: float = 1.5,
        heartbeat_check_interval: float = 0.100  # Check every 100ms
    ):
        self.risk_machine = risk_machine or RiskStateMachine()
        self.max_quote_staleness_seconds = max_quote_staleness_seconds
        self.heartbeat_check_interval = heartbeat_check_interval

        # Order & Position tracking
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.bracket_orders: Dict[str, Dict[str, Order]] = {}  # parent_id -> {"SL": Order, "TP": Order}
        self.bracket_protected_qty: Dict[str, int] = {}       # parent_id -> protected_contracts

        # Adapter
        self.broker: BrokerAdapter = SimulatedBrokerAdapter(self)

        # Heartbeat and Watchdog state (Task 4)
        self.last_quote_timestamp: float = time.monotonic()
        self.last_heartbeat_timestamp: float = time.monotonic()
        self.is_halted: bool = False
        self.watchdog_task: Optional[asyncio.Task] = None
        self._running: bool = False

        # Callbacks
        self.on_state_change_callbacks: List[Callable[[Order, OrderState, OrderState], None]] = []
        self.on_kill_switch_callbacks: List[Callable[[str], None]] = []

    # --------------------------------------------------------------------------
    # Lifecycle & Heartbeat Watchdog (Task 4)
    # --------------------------------------------------------------------------

    async def start(self):
        """Starts the background safety watchdog."""
        self._running = True
        self.last_quote_timestamp = time.monotonic()
        self.last_heartbeat_timestamp = time.monotonic()
        self.watchdog_task = asyncio.create_task(self._heartbeat_watchdog_loop())
        logger.info(
            f"🚀 [OEM STARTED] Heartbeat Watchdog active. Max latency tolerance: {self.max_quote_staleness_seconds * 1000:.0f}ms"
        )

    async def stop(self):
        """Gracefully stops the watchdog."""
        self._running = False
        if self.watchdog_task and not self.watchdog_task.done():
            self.watchdog_task.cancel()
            try:
                await self.watchdog_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 [OEM STOPPED]")

    def record_quote(self, symbol: str, bid: float, ask: float, timestamp: Optional[float] = None):
        """Updates the quote heartbeat timestamp whenever market data arrives and checks Leg 2 dynamic profit-lock."""
        self.last_quote_timestamp = time.monotonic()

        # Task 2: Dynamic Profit-Lock on Leg 2 Runner (touch +2.0R unrealized -> ratchet SL to +1.0R)
        mid_price = (bid + ask) / 2.0
        for o in list(self.orders.values()):
            if (
                getattr(o, "_is_leg2_sl", False)
                and o.symbol == symbol
                and o.state == OrderState.WORKING
                and not getattr(o, "_ratcheted_to_1r", False)
            ):
                entry_p = getattr(o, "_entry_price", 0.0)
                risk_pts = getattr(o, "_risk_points", 0.0)
                is_l = getattr(o, "_side_is_long", True)
                if risk_pts > 0:
                    if is_l and (mid_price >= entry_p + (2.0 * risk_pts)):
                        new_sl = round(entry_p + (1.0 * risk_pts), 2)
                        o.stop_price = new_sl
                        setattr(o, "_ratcheted_to_1r", True)
                        logger.info(
                            f"🔒 [LEG 2 DYNAMIC PROFIT-LOCK] Price touched +2.0R ({mid_price:.2f})! "
                            f"Order {o.order_id} Stop Loss ratcheted to +1.0R: {new_sl:.2f}"
                        )
                    elif (not is_l) and (mid_price <= entry_p - (2.0 * risk_pts)):
                        new_sl = round(entry_p - (1.0 * risk_pts), 2)
                        o.stop_price = new_sl
                        setattr(o, "_ratcheted_to_1r", True)
                        logger.info(
                            f"🔒 [LEG 2 DYNAMIC PROFIT-LOCK] Price touched +2.0R ({mid_price:.2f})! "
                            f"Order {o.order_id} Stop Loss ratcheted to +1.0R: {new_sl:.2f}"
                        )

    def record_api_heartbeat(self):
        """Updates the broker connection heartbeat timestamp."""
        self.last_heartbeat_timestamp = time.monotonic()

    async def _heartbeat_watchdog_loop(self):
        """
        Continuous millisecond-level loop enforcing the Kill-Switch protocol.
        Triggers instant 'Flatten All and Halt' if quote staleness > 1.5 seconds.
        """
        while self._running:
            await asyncio.sleep(self.heartbeat_check_interval)
            if self.is_halted:
                continue

            now = time.monotonic()
            quote_delta = now - self.last_quote_timestamp

            # Check if quote delay exceeds threshold
            if quote_delta > self.max_quote_staleness_seconds:
                has_active_exposure = self._has_active_exposure()
                if has_active_exposure:
                    reason = (
                        f"CRITICAL LAG/DISCONNECT SPIKE: Last quote age is {quote_delta:.3f}s "
                        f"(threshold: {self.max_quote_staleness_seconds}s) with active exposure!"
                    )
                    logger.critical(f"🚨 {reason}")
                    await self.flatten_all_and_halt(reason)

    def _has_active_exposure(self) -> bool:
        """Determines if there are open positions or active working orders."""
        for pos in self.positions.values():
            if pos.net_contracts != 0:
                return True
        for ord in self.orders.values():
            if ord.state in (OrderState.WORKING, OrderState.PARTIALLY_FILLED, OrderState.PENDING):
                return True
        return False

    async def flatten_all_and_halt(self, reason: str):
        """
        Task 4: Instant Kill-Switch execution.
        1. Sets system halt lock to block new orders.
        2. Cancels all pending/working orders.
        3. Emits market flatten orders for all open positions.
        4. Broadcasts emergency alerts.
        """
        self.is_halted = True
        logger.critical(f"🚨 [KILL-SWITCH ENGAGED] Initiating Flatten All and Halt! Reason: {reason}")

        # 1. Cancel all active resting orders
        cancelled_orders = await self.broker.cancel_all_orders()
        logger.warning(f"❌ Cancelled {len(cancelled_orders)} working orders.")

        # 2. Flatten all open positions at market
        for symbol, pos in list(self.positions.items()):
            if pos.net_contracts != 0:
                logger.critical(
                    f"⚠️  Flattening {symbol}: Net Position = {pos.net_contracts} contracts @ Market"
                )
                await self.broker.flatten_position(symbol)
                pos.net_contracts = 0

        # 3. Notify callbacks
        for cb in self.on_kill_switch_callbacks:
            try:
                cb(reason)
            except Exception as e:
                logger.error(f"Error in kill-switch callback: {e}")

    # --------------------------------------------------------------------------
    # Task 2: Order Lifecycle Tracking & Signal Routing
    # --------------------------------------------------------------------------

    async def execute_orb_signal(
        self,
        signal: dict,
        account_type: AccountType,
        current_balance: float,
        current_eod_drawdown_threshold: float,
        override_symbol: Optional[str] = None
    ) -> Optional[Order]:
        """
        Executes an ORB strategy signal:
          1. Validates system state (checks halt/kill-switch).
          2. Queries Risk State Machine for exact contract size and instrument.
          3. Submits parent entry order.
          4. Registers in lifecycle tracker.
        """
        if self.is_halted:
            logger.error("🛑 [EXECUTION BLOCKED] System is HALTED by Kill-Switch.")
            return None

        action = signal.get("action")
        if action not in ("LONG", "SHORT"):
            return None

        entry_price = signal.get("entry_price")
        sizing_info = signal.get("sizing", {})
        stop_points = abs(entry_price - sizing_info.get("stop_loss", entry_price - 25.0))
        if stop_points <= 0:
            stop_points = 25.0

        # Step 2: Query Risk State Machine
        try:
            sizing_res: SizingResult = self.risk_machine.calculate_sizing(
                account_type=account_type,
                current_balance=current_balance,
                current_eod_drawdown_threshold=current_eod_drawdown_threshold,
                stop_loss_points=stop_points
            )
        except Exception as e:
            logger.error(f"❌ Sizing calculation failed: {e}")
            return None

        contracts = sizing_res.contracts
        if contracts <= 0:
            logger.warning("⚠️  Calculated position size is 0 contracts. Trade aborted.")
            return None

        symbol = override_symbol or sizing_res.instrument
        side = OrderSide.BUY if action == "LONG" else OrderSide.SELL

        client_order_id = f"ORB-{action}-{uuid.uuid4().hex[:8]}"
        order_id = f"ORD-{uuid.uuid4().hex[:8]}"

        order = Order(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=contracts,
            price=entry_price,
            state=OrderState.PENDING
        )

        self.orders[order_id] = order
        self.bracket_protected_qty[order_id] = 0

        logger.info(
            f"📥 [ORB ENTRY DISPATCH] {action} {contracts}x {symbol} | "
            f"Risk Tier: {sizing_res.tier.value} | Risk $: ${sizing_res.dollar_risk:.2f} | "
            f"ClientOrderID: {client_order_id}"
        )

        # Store strategy context on order for bracket computation
        setattr(order, "_stop_points", stop_points)
        setattr(order, "_t1_points", stop_points * 1.2)  # +1.2 R:R Leg 1
        setattr(order, "_t2_points", stop_points * 2.5)  # +2.5 R:R Leg 2 Runner

        # Submit order to broker
        await self.broker.submit_order(order)
        return order

    async def on_order_update(
        self,
        order_id: str,
        new_state: OrderState,
        filled_qty: int,
        avg_price: float,
        rejection_reason: Optional[str] = None
    ):
        """
        Main API callback receiver for order state changes:
          - Transitions order states
          - Updates filled/remaining quantities
          - Adjusts real-time position
          - Triggers Two-Tier Bracket Order logic upon fill / partial fill
          - Triggers Ratchet to Entry +/- 2 ticks for Leg 2 once Leg 1 TP fills
        """
        if order_id not in self.orders:
            logger.warning(f"Received update for untracked order {order_id}")
            return

        order = self.orders[order_id]
        old_state = order.state
        delta_fill = filled_qty - order.filled_quantity

        order.state = new_state
        order.filled_quantity = filled_qty
        order.remaining_quantity = max(0, order.quantity - filled_qty)
        order.updated_at = time.time()
        if avg_price > 0:
            order.average_fill_price = avg_price
        if rejection_reason:
            order.rejection_reason = rejection_reason

        logger.info(
            f"📊 [ORDER UPDATE] {order_id} ({order.symbol} {order.side.value}) -> {new_state.value} | "
            f"Filled: {filled_qty}/{order.quantity} | Avg: {avg_price:.2f}"
        )

        # Update Net Position
        if delta_fill > 0 and not order.is_bracket:
            pos = self.positions.setdefault(order.symbol, Position(symbol=order.symbol))
            signed_contracts = delta_fill if order.side == OrderSide.BUY else -delta_fill
            pos.net_contracts += signed_contracts
            pos.average_entry_price = avg_price
            logger.info(f"💼 [POSITION UPDATE] {order.symbol}: Net Contracts = {pos.net_contracts}")

        # Check if Leg 1 TP filled to ratchet Leg 2 SL to Entry +/- 2 ticks
        if getattr(order, "_is_leg1_tp", False) and new_state == OrderState.FILLED:
            logger.info("🎯 [TWO-TIER RATCHET TRIGGER] Leg 1 TP (+1.2R) filled! Moving Leg 2 Stop to Entry +/- 2 ticks.")
            for o in list(self.orders.values()):
                if getattr(o, "_is_leg2_sl", False) and o.parent_order_id == order.parent_order_id and o.state == OrderState.WORKING:
                    entry_p = getattr(o, "_entry_price", 0.0)
                    is_l = getattr(o, "_side_is_long", True)
                    new_sl = round(entry_p + 0.50 if is_l else entry_p - 0.50, 2)
                    o.stop_price = new_sl
                    logger.info(f"🛡️  [LEG 2 SL RATCHETED] Order {o.order_id} Stop Loss ratcheted to BE+2 ticks: {new_sl:.2f}")

        # Notify state change listeners
        for cb in self.on_state_change_callbacks:
            try:
                cb(order, old_state, new_state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")

        # Task 3: Trigger Bracket Order placement if entry fills or partially fills
        if not order.is_bracket and new_state in (OrderState.PARTIALLY_FILLED, OrderState.FILLED):
            await self._handle_bracket_routing_on_fill(order, delta_fill, avg_price)

    # --------------------------------------------------------------------------
    # Task 3: Two-Tier Server-side Resting Bracket Orders (OCO) & Partial Fill Sizing
    # --------------------------------------------------------------------------

    async def _handle_bracket_routing_on_fill(self, parent_order: Order, delta_fill: int, fill_price: float):
        """
        Submits Two-Tier server-side resting OCO brackets:
          - Leg 1 (50% position): Take Profit TP1 set at +1.2R.
          - Leg 2 (50% position): Take Profit TP2 set at +2.5R runner.
        """
        if delta_fill <= 0:
            return

        stop_points = getattr(parent_order, "_stop_points", 25.0)
        tp1_points = getattr(parent_order, "_t1_points", stop_points * 1.2)
        tp2_points = getattr(parent_order, "_t2_points", stop_points * 2.5)

        is_long = (parent_order.side == OrderSide.BUY)

        # Mathematical bracket pricing
        if is_long:
            stop_side = OrderSide.SELL
            target_side = OrderSide.SELL
            stop_price = fill_price - stop_points
            tp1_price = fill_price + tp1_points
            tp2_price = fill_price + tp2_points
        else:
            stop_side = OrderSide.BUY
            target_side = OrderSide.BUY
            stop_price = fill_price + stop_points
            tp1_price = fill_price - tp1_points
            tp2_price = fill_price - tp2_points

        # Clean 50/50 two-tier split
        leg1_qty = delta_fill // 2 if delta_fill >= 2 else 1
        leg2_qty = delta_fill - leg1_qty

        # 1. Submit Leg 1 OCO bracket (TP1 @ +1.2R)
        oco_group_1 = f"OCO-L1-{uuid.uuid4().hex[:8]}"
        sl_order_1, tp_order_1 = await self.broker.submit_server_oco_bracket(
            symbol=parent_order.symbol,
            quantity=leg1_qty,
            stop_side=stop_side,
            stop_price=round(stop_price, 2),
            target_side=target_side,
            target_price=round(tp1_price, 2),
            oco_group_id=oco_group_1,
            parent_order_id=parent_order.order_id
        )
        setattr(tp_order_1, "_is_leg1_tp", True)

        # 2. Submit Leg 2 OCO bracket (TP2 @ +2.5R runner)
        sl_order_2 = None
        tp_order_2 = None
        if leg2_qty > 0:
            oco_group_2 = f"OCO-L2-{uuid.uuid4().hex[:8]}"
            sl_order_2, tp_order_2 = await self.broker.submit_server_oco_bracket(
                symbol=parent_order.symbol,
                quantity=leg2_qty,
                stop_side=stop_side,
                stop_price=round(stop_price, 2),
                target_side=target_side,
                target_price=round(tp2_price, 2),
                oco_group_id=oco_group_2,
                parent_order_id=parent_order.order_id
            )
            setattr(tp_order_2, "_is_leg2_tp", True)
            setattr(sl_order_2, "_is_leg2_sl", True)
            setattr(sl_order_2, "_entry_price", fill_price)
            setattr(sl_order_2, "_risk_points", stop_points)
            setattr(sl_order_2, "_side_is_long", is_long)
            setattr(sl_order_2, "_ratcheted_to_1r", False)

        # Track bracket orders
        self.orders[sl_order_1.order_id] = sl_order_1
        self.orders[tp_order_1.order_id] = tp_order_1
        if sl_order_2 and tp_order_2:
            self.orders[sl_order_2.order_id] = sl_order_2
            self.orders[tp_order_2.order_id] = tp_order_2

        self.bracket_orders[parent_order.order_id] = {
            "L1_SL": sl_order_1, "L1_TP": tp_order_1,
            "L2_SL": sl_order_2, "L2_TP": tp_order_2
        }
        self.bracket_protected_qty[parent_order.order_id] = (
            self.bracket_protected_qty.get(parent_order.order_id, 0) + delta_fill
        )

        logger.info(
            f"🔒 [TWO-TIER BRACKETS PLACED] Total {self.bracket_protected_qty[parent_order.order_id]}/"
            f"{parent_order.quantity} contracts enclosed: Leg 1 ({leg1_qty}x @ {tp1_price:.2f}) & "
            f"Leg 2 ({leg2_qty}x @ {tp2_price:.2f})."
        )


# ==============================================================================
# Comprehensive Verification Harness
# ==============================================================================

async def run_oem_verification():
    print("=" * 85)
    print("ORDER EXECUTION MANAGER (OEM) - VERIFICATION SUITE")
    print("=" * 85)

    oem = OrderExecutionManager(max_quote_staleness_seconds=1.5)
    await oem.start()

    # Verify Task 2 & Task 3: Signal routing, dynamic sizing, and partial fill bracket protection
    print("\n--- TEST 1: Full Order Flow with Partial Fill & OCO Brackets ---")
    mock_orb_signal = {
        "action": "LONG",
        "entry_price": 20450.00,
        "sizing": {"stop_loss": 20425.00}  # 25 points stop
    }

    # Balance $50,000, EOD Barrier $48,000 (Eval mode -> 8.5% of $2,000 = $170 risk -> 2 MNQ even split)
    parent_order = await oem.execute_orb_signal(
        signal=mock_orb_signal,
        account_type=AccountType.EVALUATION,
        current_balance=50000.0,
        current_eod_drawdown_threshold=48000.0
    )

    assert parent_order is not None
    assert parent_order.quantity == 2
    assert parent_order.symbol == "MNQ"
    print(f"Verified Entry Order: {parent_order.order_id} ({parent_order.quantity} {parent_order.symbol})")

    sim: SimulatedBrokerAdapter = oem.broker  # type: ignore

    # Simulate Partial Fill of 1 contract first (Leg 1)
    print("\nSimulating Partial Fill (1 of 2 contracts)...")
    await sim.simulate_fill(parent_order.order_id, fill_qty=1, fill_price=20450.00)
    assert oem.bracket_protected_qty[parent_order.order_id] == 1
    assert oem.positions["MNQ"].net_contracts == 1

    # Simulate Final Fill of remaining 1 contract (Leg 2)
    print("\nSimulating Final Fill (Remaining 1 contract)...")
    await sim.simulate_fill(parent_order.order_id, fill_qty=1, fill_price=20450.25)
    assert oem.bracket_protected_qty[parent_order.order_id] == 2
    assert oem.positions["MNQ"].net_contracts == 2
    print("[PASS] Two-Tier partial fill handling successfully enclosed both contracts in resting OCO brackets.")

    # Verify Task 4: Heartbeat & Kill-Switch Watchdog
    print("\n--- TEST 2: Millisecond Heartbeat & Kill-Switch Protocol ---")
    print("Simulating market data silence exceeding 1.5 seconds...")
    # Force staleness by setting last_quote_timestamp in the past
    oem.last_quote_timestamp = time.monotonic() - 2.0  # 2.0s ago

    # Give watchdog loop time to detect and execute kill-switch
    await asyncio.sleep(0.25)

    assert oem.is_halted is True
    assert oem.positions["MNQ"].net_contracts == 0
    print("[PASS] Kill-Switch engaged: Order Execution Manager halted and positions flattened at Market.")

    # Verify that new orders are blocked when halted
    blocked_order = await oem.execute_orb_signal(
        signal=mock_orb_signal,
        account_type=AccountType.EVALUATION,
        current_balance=50000.0,
        current_eod_drawdown_threshold=48000.0
    )
    assert blocked_order is None
    print("[PASS] Verified: All incoming orders strictly rejected while in HALT state.")

    await oem.stop()
    print("\n" + "=" * 85)
    print("ALL OEM TESTS COMPLETED SUCCESSFULLY WITH ZERO ERRORS.")
    print("=" * 85)


if __name__ == "__main__":
    asyncio.run(run_oem_verification())
