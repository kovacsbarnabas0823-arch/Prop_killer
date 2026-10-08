"""
5-Minute ORB Live Execution Suite
Contains:
  - OrderExecutionManager5m: Async live execution engine with 0.5R trailing ratchets & watchdog
  - BrokerAdapter, SimulatedBrokerAdapter
"""

from .order_execution_manager_5m import (
    OrderExecutionManager5m,
    Order,
    OrderState,
    OrderSide,
    OrderType,
    Position,
    BrokerAdapter,
    SimulatedBrokerAdapter,
    RealTimeEMA
)

__all__ = [
    "OrderExecutionManager5m",
    "Order",
    "OrderState",
    "OrderSide",
    "OrderType",
    "Position",
    "BrokerAdapter",
    "SimulatedBrokerAdapter",
    "RealTimeEMA"
]
