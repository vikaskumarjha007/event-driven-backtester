from __future__ import annotations

import pandas as pd

from .costs import CostModel
from .events import FillEvent, OrderEvent
from .portfolio import Portfolio


class ExecutionHandler:
    """Simulated broker. Orders placed at bar t's close are filled at bar t+1's OPEN.
    Full fill, no partial fills, no market impact beyond fixed bps."""

    def __init__(self, cost_model: CostModel):
        self.costs = cost_model
        self._pending: list[OrderEvent] = []

    def submit(self, order: OrderEvent) -> None:
        self._pending.append(order)

    def fill_pending(self, ts: pd.Timestamp, open_price: float, pf: Portfolio) -> list[FillEvent]:
        fills = []
        for order in self._pending:
            equity = pf.equity_at(open_price)
            target_units = order.target_weight * equity / open_price
            delta = target_units - pf.units
            if abs(delta) < 1e-12:
                continue
            side = "BUY" if delta > 0 else "SELL"
            qty = abs(delta)
            px = self.costs.fill_price(open_price, side)
            # Re-size buys so notional at the (worse) fill price matches the target.
            if side == "BUY":
                qty = (order.target_weight * equity - pf.units * open_price) / px
            fees = self.costs.fees(qty * px, side)
            fills.append(FillEvent(ts, side, qty, open_price, px, fees))
        self._pending.clear()
        return fills
