from __future__ import annotations

import pandas as pd

from .events import FillEvent, OrderEvent, SignalEvent


class Portfolio:
    """Single instrument, long-only, fractional units.

    Sizing rule (kept simple and documented): a move to weight w at price p buys/sells
    units so that units * p = w * equity_before_trade. Fees are then deducted from cash,
    so a fully invested account can carry a tiny negative cash balance equal to the fees.
    """

    def __init__(self, initial_cash: float = 1_000_000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.units = 0.0
        self.current_weight = 0.0
        self.fills: list[FillEvent] = []
        self._equity: list[tuple[pd.Timestamp, float]] = []
        self.total_fees = 0.0
        self.total_impact = 0.0   # (fill_price - ref_price) * qty, signed so it's a cost
        self.traded_notional = 0.0

    def on_signal(self, sig: SignalEvent) -> OrderEvent | None:
        if abs(sig.target_weight - self.current_weight) < 1e-12:
            return None
        self.current_weight = sig.target_weight
        return OrderEvent(sig.ts, sig.target_weight)

    def equity_at(self, price: float) -> float:
        return self.cash + self.units * price

    def on_fill(self, fill: FillEvent) -> None:
        signed = fill.quantity if fill.side == "BUY" else -fill.quantity
        self.units += signed
        self.cash -= signed * fill.fill_price + fill.fees
        self.total_fees += fill.fees
        self.total_impact += abs(fill.fill_price - fill.ref_price) * fill.quantity
        self.traded_notional += fill.quantity * fill.ref_price
        self.fills.append(fill)

    def mark(self, ts: pd.Timestamp, close: float) -> None:
        self._equity.append((ts, self.equity_at(close)))

    @property
    def equity_curve(self) -> pd.Series:
        idx, vals = zip(*self._equity) if self._equity else ([], [])
        return pd.Series(vals, index=pd.DatetimeIndex(idx), name="equity")
