"""The event loop. One queue, strict ordering, no peeking."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import pandas as pd

from .costs import CostModel
from .data_handler import DataHandler
from .events import FillEvent, MarketEvent, OrderEvent, SignalEvent
from .execution import ExecutionHandler
from .portfolio import Portfolio
from .strategy import Strategy


@dataclass
class BacktestResult:
    equity: pd.Series
    portfolio: Portfolio


def run_backtest(bars: pd.DataFrame, strategy: Strategy, cost_model: CostModel | None = None,
                 initial_cash: float = 1_000_000.0) -> BacktestResult:
    data = DataHandler(bars)
    pf = Portfolio(initial_cash)
    broker = ExecutionHandler(cost_model or CostModel())
    queue: deque = deque()

    while data.has_next():
        ts = data.advance()
        bar = data.current_bar()

        # 1) Orders from the previous close fill at this bar's open.
        for fill in broker.fill_pending(ts, bar["open"], pf):
            queue.append(fill)
        # 2) This bar has now closed and becomes visible.
        queue.append(MarketEvent(ts))

        while queue:
            ev = queue.popleft()
            if isinstance(ev, MarketEvent):
                w = strategy.on_bar(data.view())
                if w is not None:
                    queue.append(SignalEvent(ev.ts, float(w)))
            elif isinstance(ev, SignalEvent):
                order = pf.on_signal(ev)
                if order is not None:
                    queue.append(order)
            elif isinstance(ev, OrderEvent):
                broker.submit(ev)
            elif isinstance(ev, FillEvent):
                pf.on_fill(ev)

        pf.mark(ts, bar["close"])

    return BacktestResult(pf.equity_curve, pf)
