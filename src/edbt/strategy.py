from __future__ import annotations

from abc import ABC, abstractmethod

from .data_handler import BarView


class Strategy(ABC):
    """Sees only the past (BarView). Returns a target weight in [0, 1] or None (no change)."""

    @abstractmethod
    def on_bar(self, view: BarView) -> float | None: ...


class BuyAndHold(Strategy):
    def __init__(self):
        self._done = False

    def on_bar(self, view: BarView) -> float | None:
        if self._done:
            return None
        self._done = True
        return 1.0


class MovingAverageCrossover(Strategy):
    """Long when fast SMA of close > slow SMA, flat otherwise."""

    def __init__(self, fast: int = 20, slow: int = 60):
        if fast >= slow:
            raise ValueError("fast must be < slow")
        self.fast, self.slow = fast, slow

    def on_bar(self, view: BarView) -> float | None:
        if len(view) < self.slow:
            return None
        closes = view.history("close", self.slow)
        return 1.0 if closes.iloc[-self.fast:].mean() > closes.mean() else 0.0
