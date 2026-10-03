"""Streams bars one at a time. Strategies only ever see a BarView of the past."""
from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


class LookaheadError(Exception):
    """Raised when code asks for a bar that has not happened yet."""


class BarView:
    """Read-only window onto bars[0..t]. There is no way to reach bar t+1."""

    def __init__(self, bars: pd.DataFrame, t: int):
        self._bars = bars
        self._t = t

    @property
    def now(self) -> pd.Timestamp:
        return self._bars.index[self._t]

    def bar(self, offset: int = 0) -> pd.Series:
        """offset=0 is the current (just-closed) bar, -1 the previous one, etc."""
        if offset > 0:
            raise LookaheadError(f"Requested bar offset +{offset} from {self.now}: future data.")
        idx = self._t + offset
        if idx < 0:
            raise IndexError("Not enough history.")
        return self._bars.iloc[idx]

    def history(self, column: str = "close", n: int | None = None) -> pd.Series:
        """Copy of the last n values up to and including the current bar."""
        s = self._bars[column].iloc[: self._t + 1]
        return (s if n is None else s.iloc[-n:]).copy()

    def __len__(self) -> int:
        return self._t + 1


class DataHandler:
    def __init__(self, bars: pd.DataFrame):
        missing = [c for c in REQUIRED_COLUMNS if c not in bars.columns]
        if missing:
            raise ValueError(f"Bars missing columns: {missing}")
        if not bars.index.is_monotonic_increasing:
            raise ValueError("Bars must be sorted by time.")
        self._bars = bars[REQUIRED_COLUMNS].astype(float)
        self._t = -1

    def has_next(self) -> bool:
        return self._t + 1 < len(self._bars)

    def advance(self) -> pd.Timestamp:
        self._t += 1
        return self._bars.index[self._t]

    def view(self) -> BarView:
        return BarView(self._bars, self._t)

    def current_bar(self) -> pd.Series:
        return self._bars.iloc[self._t]
