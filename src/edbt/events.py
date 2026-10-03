"""Event types flowing through the single event queue.

Flow per bar:  MarketEvent -> Strategy -> SignalEvent -> Portfolio -> OrderEvent
               -> ExecutionHandler (fills on the NEXT bar's open) -> FillEvent -> Portfolio
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MarketEvent:
    """A new bar has closed and is now visible."""
    ts: pd.Timestamp


@dataclass(frozen=True)
class SignalEvent:
    """Strategy's desired exposure, as a target weight of equity (long-only: 0..1)."""
    ts: pd.Timestamp
    target_weight: float


@dataclass(frozen=True)
class OrderEvent:
    """Instruction to move to target_weight. Sized at fill time (next open)."""
    ts: pd.Timestamp
    target_weight: float


@dataclass(frozen=True)
class FillEvent:
    ts: pd.Timestamp
    side: str          # "BUY" or "SELL"
    quantity: float    # units, always positive
    ref_price: float   # bar open before slippage/spread
    fill_price: float  # price after slippage + half-spread
    fees: float        # explicit fees (commission, taxes) in currency
