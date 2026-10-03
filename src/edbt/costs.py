"""Transaction cost models.

Two pieces, kept separate on purpose:
  * price impact (slippage + half-spread) -> moves the fill price
  * explicit fees (brokerage, taxes)      -> charged in cash
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    half_spread_bps: float = 0.0

    def fill_price(self, ref_price: float, side: str) -> float:
        adj = (self.slippage_bps + self.half_spread_bps) / 1e4
        return ref_price * (1 + adj) if side == "BUY" else ref_price * (1 - adj)

    def fees(self, notional: float, side: str) -> float:
        return abs(notional) * self.commission_bps / 1e4


# NSE equity intraday, Zerodha schedule. Source: https://zerodha.com/charges (checked 2026-10-03).
# Rates change - re-check before relying on them.
NSE_EQUITY_INTRADAY = {
    "brokerage_rate": 0.0003,          # 0.03% ...
    "brokerage_cap": 20.0,             # ... or Rs 20 per executed order, whichever is lower
    "stt_sell_rate": 0.00025,          # 0.025% on sell side only
    "exchange_txn_rate": 0.0000307,    # NSE transaction charge 0.00307%
    "sebi_rate": 10 / 1e7,             # Rs 10 per crore
    "gst_rate": 0.18,                  # on brokerage + exchange txn + SEBI
    "stamp_buy_rate": 0.00003,         # 0.003% on buy side
}


@dataclass(frozen=True)
class IndianEquityCostModel(CostModel):
    """Explicit Indian statutory + broker fees on top of the bps slippage/spread model.
    commission_bps is ignored (brokerage comes from the schedule)."""
    schedule: tuple = tuple(sorted(NSE_EQUITY_INTRADAY.items()))

    def fees(self, notional: float, side: str) -> float:
        s = dict(self.schedule)
        n = abs(notional)
        brokerage = min(n * s["brokerage_rate"], s["brokerage_cap"])
        txn = n * s["exchange_txn_rate"]
        sebi = n * s["sebi_rate"]
        gst = s["gst_rate"] * (brokerage + txn + sebi)
        stt = n * s["stt_sell_rate"] if side == "SELL" else 0.0
        stamp = n * s["stamp_buy_rate"] if side == "BUY" else 0.0
        return brokerage + txn + sebi + gst + stt + stamp
