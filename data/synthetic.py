"""Seeded synthetic 1-minute OHLCV bars.

Assumptions (deliberately simple, stated so nobody mistakes this for real data):
  * Geometric Brownian motion on 1-min close-to-close returns, annual drift mu, vol sigma.
  * Session 09:15-15:29 IST (375 one-minute bars, NSE cash hours), weekdays only, no holidays.
  * Overnight gap: an extra GBM shock scaled to `overnight_minutes` of variance.
  * open = previous close (+ gap at session start); high/low = max/min(open, close) widened by
    a small random fraction; volume lognormal. No intraday seasonality, no jumps, no fat tails.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BARS_PER_DAY = 375


def make_bars(days: int = 60, seed: int = 7, start_price: float = 20_000.0, mu: float = 0.10,
              sigma: float = 0.15, start: str = "2026-01-05", overnight_minutes: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=days)
    minutes = pd.timedelta_range("09:15:00", periods=BARS_PER_DAY, freq="1min")
    index = pd.DatetimeIndex([d + m for d in dates for m in minutes], name="ts")

    dt = 1 / (252 * BARS_PER_DAY)
    n = len(index)
    shocks = rng.standard_normal(n)
    var_scale = np.ones(n)
    var_scale[::BARS_PER_DAY] += overnight_minutes  # bigger variance at the open
    log_r = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt * var_scale) * shocks
    log_r[0] = 0.0

    close = start_price * np.exp(np.cumsum(log_r))
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    # Session opens carry the overnight gap: split the first bar's move into gap + intraday.
    first = np.arange(BARS_PER_DAY, n, BARS_PER_DAY)
    gap_share = overnight_minutes / (overnight_minutes + 1)
    open_[first] = close[first - 1] * np.exp(log_r[first] * gap_share)

    wiggle = np.abs(rng.normal(0, sigma * np.sqrt(dt) * 0.5, n))
    high = np.maximum(open_, close) * (1 + wiggle)
    low = np.minimum(open_, close) * (1 - wiggle)
    volume = rng.lognormal(mean=8, sigma=0.5, size=n).round()

    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume},
                        index=index)
