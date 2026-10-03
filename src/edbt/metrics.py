from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import BacktestResult


def infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    """Bars per year, estimated from the data itself (bars per day x 252 trading days)."""
    days = pd.Series(index.normalize()).nunique()
    return len(index) / max(days, 1) * 252


def summary(res: BacktestResult, periods_per_year: float | None = None,
            gross_equity: pd.Series | None = None) -> dict:
    eq = res.equity
    ppy = periods_per_year or infer_periods_per_year(eq.index)
    rets = eq.pct_change().dropna()
    years = len(eq) / ppy
    total = eq.iloc[-1] / res.portfolio.initial_cash
    dd = eq / eq.cummax() - 1
    sd = rets.std(ddof=1)
    out = {
        "total_return": total - 1,
        "cagr": total ** (1 / years) - 1 if years > 0 else np.nan,
        "sharpe": float(np.sqrt(ppy) * rets.mean() / sd) if sd > 0 else np.nan,
        "max_drawdown": float(dd.min()),
        "trades": len(res.portfolio.fills),
        "turnover": res.portfolio.traded_notional / res.portfolio.initial_cash,
        "fees_paid": res.portfolio.total_fees,
        "impact_paid": res.portfolio.total_impact,
    }
    if gross_equity is not None:
        out["cost_drag"] = gross_equity.iloc[-1] / gross_equity.iloc[0] - total
    return out
