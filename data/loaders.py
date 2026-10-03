"""Both loaders return the SAME schema: DatetimeIndex 'ts' (sorted, unique),
float columns open, high, low, close, volume. The engine never knows which one it got."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .synthetic import make_bars

SCHEMA = ["open", "high", "low", "close", "volume"]
SAMPLE = Path(__file__).parent / "sample" / "synthetic_1min_sample.csv"


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    df = df[SCHEMA].astype(float)
    df.index = pd.DatetimeIndex(df.index, name="ts")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    if df.isna().any().any():
        raise ValueError("NaNs in bars - clean the source first.")
    if not ((df["low"] <= df[["open", "close"]].min(axis=1)) & (df["high"] >= df[["open", "close"]].max(axis=1))).all():
        raise ValueError("OHLC inconsistency (low > open/close or high < open/close).")
    return df


def load_real(path: str | Path, ts_col: str = "date", rename: dict | None = None,
              tz: str | None = None) -> pd.DataFrame:
    """Load a CSV/Parquet of real bars. Use `rename` to map your columns to the schema,
    e.g. {"Open": "open", "Close": "close", ...}. Raw files belong in data/raw/ (gitignored)."""
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if rename:
        df = df.rename(columns=rename)
    df.columns = [c.lower() for c in df.columns]
    ts = pd.to_datetime(df.pop(ts_col.lower()))
    if tz and ts.dt.tz is not None:
        ts = ts.dt.tz_convert(tz).dt.tz_localize(None)
    df.index = ts
    if "volume" not in df:
        df["volume"] = 0.0  # index data often has no volume
    return _validate(df)


def load_synthetic(seed: int = 7, days: int = 60, **kwargs) -> pd.DataFrame:
    return _validate(make_bars(days=days, seed=seed, **kwargs))


def load_sample() -> pd.DataFrame:
    """Small committed sample so a clean clone runs with zero setup."""
    return _validate(pd.read_csv(SAMPLE, index_col="ts", parse_dates=True))
