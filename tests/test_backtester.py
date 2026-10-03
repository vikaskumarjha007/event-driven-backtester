import numpy as np
import pandas as pd
import pytest

from data.loaders import load_sample, load_synthetic
from edbt import (BuyAndHold, CostModel, IndianEquityCostModel, LookaheadError,
                  MovingAverageCrossover, Strategy, run_backtest)


def five_bars() -> pd.DataFrame:
    idx = pd.date_range("2026-01-05 09:15", periods=5, freq="1min")
    o = [100.0, 100.0, 110.0, 121.0, 115.0]
    c = [100.0, 110.0, 121.0, 115.0, 120.0]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c),
                         "close": c, "volume": 1.0}, index=idx)


# ---------- 1. Known answer: buy-and-hold, zero cost == price return -----------------------
def test_buy_and_hold_zero_cost_equals_price_return():
    bars = load_synthetic(seed=1, days=5)
    res = run_backtest(bars, BuyAndHold())
    # Signal on bar 0's close -> filled at bar 1's open -> held to the last close.
    expected = bars["close"].iloc[-1] / bars["open"].iloc[1] - 1
    got = res.equity.iloc[-1] / res.portfolio.initial_cash - 1
    assert got == pytest.approx(expected, abs=1e-9)


# ---------- 2. No lookahead ---------------------------------------------------------------
class Cheater(Strategy):
    def on_bar(self, view):
        view.bar(+1)  # tries to read the next bar


def test_strategy_cannot_read_future_bar():
    with pytest.raises(LookaheadError):
        run_backtest(five_bars(), Cheater())


class Recorder(Strategy):
    def __init__(self):
        self.seen = []

    def on_bar(self, view):
        self.seen.append((view.now, len(view), view.history().index.max()))
        return None


def test_history_never_extends_past_now():
    rec = Recorder()
    bars = five_bars()
    run_backtest(bars, rec)
    for i, (now, n, last_ts) in enumerate(rec.seen):
        assert now == bars.index[i] and n == i + 1 and last_ts == now


def test_signal_is_filled_at_next_open_not_same_bar():
    res = run_backtest(five_bars(), BuyAndHold())
    fill = res.portfolio.fills[0]
    assert fill.ts == five_bars().index[1] and fill.ref_price == 100.0


# ---------- 3. Parity with a vectorised pandas implementation ------------------------------
def vectorised_ma(bars: pd.DataFrame, fast: int, slow: int, cash: float) -> pd.Series:
    c, o = bars["close"], bars["open"]
    f, s = c.rolling(fast).mean(), c.rolling(slow).mean()
    w = (f > s).astype(float).where(s.notna()).ffill().fillna(0.0)  # decided at close t
    held_overnight = w.shift(2).fillna(0.0)   # position held from close t-1 into open t
    held_intraday = w.shift(1).fillna(0.0)    # position after trading at open t
    gap = o / c.shift(1) - 1
    intra = c / o - 1
    growth = (1 + held_overnight * gap.fillna(0)) * (1 + held_intraday * intra)
    growth.iloc[0] = 1.0
    return cash * growth.cumprod()


def test_ma_crossover_matches_vectorised():
    bars = load_synthetic(seed=3, days=10)
    res = run_backtest(bars, MovingAverageCrossover(10, 40))
    ref = vectorised_ma(bars, 10, 40, res.portfolio.initial_cash)
    assert len(res.portfolio.fills) > 5  # make sure the test actually trades
    np.testing.assert_allclose(res.equity.values, ref.values, rtol=1e-9)


# ---------- 4. Costs reduce returns by the hand-computed amount ----------------------------
def test_costs_on_hand_built_five_bars():
    cm = CostModel(commission_bps=10, slippage_bps=10, half_spread_bps=0)
    res = run_backtest(five_bars(), BuyAndHold(), cm, initial_cash=1_000_000)
    # Hand calc: fill at 100 * 1.001 = 100.1; units = 1e6 / 100.1; fee = 1e6 * 0.001 = 1000
    units = 1_000_000 / 100.1
    expected_final = units * 120.0 - 1_000.0
    assert res.equity.iloc[-1] == pytest.approx(expected_final, rel=1e-12)
    gross = run_backtest(five_bars(), BuyAndHold()).equity.iloc[-1]   # 1.2m
    assert gross == pytest.approx(1_200_000.0)
    assert gross - res.equity.iloc[-1] == pytest.approx(1_200_000 - expected_final)


def test_round_trip_sell_side_costs():
    class InOut(Strategy):
        def on_bar(self, view):
            return {0: 1.0, 2: 0.0}.get(len(view) - 1)
    cm = CostModel(commission_bps=0, slippage_bps=0, half_spread_bps=50)
    res = run_backtest(five_bars(), InOut(), cm, initial_cash=1_000_000)
    units = 1_000_000 / 100.5            # bought at bar 1 open 100 * 1.005
    cash = units * 121.0 * 0.995         # sold at bar 3 open 121 * 0.995
    assert res.equity.iloc[-1] == pytest.approx(cash, rel=1e-12)
    assert res.portfolio.units == pytest.approx(0.0, abs=1e-9)


def test_indian_fee_schedule_known_answer():
    cm = IndianEquityCostModel()
    # Rs 1,00,000 buy: brokerage 20 (cap) + txn 3.07 + SEBI 0.10 + GST 18% of 23.17 + stamp 3
    assert cm.fees(100_000, "BUY") == pytest.approx(20 + 3.07 + 0.10 + 0.18 * 23.17 + 3.0)
    # Sell: same minus stamp plus STT 0.025% = 25
    assert cm.fees(100_000, "SELL") == pytest.approx(20 + 3.07 + 0.10 + 0.18 * 23.17 + 25.0)
    # Small order: brokerage is 0.03%, not the cap
    assert cm.fees(10_000, "BUY") == pytest.approx(3 + 0.307 + 0.01 + 0.18 * 3.317 + 0.3)


# ---------- data contract ------------------------------------------------------------------
def test_sample_and_synthetic_share_schema():
    a, b = load_sample(), load_synthetic(days=1)
    assert list(a.columns) == list(b.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(a.index, pd.DatetimeIndex) and a.index.is_monotonic_increasing


def test_synthetic_is_seeded():
    pd.testing.assert_frame_equal(load_synthetic(seed=5, days=2), load_synthetic(seed=5, days=2))
