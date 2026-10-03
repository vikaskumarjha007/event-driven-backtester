"""make demo: MA crossover gross vs net of realistic Indian costs vs buy-and-hold.

Runs on synthetic data by default. Pass --csv path/to/bars.csv (and --ts-col) to use real bars.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.loaders import load_real, load_synthetic  # noqa: E402
from edbt import (BuyAndHold, CostModel, IndianEquityCostModel,  # noqa: E402
                  MovingAverageCrossover, run_backtest, summary)

OUT = Path("outputs")
COLORS = {"Buy & hold": "#2a78d6", "MA crossover (gross)": "#1baf7a", "MA crossover (net)": "#eb6834"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--ts-col", default="date")
    ap.add_argument("--fast", type=int, default=30)
    ap.add_argument("--slow", type=int, default=120)
    ap.add_argument("--days", type=int, default=60)
    args = ap.parse_args()

    bars = load_real(args.csv, ts_col=args.ts_col) if args.csv else load_synthetic(seed=7, days=args.days)
    source = Path(args.csv).name if args.csv else f"synthetic GBM, {args.days} days of 1-min bars (seed 7)"

    net_costs = IndianEquityCostModel(slippage_bps=2.0, half_spread_bps=1.0)
    runs = {
        "Buy & hold": run_backtest(bars, BuyAndHold(), net_costs),
        "MA crossover (gross)": run_backtest(bars, MovingAverageCrossover(args.fast, args.slow), CostModel()),
        "MA crossover (net)": run_backtest(bars, MovingAverageCrossover(args.fast, args.slow), net_costs),
    }
    gross_eq = runs["MA crossover (gross)"].equity
    stats = {k: summary(v, gross_equity=gross_eq if k == "MA crossover (net)" else None) for k, v in runs.items()}

    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    for name, res in runs.items():
        curve = res.equity / res.portfolio.initial_cash * 100
        x = range(len(curve))
        ax.plot(x, curve.values, lw=2, color=COLORS[name], label=name)
        ax.annotate(f"{name}  {curve.iloc[-1]:.1f}", (len(curve) - 1, curve.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color="#333333")
    ax.axhline(100, color="#999999", lw=0.8, ls="--")
    ax.set_ylabel("Equity (start = 100)")
    ax.set_xlabel("Bar number (1-minute bars)")
    ax.set_title(f"MA({args.fast}/{args.slow}) crossover: gross vs net of costs\n{source}",
                 loc="left", fontsize=11)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.set_xlim(0, len(bars) * 1.22)
    fig.tight_layout()
    fig.savefig(OUT / "equity_curves.png")

    rows = ["| Run | Total return | Sharpe | Max DD | Trades | Turnover (x) | Fees paid | Slippage+spread paid |",
            "|---|---|---|---|---|---|---|---|"]
    for k, s in stats.items():
        rows.append(f"| {k} | {s['total_return']:+.2%} | {s['sharpe']:.2f} | {s['max_drawdown']:.2%} | "
                    f"{s['trades']} | {s['turnover']:.1f} | {s['fees_paid']:,.0f} | {s['impact_paid']:,.0f} |")
    drag = stats["MA crossover (net)"]["cost_drag"]
    rows.append(f"\nCost drag on MA crossover (gross minus net total return): {drag:.2%}. Start capital 1,000,000.")
    (OUT / "metrics.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"\nWrote {OUT/'equity_curves.png'} and {OUT/'metrics.md'}")


if __name__ == "__main__":
    main()
