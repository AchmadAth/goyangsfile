"""
Demo runner using synthetic OHLCV data (no internet required).
Simulates a trending bull market with realistic noise and volume spikes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from backtest import Config, run_backtest, compute_stats, print_report, print_trades

np.random.seed(42)

def generate_ohlcv(n: int = 1500, start_price: float = 100.0,
                   trend: float = 0.0003, vol: float = 0.015) -> pd.DataFrame:
    """Synthetic stock with a gentle upward trend + noise + realistic OHLCV."""
    dates = pd.bdate_range("2019-01-01", periods=n)
    log_returns = np.random.normal(trend, vol, n)
    close = start_price * np.exp(np.cumsum(log_returns))

    daily_range = close * np.random.uniform(0.005, 0.025, n)
    high  = close + daily_range * np.random.uniform(0.3, 1.0, n)
    low   = close - daily_range * np.random.uniform(0.3, 1.0, n)
    open_ = low + (high - low) * np.random.uniform(0, 1, n)

    base_vol = 5_000_000
    # Occasional volume spikes (simulate breakout days)
    volume = np.random.lognormal(np.log(base_vol), 0.4, n)
    spike_days = np.random.choice(n, size=int(n * 0.05), replace=False)
    volume[spike_days] *= np.random.uniform(1.5, 3.5, len(spike_days))

    return pd.DataFrame({
        "Open":   open_,
        "High":   high,
        "Low":    low,
        "Close":  close,
        "Volume": volume.astype(int),
    }, index=dates)


def generate_index(n: int = 1500, trend: float = 0.0002, vol: float = 0.010) -> pd.DataFrame:
    """Synthetic broad market index (similar but less volatile)."""
    dates = pd.bdate_range("2019-01-01", periods=n)
    log_returns = np.random.normal(trend, vol, n)
    close = 3000.0 * np.exp(np.cumsum(log_returns))
    return pd.DataFrame({
        "Open":   close * 0.999,
        "High":   close * 1.005,
        "Low":    close * 0.995,
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=dates)


def main():
    cfg = Config(initial_capital=100_000.0)
    ticker = "SYNTH_AAPL"
    start  = "2019-01-01"
    end    = "2024-12-31"

    print(f"\n[Swing Bot Backtest — SYNTHETIC DATA MODE]")
    print(f"Ticker : {ticker}  |  Period: {start} → {end}")
    print(f"Capital: ${cfg.initial_capital:,.0f}  |  Rules: 20-rule AND-gate\n")

    stock = generate_ohlcv(n=1500)
    index = generate_index(n=1500)

    print("  Running 20-rule backtest...")
    trades, equity = run_backtest(stock, index, cfg, ticker)
    stats = compute_stats(trades, equity, cfg)

    print_report(stats, ticker, start, end)
    print_trades(trades, n=15)

    # Save JSON
    import json
    out = {
        "ticker": ticker,
        "market": "US (synthetic)",
        "period": f"{start} to {end}",
        "stats": stats,
        "trades": [
            {
                "entry_date":  str(t.entry_date.date()),
                "exit_date":   str(t.exit_date.date()) if t.exit_date else None,
                "entry_price": round(t.entry_price, 4),
                "exit_price":  round(t.exit_price, 4) if t.exit_price else None,
                "pnl_pct":     round(t.pnl_pct * 100, 2),
                "pnl_usd":     round(t.pnl, 2),
                "exit_reason": t.exit_reason,
            }
            for t in trades
        ],
    }
    outfile = "/home/claude/swing_backtest/result_SYNTH_US.json"
    with open(outfile, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved → {outfile}")


if __name__ == "__main__":
    main()
