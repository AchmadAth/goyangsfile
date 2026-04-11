"""
Batch tester — runs weekly swing backtest on 6 synthetic tickers:
  IDX : BBCA, BBRI, ANTM
  US  : NVDA, GOOGL, AAPL

Yahoo Finance is unreachable in this environment, so each ticker is
simulated with realistic synthetic OHLCV data (trend, volatility,
volume profile, and periodic corrections calibrated per ticker).
"""

import numpy as np
import pandas as pd
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))

from backtest_weekly import (
    Config, run_backtest, compute_stats, print_report, print_trades
)

np.random.seed(2024)

# ──────────────────────────────────────────
#  SYNTHETIC DATA GENERATOR
# ──────────────────────────────────────────
def gen_ohlcv(
    n: int,
    start_price: float,
    trend_per_day: float,
    daily_vol: float,
    base_volume: int,
    crash_prob: float = 0.003,
    crash_size: float = 0.12,
    recovery_speed: float = 0.0004,
    seed_offset: int = 0,
) -> pd.DataFrame:
    """
    Simulate n business days of OHLCV data.
    Includes occasional crashes (for realistic drawdowns) and recovery.
    """
    rng = np.random.default_rng(2024 + seed_offset)
    dates  = pd.bdate_range("2019-01-02", periods=n)
    log_r  = rng.normal(trend_per_day, daily_vol, n)

    # Inject occasional crashes
    for i in range(n):
        if rng.random() < crash_prob:
            # Multi-day crash
            length = rng.integers(5, 15)
            for j in range(length):
                if i + j < n:
                    log_r[i + j] -= crash_size / length
            # Recovery after crash
            for j in range(length, length * 3):
                if i + j < n:
                    log_r[i + j] += recovery_speed

    close = start_price * np.exp(np.cumsum(log_r))

    # OHLC from close
    range_pct = rng.uniform(0.004, 0.022, n)
    spread    = close * range_pct
    high  = close + spread * rng.uniform(0.3, 1.0, n)
    low   = close - spread * rng.uniform(0.3, 1.0, n)
    open_ = low + (high - low) * rng.uniform(0.1, 0.9, n)

    # Volume: base + noise + spikes on big moves
    vol = rng.lognormal(np.log(base_volume), 0.5, n).astype(int)
    big_days = np.where(np.abs(log_r) > daily_vol * 1.8)[0]
    vol[big_days] = (vol[big_days] * rng.uniform(1.4, 2.8, len(big_days))).astype(int)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


def gen_index(n: int, start: float, trend: float, vol: float, seed: int) -> pd.DataFrame:
    return gen_ohlcv(n, start, trend, vol, base_volume=50_000_000,
                     crash_prob=0.002, crash_size=0.08, seed_offset=seed)


# ──────────────────────────────────────────
#  TICKER PROFILES
# ──────────────────────────────────────────
# Each entry: (display_name, market, start_price, trend/day, daily_vol, base_vol, seed)
TICKERS = [
    # IDX stocks
    ("BBCA",  "IDX", 6_000,   0.00035, 0.010, 20_000_000, 10),   # big bank, stable
    ("BBRI",  "IDX", 4_200,   0.00030, 0.013, 30_000_000, 20),   # big bank, slightly more volatile
    ("ANTM",  "IDX",   800,   0.00010, 0.028, 50_000_000, 30),   # mining, cyclical + volatile
    # US stocks
    ("NVDA",  "US",  18.0,   0.00090, 0.028, 60_000_000, 40),   # strong trend, high vol
    ("GOOGL", "US", 1050.0,  0.00045, 0.016, 25_000_000, 50),   # steady compounder
    ("AAPL",  "US",  150.0,  0.00040, 0.014, 80_000_000, 60),   # large cap, lower vol
]

# Matching index profiles
IDX_INDEX  = dict(start=6000,  trend=0.00020, vol=0.009, seed=1)
US_INDEX   = dict(start=2700,  trend=0.00035, vol=0.010, seed=2)

N_DAYS = 1500   # ~6 years


# ──────────────────────────────────────────
#  RUN ALL
# ──────────────────────────────────────────
def main():
    # Pre-generate indices (shared per market)
    idx_data = gen_index(N_DAYS, **IDX_INDEX)
    us_data  = gen_index(N_DAYS, **US_INDEX)

    summary = []

    for (name, market, start_px, trend, vol, base_vol, seed) in TICKERS:
        stock = gen_ohlcv(
            N_DAYS, start_px, trend, vol, base_vol,
            crash_prob=0.003 if market == "IDX" else 0.002,
            crash_size=0.14  if market == "IDX" else 0.10,
            seed_offset=seed,
        )
        index = idx_data if market == "IDX" else us_data

        cfg = Config(initial_capital=100_000.0)
        if market == "IDX":
            cfg.commission_pct = 0.004
            cfg.stop_max_pct   = 0.06

        trades, equity = run_backtest(stock, index, cfg)
        stats = compute_stats(trades, equity, cfg)

        period = "2019-01-02 → 2024-12-31 (synthetic)"
        print_report(stats, f"{name} [{market}]", period)
        if trades:
            print_trades(trades, n=8)

        summary.append({
            "ticker": name, "market": market,
            "trades":      stats.get("total_trades", 0),
            "win_rate":    f"{stats.get('win_rate', 0)*100:.0f}%",
            "pf":          stats.get("profit_factor", 0),
            "return_pct":  stats.get("total_return_pct", 0),
            "sharpe":      stats.get("sharpe_ratio", 0),
            "max_dd":      stats.get("max_drawdown_pct", 0),
            "avg_hold":    stats.get("avg_hold_days", 0),
            "expectancy":  stats.get("expectancy", 0),
            "final_cap":   stats.get("final_capital", 100_000),
        })

        # Save per-ticker JSON
        path = f"/home/claude/swing_backtest/result_{name}_{market}.json"
        with open(path, "w") as f:
            json.dump({"ticker": name, "market": market, "stats": stats,
                       "trades": [{"entry_date": str(t.entry_date.date()),
                                   "exit_date":  str(t.exit_date.date()),
                                   "hold_days":  t.hold_days,
                                   "entry_price": round(t.entry_price, 4),
                                   "exit_price":  round(t.exit_price, 4),
                                   "pnl_pct":     round(t.pnl_pct * 100, 2),
                                   "pnl_usd":     round(t.pnl, 2),
                                   "exit_reason": t.exit_reason}
                                  for t in trades]}, f, indent=2)

    # ── SUMMARY TABLE ──────────────────────────────────
    print("\n" + "═" * 90)
    print("  SUMMARY — ALL TICKERS")
    print("═" * 90)
    hdr = f"  {'Ticker':<8} {'Mkt':<5} {'Trades':>6} {'WinR':>5} {'PF':>6} {'Return':>8} "
    hdr += f"{'Sharpe':>7} {'MaxDD':>7} {'AvgHold':>8} {'Expect':>9} {'FinalCap':>12}"
    print(hdr)
    print("  " + "─" * 86)
    for r in summary:
        flag = "✅" if r["pf"] >= 1.5 and r["return_pct"] > 0 else "⚠ "
        print(
            f"  {flag}{r['ticker']:<7} {r['market']:<5} "
            f"{r['trades']:>6} {r['win_rate']:>5} {r['pf']:>6.2f} "
            f"{r['return_pct']:>7.1f}% {r['sharpe']:>7.2f} "
            f"{r['max_dd']:>6.1f}% {r['avg_hold']:>7.1f}d "
            f"${r['expectancy']:>8,.0f} ${r['final_cap']:>11,.0f}"
        )
    print("═" * 90)
    print("  NOTE: All results use SYNTHETIC data — run with real Yahoo Finance data locally.")
    print("═" * 90)


if __name__ == "__main__":
    main()
