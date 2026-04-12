"""
Batch tester — weekly swing engine, optimised config
  target_rr    : 2.0 → 1.5   (more profit_target exits)
  time_stop    : 7  → 10 days (more room to breathe)
  stop_atr_mult: 1.5 → 2.5 for high-volatility tickers

16 tickers total:
  IDX (6): BBCA, BBRI, ANTM, TLKM, BMRI, ASII
  US  (10): NVDA, GOOGL, AAPL, MSFT, AMZN, META, TSLA, JPM, ADBE, AMD
"""

import numpy as np
import pandas as pd
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))

from backtest_weekly import Config, run_backtest, compute_stats, print_report, print_trades

np.random.seed(2024)


# ──────────────────────────────────────────
#  SYNTHETIC DATA
# ──────────────────────────────────────────
def gen_ohlcv(n, start_price, trend_per_day, daily_vol, base_volume,
              crash_prob=0.003, crash_size=0.12, recovery_speed=0.0004,
              seed_offset=0):
    rng   = np.random.default_rng(2024 + seed_offset)
    dates = pd.bdate_range("2019-01-02", periods=n)
    log_r = rng.normal(trend_per_day, daily_vol, n)

    for i in range(n):
        if rng.random() < crash_prob:
            length = rng.integers(5, 15)
            for j in range(length):
                if i + j < n:
                    log_r[i + j] -= crash_size / length
            for j in range(length, length * 3):
                if i + j < n:
                    log_r[i + j] += recovery_speed

    close = start_price * np.exp(np.cumsum(log_r))
    rng2  = np.random.default_rng(seed_offset + 9999)
    rng_  = range_pct = rng2.uniform(0.004, 0.022, n)
    spread = close * range_pct
    high   = close + spread * rng2.uniform(0.3, 1.0, n)
    low    = close - spread * rng2.uniform(0.3, 1.0, n)
    open_  = low   + (high - low) * rng2.uniform(0.1, 0.9, n)
    vol    = rng2.lognormal(np.log(base_volume), 0.5, n).astype(int)
    big    = np.where(np.abs(log_r) > daily_vol * 1.8)[0]
    vol[big] = (vol[big] * rng2.uniform(1.4, 2.8, len(big))).astype(int)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


def gen_index(n, start, trend, vol, seed):
    return gen_ohlcv(n, start, trend, vol, 50_000_000,
                     crash_prob=0.002, crash_size=0.08, seed_offset=seed)


# ──────────────────────────────────────────
#  TICKER PROFILES
# ──────────────────────────────────────────
# (name, market, start_px, trend/day, daily_vol, base_vol, seed, high_vol?)
#  high_vol=True → stop_atr_mult bumped to 2.5

TICKERS = [
    # ── IDX ─────────────────────────────────────────────────────────────────
    ("BBCA",  "IDX",  6_000, 0.00035, 0.010, 20_000_000, 10,  False),  # big bank, stable
    ("BBRI",  "IDX",  4_200, 0.00030, 0.013, 30_000_000, 20,  False),  # bank, slightly more vol
    ("ANTM",  "IDX",    800, 0.00010, 0.028, 50_000_000, 30,  True),   # mining, cyclical
    ("TLKM",  "IDX",  4_000, 0.00020, 0.009, 35_000_000, 70,  False),  # telecoms, very stable
    ("BMRI",  "IDX",  6_500, 0.00028, 0.012, 25_000_000, 80,  False),  # bank mandiri
    ("ASII",  "IDX",  6_800, 0.00018, 0.015, 40_000_000, 90,  False),  # astra, auto/diversified
    # ── US ──────────────────────────────────────────────────────────────────
    ("NVDA",  "US",    18.0, 0.00090, 0.028, 60_000_000, 40,  True),   # strong trend, high vol
    ("GOOGL", "US",  1050.0, 0.00045, 0.016, 25_000_000, 50,  False),  # steady compounder
    ("AAPL",  "US",   150.0, 0.00040, 0.014, 80_000_000, 60,  False),  # large cap
    ("MSFT",  "US",   110.0, 0.00048, 0.013, 55_000_000, 100, False),  # stable grower
    ("AMZN",  "US",   1700.0,0.00050, 0.018, 45_000_000, 110, False),  # ecommerce/cloud
    ("META",  "US",   150.0, 0.00055, 0.023, 50_000_000, 120, True),   # social, high vol + crashes
    ("TSLA",  "US",    50.0, 0.00070, 0.038, 90_000_000, 130, True),   # EV, very high vol
    ("JPM",   "US",   100.0, 0.00030, 0.014, 35_000_000, 140, False),  # financials
    ("ADBE",  "US",   250.0, 0.00042, 0.016, 20_000_000, 150, False),  # software, steady
    ("AMD",   "US",    20.0, 0.00075, 0.030, 55_000_000, 160, True),   # semicon, high vol
]

IDX_INDEX = dict(start=6000,  trend=0.00020, vol=0.009, seed=1)
US_INDEX  = dict(start=2700,  trend=0.00035, vol=0.010, seed=2)
N_DAYS    = 1500   # ~6 years


# ──────────────────────────────────────────
#  RUN
# ──────────────────────────────────────────
def build_cfg(market: str, high_vol: bool) -> Config:
    cfg = Config()
    if market == "IDX":
        cfg.commission_pct = 0.004
        cfg.stop_max_pct   = 0.06
    if high_vol:
        cfg.stop_atr_mult  = 2.5   # wider stop for volatile tickers
    return cfg


def main():
    idx_data = gen_index(N_DAYS, **IDX_INDEX)
    us_data  = gen_index(N_DAYS, **US_INDEX)
    summary  = []

    for (name, market, start_px, trend, vol, base_vol, seed, high_vol) in TICKERS:
        stock = gen_ohlcv(N_DAYS, start_px, trend, vol, base_vol,
                          crash_prob=0.003 if market == "IDX" else 0.002,
                          crash_size=0.14  if market == "IDX" else 0.10,
                          seed_offset=seed)
        index = idx_data if market == "IDX" else us_data
        cfg   = build_cfg(market, high_vol)

        trades, equity = run_backtest(stock, index, cfg)
        stats = compute_stats(trades, equity, cfg)

        tag = f"{name} [{market}]{'  ⚡ high-vol cfg' if high_vol else ''}"
        print_report(stats, tag, "2019-01-02 → 2024-12-31 (synthetic)")
        if trades and not stats.get("error"):
            print_trades(trades, n=5)

        has_err = "error" in stats
        summary.append({
            "ticker":     name,
            "market":     market,
            "high_vol":   "⚡" if high_vol else "  ",
            "trades":     stats.get("total_trades", 0),
            "win_rate":   f"{stats.get('win_rate', 0)*100:.0f}%",
            "pf":         stats.get("profit_factor", 0),
            "return_pct": stats.get("total_return_pct", 0),
            "sharpe":     stats.get("sharpe_ratio", 0),
            "max_dd":     stats.get("max_drawdown_pct", 0),
            "avg_hold":   stats.get("avg_hold_days", 0),
            "expectancy": stats.get("expectancy", 0),
            "final_cap":  stats.get("final_capital", 100_000),
            "ok":         not has_err and stats.get("profit_factor", 0) >= 1.5
                          and stats.get("return_pct", stats.get("total_return_pct", -1)) > 0,
        })

        path = f"/home/claude/swing_backtest/result_{name}_{market}_opt.json"
        with open(path, "w") as f:
            json.dump({"ticker": name, "market": market,
                       "config": {"target_rr": cfg.target_rr,
                                  "time_stop_days": cfg.time_stop_days,
                                  "stop_atr_mult": cfg.stop_atr_mult},
                       "stats": stats,
                       "trades": [{"entry_date":  str(t.entry_date.date()),
                                   "exit_date":   str(t.exit_date.date()),
                                   "hold_days":   t.hold_days,
                                   "entry_price": round(t.entry_price, 4),
                                   "exit_price":  round(t.exit_price,  4),
                                   "pnl_pct":     round(t.pnl_pct * 100, 2),
                                   "pnl_usd":     round(t.pnl, 2),
                                   "exit_reason": t.exit_reason}
                                  for t in trades]}, f, indent=2)

    # ── SUMMARY TABLE ──────────────────────────────────
    W = 96
    print("\n" + "═" * W)
    print("  OPTIMISED SUMMARY (target_rr=1.5 | time_stop=10d | high-vol: stop_atr=2.5×)")
    print("═" * W)
    hdr = (f"  {'':2} {'Ticker':<6} {'Mkt':<4} {'':2} "
           f"{'Trd':>4} {'WR':>4} {'PF':>5} {'Ret%':>6} "
           f"{'Sharpe':>6} {'MaxDD':>6} {'Hold':>5} "
           f"{'Expect':>8}  {'FinalCap':>11}")
    print(hdr)
    print("  " + "─" * (W - 2))

    for r in summary:
        flag = "✅" if r["ok"] else "⚠ "
        print(
            f"  {flag} {r['ticker']:<6} {r['market']:<4} {r['high_vol']:>2} "
            f"{r['trades']:>4} {r['win_rate']:>4} {r['pf']:>5.2f} "
            f"{r['return_pct']:>5.1f}% {r['sharpe']:>6.2f} "
            f"{r['max_dd']:>5.1f}% {r['avg_hold']:>4.1f}d "
            f"${r['expectancy']:>7,.0f}  ${r['final_cap']:>10,.0f}"
        )

    print("═" * W)
    good = [r for r in summary if r["ok"]]
    bad  = [r for r in summary if not r["ok"]]
    print(f"\n  ✅ Layak trade ({len(good)}): {', '.join(r['ticker'] for r in good)}")
    print(f"  ⚠  Butuh tuning ({len(bad)}):  {', '.join(r['ticker'] for r in bad)}")
    print("═" * W)
    print("  NOTE: Semua angka dari SYNTHETIC data. Jalankan lokal dengan Yahoo Finance untuk real data.")
    print("═" * W)


if __name__ == "__main__":
    main()
