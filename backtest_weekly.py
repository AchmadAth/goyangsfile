"""
Weekly Swing Trader Backtesting Engine
Target holding: 3–10 trading days (1–2 minggu)
Strategy: Trend filter + EMA21 pullback + RSI(14) bounce + fast exit

Entry logic:
  - Trend template terpenuhi (price > SMA50 > SMA200, SMA200 rising)
  - Harga pullback ke EMA21 (dalam ±6%)
  - RSI(14) dalam zona 35–65 (pullback dalam uptrend)
  - Volume >= 0.8x avg (pullback = volume kering, bukan spike)

Exit logic (mana yang duluan kena):
  - Profit target: 2R (stop distance × 2)
  - Stop loss: 1.5 × ATR(10), max 5%
  - Time stop: 7 hari trading (~1.5 minggu)
  - Trail: close di bawah EMA5 setelah profit > 3%

Usage:
  python backtest_weekly.py --market US --ticker AAPL --start 2019-01-01 --end 2024-12-31
  python backtest_weekly.py --market IDX --ticker BBCA.JK --show-trades
"""

import argparse, warnings, json
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════
@dataclass
class Config:
    # MAs
    ema_5:   int = 5       # fast trail
    ema_21:  int = 21      # weekly anchor
    sma_50:  int = 50
    sma_200: int = 200

    # Trend template
    pct_above_52w_low:   float = 0.15   # price >= 15% di atas 52w low
    pct_below_52w_high:  float = 0.30   # price dalam 30% dari 52w high

    # Entry
    rsi_period:        int   = 14
    rsi_max:           float = 65.0    # RSI di bawah 65 — pullback dalam uptrend
    rsi_min:           float = 35.0    # RSI tidak di bawah 35 (bukan crash)
    ema21_band:        float = 0.06    # harga dalam ±6% dari EMA21
    volume_mult:       float = 0.8     # vol >= 0.8× avg (pullback = volume kering)
    vol_lookback:      int   = 20
    atr_period:        int   = 10

    # Risk / exit
    stop_atr_mult:     float = 1.5     # stop = 1.5 × ATR (gunakan 2.5 untuk high-vol)
    stop_max_pct:      float = 0.05    # hard cap 5%
    target_rr:         float = 1.5     # target = 1.5× stop  (was 2.0 → lebih banyak profit_target hits)
    risk_per_trade:    float = 0.01    # 1% equity
    heat_max:          float = 0.06    # max 6% portfolio at risk
    time_stop_days:    int   = 10      # max hold 10 hari / 2 minggu  (was 7)
    trail_min_profit:  float = 0.03    # mulai trail setelah +3%

    # Sim
    commission_pct:    float = 0.001
    initial_capital:   float = 100_000.0


# ══════════════════════════════════════════════════════
#  MANUAL INDICATORS (no pandas_ta dependency)
# ══════════════════════════════════════════════════════
def _rsi(series: pd.Series, length: int) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=length - 1, min_periods=length).mean()
    avg_loss = loss.ewm(com=length - 1, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=length - 1, min_periods=length).mean()


# ══════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════
def load_data(ticker: str, index_ticker: str, start: str,
              end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"  Fetching {ticker} & {index_ticker}...")
    s = yf.download(ticker,       start=start, end=end, auto_adjust=True, progress=False)
    i = yf.download(index_ticker, start=start, end=end, auto_adjust=True, progress=False)
    for df in [s, i]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
    return s, i


# ══════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════
def add_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    c, v, h, lo = df["Close"], df["Volume"], df["High"], df["Low"]

    df["ema5"]         = c.ewm(span=cfg.ema_5,  adjust=False).mean()
    df["ema21"]        = c.ewm(span=cfg.ema_21, adjust=False).mean()
    df["sma50"]        = c.rolling(cfg.sma_50).mean()
    df["sma200"]       = c.rolling(cfg.sma_200).mean()
    df["atr"]          = _atr(h, lo, c, length=cfg.atr_period)
    df["rsi14"]        = _rsi(c, length=cfg.rsi_period)
    df["high_52w"]     = c.rolling(252).max()
    df["low_52w"]      = c.rolling(252).min()
    df["sma200_slope"] = df["sma200"].diff(21)
    df["vol_avg"]      = v.rolling(cfg.vol_lookback).mean()
    df["dist_ema21"]   = (c - df["ema21"]) / df["ema21"]
    df["rs42"]         = c.pct_change(42)
    return df


def add_index_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    df["sma200"] = c.rolling(cfg.sma_200).mean()
    df["ema21"]  = c.ewm(span=cfg.ema_21, adjust=False).mean()
    df["rs42"]   = c.pct_change(42)
    return df


# ══════════════════════════════════════════════════════
#  RULES (R01–R13 = entry gate)
# ══════════════════════════════════════════════════════
ENTRY_RULE_KEYS = {
    "R01_index_above_sma200", "R02_index_healthy", "R03_vix_ok",
    "R04_above_sma200", "R05_above_sma50", "R06_sma200_rising",
    "R07_ema21_above_sma50", "R08_within_52w_high", "R09_above_52w_low",
    "R10_relative_strength", "R11_pullback_ema21", "R12_rsi_pullback",
    "R13_volume_ok",
}


def check_rules(row, idx_row, cfg: Config, idx_rs_med: float,
                vix_value: Optional[float] = None) -> dict:
    c = float(row["Close"])
    r = {}

    # Layer 1: Market regime
    r["R01_index_above_sma200"] = bool(
        pd.notna(idx_row["sma200"]) and idx_row["Close"] > idx_row["sma200"])
    r["R02_index_healthy"] = bool(
        pd.notna(idx_row["ema21"]) and idx_row["Close"] > idx_row["ema21"] * 0.97)
    r["R03_vix_ok"] = bool(vix_value < 30.0) if vix_value is not None else True

    # Layer 2: Stock trend
    r["R04_above_sma200"]      = bool(pd.notna(row["sma200"]) and c > row["sma200"])
    r["R05_above_sma50"]       = bool(pd.notna(row["sma50"])  and c > row["sma50"])
    r["R06_sma200_rising"]     = bool(pd.notna(row["sma200_slope"]) and row["sma200_slope"] > 0)
    r["R07_ema21_above_sma50"] = bool(
        pd.notna(row["ema21"]) and pd.notna(row["sma50"]) and row["ema21"] >= row["sma50"] * 0.99)
    r["R08_within_52w_high"]   = bool(
        pd.notna(row["high_52w"]) and c >= row["high_52w"] * (1 - cfg.pct_below_52w_high))
    r["R09_above_52w_low"]     = bool(
        pd.notna(row["low_52w"]) and c >= row["low_52w"] * (1 + cfg.pct_above_52w_low))
    r["R10_relative_strength"] = bool(
        pd.notna(row["rs42"]) and row["rs42"] > idx_rs_med)

    # Layer 3: Entry trigger
    r["R11_pullback_ema21"] = bool(
        pd.notna(row["dist_ema21"]) and abs(row["dist_ema21"]) <= cfg.ema21_band)
    r["R12_rsi_pullback"]   = bool(
        pd.notna(row["rsi14"]) and cfg.rsi_min <= row["rsi14"] <= cfg.rsi_max)
    r["R13_volume_ok"]      = bool(
        pd.notna(row["vol_avg"]) and row["Volume"] >= row["vol_avg"] * cfg.volume_mult)

    return r


def entry_ok(rules: dict) -> bool:
    return all(rules.get(k, False) for k in ENTRY_RULE_KEYS)


# ══════════════════════════════════════════════════════
#  TRADE
# ══════════════════════════════════════════════════════
@dataclass
class Trade:
    entry_date:   pd.Timestamp
    entry_price:  float
    stop_price:   float
    target_price: float
    size:         int
    exit_date:    Optional[pd.Timestamp] = None
    exit_price:   Optional[float]        = None
    exit_reason:  str   = ""
    pnl:          float = 0.0
    pnl_pct:      float = 0.0
    hold_days:    int   = 0


# ══════════════════════════════════════════════════════
#  BACKTEST LOOP
# ══════════════════════════════════════════════════════
def run_backtest(stock: pd.DataFrame, index: pd.DataFrame, cfg: Config,
                 vix: Optional[pd.Series] = None):
    stock = add_indicators(stock, cfg)
    index = add_index_indicators(index, cfg)

    common = stock.index.intersection(index.index)
    stock  = stock.loc[common]
    index  = index.loc[common]
    if vix is not None:
        vix = vix.reindex(common)

    stock["rs_med"] = stock["rs42"].rolling(42).median()

    capital  = cfg.initial_capital
    equity   = {}
    trades   = []
    open_t: Optional[Trade] = None
    hold_d   = 0

    for i, date in enumerate(stock.index):
        equity[date] = capital
        if i < cfg.sma_200:
            continue

        row     = stock.iloc[i]
        idx_row = index.iloc[i]
        c       = float(row["Close"])
        rs_med  = float(stock["rs_med"].iloc[i]) if pd.notna(stock["rs_med"].iloc[i]) else 0.0
        vix_val = float(vix.loc[date]) if (vix is not None and date in vix.index
                                            and pd.notna(vix.loc[date])) else None

        # ── MANAGE OPEN TRADE ──
        if open_t is not None:
            hold_d += 1
            xp, xr = None, ""

            if c <= open_t.stop_price:
                xp, xr = open_t.stop_price, "stop_loss"
            elif c >= open_t.target_price:
                xp, xr = c, "profit_target"
            elif (hold_d >= 2
                  and c >= open_t.entry_price * (1 + cfg.trail_min_profit)
                  and pd.notna(row["ema5"])
                  and c < float(row["ema5"])):
                xp, xr = c, "trail_ema5"
            elif hold_d >= cfg.time_stop_days:
                xp, xr = c, "time_stop"

            if xp is not None:
                comm = xp * open_t.size * cfg.commission_pct
                pnl  = (xp - open_t.entry_price) * open_t.size - comm
                open_t.exit_date   = date
                open_t.exit_price  = xp
                open_t.exit_reason = xr
                open_t.pnl         = pnl
                open_t.pnl_pct     = (xp - open_t.entry_price) / open_t.entry_price
                open_t.hold_days   = hold_d
                capital += pnl
                equity[date] = capital
                trades.append(open_t)
                open_t, hold_d = None, 0

        # ── SCAN ENTRY ──
        if open_t is None:
            rules = check_rules(row, idx_row, cfg, rs_med, vix_value=vix_val)
            if entry_ok(rules):
                atr  = float(row["atr"]) if pd.notna(row["atr"]) else c * 0.01
                raw_stop = c - cfg.stop_atr_mult * atr
                stop = max(raw_stop, c * (1 - cfg.stop_max_pct))
                dist = c - stop
                if dist <= 0 or dist / c > cfg.stop_max_pct:
                    equity[date] = capital
                    continue
                target = c + dist * cfg.target_rr
                size   = max(1, int((capital * cfg.risk_per_trade) / dist))
                heat   = (dist * size) / capital
                if heat <= cfg.heat_max:
                    comm_entry = c * size * cfg.commission_pct
                    if comm_entry < capital:
                        capital -= comm_entry
                        open_t  = Trade(date, c, stop, target, size)
                        hold_d  = 0
                        equity[date] = capital

    # Close open trade at end
    if open_t is not None:
        last = float(stock["Close"].iloc[-1])
        comm = last * open_t.size * cfg.commission_pct
        pnl  = (last - open_t.entry_price) * open_t.size - comm
        open_t.exit_date   = stock.index[-1]
        open_t.exit_price  = last
        open_t.exit_reason = "end_of_data"
        open_t.pnl         = pnl
        open_t.pnl_pct     = (last - open_t.entry_price) / open_t.entry_price
        open_t.hold_days   = hold_d
        capital += pnl
        trades.append(open_t)

    return trades, pd.Series(equity)


# ══════════════════════════════════════════════════════
#  STATS & REPORT
# ══════════════════════════════════════════════════════
def compute_stats(trades, equity: pd.Series, cfg: Config) -> dict:
    if not trades:
        return {"error": "Nol trade — coba perlebar RSI range atau kurangi volume_mult"}

    pnls = [t.pnl for t in trades]
    pcts = [t.pnl_pct for t in trades]
    wins = [p for p in pnls if p > 0]
    loss = [p for p in pnls if p <= 0]
    ret  = (equity.iloc[-1] - cfg.initial_capital) / cfg.initial_capital
    eq_r = equity.pct_change().dropna()
    sharpe = (eq_r.mean() / eq_r.std()) * np.sqrt(252) if eq_r.std() > 0 else 0
    dd   = ((equity - equity.cummax()) / equity.cummax()).min()
    pf   = abs(sum(wins) / sum(loss)) if loss else float("inf")

    by_exit = {}
    for t in trades:
        by_exit[t.exit_reason] = by_exit.get(t.exit_reason, 0) + 1

    return {
        "total_trades":     len(trades),
        "win_rate":         round(len(wins) / len(trades), 3),
        "profit_factor":    round(pf, 2),
        "total_return_pct": round(ret * 100, 2),
        "sharpe_ratio":     round(sharpe, 2),
        "max_drawdown_pct": round(dd * 100, 2),
        "avg_hold_days":    round(np.mean([t.hold_days for t in trades]), 1),
        "avg_win_pct":      round(np.mean([p for p in pcts if p > 0]) * 100, 2) if wins else 0,
        "avg_loss_pct":     round(np.mean([p for p in pcts if p <= 0]) * 100, 2) if loss else 0,
        "expectancy":       round(np.mean(pnls), 2),
        "exit_breakdown":   by_exit,
        "final_capital":    round(equity.iloc[-1], 2),
    }


def print_report(stats: dict, ticker: str, period: str):
    sep = "═" * 56
    print(f"\n{sep}")
    print(f"  WEEKLY SWING REPORT  |  {ticker}  |  {period}")
    print(sep)
    if "error" in stats:
        print(f"  ⚠  {stats['error']}")
        print(sep)
        return
    pf_icon = "✅" if stats["profit_factor"] >= 1.5              else "⚠ "
    wr_icon = "✅" if stats["win_rate"]       >= 0.45             else "⚠ "
    dd_icon = "✅" if stats["max_drawdown_pct"] >= -15            else "⚠ "
    hd_icon = "✅" if stats["avg_hold_days"]  <= 10               else "⚠ "
    print(f"  Trades              : {stats['total_trades']}")
    print(f"  Win Rate            : {wr_icon} {stats['win_rate']*100:.1f}%")
    print(f"  Profit Factor       : {pf_icon} {stats['profit_factor']}")
    print(f"  Total Return        : {stats['total_return_pct']}%")
    print(f"  Sharpe Ratio        : {stats['sharpe_ratio']}")
    print(f"  Max Drawdown        : {dd_icon} {stats['max_drawdown_pct']}%")
    print(f"  Avg Hold Days       : {hd_icon} {stats['avg_hold_days']} hari")
    print(f"  Avg Win             : +{stats['avg_win_pct']}%")
    print(f"  Avg Loss            : {stats['avg_loss_pct']}%")
    print(f"  Expectancy/Trade    : ${stats['expectancy']}")
    print(f"  Final Capital       : ${stats['final_capital']:,.2f}")
    print(f"\n  Exit Breakdown:")
    for r, n in sorted(stats["exit_breakdown"].items(), key=lambda x: -x[1]):
        bar = "█" * min(n, 30)
        print(f"    {r:<22} {n:>3}  {bar}")
    print(sep)


def print_trades(trades, n: int = 15):
    recent = trades[-n:]
    print(f"\n  Trades (last {len(recent)}):")
    print(f"  {'Entry':<11} {'Exit':<11} {'D':>3}  {'Entry':>8} {'Exit':>8} {'P&L%':>7}  Reason")
    print("  " + "─" * 68)
    for t in recent:
        ic = "✅" if t.pnl_pct > 0 else "❌"
        print(f"  {str(t.entry_date.date()):<11} {str(t.exit_date.date()):<11} "
              f"{t.hold_days:>3}  {t.entry_price:>8.2f} {t.exit_price:>8.2f} "
              f"{t.pnl_pct*100:>+6.1f}%  {ic} {t.exit_reason}")


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market",      choices=["US", "IDX"], default="US")
    ap.add_argument("--ticker",      default="AAPL")
    ap.add_argument("--start",       default="2019-01-01")
    ap.add_argument("--end",         default="2024-12-31")
    ap.add_argument("--capital",     type=float, default=100_000)
    ap.add_argument("--show-trades", action="store_true")
    args = ap.parse_args()

    cfg = Config(initial_capital=args.capital)
    if args.market == "IDX":
        idx_ticker         = "^JKSE"
        cfg.commission_pct = 0.004
        cfg.stop_max_pct   = 0.06
        print("  [IDX] commission=0.4%, stop cap=6%")
    else:
        idx_ticker = "^GSPC"

    print(f"\n{'═'*56}")
    print(f"  {args.market} | {args.ticker} | {args.start} → {args.end}")
    print(f"{'═'*56}")

    stock, index = load_data(args.ticker, idx_ticker, args.start, args.end)
    if stock.empty or len(stock) < 250:
        print("  ERROR: Data kurang. Perlebar range.")
        return

    trades, equity = run_backtest(stock, index, cfg)
    stats = compute_stats(trades, equity, cfg)
    print_report(stats, args.ticker, f"{args.start} → {args.end}")
    if args.show_trades and trades:
        print_trades(trades)

    fname = f"result_{args.ticker}_{args.market}.json"
    path  = f"/home/claude/swing_backtest/{fname}"
    with open(path, "w") as f:
        json.dump({
            "ticker": args.ticker, "market": args.market,
            "stats":  stats,
            "config": {
                "time_stop_days": cfg.time_stop_days,
                "stop_atr_mult":  cfg.stop_atr_mult,
                "stop_max_pct":   cfg.stop_max_pct,
                "target_rr":      cfg.target_rr,
                "rsi_zone":       f"{cfg.rsi_min}–{cfg.rsi_max}",
                "ema21_band_pct": cfg.ema21_band,
            },
            "trades": [
                {"entry_date":  str(t.entry_date.date()),
                 "exit_date":   str(t.exit_date.date()),
                 "hold_days":   t.hold_days,
                 "entry_price": round(t.entry_price, 4),
                 "exit_price":  round(t.exit_price,  4),
                 "pnl_pct":     round(t.pnl_pct * 100, 2),
                 "pnl_usd":     round(t.pnl, 2),
                 "exit_reason": t.exit_reason}
                for t in trades
            ],
        }, f, indent=2)
    print(f"\n  Saved → {path}")


if __name__ == "__main__":
    main()
