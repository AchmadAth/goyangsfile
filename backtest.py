"""
Swing Trader Backtesting Engine
20-Rule AND-Gate System (Minervini SEPA + O'Neil CAN SLIM + Connors RSI2)
Usage: python backtest.py --market US --ticker AAPL --start 2020-01-01 --end 2024-12-31
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass, field
from typing import Optional
import json

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
@dataclass
class Config:
    # Moving averages
    sma_50: int = 50
    sma_150: int = 150
    sma_200: int = 200

    # Trend template thresholds
    pct_above_52w_low: float = 0.30    # Rule 9: price >= 30% above 52w low
    pct_below_52w_high: float = 0.25   # Rule 10: price <= 25% below 52w high

    # Entry trigger
    volume_multiplier: float = 1.5     # Rule 13: breakout vol >= 1.5x avg
    volume_lookback: int = 50

    # RSI(2) mean-reversion alternative
    rsi2_oversold: float = 10.0        # Rule 12B
    rsi2_exit_sma: int = 5

    # Breakout detection
    breakout_lookback: int = 20        # pivot high over N days

    # Risk management
    stop_loss_pct: float = 0.07        # Rule 14: 7% hard stop
    risk_per_trade: float = 0.01       # Rule 15: 1% account risk per trade
    portfolio_heat_max: float = 0.05   # Rule 16: max 5% portfolio at risk
    min_rr_ratio: float = 2.0          # Rule 17: informational; profit target uses profit_target_pct directly

    # Exit management
    time_stop_days: int = 5            # Rule 18: exit if no progress in 5 days
    time_stop_min_progress_pct: float = 0.005  # Rule 18: minimum gain to avoid time-stop exit
    profit_target_pct: float = 0.20    # Rule 20: take profit at 20%
    trail_sma: int = 10                # Rule 19: trail with 10-day EMA

    # Market regime
    vix_threshold: float = 30.0        # Rule 3
    breadth_threshold: float = 0.50    # Rule 2

    # Simulation
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001      # 0.1% per side (US); use 0.004 for IDX
    slippage_bps: float = 5.0          # bps of adverse slippage per fill; 0.0 = no slippage


# ─────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────
def load_data(ticker: str, index_ticker: str, start: str,
              end: str) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
    print(f"  Fetching {ticker} and {index_ticker}...")
    stock = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    index = yf.download(index_ticker, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = stock.columns.get_level_values(0)
    if isinstance(index.columns, pd.MultiIndex):
        index.columns = index.columns.get_level_values(0)

    stock.index = pd.to_datetime(stock.index)
    index.index = pd.to_datetime(index.index)

    vix: Optional[pd.Series] = None
    try:
        vix_df = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)
        if not vix_df.empty:
            vix = vix_df["Close"]
            vix.index = pd.to_datetime(vix.index)
    except Exception:
        pass  # VIX unavailable; R03 will default to True

    return stock, index, vix


# ─────────────────────────────────────────────
# INDICATOR ENGINE
# ─────────────────────────────────────────────
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=length - 1, min_periods=length).mean()
    avg_loss = loss.ewm(com=length - 1, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def apply_slippage(price: float, is_buy: bool, slippage_bps: float) -> float:
    """Adjust fill price by fixed bps in the adverse direction."""
    adj = price * (slippage_bps / 10_000)
    return price + adj if is_buy else price - adj


def compute_indicators(df: pd.DataFrame, cfg: Config, index_close: pd.Series) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    volume = df["Volume"]

    # Moving averages
    df["sma50"]  = close.rolling(cfg.sma_50).mean()
    df["sma150"] = close.rolling(cfg.sma_150).mean()
    df["sma200"] = close.rolling(cfg.sma_200).mean()
    df["ema10"]  = close.ewm(span=cfg.trail_sma, adjust=False).mean()
    df["sma5"]   = close.rolling(cfg.rsi2_exit_sma).mean()

    # 52-week high/low
    df["high_52w"] = close.rolling(252).max()
    df["low_52w"]  = close.rolling(252).min()

    # 200-day SMA slope (1-month = 21 days)
    df["sma200_slope"] = df["sma200"].diff(21)

    # Volume average
    df["vol_avg50"] = volume.rolling(cfg.volume_lookback).mean()

    # RSI(2) — manual implementation (no pandas_ta dependency)
    df["rsi2"] = rsi(close, length=2)

    # Pivot high (breakout detection): highest high over N days
    df["pivot_high"] = close.rolling(cfg.breakout_lookback).max().shift(1)

    # Relative strength vs index: ratio of stock 63d return to index 63d return
    # Values > 1.0 mean the stock outperformed the index over the past 3 months
    stock_return_63 = close.pct_change(63)
    index_return_63 = index_close.reindex(close.index).pct_change(63)
    df["rs_raw"] = stock_return_63 / index_return_63.replace(0, np.nan)

    return df


def compute_index_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    df["sma50"]  = close.rolling(cfg.sma_50).mean()
    df["sma200"] = close.rolling(cfg.sma_200).mean()
    return df


# ─────────────────────────────────────────────
# 20-RULE CHECKLIST
# ─────────────────────────────────────────────
def evaluate_rules(row: pd.Series, index_row: pd.Series, cfg: Config,
                   index_rs_median: float, vix_value: Optional[float] = None) -> dict:
    """Returns dict of rule_name -> bool for each of the 20 rules."""
    rules = {}
    c = row["Close"]

    # --- LAYER 1: Market Regime ---
    rules["R01_index_above_sma200"] = bool(
        pd.notna(index_row["sma200"]) and index_row["Close"] > index_row["sma200"]
    )
    rules["R02_breadth_proxy"] = bool(
        pd.notna(index_row["sma50"]) and index_row["Close"] > index_row["sma50"]
    )
    # Rule 3: VIX regime filter (True if VIX data unavailable)
    if vix_value is not None:
        rules["R03_vix_ok"] = bool(vix_value < cfg.vix_threshold)
    else:
        rules["R03_vix_ok"] = True

    # --- LAYER 2: Trend Template (Minervini) ---
    rules["R04_price_above_sma150_sma200"] = bool(
        pd.notna(row["sma150"]) and pd.notna(row["sma200"]) and
        c > row["sma150"] and c > row["sma200"]
    )
    rules["R05_sma150_above_sma200"] = bool(
        pd.notna(row["sma150"]) and pd.notna(row["sma200"]) and
        row["sma150"] > row["sma200"]
    )
    rules["R06_sma200_trending_up"] = bool(
        pd.notna(row["sma200_slope"]) and row["sma200_slope"] > 0
    )
    rules["R07_sma50_stacked"] = bool(
        pd.notna(row["sma50"]) and pd.notna(row["sma150"]) and pd.notna(row["sma200"]) and
        row["sma50"] > row["sma150"] and row["sma50"] > row["sma200"]
    )
    rules["R08_price_above_sma50"] = bool(
        pd.notna(row["sma50"]) and c > row["sma50"]
    )
    rules["R09_above_52w_low_30pct"] = bool(
        pd.notna(row["low_52w"]) and
        c >= row["low_52w"] * (1 + cfg.pct_above_52w_low)
    )
    rules["R10_within_52w_high_25pct"] = bool(
        pd.notna(row["high_52w"]) and
        c >= row["high_52w"] * (1 - cfg.pct_below_52w_high)
    )
    rules["R11_relative_strength"] = bool(
        pd.notna(row["rs_raw"]) and row["rs_raw"] > index_rs_median
    )

    # --- LAYER 3: Entry Trigger ---
    # Rule 12: breakout above pivot high OR RSI(2) oversold mean-reversion (12B)
    r12_breakout = bool(pd.notna(row["pivot_high"]) and c > row["pivot_high"])
    r12b_rsi2    = bool(pd.notna(row["rsi2"]) and row["rsi2"] < cfg.rsi2_oversold)
    rules["R12_breakout_pivot"] = r12_breakout or r12b_rsi2
    # Rule 13: volume confirmation
    rules["R13_volume_confirmed"] = bool(
        pd.notna(row["vol_avg50"]) and
        row["Volume"] >= row["vol_avg50"] * cfg.volume_multiplier
    )

    # --- LAYER 4: Risk Management (pre-trade checks) ---
    rules["R14_stop_loss_defined"] = True  # always true; enforced at execution
    rules["R15_position_sized"]    = True  # always true; enforced at execution
    rules["R16_portfolio_heat_ok"] = True  # enforced at portfolio level
    rules["R17_rr_ratio_ok"]       = True  # enforced at execution

    # --- LAYER 5: Exit (informational at scan time) ---
    rules["R18_time_stop"] = True
    rules["R19_trail_stop"] = True
    rules["R20_profit_target"] = True

    return rules


ENTRY_RULE_KEYS = {
    "R01_index_above_sma200", "R02_breadth_proxy", "R03_vix_ok",
    "R04_price_above_sma150_sma200", "R05_sma150_above_sma200",
    "R06_sma200_trending_up", "R07_sma50_stacked", "R08_price_above_sma50",
    "R09_above_52w_low_30pct", "R10_within_52w_high_25pct",
    "R11_relative_strength", "R12_breakout_pivot", "R13_volume_confirmed",
}


def all_rules_pass(rules: dict) -> bool:
    return all(rules[k] for k in ENTRY_RULE_KEYS if k in rules)


# ─────────────────────────────────────────────
# BACKTESTER
# ─────────────────────────────────────────────
@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    size: int
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    rules_at_entry: dict = field(default_factory=dict)


def run_backtest(stock: pd.DataFrame, index: pd.DataFrame, cfg: Config,
                 ticker: str, vix: Optional[pd.Series] = None) -> tuple[list[Trade], pd.Series]:
    index = compute_index_indicators(index, cfg)
    # Pass index close for true relative-strength calculation (stock return / index return)
    index_aligned = index["Close"].reindex(stock.index)
    stock = compute_indicators(stock, cfg, index_close=index_aligned)

    # Align dates
    common_dates = stock.index.intersection(index.index)
    stock = stock.loc[common_dates]
    index = index.loc[common_dates]
    if vix is not None:
        vix = vix.reindex(common_dates)

    # Rolling median RS for relative strength benchmark
    stock["rs_median"] = stock["rs_raw"].rolling(63).median()

    capital = cfg.initial_capital
    equity_curve = {stock.index[0]: capital}
    trades: list[Trade] = []
    open_trade: Optional[Trade] = None
    days_in_trade = 0

    for i, date in enumerate(stock.index):
        if i < cfg.sma_200:
            equity_curve[date] = capital
            continue

        row = stock.iloc[i]
        idx_row = index.iloc[i]
        c = float(row["Close"])

        rs_median = float(stock["rs_median"].iloc[i]) if pd.notna(stock["rs_median"].iloc[i]) else 0.0

        # ── MANAGE OPEN TRADE ──
        if open_trade is not None:
            days_in_trade += 1
            exit_price = None
            exit_reason = ""

            # Rule 14: Hard stop
            if c <= open_trade.stop_price:
                exit_price = apply_slippage(open_trade.stop_price, is_buy=False, slippage_bps=cfg.slippage_bps)
                exit_reason = "stop_loss"

            # Rule 20: Profit target
            elif c >= open_trade.target_price:
                exit_price = apply_slippage(c, is_buy=False, slippage_bps=cfg.slippage_bps)
                exit_reason = "profit_target"

            # Rule 18: Time stop
            elif days_in_trade >= cfg.time_stop_days and c < open_trade.entry_price * (1 + cfg.time_stop_min_progress_pct):
                exit_price = apply_slippage(c, is_buy=False, slippage_bps=cfg.slippage_bps)
                exit_reason = "time_stop"

            # Rule 19: Trail stop (close below 10-day EMA)
            elif days_in_trade > 5 and pd.notna(row["ema10"]) and c < float(row["ema10"]):
                exit_price = apply_slippage(c, is_buy=False, slippage_bps=cfg.slippage_bps)
                exit_reason = "trail_stop"

            if exit_price is not None:
                comm = exit_price * open_trade.size * cfg.commission_pct
                pnl = (exit_price - open_trade.entry_price) * open_trade.size - comm
                open_trade.exit_date = date
                open_trade.exit_price = exit_price
                open_trade.exit_reason = exit_reason
                open_trade.pnl = pnl
                open_trade.pnl_pct = (exit_price - open_trade.entry_price) / open_trade.entry_price
                capital += pnl
                trades.append(open_trade)
                open_trade = None
                days_in_trade = 0

        # ── SCAN FOR ENTRY ──
        if open_trade is None:
            vix_val = float(vix.loc[date]) if (vix is not None and date in vix.index and pd.notna(vix.loc[date])) else None
            rules = evaluate_rules(row, idx_row, cfg, rs_median, vix_value=vix_val)
            if all_rules_pass(rules):
                fill_price  = apply_slippage(c, is_buy=True, slippage_bps=cfg.slippage_bps)
                stop        = fill_price * (1 - cfg.stop_loss_pct)
                target      = fill_price * (1 + cfg.profit_target_pct)   # was: stop_loss_pct * min_rr_ratio
                risk_per_share = fill_price - stop
                size        = max(1, int((capital * cfg.risk_per_trade) / risk_per_share))

                # Portfolio heat check (Rule 16)
                total_heat = (risk_per_share * size) / capital
                if total_heat <= cfg.portfolio_heat_max:
                    # Size down if total cost (including commission) exceeds available capital
                    trade_cost = fill_price * size
                    comm_entry = trade_cost * cfg.commission_pct
                    while size > 1 and (trade_cost + comm_entry) > capital:
                        size -= 1
                        trade_cost = fill_price * size
                        comm_entry = trade_cost * cfg.commission_pct
                    capital -= comm_entry
                    open_trade = Trade(
                        entry_date=date,
                        entry_price=fill_price,
                        stop_price=stop,
                        target_price=target,
                        size=size,
                        rules_at_entry=rules
                    )
                    days_in_trade = 0

        equity_curve[date] = capital

    # Close any open trade at end
    if open_trade is not None:
        raw_last  = float(stock["Close"].iloc[-1])
        last_price = apply_slippage(raw_last, is_buy=False, slippage_bps=cfg.slippage_bps)
        comm_exit  = last_price * open_trade.size * cfg.commission_pct
        pnl = (last_price - open_trade.entry_price) * open_trade.size - comm_exit
        open_trade.exit_date  = stock.index[-1]
        open_trade.exit_price = last_price
        open_trade.exit_reason = "end_of_data"
        open_trade.pnl = pnl
        open_trade.pnl_pct = (last_price - open_trade.entry_price) / open_trade.entry_price
        capital += pnl
        trades.append(open_trade)

    return trades, pd.Series(equity_curve)


# ─────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────
def compute_stats(trades: list[Trade], equity: pd.Series, cfg: Config) -> dict:
    if not trades:
        return {"error": "No trades generated"}

    pnls = [t.pnl for t in trades]
    pct_changes = [t.pnl_pct for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_return = (equity.iloc[-1] - cfg.initial_capital) / cfg.initial_capital
    equity_returns = equity.pct_change().dropna()
    sharpe = (equity_returns.mean() / equity_returns.std()) * np.sqrt(252) if equity_returns.std() > 0 else 0

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = float(drawdown.min())

    profit_factor = abs(sum(wins) / sum(losses)) if losses else float("inf")
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    return {
        "total_trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "profit_factor": round(profit_factor, 2),
        "total_return_pct": round(total_return * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "avg_win_pct": round(np.mean([p for p in pct_changes if p > 0]) * 100, 2) if wins else 0,
        "avg_loss_pct": round(np.mean([p for p in pct_changes if p <= 0]) * 100, 2) if losses else 0,
        "expectancy_usd": round(np.mean(pnls), 2),
        "exit_breakdown": exit_reasons,
        "final_capital": round(equity.iloc[-1], 2),
    }


def print_report(stats: dict, ticker: str, start: str, end: str):
    print("\n" + "="*55)
    print(f"  BACKTEST REPORT — {ticker} | {start} → {end}")
    print("="*55)
    print(f"  Total Trades     : {stats['total_trades']}")
    print(f"  Win Rate         : {stats['win_rate']*100:.1f}%")
    print(f"  Profit Factor    : {stats['profit_factor']}")
    print(f"  Total Return     : {stats['total_return_pct']}%")
    print(f"  Sharpe Ratio     : {stats['sharpe_ratio']}")
    print(f"  Max Drawdown     : {stats['max_drawdown_pct']}%")
    print(f"  Avg Win          : ${stats['avg_win_usd']} ({stats['avg_win_pct']}%)")
    print(f"  Avg Loss         : ${stats['avg_loss_usd']} ({stats['avg_loss_pct']}%)")
    print(f"  Expectancy/Trade : ${stats['expectancy_usd']}")
    print(f"  Final Capital    : ${stats['final_capital']:,.2f}")
    print(f"\n  Exit Breakdown:")
    for reason, count in stats["exit_breakdown"].items():
        print(f"    {reason:<20}: {count}")
    print("="*55)


def print_trades(trades: list[Trade], n: int = 10):
    print(f"\n  Last {min(n, len(trades))} Trades:")
    print(f"  {'Entry Date':<12} {'Exit Date':<12} {'Entry':>8} {'Exit':>8} {'P&L%':>7} {'Reason'}")
    print("  " + "-"*65)
    for t in trades[-n:]:
        print(f"  {str(t.entry_date.date()):<12} {str(t.exit_date.date()):<12} "
              f"{t.entry_price:>8.2f} {t.exit_price:>8.2f} "
              f"{t.pnl_pct*100:>+6.1f}%  {t.exit_reason}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Swing Trader Backtest Engine")
    parser.add_argument("--market", choices=["US", "IDX"], default="US")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--show-trades", action="store_true")
    args = parser.parse_args()

    cfg = Config(initial_capital=args.capital)

    if args.market == "IDX":
        index_ticker = "^JKSE"
        cfg.commission_pct = 0.004   # Higher for IDX
        print(f"\n[IDX MODE] Commission set to 0.4% round-trip + tax")
    else:
        index_ticker = "^GSPC"

    print(f"\n[Swing Bot Backtest] {args.market} | {args.ticker} vs {index_ticker}")
    print(f"Period: {args.start} → {args.end} | Capital: ${args.capital:,.0f}\n")

    stock, index, vix = load_data(args.ticker, index_ticker, args.start, args.end)

    if stock.empty or len(stock) < 250:
        print("ERROR: Insufficient data. Try a longer date range or different ticker.")
        return

    print("  Running 20-rule backtest...")
    trades, equity = run_backtest(stock, index, cfg, args.ticker, vix=vix)
    stats = compute_stats(trades, equity, cfg)

    print_report(stats, args.ticker, args.start, args.end)

    if args.show_trades:
        print_trades(trades)

    # Save results
    out = {
        "ticker": args.ticker,
        "market": args.market,
        "period": f"{args.start} to {args.end}",
        "stats": stats,
        "trades": [
            {
                "entry_date": str(t.entry_date.date()),
                "exit_date": str(t.exit_date.date()) if t.exit_date else None,
                "entry_price": round(t.entry_price, 4),
                "exit_price": round(t.exit_price, 4) if t.exit_price else None,
                "pnl_pct": round(t.pnl_pct * 100, 2),
                "pnl_usd": round(t.pnl, 2),
                "exit_reason": t.exit_reason,
            }
            for t in trades
        ]
    }
    outfile = f"/home/claude/swing_backtest/result_{args.ticker}_{args.market}.json"
    with open(outfile, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved → {outfile}")


if __name__ == "__main__":
    main()
