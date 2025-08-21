"""
XAUUSD Liquidity AI — Buyers vs Sellers Probability (Python)

What this script does
---------------------
1) Pulls price data for XAUUSD from MetaTrader 5 (your broker) OR a local CSV fallback.
2) Computes liquidity-sweep signals, fair value gaps (FVG), structure breaks, and a simple buy/sell pressure proxy.
3) Produces a probability score (Buy vs Sell) + risk-managed trade plan (SL/TP, R:R≥1:3).
4) (Optional) Places trades on MT5 (disabled by default; enable in CONFIG execution section).

Quick start
-----------
1) Install deps (Windows/Mac/Ubuntu):
   pip install pandas numpy MetaTrader5 pydantic ta

2) Make sure MetaTrader 5 is open and logged into an account that has XAUUSD.

3) Run:
   python xauusd_liquidity_ai.py --source mt5 --symbol XAUUSD --tf M5 --bars 1500 --risk 0.01 --equity 1000

4) CSV fallback:
   python xauusd_liquidity_ai.py --source csv --csv_path ./xauusd_M5.csv --risk 0.01 --equity 1000
   (CSV columns expected: time,open,high,low,close,tick_volume,spread,real_volume)

Notes
-----
- This is a starter framework focusing on signal quality & risk management. 
- You’re responsible for live execution testing on demo first.
- For Binance-only users: Binance doesn’t natively offer XAUUSD perpetuals. For gold, use MT5/CFD broker data or CME feeds via a vendor.

"""
from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5  # type: ignore
except Exception:  # Allow running w/o MT5 installed
    mt5 = None

# -----------------------
# Utility & Config
# -----------------------
@dataclass
class Config:
    source: str = "mt5"                   # mt5 | csv
    symbol: str = "XAUUSD"
    timeframe: str = "M5"                 # M1, M5, M15, M30, H1, H4, D1
    bars: int = 1500
    csv_path: Optional[str] = None
    equity: float = 1000.0                # Account equity for position sizing
    risk_per_trade: float = 0.01          # 1%
    min_rr: float = 3.0                   # Target R:R
    enable_trading: bool = False          # Place orders via MT5
    magic: int = 424242

# Map timeframe string to MT5 constants
MT5_TF_MAP: Dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}

# -----------------------
# Data Access
# -----------------------

def fetch_mt5(symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 module not available. Install `MetaTrader5` and run with --source mt5.")

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    # Select symbol
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Symbol {symbol} not found/visible in MT5")

    tf_min = MT5_TF_MAP.get(timeframe, 5)
    rates = mt5.copy_rates_from_pos(symbol, tf_min, 0, bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError("No rates returned from MT5")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.rename(columns={
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "tick_volume": "tick_volume",
        "real_volume": "real_volume",
        "spread": "spread",
    }, inplace=True)
    return df[["time","open","high","low","close","tick_volume","spread","real_volume"]].reset_index(drop=True)


def fetch_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    else:
        raise ValueError("CSV must contain a 'time' column in ISO or epoch milliseconds/seconds.")
    for c in ["open","high","low","close"]:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")
    # Fill optional columns
    for c in ["tick_volume","spread","real_volume"]:
        if c not in df.columns:
            df[c] = 0
    df = df.sort_values("time").reset_index(drop=True)
    return df[["time","open","high","low","close","tick_volume","spread","real_volume"]]

# -----------------------
# Features / Signals
# -----------------------

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    h_l = df["high"] - df["low"]
    h_pc = (df["high"] - prev_close).abs()
    l_pc = (df["low"] - prev_close).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.rolling(n).mean()


def rsi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    delta = df["close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up).rolling(n).mean()
    roll_down = pd.Series(down).rolling(n).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))


def body(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


def wick_top(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df[["open","close"]].max(axis=1)

def wick_bottom(df: pd.DataFrame) -> pd.Series:
    return df[["open","close"]].min(axis=1) - df["low"]


def liquidity_sweep(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Detects stop-hunts: makes a new high (or low) vs recent range then closes back inside with long wick."""
    rolling_high = df["high"].rolling(lookback).max().shift(1)
    rolling_low = df["low"].rolling(lookback).min().shift(1)

    new_high_break = df["high"] > rolling_high
    new_low_break = df["low"] < rolling_low

    # Close back inside + long wick condition
    long_upper = wick_top(df) > body(df) * 1.2
    long_lower = wick_bottom(df) > body(df) * 1.2

    sweep_up = new_high_break & (df["close"] < rolling_high) & long_upper
    sweep_down = new_low_break & (df["close"] > rolling_low) & long_lower

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[sweep_up] = -1   # after sweeping highs, bias down
    signal[sweep_down] = 1  # after sweeping lows, bias up
    return signal


def fair_value_gap(df: pd.DataFrame) -> pd.Series:
    """Marks presence of FVG (3-candle imbalance)."""
    # Bullish FVG if low[n] > high[n-2]
    bull_fvg = df["low"].shift(0) > df["high"].shift(2)
    # Bearish FVG if high[n] < low[n-2]
    bear_fvg = df["high"].shift(0) < df["low"].shift(2)

    s = pd.Series(0, index=df.index, dtype=int)
    s[bull_fvg] = 1
    s[bear_fvg] = -1
    return s


def structure_break(df: pd.DataFrame, n: int = 10) -> pd.Series:
    prev_high = df["high"].rolling(n).max().shift(1)
    prev_low = df["low"].rolling(n).min().shift(1)
    bos_up = df["close"] > prev_high
    bos_down = df["close"] < prev_low
    s = pd.Series(0, index=df.index, dtype=int)
    s[bos_up] = 1
    s[bos_down] = -1
    return s


def buy_sell_pressure(df: pd.DataFrame) -> pd.Series:
    """Proxy: candle body * tick_volume direction. +ve = buy pressure, -ve = sell pressure."""
    direction = np.sign(df["close"] - df["open"]).replace(0, method="ffill").fillna(0)
    press = direction * df["tick_volume"] * (body(df) + 1e-9)
    # Normalize to z-score
    z = (press - press.rolling(100).mean()) / (press.rolling(100).std() + 1e-9)
    return z.fillna(0)

# -----------------------
# Probability Model (simple weighted fusion)
# -----------------------

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def probability_score(df: pd.DataFrame) -> pd.DataFrame:
    ls = liquidity_sweep(df)          # +1 bull, -1 bear
    fvg = fair_value_gap(df)          # +1 bull, -1 bear
    bos = structure_break(df)         # +1 bull, -1 bear
    press = buy_sell_pressure(df)     # continuous
    atr14 = atr(df)
    rsi14 = rsi(df)

    # Combine with weights (tuneable)
    w_ls, w_fvg, w_bos, w_press, w_rsi = 1.2, 0.8, 1.0, 0.6, 0.4
    x = (w_ls*ls) + (w_fvg*fvg) + (w_bos*bos) + (w_press*np.tanh(press)) + (w_rsi*((50 - rsi14)/25.0))
    # Map to probability of BUY (0..1)
    p_buy = sigmoid(x.clip(-10, 10))
    p_sell = 1.0 - p_buy

    out = df.copy()
    out["p_buy"] = p_buy
    out["p_sell"] = p_sell
    out["signal"] = np.where(p_buy > 0.6, 1, np.where(p_sell > 0.6, -1, 0))
    out["atr14"] = atr14
    return out

# -----------------------
# Trade Plan: SL/TP, Position Size
# -----------------------

def recent_swing_levels(df: pd.DataFrame, lookback: int = 10) -> Tuple[pd.Series, pd.Series]:
    swing_high = df["high"].rolling(lookback).max().shift(1)
    swing_low = df["low"].rolling(lookback).min().shift(1)
    return swing_high, swing_low


def build_trade_plan(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = probability_score(df)
    sh, sl = recent_swing_levels(out)

    # Pip value approximation for gold (point value = 0.01 for many brokers). Use broker specs for exact.
    point = 0.01

    plans = []
    for i in range(len(out)):
        sig = out.at[i, "signal"]
        if sig == 0:
            plans.append({"entry": np.nan, "sl": np.nan, "tp": np.nan, "qty": 0.0, "rr": np.nan})
            continue

        price = out.at[i, "close"]
        atr_val = max(out.at[i, "atr14"], point*10)
        if sig == 1:  # BUY
            sl_price = min(sl.iloc[i], price - 1.5*atr_val)
            risk_per_lot = (price - sl_price)
            tp_price = price + cfg.min_rr * risk_per_lot
        else:         # SELL
            sl_price = max(sh.iloc[i], price + 1.5*atr_val)
            risk_per_lot = (sl_price - price)
            tp_price = price - cfg.min_rr * risk_per_lot

        if risk_per_lot <= 0 or math.isnan(risk_per_lot):
            plans.append({"entry": np.nan, "sl": np.nan, "tp": np.nan, "qty": 0.0, "rr": np.nan})
            continue

        # Position sizing: risk_in_$ / risk_per_unit. Assume 1 lot = 100 ounces, adjust per broker.
        risk_cash = cfg.equity * cfg.risk_per_trade
        contract_size = 100.0  # ounces per standard lot (typical for many brokers)
        tick_value_per_lot = contract_size * point  # $ per 1 point move (approx)
        units_per_price_move = tick_value_per_lot   # $ per 0.01 price move

        price_move = risk_per_lot / point          # number of points to SL
        risk_per_lot_$ = price_move * units_per_price_move
        lots = max(0.0, round(risk_cash / (risk_per_lot_$ + 1e-9), 2))

        rr = abs((tp_price - price) / (price - sl_price))
        plans.append({
            "entry": price,
            "sl": sl_price,
            "tp": tp_price,
            "qty": lots,
            "rr": rr,
        })

    plan_df = pd.DataFrame(plans)
    out = pd.concat([out, plan_df], axis=1)
    return out

# -----------------------
# MT5 Execution (optional)
# -----------------------

def place_mt5_order(symbol: str, side: int, qty_lots: float, entry: float, sl: float, tp: float, magic: int) -> bool:
    """side: +1 buy, -1 sell"""
    if mt5 is None:
        print("MT5 trading not available in this environment.")
        return False
    order_type = mt5.ORDER_TYPE_BUY if side == 1 else mt5.ORDER_TYPE_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": max(qty_lots, 0.01),
        "type": order_type,
        "price": entry,
        "sl": sl,
        "tp": tp,
        "deviation": 50,
        "magic": magic,
        "comment": "XAUUSD_LIQUIDITY_AI",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    result = mt5.order_send(request)
    print("MT5 order result:", result)
    return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE

# -----------------------
# Main
# -----------------------

def run(cfg: Config):
    if cfg.source == "mt5":
        df = fetch_mt5(cfg.symbol, cfg.timeframe, cfg.bars)
    elif cfg.source == "csv":
        if not cfg.csv_path:
            raise ValueError("--csv_path is required for source=csv")
        df = fetch_csv(cfg.csv_path)
    else:
        raise ValueError("Unknown source. Use mt5 or csv")

    df = df.dropna().reset_index(drop=True)
    out = build_trade_plan(df, cfg)

    last = out.iloc[-1]
    side = int(last["signal"])  # 1 buy, -1 sell, 0 flat

    print("\n===== XAUUSD Liquidity AI — Latest Signal =====")
    print(f"Time (UTC): {last['time']}")
    if side == 0:
        print("Signal: FLAT / NO-TRADE (probabilities not decisive)")
        print(f"P(BUY)={last['p_buy']:.2%}  P(SELL)={last['p_sell']:.2%}")
    else:
        side_txt = "BUY" if side == 1 else "SELL"
        print(f"Signal: {side_txt}")
        print(f"P(BUY)={last['p_buy']:.2%}  P(SELL)={last['p_sell']:.2%}")
        print(f"Entry={last['entry']:.2f}  SL={last['sl']:.2f}  TP={last['tp']:.2f}  Lots={last['qty']:.2f}  R:R={last['rr']:.2f}")

        if cfg.enable_trading and side != 0:
            ok = place_mt5_order(cfg.symbol, side, float(last['qty']), float(last['entry']), float(last['sl']), float(last['tp']), cfg.magic)
            print("Order placed:", ok)

    # Save full table
    out.to_csv("xauusd_liquidity_ai_output.csv", index=False)
    print("Saved: xauusd_liquidity_ai_output.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="mt5", choices=["mt5","csv"])
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--tf", dest="timeframe", default="M5")
    p.add_argument("--bars", type=int, default=1500)
    p.add_argument("--csv_path", type=str, default=None)
    p.add_argument("--equity", type=float, default=1000.0)
    p.add_argument("--risk", dest="risk_per_trade", type=float, default=0.01)
    p.add_argument("--min_rr", type=float, default=3.0)
    p.add_argument("--enable_trading", action="store_true")
    args = p.parse_args()

    cfg = Config(
        source=args.source,
        symbol=args.symbol,
        timeframe=args.timeframe,
        bars=args.bars,
        csv_path=args.csv_path,
        equity=args.equity,
        risk_per_trade=args.risk_per_trade,
        min_rr=args.min_rr,
        enable_trading=args.enable_trading,
    )
    run(cfg)
