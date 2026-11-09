#!/usr/bin/env python3
"""Technical + context analysis for the stock tracker dashboard."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo

    IST = ZoneInfo("Asia/Kolkata")
except Exception:  # no tzdata (e.g. Windows) -> fixed offset, India has no DST
    IST = timezone(timedelta(hours=5, minutes=30))

MIN_BARS = 80  # minimum to analyse at all
FULL_HISTORY_BARS = 220  # needed for SMA200 + its slope
BENCHMARK = "^NSEI"

PIVOT_ORDER = 3  # bars either side to confirm a swing point
PIVOT_LOOKBACK = 120  # ignore pivots older than this
RECENT_PIVOT_BARS = 45  # latest pivot must be this fresh to count as "current"
SUPPORT_LOOKBACK = 250

OHLCV_FIELDS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}


# --------------------------------------------------------------------------
# Ticker helpers
# --------------------------------------------------------------------------
def resolve_ticker(raw: str) -> str:
    t = (raw or "").strip().upper().replace(" ", "")
    if t.startswith("$"):
        t = t[1:]
    if not t:
        raise ValueError("Empty ticker")
    return t


def nse_candidate(ticker: str) -> str:
    """Append .NS for bare NSE symbols; leave indices (^NSEI), FX (X=) and suffixed tickers alone."""
    if "." in ticker or ticker.startswith("^") or "=" in ticker:
        return ticker
    return f"{ticker}.NS"


# --------------------------------------------------------------------------
# Data cleaning
# --------------------------------------------------------------------------
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten a yfinance MultiIndex frame. Only safe for a single ticker."""
    if df is None or df.empty or not isinstance(df.columns, pd.MultiIndex):
        return df

    field_level = None
    for lvl in range(df.columns.nlevels):
        if set(df.columns.get_level_values(lvl)) & OHLCV_FIELDS:
            field_level = lvl
            break
    if field_level is None:
        return df

    for lvl in range(df.columns.nlevels):
        if lvl == field_level:
            continue
        others = {str(x) for x in df.columns.get_level_values(lvl) if str(x) != ""}
        if len(others) > 1:
            raise ValueError(
                "Multi-ticker frame passed to flatten_columns; analyse one ticker at a time"
            )

    result = df.copy()
    result.columns = df.columns.get_level_values(field_level)
    return result.loc[:, ~result.columns.duplicated()]


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = flatten_columns(df.copy())
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()

    df = df[required].copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"]).sort_index()
    df = df[~df.index.duplicated(keep="last")]  # yfinance sometimes repeats the last bar

    # Drop today's bar while the NSE session is still open (it is incomplete).
    if len(df):
        now = datetime.now(IST)
        last_ts = pd.Timestamp(df.index[-1])
        if last_ts.tzinfo is not None:
            last_ts = last_ts.tz_convert(IST)
        if last_ts.date() == now.date() and (now.hour, now.minute) < (15, 35):
            df = df.iloc[:-1]
    return df


def _wilder(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _safe(val, default=None):
    try:
        if val is None:
            return default
        x = float(val)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# Pivots (vectorised, confirmed only)
# --------------------------------------------------------------------------
def _pivot_idx(series: pd.Series, order: int = PIVOT_ORDER, mode: str = "low") -> np.ndarray:
    """Positional indices of confirmed swing lows/highs.

    A centred rolling window is NaN for the last `order` bars, so unconfirmed
    pivots at the right edge are excluded automatically.
    """
    win = 2 * order + 1
    roll = series.rolling(win, center=True)
    ext = roll.min() if mode == "low" else roll.max()
    mask = (series == ext) & ext.notna()
    return np.flatnonzero(mask.to_numpy())


def swing_points(series: pd.Series, order: int = 3) -> list[int]:
    """Backward-compatible helper: all confirmed swing highs and lows, sorted."""
    idx = np.union1d(_pivot_idx(series, order, "low"), _pivot_idx(series, order, "high"))
    return [int(i) for i in idx]


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    df["SMA20"] = c.rolling(20).mean()
    df["SMA50"] = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()
    df["EMA12"] = c.ewm(span=12, adjust=False).mean()
    df["EMA26"] = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # RSI: no losses -> 100 (not NaN); no movement at all -> 50.
    delta = c.diff()
    avg_gain = _wilder(delta.clip(lower=0), 14)
    avg_loss = _wilder((-delta).clip(lower=0), 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    df["RSI"] = rsi

    tr = pd.concat(
        [(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()],
        axis=1,
    ).max(axis=1)
    df["ATR"] = _wilder(tr, 14)
    df["ATR_pct"] = df["ATR"] / c * 100

    up = h.diff().clip(lower=0)
    down = (-l.diff()).clip(lower=0)
    plus_dm = up.where(up > down, 0.0)
    minus_dm = down.where(down > up, 0.0)
    atr14 = df["ATR"].replace(0, np.nan)
    plus_di = 100 * _wilder(plus_dm, 14) / atr14
    minus_di = 100 * _wilder(minus_dm, 14) / atr14
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    df["ADX"] = _wilder(dx, 14)
    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di

    mid = df["SMA20"]
    std = c.rolling(20).std()
    df["BB_upper"] = mid + 2 * std
    df["BB_lower"] = mid - 2 * std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / mid.replace(0, np.nan)
    df["BB_pct"] = (c - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)

    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    df["STOCH_K"] = (c - low14) / (high14 - low14).replace(0, np.nan) * 100
    df["STOCH_D"] = df["STOCH_K"].rolling(3).mean()

    # Relative volume vs the *prior* 20 sessions (current bar excluded).
    df["Vol_SMA20"] = v.rolling(20).mean()
    df["Rel_Vol"] = v / df["Vol_SMA20"].shift(1).replace(0, np.nan)
    obv = np.sign(c.diff().fillna(0)) * v
    df["OBV"] = obv.cumsum()
    df["OBV_slope"] = df["OBV"].diff(10)  # OBV is signed; pct_change is meaningless

    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    mfv = mfm.fillna(0) * v
    df["CMF"] = mfv.rolling(21).sum() / v.rolling(21).sum().replace(0, np.nan)

    df["ROC_20"] = c.pct_change(20) * 100
    df["ROC_60"] = c.pct_change(60) * 100
    roll_max_60 = c.rolling(60, min_periods=20).max()
    roll_max_252 = c.rolling(252, min_periods=60).max()
    roll_min_252 = c.rolling(252, min_periods=60).min()
    df["DD_20"] = (c / c.rolling(20, min_periods=5).max() - 1) * 100
    df["DD_60"] = (c / roll_max_60 - 1) * 100
    df["Pos_52w"] = (c - roll_min_252) / (roll_max_252 - roll_min_252).replace(0, np.nan)

    df["SMA20_slope"] = df["SMA20"].pct_change(10) * 100
    df["SMA50_slope"] = df["SMA50"].pct_change(10) * 100
    df["SMA200_slope"] = df["SMA200"].pct_change(20) * 100
    return df


# --------------------------------------------------------------------------
# Structure, support, fibonacci
# --------------------------------------------------------------------------
def _fib_zone(
    price: float,
    swing_low: float,
    swing_high: float,
    low_idx: Optional[int] = None,
    high_idx: Optional[int] = None,
) -> dict[str, Any]:
    empty = {"in_zone": False, "level": None, "retracement": None}
    if not swing_high or not swing_low or swing_high <= swing_low:
        return empty
    # A pullback retracement only makes sense when the low came BEFORE the high.
    if low_idx is not None and high_idx is not None and low_idx >= high_idx:
        return empty

    span = swing_high - swing_low
    retr = (swing_high - price) / span
    levels = {
        "0.382": swing_high - 0.382 * span,
        "0.5": swing_high - 0.5 * span,
        "0.618": swing_high - 0.618 * span,
    }
    in_zone = 0.35 <= retr <= 0.68
    nearest = min(levels.items(), key=lambda kv: abs(price - kv[1]))
    return {
        "in_zone": in_zone,
        "level": nearest[0],
        "retracement": round(float(retr), 3),
        "prices": {k: round(v, 2) for k, v in levels.items()},
    }


def _nearest_support(df: pd.DataFrame, price: float) -> Optional[float]:
    """Highest recent swing low at/below price, or None if price is at fresh lows."""
    n = len(df)
    start = max(0, n - SUPPORT_LOOKBACK)
    lows = df["Low"]
    idx = _pivot_idx(lows, PIVOT_ORDER, "low")
    idx = idx[idx >= start]
    candidates = [float(lows.iloc[i]) for i in idx]
    below = [x for x in candidates if x <= price * 1.01]
    if below:
        return max(below)
    # No pivot below price: fall back to the 20-day low only if it is below price.
    fallback = _safe(lows.rolling(20).min().iloc[-1])
    if fallback is not None and fallback <= price:
        return fallback
    return None


def _structure(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    hv = df["High"].to_numpy()
    lv = df["Low"].to_numpy()
    cutoff = n - PIVOT_LOOKBACK

    hi_idx = _pivot_idx(df["High"], PIVOT_ORDER, "high")
    lo_idx = _pivot_idx(df["Low"], PIVOT_ORDER, "low")
    hi_idx = hi_idx[hi_idx >= cutoff]
    lo_idx = lo_idx[lo_idx >= cutoff]

    recent_high = len(hi_idx) >= 1 and (n - 1 - hi_idx[-1]) <= RECENT_PIVOT_BARS
    recent_low = len(lo_idx) >= 1 and (n - 1 - lo_idx[-1]) <= RECENT_PIVOT_BARS

    hh = len(hi_idx) >= 2 and recent_high and hv[hi_idx[-1]] > hv[hi_idx[-2]]
    lh = len(hi_idx) >= 2 and recent_high and hv[hi_idx[-1]] < hv[hi_idx[-2]]
    hl = len(lo_idx) >= 2 and recent_low and lv[lo_idx[-1]] > lv[lo_idx[-2]]
    ll = len(lo_idx) >= 2 and recent_low and lv[lo_idx[-1]] < lv[lo_idx[-2]]

    rsi_div = False
    if len(lo_idx) >= 2 and recent_low and "RSI" in df.columns:
        i1, i2 = lo_idx[-2], lo_idx[-1]
        r1, r2 = float(df["RSI"].iloc[i1]), float(df["RSI"].iloc[i2])
        rsi_div = bool(
            lv[i2] < lv[i1] and np.isfinite(r1) and np.isfinite(r2) and r2 > r1
        )

    window = 60
    tail = df.iloc[-window:]
    if len(hi_idx):
        high_i = int(hi_idx[-1])
        last_high = float(hv[high_i])
    else:
        high_i = n - len(tail) + int(np.argmax(tail["High"].to_numpy()))
        last_high = float(tail["High"].max())
    if len(lo_idx):
        low_i = int(lo_idx[-1])
        last_low = float(lv[low_i])
    else:
        low_i = n - len(tail) + int(np.argmin(tail["Low"].to_numpy()))
        last_low = float(tail["Low"].min())

    label = "range"
    if hh and hl:
        label = "uptrend"
    elif lh and ll:
        label = "downtrend"
    elif hl and lh:
        label = "contracting"  # higher lows + lower highs: coiling range
    elif ll and hh:
        label = "expanding"  # lower lows + higher highs: widening range

    return {
        "hh": bool(hh),
        "hl": bool(hl),
        "lh": bool(lh),
        "ll": bool(ll),
        "rsi_div": bool(rsi_div),
        "label": label,
        "last_swing_high": last_high,
        "last_swing_low": last_low,
        "high_idx": high_i,
        "low_idx": low_i,
    }


def _today_ist():
    return datetime.now(IST).date()


def _earnings_soon(calendar) -> bool:
    """True if an earnings date falls within the next 10 days (IST)."""
    try:
        if calendar is None:
            return False
        today = _today_ist()
        dates: list = []
        if isinstance(calendar, pd.DataFrame):
            if calendar.empty:
                return False
            if "Earnings Date" in calendar.columns:
                dates = calendar["Earnings Date"].dropna().tolist()
            elif "Earnings Date" in calendar.index:  # older yfinance: row layout
                row = calendar.loc["Earnings Date"]
                dates = row.dropna().tolist() if hasattr(row, "dropna") else [row]
        elif isinstance(calendar, dict):
            d = calendar.get("Earnings Date", [])
            dates = d if isinstance(d, (list, tuple)) else [d]
        for ed in dates:
            try:
                diff = (pd.Timestamp(ed).date() - today).days
                if 0 <= diff <= 10:
                    return True
            except Exception:
                continue
    except Exception:
        return False
    return False


def classify_zone(score: float, setup: str) -> str:
    if setup in ("Falling knife", "Breakdown"):
        return "avoid"
    if setup == "Extended":
        return "extended"
    if score >= 75:
        return "strong_buy_zone"
    if score >= 62:
        return "buy_zone"
    if score >= 50:
        return "watch"
    return "neutral"


# --------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------
def analyze_frame(
    ticker: str,
    df: pd.DataFrame,
    bench: Optional[pd.DataFrame] = None,
    info: Optional[dict] = None,
    calendar=None,
) -> dict[str, Any]:
    df = clean_ohlcv(df)
    if len(df) < MIN_BARS:
        raise ValueError(f"{ticker}: need at least {MIN_BARS} daily bars")
    limited_history = len(df) < FULL_HISTORY_BARS
    df = add_indicators(df)
    info = info or {}
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    price = float(last["Close"])

    rsi = _safe(last["RSI"], 50.0)
    rsi_prev = _safe(prev["RSI"], rsi)
    rsi_prev2 = _safe(prev2["RSI"], rsi_prev)
    macd_hist = _safe(last["MACD_hist"], 0.0)
    macd_hist_prev = _safe(prev["MACD_hist"], 0.0)
    macd_hist_prev2 = _safe(prev2["MACD_hist"], macd_hist_prev)

    struct = _structure(df)
    support = _nearest_support(df, price)
    fib = _fib_zone(
        price,
        struct["last_swing_low"],
        struct["last_swing_high"],
        struct["low_idx"],
        struct["high_idx"],
    )

    sma20 = _safe(last["SMA20"])
    sma50 = _safe(last["SMA50"])
    sma200 = _safe(last["SMA200"])
    has_200 = sma200 is not None
    above_20 = sma20 is not None and price > sma20
    above_50 = sma50 is not None and price > sma50
    above_200 = has_200 and price > sma200
    # With <200 bars, SMA200 is unknown (not "bearish"): fall back to the 50-day.
    long_trend_ok = above_200 if has_200 else above_50
    ma_stack_bull = (
        sma20 is not None
        and sma50 is not None
        and has_200
        and sma20 > sma50 > sma200
    )

    dist_20 = ((price / sma20) - 1) * 100 if sma20 else None
    dist_50 = ((price / sma50) - 1) * 100 if sma50 else None
    dist_200 = ((price / sma200) - 1) * 100 if sma200 else None
    dist_support = ((price / support) - 1) * 100 if support else None

    rel_vol = _safe(last["Rel_Vol"], 1.0)
    cmf = _safe(last["CMF"], 0.0)
    adx = _safe(last["ADX"], 15.0)
    plus_di = _safe(last["PLUS_DI"], 20.0)
    minus_di = _safe(last["MINUS_DI"], 20.0)
    bb_pct = _safe(last["BB_pct"], 0.5)
    bb_width = _safe(last["BB_width"], 0.1)
    bb_width_med = _safe(df["BB_width"].rolling(120).median().iloc[-1], bb_width)
    squeeze = bb_width < 0.75 * bb_width_med
    stoch_k = _safe(last["STOCH_K"], 50.0)
    stoch_d = _safe(last["STOCH_D"], 50.0)
    stoch_prev_k = _safe(prev["STOCH_K"], stoch_k)
    stoch_prev_d = _safe(prev["STOCH_D"], stoch_d)
    roc20 = _safe(last["ROC_20"], 0.0)
    dd60 = _safe(last["DD_60"], 0.0)
    pos52 = _safe(last["Pos_52w"], 0.5)
    atr_pct = _safe(last["ATR_pct"], 2.0)

    # Relative strength vs benchmark; None means "unknown", not "neutral".
    rs_20: Optional[float] = None
    if bench is not None:
        b = clean_ohlcv(bench)
        if len(b) >= 25:
            common = df.index.intersection(b.index)
            if len(common) >= 25:
                s = df.loc[common, "Close"]
                x = b.loc[common, "Close"]
                rs_20 = float((s.iloc[-1] / s.iloc[-21] - 1) - (x.iloc[-1] / x.iloc[-21] - 1)) * 100

    # Two-bar confirmation instead of a single-bar uptick.
    rsi_turning_up = rsi > rsi_prev > rsi_prev2 and min(rsi_prev, rsi_prev2) < 45
    macd_turning_up = macd_hist > macd_hist_prev > macd_hist_prev2 and macd_hist_prev < 0
    stoch_cross_up = stoch_k > stoch_d and stoch_prev_k <= stoch_prev_d

    near_ma = any(d is not None and abs(d) <= 2.5 for d in (dist_20, dist_50, dist_200))
    near_support = dist_support is not None and 0 <= dist_support <= 3.5
    oversold_band = 28 <= rsi <= 48
    not_crushed = rsi >= 28 or struct["hl"] or struct["rsi_div"]

    # ---- sub-scores -------------------------------------------------------
    location = 40
    if near_support:
        location += 22
    if near_ma and long_trend_ok:
        location += 18
    elif near_ma:
        location += 10
    if fib["in_zone"]:
        location += 12
    if bb_pct < 0.25:
        location += 10
    if dist_20 is not None and dist_20 > 8:
        location -= 18
    if pos52 > 0.92:
        location -= 12
    if dist_support is not None and dist_support > 12:
        location -= 8
    location = int(np.clip(location, 0, 100))

    momentum = 40
    if rsi_turning_up:
        momentum += 16
    if macd_turning_up:
        momentum += 14
    if stoch_cross_up and stoch_k < 55:
        momentum += 10
    if struct["rsi_div"]:
        momentum += 16
    if struct["hl"]:
        momentum += 12
    if plus_di > minus_di and adx >= 18:
        momentum += 8
    if rsi < 25 and not struct["hl"]:
        momentum -= 18
    if macd_hist < macd_hist_prev and macd_hist < 0:
        momentum -= 8
    momentum = int(np.clip(momentum, 0, 100))

    trend = 40
    if above_200:
        trend += 18
    if above_50:
        trend += 10
    if ma_stack_bull:
        trend += 14
    if _safe(last["SMA200_slope"], 0) > 0:
        trend += 10
    if rs_20 is not None:
        if rs_20 > 2:
            trend += 10
        elif rs_20 < -4:
            trend -= 12
    if struct["label"] == "uptrend":
        trend += 10
    elif struct["label"] == "downtrend":
        trend -= 16
    if adx < 15:
        trend -= 4
    trend = int(np.clip(trend, 0, 100))

    volume = 45
    if rel_vol < 0.85 and (near_ma or near_support):
        volume += 10  # quiet pullback
    if rel_vol > 1.4 and (rsi_turning_up or macd_turning_up):
        volume += 16  # demand returning
    if cmf > 0.05:
        volume += 10
    elif cmf < -0.08:
        volume -= 10
    if _safe(last["OBV_slope"], 0) > 0:
        volume += 8
    volume = int(np.clip(volume, 0, 100))

    # ---- setup classification --------------------------------------------
    earnings_risk = _earnings_soon(calendar)
    falling_knife = (
        not above_50
        and (not has_200 or not above_200)
        and _safe(last["SMA50_slope"], 0) < -2
        and rsi < 32
        and struct["ll"]
        and not struct["hl"]
    )
    extended = dist_20 is not None and dist_20 > 7 and rsi > 65 and pos52 > 0.85
    quality_pullback = (
        long_trend_ok
        and oversold_band
        and (near_ma or near_support or fib["in_zone"])
        and (struct["hl"] or rsi_turning_up or macd_turning_up)
        and not falling_knife
    )
    momentum_dip = (
        above_50
        and abs(dd60) < 12
        and rs_20 is not None
        and rs_20 > 1
        and (near_ma or oversold_band)
        and not extended
    )
    reversal = (
        (struct["rsi_div"] or (rsi_turning_up and rsi < 40))
        and not long_trend_ok
        and not_crushed
    )
    squeeze_break = squeeze and macd_turning_up and plus_di > minus_di

    setup = "Neutral"
    setup_mult = 1.0
    if falling_knife:
        setup, setup_mult = "Falling knife", 0.45
    elif extended:
        setup, setup_mult = "Extended", 0.55
    elif quality_pullback:
        setup, setup_mult = "Quality pullback", 1.18
    elif momentum_dip:
        setup, setup_mult = "Momentum dip", 1.08
    elif squeeze_break:
        setup, setup_mult = "Squeeze expansion", 1.05
    elif reversal:
        setup, setup_mult = "Reversal attempt", 0.92
    elif struct["label"] == "downtrend" and rsi < 40:
        setup, setup_mult = "Breakdown", 0.5

    risk_adj = 1.0
    if earnings_risk:
        risk_adj *= 0.9
    if atr_pct > 5:
        risk_adj *= 0.95

    raw = location * 0.32 + momentum * 0.28 + trend * 0.22 + volume * 0.18
    score = round(float(np.clip(raw * setup_mult * risk_adj, 0, 100)), 1)
    zone = classify_zone(score, setup)

    # ---- signals ------------------------------------------------------------
    signals = []
    if quality_pullback:
        signals.append("Healthy trend, price back into a demand area")
    if momentum_dip:
        signals.append("Shallow dip while relative strength stays positive")
    if rsi_turning_up:
        signals.append(f"RSI rising two bars running ({rsi_prev2:.0f} -> {rsi:.0f})")
    if macd_turning_up:
        signals.append("MACD histogram improving from below zero")
    if struct["hl"]:
        signals.append("Higher swing low in place")
    if struct["rsi_div"]:
        signals.append("Bullish RSI divergence at the last two swing lows")
    if fib["in_zone"]:
        signals.append(f"Trading in the {fib['level']} Fibonacci retracement band")
    if near_support:
        signals.append(f"Within {dist_support:.1f}% of swing support {support:.2f}")
    if near_ma and long_trend_ok:
        signals.append("Holding the moving-average cluster in a healthy trend")
    if squeeze:
        signals.append("Bollinger squeeze: volatility compressed, move likely soon")
    if rs_20 is not None and rs_20 > 2:
        signals.append(f"Outperforming Nifty by {rs_20:.1f} pts over 20 sessions")
    if cmf > 0.05:
        signals.append("Chaikin money flow positive (buying pressure)")
    if earnings_risk:
        signals.append("Earnings within 10 days - size down or wait")
    if falling_knife:
        signals.append("Declining MAs, lower lows, weak RSI - not a buy zone")
    if extended:
        signals.append("Stretched above the 20-day and near the top of its 52-week range - chase risk")
    if limited_history:
        signals.append(f"Under {FULL_HISTORY_BARS} bars of history: 200-day average unavailable")
    if not signals:
        signals.append("No strong edge: wait for a test of support or a momentum turn")

    # ---- buy band / invalidation / risk-reward ----------------------------
    buy_from = buy_to = invalidation = None
    if support and zone not in ("avoid", "extended"):
        ceilings = [support * 1.035]
        for ma in (sma20, sma50, sma200):
            if ma and ma >= support and (ma - support) / support <= 0.06:
                ceilings.append(ma * 1.008)
        buy_from = round(support * 0.992, 2)
        buy_to = round(min(ceilings), 2)
        invalidation = round(support * 0.97, 2)

    resistance = struct["last_swing_high"]
    risk_reward = None
    if invalidation is not None and resistance > price > invalidation:
        risk_reward = round((resistance - price) / (price - invalidation), 2)

    change_1d = float(df["Close"].pct_change().iloc[-1] * 100)

    name = info.get("shortName") or info.get("longName") or ticker.replace(".NS", "")
    sector = info.get("sector") or "Unknown"
    pe = _safe(info.get("trailingPE"))
    mcap = _safe(info.get("marketCap"))
    beta = _safe(info.get("beta"))

    chart = _chart_payload(df.tail(180))

    return {
        "ticker": ticker,
        "symbol": ticker.replace(".NS", "").replace(".BO", ""),
        "name": name,
        "sector": sector,
        "price": round(price, 2),
        "change_1d": round(change_1d, 2),
        "change_20d": round(roc20, 2),
        "buy_score": score,
        "zone": zone,
        "setup": setup,
        "signals": signals[:8],
        "buy_from": buy_from,
        "buy_to": buy_to,
        "invalidation": invalidation,
        "support": round(support, 2) if support else None,
        "resistance": round(resistance, 2),
        "risk_reward": risk_reward,
        "in_buy_band": bool(buy_from and buy_to and buy_from <= price <= buy_to),
        "limited_history": limited_history,
        "scores": {
            "location": location,
            "momentum": momentum,
            "trend": trend,
            "volume": volume,
        },
        "technicals": {
            "rsi": round(rsi, 1),
            "rsi_prev": round(rsi_prev, 1),
            "macd_hist": round(macd_hist, 4),
            "adx": round(adx, 1),
            "plus_di": round(plus_di, 1),
            "minus_di": round(minus_di, 1),
            "stoch_k": round(stoch_k, 1),
            "stoch_d": round(stoch_d, 1),
            "rel_vol": round(rel_vol, 2),
            "cmf": round(cmf, 3),
            "bb_pct": round(bb_pct, 2),
            "atr_pct": round(atr_pct, 2),
            "sma20": round(sma20, 2) if sma20 else None,
            "sma50": round(sma50, 2) if sma50 else None,
            "sma200": round(sma200, 2) if sma200 else None,
            "dist_sma20": round(dist_20, 2) if dist_20 is not None else None,
            "dist_sma50": round(dist_50, 2) if dist_50 is not None else None,
            "dist_sma200": round(dist_200, 2) if dist_200 is not None else None,
            "dd_60": round(dd60, 1),
            "pos_52w": round(pos52 * 100, 1),
            "rs_20": round(rs_20, 2) if rs_20 is not None else None,
            "structure": struct["label"],
            "higher_low": struct["hl"],
            "higher_high": struct["hh"],
            "rsi_divergence": struct["rsi_div"],
            "squeeze": bool(squeeze),
            "above_200dma": bool(above_200),
            "ma_stack_bull": bool(ma_stack_bull),
        },
        "fundamentals": {
            "pe": round(pe, 1) if pe else None,
            "market_cap": mcap,
            "beta": round(beta, 2) if beta else None,
            "earnings_soon": earnings_risk,
        },
        "fib": fib,
        "as_of": str(df.index[-1].date()),
        "chart": chart,
    }


def _chart_payload(df: pd.DataFrame) -> dict[str, Any]:
    dates = [str(i.date()) for i in df.index]

    def series(col):
        if col not in df.columns:
            return []
        return [None if pd.isna(x) else round(float(x), 2) for x in df[col].tolist()]

    return {
        "dates": dates,
        "open": series("Open"),
        "high": series("High"),
        "low": series("Low"),
        "close": series("Close"),
        "volume": [None if pd.isna(x) else int(x) for x in df["Volume"].tolist()],
        "sma20": series("SMA20"),
        "sma50": series("SMA50"),
        "sma200": series("SMA200"),
        "rsi": [None if pd.isna(x) else round(float(x), 1) for x in df["RSI"].tolist()],
        "macd_hist": [
            None if pd.isna(x) else round(float(x), 4) for x in df["MACD_hist"].tolist()
        ],
    }


def empty_error(ticker: str, message: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "symbol": ticker.replace(".NS", ""),
        "error": message,
        "buy_score": None,
        "zone": "error",
    }