#!/usr/bin/env python3
"""
Smallcap Momentum & Volume Scanner  (stateless edition, v3-small)
=================================================================

Small-cap adaptation of midcap_momentum_scanner.py. Finds fundamentally
healthy small caps that have had an unusually large decline where selling
pressure is fading and MOMENTUM + VOLUME are turning up.

What is different from the midcap version
-----------------------------------------
  UNIVERSE
    * Default --universe is smallcap250 (also smallcap100, smallcap50, nifty500).
    * Benchmark is the Nifty Smallcap index (^CNXSC) with fallbacks.
    * --extra "IFCI,HFCL" appends your own names; --tickers-file for a list.

  SMALL-CAP RISK FILTERS (new)
    * Liquidity floor lowered to Rs 2 Cr median daily turnover (--min-adv),
      but a score haircut below Rs 5 Cr and a suggested max position size
      (1% of ADV) is printed, because slippage is the real small-cap cost.
    * Price floor (--min-price, default Rs 20) and market-cap floor
      (--min-mcap, default Rs 500 Cr) to avoid penny/micro-cap junk.
    * Volatility cap (--max-atr-pct, default 7%): ATR/price above this is
      too wild to size sensibly.
    * Circuit-lock filter: stocks with several high==low days in the last
      20 sessions are skipped (you often cannot exit them).
    * Hype-spike guard: huge relative-volume spike + >20% 5-day jump is
      flagged and penalised (classic operator/pump pattern).
    * Stricter fundamentals: lower debt/equity ceiling, loss-making and
      deteriorating names rejected, low insider-holding noted.

  SCALING
    * Drawdown, ROC, relative-strength and MACD scales are widened to
      small-cap volatility. Breakouts need stronger volume confirmation.

Stateless by default: no SQLite, no caches. Pass --export-csv PATH to save.

Usage:
  python smallcap_momentum_scanner.py
  python smallcap_momentum_scanner.py --universe smallcap100
  python smallcap_momentum_scanner.py --extra "IFCI,HFCL,SUZLON"
  python smallcap_momentum_scanner.py --lenient --top 30 --min-score 45
  python smallcap_momentum_scanner.py --min-adv 1 --min-mcap 300
  python smallcap_momentum_scanner.py --export-csv small.csv
  python smallcap_momentum_scanner.py --backtest

DISCLAIMER: heuristics, not validated alpha. Not investment advice. Backtest
numbers are in-sample, suffer survivorship bias, and ignore costs/slippage/
taxes (which are large for small caps).
"""

from __future__ import annotations

import argparse
import io
import warnings
from urllib.parse import unquote
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:
    IST = None

warnings.filterwarnings("ignore")


# =============================================================================
# CONSTANTS
# =============================================================================

NSE_CSV_BASE = "https://www.niftyindices.com/IndexConstituent/"

# name -> (index-constituent CSV, human label, benchmark ticker for Yahoo Finance)
UNIVERSES = {
    "smallcap250": ("ind_niftysmallcap250list.csv", "Nifty Smallcap 250", "^CNXSC"),
    "smallcap100": ("ind_niftysmallcap100list.csv", "Nifty Smallcap 100", "^CNXSC"),
    "smallcap50":  ("ind_niftysmallcap50list.csv",  "Nifty Smallcap 50",  "^CNXSC"),
    "nifty500":    ("ind_nifty500list.csv",         "Nifty 500",          "^CRSLDX"),
}

# Benchmark fallbacks, tried in order (all resolve on Yahoo Finance).
INDEX_TICKERS = ["^CNXSC", "^CRSLDX", "^NSEI"]

MIN_BARS = 200
HISTORY_DAYS = 1100

# -----------------------------------------------------------------------------
# Fallback snapshot ONLY if the live NSE download fails. Symbols that have been
# renamed/delisted simply fail the history check and are skipped.
# -----------------------------------------------------------------------------
SMALLCAP_TICKERS = [
    "AARTIIND", "ACE", "AETHER", "AHLUCONT", "AMBER", "ANANTRAJ", "ANURAS",
    "APARINDS", "ARVINDFASN", "ASTERDM", "ATUL", "BALRAMCHIN", "BASF", "BBTC",
    "BEML", "BIKAJI", "BIRLACORPN", "BLS", "BORORENEW", "BRIGADE", "CAMPUS",
    "CAPLIPOINT", "CASTROLIND", "CEATLTD", "CENTURYPLY", "CERA", "CHALET",
    "CHOLAHLDNG", "CLEAN", "CRAFTSMAN", "CREDITACC", "CUB", "DATAPATTNS",
    "DCMSHRIRAM", "DELHIVERY", "DEEPAKFERT", "EASEMYTRIP", "ECLERX",
    "EIDPARRY", "EIHOTEL", "ELECON", "ENGINERSIN", "EQUITASBNK", "FINCABLES",
    "FINPIPE", "FSL", "GESHIP", "GMDCLTD", "GNFC", "GPIL", "GRSE", "GSFC",
    "GSPL", "HBLENGINE", "HCG", "HFCL", "HGINFRA", "HINDCOPPER", "HSCL",
    "IFCI", "IIFL", "INOXWIND", "INTELLECT", "IONEXCHANG", "JBMA",
    "JKLAKSHMI", "JKTYRE", "JMFINANCIL", "JPPOWER", "JUSTDIAL", "JWL",
    "KALPATPOWR", "KFINTECH", "KIRLOSENG", "KNRCON", "KPIL", "LEMONTREE",
    "LXCHEM", "MAHSEAMLES", "MANAPPURAM", "MASTEK", "MEDPLUS", "MGL",
    "MINDACORP", "MOIL", "MRPL", "NATCOPHARM", "NAZARA", "NETWEB", "NEWGEN",
    "NOCIL", "OLECTRA", "ORIENTCEM", "PCBL", "PGEL", "PNCINFRA", "PPLPHARMA",
    "PRAJIND", "PRINCEPIPE", "PRSMJOHNSN", "RAILTEL", "RAIN", "RALLIS",
    "RATNAMANI", "RCF", "REDINGTON", "ROUTE", "RPOWER", "RTNINDIA", "SANOFI",
    "SAPPHIRE", "SHAREINDIA", "SHILPAMED", "SHOPERSTOP", "SKFINDIA", "SOBHA",
    "SOUTHBANK", "SPARC", "STLTECH", "SUNTECK", "SUVENPHAR", "SWANENERGY",
    "TANLA", "TEJASNET", "TEXRAIL", "TMB", "TRITURBINE", "TVSSCS",
    "UJJIVANSFB", "USHAMART", "VAIBHAVGBL", "VIPIND", "VTL", "WELCORP",
    "WELSPUNLIV", "WOCKPHARMA", "ZEEL", "ZENSARTECH", "ZENTEC",
    "AAVAS", "AEGISLOG", "AFFLE", "ALKYLAMINE", "ANGELONE", "APTUS",
    "BSOFT", "CAMS", "CANFINHOME", "CDSL", "CESC", "CHAMBLFERT", "CYIENT",
    "FINEORG", "GRANULES", "HAPPSTMNDS", "HOMEFIRST", "IEX", "IRB", "KEC",
    "KRBL", "LATENTVIEW", "MAPMYINDIA", "NBCC", "NCC", "RBLBANK", "RITES",
    "SAREGAMA", "TATATECH", "TITAGARH", "TRIDENT", "UCOBANK", "CENTRALBK",
]


# =============================================================================
# HELPERS
# =============================================================================

def cprint(text: str, color: str = "white"):
    if HAS_COLOR:
        colors = {
            "red": Fore.RED, "green": Fore.GREEN, "yellow": Fore.YELLOW,
            "cyan": Fore.CYAN, "white": Fore.WHITE, "magenta": Fore.MAGENTA,
        }
        print(f"{colors.get(color, '')}{text}{Style.RESET_ALL}")
    else:
        print(text)


def safe_float(val, default=0.0):
    try:
        if val is None:
            return default
        x = float(val)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def optional_float(val):
    try:
        if val is None:
            return None
        x = float(val)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def optional_pct(val):
    x = optional_float(val)
    return "N/A" if x is None else f"{x:.2%}"


def scale01(value, lo, hi) -> float:
    """Linearly map a value into 0..100, clamped. 50 when undefined."""
    x = optional_float(value)
    if x is None or hi == lo:
        return 50.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0) * 100.0)


def flatten_yfinance_columns(df):
    """Convert Yahoo MultiIndex columns into plain Open/High/Low/Close/Volume."""
    if df is None or df.empty:
        return df
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    new_columns = []
    for col in df.columns:
        if not isinstance(col, tuple):
            new_columns.append(col)
            continue
        selected = next((item for item in col if item in fields), None)
        if selected is None:
            selected = "_".join(str(x) for x in col if str(x) != "")
        new_columns.append(selected)

    result = df.copy()
    result.columns = new_columns
    return result.loc[:, ~result.columns.duplicated()]


def extract_ticker_dataframe(all_data, ticker):
    if all_data is None or all_data.empty:
        return pd.DataFrame()
    if not isinstance(all_data.columns, pd.MultiIndex):
        return flatten_yfinance_columns(all_data.copy())

    for level in (None, 0, 1):
        try:
            if level is None:
                df = all_data[ticker].copy()
            else:
                df = all_data.xs(ticker, axis=1, level=level).copy()
            return flatten_yfinance_columns(df)
        except Exception:
            continue
    return pd.DataFrame()


def clean_price_dataframe(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = flatten_yfinance_columns(df.copy())
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()
    df = df[required].copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Close"]).sort_index()


# =============================================================================
# UNIVERSE / TICKERS
# =============================================================================

def normalize_nse_ticker(symbol: object) -> str:
    """Return a Yahoo NSE ticker while preserving valid NSE punctuation
    (e.g. M&MFIN keeps its ampersand; yfinance encodes it on request)."""
    symbol = unquote(str(symbol)).strip().upper()
    if not symbol:
        return ""
    return symbol if symbol.endswith(".NS") else f"{symbol}.NS"


def parse_nse_tickers(raw_tickers) -> list[str]:
    return sorted({t for raw in raw_tickers if (t := normalize_nse_ticker(raw))})


def fetch_index_constituents(csv_name: str, label: str, timeout: int = 15) -> list[str]:
    """Download an NSE index constituent list; raise on failure."""
    import requests

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/151.0 Safari/537.36"
        )
    }
    resp = requests.get(NSE_CSV_BASE + csv_name, headers=headers, timeout=timeout)
    resp.raise_for_status()
    table = pd.read_csv(io.StringIO(resp.text))

    symbol_col = next((c for c in table.columns if "symbol" in c.lower()), None)
    if symbol_col is None:
        raise ValueError("No symbol column found")

    tickers = parse_nse_tickers(table[symbol_col].dropna())
    if len(tickers) < 20:
        raise ValueError(f"Too few constituents ({len(tickers)})")

    cprint(f"✅ Loaded {len(tickers)} live {label} constituents.", "green")
    return tickers


def fallback_tickers() -> list[str]:
    return parse_nse_tickers(SMALLCAP_TICKERS)


def load_tickers(args) -> list[str]:
    if args.tickers_file:
        raw = Path(args.tickers_file).read_text().splitlines()
        tickers = parse_nse_tickers(l for l in raw if l.strip() and not l.lstrip().startswith("#"))
        cprint(f"✅ Loaded {len(tickers)} tickers from {args.tickers_file}.", "green")
    else:
        csv_name, label, _ = UNIVERSES[args.universe]
        try:
            tickers = fetch_index_constituents(csv_name, label)
        except Exception as exc:
            cprint(f"⚠️ Live fetch failed ({exc}); using built-in fallback list.", "yellow")
            tickers = fallback_tickers()

    if args.extra:
        extra = parse_nse_tickers(args.extra.split(","))
        tickers = sorted(set(tickers) | set(extra))
        cprint(f"➕ Added {len(extra)} extra tickers from --extra.", "green")

    return sorted(set(tickers))


# =============================================================================
# TECHNICAL ANALYSIS  (momentum + volume heavy)
# =============================================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_price_dataframe(df)
    if len(df) < MIN_BARS:
        return df

    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # ---- Moving averages -------------------------------------------------
    for n in (20, 50, 150, 200):
        df[f"SMA{n}"] = c.rolling(n).mean()
    df["EMA21"] = c.ewm(span=21, adjust=False).mean()

    df["SMA20_slope"] = df["SMA20"].pct_change(10) * 100
    df["SMA50_slope"] = df["SMA50"].pct_change(10) * 100
    df["SMA200_slope"] = df["SMA200"].pct_change(20) * 100

    # ---- RSI (Wilder) ----------------------------------------------------
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ---- ATR -------------------------------------------------------------
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    df["ATR_pct"] = df["ATR"] / c * 100

    # ---- MACD (momentum) -------------------------------------------------
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_sig"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_sig"]

    # ---- ADX / DMI (trend strength + direction) --------------------------
    up = h.diff()
    dn = -l.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    atr_w = df["ATR"].replace(0, np.nan)
    df["plus_DI"] = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr_w
    df["minus_DI"] = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr_w
    di_sum = (df["plus_DI"] + df["minus_DI"]).replace(0, np.nan)
    dx = 100 * (df["plus_DI"] - df["minus_DI"]).abs() / di_sum
    df["ADX"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    # ---- Stochastic + StochRSI ------------------------------------------
    ll = l.rolling(14).min()
    hh = h.rolling(14).max()
    df["Stoch_K"] = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()
    rmin = df["RSI"].rolling(14).min()
    rmax = df["RSI"].rolling(14).max()
    df["StochRSI"] = (df["RSI"] - rmin) / (rmax - rmin).replace(0, np.nan) * 100

    # ---- Bollinger Bands (%B + squeeze width) ----------------------------
    sd = c.rolling(20).std(ddof=0)
    df["BB_mid"] = df["SMA20"]
    df["BB_up"] = df["SMA20"] + 2 * sd
    df["BB_dn"] = df["SMA20"] - 2 * sd
    df["BB_pctB"] = (c - df["BB_dn"]) / (df["BB_up"] - df["BB_dn"]).replace(0, np.nan)
    df["BB_width"] = (df["BB_up"] - df["BB_dn"]) / df["BB_mid"]
    df["BB_width_pct"] = df["BB_width"].rolling(252, min_periods=60).rank(pct=True)

    # ---- Multi-period rate of change (momentum) --------------------------
    for n in (5, 10, 20, 60, 120):
        df[f"ROC_{n}"] = c.pct_change(n) * 100

    # ---- Drawdowns -------------------------------------------------------
    for d in (5, 10, 20, 30, 60, 252):
        roll_max = c.rolling(d, min_periods=1).max()
        df[f"DD_{d}"] = (c / roll_max - 1) * 100

    # ---- 52-week position ------------------------------------------------
    hi252 = h.rolling(252, min_periods=60).max()
    lo252 = l.rolling(252, min_periods=60).min()
    df["Dist_52wH"] = (c / hi252 - 1) * 100
    df["Dist_52wL"] = (c / lo252 - 1) * 100

    # ---- Volume ----------------------------------------------------------
    df["Vol_SMA20"] = v.rolling(20).mean()
    df["Vol_SMA60"] = v.rolling(60).mean()
    df["Rel_Vol"] = v / df["Vol_SMA20"].replace(0, np.nan)
    df["Vol_Surge"] = v / df["Vol_SMA60"].replace(0, np.nan)
    df["Vol_DryUp"] = df["Vol_SMA20"] / df["Vol_SMA60"].replace(0, np.nan)

    # ---- OBV (cumulative signed volume) ----------------------------------
    df["OBV"] = (np.sign(c.diff()).fillna(0) * v).cumsum()
    obv_vol = v.rolling(20).sum().replace(0, np.nan)
    df["OBV_slope20"] = (df["OBV"] - df["OBV"].shift(20)) / obv_vol

    # ---- Chaikin Money Flow / A-D line -----------------------------------
    rng = (h - l).replace(0, np.nan)
    mfm = ((c - l) - (h - c)) / rng
    df["AD_line"] = (mfm.fillna(0) * v).cumsum()
    df["CMF"] = (mfm * v).rolling(21).sum() / v.rolling(21).sum().replace(0, np.nan)

    # ---- Money Flow Index (volume-weighted RSI) --------------------------
    tp = (h + l + c) / 3
    rmf = tp * v
    pos_mf = rmf.where(tp > tp.shift(1), 0.0).rolling(14).sum()
    neg_mf = rmf.where(tp < tp.shift(1), 0.0).rolling(14).sum()
    df["MFI"] = 100 - 100 / (1 + pos_mf / neg_mf.replace(0, np.nan))

    # ---- Donchian channels (breakout reference) --------------------------
    df["DC_high20"] = h.rolling(20).max().shift(1)
    df["DC_high55"] = h.rolling(55).max().shift(1)
    df["DC_low20"] = l.rolling(20).min().shift(1)

    # ---- Accumulation vs distribution days -------------------------------
    up_bar = c.diff() > 0
    dn_bar = c.diff() < 0
    vol_higher = v > v.shift(1)
    df["Acc_days25"] = (up_bar & vol_higher).astype(float).rolling(25).sum()
    df["Dist_days25"] = (dn_bar & vol_higher).astype(float).rolling(25).sum()
    up_vol = v.where(up_bar, 0.0).rolling(25).sum()
    dn_vol = v.where(dn_bar, 0.0).rolling(25).sum()
    df["UpDnVol25"] = up_vol / dn_vol.replace(0, np.nan)

    # ---- Pocket pivot: up bar on more volume than any down bar (last 10) --
    prior_down_vol = v.where(dn_bar, np.nan).rolling(10).max().shift(1)
    df["PocketPivot"] = (up_bar & (v > prior_down_vol)).astype(int)

    # ---- Small-cap risk helpers ------------------------------------------
    # Days where high == low (typically a circuit-locked, untradeable session)
    df["Lock_day"] = ((h == l) & (h > 0)).astype(float)
    df["Lock_days20"] = df["Lock_day"].rolling(20).sum()
    df["Turnover"] = c * v   # Rs traded per day

    return df


def find_swing_points(df: pd.DataFrame, order: int = 3):
    highs, lows = [], []
    hi = df["High"].values
    lo = df["Low"].values
    idx = df.index

    for i in range(order, len(df) - order):
        h = hi[i]
        if all(h >= hi[i - j] for j in range(1, order + 1)) and all(
            h >= hi[i + j] for j in range(1, order + 1)
        ):
            highs.append((i, h, idx[i]))

        l = lo[i]
        if all(l <= lo[i - j] for j in range(1, order + 1)) and all(
            l <= lo[i + j] for j in range(1, order + 1)
        ):
            lows.append((i, l, idx[i]))

    return highs, lows


def _relative_return(df, index_df, common_idx, n):
    stock = (df.loc[common_idx[-1], "Close"] / df.loc[common_idx[-n], "Close"] - 1) * 100
    idx = (index_df.loc[common_idx[-1], "Close"] / index_df.loc[common_idx[-n], "Close"] - 1) * 100
    return stock - idx


def _momentum_score(pos: dict) -> tuple[float, dict]:
    """Blend sub-scores (0-100) into one momentum score. Scales are widened
    for small-cap volatility."""
    parts = {
        "20d ROC": scale01(pos["roc_20"], -18, 18),
        "60d ROC": scale01(pos["roc_60"], -35, 35),
        "RS vs index (20d)": scale01(pos["rs_20"], -12, 12),
        "MACD momentum": scale01(pos["macd_hist_pct"], -3.0, 3.0),
        "Trend strength (ADX)": (
            scale01(pos["adx"], 0, 40) if pos["dmi_bull"] else 100 - scale01(pos["adx"], 0, 40)
        ),
        "Position vs 52w high": scale01(pos["dist_52wH"], -50, 0),
    }
    score = float(np.mean(list(parts.values())))

    if pos["macd_bull"] and pos["macd_hist_rising"]:
        score = min(100.0, score + 4)
    if pos["adx"] >= 25 and pos["dmi_bull"]:
        score = min(100.0, score + 4)
    if pos["stoch_k"] is not None and pos["stoch_k"] > 80:
        score = min(100.0, score + 2)

    return round(score, 1), parts


def _volume_score(pos: dict) -> tuple[float, dict]:
    """Blend participation/accumulation signals into one volume score."""
    parts = {
        "Chaikin Money Flow": scale01(pos["cmf"], -0.25, 0.25),
        "OBV trend (20d)": scale01(pos["obv_slope20"], -0.30, 0.30),
        "Up/down volume": scale01(pos["up_dn_vol"], 0.5, 2.0),
        "Relative volume (5d)": scale01(pos["rel_vol5"], 0.7, 2.5),
        "Money Flow Index": scale01(pos["mfi"], 20, 80),
    }
    score = float(np.mean(list(parts.values())))

    acc_bal = (pos["acc_days25"] or 0) - (pos["dist_days25"] or 0)
    score = float(np.clip(score + np.clip(acc_bal, -10, 10) * 1.5, 0, 100))
    if pos["pocket_pivot"]:
        score = min(100.0, score + 5)

    return round(score, 1), parts


def analyze_technicals(df: pd.DataFrame, index_df: pd.DataFrame = None):
    if len(df) < MIN_BARS:
        return None

    df = calculate_indicators(df)
    if df.empty or "RSI" not in df.columns:
        return None

    highs, lows = find_swing_points(df)
    last_i = len(df) - 1
    last = df.iloc[-1]

    curr_price = safe_float(last["Close"])
    curr_rsi = safe_float(last["RSI"], 50.0)
    curr_atr = safe_float(last["ATR"], curr_price * 0.02)

    conf_lows = [x for x in lows if x[0] <= last_i - 3]
    conf_highs = [x for x in highs if x[0] <= last_i - 3]

    hl = len(conf_lows) >= 2 and conf_lows[-1][1] > conf_lows[-2][1]
    hh = len(conf_highs) >= 2 and conf_highs[-1][1] > conf_highs[-2][1]

    # RSI bullish divergence
    rsi_div = False
    if len(conf_lows) >= 2:
        i1, p1, _ = conf_lows[-2]
        i2, p2, _ = conf_lows[-1]
        rsi1 = safe_float(df["RSI"].iloc[i1], 50.0)
        rsi2 = safe_float(df["RSI"].iloc[i2], 50.0)
        rsi_div = p2 < p1 and rsi2 > rsi1

    # Volume near support
    vol_near_support = 0.0
    if conf_lows:
        support = conf_lows[-1][1]
        mask = df["Close"] <= support * 1.05
        total_volume = df["Volume"].sum()
        if total_volume > 0:
            vol_near_support = df.loc[mask, "Volume"].sum() / total_volume

    # Volume contraction (decline-leg volume vs recent volume)
    vol_contract_expand = 0.0
    if conf_highs and conf_lows:
        high_i = conf_highs[-1][0]
        low_i = conf_lows[-1][0]
        if high_i < low_i:
            decline_volume = df["Volume"].iloc[high_i:low_i].mean()
            recent_volume = df["Volume"].iloc[-3:].mean()
            if decline_volume > 0 and recent_volume > 0:
                vol_contract_expand = decline_volume / recent_volume

    # Relative strength vs index
    rs_20 = 0.0
    rs_60 = 0.0
    if index_df is not None and len(index_df) >= 60:
        index_clean = clean_price_dataframe(index_df)
        common_idx = df.index.intersection(index_clean.index)
        if len(common_idx) >= 60:
            rs_20 = _relative_return(df, index_clean, common_idx, 20)
            rs_60 = _relative_return(df, index_clean, common_idx, 60)

    # Volatility-normalised drawdown
    dd_60 = safe_float(last["DD_60"], 0.0)
    atr_pct = (curr_atr / curr_price) * 100 if curr_price > 0 else 0
    vol_norm_dd = abs(dd_60) / atr_pct if atr_pct > 0 else 0

    sma200_last = last["SMA200"]
    sma50_last = last["SMA50"]
    above_200 = bool(curr_price > sma200_last) if not pd.isna(sma200_last) else False
    above_50 = bool(curr_price > sma50_last) if not pd.isna(sma50_last) else False
    above_150 = bool(curr_price > last["SMA150"]) if not pd.isna(last["SMA150"]) else False

    plus_di = safe_float(last["plus_DI"])
    minus_di = safe_float(last["minus_DI"])
    adx = safe_float(last["ADX"])
    dmi_bull = plus_di > minus_di
    macd_bull = safe_float(last["MACD"]) > safe_float(last["MACD_sig"])
    macd_hist = safe_float(last["MACD_hist"])
    macd_hist_rising = macd_hist > safe_float(df["MACD_hist"].iloc[-2])

    rel_vol5 = safe_float(df["Rel_Vol"].iloc[-5:].mean(), 1.0)
    rel_vol_max5 = safe_float(df["Rel_Vol"].iloc[-5:].max(), 1.0)
    vol_surge5 = safe_float(df["Vol_Surge"].iloc[-5:].mean(), 1.0)
    pocket_pivot = int(df["PocketPivot"].iloc[-5:].sum()) > 0

    # Small-cap risk metrics
    adv_cr = safe_float(df["Turnover"].rolling(20).median().iloc[-1], 0.0) / 1e7
    roc_5 = safe_float(last["ROC_5"])
    hype_spike = bool(rel_vol_max5 >= 6.0 and roc_5 > 20.0)

    pos = {
        "roc_20": safe_float(last["ROC_20"]),
        "roc_60": safe_float(last["ROC_60"]),
        "rs_20": rs_20,
        "macd_hist_pct": (macd_hist / curr_price * 100) if curr_price > 0 else 0.0,
        "adx": adx,
        "dmi_bull": dmi_bull,
        "dist_52wH": safe_float(last["Dist_52wH"], -20.0),
        "macd_bull": macd_bull,
        "macd_hist_rising": macd_hist_rising,
        "stoch_k": optional_float(last["Stoch_K"]),
        "cmf": safe_float(last["CMF"]),
        "obv_slope20": safe_float(last["OBV_slope20"]),
        "up_dn_vol": safe_float(last["UpDnVol25"], 1.0),
        "rel_vol5": rel_vol5,
        "mfi": safe_float(last["MFI"], 50.0),
        "acc_days25": safe_float(last["Acc_days25"]),
        "dist_days25": safe_float(last["Dist_days25"]),
        "pocket_pivot": pocket_pivot,
    }
    mom_score, mom_parts = _momentum_score(pos)
    vol_score, vol_parts = _volume_score(pos)

    return {
        "df": df,
        "curr_price": curr_price,
        "curr_rsi": curr_rsi,
        "curr_atr": curr_atr,
        "atr_pct": atr_pct,
        # structure
        "hl": hl,
        "hh": hh,
        "rsi_div": rsi_div,
        "vol_near_support": vol_near_support,
        "vol_contract_expand": vol_contract_expand,
        # relative strength
        "rs_20": rs_20,
        "rs_60": rs_60,
        "vol_norm_dd": vol_norm_dd,
        # trend
        "sma20_slope": safe_float(last["SMA20_slope"]),
        "sma50_slope": safe_float(last["SMA50_slope"]),
        "sma200_slope": safe_float(last["SMA200_slope"]),
        "above_200dma": above_200,
        "above_50dma": above_50,
        "above_150dma": above_150,
        # drawdowns
        "dd_5": safe_float(last["DD_5"]),
        "dd_10": safe_float(last["DD_10"]),
        "dd_20": safe_float(last["DD_20"]),
        "dd_60": dd_60,
        "dd_252": safe_float(last["DD_252"]),
        # momentum block
        "mom_score": mom_score,
        "mom_parts": mom_parts,
        "macd": safe_float(last["MACD"]),
        "macd_hist": macd_hist,
        "macd_bull": macd_bull,
        "macd_hist_rising": macd_hist_rising,
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "dmi_bull": dmi_bull,
        "stoch_k": safe_float(last["Stoch_K"], 50.0),
        "stoch_d": safe_float(last["Stoch_D"], 50.0),
        "stochrsi": safe_float(last["StochRSI"], 50.0),
        "bb_pctb": safe_float(last["BB_pctB"], 0.5),
        "bb_width_pct": safe_float(last["BB_width_pct"], 0.5),
        "roc_5": roc_5,
        "roc_20": pos["roc_20"],
        "roc_60": pos["roc_60"],
        "dist_52wH": pos["dist_52wH"],
        "dist_52wL": safe_float(last["Dist_52wL"]),
        # volume block
        "vol_score": vol_score,
        "vol_parts": vol_parts,
        "cmf": pos["cmf"],
        "mfi": pos["mfi"],
        "obv_slope20": pos["obv_slope20"],
        "up_dn_vol": pos["up_dn_vol"],
        "rel_vol5": rel_vol5,
        "rel_vol_max5": rel_vol_max5,
        "vol_surge5": vol_surge5,
        "vol_dryup": safe_float(last["Vol_DryUp"], 1.0),
        "acc_days25": pos["acc_days25"],
        "dist_days25": pos["dist_days25"],
        "pocket_pivot": pocket_pivot,
        "near_20d_high": bool(curr_price >= safe_float(last["DC_high20"], np.inf)),
        # small-cap risk block
        "adv_cr": adv_cr,
        "lock_days20": safe_float(last["Lock_days20"]),
        "hype_spike": hype_spike,
    }


def smallcap_filter_reason(tech: dict, args):
    """Return a rejection reason string, or None if the stock passes."""
    if tech["curr_price"] < args.min_price:
        return f"Price < Rs {args.min_price:g}"
    if tech["adv_cr"] < args.min_adv:
        return f"Low Liquidity (ADV {tech['adv_cr']:.1f} Cr)"
    if tech["atr_pct"] > args.max_atr_pct:
        return f"Too Volatile (ATR {tech['atr_pct']:.1f}%)"
    if tech["lock_days20"] >= 3:
        return "Circuit-locked sessions"
    return None


# =============================================================================
# FUNDAMENTAL ANALYSIS
# =============================================================================

def get_point_in_time_financials(t_obj, current_date):
    fin_data = {}
    try:
        q_fin = t_obj.quarterly_financials
        a_fin = t_obj.financials
        fin = q_fin if q_fin is not None and not q_fin.empty else a_fin

        if fin is not None and not fin.empty:
            c_date = pd.Timestamp(current_date)
            valid_cols = [c for c in fin.columns if pd.Timestamp(c) <= c_date]
            if valid_cols:
                fin_data = fin[valid_cols[0]].dropna().to_dict()
    except Exception:
        pass
    return fin_data


def _check_earnings_risk(t_obj, current_date) -> bool:
    if t_obj is None:
        return False
    try:
        cal = t_obj.calendar
        if cal is None:
            return False

        edates = []
        if isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.columns:
                edates = cal["Earnings Date"].dropna().tolist()
        elif isinstance(cal, dict):
            edates = cal.get("Earnings Date", [])
            if not isinstance(edates, list):
                edates = [edates]
        elif isinstance(cal, list):
            edates = cal

        c_date = pd.Timestamp(current_date)
        for ed in edates:
            try:
                diff = (pd.Timestamp(ed) - c_date).days
                if 0 <= diff <= 7:
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def analyze_fundamentals(info, pt_fin, pe_history, current_date, t_obj=None, min_mcap_cr=0.0):
    if not isinstance(info, dict):
        info = {}

    sector = str(info.get("sector", "")).lower()
    is_financial = any(
        s in sector for s in ["bank", "financial", "insurance", "nbfc", "credit"]
    )

    pe = optional_float(info.get("trailingPE"))

    valid_pe_history = []
    for x in pe_history:
        fx = optional_float(x)
        if fx is not None and fx > 0:
            valid_pe_history.append(fx)
    pe_median = float(np.median(valid_pe_history)) if len(valid_pe_history) > 20 else None

    rev_growth = optional_float(info.get("revenueGrowth"))
    earn_growth = optional_float(info.get("earningsGrowth"))
    roe = optional_float(info.get("returnOnEquity"))
    roa = optional_float(info.get("returnOnAssets"))
    roic = optional_float(info.get("returnOnCapital"))
    de = optional_float(info.get("debtToEquity"))
    margin = optional_float(info.get("profitMargins"))
    current_ratio = optional_float(info.get("currentRatio"))
    insiders = optional_float(info.get("heldPercentInsiders"))
    mcap = optional_float(info.get("marketCap"))
    mcap_cr = mcap / 1e7 if mcap is not None else None

    earnings_risk = _check_earnings_risk(t_obj, current_date)

    scores = {"val": 50, "mom": 50}
    reasons = []
    rejected = False
    reject_reason = ""

    # Market-cap floor (only when Yahoo provides it)
    if mcap_cr is not None and min_mcap_cr > 0 and mcap_cr < min_mcap_cr:
        rejected = True
        reject_reason = f"Market cap {mcap_cr:,.0f} Cr < {min_mcap_cr:,.0f} Cr"

    # Loss-making AND deteriorating -> reject outright
    if (
        not rejected
        and margin is not None and margin < 0
        and earn_growth is not None and earn_growth < -0.20
    ):
        rejected = True
        reject_reason = "Loss-making and earnings deteriorating"

    if is_financial:
        if roa is not None and roa < 0.005:
            rejected = True
            reject_reason = "Poor ROA for Financial"
        if roe is not None and roe < 0.08:
            rejected = True
            reject_reason = "Poor ROE for Financial"
        reasons.append(
            f"Financial Sector Eval: ROA={optional_pct(roa)}, ROE={optional_pct(roe)}"
        )
    else:
        if de is not None and de > 150:     # stricter than midcap (200)
            rejected = True
            reject_reason = f"High Debt/Equity ({de:.0f}%)"
        if current_ratio is not None and current_ratio < 0.8:
            scores["val"] -= 10
            reasons.append(f"Weak liquidity (current ratio {current_ratio:.2f})")
        if roic is not None and roic < 0.08:
            scores["val"] -= 20
            reasons.append(f"Low ROIC ({roic:.2%})")
        elif roic is not None and roic > 0.15:
            scores["val"] += 20
            reasons.append(f"Strong ROIC ({roic:.2%})")

    if margin is not None:
        if margin > 0.10:
            scores["val"] += 8
            reasons.append(f"Healthy net margin ({margin:.1%})")
        elif margin < 0:
            scores["val"] -= 15
            reasons.append(f"Negative net margin ({margin:.1%})")

    if insiders is not None and insiders < 0.25:
        scores["val"] -= 5
        reasons.append(f"Low insider/promoter holding ({insiders:.0%})")

    if pe is not None and pe > 0 and pe_median is not None:
        if pe < pe_median * 0.8:
            scores["val"] += 25
            reasons.append(f"Undervalued: PE {pe:.1f} < Median {pe_median:.1f}")
        elif pe > pe_median * 1.5:
            scores["val"] -= 25
            reasons.append(f"Overvalued: PE {pe:.1f} > Median {pe_median:.1f}")

    if rev_growth is not None:
        if rev_growth > 0.15:
            scores["mom"] += 20
            reasons.append(f"Strong Revenue Growth ({rev_growth:.1%})")
        elif rev_growth < -0.05:
            scores["mom"] -= 30
            reasons.append(f"Weak Revenue Growth ({rev_growth:.1%})")

    if earn_growth is not None:
        if earn_growth > 0.15:
            scores["mom"] += 20
            reasons.append(f"Strong Earnings Growth ({earn_growth:.1%})")
        elif earn_growth < -0.10:
            scores["mom"] -= 30
            reasons.append(f"Weak Earnings Growth ({earn_growth:.1%})")

    scores["val"] = max(0, min(100, scores["val"]))
    scores["mom"] = max(0, min(100, scores["mom"]))

    return {
        "scores": scores,
        "reasons": reasons,
        "rejected": rejected,
        "reject_reason": reject_reason,
        "is_financial": is_financial,
        "pe": pe,
        "pe_median": pe_median,
        "mcap_cr": mcap_cr,
        "earnings_risk": earnings_risk,
        "sector": info.get("sector", "Unknown"),
    }


# =============================================================================
# REGIME / SCORING ENGINE
# =============================================================================

# Composite weights (sum to 1.0). Momentum+volume together carry 22%.
WEIGHTS = {"opp": 0.18, "stab": 0.20, "trend": 0.15, "momv": 0.22, "val": 0.10, "fmom": 0.15}


def calculate_regime_and_scores(tech: dict, fund: dict):
    scores = {"opp": 50, "stab": 50, "trend": 50}
    reasons = []
    tech_notes = []

    # ---- Opportunity (how big is the decline; small-cap scaled) ----------
    dd_60 = tech["dd_60"]
    if dd_60 < -15:
        scores["opp"] += 20
    if dd_60 < -30:
        scores["opp"] += 20
    if tech["vol_norm_dd"] > 3.5:
        scores["opp"] += 15
    tech_notes.append(
        f"60d Drawdown: {dd_60:.1f}% | Vol-Norm DD: {tech['vol_norm_dd']:.1f} ATRs"
    )

    # ---- Stabilization (is selling pressure fading) ----------------------
    rsi = tech["curr_rsi"]
    if 30 <= rsi <= 45:
        scores["stab"] += 20
    elif rsi < 25:
        scores["stab"] -= 20
    if tech["hl"]:
        scores["stab"] += 25
        tech_notes.append("Recent higher low (structure improving)")
    if tech["rsi_div"]:
        scores["stab"] += 25
        tech_notes.append("Bullish RSI divergence")
    if tech["vol_contract_expand"] > 1.5:
        scores["stab"] += 15
        tech_notes.append("Volume drying up vs the decline leg")

    # ---- Trend (position & slope) ----------------------------------------
    if tech["above_200dma"]:
        scores["trend"] += 20
    if tech["above_50dma"]:
        scores["trend"] += 10
    if tech["sma200_slope"] > 0:
        scores["trend"] += 15
    if tech["rs_20"] > 3:
        scores["trend"] += 15
        tech_notes.append(f"Outperforming smallcap index (20d RS {tech['rs_20']:+.1f}%)")
    if tech["adx"] >= 25 and tech["dmi_bull"]:
        scores["trend"] += 10
        tech_notes.append(f"Strong directional trend (ADX {tech['adx']:.0f}, +DI>-DI)")

    # ---- Momentum + Volume ----------------------------------------------
    momv = round(0.5 * tech["mom_score"] + 0.5 * tech["vol_score"], 1)
    scores["momv"] = momv

    if tech["macd_bull"] and tech["macd_hist_rising"]:
        tech_notes.append("MACD above signal and histogram rising")
    if tech["rel_vol5"] >= 1.3:
        tech_notes.append(f"Volume expansion (5d rel-vol {tech['rel_vol5']:.2f}x)")
    if tech["cmf"] > 0.10:
        tech_notes.append(f"Accumulation (CMF {tech['cmf']:+.2f})")
    elif tech["cmf"] < -0.10:
        tech_notes.append(f"Distribution (CMF {tech['cmf']:+.2f})")
    if tech["obv_slope20"] > 0.10:
        tech_notes.append(f"OBV rising ({tech['obv_slope20']:+.2f})")
    if tech["up_dn_vol"] > 1.2:
        tech_notes.append(f"Up-volume dominates (up/dn vol {tech['up_dn_vol']:.2f})")
    if tech["acc_days25"] - tech["dist_days25"] >= 3:
        tech_notes.append(
            f"Net accumulation days ({int(tech['acc_days25'])} acc vs "
            f"{int(tech['dist_days25'])} dist over 25d)"
        )
    if tech["pocket_pivot"]:
        tech_notes.append("Pocket pivot (up bar on more volume than any recent down bar)")
    if tech["mfi"] < 20:
        tech_notes.append(f"Money Flow Index oversold ({tech['mfi']:.0f})")
    if tech["bb_width_pct"] <= 0.20:
        tech_notes.append("Volatility squeeze (narrow Bollinger width)")

    # ---- Small-cap specific notes ---------------------------------------
    if tech.get("hype_spike"):
        tech_notes.append(
            f"⚠️ Hype spike: peak rel-vol {tech['rel_vol_max5']:.1f}x with "
            f"{tech['roc_5']:+.0f}% in 5d (possible operator/pump move)"
        )
    if tech.get("adv_cr", 99) < 5:
        tech_notes.append(f"Thin liquidity (ADV {tech['adv_cr']:.1f} Cr) - size positions carefully")

    # ---- Regime / setup --------------------------------------------------
    setup = "Neutral"
    composite_mult = 1.0

    is_falling_knife = (
        not tech["above_50dma"]
        and not tech["above_200dma"]
        and tech["sma50_slope"] < -3
        and tech["sma200_slope"] < 0
        and rsi < 30
        and not tech["hl"]
        and tech["vol_norm_dd"] > 4.0
    )
    is_quality_pullback = (
        tech["above_200dma"]
        and 30 <= rsi <= 45
        and tech["hl"]
        and fund["scores"]["val"] > 60
        and fund["scores"]["mom"] > 60
        and tech["rs_20"] > 0
    )
    is_momentum_breakout = (
        tech["dist_52wH"] > -8
        and tech["adx"] >= 25
        and tech["dmi_bull"]
        and tech["macd_bull"]
        and tech["rs_20"] > 4
        and tech["rel_vol5"] >= 1.3          # small caps need real volume confirmation
        and not tech["hype_spike"]
    )
    is_accumulation_breakout = (
        tech["acc_days25"] - tech["dist_days25"] >= 3
        and tech["obv_slope20"] > 0.10
        and tech["up_dn_vol"] > 1.2
        and (tech["bb_pctb"] > 0.8 or tech["near_20d_high"])
        and not tech["hype_spike"]
    )
    is_volume_thrust = (
        tech["pocket_pivot"]
        and tech["above_50dma"]
        and tech["rs_20"] > 0
        and tech["rel_vol5"] >= 1.3
        and tech["cmf"] > 0
        and not tech["hype_spike"]
    )
    is_momentum_pullback = (
        tech["above_50dma"]
        and abs(dd_60) < 15
        and tech["rs_20"] > 6
        and fund["scores"]["mom"] > 70
    )

    if is_falling_knife:
        setup, composite_mult = "Falling Knife", 0.3
        reasons.append("⚠️ REGIME: Falling Knife (declining DMAs, lower lows, extreme RSI)")
    elif is_quality_pullback:
        setup, composite_mult = "Quality Pullback", 1.2
        reasons.append("✅ REGIME: Quality Pullback (healthy fundamentals, higher lows, RS improving)")
    elif is_momentum_breakout:
        setup, composite_mult = "Momentum Breakout", 1.15
        reasons.append("🚀 REGIME: Momentum Breakout (near 52w high, strong ADX/DI, volume expansion)")
    elif is_accumulation_breakout:
        setup, composite_mult = "Accumulation Breakout", 1.15
        reasons.append("🏦 REGIME: Accumulation Breakout (net accumulation + OBV + breakout)")
    elif is_volume_thrust:
        setup, composite_mult = "Volume Thrust", 1.1
        reasons.append("📶 REGIME: Volume Thrust (pocket pivot with positive money flow)")
    elif is_momentum_pullback:
        setup, composite_mult = "Momentum Pullback", 1.1
        reasons.append("🔥 REGIME: Momentum Pullback (strong uptrend, shallow dip, high RS)")
    elif tech["rsi_div"] and tech["vol_contract_expand"] > 1.5 and not tech["above_200dma"]:
        setup, composite_mult = "Reversal", 0.9
        reasons.append("🔄 REGIME: Reversal (deep dip, divergence & volume climax)")
    elif fund["scores"]["mom"] < 30:
        setup, composite_mult = "Fundamental Deterioration", 0.5
        reasons.append("📉 REGIME: Fundamental Deterioration (poor growth/margins)")

    # Small-cap haircuts
    if tech.get("hype_spike"):
        composite_mult *= 0.8
    if tech.get("adv_cr", 99) < 5:
        composite_mult *= 0.95

    composite = (
        scores["opp"] * WEIGHTS["opp"]
        + scores["stab"] * WEIGHTS["stab"]
        + scores["trend"] * WEIGHTS["trend"]
        + scores["momv"] * WEIGHTS["momv"]
        + fund["scores"]["val"] * WEIGHTS["val"]
        + fund["scores"]["mom"] * WEIGHTS["fmom"]
    ) * composite_mult

    checks = [
        tech["hl"], tech["rsi_div"], tech["vol_contract_expand"] > 0,
        tech["rs_20"] != 0, tech["macd_bull"], tech["adx"] >= 20,
        tech["cmf"] > 0, tech["pocket_pivot"],
    ]
    valid = sum(1 for c in checks if c)
    confidence = min(100, (valid / len(checks)) * 100 + 15)

    scores["composite"] = round(composite, 1)
    scores["confidence"] = round(confidence, 1)
    scores["mom"] = tech["mom_score"]
    scores["vol"] = tech["vol_score"]

    return setup, scores, reasons, tech_notes


# =============================================================================
# SINGLE TICKER PROCESSOR (live scan)
# =============================================================================

def process_ticker(ticker, df, index_df, args, current_date):
    try:
        df = clean_price_dataframe(df)
        if len(df) < MIN_BARS:
            return "filtered", "Insufficient History"

        # Drop incomplete current-day candle (before NSE close + buffer)
        if IST is not None:
            now = datetime.now(IST)
            last_ts = df.index[-1]
            if getattr(last_ts, "tzinfo", None):
                last_date = last_ts.tz_convert(IST).date()
            else:
                last_date = last_ts.date()
            if last_date == now.date() and (now.hour, now.minute) < (15, 35):
                df = df.iloc[:-1]

        if len(df) < MIN_BARS:
            return "filtered", "Insufficient History"

        # Small caps trade thinly; tolerate a few more zero-volume days
        if (df["Volume"].tail(60) == 0).sum() > 8:
            return "filtered", "Too many zero-volume days"

        tech = analyze_technicals(df, index_df)
        if not tech:
            return "filtered", "Tech Calc Failed"

        reason = smallcap_filter_reason(tech, args)
        if reason:
            return "filtered", reason

        # Always fresh fundamentals (nothing cached between runs)
        try:
            t_obj = yf.Ticker(ticker)
            info = t_obj.info
        except Exception as e:
            return "error", f"Info fetch failed: {e}"

        if not info or not isinstance(info, dict):
            return "filtered", "Empty Info"

        pt_fin = get_point_in_time_financials(t_obj, current_date)
        fund = analyze_fundamentals(
            info, pt_fin, [], current_date, t_obj, min_mcap_cr=args.min_mcap
        )

        if fund["rejected"]:
            return "filtered", fund["reject_reason"]

        setup, scores, reasons, tech_notes = calculate_regime_and_scores(tech, fund)

        if setup in ("Falling Knife", "Fundamental Deterioration") and not args.lenient:
            return "filtered", f"Rejected Setup: {setup}"

        if scores["composite"] < args.min_score:
            return "filtered", f"Score {scores['composite']} < {args.min_score}"

        pe_output = f"{fund['pe']:.1f}" if fund["pe"] is not None else "-"
        mcap_output = f"{fund['mcap_cr']:,.0f}" if fund["mcap_cr"] is not None else "-"

        return "ok", {
            "Ticker": ticker.replace(".NS", ""),
            "Setup": setup,
            "Conv": scores["confidence"],
            "Score": scores["composite"],
            "Mom": tech["mom_score"],
            "Vol": tech["vol_score"],
            "Price": tech["curr_price"],
            "RSI": tech["curr_rsi"],
            "ADX": tech["adx"],
            "RelVol": tech["rel_vol5"],
            "CMF": tech["cmf"],
            "MFI": tech["mfi"],
            "DD_60": tech["dd_60"],
            "RS_20": tech["rs_20"],
            "ATR_pct": tech["atr_pct"],
            "ADV_Cr": tech["adv_cr"],
            "MaxPos_L": tech["adv_cr"] * 0.01 * 100,   # 1% of ADV, in Rs lakh
            "MCap_Cr": mcap_output,
            "Hype": tech["hype_spike"],
            "PE": pe_output,
            "Sector": str(fund["sector"])[:15],
            "Earnings_Soon": fund["earnings_risk"],
            "Tech_Reasons": [r for r in (tech_notes + fund["reasons"]) if r],
            "Regime_Reasons": reasons,
        }

    except Exception as e:
        # One bad ticker must never kill the whole scan.
        return "error", f"{type(e).__name__}: {e}"


# =============================================================================
# LIVE SCANNER
# =============================================================================

def get_index_data(benchmark: str):
    for idx in [benchmark] + [t for t in INDEX_TICKERS if t != benchmark]:
        try:
            idx_df = yf.download(idx, period="3y", progress=False, auto_adjust=True, threads=False)
            idx_df = clean_price_dataframe(idx_df)
            if not idx_df.empty and len(idx_df) > 200:
                cprint(f"   benchmark: {idx}", "white")
                return calculate_indicators(idx_df)
        except Exception as e:
            cprint(f"⚠️ Index {idx} failed: {e}", "yellow")
    return None


def scan_stocks(args):
    current_date = datetime.now().date()
    _, label, benchmark = UNIVERSES[args.universe]

    cprint(f"⏳ Fetching benchmark index data for {label}...", "cyan")
    index_df = get_index_data(benchmark)

    market_regime = "Unknown"
    if index_df is not None and "SMA200" in index_df.columns and len(index_df) > 200:
        close = safe_float(index_df["Close"].iloc[-1])
        sma200 = safe_float(index_df["SMA200"].iloc[-1])
        if sma200 > 0:
            market_regime = (
                "Bullish (Index > 200DMA)" if close > sma200 else "Bearish (Index < 200DMA)"
            )

    tickers = load_tickers(args)

    cprint(
        f"⏳ Downloading ~{HISTORY_DAYS} days of price data for {len(tickers)} tickers...",
        "cyan",
    )

    try:
        all_data = yf.download(
            tickers,
            period=f"{HISTORY_DAYS}d",
            group_by="ticker",
            threads=True,
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        cprint(f"❌ Data download failed: {e}", "red")
        return

    if all_data is None or all_data.empty:
        cprint("❌ Yahoo returned no price data.", "red")
        return

    results = []
    filtered = Counter()
    errors = []

    pbar = tqdm(total=len(tickers), desc="Scanning", unit="stock") if HAS_TQDM else None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}

        for ticker in tickers:
            try:
                df = clean_price_dataframe(extract_ticker_dataframe(all_data, ticker))
                if len(df) < MIN_BARS:
                    filtered["Insufficient History"] += 1
                    if pbar:
                        pbar.update(1)
                    continue

                fut = pool.submit(process_ticker, ticker, df, index_df, args, current_date)
                futures[fut] = ticker
            except Exception as e:
                errors.append(f"{ticker}: {type(e).__name__}: {e}")
                if pbar:
                    pbar.update(1)

        for fut in as_completed(futures):
            if pbar:
                pbar.update(1)
            ticker = futures[fut]
            try:
                status, payload = fut.result()
                if status == "ok":
                    results.append(payload)
                elif status == "filtered":
                    # Bucket parameterised reasons so the summary stays readable
                    key = payload.split(" (")[0] if payload.startswith(("Low Liquidity", "Too Volatile")) else payload
                    if payload.startswith("Score "):
                        key = "Below min score"
                    elif payload.startswith("Market cap"):
                        key = "Market cap below floor"
                    filtered[key] += 1
                else:
                    errors.append(f"{ticker}: {payload}")
            except Exception as e:
                errors.append(f"{ticker}: {type(e).__name__}: {e}")

    if pbar:
        pbar.close()

    # ---- Output ----------------------------------------------------------
    print("\n" + "=" * 100)
    cprint(f"  SMALLCAP MOMENTUM & VOLUME SCANNER — {current_date} | {label} | Regime: {market_regime}", "cyan")
    cprint(
        "  Objective: healthy small caps, deep decline, selling pressure fading, "
        "momentum + volume turning up",
        "cyan",
    )
    print("=" * 100)

    if filtered:
        cprint("\nExcluded by Filters:", "white")
        for reason, n in filtered.most_common(15):
            print(f"   {n:>3} × {reason}")

    if errors:
        cprint("\nTicker Errors:", "yellow")
        for err in errors[:20]:
            print(f"   ⚠️ {err}")
        if len(errors) > 20:
            print(f"   ... and {len(errors) - 20} more")

    if not results:
        cprint("\nNo stocks matched the criteria.", "yellow")
        return

    df_res = pd.DataFrame(results).sort_values("Score", ascending=False)
    shown = df_res.head(args.top) if args.top else df_res

    for _, row in shown.iterrows():
        if row["Setup"] in ("Quality Pullback", "Momentum Breakout", "Accumulation Breakout", "Volume Thrust"):
            color = "green"
        elif row["Setup"] == "Momentum Pullback":
            color = "cyan"
        elif row["Setup"] == "Falling Knife":
            color = "red"
        else:
            color = "white"

        cprint(f"\n{'─' * 90}", color)
        cprint(
            f"  SETUP: {row['Setup']} | CONVICTION: {row['Conv']}% | SCORE: {row['Score']}",
            color,
        )
        earn = " | ⚠️ EARNINGS ≤7d" if row["Earnings_Soon"] else ""
        hype = " | ⚠️ HYPE SPIKE" if row["Hype"] else ""
        cprint(
            f"  {row['Ticker']} | ₹{row['Price']:.2f} | MCap {row['MCap_Cr']} Cr | {row['Sector']} | "
            f"RSI: {row['RSI']:.1f} | DD60: {row['DD_60']:.1f}% | "
            f"RS20: {row['RS_20']:+.1f}% | PE: {row['PE']}{earn}{hype}",
            color,
        )
        cprint(
            f"  MOMENTUM {row['Mom']:.0f}/100 | VOLUME {row['Vol']:.0f}/100 | "
            f"ADX {row['ADX']:.0f} | RelVol {row['RelVol']:.2f}x | "
            f"CMF {row['CMF']:+.2f} | MFI {row['MFI']:.0f}",
            color,
        )
        cprint(
            f"  RISK: ATR {row['ATR_pct']:.1f}% | ADV ₹{row['ADV_Cr']:.1f} Cr | "
            f"Suggested max position ≈ ₹{row['MaxPos_L']:.1f} lakh (1% of ADV)",
            color,
        )
        for r in row["Regime_Reasons"]:
            print(f"    {r}")
        print("    Technical & Fundamental Reasons:")
        for r in row["Tech_Reasons"]:
            print(f"      • {r}")

    # ---- Optional export -------------------------------------------------
    if args.export_csv:
        out = df_res.copy()
        out["Regime_Reasons"] = out["Regime_Reasons"].apply(lambda x: " | ".join(x))
        out["Tech_Reasons"] = out["Tech_Reasons"].apply(lambda x: " | ".join(x))
        Path(args.export_csv).write_text(out.to_csv(index=False))
        cprint(f"\n💾 Wrote {len(out)} rows to {args.export_csv}", "cyan")
    else:
        cprint("\n(Results are shown only; nothing was saved. Use --export-csv to save.)", "cyan")


# =============================================================================
# BACKTESTER (in-memory only)
# =============================================================================

def run_backtest(args):
    _, label, benchmark = UNIVERSES[args.universe]
    cprint("🚀 Starting No-Look-Ahead (prices) Backtester...", "magenta")
    cprint(
        "⚠️ Caveats: Yahoo 'info' fundamentals are CURRENT values, not historical, "
        "and the constituent list is today's (survivorship bias). Small-cap slippage, "
        "impact cost and circuit-locks are NOT modelled. Treat results as optimistic.",
        "yellow",
    )

    tickers = load_tickers(args)

    cprint(f"⏳ Downloading 3 years of data for {len(tickers)} tickers...", "cyan")
    all_data = yf.download(
        tickers, period="3y", group_by="ticker",
        threads=True, progress=False, auto_adjust=True,
    )

    index_df = pd.DataFrame()
    for idx in [benchmark] + [t for t in INDEX_TICKERS if t != benchmark]:
        index_df = clean_price_dataframe(
            yf.download(idx, period="3y", progress=False, auto_adjust=True, threads=False)
        )
        if not index_df.empty:
            break
    if not index_df.empty:
        index_df = calculate_indicators(index_df)

    price_cache = {}
    for t in tickers:
        d = clean_price_dataframe(extract_ticker_dataframe(all_data, t))
        if len(d) >= MIN_BARS:
            price_cache[t] = d

    cprint("⏳ Fetching fundamentals...", "cyan")
    fin_cache = {}
    for t in price_cache:
        try:
            t_obj = yf.Ticker(t)
            fin_cache[t] = {
                "info": t_obj.info,
                "q_fin": t_obj.quarterly_financials,
                "a_fin": t_obj.financials,
            }
        except Exception:
            fin_cache[t] = {}

    start_date = datetime.now() - timedelta(days=365 * 2)
    test_dates = pd.date_range(
        start=start_date,
        end=datetime.now() - timedelta(days=60),
        freq=f"{args.backtest_days}B",
    )

    cprint(f"⏳ Running backtest over {len(test_dates)} checkpoints...", "cyan")
    pbar = tqdm(total=len(test_dates), desc="Backtesting") if HAS_TQDM else None

    records = []

    for dt in test_dates:
        if pbar:
            pbar.update(1)

        for ticker, df_full in price_cache.items():
            try:
                pos = df_full.index.searchsorted(pd.Timestamp(dt), side="right") - 1
                if pos < MIN_BARS:
                    continue
                if pos + 30 >= len(df_full):
                    continue

                df = df_full.iloc[: pos + 1]
                idx_slice = index_df[index_df.index <= df.index[-1]] if not index_df.empty else None

                tech = analyze_technicals(df, idx_slice)
                if not tech:
                    continue

                if smallcap_filter_reason(tech, args):
                    continue

                fin_data = fin_cache.get(ticker, {})
                q_fin, a_fin = fin_data.get("q_fin"), fin_data.get("a_fin")
                fin = q_fin if (q_fin is not None and not q_fin.empty) else a_fin

                pt_fin = {}
                if fin is not None and not fin.empty:
                    valid_cols = [c for c in fin.columns if pd.Timestamp(c) <= pd.Timestamp(dt)]
                    if valid_cols:
                        pt_fin = fin[valid_cols[0]].dropna().to_dict()

                fund = analyze_fundamentals(fin_data.get("info", {}), pt_fin, [], dt, None, min_mcap_cr=0.0)
                if fund["rejected"]:
                    continue

                setup, scores, _, _ = calculate_regime_and_scores(tech, fund)
                if setup in ("Falling Knife", "Fundamental Deterioration"):
                    continue
                if scores["composite"] < args.min_score:
                    continue

                entry_p = df_full["Close"].iloc[pos]
                fut = df_full["Close"].iloc[pos + 1: pos + 31]
                if len(fut) < 30:
                    continue

                def r(n):
                    return (fut.iloc[n - 1] / entry_p - 1) * 100

                records.append({
                    "date": df.index[-1],
                    "ticker": ticker,
                    "setup": setup,
                    "composite": scores["composite"],
                    "mom": scores["mom"],
                    "vol": scores["vol"],
                    "entry": entry_p,
                    "r1": r(1), "r3": r(3), "r5": r(5),
                    "r10": r(10), "r20": r(20), "r30": r(30),
                    "mfe": (fut.max() / entry_p - 1) * 100,
                    "mae": (fut.min() / entry_p - 1) * 100,
                })
            except Exception:
                continue

    if pbar:
        pbar.close()

    cprint("✅ Backtest complete.", "green")
    print_backtest_stats(pd.DataFrame(records), args.cost_bps)


def print_backtest_stats(df: pd.DataFrame, cost_bps: float = 0.0):
    if df is None or df.empty:
        cprint("No backtest signals were generated.", "yellow")
        return

    # Optional round-trip cost haircut (small-cap impact cost is material)
    cost_pct = cost_bps / 100.0
    if cost_pct:
        df = df.copy()
        for col in ("r1", "r3", "r5", "r10", "r20", "r30"):
            df[col] = df[col] - cost_pct

    print("\n" + "=" * 100)
    cprint("  BACKTEST AGGREGATE STATISTICS", "magenta")
    print("=" * 100)
    print(f"  Total signals: {len(df)}" + (f" | round-trip cost applied: {cost_bps:g} bps" if cost_pct else ""))

    for setup in df["setup"].dropna().unique():
        sub = df[df["setup"] == setup]
        cprint(f"\n  SETUP: {setup} (N={len(sub)})", "cyan")
        print(
            f"    Win Rate (20d): {(sub['r20'] > 0).mean() * 100:.1f}% | "
            f"Median Return (20d): {sub['r20'].median():.2f}%"
        )
        print(f"    Avg MFE: {sub['mfe'].mean():.2f}% | Avg MAE: {sub['mae'].mean():.2f}%")

    print("\n  SCORE BUCKET PERFORMANCE (20d Returns):")
    df = df.copy()
    df["bucket"] = pd.cut(
        df["composite"], bins=[0, 40, 60, 80, 100, np.inf],
        labels=["Low", "Med", "High", "Ultra", "Extreme"],
    )
    print(df.groupby("bucket", observed=True)["r20"].agg(["count", "mean", "median"]).to_string())

    print("\n  MOMENTUM-SCORE BUCKET (20d Returns):")
    df["mom_bucket"] = pd.cut(
        df["mom"], bins=[0, 40, 60, 80, 100], labels=["Low", "Med", "High", "Ultra"],
    )
    print(df.groupby("mom_bucket", observed=True)["r20"].agg(["count", "mean", "median"]).to_string())

    print("\n  VOLUME-SCORE BUCKET (20d Returns):")
    df["vol_bucket"] = pd.cut(
        df["vol"], bins=[0, 40, 60, 80, 100], labels=["Low", "Med", "High", "Ultra"],
    )
    print(df.groupby("vol_bucket", observed=True)["r20"].agg(["count", "mean", "median"]).to_string())


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Smallcap Momentum & Volume Scanner / Backtester (stateless)"
    )
    p.add_argument("--backtest", action="store_true", help="Run historical backtester")
    p.add_argument("--backtest-days", type=int, default=5, help="Backtest every N business days")
    p.add_argument("--cost-bps", type=float, default=100.0,
                   help="Round-trip cost in bps applied to backtest returns (default 100 = 1%%)")
    p.add_argument("--min-score", type=float, default=50.0, help="Minimum composite score to show")
    p.add_argument("--lenient", action="store_true", help="Include Falling Knife / Deterioration setups")
    p.add_argument("--workers", type=int, default=6, help="Concurrent workers")
    p.add_argument("--top", type=int, default=15, help="Show top N results (0 = all)")
    p.add_argument("--tickers-file", metavar="PATH", help="Custom ticker list")
    p.add_argument(
        "--universe", choices=sorted(UNIVERSES.keys()), default="smallcap250",
        help="Index universe to scan (default: smallcap250)",
    )
    p.add_argument(
        "--extra", metavar="CSV",
        help="Extra comma-separated tickers to append (quote shell-special symbols such as M&MFIN)",
    )
    p.add_argument("--min-adv", type=float, default=2.0,
                   help="Liquidity floor, median daily turnover in Rs crore (default 2)")
    p.add_argument("--min-price", type=float, default=20.0, help="Minimum share price in Rs (default 20)")
    p.add_argument("--min-mcap", type=float, default=500.0,
                   help="Minimum market cap in Rs crore (default 500; 0 disables)")
    p.add_argument("--max-atr-pct", type=float, default=7.0,
                   help="Reject if ATR/price exceeds this percent (default 7)")
    p.add_argument("--export-csv", metavar="PATH", help="Write full results to this CSV (opt-in)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.backtest:
        run_backtest(args)
    else:
        scan_stocks(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())