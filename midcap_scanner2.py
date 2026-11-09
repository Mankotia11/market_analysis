#!/usr/bin/env python3
"""
Midcap Stock Scanner v2 — Nifty Midcap 100
==========================================

Goal: find fundamentally healthy stocks that have had an unusually large
decline where selling pressure is fading and relative strength is improving.

What changed vs v1 (see review):
  * Backtest rows are keyed by run_id (no silent accumulation across runs).
  * All price indexes are normalised to tz-naive dates; backtest exceptions
    are counted and printed instead of silently skipped.
  * Fundamentals are point-in-time: YoY revenue / net-income growth is derived
    from dated financial statements with a reporting lag. Snapshot-only fields
    (PE, ROE, ROA, D/E) are used in LIVE mode only, so the backtest no longer
    leaks today's values. In the backtest the valuation pillar is neutral.
  * Hard drawdown gate (--min-dd) so a no-signal stock can't pass on the 50
    neutral baseline. Every pillar is clamped to 0-100, composite to 0-100.
  * --lenient now actually works (live and backtest).
  * "Climax" logic fixed: volume dry-up (contraction) and capitulation-held
    (high-volume down bar whose low later holds) are separate signals.
  * Higher-low / RSI-divergence need a recent, well-spaced swing pair.
    Swing detection is vectorised with strict/non-strict tie handling.
  * Unused features removed; earnings proximity now penalises the score.
  * "Conv" renamed "Coverage" (it measures data availability, not conviction).
  * Backtest: enters at NEXT open, per-ticker cooldown (de-overlap), costs,
    excess return vs index, split by market regime, indicators computed once.
  * Stale-data check, IST-consistent dates, index fallback is logged.
"""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
import threading
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from numpy.lib.stride_tricks import sliding_window_view

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

NSE_MIDCAP100_CSV = (
    "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv"
)

# Tried in order; whichever is used is printed.
INDEX_CANDIDATES = ["NIFTY_MIDCAP_100.NS", "^CNXMIDCAP", "^NSEI"]

# Fallback snapshot ONLY if the live NSE download fails.
MIDCAP_TICKERS = [
    "360ONE.NS", "3MINDIA.NS", "ABBOTINDIA.NS", "ABCAPITAL.NS", "ACC.NS",
    "AIAENG.NS", "AIIL.NS", "AJANTPHARM.NS", "ALKEM.NS", "ANTHEM.NS",
    "APARINDS.NS", "APLAPOLLO.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS", "ASTRAL.NS",
    "ATGL.NS", "AUBANK.NS", "AUROPHARMA.NS", "AWL.NS", "BAJAJHFL.NS",
    "BALKRISIND.NS", "BANKINDIA.NS", "BDL.NS", "BERGEPAINT.NS", "BHARATFORG.NS",
    "BHARTIHEXA.NS", "BHEL.NS", "BIOCON.NS", "BLUESTARCO.NS", "BSE.NS",
    "COCHINSHIP.NS", "COFORGE.NS", "COLPAL.NS", "CONCOR.NS", "COROMANDEL.NS",
    "CRISIL.NS", "DABUR.NS", "DALBHARAT.NS", "DIXON.NS", "ENDURANCE.NS",
    "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS", "FLUOROCHEM.NS", "FORTIS.NS",
    "GICRE.NS", "GLAXO.NS", "GLENMARK.NS", "GMRAIRPORT.NS", "GODFRYPHLP.NS",
    "GODREJIND.NS", "GODREJPROP.NS", "GROWW.NS", "GVT&D.NS", "HAVELLS.NS",
    "HDBFS.NS", "HEROMOTOCO.NS", "HEXT.NS", "HINDPETRO.NS", "HONAUT.NS",
    "HUDCO.NS", "ICICIAMC.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IDEA.NS",
    "IDFCFIRSTB.NS", "INDIANB.NS", "INDUSINDBK.NS", "INDUSTOWER.NS", "IPCALAB.NS",
    "IRCTC.NS", "IREDA.NS", "ITCHOTELS.NS", "JKCEMENT.NS", "JSL.NS",
    "JSWENERGY.NS", "JSWINFRA.NS", "JUBLFOOD.NS", "KALYANKJIL.NS", "KEI.NS",
    "KPITTECH.NS", "KPRMILL.NS", "LAURUSLABS.NS", "LENSKART.NS", "LGEINDIA.NS",
    "LICHSGFIN.NS", "LICI.NS", "LINDEINDIA.NS", "LLOYDSME.NS", "LTFL.NS",
    "LUPIN.NS", "M&MFIN.NS", "MAHABANK.NS", "MANKIND.NS", "MARICO.NS",
    "MCX.NS", "MEDANTA.NS", "MFSL.NS", "MOTILALOFS.NS", "MPHASIS.NS",
    "MRF.NS", "NAM-INDIA.NS", "NATIONALUM.NS", "NAUKRI.NS", "NHPC.NS",
    "NIACL.NS", "NLCINDIA.NS", "NMDC.NS", "NTPCGREEN.NS", "NYKAA.NS",
    "OBEROIRLTY.NS", "OFSS.NS", "OIL.NS", "PAGEIND.NS", "PATANJALI.NS",
    "PAYTM.NS", "PERSISTENT.NS", "PETRONET.NS", "PHOENIXLTD.NS", "PIIND.NS",
    "POLICYBZR.NS", "POLYCAB.NS", "POWERINDIA.NS", "PREMIERENE.NS", "PRESTIGE.NS",
    "RADICO.NS", "RVNL.NS", "SAIL.NS", "SBICARD.NS", "SCHAEFFLER.NS",
    "SJVNL.NS", "SRF.NS", "SUNDARMFIN.NS", "SUPREMEIND.NS", "SUZLON.NS",
    "SWIGGY.NS", "TATACOMM.NS", "TATAELXSI.NS", "TATAINVEST.NS", "THERMAX.NS",
    "TIINDIA.NS", "TORNTPOWER.NS", "UBL.NS", "UNOMINDAUPL.NS", "VMM.NS",
    "VOLTAS.NS", "WAAREEENERGY.NS", "YESBANK.NS",
]

MIN_BARS = 200
HORIZONS = [1, 3, 5, 10, 20, 30]
MAX_HORIZON = max(HORIZONS)
MIN_PE_SNAPSHOTS = 30
BAD_SETUPS = ("Falling Knife", "Fundamental Deterioration")
DB_NAME = "midcap_scanner_v2.db"  # new schema; old v1 DB is left untouched

REV_KEYS = ["Total Revenue", "Operating Revenue", "Revenue"]
NI_KEYS = ["Net Income", "Net Income Common Stockholders"]


# =============================================================================
# SMALL HELPERS
# =============================================================================

def now_ist() -> datetime:
    return datetime.now(IST) if IST else datetime.now()


def cprint(text: str, color: str = "white"):
    if HAS_COLOR:
        colors = {
            "red": Fore.RED,
            "green": Fore.GREEN,
            "yellow": Fore.YELLOW,
            "cyan": Fore.CYAN,
            "white": Fore.WHITE,
            "magenta": Fore.MAGENTA,
        }
        print(f"{colors.get(color, '')}{text}{Style.RESET_ALL}")
    else:
        print(text)


def optional_float(val):
    try:
        if val is None:
            return None
        x = float(val)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def safe_float(val, default=0.0):
    x = optional_float(val)
    return default if x is None else x


def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


# =============================================================================
# DATABASE
# =============================================================================

class ScannerDB:
    def __init__(self, db_path: str = DB_NAME):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.lock = threading.Lock()
        self._init_tables()

    def _init_tables(self):
        with self.lock:
            c = self.conn.cursor()
            c.execute(
                """CREATE TABLE IF NOT EXISTS bundles (
                    ticker TEXT PRIMARY KEY, fetch_date TEXT, data TEXT)"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS pe_history (
                    ticker TEXT, date TEXT, pe REAL,
                    PRIMARY KEY (ticker, date))"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS signals (
                    date TEXT, ticker TEXT, setup TEXT, composite REAL,
                    coverage REAL, entry_price REAL, opp REAL, stab REAL,
                    trend REAL, val REAL, mom REAL, reasons TEXT,
                    PRIMARY KEY (date, ticker))"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS backtest_results (
                    run_id TEXT, signal_date TEXT, ticker TEXT, setup TEXT,
                    regime TEXT, composite REAL, entry_price REAL,
                    ret_1d REAL, ret_3d REAL, ret_5d REAL, ret_10d REAL,
                    ret_20d REAL, ret_30d REAL, ex_20d REAL,
                    mfe REAL, mae REAL)"""
            )
            self.conn.commit()

    # -- fundamentals bundle ---------------------------------------------------
    def get_bundle(self, ticker, max_age_days=7):
        with self.lock:
            row = self.conn.execute(
                "SELECT fetch_date, data FROM bundles WHERE ticker=?", (ticker,)
            ).fetchone()
        if not row:
            return None
        try:
            age = (datetime.now() - datetime.strptime(row[0], "%Y-%m-%d")).days
            if age <= max_age_days:
                return json.loads(row[1])
        except Exception:
            return None
        return None

    def put_bundle(self, ticker, bundle):
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO bundles VALUES (?, ?, ?)",
                (
                    ticker,
                    datetime.now().strftime("%Y-%m-%d"),
                    json.dumps(bundle, default=str),
                ),
            )
            self.conn.commit()

    # -- PE snapshots (one per scan day; NOT a true 3Y history) ----------------
    def cache_pe(self, ticker, dt, pe):
        pe = optional_float(pe)
        if pe is None:
            return
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO pe_history VALUES (?, ?, ?)",
                (ticker, pd.Timestamp(dt).strftime("%Y-%m-%d"), pe),
            )
            self.conn.commit()

    def get_pe_history(self, ticker):
        with self.lock:
            rows = self.conn.execute(
                "SELECT pe FROM pe_history WHERE ticker=? ORDER BY date",
                (ticker,),
            ).fetchall()
        return [r[0] for r in rows if r[0] is not None]

    # -- signals ---------------------------------------------------------------
    def log_signal(self, dt, ticker, setup, scores, price, reasons):
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO signals VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    ticker,
                    setup,
                    scores["composite"],
                    scores["coverage"],
                    price,
                    scores["opp"],
                    scores["stab"],
                    scores["trend"],
                    scores["val"],
                    scores["mom"],
                    json.dumps(reasons),
                ),
            )
            self.conn.commit()

    # -- backtest --------------------------------------------------------------
    def insert_backtest(self, run_id, records):
        if not records:
            return
        rows = [
            (
                run_id,
                r["signal_date"],
                r["ticker"],
                r["setup"],
                r["regime"],
                r["composite"],
                r["entry_price"],
                r["ret_1d"],
                r["ret_3d"],
                r["ret_5d"],
                r["ret_10d"],
                r["ret_20d"],
                r["ret_30d"],
                r["ex_20d"],
                r["mfe"],
                r["mae"],
            )
            for r in records
        ]
        with self.lock:
            self.conn.executemany(
                "INSERT INTO backtest_results VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            self.conn.commit()

    def load_run(self, run_id):
        with self.lock:
            return pd.read_sql_query(
                "SELECT * FROM backtest_results WHERE run_id=?",
                self.conn,
                params=(run_id,),
            )

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


# =============================================================================
# PRICE DATA HELPERS
# =============================================================================

def flatten_yfinance_columns(df):
    """Collapse yfinance MultiIndex columns to plain OHLCV names."""
    if df is None or df.empty or not isinstance(df.columns, pd.MultiIndex):
        return df
    fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    new_cols = []
    for col in df.columns:
        pick = next((c for c in col if c in fields), None)
        new_cols.append(pick if pick else "_".join(str(c) for c in col if c))
    out = df.copy()
    out.columns = new_cols
    return out.loc[:, ~out.columns.duplicated()]


def clean_price_dataframe(df):
    """OHLCV frame with numeric columns and a tz-naive, normalised date index."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = flatten_yfinance_columns(df.copy())
    req = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in req):
        return pd.DataFrame()
    df = df[req].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Close"])
    if df.empty:
        return pd.DataFrame()
    for c in ("Open", "High", "Low"):
        df[c] = df[c].fillna(df["Close"])
    df["Volume"] = df["Volume"].fillna(0)

    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def extract_ticker_dataframe(all_data, ticker):
    if all_data is None or all_data.empty:
        return pd.DataFrame()
    if not isinstance(all_data.columns, pd.MultiIndex):
        return all_data.copy()
    try:
        if ticker in all_data.columns.get_level_values(0):
            df = all_data[ticker]
        elif ticker in all_data.columns.get_level_values(1):
            df = all_data.xs(ticker, axis=1, level=1)
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    return df.dropna(how="all").copy()


def drop_incomplete_candle(df):
    """Drop today's bar if the NSE session has not closed yet (IST)."""
    if df.empty:
        return df
    now = now_ist()
    if df.index[-1].date() == now.date() and (now.hour, now.minute) < (15, 45):
        return df.iloc[:-1]
    return df


# =============================================================================
# CONSTITUENTS / INDEX
# =============================================================================

def fetch_index_constituents(timeout: int = 15) -> list[str]:
    try:
        import requests

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
            )
        }
        resp = requests.get(NSE_MIDCAP100_CSV, headers=headers, timeout=timeout)
        resp.raise_for_status()
        table = pd.read_csv(io.StringIO(resp.text))
        col = next((c for c in table.columns if "symbol" in c.lower()), None)
        if col is None:
            raise ValueError("no symbol column")
        tickers = sorted(
            {f"{str(s).strip().upper()}.NS" for s in table[col].dropna()}
        )
        if len(tickers) < 50:
            raise ValueError(f"too few constituents ({len(tickers)})")
        cprint(f"✅ Loaded {len(tickers)} live Nifty Midcap 100 constituents.", "green")
        return tickers
    except Exception as exc:
        cprint(f"⚠️ Live constituent fetch failed ({exc}); using fallback list.", "yellow")
        return list(MIDCAP_TICKERS)


def resolve_tickers(args) -> list[str]:
    if args.tickers_file:
        raw = [
            l.strip().upper()
            for l in Path(args.tickers_file).read_text().splitlines()
            if l.strip() and not l.startswith("#")
        ]
        tickers = [t if t.endswith(".NS") else f"{t}.NS" for t in raw]
    else:
        tickers = fetch_index_constituents()
    return sorted(set(tickers))


def load_index(period: str):
    """First usable index from INDEX_CANDIDATES, with indicators. Logs choice."""
    for name in INDEX_CANDIDATES:
        try:
            raw = yf.download(
                name, period=period, progress=False,
                auto_adjust=True, threads=False,
            )
            df = clean_price_dataframe(raw)
            if len(df) >= MIN_BARS:
                cprint(f"📈 Benchmark index: {name} ({len(df)} bars)", "cyan")
                return name, calculate_indicators(df)
            cprint(f"⚠️ Index {name} returned too little data.", "yellow")
        except Exception as e:
            cprint(f"⚠️ Index {name} failed: {e}", "yellow")
    cprint("⚠️ No benchmark index available; relative strength disabled.", "red")
    return None, None


# =============================================================================
# TECHNICAL ANALYSIS
# =============================================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Causal indicators only (safe to compute once and slice by date)."""
    df = clean_price_dataframe(df)
    if len(df) < MIN_BARS:
        return df
    df = df.copy()
    close = df["Close"]

    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()
    df["SMA50_slope"] = df["SMA50"].pct_change(10) * 100
    df["SMA200_slope"] = df["SMA200"].pct_change(20) * 100

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - close.shift()).abs(),
            (df["Low"] - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    for d in (20, 60, 252):
        df[f"DD_{d}"] = (close / close.rolling(d, min_periods=1).max() - 1) * 100

    df["Vol_SMA20"] = df["Volume"].rolling(20).mean()
    df["Rel_Vol"] = df["Volume"] / df["Vol_SMA20"].replace(0, np.nan)
    df["ADV20"] = (close * df["Volume"]).rolling(20).mean()
    df["ZeroVol60"] = (df["Volume"] == 0).rolling(60).sum()
    return df


def find_swing_points(high: np.ndarray, low: np.ndarray, order: int = 3):
    """
    Vectorised swing detection. A swing high must be strictly above the
    `order` bars before it and >= the `order` bars after (avoids duplicate
    points on flat tops). Last `order` bars can't qualify, so every point
    returned is already confirmed.
    """
    n = len(high)
    w = 2 * order + 1
    if n < w:
        return [], []
    hw = sliding_window_view(high, w)
    lw = sliding_window_view(low, w)
    mid_h, mid_l = hw[:, order], lw[:, order]
    is_high = (mid_h > hw[:, :order].max(axis=1)) & (mid_h >= hw[:, order + 1:].max(axis=1))
    is_low = (mid_l < lw[:, :order].min(axis=1)) & (mid_l <= lw[:, order + 1:].min(axis=1))
    hi = np.where(is_high)[0] + order
    lo = np.where(is_low)[0] + order
    return (
        [(int(i), float(high[i])) for i in hi],
        [(int(i), float(low[i])) for i in lo],
    )


def _recent_pair(points, n, max_age, min_gap):
    """Last two swing points if the latest is recent and they are spaced out."""
    if len(points) < 2:
        return None
    (i1, p1), (i2, p2) = points[-2], points[-1]
    if (n - 1 - i2) > max_age or (i2 - i1) < min_gap:
        return None
    return (i1, p1), (i2, p2)


def relative_strength(df, index_df, n_bars):
    """Stock return minus index return over n_bars, in %. None if unavailable."""
    if index_df is None or len(df) <= n_bars or index_df.empty:
        return None
    ic = index_df["Close"].reindex(df.index, method="ffill")
    s0, s1 = df["Close"].iloc[-1 - n_bars], df["Close"].iloc[-1]
    i0, i1 = ic.iloc[-1 - n_bars], ic.iloc[-1]
    if any(pd.isna(x) or x <= 0 for x in (s0, s1, i0, i1)):
        return None
    return float((s1 / s0 - 1) * 100 - (i1 / i0 - 1) * 100)


def analyze_technicals(df: pd.DataFrame, index_df: pd.DataFrame | None = None):
    """`df` must already carry indicators and end at the evaluation bar."""
    if df is None or len(df) < MIN_BARS or "SMA200" not in df.columns:
        return None
    n = len(df)
    last = df.iloc[-1]
    price = safe_float(last["Close"])
    if price <= 0:
        return None

    rsi_arr = df["RSI"].to_numpy(float)
    rsi = safe_float(last["RSI"], 50.0)
    atr = safe_float(last["ATR"], price * 0.02)

    highs, lows = find_swing_points(
        df["High"].to_numpy(float), df["Low"].to_numpy(float)
    )

    # Higher low / bullish RSI divergence on a recent, well-spaced swing pair
    low_pair = _recent_pair(lows, n, max_age=40, min_gap=5)
    hl = rsi_div = False
    if low_pair:
        (i1, p1), (i2, p2) = low_pair
        hl = p2 > p1
        r1, r2 = rsi_arr[i1], rsi_arr[i2]
        rsi_div = bool(p2 < p1 and r2 > r1 and r1 < 45)

    high_pair = _recent_pair(highs, n, max_age=60, min_gap=5)
    hh = bool(high_pair and high_pair[1][1] > high_pair[0][1])

    # Volume dry-up on the decline leg (selling pressure fading)
    vol = df["Volume"].to_numpy(float)
    vol_dryup = False
    if highs:
        hi_idx = highs[-1][0]
        leg = n - 1 - hi_idx
        if 12 <= leg <= 150:
            decline_vol = vol[hi_idx: n - 5].mean()
            recent_vol = vol[-5:].mean()
            vol_dryup = bool(
                decline_vol > 0 and recent_vol > 0 and recent_vol / decline_vol < 0.75
            )

    # Capitulation held: recent high-volume DOWN bar whose low later holds
    rel = df["Rel_Vol"].to_numpy(float)
    close = df["Close"].to_numpy(float)
    low_arr = df["Low"].to_numpy(float)
    cap_held = False
    for k in range(n - 1, max(n - 16, 1), -1):
        if rel[k] >= 2.5 and close[k] < close[k - 1]:
            if k < n - 3 and low_arr[k + 1:].min() >= low_arr[k] * 0.995:
                cap_held = True
            break

    dd_60 = safe_float(last["DD_60"])
    atr_pct = atr / price * 100
    vol_norm_dd = abs(dd_60) / atr_pct if atr_pct > 0 else 0.0

    def above(col):
        v = last[col]
        return bool(price > v) if not pd.isna(v) else False

    return {
        "curr_price": price,
        "curr_rsi": rsi,
        "hl": hl,
        "hh": hh,
        "rsi_div": rsi_div,
        "vol_dryup": vol_dryup,
        "cap_held": cap_held,
        "n_lows": len(lows),
        "rs_20": relative_strength(df, index_df, 20),
        "rs_60": relative_strength(df, index_df, 60),
        "vol_norm_dd": vol_norm_dd,
        "sma50_slope": safe_float(last["SMA50_slope"]),
        "sma200_slope": safe_float(last["SMA200_slope"]),
        "above_20dma": above("SMA20"),
        "above_50dma": above("SMA50"),
        "above_200dma": above("SMA200"),
        "dd_20": safe_float(last["DD_20"]),
        "dd_60": dd_60,
        "dd_252": safe_float(last["DD_252"]),
    }


# =============================================================================
# FUNDAMENTALS (bundle fetch + point-in-time growth)
# =============================================================================

def _extract_series(frame, keys):
    if frame is None or getattr(frame, "empty", True):
        return {}
    for k in keys:
        if k in frame.index:
            s = frame.loc[k].dropna()
            out = {}
            for c, v in s.items():
                fv = optional_float(v)
                if fv is not None:
                    out[pd.Timestamp(c).strftime("%Y-%m-%d")] = fv
            return out
    return {}


def _parse_earnings_dates(cal):
    out = []
    try:
        if isinstance(cal, dict):
            vals = cal.get("Earnings Date", [])
        elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.columns:
            vals = cal["Earnings Date"].dropna().tolist()
        else:
            vals = []
        if not isinstance(vals, (list, tuple)):
            vals = [vals]
        for e in vals:
            try:
                out.append(pd.Timestamp(e).strftime("%Y-%m-%d"))
            except Exception:
                continue
    except Exception:
        pass
    return out


def fetch_bundle(ticker: str):
    """Network fetch: info subset + dated financial series + earnings dates."""
    t = yf.Ticker(ticker)
    info = {}
    try:
        raw = t.info or {}
        for k in (
            "sector", "trailingPE", "revenueGrowth", "earningsGrowth",
            "returnOnEquity", "returnOnAssets", "debtToEquity",
        ):
            if k in raw:
                info[k] = raw[k]
    except Exception:
        pass

    fin = {"q_rev": {}, "q_ni": {}, "a_rev": {}, "a_ni": {}}
    try:
        q = t.quarterly_financials
        fin["q_rev"] = _extract_series(q, REV_KEYS)
        fin["q_ni"] = _extract_series(q, NI_KEYS)
    except Exception:
        pass
    try:
        a = t.financials
        fin["a_rev"] = _extract_series(a, REV_KEYS)
        fin["a_ni"] = _extract_series(a, NI_KEYS)
    except Exception:
        pass

    try:
        edates = _parse_earnings_dates(t.calendar)
    except Exception:
        edates = []

    if not info and not any(fin.values()):
        return None
    return {"info": info, "fin": fin, "earnings_dates": edates}


def get_bundle(ticker: str, db: ScannerDB):
    cached = db.get_bundle(ticker)
    if cached and "fin" in cached:
        return cached
    bundle = fetch_bundle(ticker)
    if bundle:
        db.put_bundle(ticker, bundle)
    return bundle


def fetch_bundles(tickers, db, workers):
    out = {}
    pbar = tqdm(total=len(tickers), desc="Fundamentals", unit="stock") if HAS_TQDM else None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(get_bundle, t, db): t for t in tickers}
        for f in as_completed(futs):
            if pbar:
                pbar.update(1)
            try:
                out[futs[f]] = f.result()
            except Exception:
                out[futs[f]] = None
    if pbar:
        pbar.close()
    return out


def _yoy(series: dict, as_of, lag_days: int, max_stale_days: int):
    """YoY growth using only statements that were public by `as_of`."""
    if not series:
        return None
    as_of = pd.Timestamp(as_of)
    cutoff = as_of - pd.Timedelta(days=lag_days)
    pts = sorted(
        ((pd.Timestamp(k), v) for k, v in series.items() if pd.Timestamp(k) <= cutoff),
        reverse=True,
    )
    if not pts or (as_of - pts[0][0]).days > max_stale_days:
        return None
    d0, v0 = pts[0]
    for d1, v1 in pts[1:]:
        if 330 <= (d0 - d1).days <= 400:
            if v1 is None or v1 <= 0 or v0 is None:
                return None
            return v0 / v1 - 1
    return None


def pit_growth(fin: dict, kind: str, as_of):
    """Quarterly YoY first (45d reporting lag), annual YoY as fallback (90d)."""
    g = _yoy(fin.get(f"q_{kind}", {}), as_of, 45, 200)
    if g is None:
        g = _yoy(fin.get(f"a_{kind}", {}), as_of, 90, 500)
    return g


def analyze_fundamentals(bundle, as_of, pe_history, live: bool):
    """
    live=True : may use snapshot fields from Yahoo `info` (PE, ROE, ROA, D/E).
    live=False: backtest — only dated financials; snapshot fields are ignored
                so today's values cannot leak into past dates. Valuation
                pillar stays neutral (50).
    """
    bundle = bundle or {}
    info = bundle.get("info") or {}
    fin = bundle.get("fin") or {}
    sector = str(info.get("sector") or "Unknown")
    is_financial = any(s in sector.lower() for s in ("bank", "financial", "insurance"))

    rev_g = pit_growth(fin, "rev", as_of)
    earn_g = pit_growth(fin, "ni", as_of)

    pe = roe = roa = de = None
    if live:
        if rev_g is None:
            rev_g = optional_float(info.get("revenueGrowth"))
        if earn_g is None:
            earn_g = optional_float(info.get("earningsGrowth"))
        pe = optional_float(info.get("trailingPE"))
        roe = optional_float(info.get("returnOnEquity"))
        roa = optional_float(info.get("returnOnAssets"))
        de = optional_float(info.get("debtToEquity"))

    valid_pe_hist = [x for x in (optional_float(v) for v in pe_history) if x and x > 0]
    pe_median = (
        float(np.median(valid_pe_hist)) if len(valid_pe_hist) >= MIN_PE_SNAPSHOTS else None
    )

    # Earnings within 7 days (live only; calendar is a forward-looking snapshot)
    earnings_risk = False
    if live:
        ref = pd.Timestamp(as_of)
        for e in bundle.get("earnings_dates", []):
            try:
                if 0 <= (pd.Timestamp(e) - ref).days <= 7:
                    earnings_risk = True
                    break
            except Exception:
                continue

    val, mom = 50.0, 50.0
    reasons = []
    rejected, reject_reason = False, ""

    if is_financial:
        if roa is not None and roa < 0.005:
            rejected, reject_reason = True, "Poor ROA for Financial"
        if roe is not None and roe < 0.08:
            rejected, reject_reason = True, "Poor ROE for Financial"
        if live:
            reasons.append(
                "Financial sector: "
                f"ROA={'N/A' if roa is None else f'{roa:.2%}'}, "
                f"ROE={'N/A' if roe is None else f'{roe:.2%}'}"
            )
    else:
        if de is not None and de > 200:
            rejected, reject_reason = True, f"High Debt/Equity ({de:.0f}%)"
        if roe is not None:
            if roe > 0.15:
                val += 15
                reasons.append(f"Strong ROE ({roe:.1%})")
            elif roe < 0.08:
                val -= 15
                reasons.append(f"Low ROE ({roe:.1%})")

    if pe is not None and pe > 0 and pe_median is not None:
        if pe < pe_median * 0.8:
            val += 25
            reasons.append(
                f"Cheap vs own history: PE {pe:.1f} < median {pe_median:.1f} "
                f"(n={len(valid_pe_hist)} snapshots)"
            )
        elif pe > pe_median * 1.5:
            val -= 25
            reasons.append(
                f"Rich vs own history: PE {pe:.1f} > median {pe_median:.1f} "
                f"(n={len(valid_pe_hist)} snapshots)"
            )

    if rev_g is not None:
        if rev_g > 0.15:
            mom += 20
            reasons.append(f"Strong revenue growth ({rev_g:.1%} YoY)")
        elif rev_g < -0.05:
            mom -= 30
            reasons.append(f"Weak revenue growth ({rev_g:.1%} YoY)")
    if earn_g is not None:
        if earn_g > 0.15:
            mom += 20
            reasons.append(f"Strong earnings growth ({earn_g:.1%} YoY)")
        elif earn_g < -0.10:
            mom -= 30
            reasons.append(f"Weak earnings growth ({earn_g:.1%} YoY)")

    return {
        "val": clamp(val),
        "mom": clamp(mom),
        "reasons": reasons,
        "rejected": rejected,
        "reject_reason": reject_reason,
        "is_financial": is_financial,
        "pe": pe,
        "pe_median": pe_median,
        "rev_g": rev_g,
        "earn_g": earn_g,
        "earnings_risk": earnings_risk,
        "sector": sector,
    }


# =============================================================================
# REGIME / SCORING
# =============================================================================

def calculate_regime_and_scores(tech: dict, fund: dict):
    rs20 = tech["rs_20"] if tech["rs_20"] is not None else 0.0
    rs60 = tech["rs_60"] if tech["rs_60"] is not None else 0.0
    rsi = tech["curr_rsi"]
    dd60 = tech["dd_60"]
    vn = tech["vol_norm_dd"]
    reasons = []

    # Opportunity: depth of decline (raw %) + volatility-adjusted depth
    opp = min(-dd60, 40) / 40 * 60 + min(vn, 6) / 6 * 40 if dd60 < 0 else 0.0
    reasons.append(f"60d drawdown {dd60:.1f}% | {vn:.1f} ATRs deep")

    # Stabilization
    stab = 10.0
    if 30 <= rsi <= 45:
        stab += 20
    elif rsi < 25:
        stab -= 10
    if tech["hl"]:
        stab += 25
        reasons.append("Recent higher low")
    if tech["rsi_div"]:
        stab += 25
        reasons.append("Bullish RSI divergence")
    if tech["vol_dryup"]:
        stab += 15
        reasons.append("Volume drying up vs decline leg")
    if tech["cap_held"]:
        stab += 15
        reasons.append("High-volume capitulation bar, low has held")
    if tech["above_20dma"]:
        stab += 10

    # Trend
    trend = 20.0
    if tech["above_200dma"]:
        trend += 20
    if tech["above_50dma"]:
        trend += 10
    if tech["sma200_slope"] > 0:
        trend += 15
    if rs20 > 2:
        trend += 15
        reasons.append(f"Outperforming index (20d RS {rs20:+.1f}%)")
    if rs60 > 0:
        trend += 10
    if tech["hh"]:
        trend += 10

    opp, stab, trend = clamp(opp), clamp(stab), clamp(trend)
    val, mom = fund["val"], fund["mom"]

    falling_knife = (
        not tech["above_50dma"]
        and not tech["above_200dma"]
        and tech["sma50_slope"] < -2
        and tech["sma200_slope"] < 0
        and not tech["hl"]
        and rs20 < 0
    )
    quality_pullback = (
        tech["above_200dma"]
        and 30 <= rsi <= 50
        and tech["hl"]
        and val >= 50
        and mom > 60
        and rs20 > 0
    )
    momentum_pullback = (
        tech["above_50dma"]
        and tech["above_200dma"]
        and 10 <= -dd60 <= 18
        and rs20 > 2
        and mom > 70
    )
    reversal = (
        (tech["rsi_div"] or tech["cap_held"])
        and tech["vol_dryup"]
        and not tech["above_200dma"]
    )

    setup, mult = "Neutral", 1.0
    if falling_knife:
        setup, mult = "Falling Knife", 0.3
        reasons.append("⚠️ REGIME: Falling Knife (both DMAs broken, 50DMA falling, no higher low)")
    elif mom < 30:
        setup, mult = "Fundamental Deterioration", 0.5
        reasons.append("📉 REGIME: Fundamental Deterioration (weak growth)")
    elif quality_pullback:
        setup, mult = "Quality Pullback", 1.2
        reasons.append("✅ REGIME: Quality Pullback (above 200DMA, higher low, healthy growth)")
    elif momentum_pullback:
        setup, mult = "Momentum Pullback", 1.1
        reasons.append("🔥 REGIME: Momentum Pullback (uptrend intact, strong RS)")
    elif reversal:
        setup, mult = "Reversal", 0.9
        reasons.append("🔄 REGIME: Reversal (divergence/capitulation + volume dry-up)")

    raw = opp * 0.20 + stab * 0.25 + trend * 0.20 + val * 0.15 + mom * 0.20
    composite = clamp(raw * mult)
    if fund["earnings_risk"]:
        composite = clamp(composite - 10)
        reasons.append("⏰ Earnings within 7 days (-10)")

    # Coverage = share of key inputs actually available (NOT conviction)
    have = [
        tech["rs_20"] is not None,
        tech["n_lows"] >= 2,
        fund["pe"] is not None,
        fund["pe_median"] is not None,
        fund["rev_g"] is not None,
        fund["earn_g"] is not None,
    ]
    scores = {
        "opp": round(opp, 1),
        "stab": round(stab, 1),
        "trend": round(trend, 1),
        "val": round(val, 1),
        "mom": round(mom, 1),
        "composite": round(composite, 1),
        "coverage": round(sum(have) / len(have) * 100, 1),
    }
    return setup, scores, reasons


# =============================================================================
# LIVE SCAN
# =============================================================================

def process_ticker(ticker, raw_df, index_df, db, args):
    try:
        df = drop_incomplete_candle(clean_price_dataframe(raw_df))
        if len(df) < MIN_BARS:
            return "filtered", "Insufficient history"

        last_date = df.index[-1].date()
        if (now_ist().date() - last_date).days > 7:
            return "filtered", "Stale data (last bar > 7 days old)"
        if (df["Volume"].iloc[-60:] == 0).sum() > 5:
            return "filtered", "Too many zero-volume days (last 60)"

        df = calculate_indicators(df)
        adv = safe_float(df["ADV20"].iloc[-1])
        if adv < args.min_adv_cr * 1e7:
            return "filtered", f"Low liquidity (< {args.min_adv_cr:g} Cr ADV)"

        tech = analyze_technicals(df, index_df)
        if tech is None:
            return "filtered", "Technical calc failed"
        if tech["dd_60"] > -args.min_dd:
            return "filtered", f"60d drawdown smaller than {args.min_dd:g}%"

        bundle = get_bundle(ticker, db)
        if not bundle:
            return "filtered", "No fundamentals data"

        sig_date = pd.Timestamp(last_date)
        pe_hist = db.get_pe_history(ticker)
        db.cache_pe(ticker, sig_date, (bundle.get("info") or {}).get("trailingPE"))
        fund = analyze_fundamentals(bundle, sig_date, pe_hist, live=True)
        if fund["rejected"]:
            return "filtered", fund["reject_reason"]

        setup, scores, reasons = calculate_regime_and_scores(tech, fund)
        bad = setup in BAD_SETUPS
        if bad and not args.lenient:
            return "filtered", f"Rejected setup: {setup}"
        if scores["composite"] < args.min_score and not (bad and args.lenient):
            return "filtered", f"Score below {args.min_score:g}"

        db.log_signal(sig_date, ticker, setup, scores, tech["curr_price"], reasons)

        all_reasons = reasons + fund["reasons"]
        return "ok", {
            "Ticker": ticker.replace(".NS", ""),
            "Setup": setup,
            "Coverage": scores["coverage"],
            "Score": scores["composite"],
            "Price": tech["curr_price"],
            "RSI": tech["curr_rsi"],
            "DD_60": tech["dd_60"],
            "RS_20": tech["rs_20"],
            "PE": fund["pe"],
            "Sector": fund["sector"][:18],
            "Regime_Reasons": [r for r in all_reasons if "REGIME" in r],
            "Reasons": [r for r in all_reasons if "REGIME" not in r],
        }
    except Exception as e:  # one bad ticker must never kill the scan
        return "error", f"{type(e).__name__}: {e}"


def scan_stocks(args):
    db = ScannerDB()
    try:
        cprint("⏳ Fetching benchmark index...", "cyan")
        idx_name, index_df = load_index("3y")

        market_regime = "Unknown"
        if index_df is not None and "SMA200" in index_df and len(index_df) > 200:
            c, s = safe_float(index_df["Close"].iloc[-1]), safe_float(index_df["SMA200"].iloc[-1])
            if s > 0:
                market_regime = "Bullish (index > 200DMA)" if c > s else "Bearish (index < 200DMA)"

        tickers = resolve_tickers(args)
        cprint(f"⏳ Downloading 3y of prices for {len(tickers)} tickers...", "cyan")
        try:
            all_data = yf.download(
                tickers, period="3y", group_by="ticker",
                threads=True, progress=False, auto_adjust=True,
            )
        except Exception as e:
            cprint(f"❌ Data download failed: {e}", "red")
            return
        if all_data is None or all_data.empty:
            cprint("❌ Yahoo returned no price data.", "red")
            return

        results, errors, filtered = [], [], Counter()
        pbar = tqdm(total=len(tickers), desc="Scanning", unit="stock") if HAS_TQDM else None

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for t in tickers:
                df = extract_ticker_dataframe(all_data, t)
                if df.empty:
                    filtered["No price data"] += 1
                    if pbar:
                        pbar.update(1)
                    continue
                futures[pool.submit(process_ticker, t, df, index_df, db, args)] = t

            for fut in as_completed(futures):
                if pbar:
                    pbar.update(1)
                t = futures[fut]
                try:
                    status, payload = fut.result()
                except Exception as e:
                    errors.append(f"{t}: {type(e).__name__}: {e}")
                    continue
                if status == "ok":
                    results.append(payload)
                elif status == "filtered":
                    filtered[payload] += 1
                else:
                    errors.append(f"{t}: {payload}")
        if pbar:
            pbar.close()
    finally:
        db.close()

    today = now_ist().date()
    print("\n" + "=" * 100)
    cprint(
        f"  MIDCAP SCANNER — {today} | Benchmark: {idx_name or 'none'} | "
        f"Regime: {market_regime}",
        "cyan",
    )
    cprint(
        f"  Gate: 60d drawdown ≥ {args.min_dd:g}% | min score {args.min_score:g} | "
        f"min ADV {args.min_adv_cr:g} Cr",
        "cyan",
    )
    print("=" * 100)

    if filtered:
        cprint("\nExcluded by filters:", "white")
        for reason, n in filtered.most_common(15):
            print(f"   {n:>3} × {reason}")
    if errors:
        cprint("\nTicker errors:", "yellow")
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
        color = (
            "green" if row["Setup"] in ("Quality Pullback", "Momentum Pullback")
            else "red" if row["Setup"] in BAD_SETUPS
            else "white"
        )
        rs = "n/a" if pd.isna(row["RS_20"]) else f"{row['RS_20']:+.1f}%"
        pe = "-" if pd.isna(row["PE"]) else f"{row['PE']:.1f}"
        cprint(f"\n{'─' * 90}", color)
        cprint(
            f"  {row['Setup']} | Score {row['Score']} | Coverage {row['Coverage']}%",
            color,
        )
        cprint(
            f"  {row['Ticker']} | ₹{row['Price']:.2f} | {row['Sector']} | "
            f"RSI {row['RSI']:.1f} | DD60 {row['DD_60']:.1f}% | RS20 {rs} | PE {pe}",
            color,
        )
        for r in row["Regime_Reasons"]:
            print(f"    {r}")
        for r in row["Reasons"]:
            print(f"      • {r}")

    out = df_res.copy()
    out["Regime_Reasons"] = out["Regime_Reasons"].apply(" | ".join)
    out["Reasons"] = out["Reasons"].apply(" | ".join)
    out_file = f"{args.output_prefix}_{today}.csv"
    out.to_csv(out_file, index=False)
    cprint(f"\n💾 Saved {len(out)} results to {out_file}", "green")


# =============================================================================
# BACKTEST
# =============================================================================

def run_backtest(args):
    db = ScannerDB()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    cprint(f"🚀 Backtest run {run_id}", "magenta")
    cprint(
        "⚠️ Caveats: current constituents (survivorship bias); fundamentals are "
        "point-in-time growth only (valuation pillar neutral); samples are "
        "de-overlapped by cooldown but still correlated within market regimes.",
        "yellow",
    )

    records, errs = [], Counter()
    try:
        tickers = resolve_tickers(args)
        cprint(f"⏳ Downloading 4y of prices for {len(tickers)} tickers...", "cyan")
        all_data = yf.download(
            tickers, period="4y", group_by="ticker",
            threads=True, progress=False, auto_adjust=True,
        )
        idx_name, index_df = load_index("4y")

        prices = {}
        for t in tickers:
            df = clean_price_dataframe(extract_ticker_dataframe(all_data, t))
            if len(df) >= MIN_BARS + MAX_HORIZON + 5:
                prices[t] = calculate_indicators(df)
        cprint(f"   {len(prices)} tickers have enough history.", "cyan")
        if not prices:
            cprint("❌ No usable price history.", "red")
            return

        cprint("⏳ Loading fundamentals (cached 7 days)...", "cyan")
        bundles = fetch_bundles(list(prices), db, args.workers)

        regime_series = None
        if index_df is not None and "SMA200" in index_df:
            regime_series = index_df["Close"] > index_df["SMA200"]

        cost_pct = args.cost_bps / 100.0
        start = pd.Timestamp(now_ist().date()) - pd.Timedelta(days=730)
        end = pd.Timestamp(now_ist().date()) - pd.Timedelta(days=45)
        test_dates = pd.date_range(start, end, freq=f"{args.backtest_days}B")
        cprint(f"⏳ {len(test_dates)} checkpoints × {len(prices)} tickers...", "cyan")

        last_signal = {}
        pbar = tqdm(total=len(test_dates), desc="Backtesting") if HAS_TQDM else None

        for dt in test_dates:
            if pbar:
                pbar.update(1)

            regime = "Unknown"
            idx_slice = None
            if index_df is not None:
                ip = index_df.index.searchsorted(dt, side="right") - 1
                if ip >= 0:
                    idx_slice = index_df.iloc[: ip + 1]
                    if regime_series is not None and not pd.isna(index_df["SMA200"].iloc[ip]):
                        regime = "Bull" if regime_series.iloc[ip] else "Bear"

            for ticker, df in prices.items():
                try:
                    pos = df.index.searchsorted(dt, side="right") - 1
                    if pos < MIN_BARS - 1 or pos + MAX_HORIZON >= len(df):
                        continue
                    sig_day = df.index[pos]
                    if (dt - sig_day).days > 4:  # ticker not trading around dt
                        continue
                    prev = last_signal.get(ticker)
                    if prev is not None and (sig_day - prev).days < args.cooldown_days:
                        continue
                    if df["ADV20"].iloc[pos] < args.min_adv_cr * 1e7:
                        continue
                    if df["ZeroVol60"].iloc[pos] > 5:
                        continue

                    window = df.iloc[: pos + 1]
                    tech = analyze_technicals(window, idx_slice)
                    if tech is None or tech["dd_60"] > -args.min_dd:
                        continue

                    fund = analyze_fundamentals(bundles.get(ticker), sig_day, [], live=False)
                    if fund["rejected"]:
                        continue

                    setup, scores, _ = calculate_regime_and_scores(tech, fund)
                    bad = setup in BAD_SETUPS
                    if bad and not args.lenient:
                        continue
                    if scores["composite"] < args.min_score and not (bad and args.lenient):
                        continue

                    # Enter at NEXT session's open; exit horizons are closes
                    entry = float(df["Open"].iloc[pos + 1])
                    if not entry > 0:
                        continue
                    entry_day = df.index[pos + 1]
                    fut_days = df.index[pos + 1: pos + 1 + MAX_HORIZON]
                    rets = {
                        h: (float(df["Close"].iloc[pos + h]) / entry - 1) * 100 - cost_pct
                        for h in HORIZONS
                    }
                    mfe = (float(df["High"].iloc[pos + 1: pos + 1 + MAX_HORIZON].max()) / entry - 1) * 100
                    mae = (float(df["Low"].iloc[pos + 1: pos + 1 + MAX_HORIZON].min()) / entry - 1) * 100

                    ex20 = None
                    if index_df is not None:
                        ic = index_df["Close"].reindex(fut_days, method="ffill")
                        io = index_df["Open"].reindex([entry_day], method="ffill").iloc[0]
                        i20 = ic.iloc[19]
                        if pd.notna(io) and pd.notna(i20) and io > 0:
                            ex20 = rets[20] - ((i20 / io - 1) * 100)

                    last_signal[ticker] = sig_day
                    records.append(
                        {
                            "signal_date": sig_day.strftime("%Y-%m-%d"),
                            "ticker": ticker,
                            "setup": setup,
                            "regime": regime,
                            "composite": scores["composite"],
                            "entry_price": entry,
                            "ret_1d": rets[1],
                            "ret_3d": rets[3],
                            "ret_5d": rets[5],
                            "ret_10d": rets[10],
                            "ret_20d": rets[20],
                            "ret_30d": rets[30],
                            "ex_20d": ex20,
                            "mfe": mfe,
                            "mae": mae,
                        }
                    )
                except Exception as e:
                    errs[f"{type(e).__name__}: {str(e)[:70]}"] += 1
        if pbar:
            pbar.close()

        db.insert_backtest(run_id, records)
    finally:
        db.close()

    if errs:
        cprint("\nBacktest exceptions (counted, not silent):", "yellow")
        for msg, n in errs.most_common(10):
            print(f"   {n:>5} × {msg}")

    cprint(f"\n✅ Backtest complete: {len(records)} signals. run_id={run_id}", "green")
    if records:
        print_backtest_stats(pd.DataFrame(records))
    else:
        cprint("No signals generated (try lowering --min-score or --min-dd).", "yellow")


def _summarize(sub: pd.DataFrame, label: str):
    r = sub["ret_20d"].dropna()
    if r.empty:
        return
    cprint(f"\n  {label} (N={len(r)}, tickers={sub['ticker'].nunique()})", "cyan")
    ex = sub["ex_20d"].dropna()
    print(
        f"    20d net: win {((r > 0).mean() * 100):.1f}% | "
        f"median {r.median():.2f}% | mean {r.mean():.2f}%"
        + (f" | mean excess vs index {ex.mean():.2f}%" if len(ex) else "")
    )
    print(
        f"    30d window: avg MFE {sub['mfe'].mean():.2f}% | avg MAE {sub['mae'].mean():.2f}%"
    )


def print_backtest_stats(df: pd.DataFrame):
    print("\n" + "=" * 100)
    cprint("  BACKTEST STATISTICS (net of costs, entry = next open)", "magenta")
    print("=" * 100)

    _summarize(df, "ALL SIGNALS")
    for setup in sorted(df["setup"].dropna().unique()):
        _summarize(df[df["setup"] == setup], f"SETUP: {setup}")
    for regime in sorted(df["regime"].dropna().unique()):
        _summarize(df[df["regime"] == regime], f"MARKET REGIME: {regime}")

    df = df.copy()
    df["bucket"] = pd.cut(
        df["composite"],
        bins=[0, 50, 60, 70, 80, 100.01],
        labels=["<50", "50-60", "60-70", "70-80", "80+"],
        include_lowest=True,
    )
    print("\n  SCORE BUCKETS (20d net return, %):")
    print(
        df.groupby("bucket", observed=True)["ret_20d"]
        .agg(["count", "mean", "median"])
        .round(2)
        .to_string()
    )


def report_run(run_id):
    db = ScannerDB()
    try:
        df = db.load_run(run_id)
    finally:
        db.close()
    if df.empty:
        cprint(f"No rows for run_id {run_id}.", "yellow")
        return
    print_backtest_stats(df)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Midcap Scanner v2 & Backtester")
    p.add_argument("--backtest", action="store_true", help="Run historical backtest")
    p.add_argument("--report-run", metavar="RUN_ID", help="Re-print stats for a saved backtest run")
    p.add_argument("--backtest-days", type=int, default=5, help="Checkpoint every N business days")
    p.add_argument("--cooldown-days", type=int, default=28, help="Min calendar days between signals per ticker (backtest)")
    p.add_argument("--cost-bps", type=float, default=20.0, help="Round-trip cost in bps deducted from returns")
    p.add_argument("--min-score", type=float, default=55.0, help="Minimum composite score")
    p.add_argument("--min-dd", type=float, default=10.0, help="Required 60d drawdown from high, in %%")
    p.add_argument("--min-adv-cr", type=float, default=5.0, help="Min 20d avg traded value, ₹ crore")
    p.add_argument("--lenient", action="store_true", help="Also show Falling Knife / Deterioration setups")
    p.add_argument("--workers", type=int, default=4, help="Concurrent workers")
    p.add_argument("--top", type=int, default=15, help="Show top N results (0 = all)")
    p.add_argument("--output-prefix", default="midcap_scan", help="CSV output prefix")
    p.add_argument("--tickers-file", metavar="PATH", help="Custom ticker list (one per line)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.report_run:
        report_run(args.report_run)
    elif args.backtest:
        run_backtest(args)
    else:
        scan_stocks(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())