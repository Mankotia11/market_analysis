#!/usr/bin/env python3
"""
Midcap Stock Scanner — Nifty Midcap 100 (stateless edition)
===========================================================

Finds fundamentally healthy stocks with an unusually large decline where
selling pressure is weakening and relative strength is improving.

This version keeps NO state between runs:
  - no SQLite database, no caches, no PE-history accumulation
  - no CSV / result files written
Every run starts clean and fetches fresh data, so results do not drift
because of leftover data from earlier runs.

Usage:
  python midcap_scanner.py                  # live scan
  python midcap_scanner.py --backtest       # historical backtest
  python midcap_scanner.py --lenient --top 30 --min-score 45
"""

from __future__ import annotations

import argparse
import io
import warnings
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

NSE_MIDCAP100_CSV = (
    "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv"
)

INDEX_TICKERS = ["NIFTY_MIDCAP_100.NS", "^NSEI"]

# Fallback snapshot ONLY if live NSE download fails.
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
HISTORY_DAYS = 1100


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


def load_tickers(args) -> list[str]:
    tickers = fetch_index_constituents()
    if args.tickers_file:
        tickers = [
            l.strip().upper()
            for l in Path(args.tickers_file).read_text().splitlines()
            if l.strip()
        ]
        tickers = [t if t.endswith(".NS") else f"{t}.NS" for t in tickers]
    return sorted(set(tickers))


# =============================================================================
# NSE CONSTITUENTS
# =============================================================================

def fetch_index_constituents(timeout: int = 15) -> list[str]:
    try:
        import requests

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 Chrome/151.0 Safari/537.36"
            )
        }
        resp = requests.get(NSE_MIDCAP100_CSV, headers=headers, timeout=timeout)
        resp.raise_for_status()
        table = pd.read_csv(io.StringIO(resp.text))

        symbol_col = next((c for c in table.columns if "symbol" in c.lower()), None)
        if symbol_col is None:
            raise ValueError("No symbol column found")

        tickers = sorted(
            {f"{str(s).strip().upper()}.NS" for s in table[symbol_col].dropna()}
        )
        if len(tickers) < 50:
            raise ValueError(f"Too few constituents ({len(tickers)})")

        cprint(f"✅ Loaded {len(tickers)} live Nifty Midcap 100 constituents.", "green")
        return tickers

    except Exception as exc:
        cprint(f"⚠️ Live fetch failed ({exc}); using fallback list.", "yellow")
        return list(MIDCAP_TICKERS)


# =============================================================================
# TECHNICAL ANALYSIS
# =============================================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_price_dataframe(df)
    if len(df) < MIN_BARS:
        return df

    df = df.copy()

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    df["SMA20_slope"] = df["SMA20"].pct_change(10) * 100
    df["SMA50_slope"] = df["SMA50"].pct_change(10) * 100
    df["SMA200_slope"] = df["SMA200"].pct_change(20) * 100

    # RSI (Wilder)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ATR
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # Drawdowns
    for d in [5, 10, 20, 30, 60, 252]:
        roll_max = df["Close"].rolling(d, min_periods=1).max()
        df[f"DD_{d}"] = (df["Close"] / roll_max - 1) * 100

    # Volume
    df["Vol_SMA20"] = df["Volume"].rolling(20).mean()
    df["Vol_SMA60"] = df["Volume"].rolling(60).mean()
    df["Rel_Vol"] = df["Volume"] / df["Vol_SMA20"].replace(0, np.nan)

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


def analyze_technicals(df: pd.DataFrame, index_df: pd.DataFrame = None):
    if len(df) < MIN_BARS:
        return None

    df = calculate_indicators(df)
    if df.empty or "RSI" not in df.columns:
        return None

    highs, lows = find_swing_points(df)
    last_i = len(df) - 1

    curr_price = safe_float(df["Close"].iloc[-1])
    curr_rsi = safe_float(df["RSI"].iloc[-1], 50.0)
    curr_atr = safe_float(df["ATR"].iloc[-1], curr_price * 0.02)

    # Swing points need 3 bars after them to be confirmed
    conf_lows = [l for l in lows if l[0] <= last_i - 3]
    conf_highs = [h for h in highs if h[0] <= last_i - 3]

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

    # Volume contraction (decline-phase volume vs recent volume)
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
    dd_60 = safe_float(df["DD_60"].iloc[-1], 0.0)
    atr_pct = (curr_atr / curr_price) * 100 if curr_price > 0 else 0
    vol_norm_dd = abs(dd_60) / atr_pct if atr_pct > 0 else 0

    sma200_last = df["SMA200"].iloc[-1]
    sma50_last = df["SMA50"].iloc[-1]

    return {
        "df": df,
        "curr_price": curr_price,
        "curr_rsi": curr_rsi,
        "curr_atr": curr_atr,
        "hl": hl,
        "hh": hh,
        "rsi_div": rsi_div,
        "vol_near_support": vol_near_support,
        "vol_contract_expand": vol_contract_expand,
        "rs_20": rs_20,
        "rs_60": rs_60,
        "vol_norm_dd": vol_norm_dd,
        "sma20_slope": safe_float(df["SMA20_slope"].iloc[-1]),
        "sma50_slope": safe_float(df["SMA50_slope"].iloc[-1]),
        "sma200_slope": safe_float(df["SMA200_slope"].iloc[-1]),
        "above_200dma": bool(curr_price > sma200_last) if not pd.isna(sma200_last) else False,
        "above_50dma": bool(curr_price > sma50_last) if not pd.isna(sma50_last) else False,
        "dd_5": safe_float(df["DD_5"].iloc[-1]),
        "dd_10": safe_float(df["DD_10"].iloc[-1]),
        "dd_20": safe_float(df["DD_20"].iloc[-1]),
        "dd_60": dd_60,
        "dd_252": safe_float(df["DD_252"].iloc[-1]),
    }


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


def analyze_fundamentals(info, pt_fin, pe_history, current_date, t_obj=None):
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

    earnings_risk = _check_earnings_risk(t_obj, current_date)

    scores = {"val": 50, "mom": 50}
    reasons = []
    rejected = False
    reject_reason = ""

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
        if de is not None and de > 200:
            rejected = True
            reject_reason = f"High Debt/Equity ({de:.0f}%)"
        if roic is not None and roic < 0.08:
            scores["val"] -= 20
            reasons.append(f"Low ROIC ({roic:.2%})")
        elif roic is not None and roic > 0.15:
            scores["val"] += 20
            reasons.append(f"Strong ROIC ({roic:.2%})")

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
        "earnings_risk": earnings_risk,
        "sector": info.get("sector", "Unknown"),
    }


# =============================================================================
# REGIME / SCORING ENGINE
# =============================================================================

def calculate_regime_and_scores(tech: dict, fund: dict):
    scores = {"opp": 50, "stab": 50, "trend": 50}
    reasons = []

    # Opportunity
    dd_60 = tech["dd_60"]
    if dd_60 < -10:
        scores["opp"] += 20
    if dd_60 < -20:
        scores["opp"] += 20
    if tech["vol_norm_dd"] > 3.0:
        scores["opp"] += 15

    reasons.append(
        f"60d Drawdown: {dd_60:.1f}% | Vol-Norm DD: {tech['vol_norm_dd']:.1f} ATRs"
    )

    # Stabilization
    rsi = tech["curr_rsi"]
    if 30 <= rsi <= 45:
        scores["stab"] += 20
    elif rsi < 25:
        scores["stab"] -= 20
    if tech["hl"]:
        scores["stab"] += 25
    if tech["rsi_div"]:
        scores["stab"] += 25
    if tech["vol_contract_expand"] > 1.5:
        scores["stab"] += 15

    # Trend
    if tech["above_200dma"]:
        scores["trend"] += 20
    if tech["above_50dma"]:
        scores["trend"] += 10
    if tech["sma200_slope"] > 0:
        scores["trend"] += 15
    if tech["rs_20"] > 2:
        scores["trend"] += 15

    setup = "Neutral"
    composite_mult = 1.0

    is_falling_knife = (
        not tech["above_50dma"]
        and not tech["above_200dma"]
        and tech["sma50_slope"] < -2
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

    is_momentum_pullback = (
        tech["above_50dma"]
        and abs(dd_60) < 10
        and tech["rs_20"] > 5
        and fund["scores"]["mom"] > 70
    )

    if is_falling_knife:
        setup, composite_mult = "Falling Knife", 0.3
        reasons.append("⚠️ REGIME: Falling Knife (Declining DMAs, Lower Lows, Extreme RSI)")
    elif is_quality_pullback:
        setup, composite_mult = "Quality Pullback", 1.2
        reasons.append("✅ REGIME: Quality Pullback (Healthy Fundamentals, Higher Lows, RS Improving)")
    elif is_momentum_pullback:
        setup, composite_mult = "Momentum Pullback", 1.1
        reasons.append("🔥 REGIME: Momentum Pullback (Strong Uptrend, Shallow Dip, High RS)")
    elif tech["rsi_div"] and tech["vol_contract_expand"] > 1.5 and not tech["above_200dma"]:
        setup, composite_mult = "Reversal", 0.9
        reasons.append("🔄 REGIME: Reversal (Deep Dip, Divergence & Volume Climax)")
    elif fund["scores"]["mom"] < 30:
        setup, composite_mult = "Fundamental Deterioration", 0.5
        reasons.append("📉 REGIME: Fundamental Deterioration (Poor Growth/Margins)")

    composite = (
        scores["opp"] * 0.20
        + scores["stab"] * 0.25
        + scores["trend"] * 0.20
        + fund["scores"]["val"] * 0.15
        + fund["scores"]["mom"] * 0.20
    ) * composite_mult

    valid_signals = sum(
        [
            1 if tech["hl"] else 0,
            1 if tech["rsi_div"] else 0,
            1 if tech["vol_contract_expand"] > 0 else 0,
            1 if tech["rs_20"] != 0 else 0,
            1 if fund["pe"] else 0,
            1 if fund["pe_median"] else 0,
        ]
    )
    confidence = min(100, (valid_signals / 6) * 100 + 20)

    scores["composite"] = round(composite, 1)
    scores["confidence"] = round(confidence, 1)

    return setup, scores, reasons


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

        if (df["Volume"] == 0).sum() > 5:
            return "filtered", "Too many zero-volume days"

        tech = analyze_technicals(df, index_df)
        if not tech:
            return "filtered", "Tech Calc Failed"

        adv = safe_float((df["Close"] * df["Volume"]).rolling(20).mean().iloc[-1], 0)
        if adv < 50_000_000:
            return "filtered", f"Low Liquidity (ADV {adv / 10000000:.1f} Cr)"

        # Always fresh fundamentals (nothing cached between runs)
        try:
            t_obj = yf.Ticker(ticker)
            info = t_obj.info
        except Exception as e:
            return "error", f"Info fetch failed: {e}"

        if not info or not isinstance(info, dict):
            return "filtered", "Empty Info"

        pt_fin = get_point_in_time_financials(t_obj, current_date)

        # No stored PE history -> no self-built "median PE" drifting across runs
        fund = analyze_fundamentals(info, pt_fin, [], current_date, t_obj)

        if fund["rejected"]:
            return "filtered", fund["reject_reason"]

        setup, scores, reasons = calculate_regime_and_scores(tech, fund)

        if setup in ("Falling Knife", "Fundamental Deterioration") and not args.lenient:
            return "filtered", f"Rejected Setup: {setup}"

        if scores["composite"] < args.min_score:
            return "filtered", f"Score {scores['composite']} < {args.min_score}"

        pe_output = f"{fund['pe']:.1f}" if fund["pe"] is not None else "-"

        return "ok", {
            "Ticker": ticker.replace(".NS", ""),
            "Setup": setup,
            "Conv": scores["confidence"],
            "Score": scores["composite"],
            "Price": tech["curr_price"],
            "RSI": tech["curr_rsi"],
            "DD_60": tech["dd_60"],
            "RS_20": tech["rs_20"],
            "PE": pe_output,
            "Sector": str(fund["sector"])[:15],
            "Earnings_Soon": fund["earnings_risk"],
            "Tech_Reasons": [
                r for r in (reasons + fund["reasons"]) if "REGIME" not in r
            ],
            "Regime_Reasons": [r for r in reasons if "REGIME" in r],
        }

    except Exception as e:
        # One bad ticker must never kill the whole scan.
        return "error", f"{type(e).__name__}: {e}"


# =============================================================================
# LIVE SCANNER
# =============================================================================

def get_index_data():
    index_data = {}
    for idx in INDEX_TICKERS:
        try:
            idx_df = yf.download(
                idx, period="3y", progress=False, auto_adjust=True, threads=False
            )
            idx_df = clean_price_dataframe(idx_df)
            if not idx_df.empty:
                index_data[idx] = calculate_indicators(idx_df)
        except Exception as e:
            cprint(f"⚠️ Index {idx} failed: {e}", "yellow")

    return index_data.get("NIFTY_MIDCAP_100.NS") if "NIFTY_MIDCAP_100.NS" in index_data \
        else index_data.get("^NSEI")


def scan_stocks(args):
    current_date = datetime.now().date()

    cprint("⏳ Fetching Market Index Data (Midcap 100 / Nifty 50)...", "cyan")
    index_df = get_index_data()

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
                    filtered[payload] += 1
                else:
                    errors.append(f"{ticker}: {payload}")
            except Exception as e:
                errors.append(f"{ticker}: {type(e).__name__}: {e}")

    if pbar:
        pbar.close()

    # ---- Output ---------------------------------------------------------------
    print("\n" + "=" * 100)
    cprint(f"  MIDCAP SCANNER RESULTS — {current_date} | Market Regime: {market_regime}", "cyan")
    cprint(
        "  Objective: Find fundamentally healthy stocks with weakening selling "
        "pressure & improving RS",
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
        if "Quality" in row["Setup"] or "Momentum" in row["Setup"]:
            color = "green"
        elif "Falling" in row["Setup"]:
            color = "red"
        else:
            color = "white"

        cprint(f"\n{'─' * 90}", color)
        cprint(
            f"  SETUP: {row['Setup']} | CONVICTION: {row['Conv']}% | SCORE: {row['Score']}",
            color,
        )
        earn = " | ⚠️ EARNINGS ≤7d" if row["Earnings_Soon"] else ""
        cprint(
            f"  {row['Ticker']} | ₹{row['Price']:.2f} | {row['Sector']} | "
            f"RSI: {row['RSI']:.1f} | DD60: {row['DD_60']:.1f}% | "
            f"RS20: {row['RS_20']:+.1f}% | PE: {row['PE']}{earn}",
            color,
        )
        for r in row["Regime_Reasons"]:
            print(f"    {r}")
        print("    Technical & Fundamental Reasons:")
        for r in row["Tech_Reasons"]:
            print(f"      • {r}")

    # Nothing is saved to disk by design.
    cprint("\n(Results are shown only; nothing was saved to disk.)", "cyan")


# =============================================================================
# BACKTESTER (in-memory only)
# =============================================================================

def run_backtest(args):
    cprint("🚀 Starting No-Look-Ahead (prices) Backtester...", "magenta")
    cprint(
        "⚠️ Caveats: Yahoo 'info' fundamentals are CURRENT values, not historical, "
        "and the constituent list is today's (survivorship bias). Treat results "
        "as optimistic.",
        "yellow",
    )

    tickers = load_tickers(args)

    cprint(f"⏳ Downloading 3 years of data for {len(tickers)} tickers...", "cyan")
    all_data = yf.download(
        tickers, period="3y", group_by="ticker",
        threads=True, progress=False, auto_adjust=True,
    )

    index_df = clean_price_dataframe(
        yf.download(
            "NIFTY_MIDCAP_100.NS", period="3y",
            progress=False, auto_adjust=True, threads=False,
        )
    )
    if index_df.empty:
        index_df = clean_price_dataframe(
            yf.download("^NSEI", period="3y", progress=False,
                        auto_adjust=True, threads=False)
        )
    if not index_df.empty:
        index_df = calculate_indicators(index_df)

    # Clean each ticker's price frame ONCE
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

    records = []  # in-memory only

    for dt in test_dates:
        if pbar:
            pbar.update(1)

        for ticker, df_full in price_cache.items():
            try:
                # Align checkpoint to the last trading day <= dt
                pos = df_full.index.searchsorted(pd.Timestamp(dt), side="right") - 1
                if pos < MIN_BARS:
                    continue
                if pos + 30 >= len(df_full):
                    continue

                df = df_full.iloc[: pos + 1]
                idx_slice = index_df[index_df.index <= df.index[-1]] \
                    if not index_df.empty else None

                tech = analyze_technicals(df, idx_slice)
                if not tech:
                    continue

                fin_data = fin_cache.get(ticker, {})
                q_fin, a_fin = fin_data.get("q_fin"), fin_data.get("a_fin")
                fin = q_fin if (q_fin is not None and not q_fin.empty) else a_fin

                pt_fin = {}
                if fin is not None and not fin.empty:
                    valid_cols = [c for c in fin.columns if pd.Timestamp(c) <= pd.Timestamp(dt)]
                    if valid_cols:
                        pt_fin = fin[valid_cols[0]].dropna().to_dict()

                fund = analyze_fundamentals(fin_data.get("info", {}), pt_fin, [], dt, None)
                if fund["rejected"]:
                    continue

                setup, scores, _ = calculate_regime_and_scores(tech, fund)
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
    print_backtest_stats(pd.DataFrame(records))


def print_backtest_stats(df: pd.DataFrame):
    if df is None or df.empty:
        cprint("No backtest signals were generated.", "yellow")
        return

    print("\n" + "=" * 100)
    cprint("  BACKTEST AGGREGATE STATISTICS", "magenta")
    print("=" * 100)
    print(f"  Total signals: {len(df)}")

    for setup in df["setup"].dropna().unique():
        sub = df[df["setup"] == setup]
        cprint(f"\n  SETUP: {setup} (N={len(sub)})", "cyan")
        print(
            f"    Win Rate (20d): {(sub['r20'] > 0).mean() * 100:.1f}% | "
            f"Median Return (20d): {sub['r20'].median():.2f}%"
        )
        print(
            f"    Avg MFE: {sub['mfe'].mean():.2f}% | Avg MAE: {sub['mae'].mean():.2f}%"
        )

    print("\n  SCORE BUCKET PERFORMANCE (20d Returns):")
    df = df.copy()
    df["bucket"] = pd.cut(
        df["composite"],
        bins=[0, 40, 60, 80, 100, np.inf],
        labels=["Low", "Med", "High", "Ultra", "Extreme"],
    )
    print(
        df.groupby("bucket", observed=True)["r20"]
        .agg(["count", "mean", "median"])
        .to_string()
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Advanced Midcap Scanner & Backtester (stateless)")
    p.add_argument("--backtest", action="store_true", help="Run historical backtester")
    p.add_argument("--backtest-days", type=int, default=5, help="Backtest every N business days")
    p.add_argument("--min-score", type=float, default=50.0, help="Minimum composite score to show")
    p.add_argument("--lenient", action="store_true", help="Include Falling Knife / Deterioration setups")
    p.add_argument("--workers", type=int, default=4, help="Concurrent workers")
    p.add_argument("--top", type=int, default=15, help="Show top N results (0 = all)")
    p.add_argument("--tickers-file", metavar="PATH", help="Custom ticker list")
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