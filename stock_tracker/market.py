#!/usr/bin/env python3
"""Yahoo Finance fetch + short TTL cache."""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

import pandas as pd
import yfinance as yf

_YF_LOCK = threading.Lock()

from analysis import (
    BENCHMARK,
    analyze_frame,
    clean_ohlcv,
    empty_error,
    nse_candidate,
    resolve_ticker,
)

CACHE_TTL = 15 * 60
_price_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_info_cache: dict[str, tuple[float, dict]] = {}
_bench_cache: tuple[float, Optional[pd.DataFrame]] | None = None


def _fresh(entry, ttl=CACHE_TTL):
    if not entry:
        return None
    ts, payload = entry
    if time.time() - ts > ttl:
        return None
    return payload


def download_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    cached = _fresh(_price_cache.get(ticker))
    if cached is not None:
        return cached.copy()
    with _YF_LOCK:
        cached = _fresh(_price_cache.get(ticker))
        if cached is not None:
            return cached.copy()
        df = yf.download(
            ticker, period=period, progress=False, auto_adjust=True, threads=False
        )
        df = clean_ohlcv(df)
        _price_cache[ticker] = (time.time(), df)
        return df.copy()


def fetch_info(ticker: str) -> tuple[dict, Any]:
    cached = _fresh(_info_cache.get(ticker), ttl=6 * 3600)
    if cached is not None:
        return cached.get("info") or {}, cached.get("calendar")
    with _YF_LOCK:
        cached = _fresh(_info_cache.get(ticker), ttl=6 * 3600)
        if cached is not None:
            return cached.get("info") or {}, cached.get("calendar")
        t = yf.Ticker(ticker)
        info = {}
        calendar = None
        try:
            info = t.info or {}
        except Exception:
            info = {}
        try:
            calendar = t.calendar
        except Exception:
            calendar = None
        _info_cache[ticker] = (time.time(), {"info": info, "calendar": calendar})
        return info, calendar


def get_benchmark() -> Optional[pd.DataFrame]:
    global _bench_cache
    cached = _fresh(_bench_cache) if _bench_cache else None
    if cached is not None:
        return cached
    df = download_history(BENCHMARK, period="2y")
    _bench_cache = (time.time(), df)
    return df


def resolve_listed_ticker(raw: str) -> str:
    ticker = resolve_ticker(raw)
    # Bare symbols in this project are usually NSE. Try .NS first, then the
    # raw Yahoo symbol (US / other listings). Explicit suffixes win as-is.
    candidates = [ticker] if "." in ticker else [nse_candidate(ticker), ticker]
    seen = []
    for cand in candidates:
        if cand in seen:
            continue
        seen.append(cand)
        df = download_history(cand)
        if len(df) >= 80:
            return cand
    if seen:
        raise ValueError(f"No price history for {raw}")
    raise ValueError(f"No price history for {raw}")


def analyze_ticker(raw: str) -> dict[str, Any]:
    try:
        ticker = resolve_listed_ticker(raw)
    except Exception as exc:
        return empty_error(raw.upper(), str(exc))
    try:
        df = download_history(ticker)
        if df.empty:
            return empty_error(ticker, "No price data")
        info, calendar = fetch_info(ticker)
        bench = get_benchmark()
        return analyze_frame(ticker, df, bench=bench, info=info, calendar=calendar)
    except Exception as exc:
        return empty_error(ticker, f"{type(exc).__name__}: {exc}")


def invalidate_cache(ticker: Optional[str] = None) -> None:
    global _bench_cache
    if ticker:
        _price_cache.pop(ticker, None)
        _info_cache.pop(ticker, None)
    else:
        _price_cache.clear()
        _info_cache.clear()
        _bench_cache = None
