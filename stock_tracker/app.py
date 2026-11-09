#!/usr/bin/env python3
"""Stock tracker dashboard: add tickers, score buy zones, monitor setups."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import store
from market import analyze_ticker, invalidate_cache, resolve_listed_ticker

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

app = FastAPI(title="Stock Tracker", version="1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class AddTicker(BaseModel):
    ticker: str
    notes: str = ""


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/watchlist")
def get_watchlist():
    return {"tickers": store.list_tickers()}


@app.post("/api/watchlist")
def post_watchlist(body: AddTicker):
    raw = (body.ticker or "").strip()
    if not raw:
        raise HTTPException(400, "Ticker is required")
    try:
        ticker = resolve_listed_ticker(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    row = store.add_ticker(ticker, body.notes)
    snapshot = analyze_ticker(ticker)
    return {"item": row, "snapshot": snapshot}


@app.delete("/api/watchlist/{ticker}")
def delete_watchlist(ticker: str):
    ok = store.remove_ticker(ticker.upper())
    if not ok:
        store.remove_ticker(ticker)
    return {"ok": True}


@app.get("/api/snapshot")
def snapshot():
    rows = store.list_tickers()
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(analyze_ticker, r["ticker"]): r for r in rows}
        for fut in as_completed(futs):
            data = fut.result()
            if data.get("error"):
                errors.append(data)
            else:
                results.append(data)
    results.sort(key=lambda x: (-(x.get("buy_score") or 0), x.get("symbol") or ""))
    zones = {}
    for r in results:
        zones[r["zone"]] = zones.get(r["zone"], 0) + 1
    return {
        "count": len(results),
        "zones": zones,
        "stocks": results,
        "errors": errors,
    }


@app.get("/api/stock/{ticker}")
def stock_detail(ticker: str):
    data = analyze_ticker(ticker)
    if data.get("error"):
        raise HTTPException(404, data["error"])
    return data


@app.post("/api/refresh")
def refresh():
    invalidate_cache()
    return snapshot()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=5000 , reload=True)
