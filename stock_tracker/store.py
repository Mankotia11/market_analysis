#!/usr/bin/env python3
"""SQLite watchlist."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "watchlist.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlist (
            ticker TEXT PRIMARY KEY,
            added_at TEXT NOT NULL,
            notes TEXT DEFAULT ''
        )
        """
    )
    return conn


def list_tickers() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT ticker, added_at, notes FROM watchlist ORDER BY added_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def add_ticker(ticker: str, notes: str = "") -> dict:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO watchlist (ticker, added_at, notes) VALUES (?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET notes = excluded.notes
            """,
            (ticker, now, notes),
        )
    return {"ticker": ticker, "added_at": now, "notes": notes}


def remove_ticker(ticker: str) -> bool:
    with _conn() as conn:
        cur = conn.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker,))
        return cur.rowcount > 0
