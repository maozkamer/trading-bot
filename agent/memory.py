"""
SQLite long-term memory for the trading agent.
DB lives at /data/agent_memory.db (Fly.io persistent volume).
Falls back to ./agent_memory.db in local dev.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

DB_PATH = os.environ.get("AGENT_DB_PATH", "/data/agent_memory.db")


def _get_conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_memory_db() -> None:
    """Create tables if they don't exist."""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS insights (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                category         TEXT    NOT NULL,
                content          TEXT    NOT NULL,
                created_at       TEXT    NOT NULL,
                referenced_count INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_category ON insights(category);

            CREATE TABLE IF NOT EXISTS pattern_alerts (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol         TEXT    NOT NULL,
                pattern        TEXT    NOT NULL,
                confidence     REAL,
                price_at_alert REAL,
                sent_at        TEXT    NOT NULL,
                outcome        TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_symbol ON pattern_alerts(symbol);
        """)
    log.info("✅ Agent memory DB initialised at %s", DB_PATH)


# ─────────────────────────────────────────────────────────────
#  Insights
# ─────────────────────────────────────────────────────────────

def save_insight(category: str, content: str) -> bool:
    """Persist a long-term insight. Returns True on success."""
    try:
        now = datetime.now(timezone.utc).isoformat()
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO insights (category, content, created_at) VALUES (?,?,?)",
                (category, content, now),
            )
        log.info("💾 Insight saved [%s]: %.80s", category, content)
        return True
    except Exception as exc:
        log.error("save_insight failed: %s", exc)
        return False


def recall_memory(query: str, limit: int = 5) -> list[dict]:
    """
    Return the most relevant insights for *query*.
    Simple keyword search across content + category.
    Increments referenced_count for returned rows.
    """
    try:
        tokens = [t.strip().lower() for t in query.split() if len(t.strip()) > 1]
        if not tokens:
            return []

        with _get_conn() as conn:
            # Fetch all (small table) and rank client-side
            rows = conn.execute(
                "SELECT id, category, content, created_at, referenced_count "
                "FROM insights ORDER BY created_at DESC LIMIT 200"
            ).fetchall()

            scored: list[tuple[int, dict]] = []
            for row in rows:
                text = (row["category"] + " " + row["content"]).lower()
                score = sum(1 for t in tokens if t in text)
                if score > 0:
                    scored.append((score, dict(row)))

            scored.sort(key=lambda x: -x[0])
            top = [item for _, item in scored[:limit]]

            # Bump reference counts
            ids = [r["id"] for r in top]
            if ids:
                conn.execute(
                    f"UPDATE insights SET referenced_count = referenced_count + 1 "
                    f"WHERE id IN ({','.join('?' * len(ids))})",
                    ids,
                )

        return top
    except Exception as exc:
        log.error("recall_memory failed: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────
#  Pattern alerts log
# ─────────────────────────────────────────────────────────────

def save_pattern_alert(
    symbol: str,
    pattern: str,
    confidence: float,
    price: float,
) -> bool:
    """Log an alert that was sent to the user."""
    try:
        now = datetime.now(timezone.utc).isoformat()
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO pattern_alerts "
                "(symbol, pattern, confidence, price_at_alert, sent_at) "
                "VALUES (?,?,?,?,?)",
                (symbol, pattern, confidence, price, now),
            )
        return True
    except Exception as exc:
        log.error("save_pattern_alert failed: %s", exc)
        return False


def was_alert_sent_recently(symbol: str, pattern: str, hours: int = 6) -> bool:
    """True if the same symbol+pattern alert was sent within *hours*."""
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM pattern_alerts "
                "WHERE symbol=? AND pattern=? "
                "AND sent_at > datetime('now', ? || ' hours') LIMIT 1",
                (symbol, pattern, f"-{hours}"),
            ).fetchone()
        return row is not None
    except Exception as exc:
        log.error("was_alert_sent_recently failed: %s", exc)
        return False
