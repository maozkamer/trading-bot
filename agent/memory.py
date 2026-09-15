"""
SQLite long-term memory for the trading agent.
DB lives at /data/agent_memory.db (Fly.io persistent volume).
Falls back to ./agent_memory.db in local dev.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
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

            CREATE TABLE IF NOT EXISTS conversation_history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id    TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_chat_id ON conversation_history(chat_id);

            CREATE TABLE IF NOT EXISTS recommendations (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol       TEXT    NOT NULL,
                setup        TEXT,
                direction    TEXT    NOT NULL DEFAULT 'long',
                confidence   REAL,
                det_score    REAL,
                entry        REAL    NOT NULL,
                target       REAL    NOT NULL,
                stop         REAL    NOT NULL,
                rr           REAL,
                reasoning    TEXT,
                risks        TEXT,
                snapshot     TEXT,
                status       TEXT    NOT NULL DEFAULT 'open',
                opened_at    TEXT    NOT NULL,
                deadline_at  TEXT,
                closed_at    TEXT,
                close_price  REAL,
                close_reason TEXT,
                pnl_pct      REAL,
                mfe_pct      REAL,
                mae_pct      REAL
            );
            CREATE INDEX IF NOT EXISTS idx_rec_status ON recommendations(status);
            CREATE INDEX IF NOT EXISTS idx_rec_symbol ON recommendations(symbol);
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


# ─────────────────────────────────────────────────────────────
#  Conversation history
# ─────────────────────────────────────────────────────────────

def save_message(chat_id: int | str, role: str, content: str) -> None:
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO conversation_history (chat_id, role, content) VALUES (?,?,?)",
                (str(chat_id), role, content),
            )
    except Exception as exc:
        log.error("save_message failed: %s", exc)


def get_history(chat_id: int | str, limit: int = 10) -> list[dict]:
    """Return the last *limit* messages for a chat in chronological order."""
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM conversation_history "
                "WHERE chat_id=? ORDER BY id DESC LIMIT ?",
                (str(chat_id), limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    except Exception as exc:
        log.error("get_history failed: %s", exc)
        return []


def clear_history(chat_id: int | str) -> None:
    try:
        with _get_conn() as conn:
            conn.execute(
                "DELETE FROM conversation_history WHERE chat_id=?",
                (str(chat_id),),
            )
        log.info("🗑️ Conversation history cleared for chat %s", chat_id)
    except Exception as exc:
        log.error("clear_history failed: %s", exc)


# ─────────────────────────────────────────────────────────────
#  "הממליץ" recommendations
# ─────────────────────────────────────────────────────────────

def save_recommendation(rec: dict) -> int:
    """Persist an accepted recommendation. Returns the new row id."""
    now = datetime.now(timezone.utc)
    deadline = now + timedelta(days=int(rec.get("hold_days", 14)))
    with _get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO recommendations
                 (symbol, setup, direction, confidence, det_score, entry, target, stop, rr,
                  reasoning, risks, snapshot, status, opened_at, deadline_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?, 'open', ?, ?)""",
            (
                rec["symbol"], rec.get("setup", ""), rec.get("direction", "long"),
                rec.get("confidence"), rec.get("det_score"),
                rec["entry"], rec["target"], rec["stop"], rec.get("rr"),
                rec.get("reasoning", ""), rec.get("risks", ""), rec.get("snapshot", ""),
                now.isoformat(), deadline.isoformat(),
            ),
        )
        return int(cur.lastrowid)


def get_open_recommendations() -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM recommendations WHERE status='open' ORDER BY opened_at"
        ).fetchall()
    return [dict(r) for r in rows]


def update_recommendation_excursions(rec_id: int, mfe_pct: float, mae_pct: float) -> None:
    try:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE recommendations SET mfe_pct=?, mae_pct=? WHERE id=?",
                (mfe_pct, mae_pct, rec_id),
            )
    except Exception as exc:
        log.error("update_recommendation_excursions failed: %s", exc)


def close_recommendation(
    rec_id: int,
    status: str,
    close_price: float,
    close_reason: str,
    pnl_pct: float,
    mfe_pct: float,
    mae_pct: float,
) -> None:
    try:
        with _get_conn() as conn:
            conn.execute(
                """UPDATE recommendations
                     SET status=?, close_price=?, close_reason=?, pnl_pct=?,
                         mfe_pct=?, mae_pct=?, closed_at=?
                   WHERE id=?""",
                (
                    status, close_price, close_reason, pnl_pct, mfe_pct, mae_pct,
                    datetime.now(timezone.utc).isoformat(), rec_id,
                ),
            )
        log.info("הממליץ: closed #%s as %s (%.2f%%)", rec_id, status, pnl_pct)
    except Exception as exc:
        log.error("close_recommendation failed: %s", exc)


def was_symbol_recommended_recently(symbol: str, hours: int = 72) -> bool:
    """True if *symbol* has an open recommendation or was opened within *hours*."""
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM recommendations "
                "WHERE symbol=? AND (status='open' OR opened_at > datetime('now', ?)) LIMIT 1",
                (symbol, f"-{hours} hours"),
            ).fetchone()
        return row is not None
    except Exception as exc:
        log.error("was_symbol_recommended_recently failed: %s", exc)
        return False


def get_recommendation_stats(since_days: int | None = None) -> dict:
    with _get_conn() as conn:
        q = "SELECT * FROM recommendations WHERE status != 'open'"
        params: list = []
        if since_days:
            q += " AND closed_at > datetime('now', ?)"
            params.append(f"-{int(since_days)} days")
        rows = [dict(r) for r in conn.execute(q, params).fetchall()]
        open_count = conn.execute(
            "SELECT COUNT(*) FROM recommendations WHERE status='open'"
        ).fetchone()[0]

    closed = len(rows)
    wins = sum(1 for r in rows if (r.get("pnl_pct") or 0) > 0)
    losses = closed - wins
    avg_pnl = round(sum((r.get("pnl_pct") or 0) for r in rows) / closed, 2) if closed else None
    win_rate = round(wins / closed * 100, 1) if closed else None
    best = max(rows, key=lambda r: r.get("pnl_pct") or -1e9, default=None)
    worst = min(rows, key=lambda r: r.get("pnl_pct") or 1e9, default=None)
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    return {
        "closed": closed,
        "open": open_count,
        "wins": wins,
        "losses": losses,
        "avg_pnl": avg_pnl,
        "win_rate": win_rate,
        "by_status": by_status,
        "best": {"symbol": best["symbol"], "pnl_pct": best["pnl_pct"]}
        if best and best.get("pnl_pct") is not None
        else None,
        "worst": {"symbol": worst["symbol"], "pnl_pct": worst["pnl_pct"]}
        if worst and worst.get("pnl_pct") is not None
        else None,
    }


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
