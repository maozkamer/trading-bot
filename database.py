import sqlite3
from datetime import datetime, timedelta, timezone

DB_PATH = "trading.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sent_alerts (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol    TEXT NOT NULL,
                alert_key TEXT NOT NULL,
                sent_at   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_alerts
                ON sent_alerts (symbol, alert_key, sent_at);

            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)


def is_alert_recent(symbol: str, alert_key: str, hours: int) -> bool:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM sent_alerts "
            "WHERE symbol=? AND alert_key=? AND sent_at > ? LIMIT 1",
            (symbol, alert_key, cutoff),
        ).fetchone()
    return row is not None


def save_alert(symbol: str, alert_key: str):
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sent_alerts (symbol, alert_key, sent_at) VALUES (?,?,?)",
            (symbol, alert_key, now),
        )
        # prune rows older than 7 days
        conn.execute("DELETE FROM sent_alerts WHERE sent_at < datetime('now','-7 days')")


def get_setting(key: str) -> str | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT value FROM settings WHERE key=?", (key,)
        ).fetchone()
    return row["value"] if row else None


def save_setting(key: str, value: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?,?)",
            (key, value),
        )
