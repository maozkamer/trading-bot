"""
Universe builder and pattern scanner for the trading agent.
Combines a static watchlist with dynamic additions from DB and news.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger(__name__)

STATIC_UNIVERSE = [
    "MARA", "RIOT", "COIN", "CIFR", "IREN", "CLSK",
    "NNE", "CEG", "VST", "SMR",
    "NVDA", "AMD", "TSLA", "PLTR", "RKLB",
]

_BOT_DB_PATH = os.environ.get("DB_PATH", "/data/trading.db")
_NEWSAPI_KEY = os.environ.get("NEWS_API_KEY") or os.environ.get("NEWSAPI_KEY")


def _get_watchlist_symbols() -> list[str]:
    """Pull symbols that are in the bot's watchlist table."""
    try:
        db = Path(_BOT_DB_PATH)
        if not db.exists():
            return []
        with sqlite3.connect(str(db)) as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM watchlist"
            ).fetchall()
        return [r[0].upper() for r in rows if r[0]]
    except Exception as exc:
        log.debug("_get_watchlist_symbols failed: %s", exc)
        return []


def _get_news_mentioned_symbols(limit: int = 10) -> list[str]:
    """
    Ask NewsAPI for recent finance headlines and extract ticker-like tokens
    (2-5 uppercase letters preceded by $ or common context words).
    Falls back to empty list silently.
    """
    if not _NEWSAPI_KEY:
        return []
    try:
        from newsapi import NewsApiClient
        import re

        client = NewsApiClient(api_key=_NEWSAPI_KEY)
        resp = client.get_top_headlines(category="business", language="en", page_size=20)
        articles = resp.get("articles") or []
        text = " ".join(
            (a.get("title") or "") + " " + (a.get("description") or "")
            for a in articles
        )
        # Match $TICKER or standalone known-ish tickers
        found = re.findall(r"\$([A-Z]{2,5})\b", text)
        # Deduplicate, cap at limit
        seen: dict[str, int] = {}
        for sym in found:
            seen[sym] = seen.get(sym, 0) + 1
        sorted_syms = sorted(seen, key=lambda s: -seen[s])
        return sorted_syms[:limit]
    except Exception as exc:
        log.debug("_get_news_mentioned_symbols failed: %s", exc)
        return []


def build_dynamic_universe() -> list[str]:
    """
    Return deduplicated symbol list: static + DB watchlist + news mentions.
    Order preserved (static first).
    """
    combined = list(STATIC_UNIVERSE)
    for sym in _get_watchlist_symbols():
        if sym not in combined:
            combined.append(sym)
    for sym in _get_news_mentioned_symbols():
        if sym not in combined:
            combined.append(sym)
    log.info("🌐 Universe: %d symbols", len(combined))
    return combined


# ─────────────────────────────────────────────────────────────
#  Pattern scanner
# ─────────────────────────────────────────────────────────────

def _scan_one(symbol: str) -> list[dict]:
    """Run detect_patterns for a single symbol. Returns list of signal dicts."""
    try:
        # Import here to avoid circular imports at module load
        from agent.tools import detect_patterns  # type: ignore

        result = detect_patterns(symbol)
        signals = []

        if isinstance(result, dict):
            patterns = result.get("patterns") or []
            for p in patterns:
                confidence = float(p.get("confidence", 0))
                if confidence >= 0.75:
                    signals.append({
                        "symbol": symbol,
                        "pattern": p.get("name") or p.get("pattern") or "unknown",
                        "confidence": confidence,
                        "price": result.get("price") or p.get("price"),
                        "details": p,
                    })

        return signals
    except Exception as exc:
        log.debug("_scan_one(%s) failed: %s", symbol, exc)
        return []


def scan_for_patterns(universe: list[str] | None = None, max_workers: int = 6) -> list[dict]:
    """
    Scan the full universe for high-confidence patterns (≥0.75).
    Returns a flat list of signal dicts, sorted by confidence descending.
    """
    if universe is None:
        universe = build_dynamic_universe()

    all_signals: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_scan_one, sym): sym for sym in universe}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                signals = fut.result(timeout=30)
                all_signals.extend(signals)
            except Exception as exc:
                log.debug("scan future(%s) error: %s", sym, exc)

    all_signals.sort(key=lambda s: -s.get("confidence", 0))
    log.info("🔍 Pattern scan done: %d signals across %d symbols", len(all_signals), len(universe))
    return all_signals
