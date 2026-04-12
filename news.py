"""
Morning news digest (RSS via feedparser) + Earnings alert (FMP API).
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta

import feedparser
import requests

from analysis import WATCHLIST

log = logging.getLogger(__name__)

FMP_KEY      = os.environ.get("FMP_KEY")
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# ─────────────────────────────────────────────────────────────
#  RSS feeds (no API key needed)
# ─────────────────────────────────────────────────────────────

RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/technologyNews",
]

MAX_ITEMS_PER_FEED = 5


def fetch_news() -> tuple[list[str], list[str]]:
    """
    Returns (headlines, mentioned_symbols).
    headlines      — list of title strings
    mentioned_symbols — symbols from WATCHLIST found in headlines
    """
    headlines: list[str] = []
    watchlist_upper = [s.upper() for s in WATCHLIST]

    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:MAX_ITEMS_PER_FEED]:
                title = (entry.get("title") or "").strip()
                if title and title not in headlines:
                    headlines.append(title)
        except Exception as exc:
            log.warning("RSS fetch error %s: %s", url, exc)

    # find which watchlist symbols are mentioned in any headline
    text = " ".join(headlines).upper()
    mentioned = [s for s in watchlist_upper if s in text]

    return headlines[:15], mentioned


def build_morning_message() -> str:
    headlines, mentioned = fetch_news()

    if not headlines:
        return "🌅 בוקר טוב! לא הצלחתי לשלוף חדשות כרגע."

    lines = ["🌅 *בוקר טוב! סיכום חדשות שוק:*\n"]
    for h in headlines[:10]:
        lines.append(f"📰 {h}")

    if mentioned:
        lines.append(f"\n📊 *מניות שמוזכרות היום:* {', '.join(mentioned)}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  Earnings calendar (FMP)
# ─────────────────────────────────────────────────────────────

EARNINGS_DAYS_AHEAD = 5


def fetch_upcoming_earnings() -> list[dict]:
    """
    Returns list of dicts: {symbol, date, days_left}
    for WATCHLIST symbols reporting in the next EARNINGS_DAYS_AHEAD days.
    """
    if not FMP_KEY:
        log.warning("FMP_KEY not set — skipping earnings check")
        return []

    today = date.today()
    to_date = today + timedelta(days=EARNINGS_DAYS_AHEAD)

    try:
        resp = requests.get(
            f"{FMP_BASE_URL}/earning_calendar",
            params={
                "from":   today.isoformat(),
                "to":     to_date.isoformat(),
                "apikey": FMP_KEY,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.error("Earnings fetch error: %s", exc)
        return []

    watchlist_upper = {s.upper() for s in WATCHLIST}
    results: list[dict] = []

    for item in data:
        sym = (item.get("symbol") or "").upper()
        if sym not in watchlist_upper:
            continue
        date_str = item.get("date") or item.get("reportDate") or ""
        if not date_str:
            continue
        try:
            report_date = date.fromisoformat(date_str[:10])
        except ValueError:
            continue
        days_left = (report_date - today).days
        if 0 <= days_left <= EARNINGS_DAYS_AHEAD:
            results.append({
                "symbol":    sym,
                "date":      report_date.strftime("%d/%m/%Y"),
                "days_left": days_left,
            })

    return results


def build_earnings_messages() -> list[str]:
    """Returns one message string per upcoming earnings event."""
    upcoming = fetch_upcoming_earnings()
    messages: list[str] = []

    for e in upcoming:
        days = e["days_left"]
        days_str = "היום" if days == 0 else f"בעוד {days} יום" if days == 1 else f"בעוד {days} ימים"
        messages.append(
            f"📅 *תזכורת Earnings!*\n"
            f"📊 *{e['symbol']}* מדווחת תוצאות {days_str} ({e['date']})\n"
            f"⚡ שקול לנהל סיכון לפני הדוח"
        )

    return messages
