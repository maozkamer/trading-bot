"""
Morning news digest (RSS via feedparser) + Earnings alert (FMP API).
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta

import feedparser
import requests

from analysis import WATCHLIST, _fetch_daily, fetch_fear_greed, get_top_morning_pick

log = logging.getLogger(__name__)

FMP_KEY      = os.environ.get("FMP_KEY")
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.1-8b-instant"

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


def _groq_summarize(headlines: list[str], watchlist_symbols: list[str]) -> str:
    """Summarise headlines in Hebrew using Groq. Falls back to raw headlines."""
    if not GROQ_API_KEY or not headlines:
        return "\n".join(f"  - {h}" for h in headlines[:5])

    headlines_text = "\n".join(f"- {h}" for h in headlines[:15])
    symbols_text   = ", ".join(watchlist_symbols)
    prompt = (
        f"אתה עוזר מסחר. קיבלת את הכותרות האלה:\n{headlines_text}\n\n"
        f"המניות שהטריידר עוקב: {symbols_text}\n"
        f"תן סיכום של 3-4 נקודות בעברית קצרות וחדות — מה חשוב לטריידר הבוקר, "
        f"האם יש השפעה על המניות ברשימה, ומה כדאי לעקוב."
    )
    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400,
                "temperature": 0.3,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        log.warning("Groq summarize failed: %s", exc)
        return "\n".join(f"  - {h}" for h in headlines[:5])


def build_morning_message() -> str:
    from datetime import date as _date
    today_str = _date.today().strftime("%d/%m/%Y")

    headlines, _ = fetch_news()

    fg   = fetch_fear_greed()
    pick = get_top_morning_pick()

    lines = [f"🌅 *בוקר טוב! סיכום שוק — {today_str}*\n"]

    # Fear & Greed
    if fg["value"] is not None:
        lines.append(f"😱 *Fear & Greed:* {fg['value']}/100 — {fg['classification']}\n")

    # Top 3 headlines in Hebrew
    lines.append("📰 *חדשות מרכזיות:*")
    if headlines:
        lines.append(_groq_summarize(headlines, WATCHLIST))
    else:
        lines.append("  לא הצלחתי לשלוף חדשות כרגע.")

    # Top morning pick
    if pick:
        lines.append(f"\n⭐ *מנייה לעקוב היום:* {pick['symbol']}")
        lines.append(f"  {pick['reason']}")

    lines.append("\nבהצלחה במסחר! 📈")
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
