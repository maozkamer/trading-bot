"""
Morning news digest (RSS via feedparser) + Earnings alert (FMP API).
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta

import feedparser
import requests
from deep_translator import GoogleTranslator


def _translate(text: str) -> str:
    try:
        return GoogleTranslator(source="auto", target="iw").translate(text)
    except Exception:
        return text

from analysis import WATCHLIST, _fetch_daily, _rsi, _find_sr

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
        lines.append(f"📰 {_translate(h)}")

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


def build_screener_message() -> str:
    """
    Morning screener: iterates WATCHLIST, computes key indicators,
    and returns a formatted Hebrew message split into 4 sections.
    """
    today_str = date.today().strftime("%d/%m/%Y")

    breakout_candidates = []   # price within 3% below nearest resistance, volume ratio > 1.3
    abnormal_volume     = []   # volume ratio >= 2.0
    rsi_oversold        = []   # RSI < 35
    rsi_overbought      = []   # RSI > 70

    for symbol in WATCHLIST:
        try:
            df = _fetch_daily(symbol)
            if df.empty or len(df) < 21:
                continue
            df = df.dropna(subset=["High", "Low", "Close", "Volume"])

            closes  = df["Close"]
            volumes = df["Volume"]
            price   = float(closes.iloc[-1])

            # Volume ratio
            avg_20_vol = float(volumes.iloc[-21:-1].mean())
            curr_vol   = float(volumes.iloc[-1])
            vol_ratio  = (curr_vol / avg_20_vol) if avg_20_vol > 0 else 0.0

            # RSI
            rsi = _rsi(closes) if len(closes) >= 15 else None

            # Support / Resistance
            _, resistances = _find_sr(df)
            nearest_res = resistances[0] if resistances else None

            # Section 1 — breakout candidates
            if nearest_res is not None:
                dist_pct = (nearest_res - price) / nearest_res * 100
                if 0 < dist_pct <= 3.0 and vol_ratio > 1.3:
                    breakout_candidates.append({
                        "symbol":   symbol,
                        "price":    price,
                        "res":      nearest_res,
                        "dist_pct": dist_pct,
                        "vol_ratio": vol_ratio,
                    })

            # Section 2 — abnormal volume
            if vol_ratio >= 2.0:
                abnormal_volume.append({
                    "symbol":    symbol,
                    "price":     price,
                    "vol_ratio": vol_ratio,
                    "curr_vol":  curr_vol,
                    "avg_vol":   avg_20_vol,
                })

            # Sections 3 & 4 — RSI zones
            if rsi is not None:
                if rsi < 35:
                    rsi_oversold.append({"symbol": symbol, "price": price, "rsi": rsi})
                elif rsi > 70:
                    rsi_overbought.append({"symbol": symbol, "price": price, "rsi": rsi})

        except Exception as exc:
            log.warning("screener error %s: %s", symbol, exc)

    lines = [f"🔍 *סקרינר בוקר — {today_str}*\n"]

    lines.append("🚀 *קרובים לפריצה* (מחיר עד 3% מתחת להתנגדות + נפח x1.3+)")
    if breakout_candidates:
        for e in sorted(breakout_candidates, key=lambda x: x["dist_pct"]):
            lines.append(
                f"  • *{e['symbol']}*  ${e['price']:.2f}  |  "
                f"התנגדות: ${e['res']:.2f} ({e['dist_pct']:.1f}% מעל)  |  "
                f"נפח: x{e['vol_ratio']:.1f}\n"
                f"    📌 סיבה: קרוב לפריצת התנגדות עם עלייה בנפח"
            )
    else:
        lines.append("  אין מניות מתאימות כרגע")

    lines.append("\n📊 *נפח חריג* (נפח x2.0+ מהממוצע)")
    if abnormal_volume:
        for e in sorted(abnormal_volume, key=lambda x: -x["vol_ratio"]):
            lines.append(
                f"  • *{e['symbol']}*  ${e['price']:.2f}  |  "
                f"נפח: {e['curr_vol']/1e6:.1f}M (x{e['vol_ratio']:.1f} ממוצע)\n"
                f"    📌 סיבה: נפח חריג — ייתכן כסף גדול או קטליזטור"
            )
    else:
        lines.append("  אין מניות עם נפח חריג")

    lines.append("\n💚 *RSI באזור קנייה* (RSI < 35)")
    if rsi_oversold:
        for e in sorted(rsi_oversold, key=lambda x: x["rsi"]):
            lines.append(
                f"  • *{e['symbol']}*  ${e['price']:.2f}  |  RSI: {e['rsi']:.1f}\n"
                f"    📌 סיבה: oversold — הזדמנות כניסה פוטנציאלית לסווינג"
            )
    else:
        lines.append("  אין מניות ב-oversold")

    lines.append("\n❤️ *RSI אזור מכירה* (RSI > 70)")
    if rsi_overbought:
        for e in sorted(rsi_overbought, key=lambda x: -x["rsi"]):
            lines.append(
                f"  • *{e['symbol']}*  ${e['price']:.2f}  |  RSI: {e['rsi']:.1f}\n"
                f"    📌 סיבה: overbought — שקול יציאה חלקית או סטופ"
            )
    else:
        lines.append("  אין מניות ב-overbought")

    return "\n".join(lines)


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
