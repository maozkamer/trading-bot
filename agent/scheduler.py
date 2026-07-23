"""
APScheduler jobs for the trading agent.
All times are Jerusalem (UTC+2/+3).
"""

from __future__ import annotations

import logging
import os

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from agent.core import run_agent_single_shot_async
from agent.memory import save_pattern_alert, was_alert_sent_recently
from agent.scanner import scan_for_patterns
from agent.tools import fetch_news, get_market_overview, scan_universe, send_telegram
from agent.transparency import error_step

log = logging.getLogger(__name__)

_OWNER_CHAT_ID = int(os.environ.get("OWNER_CHAT_ID", "0"))
_TZ = "Asia/Jerusalem"
_ENRICH_PATTERN_ALERTS = os.environ.get("ENRICH_PATTERN_ALERTS") == "1"


# ─────────────────────────────────────────────────────────────
#  Job definitions
# ─────────────────────────────────────────────────────────────

async def morning_briefing_job() -> None:
    """08:00 daily — market overview + watchlist scan."""
    log.info("⏰ Running morning briefing")
    try:
        context = {
            "market_overview": get_market_overview(),
            "news": fetch_news(since_hours=8),
            "universe_scan": scan_universe(),
        }
        instruction = (
            "כתוב תדריך בוקר קצר וברור בעברית למשקיע, על סמך הנתונים בהודעה בלבד "
            "(סקירת שוק, חדשות, סריקת יוניברס). כלול מצב שוק כללי, חדשות בולטות, "
            "ומניות שראויות לתשומת לב. פורמט התראת בוקר קריאה לטלגרם. אל תמציא נתונים "
            "שלא סופקו."
        )
        summary = await run_agent_single_shot_async(context, instruction, max_tokens=500)
        send_telegram(summary, str(_OWNER_CHAT_ID))
    except Exception as exc:
        log.error("morning_briefing_job failed: %s", exc)
        error_step(_OWNER_CHAT_ID, f"שגיאה בתדריך בוקר: {exc}")


async def market_open_scan_job() -> None:
    """15:30 — 5 min before US market open, quick setup scan."""
    log.info("⏰ Running pre-open scan")
    try:
        universe = scan_universe()
        candidates = [
            s for s in universe
            if s.get("price") is not None
            and (
                (s.get("rsi") is not None and (s["rsi"] <= 30 or s["rsi"] >= 70))
                or (s.get("vol_ratio") is not None and s["vol_ratio"] >= 2.0)
            )
        ]
        if not candidates:
            log.info("No extreme RSI / high volume candidates — skipping Claude call")
            return

        context = {"candidates": candidates}
        instruction = (
            "שוק ארה\"ב נפתח בעוד 5 דקות. אלה מניות עם RSI קיצוני או נפח מסחר חריג, "
            "על סמך הנתונים בהודעה בלבד. כתוב התראת פתיחת שוק קצרה בעברית לכל מניה "
            "רלוונטית. אל תמציא נתונים שלא סופקו."
        )
        summary = await run_agent_single_shot_async(context, instruction, max_tokens=500)
        send_telegram(summary, str(_OWNER_CHAT_ID))
    except Exception as exc:
        log.error("market_open_scan_job failed: %s", exc)
        error_step(_OWNER_CHAT_ID, f"שגיאה בסריקת פתיחה: {exc}")


def _format_pattern_alert(symbol: str, pattern: str, confidence: float, price: float | None) -> str:
    price_str = f"${price}" if price else "לא ידוע"
    return (
        f"🔔 *זוהה פטרן*\n"
        f"מניה: {symbol}\n"
        f"פטרן: {pattern}\n"
        f"ביטחון: {confidence:.0%}\n"
        f"מחיר נוכחי: {price_str}"
    )


async def pattern_scan_job() -> None:
    """Every 30 min during market hours (16:30–23:00 Jerusalem)."""
    log.info("⏰ Running intraday pattern scan")
    try:
        signals = scan_for_patterns()
        if not signals:
            log.info("No high-confidence signals found")
            return

        for sig in signals:
            symbol = sig["symbol"]
            pattern = sig["pattern"]
            confidence = sig.get("confidence", 0)
            price = sig.get("price")

            if was_alert_sent_recently(symbol, pattern, hours=6):
                log.debug("Skipping duplicate alert: %s %s", symbol, pattern)
                continue

            if _ENRICH_PATTERN_ALERTS:
                context = {
                    "symbol": symbol,
                    "pattern": pattern,
                    "confidence": confidence,
                    "price": price,
                    "details": sig.get("details"),
                }
                instruction = (
                    "נסח התראת פטרן קצרה וטבעית בעברית על סמך הנתונים בהודעה בלבד. "
                    "אל תמציא מידע נוסף."
                )
                message = await run_agent_single_shot_async(context, instruction, max_tokens=250)
            else:
                message = _format_pattern_alert(symbol, pattern, confidence, price)

            send_telegram(message, str(_OWNER_CHAT_ID))
            save_pattern_alert(symbol, pattern, confidence, price or 0.0)

    except Exception as exc:
        log.error("pattern_scan_job failed: %s", exc)
        error_step(_OWNER_CHAT_ID, f"שגיאה בסריקת פטרנים: {exc}")


# ─────────────────────────────────────────────────────────────
#  Scheduler setup
# ─────────────────────────────────────────────────────────────

def setup_scheduler() -> AsyncIOScheduler:
    """
    Create and start the AsyncIOScheduler with all agent jobs.
    Returns the scheduler so the caller can shut it down on exit.
    """
    scheduler = AsyncIOScheduler(timezone=_TZ)

    # 08:00 every weekday
    scheduler.add_job(
        morning_briefing_job,
        CronTrigger(day_of_week="mon-fri", hour=8, minute=0, timezone=_TZ),
        id="morning_briefing",
        replace_existing=True,
        misfire_grace_time=300,
    )

    # 15:30 every weekday (5 min before US open)
    scheduler.add_job(
        market_open_scan_job,
        CronTrigger(day_of_week="mon-fri", hour=15, minute=30, timezone=_TZ),
        id="market_open_scan",
        replace_existing=True,
        misfire_grace_time=120,
    )

    # Every 30 min, 16:30–23:00, weekdays
    scheduler.add_job(
        pattern_scan_job,
        CronTrigger(
            day_of_week="mon-fri",
            hour="16-22",
            minute="0,30",
            timezone=_TZ,
        ),
        id="pattern_scan",
        replace_existing=True,
        misfire_grace_time=120,
    )

    scheduler.start()
    log.info("✅ Agent scheduler started (timezone=%s)", _TZ)
    return scheduler
