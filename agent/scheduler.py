"""
APScheduler jobs for the trading agent.
All times are Jerusalem (UTC+2/+3).
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from agent.core import run_agent_single_shot_async
from agent.memory import save_pattern_alert, was_alert_sent_recently
from agent.scanner import scan_for_patterns
from agent.tools import send_telegram
from agent.transparency import error_step

log = logging.getLogger(__name__)

_OWNER_CHAT_ID = int(os.environ.get("OWNER_CHAT_ID", "0"))
_TZ = "Asia/Jerusalem"
_ENRICH_PATTERN_ALERTS = os.environ.get("ENRICH_PATTERN_ALERTS") == "1"


# ─────────────────────────────────────────────────────────────
#  Job definitions
# ─────────────────────────────────────────────────────────────

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
        # Run off the event loop — scan_for_patterns can block for tens of seconds
        # on a slow/rate-limited data fetch, which was stalling Discord heartbeats.
        signals = await asyncio.to_thread(scan_for_patterns)
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


async def recommender_job() -> None:
    """Tue–Sat 07:00 Israel — "הממליץ" post-close scan. Silent if nothing passes."""
    log.info("⏰ Running הממליץ scan")
    try:
        from agent.recommender import run_recommender

        await asyncio.to_thread(run_recommender)
    except Exception as exc:
        log.error("recommender_job failed: %s", exc, exc_info=True)
        error_step(_OWNER_CHAT_ID, f"שגיאה בהממליץ: {exc}")


async def recommender_tracking_job() -> None:
    """Mon–Sat 08:00 Israel — evaluate open recommendations, post verdicts."""
    log.info("⏰ Running הממליץ tracking")
    try:
        from agent.recommender_tracking import run_tracking

        weekly = datetime.now(pytz.timezone(_TZ)).weekday() == 5  # Saturday
        await asyncio.to_thread(run_tracking, weekly)
    except Exception as exc:
        log.error("recommender_tracking_job failed: %s", exc, exc_info=True)
        error_step(_OWNER_CHAT_ID, f"שגיאה במעקב הממליץ: {exc}")


# ─────────────────────────────────────────────────────────────
#  Scheduler setup
# ─────────────────────────────────────────────────────────────

def setup_scheduler() -> AsyncIOScheduler:
    """
    Create and start the AsyncIOScheduler with all agent jobs.
    Returns the scheduler so the caller can shut it down on exit.
    """
    scheduler = AsyncIOScheduler(timezone=_TZ)

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

    # "הממליץ" — post-close scan, Tue–Sat 07:00 (covers each Mon–Fri US close)
    scheduler.add_job(
        recommender_job,
        CronTrigger(day_of_week="tue-sat", hour=7, minute=0, timezone=_TZ),
        id="recommender",
        replace_existing=True,
        misfire_grace_time=1800,
    )

    # "הממליץ" tracking — Mon–Sat 08:00, weekly summary folded in on Saturdays
    scheduler.add_job(
        recommender_tracking_job,
        CronTrigger(day_of_week="mon-sat", hour=8, minute=0, timezone=_TZ),
        id="recommender_tracking",
        replace_existing=True,
        misfire_grace_time=1800,
    )

    scheduler.start()
    log.info("✅ Agent scheduler started (timezone=%s)", _TZ)
    return scheduler
