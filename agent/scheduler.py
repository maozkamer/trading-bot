"""
APScheduler jobs for the trading agent.
All times are Jerusalem (UTC+2/+3).
"""

from __future__ import annotations

import logging
import os

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from agent.core import run_agent_async
from agent.memory import save_pattern_alert, was_alert_sent_recently
from agent.scanner import scan_for_patterns
from agent.transparency import error_step

log = logging.getLogger(__name__)

_OWNER_CHAT_ID = int(os.environ.get("OWNER_CHAT_ID", "0"))
_TZ = "Asia/Jerusalem"


# ─────────────────────────────────────────────────────────────
#  Job definitions
# ─────────────────────────────────────────────────────────────

async def morning_briefing_job() -> None:
    """08:00 daily — market overview + watchlist scan."""
    log.info("⏰ Running morning briefing")
    try:
        prompt = (
            "תכין לי תדריך בוקר מלא: "
            "1) סקירת שוק כללית (SPY, QQQ, VIX, DXY). "
            "2) חדשות חמות מ-8 השעות האחרונות. "
            "3) סריקת Universe — כל מה שראוי לתשומת לב. "
            "שלח את הסיכום לטלגרם בפורמט התראת בוקר."
        )
        await run_agent_async(prompt, _OWNER_CHAT_ID)
    except Exception as exc:
        log.error("morning_briefing_job failed: %s", exc)
        error_step(_OWNER_CHAT_ID, f"שגיאה בתדריך בוקר: {exc}")


async def market_open_scan_job() -> None:
    """15:30 — 5 min before US market open, quick setup scan."""
    log.info("⏰ Running pre-open scan")
    try:
        prompt = (
            "שוק ארה\"ב נפתח בעוד 5 דקות. "
            "סרוק את ה-Universe לפטרנים חמים עם confidence >= 0.8. "
            "אם יש סיגנלים — שלח התראה לטלגרם לכל אחד."
        )
        await run_agent_async(prompt, _OWNER_CHAT_ID)
    except Exception as exc:
        log.error("market_open_scan_job failed: %s", exc)
        error_step(_OWNER_CHAT_ID, f"שגיאה בסריקת פתיחה: {exc}")


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

            # Ask the agent to compose and send the alert
            prompt = (
                f"זוהה פטרן {pattern} ב-{symbol} עם confidence {confidence:.0%}. "
                f"מחיר נוכחי: {'$'+str(price) if price else 'לא ידוע'}. "
                f"בצע ניתוח טכני מהיר ואם הפטרן מאושר — שלח התראת פטרן לטלגרם."
            )
            await run_agent_async(prompt, _OWNER_CHAT_ID)
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
