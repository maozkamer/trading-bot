"""
Swing-trading Telegram bot — main entry point.
Scans WATCHLIST every hour during US market hours, every 4 hours at night.
All messages in Hebrew.
"""

import asyncio
import logging
import os
from datetime import datetime

import pytz
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from news import build_morning_message, build_earnings_messages, build_screener_message
from analysis import (
    WATCHLIST,
    Alert,
    analyze_symbol,
    check_sr_proximity,
    get_full_analysis,
    get_levels,
    get_quick_status,
    get_bollinger_levels,
    get_fibonacci_levels,
    get_vwap,
    _fetch_daily,
    _find_sr,
)
from charts import build_chart
from database import (
    get_setting,
    init_db,
    is_alert_recent,
    save_alert,
    save_setting,
)

# ─────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────
TOKEN  = os.environ.get("TELEGRAM_TOKEN")
EST_TZ = pytz.timezone("US/Eastern")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

OWNER_CHAT_ID: int | None = None
alerts_paused: bool = False


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def format_alert(alert: Alert) -> str:
    sup = f"${alert.support:.2f}"    if alert.support    else "—"
    res = f"${alert.resistance:.2f}" if alert.resistance else "—"
    return (
        f"🚨 *{alert.symbol}* — {alert.title}\n"
        f"💵 מחיר נוכחי: ${alert.price:.2f}\n"
        f"📊 מה קרה: {alert.description}\n"
        f"🎯 רמות חשובות: תמיכה {sup} / התנגדות {res}\n"
        f"⚡ המלצה: {alert.recommendation}"
    )


def format_sr_alert(alert: Alert) -> str:
    """Format for S/R proximity alerts (different layout)."""
    level_type = "תמיכה" if "sup" in alert.key else "התנגדות"
    level_price = alert.support if "sup" in alert.key else alert.resistance
    dist_pct = abs(alert.price - level_price) / alert.price * 100 if level_price else 0
    return (
        f"📊 *{alert.symbol}* מתקרב ל{level_type} ב-${level_price:.2f}\n"
        f"💵 מחיר נוכחי: ${alert.price:.2f}\n"
        f"📉 מרחק: {dist_pct:.1f}% מה{level_type}\n"
        f"⚡ שים לב — {alert.recommendation}"
    )


def is_market_hours() -> bool:
    now = datetime.now(EST_TZ)
    if now.weekday() >= 5:
        return False
    open_t  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_t <= now <= close_t


# ─────────────────────────────────────────────────────────────
#  Scanner
# ─────────────────────────────────────────────────────────────

async def run_scan(app: Application) -> None:
    global OWNER_CHAT_ID
    if OWNER_CHAT_ID is None:
        log.info("No owner chat_id — skipping scan")
        return
    if alerts_paused:
        log.info("Alerts paused — skipping scan")
        return

    log.info("Scanning %d symbols…", len(WATCHLIST))
    sent = 0

    for symbol in WATCHLIST:
        try:
            # ── Technical alerts ─────────────────────────────
            for alert in analyze_symbol(symbol):
                if is_alert_recent(symbol, alert.key, alert.cooldown_hours):
                    continue
                save_alert(symbol, alert.key)
                await app.bot.send_message(
                    chat_id=OWNER_CHAT_ID,
                    text=format_alert(alert),
                    parse_mode="Markdown",
                )
                sent += 1
                await asyncio.sleep(0.4)

            # ── S/R proximity alerts ──────────────────────────
            if is_market_hours():
                for alert in check_sr_proximity(symbol):
                    if is_alert_recent(symbol, alert.key, alert.cooldown_hours):
                        continue
                    save_alert(symbol, alert.key)
                    await app.bot.send_message(
                        chat_id=OWNER_CHAT_ID,
                        text=format_sr_alert(alert),
                        parse_mode="Markdown",
                    )
                    sent += 1
                    await asyncio.sleep(0.4)

            await asyncio.sleep(1.5)
        except Exception as exc:
            log.error("Scan error %s: %s", symbol, exc)

    log.info("Scan done — %d alerts sent", sent)


async def scan_loop(app: Application) -> None:
    await asyncio.sleep(20)          # startup grace period
    while True:
        try:
            await run_scan(app)
        except Exception as exc:
            log.error("scan_loop error: %s", exc)

        in_market  = is_market_hours()
        sleep_secs = 3600 if in_market else 14400
        log.info("Next scan in %.0f min  (market_hours=%s)", sleep_secs / 60, in_market)
        await asyncio.sleep(sleep_secs)


def _seconds_until_time(hour: int, minute: int) -> float:
    """Seconds until the next occurrence of HH:MM in Israel time."""
    import pytz as _pytz
    tz  = _pytz.timezone("Asia/Jerusalem")
    now = datetime.now(tz)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target = target.replace(day=target.day + 1)
    return (target - now).total_seconds()


async def morning_news_loop(app: Application) -> None:
    """Sends morning news digest every day at 08:00 Israel time."""
    while True:
        await asyncio.sleep(_seconds_until_time(8, 0))
        if OWNER_CHAT_ID is None:
            await asyncio.sleep(60)
            continue
        try:
            msg = await asyncio.get_event_loop().run_in_executor(
                None, build_morning_message
            )
            await app.bot.send_message(
                chat_id=OWNER_CHAT_ID, text=msg, parse_mode="Markdown"
            )
            log.info("Morning news sent")
        except Exception as exc:
            log.error("morning_news_loop error: %s", exc)
        await asyncio.sleep(61)   # avoid double-fire in the same minute


async def screener_loop(app: Application) -> None:
    """Sends morning screener every day at 09:00 Israel time."""
    while True:
        await asyncio.sleep(_seconds_until_time(9, 0))
        if OWNER_CHAT_ID is None:
            await asyncio.sleep(60)
            continue
        try:
            msg = await asyncio.get_event_loop().run_in_executor(
                None, build_screener_message
            )
            await app.bot.send_message(
                chat_id=OWNER_CHAT_ID, text=msg, parse_mode="Markdown"
            )
            log.info("Morning screener sent")
        except Exception as exc:
            log.error("screener_loop error: %s", exc)
        await asyncio.sleep(61)   # avoid double-fire in the same minute


async def earnings_loop(app: Application) -> None:
    """Sends earnings reminders every day at 08:30 Israel time."""
    while True:
        await asyncio.sleep(_seconds_until_time(8, 30))
        if OWNER_CHAT_ID is None:
            await asyncio.sleep(60)
            continue
        try:
            messages = await asyncio.get_event_loop().run_in_executor(
                None, build_earnings_messages
            )
            for msg in messages:
                symbol = msg.split("*")[1] if "*" in msg else "earnings"
                key    = f"earnings_{symbol}_{datetime.now().strftime('%Y%m%d')}"
                if is_alert_recent(symbol, key, 24):
                    continue
                save_alert(symbol, key)
                await app.bot.send_message(
                    chat_id=OWNER_CHAT_ID, text=msg, parse_mode="Markdown"
                )
                await asyncio.sleep(0.4)
            log.info("Earnings check done — %d alerts", len(messages))
        except Exception as exc:
            log.error("earnings_loop error: %s", exc)
        await asyncio.sleep(61)


# ─────────────────────────────────────────────────────────────
#  Command handlers
# ─────────────────────────────────────────────────────────────

async def cmd_start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    global OWNER_CHAT_ID
    OWNER_CHAT_ID = update.effective_chat.id
    save_setting("chat_id", str(OWNER_CHAT_ID))

    symbols_str = " | ".join(WATCHLIST)
    text = (
        "📈 *בוט מסחר סווינג — פעיל!*\n\n"
        f"🔍 מנטר *{len(WATCHLIST)} מניות:*\n"
        f"`{symbols_str}`\n\n"
        "⏰ *תדירות סריקה:*\n"
        "  • שעות מסחר (09:30‑16:00 EST): כל שעה\n"
        "  • לילה / סוף שבוע: כל 4 שעות\n\n"
        "📋 *פקודות:*\n"
        "  /status — מחירים נוכחיים + RSI\n"
        "  /analysis AAPL — ניתוח מלא\n"
        "  /levels AAPL — תמיכות והתנגדויות\n"
        "  /chart AAPL — גרף נרות 30 ימים\n"
        "  /stop — השהה התראות\n"
        "  /resume — חדש התראות\n\n"
        "🚨 *סוגי התראות:*\n"
        "  📊 תמיכה / התנגדות / פריצה\n"
        "  📈 Bull/Bear Flag, Double Top/Bottom,\n"
        "      Head & Shoulders, Triangle, Breakout, Cup & Handle\n"
        "  🔢 RSI, MACD, MA50/200, Golden/Death Cross, נפח\n"
        "  🕯️ Doji, Hammer, Engulfing, Morning/Evening Star\n"
        "  💰 ירידה/עלייה חדה, Gap Up/Down"
    )
    await update.message.reply_text(text, parse_mode="Markdown", reply_markup=MAIN_KEYBOARD)


async def cmd_status(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("⏳ מושך נתונים… (20‑40 שניות)")
    data  = get_quick_status()
    lines = ["📊 *סטטוס מניות נוכחי*\n"]

    for d in data:
        if d["price"] is None:
            lines.append(f"  ❌ {d['symbol']:6s} — שגיאה")
            continue
        chg   = d["change"]
        rsi   = d["rsi"]
        arrow = "🟢" if chg >= 0 else "🔴"
        chg_s = f"{'+'if chg>=0 else ''}{chg:.1f}%"
        rsi_s = f"RSI {rsi:.0f}" if rsi is not None else "     "
        flag  = (" 🔥" if rsi < 30 else " ⚠️" if rsi > 70 else "") if rsi else ""
        lines.append(f"  {arrow} {d['symbol']:6s}  ${d['price']:8.2f}  {chg_s:>7}   {rsi_s}{flag}")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    symbol = " ".join(context.args).upper().strip()
    if not symbol:
        await update.message.reply_text("שלח: /analysis NNE")
        return
    await update.message.reply_text(f"⏳ מנתח את {symbol}…")
    await update.message.reply_text(get_full_analysis(symbol), parse_mode="Markdown")


async def cmd_levels(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    symbol = " ".join(context.args).upper().strip()
    if not symbol:
        await update.message.reply_text("שלח: /levels NNE")
        return
    await update.message.reply_text(f"⏳ מחשב רמות עבור {symbol}…")
    await update.message.reply_text(get_levels(symbol), parse_mode="Markdown")


async def cmd_chart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    symbol = " ".join(context.args).upper().strip()
    if not symbol:
        await update.message.reply_text("שלח: /chart NNE")
        return
    msg = await update.message.reply_text(f"⏳ מייצר גרף עבור {symbol}…")
    try:
        df = _fetch_daily(symbol)
        if df.empty or len(df) < 5:
            await msg.edit_text(f"❌ אין מספיק נתונים עבור {symbol}")
            return
        supports, resistances = _find_sr(df)
        buf = await asyncio.get_event_loop().run_in_executor(
            None, build_chart, symbol, df, supports, resistances
        )
        caption = (
            f"📈 *{symbol}* — 30 ימים אחרונים\n"
            f"🛡️ תמיכות: {', '.join(f'${s:.2f}' for s in supports[:3]) or '—'}\n"
            f"🔺 התנגדויות: {', '.join(f'${r:.2f}' for r in resistances[:3]) or '—'}"
        )
        await update.message.reply_photo(photo=buf, caption=caption, parse_mode="Markdown")
        await msg.delete()
    except Exception as exc:
        log.error("Chart error %s: %s", symbol, exc)
        await msg.edit_text(f"❌ שגיאה ביצירת גרף עבור {symbol}: {exc}")


async def cmd_bb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    symbol = " ".join(context.args).upper().strip()
    if not symbol:
        await update.message.reply_text("שלח: /bb NNE")
        return
    await update.message.reply_text(f"⏳ מחשב Bollinger Bands עבור {symbol}…")
    await update.message.reply_text(get_bollinger_levels(symbol), parse_mode="Markdown")


async def cmd_fib(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    symbol = " ".join(context.args).upper().strip()
    if not symbol:
        await update.message.reply_text("שלח: /fib NNE")
        return
    await update.message.reply_text(f"⏳ מחשב רמות פיבונאצ'י עבור {symbol}…")
    await update.message.reply_text(get_fibonacci_levels(symbol), parse_mode="Markdown")


async def cmd_vwap(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    symbol = " ".join(context.args).upper().strip()
    if not symbol:
        await update.message.reply_text("שלח: /vwap NNE")
        return
    await update.message.reply_text(f"⏳ מחשב VWAP עבור {symbol}…")
    await update.message.reply_text(get_vwap(symbol), parse_mode="Markdown")


async def cmd_stop(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    global alerts_paused
    alerts_paused = True
    await update.message.reply_text("⏸️ התראות הושהו. שלח /resume להמשך.")


async def cmd_resume(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    global alerts_paused
    alerts_paused = False
    await update.message.reply_text("▶️ התראות חודשו!")


# ─────────────────────────────────────────────────────────────
#  Persistent Reply Keyboard
# ─────────────────────────────────────────────────────────────

# Buttons that trigger "ask for symbol" flow
SYMBOL_ACTIONS = {"📊 גרף", "🔍 ניתוח", "📉 רמות", "📐 Fib", "⚡ VWAP", "📊 BB"}

# Key stored in context.user_data while waiting for a symbol
PENDING_KEY = "pending_action"

MAIN_KEYBOARD = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton("📈 סטטוס מניות"), KeyboardButton("❓ עזרה")],
        [KeyboardButton("📊 גרף"),         KeyboardButton("🔍 ניתוח")],
        [KeyboardButton("📉 רמות"),        KeyboardButton("🗂 תפריט")],
        [KeyboardButton("📐 Fib"),         KeyboardButton("⚡ VWAP"),   KeyboardButton("📊 BB")],
    ],
    resize_keyboard=True,
)


async def cmd_menu(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("תפריט:", reply_markup=MAIN_KEYBOARD)


async def _send_status(msg) -> None:
    """Shared status logic used by both the command and the menu button."""
    await msg.reply_text("⏳ מושך נתונים… (20‑40 שניות)")
    data_list = get_quick_status()
    lines = ["📊 *סטטוס מניות נוכחי*\n"]
    for d in data_list:
        if d["price"] is None:
            lines.append(f"  ❌ {d['symbol']:6s} — שגיאה")
            continue
        chg   = d["change"]
        rsi   = d["rsi"]
        arrow = "🟢" if chg >= 0 else "🔴"
        chg_s = f"{'+'if chg>=0 else ''}{chg:.1f}%"
        rsi_s = f"RSI {rsi:.0f}" if rsi is not None else "     "
        flag  = (" 🔥" if rsi < 30 else " ⚠️" if rsi > 70 else "") if rsi else ""
        lines.append(f"  {arrow} {d['symbol']:6s}  ${d['price']:8.2f}  {chg_s:>7}   {rsi_s}{flag}")
    await msg.reply_text("\n".join(lines), parse_mode="Markdown")


async def _run_symbol_action(action: str, symbol: str, msg) -> None:
    """Execute chart / analysis / levels for a given symbol."""
    if action == "📊 גרף":
        loading = await msg.reply_text(f"⏳ מייצר גרף עבור {symbol}…")
        try:
            df = _fetch_daily(symbol)
            if df.empty or len(df) < 5:
                await loading.edit_text(f"❌ אין מספיק נתונים עבור {symbol}")
                return
            supports, resistances = _find_sr(df)
            buf = await asyncio.get_event_loop().run_in_executor(
                None, build_chart, symbol, df, supports, resistances
            )
            caption = (
                f"📈 *{symbol}* — 30 ימים אחרונים\n"
                f"🛡️ תמיכות: {', '.join(f'${s:.2f}' for s in supports[:3]) or '—'}\n"
                f"🔺 התנגדויות: {', '.join(f'${r:.2f}' for r in resistances[:3]) or '—'}"
            )
            await msg.reply_photo(photo=buf, caption=caption, parse_mode="Markdown")
            await loading.delete()
        except Exception as exc:
            log.error("Chart error %s: %s", symbol, exc)
            await loading.edit_text(f"❌ שגיאה: {exc}")

    elif action == "🔍 ניתוח":
        await msg.reply_text(f"⏳ מנתח את {symbol}…")
        await msg.reply_text(get_full_analysis(symbol), parse_mode="Markdown")

    elif action == "📉 רמות":
        await msg.reply_text(f"⏳ מחשב רמות עבור {symbol}…")
        await msg.reply_text(get_levels(symbol), parse_mode="Markdown")

    elif action == "📐 Fib":
        await msg.reply_text(f"⏳ מחשב רמות פיבונאצ'י עבור {symbol}…")
        await msg.reply_text(get_fibonacci_levels(symbol), parse_mode="Markdown")

    elif action == "⚡ VWAP":
        await msg.reply_text(f"⏳ מחשב VWAP עבור {symbol}…")
        await msg.reply_text(get_vwap(symbol), parse_mode="Markdown")

    elif action == "📊 BB":
        await msg.reply_text(f"⏳ מחשב Bollinger Bands עבור {symbol}…")
        await msg.reply_text(get_bollinger_levels(symbol), parse_mode="Markdown")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles all non-command text: menu button presses + symbol replies."""
    text = (update.message.text or "").strip()

    # ── User replied with a symbol after we asked ─────────────
    pending = context.user_data.get(PENDING_KEY)
    if pending and text not in SYMBOL_ACTIONS and text != "🗂 תפריט":
        context.user_data.pop(PENDING_KEY)
        symbol = text.upper()
        await _run_symbol_action(pending, symbol, update.message)
        return

    # ── Menu buttons ──────────────────────────────────────────
    if text == "📈 סטטוס מניות":
        await _send_status(update.message)

    elif text in SYMBOL_ACTIONS:
        context.user_data[PENDING_KEY] = text
        await update.message.reply_text("איזו מנייה? (למשל: NNE, MARA, PLTR)")

    elif text == "🗂 תפריט":
        await update.message.reply_text("תפריט:", reply_markup=MAIN_KEYBOARD)

    elif text == "❓ עזרה":
        await update.message.reply_text(
            "📋 *פקודות זמינות:*\n"
            "  /status — מחירים נוכחיים + RSI\n"
            "  /analysis NNE — ניתוח מלא\n"
            "  /levels NNE — תמיכות והתנגדויות\n"
            "  /chart NNE — גרף נרות 30 ימים\n"
            "  /stop — השהה התראות\n"
            "  /resume — חדש התראות",
            parse_mode="Markdown",
        )


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    global OWNER_CHAT_ID

    if not TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN לא מוגדר! הוסף אותו כ-environment variable")

    init_db()

    # Restore owner chat_id from previous session
    saved = get_setting("chat_id")
    if saved:
        OWNER_CHAT_ID = int(saved)
        log.info("Restored owner chat_id: %d", OWNER_CHAT_ID)

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("menu",     cmd_menu))
    app.add_handler(CommandHandler("status",   cmd_status))
    app.add_handler(CommandHandler("analysis", cmd_analysis))
    app.add_handler(CommandHandler("levels",   cmd_levels))
    app.add_handler(CommandHandler("chart",    cmd_chart))
    app.add_handler(CommandHandler("bb",       cmd_bb))
    app.add_handler(CommandHandler("fib",      cmd_fib))
    app.add_handler(CommandHandler("vwap",     cmd_vwap))
    app.add_handler(CommandHandler("stop",     cmd_stop))
    app.add_handler(CommandHandler("resume",   cmd_resume))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    async def post_init(application: Application) -> None:
        asyncio.create_task(scan_loop(application))
        asyncio.create_task(morning_news_loop(application))
        asyncio.create_task(earnings_loop(application))
        asyncio.create_task(screener_loop(application))

    app.post_init = post_init

    log.info("✅ Trading bot started — watching %d symbols", len(WATCHLIST))
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
