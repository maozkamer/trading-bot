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
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application,
    CallbackQueryHandler,
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
    get_rich_analysis,
    get_levels,
    get_quick_status,
    get_bollinger_levels,
    get_fibonacci_levels,
    get_vwap,
    build_fear_greed_message,
    build_setups_message,
    get_active_setups,
    get_ichimoku,
    get_stoch_rsi,
    get_pivot_points,
    get_obv,
    get_atr,
    get_stoploss,
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
PORT   = int(os.environ.get("PORT", 8080))

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

OWNER_CHAT_ID: int | None = None
alerts_paused: bool = False

# Mutable watchlist (copy of WATCHLIST so we can edit it at runtime)
_dynamic_watchlist: list[str] = list(WATCHLIST)


# ─────────────────────────────────────────────────────────────
#  FastAPI — Mini App backend
# ─────────────────────────────────────────────────────────────

api = FastAPI()
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SymbolRequest(BaseModel):
    symbol: str


@api.get("/api/watchlist")
async def api_watchlist():
    """Returns watchlist with current price + RSI for each symbol."""
    try:
        data = await asyncio.get_event_loop().run_in_executor(None, get_quick_status)
        # Filter to only symbols in our dynamic watchlist
        filtered = [d for d in data if d["symbol"] in _dynamic_watchlist]
        return {"symbols": filtered, "watchlist": _dynamic_watchlist}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/api/screener")
async def api_screener():
    """Returns screener results: breakouts, volume spikes, RSI zones."""
    try:
        results = {"breakouts": [], "volume": [], "oversold": [], "overbought": []}
        data = await asyncio.get_event_loop().run_in_executor(None, get_quick_status)

        for d in data:
            if d["symbol"] not in _dynamic_watchlist:
                continue
            if d["price"] is None:
                continue
            rsi = d.get("rsi")
            vol_ratio = d.get("vol_ratio", 1.0) or 1.0
            change = d.get("change", 0) or 0

            entry = {
                "symbol": d["symbol"],
                "price": d["price"],
                "change": change,
                "rsi": rsi,
                "vol_ratio": vol_ratio,
            }

            if rsi and rsi < 35:
                results["oversold"].append(entry)
            elif rsi and rsi > 70:
                results["overbought"].append(entry)

            if vol_ratio >= 2.0:
                results["volume"].append(entry)

            if change >= 3.0 and vol_ratio >= 1.3:
                results["breakouts"].append(entry)

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/api/analysis/{symbol}")
async def api_analysis(symbol: str):
    """Returns full technical analysis for a symbol."""
    symbol = symbol.upper()
    try:
        log.info("API analysis request for %s", symbol)
        df = await asyncio.get_event_loop().run_in_executor(None, _fetch_daily, symbol)
        log.info("DataFrame for %s: is_none=%s, len=%d", symbol, df is None, len(df) if df is not None else 0)

        if df is None or df.empty or len(df) < 5:
            log.warning("Insufficient data for %s — returning 404", symbol)
            raise HTTPException(status_code=404, detail=f"אין נתונים עבור {symbol}")

        supports, resistances = _find_sr(df)
        closes = df["Close"].tolist()
        latest = float(df["Close"].iloc[-1])
        prev   = float(df["Close"].iloc[-2]) if len(df) > 1 else latest
        change = (latest - prev) / prev * 100 if prev else 0

        # RSI
        from analysis import _rsi, _macd, _bollinger, _sma
        rsi_val = _rsi(closes)
        log.info("%s — price=%.2f change=%.2f%% rsi=%s", symbol, latest, change, rsi_val)

        # MACD
        macd_val, signal_val, hist_val, _ = _macd(closes)

        # Bollinger
        bb_upper, bb_mid, bb_lower = _bollinger(closes)

        # MA50 / MA200
        ma50_cur,  _ = _sma(closes, 50)
        ma200_cur, _ = _sma(closes, 200)

        # Recent candles for chart data (last 30)
        candles = []
        df_recent = df.tail(30)
        for idx, row in df_recent.iterrows():
            candles.append({
                "date":   idx.strftime("%d/%m"),
                "open":   round(float(row["Open"]),  2),
                "high":   round(float(row["High"]),  2),
                "low":    round(float(row["Low"]),   2),
                "close":  round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            })

        log.info("%s — returning %d candles, %d supports, %d resistances",
                 symbol, len(candles), len(supports), len(resistances))

        return {
            "symbol":      symbol,
            "price":       round(latest, 2),
            "change":      round(change, 2),
            "rsi":         round(rsi_val, 1) if rsi_val else None,
            "macd":        round(macd_val, 3) if macd_val else None,
            "macd_signal": round(signal_val, 3) if signal_val else None,
            "macd_hist":   round(hist_val, 3) if hist_val else None,
            "bb_upper":    round(bb_upper, 2) if bb_upper else None,
            "bb_mid":      round(bb_mid, 2) if bb_mid else None,
            "bb_lower":    round(bb_lower, 2) if bb_lower else None,
            "ma50":        round(ma50_cur, 2) if ma50_cur else None,
            "ma200":       round(ma200_cur, 2) if ma200_cur else None,
            "supports":    [round(s, 2) for s in supports[:3]],
            "resistances": [round(r, 2) for r in resistances[:3]],
            "candles":     candles,
        }
    except HTTPException:
        raise
    except Exception as e:
        log.error("API analysis error for %s: %s", symbol, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/watchlist/add")
async def api_watchlist_add(req: SymbolRequest):
    symbol = req.symbol.upper().strip()
    if symbol in _dynamic_watchlist:
        return {"ok": True, "watchlist": _dynamic_watchlist}
    _dynamic_watchlist.append(symbol)
    save_setting("dynamic_watchlist", ",".join(_dynamic_watchlist))
    return {"ok": True, "watchlist": _dynamic_watchlist}


@api.post("/api/watchlist/remove")
async def api_watchlist_remove(req: SymbolRequest):
    symbol = req.symbol.upper().strip()
    if symbol in _dynamic_watchlist:
        _dynamic_watchlist.remove(symbol)
        save_setting("dynamic_watchlist", ",".join(_dynamic_watchlist))
    return {"ok": True, "watchlist": _dynamic_watchlist}


@api.get("/api/fear_greed")
async def api_fear_greed():
    try:
        from analysis import fetch_fear_greed
        data = await asyncio.get_event_loop().run_in_executor(None, fetch_fear_greed)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve Mini App static files
api.mount("/app", StaticFiles(directory="miniapp", html=True), name="miniapp")


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

    log.info("Scanning %d symbols…", len(_dynamic_watchlist))
    sent = 0

    for symbol in _dynamic_watchlist:
        try:
            _MUTED_KEYS = (
                "rsi_oversold", "rsi_overbought",
                "macd_bullish", "macd_bearish",
                "bb_upper_break", "bb_lower_break",
                "golden_cross", "death_cross",
                "vwap_cross_up", "vwap_cross_down",
            )
            for alert in analyze_symbol(symbol):
                if any(alert.key.startswith(k) for k in _MUTED_KEYS):
                    continue
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
    await asyncio.sleep(20)
    while True:
        try:
            await run_scan(app)
        except Exception as exc:
            log.error("scan_loop error: %s", exc)
        in_market  = is_market_hours()
        sleep_secs = 3600 if in_market else 14400
        log.info("Next scan in %.0f min  (market_hours=%s)", sleep_secs / 60, in_market)
        await asyncio.sleep(sleep_secs)


def _seconds_until_utc(hour: int, minute: int) -> float:
    import pytz as _pytz
    tz  = _pytz.utc
    now = datetime.now(tz)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        from datetime import timedelta
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _seconds_until_time(hour: int, minute: int) -> float:
    import pytz as _pytz
    tz  = _pytz.timezone("Asia/Jerusalem")
    now = datetime.now(tz)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        from datetime import timedelta
        target += timedelta(days=1)
    return (target - now).total_seconds()


async def morning_news_loop(app: Application) -> None:
    while True:
        secs = _seconds_until_utc(6, 0)
        log.info("Morning news scheduled in %.0f min", secs / 60)
        await asyncio.sleep(secs)
        if OWNER_CHAT_ID is None:
            await asyncio.sleep(60)
            continue
        try:
            msg = await asyncio.get_event_loop().run_in_executor(None, build_morning_message)
            await app.bot.send_message(chat_id=OWNER_CHAT_ID, text=msg, parse_mode="Markdown")
        except Exception as exc:
            log.error("morning_news_loop error: %s", exc)
        await asyncio.sleep(61)


async def screener_loop(app: Application) -> None:
    while True:
        await asyncio.sleep(_seconds_until_time(9, 0))
        if OWNER_CHAT_ID is None:
            await asyncio.sleep(60)
            continue
        try:
            msg = await asyncio.get_event_loop().run_in_executor(None, build_screener_message)
            await app.bot.send_message(chat_id=OWNER_CHAT_ID, text=msg, parse_mode="Markdown")
        except Exception as exc:
            log.error("screener_loop error: %s", exc)
        await asyncio.sleep(61)


async def earnings_loop(app: Application) -> None:
    while True:
        await asyncio.sleep(_seconds_until_time(8, 30))
        if OWNER_CHAT_ID is None:
            await asyncio.sleep(60)
            continue
        try:
            messages = await asyncio.get_event_loop().run_in_executor(None, build_earnings_messages)
            for msg in messages:
                symbol = msg.split("*")[1] if "*" in msg else "earnings"
                key    = f"earnings_{symbol}_{datetime.now().strftime('%Y%m%d')}"
                if is_alert_recent(symbol, key, 24):
                    continue
                save_alert(symbol, key)
                await app.bot.send_message(chat_id=OWNER_CHAT_ID, text=msg, parse_mode="Markdown")
                await asyncio.sleep(0.4)
        except Exception as exc:
            log.error("earnings_loop error: %s", exc)
        await asyncio.sleep(61)


async def setup_scan_loop(app: Application) -> None:
    await asyncio.sleep(30)
    while True:
        try:
            if OWNER_CHAT_ID is not None and not alerts_paused and is_market_hours():
                sent = 0
                for symbol in _dynamic_watchlist:
                    try:
                        setups = await asyncio.get_event_loop().run_in_executor(
                            None, get_active_setups, symbol
                        )
                        for s in setups:
                            key = f"setup_{symbol}_{s['name'].split()[0].lower()}"
                            if is_alert_recent(symbol, key, 12):
                                continue
                            save_alert(symbol, key)
                            dist_pct = (s["entry"] - s["price"]) / s["price"] * 100
                            vol_note = f"גבוה x{s['vol_ratio']:.1f} — מחזק את הסט-אפ ⚡" if s["vol_ratio"] >= 1.5 else "תקין"
                            text = (
                                f"🚨 *סט-אפ מזוהה — {symbol}!*\n\n"
                                f"📊 פטרן: {s['name']}\n"
                                f"💵 מחיר נוכחי: ${s['price']:.2f}\n"
                                f"🎯 נקודת פריצה: ${s['entry']:.2f} ({dist_pct:+.1f}%)\n"
                                f"📈 יעד אחרי פריצה: ${s['target']:.2f}\n"
                                f"🛑 סטופ מומלץ: ${s['stop']:.2f}\n"
                                f"📊 נפח: {vol_note}\n\n"
                                f"📌 סיבה: {s['reason']}"
                            )
                            await app.bot.send_message(
                                chat_id=OWNER_CHAT_ID, text=text, parse_mode="Markdown"
                            )
                            sent += 1
                            await asyncio.sleep(0.4)
                        await asyncio.sleep(1.0)
                    except Exception as exc:
                        log.warning("setup_scan error %s: %s", symbol, exc)
        except Exception as exc:
            log.error("setup_scan_loop error: %s", exc)
        await asyncio.sleep(3600)


# ─────────────────────────────────────────────────────────────
#  Command handlers
# ─────────────────────────────────────────────────────────────

async def cmd_start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    global OWNER_CHAT_ID
    OWNER_CHAT_ID = update.effective_chat.id
    save_setting("chat_id", str(OWNER_CHAT_ID))

    mini_app_url = f"https://trading-bot-production-5207.up.railway.app/app"
    symbols_str = " | ".join(_dynamic_watchlist)
    text = (
        "📈 *בוט מסחר סווינג — פעיל!*\n\n"
        f"🔍 מנטר *{len(_dynamic_watchlist)} מניות:*\n"
        f"`{symbols_str}`\n\n"
        "⏰ *תדירות סריקה:*\n"
        "  • שעות מסחר (09:30‑16:00 EST): כל שעה\n"
        "  • לילה / סוף שבוע: כל 4 שעות\n\n"
        "📱 *Dashboard:* לחץ על הכפתור למטה\n\n"
        "📋 *פקודות:*\n"
        "  /status — מחירים נוכחיים + RSI\n"
        "  /analysis AAPL — ניתוח מלא\n"
        "  /levels AAPL — תמיכות והתנגדויות\n"
        "  /chart AAPL — גרף נרות 30 ימים\n"
        "  /stop — השהה התראות\n"
        "  /resume — חדש התראות"
    )

    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton("📱 פתח Dashboard", web_app={"url": mini_app_url})],
            [KeyboardButton("📈 סטטוס"),        KeyboardButton("🔍 סקרינר")],
            [KeyboardButton("📊 גרף"),           KeyboardButton("🎯 סט-אפים")],
            [KeyboardButton("📉 BB"),            KeyboardButton("⚡ VWAP")],
            [KeyboardButton("📐 פיבונאצ'י"),    KeyboardButton("🔎 ניתוח")],
            [KeyboardButton("☁️ Ichimoku"),      KeyboardButton("📉 Stoch RSI"), KeyboardButton("📐 Pivot")],
            [KeyboardButton("💹 OBV"),           KeyboardButton("🛑 Stop Loss")],
            [KeyboardButton("😱 Fear & Greed"), KeyboardButton("🗂 עוד פקודות")],
        ],
        resize_keyboard=True,
    )
    await update.message.reply_text(text, parse_mode="Markdown", reply_markup=keyboard)


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


async def cmd_ichimoku(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    symbol = " ".join(context.args).upper().strip()
    if not symbol:
        await update.message.reply_text("שלח: /ichimoku NNE")
        return
    await update.message.reply_text(f"⏳ מחשב Ichimoku Cloud עבור {symbol}…")
    result = await asyncio.get_event_loop().run_in_executor(None, get_ichimoku, symbol)
    await update.message.reply_text(result, parse_mode="Markdown")


async def cmd_stoch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    symbol = " ".join(context.args).upper().strip()
    if not symbol:
        await update.message.reply_text("שלח: /stoch NNE")
        return
    await update.message.reply_text(f"⏳ מחשב Stochastic RSI עבור {symbol}…")
    await update.message.reply_text(get_stoch_rsi(symbol), parse_mode="Markdown")


async def cmd_pivot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    symbol = " ".join(context.args).upper().strip()
    if not symbol:
        await update.message.reply_text("שלח: /pivot NNE")
        return
    await update.message.reply_text(f"⏳ מחשב Pivot Points עבור {symbol}…")
    await update.message.reply_text(get_pivot_points(symbol), parse_mode="Markdown")


async def cmd_obv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    symbol = " ".join(context.args).upper().strip()
    if not symbol:
        await update.message.reply_text("שלח: /obv NNE")
        return
    await update.message.reply_text(f"⏳ מחשב OBV עבור {symbol}…")
    await update.message.reply_text(get_obv(symbol), parse_mode="Markdown")


async def cmd_atr(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    symbol = " ".join(context.args).upper().strip()
    if not symbol:
        await update.message.reply_text("שלח: /atr NNE")
        return
    await update.message.reply_text(f"⏳ מחשב ATR עבור {symbol}…")
    await update.message.reply_text(get_atr(symbol), parse_mode="Markdown")


async def cmd_stoploss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if len(context.args) < 2:
        await update.message.reply_text("שלח: /stoploss NNE 45.50")
        return
    symbol = context.args[0].upper().strip()
    try:
        entry = float(context.args[1])
    except ValueError:
        await update.message.reply_text("❌ מחיר כניסה לא תקין.")
        return
    await update.message.reply_text(f"⏳ מחשב Stop Loss עבור {symbol}…")
    result = await asyncio.get_event_loop().run_in_executor(None, get_stoploss, symbol, entry)
    await update.message.reply_text(result, parse_mode="Markdown")


async def cmd_stop(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    global alerts_paused
    alerts_paused = True
    await update.message.reply_text("⏸️ התראות הושהו. שלח /resume להמשך.")


async def cmd_resume(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    global alerts_paused
    alerts_paused = False
    await update.message.reply_text("▶️ התראות חודשו!")


# ─────────────────────────────────────────────────────────────
#  Keyboards
# ─────────────────────────────────────────────────────────────

PENDING_KEY = "pending_action"

MAIN_KEYBOARD = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton("📈 סטטוס"),        KeyboardButton("🔍 סקרינר")],
        [KeyboardButton("📊 גרף"),           KeyboardButton("🎯 סט-אפים")],
        [KeyboardButton("📉 BB"),            KeyboardButton("⚡ VWAP")],
        [KeyboardButton("📐 פיבונאצ'י"),    KeyboardButton("🔎 ניתוח")],
        [KeyboardButton("☁️ Ichimoku"),      KeyboardButton("📉 Stoch RSI"), KeyboardButton("📐 Pivot")],
        [KeyboardButton("💹 OBV"),           KeyboardButton("🛑 Stop Loss")],
        [KeyboardButton("😱 Fear & Greed"), KeyboardButton("🗂 עוד פקודות")],
    ],
    resize_keyboard=True,
)

_SYMBOLS_ROWS = [
    ["NNE",  "MARA", "PLTR", "IREN"],
    ["SOFI", "AAPL", "NVDA", "TSLA"],
    ["CIFR", "HOOD", "MSFT", "OKLO"],
    ["SMR",  "RKLB", "COIN", "RIOT"],
    ["AMD",  "META", "GOOGL","AMZN"],
    ["BTC-USD", "ETH-USD"],
]

def build_symbol_picker(action_prefix: str) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(sym, callback_data=f"{action_prefix}:{sym}") for sym in row]
        for row in _SYMBOLS_ROWS
    ]
    return InlineKeyboardMarkup(rows)

def build_more_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📅 Earnings",  callback_data="more:earnings"),
            InlineKeyboardButton("🔔 רמות",      callback_data="more:levels"),
        ],
        [
            InlineKeyboardButton("📰 חדשות",     callback_data="more:news"),
            InlineKeyboardButton("❓ עזרה",       callback_data="more:help"),
        ],
    ])

SYMBOL_ACTIONS = {
    "📊 גרף", "📐 פיבונאצ'י", "📉 BB", "⚡ VWAP", "🎯 סט-אפים", "🔎 ניתוח",
    "☁️ Ichimoku", "📉 Stoch RSI", "📐 Pivot", "💹 OBV", "🛑 Stop Loss",
}

_ACTION_PREFIX = {
    "📊 גרף":        "chart",
    "📐 פיבונאצ'י":  "fib",
    "📉 BB":          "bb",
    "⚡ VWAP":        "vwap",
    "🎯 סט-אפים":    "setups",
    "🔎 ניתוח":      "richanalysis",
    "☁️ Ichimoku":   "ichimoku",
    "📉 Stoch RSI":  "stoch",
    "📐 Pivot":       "pivot",
    "💹 OBV":         "obv",
    "🛑 Stop Loss":  "stoploss",
}


async def cmd_menu(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("תפריט:", reply_markup=MAIN_KEYBOARD)


# ─────────────────────────────────────────────────────────────
#  Shared action executor
# ─────────────────────────────────────────────────────────────

async def _send_status(msg) -> None:
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


async def _run_symbol_action(action: str, symbol: str, msg,
                             ctx: ContextTypes.DEFAULT_TYPE | None = None) -> None:
    if action == "chart":
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
            await loading.edit_text(f"❌ שגיאה: {exc}")
    elif action == "fib":
        await msg.reply_text(f"⏳ מחשב רמות פיבונאצ'י עבור {symbol}…")
        await msg.reply_text(get_fibonacci_levels(symbol), parse_mode="Markdown")
    elif action == "bb":
        await msg.reply_text(f"⏳ מחשב Bollinger Bands עבור {symbol}…")
        await msg.reply_text(get_bollinger_levels(symbol), parse_mode="Markdown")
    elif action == "vwap":
        await msg.reply_text(f"⏳ מחשב VWAP עבור {symbol}…")
        await msg.reply_text(get_vwap(symbol), parse_mode="Markdown")
    elif action == "levels":
        await msg.reply_text(f"⏳ מחשב רמות עבור {symbol}…")
        await msg.reply_text(get_levels(symbol), parse_mode="Markdown")
    elif action == "analysis":
        await msg.reply_text(f"⏳ מנתח את {symbol}…")
        await msg.reply_text(get_full_analysis(symbol), parse_mode="Markdown")
    elif action == "setups":
        await msg.reply_text(f"⏳ מחפש סט-אפים עבור {symbol}…")
        result = await asyncio.get_event_loop().run_in_executor(None, build_setups_message, symbol)
        await msg.reply_text(result, parse_mode="Markdown")
    elif action == "richanalysis":
        await msg.reply_text(f"⏳ מנתח את {symbol}…")
        result = await asyncio.get_event_loop().run_in_executor(None, get_rich_analysis, symbol)
        await msg.reply_text(result, parse_mode="Markdown")
    elif action == "ichimoku":
        await msg.reply_text(f"⏳ מחשב Ichimoku Cloud עבור {symbol}…")
        result = await asyncio.get_event_loop().run_in_executor(None, get_ichimoku, symbol)
        await msg.reply_text(result, parse_mode="Markdown")
    elif action == "stoch":
        await msg.reply_text(f"⏳ מחשב Stochastic RSI עבור {symbol}…")
        result = await asyncio.get_event_loop().run_in_executor(None, get_stoch_rsi, symbol)
        await msg.reply_text(result, parse_mode="Markdown")
    elif action == "pivot":
        await msg.reply_text(f"⏳ מחשב Pivot Points עבור {symbol}…")
        result = await asyncio.get_event_loop().run_in_executor(None, get_pivot_points, symbol)
        await msg.reply_text(result, parse_mode="Markdown")
    elif action == "obv":
        await msg.reply_text(f"⏳ מחשב OBV עבור {symbol}…")
        result = await asyncio.get_event_loop().run_in_executor(None, get_obv, symbol)
        await msg.reply_text(result, parse_mode="Markdown")
    elif action == "atr":
        await msg.reply_text(f"⏳ מחשב ATR עבור {symbol}…")
        result = await asyncio.get_event_loop().run_in_executor(None, get_atr, symbol)
        await msg.reply_text(result, parse_mode="Markdown")
    elif action == "stoploss":
        await msg.reply_text(
            f"✅ *{symbol}* נבחר\n\nשלח את מחיר הכניסה (למשל: `45.50`)",
            parse_mode="Markdown",
        )
        if ctx is not None:
            ctx.user_data["pending_stoploss"] = symbol


# ─────────────────────────────────────────────────────────────
#  Inline callback handler
# ─────────────────────────────────────────────────────────────

async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data
    prefix, _, payload = data.partition(":")

    if prefix in ("chart", "fib", "bb", "vwap", "levels", "analysis",
                  "setups", "richanalysis",
                  "ichimoku", "stoch", "pivot", "obv", "atr", "stoploss"):
        await _run_symbol_action(prefix, payload, query.message, ctx)

    elif prefix == "more":
        if payload == "earnings":
            await query.message.reply_text("⏳ בודק Earnings…")
            msgs = await asyncio.get_event_loop().run_in_executor(None, build_earnings_messages)
            if msgs:
                for m in msgs:
                    await query.message.reply_text(m, parse_mode="Markdown")
            else:
                await query.message.reply_text("אין Earnings קרובים ברשימה.")
        elif payload == "levels":
            await query.message.reply_text(
                "בחר מנייה לרמות תמיכה/התנגדות:",
                reply_markup=build_symbol_picker("levels"),
            )
        elif payload == "news":
            await query.message.reply_text("⏳ מושך חדשות…")
            msg = await asyncio.get_event_loop().run_in_executor(None, build_morning_message)
            await query.message.reply_text(msg, parse_mode="Markdown")
        elif payload == "help":
            await query.message.reply_text(
                "📋 *פקודות זמינות:*\n"
                "  /status — מחירים נוכחיים + RSI\n"
                "  /analysis NNE — ניתוח מלא\n"
                "  /levels NNE — תמיכות והתנגדויות\n"
                "  /chart NNE — גרף נרות 30 ימים\n"
                "  /bb NNE — Bollinger Bands\n"
                "  /fib NNE — רמות פיבונאצ'י\n"
                "  /vwap NNE — VWAP\n"
                "  /ichimoku NNE — Ichimoku Cloud\n"
                "  /stoch NNE — Stochastic RSI\n"
                "  /pivot NNE — Pivot Points יומיים\n"
                "  /obv NNE — On Balance Volume\n"
                "  /atr NNE — ATR (תנודתיות)\n"
                "  /stoploss NNE 45.50 — Stop Loss אוטומטי\n"
                "  /stop — השהה התראות\n"
                "  /resume — חדש התראות",
                parse_mode="Markdown",
            )


# ─────────────────────────────────────────────────────────────
#  Reply-keyboard text handler
# ─────────────────────────────────────────────────────────────

async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()

    pending_sl = ctx.user_data.get("pending_stoploss")
    if pending_sl:
        try:
            entry = float(text.replace(",", "."))
            ctx.user_data.pop("pending_stoploss", None)
            await update.message.reply_text(f"⏳ מחשב Stop Loss עבור {pending_sl}…")
            result = await asyncio.get_event_loop().run_in_executor(None, get_stoploss, pending_sl, entry)
            await update.message.reply_text(result, parse_mode="Markdown")
            return
        except ValueError:
            await update.message.reply_text(
                "❌ מחיר לא תקין. שלח מספר בלבד (למשל: `45.50`)",
                parse_mode="Markdown",
            )
            return

    if text == "📈 סטטוס":
        await _send_status(update.message)
    elif text == "🔍 סקרינר":
        await update.message.reply_text("⏳ מריץ סקרינר… (עשוי לקחת זמן)")
        msg = await asyncio.get_event_loop().run_in_executor(None, build_screener_message)
        await update.message.reply_text(msg, parse_mode="Markdown")
    elif text in SYMBOL_ACTIONS:
        prefix = _ACTION_PREFIX[text]
        labels = {
            "chart": "📊 גרף", "fib": "📐 פיבונאצ'י", "bb": "📉 BB",
            "vwap": "⚡ VWAP", "setups": "🎯 סט-אפים", "richanalysis": "🔎 ניתוח",
            "ichimoku": "☁️ Ichimoku", "stoch": "📉 Stoch RSI", "pivot": "📐 Pivot",
            "obv": "💹 OBV", "stoploss": "🛑 Stop Loss",
        }
        await update.message.reply_text(
            f"בחר מנייה עבור {labels.get(prefix, prefix)}:",
            reply_markup=build_symbol_picker(prefix),
        )
    elif text == "😱 Fear & Greed":
        await update.message.reply_text("⏳ מושך Fear & Greed Index…")
        msg = await asyncio.get_event_loop().run_in_executor(None, build_fear_greed_message)
        await update.message.reply_text(msg, parse_mode="Markdown")
    elif text == "🗂 עוד פקודות":
        await update.message.reply_text("בחר פעולה:", reply_markup=build_more_keyboard())


# ─────────────────────────────────────────────────────────────
#  Main — runs FastAPI + Telegram bot concurrently
# ─────────────────────────────────────────────────────────────

def main() -> None:
    global OWNER_CHAT_ID, _dynamic_watchlist

    if not TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN לא מוגדר!")

    init_db()

    # Restore chat_id
    saved = get_setting("chat_id")
    if saved:
        OWNER_CHAT_ID = int(saved)
        log.info("Restored owner chat_id: %d", OWNER_CHAT_ID)

    # Restore dynamic watchlist
    saved_wl = get_setting("dynamic_watchlist")
    if saved_wl:
        _dynamic_watchlist = saved_wl.split(",")
        log.info("Restored watchlist: %s", _dynamic_watchlist)

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
    app.add_handler(CommandHandler("ichimoku", cmd_ichimoku))
    app.add_handler(CommandHandler("stoch",    cmd_stoch))
    app.add_handler(CommandHandler("pivot",    cmd_pivot))
    app.add_handler(CommandHandler("obv",      cmd_obv))
    app.add_handler(CommandHandler("atr",      cmd_atr))
    app.add_handler(CommandHandler("stoploss", cmd_stoploss))
    app.add_handler(CommandHandler("stop",     cmd_stop))
    app.add_handler(CommandHandler("resume",   cmd_resume))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    async def post_init(application: Application) -> None:
        asyncio.create_task(scan_loop(application))
        asyncio.create_task(morning_news_loop(application))
        asyncio.create_task(earnings_loop(application))
        asyncio.create_task(screener_loop(application))
        asyncio.create_task(setup_scan_loop(application))
        # Start FastAPI web server
        config = uvicorn.Config(api, host="0.0.0.0", port=PORT, log_level="warning")
        server = uvicorn.Server(config)
        asyncio.create_task(server.serve())
        log.info("✅ FastAPI server started on port %d", PORT)

    app.post_init = post_init

    log.info("✅ Trading bot started — watching %d symbols", len(_dynamic_watchlist))
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()