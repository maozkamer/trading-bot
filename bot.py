"""
Swing-trading Discord bot — main entry point.

Ported from the original Telegram bot. Everything below the messenger layer
(analysis.py, charts.py, news.py, database.py, agent/) is reused untouched;
only the presentation/handler layer is Discord-native.

Structure on the server (created by /setup):
  📈 מסחר
    #התראות     — automated scan alerts
    #ניתוח      — commands, plain-text tickers, and the pinned control panel

Data is pulled per-symbol on demand only. There is deliberately no bulk
watchlist status sweep: Twelve Data's free tier allows ~8 requests/minute,
and polling all 22 symbols on a timer exhausted it (429s).
  🤖 עוזר אישי
    #עוזר-אישי  — the separate Gemini assistant bot lives here

All user-facing text is Hebrew.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import re
from datetime import datetime

import discord
import pytz
from aiohttp import web
from discord import app_commands

from analysis import (
    WATCHLIST,
    Alert,
    analyze_symbol,
    check_sr_proximity,
    get_full_analysis,
    get_rich_analysis,
    get_levels,
    get_bollinger_levels,
    get_fibonacci_levels,
    get_vwap,
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
from agent import setup_scheduler
from agent.memory import init_memory_db, clear_history

# ─────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────

DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
OWNER_ID = int(os.environ.get("DISCORD_OWNER_ID", "0"))
EST_TZ = pytz.timezone("US/Eastern")
TZ = pytz.timezone("Asia/Jerusalem")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN חסר — הגדר אותו ב-fly secrets")

# Shared secret with the personal-assistant bot, which hands trading questions to
# this agent over HTTP. Without it the /ask endpoint stays closed.
AGENT_LINK_TOKEN = os.environ.get("AGENT_LINK_TOKEN")
AGENT_LINK_PORT = int(os.environ.get("AGENT_LINK_PORT", "8080"))
AGENT_LINK_CHAT_ID = "quarterback"  # its own memory thread, separate from the Discord channels

alerts_paused: bool = False
_dynamic_watchlist: list[str] = list(WATCHLIST)

CATEGORY_TRADING = "📈 מסחר"
CATEGORY_ASSISTANT = "🤖 עוזר אישי"
CH_ALERTS = "התראות"
CH_ANALYSIS = "ניתוח"
CH_RECOMMENDER = "הממליץ"
CH_ASSISTANT = "עוזר-אישי"

DISCORD_LIMIT = 1900  # hard cap is 2000; leave room


# ─────────────────────────────────────────────────────────────
#  Formatting helpers
# ─────────────────────────────────────────────────────────────

def tg_to_discord_md(text: str) -> str:
    """The analysis modules emit Telegram Markdown, where *x* is BOLD. In Discord
    *x* is italic and **x** is bold, so single-star runs must be doubled or every
    heading in the ported output silently turns into italics."""
    if not text:
        return text
    # Leave existing **bold** alone; only promote lone *…* runs.
    return re.sub(r"(?<!\*)\*(?!\*)([^*\n]+?)(?<!\*)\*(?!\*)", r"**\1**", text)


def chunk(text: str) -> list[str]:
    return [text[i : i + DISCORD_LIMIT] for i in range(0, len(text), DISCORD_LIMIT)] or [""]


async def send_long(target, text: str, **kwargs) -> None:
    """Send text that may exceed Discord's per-message cap. kwargs (view/file) ride
    on the final chunk so controls sit at the bottom of the output."""
    parts = chunk(tg_to_discord_md(text))
    for part in parts[:-1]:
        await target.send(part)
    await target.send(parts[-1], **kwargs)


async def followup_long(interaction: discord.Interaction, text: str, **kwargs) -> None:
    parts = chunk(tg_to_discord_md(text))
    for part in parts[:-1]:
        await interaction.followup.send(part)
    await interaction.followup.send(parts[-1], **kwargs)


def is_market_hours() -> bool:
    now = datetime.now(EST_TZ)
    if now.weekday() >= 5:
        return False
    open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now <= close_t


def format_alert(alert: Alert) -> str:
    sup = f"${alert.support:.2f}" if alert.support else "—"
    res = f"${alert.resistance:.2f}" if alert.resistance else "—"
    return (
        f"🚨 **{alert.symbol}** — {alert.title}\n"
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
        f"📊 **{alert.symbol}** מתקרב ל{level_type} ב-${level_price:.2f}\n"
        f"💵 מחיר נוכחי: ${alert.price:.2f}\n"
        f"📉 מרחק: {dist_pct:.1f}% מה{level_type}\n"
        f"⚡ שים לב — {alert.recommendation}"
    )


# ─────────────────────────────────────────────────────────────
#  Discord client
# ─────────────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


def _channel(name: str) -> discord.TextChannel | None:
    """Look up a bot channel by the id stashed in settings at /setup time."""
    raw = get_setting(f"ch_{name}")
    if not raw:
        return None
    ch = client.get_channel(int(raw))
    return ch if isinstance(ch, discord.TextChannel) else None


async def _post(name: str, text: str, fallback_owner: bool = True, **kwargs) -> None:
    """Post to one of the bot's channels, falling back to a DM if /setup never ran."""
    ch = _channel(name)
    if ch is None and fallback_owner and OWNER_ID:
        try:
            ch = await client.fetch_user(OWNER_ID)
        except Exception as exc:
            log.error("Could not resolve owner for DM: %s", exc)
            return
    if ch is None:
        log.warning("No channel %r and no owner fallback — dropping message", name)
        return
    try:
        await send_long(ch, text, **kwargs)
    except Exception as exc:
        log.error("Failed posting to %s: %s", name, exc)


# ─────────────────────────────────────────────────────────────
#  Autocomplete
# ─────────────────────────────────────────────────────────────

async def symbol_autocomplete(
    interaction: discord.Interaction, current: str
) -> list[app_commands.Choice[str]]:
    cur = (current or "").upper()
    matches = [s for s in _dynamic_watchlist if cur in s]
    # Let the user analyse something off-watchlist too, without losing the suggestions.
    if cur and cur not in matches:
        matches = matches + [cur]
    return [app_commands.Choice(name=s, value=s) for s in matches[:25]]


# ─────────────────────────────────────────────────────────────
#  Action buttons
# ─────────────────────────────────────────────────────────────

class SymbolView(discord.ui.View):
    """Refresh / chart / news controls attached to an analysis result."""

    def __init__(self, symbol: str):
        super().__init__(timeout=900)
        self.symbol = symbol

    @discord.ui.button(label="רענן", emoji="🔄", style=discord.ButtonStyle.secondary)
    async def refresh(self, interaction: discord.Interaction, _: discord.ui.Button) -> None:
        await interaction.response.defer(thinking=True)
        # Drop the cached frame so this really re-fetches instead of replaying the
        # 55-minute cache the scheduled scans share.
        from analysis import _CACHE
        for key in [k for k in _CACHE if k.startswith(f"{self.symbol}:")]:
            _CACHE.pop(key, None)
        text = await asyncio.to_thread(get_full_analysis, self.symbol)
        await followup_long(interaction, text, view=SymbolView(self.symbol))

    @discord.ui.button(label="גרף", emoji="📈", style=discord.ButtonStyle.secondary)
    async def chart(self, interaction: discord.Interaction, _: discord.ui.Button) -> None:
        await interaction.response.defer(thinking=True)
        file, caption = await _build_chart_file(self.symbol)
        if file is None:
            await interaction.followup.send(caption)
            return
        await interaction.followup.send(tg_to_discord_md(caption), file=file)

    @discord.ui.button(label="רמות", emoji="📊", style=discord.ButtonStyle.secondary)
    async def levels(self, interaction: discord.Interaction, _: discord.ui.Button) -> None:
        await interaction.response.defer(thinking=True)
        text = await asyncio.to_thread(get_levels, self.symbol)
        await followup_long(interaction, text)


class TickerSelect(discord.ui.Select):
    """Dropdown of the watchlist — the no-typing path to an analysis."""

    def __init__(self) -> None:
        options = [
            discord.SelectOption(label=s, description=f"ניתוח טכני מלא ל-{s}")
            for s in _dynamic_watchlist[:25]  # Discord caps a select at 25 options
        ]
        super().__init__(
            placeholder="בחר מניה לניתוח…",
            options=options,
            custom_id="panel:ticker_select",
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True)
        symbol = self.values[0]
        text = await asyncio.to_thread(get_full_analysis, symbol)
        await followup_long(interaction, text, view=SymbolView(symbol))


class PanelView(discord.ui.View):
    """Pinned control panel. timeout=None + fixed custom_ids so the buttons keep
    working after a redeploy instead of going dead."""

    def __init__(self) -> None:
        super().__init__(timeout=None)
        self.add_item(TickerSelect())


async def _build_chart_file(symbol: str) -> tuple[discord.File | None, str]:
    try:
        df = await asyncio.to_thread(_fetch_daily, symbol)
        if df.empty or len(df) < 5:
            return None, f"❌ אין מספיק נתונים עבור {symbol}"
        supports, resistances = _find_sr(df)
        buf = await asyncio.to_thread(build_chart, symbol, df, supports, resistances)
        caption = (
            f"📈 *{symbol}* — 30 ימים אחרונים\n"
            f"🛡️ תמיכות: {', '.join(f'${s:.2f}' for s in supports[:3]) or '—'}\n"
            f"🔺 התנגדויות: {', '.join(f'${r:.2f}' for r in resistances[:3]) or '—'}"
        )
        return discord.File(buf, filename=f"{symbol}.png"), caption
    except Exception as exc:
        log.error("chart failed for %s: %s", symbol, exc)
        return None, f"❌ שגיאה בבניית הגרף: {exc}"


# ─────────────────────────────────────────────────────────────
#  Slash commands
# ─────────────────────────────────────────────────────────────

@tree.command(name="setup", description="יוצר את קטגוריות וערוצי הבוט בשרת")
async def cmd_setup(interaction: discord.Interaction) -> None:
    if interaction.guild is None:
        await interaction.response.send_message("צריך להריץ את זה בתוך שרת, לא ב-DM.")
        return
    await interaction.response.defer(thinking=True)

    guild = interaction.guild
    created: list[str] = []

    async def ensure_category(name: str) -> discord.CategoryChannel:
        existing = discord.utils.get(guild.categories, name=name)
        if existing:
            return existing
        created.append(name)
        return await guild.create_category(name)

    async def ensure_channel(cat: discord.CategoryChannel, name: str) -> discord.TextChannel:
        existing = discord.utils.get(guild.text_channels, name=name, category=cat)
        if existing is None:
            existing = await guild.create_text_channel(name, category=cat)
            created.append(f"#{name}")
        save_setting(f"ch_{name}", str(existing.id))
        return existing

    try:
        trading = await ensure_category(CATEGORY_TRADING)
        for ch_name in (CH_ALERTS, CH_ANALYSIS, CH_RECOMMENDER):
            await ensure_channel(trading, ch_name)

        assistant_cat = await ensure_category(CATEGORY_ASSISTANT)
        await ensure_channel(assistant_cat, CH_ASSISTANT)
    except discord.Forbidden:
        await interaction.followup.send(
            "❌ אין לי הרשאת **Manage Channels**. תוסיף אותה לבוט בהגדרות השרת "
            "(Server Settings → Roles) ואז תריץ שוב `/setup`."
        )
        return

    summary = "\n".join(f"• {c}" for c in created) if created else "הכל כבר היה קיים."
    await interaction.followup.send(f"✅ המבנה מוכן:\n{summary}")


@tree.command(name="analysis", description="ניתוח טכני מלא למניה")
@app_commands.describe(symbol="סימול המניה")
@app_commands.autocomplete(symbol=symbol_autocomplete)
async def cmd_analysis(interaction: discord.Interaction, symbol: str) -> None:
    symbol = symbol.upper().strip()
    await interaction.response.defer(thinking=True)
    text = await asyncio.to_thread(get_full_analysis, symbol)
    await followup_long(interaction, text, view=SymbolView(symbol))


@tree.command(name="deep", description="ניתוח מעמיק (איטי יותר) למניה")
@app_commands.describe(symbol="סימול המניה")
@app_commands.autocomplete(symbol=symbol_autocomplete)
async def cmd_deep(interaction: discord.Interaction, symbol: str) -> None:
    symbol = symbol.upper().strip()
    await interaction.response.defer(thinking=True)
    text = await asyncio.to_thread(get_rich_analysis, symbol)
    await followup_long(interaction, text, view=SymbolView(symbol))


@tree.command(name="chart", description="גרף נרות עם תמיכות והתנגדויות")
@app_commands.describe(symbol="סימול המניה")
@app_commands.autocomplete(symbol=symbol_autocomplete)
async def cmd_chart(interaction: discord.Interaction, symbol: str) -> None:
    symbol = symbol.upper().strip()
    await interaction.response.defer(thinking=True)
    file, caption = await _build_chart_file(symbol)
    if file is None:
        await interaction.followup.send(caption)
        return
    await interaction.followup.send(tg_to_discord_md(caption), file=file, view=SymbolView(symbol))


@tree.command(name="levels", description="רמות תמיכה והתנגדות")
@app_commands.describe(symbol="סימול המניה")
@app_commands.autocomplete(symbol=symbol_autocomplete)
async def cmd_levels(interaction: discord.Interaction, symbol: str) -> None:
    symbol = symbol.upper().strip()
    await interaction.response.defer(thinking=True)
    text = await asyncio.to_thread(get_levels, symbol)
    await followup_long(interaction, text, view=SymbolView(symbol))


# The eight single-indicator commands collapse into one command with a dropdown —
# same coverage, one entry in the slash-command list instead of eight.
_INDICATORS = {
    "bb": ("Bollinger Bands", get_bollinger_levels),
    "fib": ("Fibonacci", get_fibonacci_levels),
    "vwap": ("VWAP", get_vwap),
    "ichimoku": ("Ichimoku Cloud", get_ichimoku),
    "stoch": ("Stochastic RSI", get_stoch_rsi),
    "pivot": ("Pivot Points", get_pivot_points),
    "obv": ("OBV", get_obv),
    "atr": ("ATR", get_atr),
}


@tree.command(name="indicator", description="אינדיקטור בודד למניה")
@app_commands.describe(kind="איזה אינדיקטור", symbol="סימול המניה")
@app_commands.choices(
    kind=[app_commands.Choice(name=label, value=key) for key, (label, _) in _INDICATORS.items()]
)
@app_commands.autocomplete(symbol=symbol_autocomplete)
async def cmd_indicator(
    interaction: discord.Interaction, kind: app_commands.Choice[str], symbol: str
) -> None:
    symbol = symbol.upper().strip()
    await interaction.response.defer(thinking=True)
    _, fn = _INDICATORS[kind.value]
    text = await asyncio.to_thread(fn, symbol)
    await followup_long(interaction, text, view=SymbolView(symbol))


@tree.command(name="stoploss", description="חישוב סטופ לוס לפי מחיר כניסה")
@app_commands.describe(symbol="סימול המניה", entry="מחיר הכניסה")
@app_commands.autocomplete(symbol=symbol_autocomplete)
async def cmd_stoploss(interaction: discord.Interaction, symbol: str, entry: float) -> None:
    symbol = symbol.upper().strip()
    await interaction.response.defer(thinking=True)
    text = await asyncio.to_thread(get_stoploss, symbol, entry)
    await followup_long(interaction, text)


@tree.command(name="agent", description="שאלה חופשית לסוכן ה-AI")
@app_commands.describe(question="מה לשאול")
async def cmd_agent(interaction: discord.Interaction, question: str) -> None:
    await interaction.response.defer(thinking=True)
    try:
        from agent.core import run_agent_async
        result = await run_agent_async(question, interaction.channel_id)
        await followup_long(interaction, result or "אין תוצאה.")
    except Exception as exc:
        log.error("cmd_agent error: %s", exc, exc_info=True)
        await interaction.followup.send(f"❌ שגיאה: {exc}"[:DISCORD_LIMIT])


@tree.command(name="reset", description="מוחק את היסטוריית השיחה מול הסוכן")
async def cmd_reset(interaction: discord.Interaction) -> None:
    clear_history(str(interaction.channel_id))
    await interaction.response.send_message("היסטוריית השיחה נמחקה ✅")


@tree.command(name="recommender", description="הממליץ — המלצות פתוחות וסקורבורד")
async def cmd_recommender(interaction: discord.Interaction) -> None:
    await interaction.response.defer(thinking=True)
    try:
        from agent.recommender import format_status

        text = await asyncio.to_thread(format_status)
        await followup_long(interaction, text)
    except Exception as exc:
        log.error("cmd_recommender error: %s", exc, exc_info=True)
        await interaction.followup.send(f"❌ שגיאה: {exc}"[:DISCORD_LIMIT])


@tree.command(name="recommender-run", description="מריץ את סריקת הממליץ עכשיו (בעלים בלבד)")
async def cmd_recommender_run(interaction: discord.Interaction) -> None:
    if OWNER_ID and interaction.user.id != OWNER_ID:
        await interaction.response.send_message("רק הבעלים יכול להריץ את זה.", ephemeral=True)
        return
    await interaction.response.defer(thinking=True)
    try:
        from agent.recommender import run_recommender

        recs = await asyncio.to_thread(run_recommender)
        n = len(recs) if recs is not None else 0
        await interaction.followup.send(
            f"✅ סריקה הושלמה — {n} המלצות חדשות פורסמו ל-#{CH_RECOMMENDER}."
            if n
            else "✅ סריקה הושלמה — שום סטאפ לא עבר את שני הסינונים."
        )
    except Exception as exc:
        log.error("cmd_recommender_run error: %s", exc, exc_info=True)
        await interaction.followup.send(f"❌ שגיאה: {exc}"[:DISCORD_LIMIT])


watchlist_group = app_commands.Group(name="watchlist", description="ניהול רשימת המעקב")


@watchlist_group.command(name="show", description="הצג את רשימת המעקב")
async def wl_show(interaction: discord.Interaction) -> None:
    await interaction.response.send_message(
        "👁️ **רשימת מעקב** (" + str(len(_dynamic_watchlist)) + "):\n"
        + ", ".join(f"`{s}`" for s in _dynamic_watchlist)
    )


@watchlist_group.command(name="add", description="הוסף מניה לרשימת המעקב")
@app_commands.describe(symbol="סימול להוספה")
async def wl_add(interaction: discord.Interaction, symbol: str) -> None:
    symbol = symbol.upper().strip()
    if symbol in _dynamic_watchlist:
        await interaction.response.send_message(f"`{symbol}` כבר ברשימה.")
        return
    _dynamic_watchlist.append(symbol)
    save_setting("watchlist", ",".join(_dynamic_watchlist))
    await interaction.response.send_message(f"✅ `{symbol}` נוסף לרשימת המעקב.")


@watchlist_group.command(name="remove", description="הסר מניה מרשימת המעקב")
@app_commands.describe(symbol="סימול להסרה")
@app_commands.autocomplete(symbol=symbol_autocomplete)
async def wl_remove(interaction: discord.Interaction, symbol: str) -> None:
    symbol = symbol.upper().strip()
    if symbol not in _dynamic_watchlist:
        await interaction.response.send_message(f"`{symbol}` לא נמצא ברשימה.")
        return
    _dynamic_watchlist.remove(symbol)
    save_setting("watchlist", ",".join(_dynamic_watchlist))
    await interaction.response.send_message(f"🗑️ `{symbol}` הוסר מרשימת המעקב.")


tree.add_command(watchlist_group)


@tree.command(name="alerts", description="הפעלה או השהיה של ההתראות האוטומטיות")
@app_commands.describe(mode="on להפעלה, off להשהיה")
@app_commands.choices(
    mode=[
        app_commands.Choice(name="on", value="on"),
        app_commands.Choice(name="off", value="off"),
    ]
)
async def cmd_alerts(interaction: discord.Interaction, mode: app_commands.Choice[str]) -> None:
    global alerts_paused
    alerts_paused = mode.value == "off"
    save_setting("alerts_paused", "1" if alerts_paused else "0")
    await interaction.response.send_message(
        "⏸️ ההתראות הושהו." if alerts_paused else "▶️ ההתראות פעילות."
    )


@tree.command(name="panel", description="שולח לוח בקרה נעוץ עם כפתורים")
async def cmd_panel(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)
    ok = await ensure_panel(force_new=True)
    await interaction.followup.send(
        "✅ לוח הבקרה רוענן." if ok else "❌ לא מצאתי את ערוץ הניתוח. תריץ `/setup` קודם.",
        ephemeral=True,
    )


PANEL_TEXT = (
    "🎛️ **לוח בקרה**\n\n"
    "בחר מניה מהתפריט למטה, או לחץ על כפתור.\n"
    "אפשר גם פשוט **לכתוב כאן** — סימול (`NVDA`) לניתוח, "
    "או שאלה חופשית (\"איזו מניה הכי חזקה?\") שתלך לסוכן ה-AI.\n"
    "אין צורך להפעיל שום דבר — זה תמיד פעיל."
)

CHANNEL_HEADERS = {
    CH_ALERTS: (
        "🔔 **ערוץ ההתראות**\n"
        "כאן יגיעו התראות אוטומטיות מסריקת המניות.\n"
        "כרגע ההתראות **כבויות** — `/alerts on` בערוץ הניתוח יפעיל אותן."
    ),
    CH_RECOMMENDER: (
        "📈 **הממליץ**\n"
        "סריקה יומית אחרי סגירת השוק (07:00). ממליץ לונג רק על סטאפ שעובר "
        "סינון דטרמיניסטי **וגם** שער ביטחון של Claude — אם אין, לא נשלח כלום.\n"
        "כל המלצה נבדקת אוטומטית 14 יום, ואז מגיע 'פסק דין': מה נגע קודם "
        "(יעד/סטופ/זמן) ומה התשואה נטו. `/recommender` למצב ולסקורבורד."
    ),
}


async def _ensure_pinned(channel_name: str, key: str, text: str, view=None):
    """Keep exactly one bot-owned pinned message per channel: edit the stored one if
    it still exists, otherwise post and pin a fresh one."""
    ch = _channel(channel_name)
    if ch is None:
        return None
    raw = get_setting(key)
    if raw:
        try:
            msg = await ch.fetch_message(int(raw))
            await msg.edit(content=text, view=view)
            return msg
        except discord.NotFound:
            pass  # deleted by hand — fall through and make a new one
        except Exception as exc:
            log.warning("Could not refresh pinned message in %s: %s", channel_name, exc)
            return None
    try:
        msg = await ch.send(text, view=view)
        await msg.pin()
        save_setting(key, str(msg.id))
        return msg
    except discord.Forbidden:
        log.error("Missing permission to post/pin in %s", channel_name)
    except Exception as exc:
        log.error("Could not create pinned message in %s: %s", channel_name, exc)
    return None


async def ensure_panel(force_new: bool = False) -> bool:
    """Posted automatically on every startup so the panel is simply always there."""
    if force_new:
        save_setting("panel_msg_id", "")
    msg = await _ensure_pinned(CH_ANALYSIS, "panel_msg_id", PANEL_TEXT, view=PanelView())
    return msg is not None


@tree.command(name="cleanup", description="מוחק ערוצים ישנים וקטגוריות ריקות")
async def cmd_cleanup(interaction: discord.Interaction) -> None:
    """Deletes only leftovers: the retired live channel, and Discord's default empty
    categories. Anything with real content is left alone."""
    if interaction.guild is None:
        await interaction.response.send_message("צריך להריץ בתוך שרת.", ephemeral=True)
        return
    await interaction.response.defer(thinking=True)

    guild = interaction.guild
    removed: list[str] = []
    skipped: list[str] = []

    async def is_empty(ch: discord.TextChannel) -> bool:
        try:
            async for _ in ch.history(limit=1):
                return False
            return True
        except Exception:
            return False

    # 1. the retired live-dashboard channel (bot-created, nothing writes to it now)
    stale = discord.utils.get(guild.text_channels, name="מעקב-חי")
    if stale:
        try:
            await stale.delete(reason="dashboard retired")
            removed.append("#מעקב-חי")
        except Exception as exc:
            skipped.append(f"#מעקב-חי ({exc})")

    # 2. Discord's default categories, but only when genuinely unused
    for cat_name in ("Text Channels", "Voice Channels"):
        cat = discord.utils.get(guild.categories, name=cat_name)
        if cat is None:
            continue
        for ch in list(cat.channels):
            if isinstance(ch, discord.TextChannel) and not await is_empty(ch):
                skipped.append(f"#{ch.name} (יש בו הודעות)")
                continue
            try:
                await ch.delete(reason="unused default channel")
                removed.append(f"#{ch.name}")
            except Exception as exc:
                skipped.append(f"#{ch.name} ({exc})")
        if not cat.channels:
            try:
                await cat.delete(reason="empty default category")
                removed.append(cat_name)
            except Exception as exc:
                skipped.append(f"{cat_name} ({exc})")

    lines = []
    if removed:
        lines.append("🗑️ **נמחק:**\n" + "\n".join(f"• {r}" for r in removed))
    if skipped:
        lines.append("⏭️ **דילגתי:**\n" + "\n".join(f"• {s}" for s in skipped))
    await interaction.followup.send("\n\n".join(lines) or "אין מה לנקות ✨")


# ─────────────────────────────────────────────────────────────
#  Plain-text handling — no slash command needed
# ─────────────────────────────────────────────────────────────

# A bare ticker: 1-5 letters, optionally the -USD crypto suffix.
_TICKER_RE = re.compile(r"^[A-Za-z]{1,5}(-USD)?$")


@client.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return

    analysis_ch = _channel(CH_ANALYSIS)
    in_analysis = analysis_ch is not None and message.channel.id == analysis_ch.id
    is_dm = isinstance(message.channel, discord.DMChannel)
    mentioned = client.user in message.mentions
    if not (in_analysis or is_dm or mentioned):
        return

    text = (message.content or "").strip()
    if mentioned:
        text = re.sub(rf"<@!?{client.user.id}>", "", text).strip()
    if not text or text.startswith("/"):
        return

    async with message.channel.typing():
        try:
            if _TICKER_RE.match(text):
                symbol = text.upper()
                result = await asyncio.to_thread(get_full_analysis, symbol)
                await send_long(message.channel, result, view=SymbolView(symbol))
            else:
                # Anything else is a question for the agent.
                from agent.core import run_agent_async
                result = await run_agent_async(text, message.channel.id)
                await send_long(message.channel, result or "אין תוצאה.")
        except Exception as exc:
            log.error("on_message failed for %r: %s", text, exc, exc_info=True)
            await message.channel.send(f"❌ שגיאה: {exc}"[:DISCORD_LIMIT])


# ─────────────────────────────────────────────────────────────
#  Scanner → #התראות
# ─────────────────────────────────────────────────────────────

_MUTED_KEYS = (
    "rsi_oversold", "rsi_overbought",
    "macd_bullish", "macd_bearish",
    "bb_upper_break", "bb_lower_break",
    "golden_cross", "death_cross",
    "vwap_cross_up", "vwap_cross_down",
)


async def run_scan() -> None:
    if alerts_paused:
        log.info("Alerts paused — skipping scan")
        return

    log.info("Scanning %d symbols…", len(_dynamic_watchlist))
    sent = 0

    for symbol in _dynamic_watchlist:
        try:
            for alert in await asyncio.to_thread(analyze_symbol, symbol):
                if any(alert.key.startswith(k) for k in _MUTED_KEYS):
                    continue
                if is_alert_recent(symbol, alert.key, alert.cooldown_hours):
                    continue
                save_alert(symbol, alert.key)
                await _post(CH_ALERTS, format_alert(alert), view=SymbolView(symbol))
                sent += 1
                await asyncio.sleep(0.4)

            if is_market_hours():
                for alert in await asyncio.to_thread(check_sr_proximity, symbol):
                    if is_alert_recent(symbol, alert.key, alert.cooldown_hours):
                        continue
                    save_alert(symbol, alert.key)
                    await _post(CH_ALERTS, format_sr_alert(alert), view=SymbolView(symbol))
                    sent += 1
                    await asyncio.sleep(0.4)

            await asyncio.sleep(1.5)
        except Exception as exc:
            log.error("Scan error %s: %s", symbol, exc)

    log.info("Scan done — %d alerts sent", sent)


async def scan_loop() -> None:
    await client.wait_until_ready()
    await asyncio.sleep(20)
    while True:
        try:
            await run_scan()
        except Exception as exc:
            log.error("scan_loop error: %s", exc)
        sleep_secs = 3600 if is_market_hours() else 14400
        log.info("Next scan in %.0f min (market_hours=%s)", sleep_secs / 60, is_market_hours())
        await asyncio.sleep(sleep_secs)


# ─────────────────────────────────────────────────────────────
#  Agent push sink — the scheduler jobs call agent.tools.send_telegram
# ─────────────────────────────────────────────────────────────

def _install_agent_sink() -> None:
    """agent/scheduler.py does `from agent.tools import send_telegram` at import time,
    so the function object itself has to keep working — we swap its body's sink rather
    than rebinding the name."""
    import agent.tools as tools

    def discord_sink(message: str, chat_id: str | None = None) -> dict:
        try:
            asyncio.run_coroutine_threadsafe(
                _post(CH_ALERTS, message), client.loop
            )
            return {"sent": True, "channel": CH_ALERTS}
        except Exception as exc:
            log.error("discord_sink failed: %s", exc)
            return {"sent": False, "error": str(exc)}

    tools.send_telegram = discord_sink
    if hasattr(tools, "TOOL_REGISTRY"):
        tools.TOOL_REGISTRY["send_telegram"] = discord_sink

    # scheduler.py already captured the old reference — rebind it there too.
    try:
        import agent.scheduler as sched
        sched.send_telegram = discord_sink
    except Exception as exc:
        log.warning("Could not patch scheduler sink: %s", exc)

    # "הממליץ" posts to its own channel.
    try:
        import agent.recommender as rec

        def recommender_sink(message: str) -> dict:
            try:
                asyncio.run_coroutine_threadsafe(
                    _post(CH_RECOMMENDER, message), client.loop
                )
                return {"sent": True, "channel": CH_RECOMMENDER}
            except Exception as exc:
                log.error("recommender_sink failed: %s", exc)
                return {"sent": False, "error": str(exc)}

        rec.set_sink(recommender_sink)
    except Exception as exc:
        log.warning("Could not install recommender sink: %s", exc)


# ─────────────────────────────────────────────────────────────
#  Agent link — HTTP door for the personal-assistant bot
# ─────────────────────────────────────────────────────────────

async def _handle_healthz(_request: web.Request) -> web.Response:
    return web.json_response(
        {
            "ok": True,
            "bot": "trading",
            "discord_ready": client.is_ready(),
            "alerts_paused": alerts_paused,
            "watchlist": len(_dynamic_watchlist),
        }
    )



async def _handle_ask(request: web.Request) -> web.Response:
    supplied = request.headers.get("X-Auth-Token", "")
    if not AGENT_LINK_TOKEN or not hmac.compare_digest(supplied, AGENT_LINK_TOKEN):
        log.warning("Rejected /ask from %s (bad token)", request.remote)
        return web.json_response({"error": "unauthorized"}, status=401)
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    question = str(payload.get("question", "")).strip() if isinstance(payload, dict) else ""
    if not question:
        return web.json_response({"error": "question is required"}, status=400)

    log.info("Agent link question: %r", question[:200])
    try:
        from agent.core import run_agent_async
        answer = await run_agent_async(question, AGENT_LINK_CHAT_ID)
    except Exception as exc:
        log.error("Agent link failed for %r: %s", question, exc, exc_info=True)
        return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
    return web.json_response({"answer": answer or "אין תוצאה."})


_link_runner: web.AppRunner | None = None


async def start_agent_link_server() -> None:
    global _link_runner
    if _link_runner is not None:  # on_ready fires again on every reconnect
        return
    app = web.Application()
    app.router.add_get("/healthz", _handle_healthz)
    app.router.add_post("/ask", _handle_ask)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", AGENT_LINK_PORT).start()
    _link_runner = runner
    log.info("Agent link listening on :%d (/healthz, /ask%s)",
             AGENT_LINK_PORT, "" if AGENT_LINK_TOKEN else " — disabled, no AGENT_LINK_TOKEN")


# ─────────────────────────────────────────────────────────────
#  Lifecycle
# ─────────────────────────────────────────────────────────────

@client.event
async def on_ready() -> None:
    log.info("Logged in as %s", client.user)
    try:
        synced = await tree.sync()
        log.info("Synced %d slash command(s)", len(synced))
    except Exception as exc:
        log.error("Slash sync failed: %s", exc, exc_info=True)

    # Re-register the panel so its buttons survive a restart.
    try:
        client.add_view(PanelView())
    except Exception as exc:
        log.warning("Could not register persistent panel view: %s", exc)

    # Everything the user needs is posted for them — no command to "start" anything.
    try:
        await ensure_panel()
        for ch_name, header in CHANNEL_HEADERS.items():
            await _ensure_pinned(ch_name, f"header_{ch_name}", header)
        log.info("Pinned panel + channel headers are in place")
    except Exception as exc:
        log.error("Could not set up pinned messages: %s", exc, exc_info=True)

    _install_agent_sink()
    try:
        setup_scheduler()
        log.info("Agent scheduler started")
    except Exception as exc:
        log.error("setup_scheduler failed: %s", exc, exc_info=True)

    client.loop.create_task(scan_loop())

    try:
        await start_agent_link_server()
    except Exception as exc:
        log.error("Agent link server failed to start: %s", exc, exc_info=True)


def main() -> None:
    global alerts_paused, _dynamic_watchlist

    init_db()
    try:
        init_memory_db()
    except Exception as exc:
        log.warning("init_memory_db failed: %s", exc)

    saved = get_setting("watchlist")
    if saved:
        _dynamic_watchlist = [s for s in saved.split(",") if s]
    # Alerts default to OFF — the hourly sweep was far too noisy in practice.
    # Only an explicit `/alerts on` (which stores "0") turns them back on.
    alerts_paused = get_setting("alerts_paused") != "0"

    log.info("Starting trading bot (watchlist=%d symbols)", len(_dynamic_watchlist))
    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
