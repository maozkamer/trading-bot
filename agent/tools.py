"""
9 tools available to the trading agent.
All functions are synchronous (run inside ThreadPoolExecutor in core.py).
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

log = logging.getLogger(__name__)

TWELVE_DATA_KEY = (
    os.environ.get("TWELVE_DATA_KEY") or
    os.environ.get("TWELVEDATAKEY") or
    os.environ.get("TWELVE_DATA_API_KEY")
)
TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
USER_CHAT_ID     = os.environ.get("USER_TELEGRAM_CHAT_ID", "468005359")
NEWSAPI_KEY      = os.environ.get("NEWSAPI_KEY", "")

# ─────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────

def _fetch_intraday(symbol: str, interval: str = "15min", outputsize: int = 96) -> list[dict]:
    """
    Fetch intraday OHLCV bars from Twelve Data.
    Returns list of raw value dicts (newest last).
    """
    api_key = TWELVE_DATA_KEY
    if not api_key:
        raise RuntimeError("TWELVEDATAKEY not set")

    api_sym = symbol.replace("-USD", "/USD") if symbol.endswith("-USD") else symbol
    resp = requests.get(
        TWELVE_DATA_URL,
        params={
            "symbol":     api_sym,
            "interval":   interval,
            "outputsize": outputsize,
            "apikey":     api_key,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "code" in data:
        raise ValueError(f"Twelve Data error: {data.get('message', data)}")
    values = data.get("values")
    if not values:
        raise ValueError(f"No intraday values for {symbol}")
    return list(reversed(values))   # oldest → newest


# ─────────────────────────────────────────────────────────────
#  Tool 1 — analyze_stock
# ─────────────────────────────────────────────────────────────

def analyze_stock(symbol: str) -> dict:
    """
    Full technical analysis for *symbol*.
    Delegates to analysis.get_rich_analysis which returns a formatted string,
    plus returns raw indicator values for the agent to reason over.
    """
    symbol = symbol.upper().strip()
    try:
        from analysis import (
            _fetch_daily, _rsi, _macd, _bollinger, _sma,
            _find_sr, _atr, _vwap, get_rich_analysis,
        )

        df = _fetch_daily(symbol)
        if df is None or df.empty or len(df) < 20:
            return {"error": f"אין מספיק נתונים עבור {symbol}"}

        closes  = df["Close"]
        price   = float(closes.iloc[-1])
        prev    = float(closes.iloc[-2])
        change  = (price - prev) / prev * 100

        rsi_val             = _rsi(closes) if len(closes) >= 15 else None
        macd_v, sig_v, hist_v, _ = _macd(closes) if len(closes) >= 30 else (None, None, None, None)
        bb_upper, bb_mid, bb_lower = _bollinger(closes)
        ma50,  _  = _sma(closes, 50)
        ma200, _  = _sma(closes, 200)
        atr_val   = _atr(df)
        vwap_val, _ = _vwap(df)
        supports, resistances = _find_sr(df)

        formatted = get_rich_analysis(symbol)

        return {
            "symbol":      symbol,
            "price":       round(price, 2),
            "change_pct":  round(change, 2),
            "rsi":         round(rsi_val, 2) if rsi_val is not None else None,
            "macd":        round(macd_v,  3) if macd_v  is not None else None,
            "macd_signal": round(sig_v,   3) if sig_v   is not None else None,
            "macd_hist":   round(hist_v,  3) if hist_v  is not None else None,
            "bb_upper":    round(bb_upper, 2) if bb_upper is not None else None,
            "bb_lower":    round(bb_lower, 2) if bb_lower is not None else None,
            "ma50":        round(ma50,  2) if ma50  is not None else None,
            "ma200":       round(ma200, 2) if ma200 is not None else None,
            "atr":         round(atr_val, 3) if atr_val is not None else None,
            "vwap":        round(vwap_val, 2) if vwap_val is not None else None,
            "supports":    [round(s, 2) for s in supports[:3]],
            "resistances": [round(r, 2) for r in resistances[:3]],
            "formatted":   formatted,
        }
    except Exception as exc:
        log.error("analyze_stock %s: %s", symbol, exc)
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────
#  Tool 2 — scan_universe
# ─────────────────────────────────────────────────────────────

def scan_universe() -> list[dict]:
    """
    Parallel scan of the full trading universe.
    Returns list of dicts with price, change, RSI, sorted by RSI ascending.
    """
    from agent.scanner import build_dynamic_universe
    from analysis import _fetch_daily, _rsi

    universe = build_dynamic_universe()
    results: list[dict] = []

    def _fetch_one(sym: str) -> dict:
        try:
            df = _fetch_daily(sym)
            if df is None or df.empty or len(df) < 15:
                return {"symbol": sym, "price": None}
            closes = df["Close"]
            price  = float(closes.iloc[-1])
            prev   = float(closes.iloc[-2])
            change = (price - prev) / prev * 100
            rsi    = _rsi(closes)
            vol    = float(df["Volume"].iloc[-1])
            avg_vol = float(df["Volume"].iloc[-21:-1].mean()) if len(df) >= 21 else vol
            return {
                "symbol":    sym,
                "price":     round(price, 2),
                "change":    round(change, 2),
                "rsi":       round(rsi, 1) if rsi is not None else None,
                "vol_ratio": round(vol / avg_vol, 2) if avg_vol > 0 else None,
            }
        except Exception as exc:
            log.warning("scan_universe %s: %s", sym, exc)
            return {"symbol": sym, "price": None}

    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(_fetch_one, sym): sym for sym in universe}
        for fut in as_completed(futures):
            results.append(fut.result())

    # Sort: symbols with data first, by RSI ascending (oversold first)
    results.sort(key=lambda x: (x["price"] is None, x.get("rsi") or 999))
    return results


# ─────────────────────────────────────────────────────────────
#  Tool 3 — detect_patterns
# ─────────────────────────────────────────────────────────────

def detect_patterns(symbol: str) -> list[dict]:
    """
    Detect classic chart patterns for *symbol* using daily + intraday data.
    Returns list of dicts: {pattern, confidence, entry, target, stop, reason}.
    """
    symbol = symbol.upper().strip()
    try:
        from analysis import get_active_setups, _fetch_daily, _find_sr, _rsi

        # Daily patterns via existing engine
        setups = get_active_setups(symbol)
        patterns: list[dict] = []
        for s in setups:
            patterns.append({
                "pattern":    s.get("name", "Unknown"),
                "confidence": min(1.0, s.get("vol_ratio", 1.0) / 2),
                "entry":      s.get("entry"),
                "target":     s.get("target"),
                "stop":       s.get("stop"),
                "timeframe":  "daily",
                "reason":     s.get("reason", ""),
            })

        # Intraday consolidation break (15min)
        try:
            bars = _fetch_intraday(symbol, interval="15min", outputsize=96)
            if bars and len(bars) >= 20:
                highs  = [float(b["high"])  for b in bars[-20:]]
                lows   = [float(b["low"])   for b in bars[-20:]]
                closes = [float(b["close"]) for b in bars[-20:]]
                price  = closes[-1]

                range_high = max(highs[:-3])
                range_low  = min(lows[:-3])
                band       = range_high - range_low

                if band > 0:
                    # Consolidation break up
                    if price > range_high and closes[-2] <= range_high:
                        patterns.append({
                            "pattern":    "Consolidation Break (bullish)",
                            "confidence": 0.70,
                            "entry":      round(range_high, 2),
                            "target":     round(range_high + band, 2),
                            "stop":       round(range_low, 2),
                            "timeframe":  "15min",
                            "reason":     f"פריצה מעל {range_high:.2f} אחרי קונסולידציה",
                        })
                    # Consolidation break down
                    elif price < range_low and closes[-2] >= range_low:
                        patterns.append({
                            "pattern":    "Consolidation Break (bearish)",
                            "confidence": 0.65,
                            "entry":      round(range_low, 2),
                            "target":     round(range_low - band, 2),
                            "stop":       round(range_high, 2),
                            "timeframe":  "15min",
                            "reason":     f"שבירה מתחת {range_low:.2f} אחרי קונסולידציה",
                        })
        except Exception as exc:
            log.debug("intraday pattern check skipped for %s: %s", symbol, exc)

        # Sort by confidence descending
        patterns.sort(key=lambda x: -(x.get("confidence") or 0))
        return patterns

    except Exception as exc:
        log.error("detect_patterns %s: %s", symbol, exc)
        return []


# ─────────────────────────────────────────────────────────────
#  Tool 4 — fetch_news
# ─────────────────────────────────────────────────────────────

def fetch_news(query: str = "", since_hours: int = 24) -> list[dict]:
    """
    Fetch news from NewsAPI (if key available) + RSS feeds.
    Returns list of {title, source, url, published_at}.
    """
    articles: list[dict] = []

    # NewsAPI
    if NEWSAPI_KEY:
        try:
            from datetime import datetime, timedelta, timezone
            from_dt = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
            params: dict = {
                "apiKey":   NEWSAPI_KEY,
                "language": "en",
                "sortBy":   "publishedAt",
                "from":     from_dt,
                "pageSize": 20,
            }
            if query:
                params["q"] = query
                url = "https://newsapi.org/v2/everything"
            else:
                params["category"] = "business"
                url = "https://newsapi.org/v2/top-headlines"
                params["country"] = "us"

            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            for a in resp.json().get("articles", []):
                articles.append({
                    "title":        a.get("title", ""),
                    "source":       a.get("source", {}).get("name", ""),
                    "url":          a.get("url", ""),
                    "published_at": a.get("publishedAt", ""),
                })
        except Exception as exc:
            log.warning("NewsAPI fetch failed: %s", exc)

    # RSS fallback
    if not articles:
        try:
            from news import fetch_news as rss_fetch
            headlines, _ = rss_fetch()
            for h in headlines:
                articles.append({"title": h, "source": "RSS", "url": "", "published_at": ""})
        except Exception as exc:
            log.warning("RSS fetch failed: %s", exc)

    return articles[:15]


# ─────────────────────────────────────────────────────────────
#  Tool 5 — check_earnings
# ─────────────────────────────────────────────────────────────

def check_earnings(symbol: str) -> dict:
    """
    Return upcoming earnings info for *symbol*.
    Uses FMP API if key available, otherwise returns placeholder.
    """
    symbol = symbol.upper().strip()
    try:
        from news import fetch_upcoming_earnings
        all_upcoming = fetch_upcoming_earnings()
        for e in all_upcoming:
            if e.get("symbol") == symbol:
                return e
        return {"symbol": symbol, "message": "אין earnings קרובים ב-5 הימים הבאים"}
    except Exception as exc:
        log.error("check_earnings %s: %s", symbol, exc)
        return {"symbol": symbol, "error": str(exc)}


# ─────────────────────────────────────────────────────────────
#  Tool 6 — get_market_overview
# ─────────────────────────────────────────────────────────────

def get_market_overview() -> dict:
    """
    Fetch SPY, QQQ, VIX data via Twelve Data.
    Returns dict with prices and daily changes.
    """
    from analysis import _fetch_daily, _rsi

    symbols = {"SPY": None, "QQQ": None, "VIX": None}
    overview: dict = {}

    for sym in symbols:
        try:
            df = _fetch_daily(sym)
            if df is None or df.empty:
                overview[sym] = {"error": "no data"}
                continue
            price  = float(df["Close"].iloc[-1])
            prev   = float(df["Close"].iloc[-2])
            change = (price - prev) / prev * 100
            rsi    = _rsi(df["Close"]) if len(df) >= 15 else None
            overview[sym] = {
                "price":      round(price, 2),
                "change_pct": round(change, 2),
                "rsi":        round(rsi, 1) if rsi is not None else None,
            }
        except Exception as exc:
            log.warning("get_market_overview %s: %s", sym, exc)
            overview[sym] = {"error": str(exc)}

    return overview


# ─────────────────────────────────────────────────────────────
#  Tool 7 — save_insight
# ─────────────────────────────────────────────────────────────

def save_insight(category: str, content: str) -> dict:
    """Persist a long-term insight. Returns status dict."""
    from agent.memory import save_insight as _save
    ok = _save(category, content)
    return {"saved": ok, "category": category}


# ─────────────────────────────────────────────────────────────
#  Tool 8 — recall_memory
# ─────────────────────────────────────────────────────────────

def recall_memory(query: str) -> list[dict]:
    """Retrieve relevant past insights for *query*."""
    from agent.memory import recall_memory as _recall
    return _recall(query, limit=5)


# ─────────────────────────────────────────────────────────────
#  Tool 9 — send_telegram
# ─────────────────────────────────────────────────────────────

def send_telegram(message: str, chat_id: str | None = None) -> dict:
    """
    Send a message to the user via Telegram Bot API.
    Uses USER_TELEGRAM_CHAT_ID env var as default recipient.
    """
    target = chat_id or USER_CHAT_ID
    if not TELEGRAM_TOKEN:
        return {"sent": False, "error": "TELEGRAM_TOKEN not set"}
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": target, "text": message, "parse_mode": "Markdown"},
            timeout=15,
        )
        resp.raise_for_status()
        return {"sent": True, "chat_id": target}
    except Exception as exc:
        log.error("send_telegram failed: %s", exc)
        return {"sent": False, "error": str(exc)}


# ─────────────────────────────────────────────────────────────
#  Tool registry — used by core.py to build Claude tool defs
# ─────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, callable] = {
    "analyze_stock":      analyze_stock,
    "scan_universe":      scan_universe,
    "detect_patterns":    detect_patterns,
    "fetch_news":         fetch_news,
    "check_earnings":     check_earnings,
    "get_market_overview": get_market_overview,
    "save_insight":       save_insight,
    "recall_memory":      recall_memory,
    "send_telegram":      send_telegram,
}

TOOL_DEFINITIONS = [
    {
        "name": "analyze_stock",
        "description": "ניתוח טכני מלא של מנייה: מחיר, RSI, MACD, Bollinger Bands, MA50/200, ATR, VWAP, תמיכות והתנגדויות.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "סימול המנייה, למשל NNE או BTC-USD"}
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "scan_universe",
        "description": "סריקה מקבילית של כל Universe המניות. מחזיר רשימה עם מחיר, שינוי%, RSI, ויחס נפח.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "detect_patterns",
        "description": "זיהוי פטרנים קלאסיים (Breakout, Bull Flag, Double Bottom וכו') במנייה ב-Daily ו-15min.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "סימול המנייה"}
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "fetch_news",
        "description": "שליפת חדשות פיננסיות. ניתן לסנן לפי query ולפי כמה שעות אחורה.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":       {"type": "string",  "description": "מונח חיפוש, לדוגמה 'MARA bitcoin'"},
                "since_hours": {"type": "integer", "description": "כמה שעות אחורה (ברירת מחדל: 24)"},
            },
        },
    },
    {
        "name": "check_earnings",
        "description": "בדיקת מועד דוח הרווחים הקרוב של מנייה ספציפית.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "סימול המנייה"}
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_market_overview",
        "description": "מצב השוק הכללי: SPY, QQQ, VIX — מחיר ושינוי יומי.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "save_insight",
        "description": "שמירת תובנה ארוכת טווח לזיכרון הסוכן.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["trading_patterns", "market_correlations", "user_preferences", "lessons_learned"],
                    "description": "קטגוריית התובנה",
                },
                "content": {"type": "string", "description": "תוכן התובנה"},
            },
            "required": ["category", "content"],
        },
    },
    {
        "name": "recall_memory",
        "description": "שליפת תובנות עבר רלוונטיות מהזיכרון ארוך הטווח.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "מה לחפש בזיכרון"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "send_telegram",
        "description": "שליחת התראה למעוז בטלגרם.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "תוכן ההודעה (Markdown)"}
            },
            "required": ["message"],
        },
    },
]
