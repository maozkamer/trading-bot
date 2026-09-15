"""
"הממליץ" — daily long-only setup recommender.

Flow (runs post-close, Tue–Sat 07:00 Israel):

  Gate 1 — deterministic scanner
    Runs every long-side detector on closed daily bars. A symbol passes only
    if independent signal *families* align (structure + at least one of
    momentum / trend / volatility), volume confirms, and the combined score
    clears a bar.

  Gate 2 — Claude judgement
    Only gate-1 survivors are sent to Claude with the full picture (signals,
    indicators, S/R, gap, market overview, news, earnings). Claude returns a
    0–100 confidence and rejects anything that is not a clean setup. Only
    confidence >= CONFIDENCE_THRESHOLD is published.

  If nothing passes, nothing is sent. Silence = no opportunity.

Accepted recommendations are stored (agent.memory.recommendations) and
tracked for 14 calendar days by agent.recommender_tracking.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── tunables (env-overridable) ───────────────────────────────
CONFIDENCE_THRESHOLD = int(os.environ.get("RECOMMENDER_CONFIDENCE", "75"))
HOLD_DAYS = int(os.environ.get("RECOMMENDER_HOLD_DAYS", "14"))
MIN_FAMILIES = 2
MIN_DISTINCT_SIGNALS = 3
MIN_VOLUME_RATIO = 1.3
MIN_PRIMARY_STRENGTH = 0.7
PRIMARY_FAMILIES = {"breakout", "reversal", "trend"}
GATE1_MIN_SCORE = 62.0
MAX_STOP_PCT = 0.09  # cap single-position risk
REDUNDANCY_HOURS = 72          # don't re-recommend the same symbol within this window
FETCH_PACING_SECONDS = float(os.environ.get("RECOMMENDER_PACING", "8.5"))  # TD free ≈ 8/min
JUDGE_MODEL = os.environ.get("RECOMMENDER_MODEL", "claude-sonnet-5")

_JUDGE_SYSTEM = """אתה השופט הסופי של "הממליץ" — מנגנון שמזהה סטאפים לונג במניות של מעוז.
לפניך מועמד יחיד שכבר עבר סינון דטרמיניסטי (כמה סיגנלים עצמאיים התיישרו + אישור נפח).
התפקיד שלך: להגיד אם זה סטאפ *נקי מאוד* או לא.

עברית בלבד, יבש ומדויק. בלי הקדמות.

פסול (confidence נמוך, verdict=reject) אם מתקיים ולו אחד מאלה:
- נר הפריצה/הכניסה חלש (גוף קטן, פתיל עליון ארוך, נסגר בתחתית הטווח)
- דוח רווחים צפוי בתוך 10 ימי מסחר
- מצב השוק הכללי שלילי (SPY יורד בצורה משמעותית / VIX גבוה ועולה)
- הגאפ מסווג exhaustion, או שהכניסה נשענת על גאפ שכבר נסגר
- יחס סיכוי/סיכון קטן מ-1.8
- הסיגנלים סותרים זה את זה, או שהמבנה לא ברור

אתה רשאי לכוונן entry/target/stop סביב ההצעה שקיבלת, אבל השאר אותם ריאליים
ביחס לרמות התמיכה/התנגדות והמחיר.

החזר JSON תקין אחד בלבד, בלי טקסט מסביב:
{"confidence": <0-100>, "verdict": "accept"|"reject",
 "entry": <number>, "target": <number>, "stop": <number>,
 "reasoning": "<2-3 משפטים>", "risks": "<אזהרה קצרה או מחרוזת ריקה>"}

אל תמציא נתונים שלא סופקו."""

# ── output sink (bot.py wires this to the #הממליץ channel) ────
_sink = None


def set_sink(fn) -> None:
    global _sink
    _sink = fn


def _emit(message: str) -> None:
    if _sink:
        try:
            _sink(message)
            return
        except Exception as exc:
            log.error("recommender sink failed: %s", exc)
    log.warning("recommender: no sink — message dropped:\n%s", message)


# ─────────────────────────────────────────────────────────────
#  Context
# ─────────────────────────────────────────────────────────────

@dataclass
class Signal:
    name: str
    family: str          # structure | momentum | trend | volatility
    strength: float      # 0..1
    detail: str


@dataclass
class Ctx:
    symbol: str
    df: pd.DataFrame
    price: float
    prev_close: float
    rsi: float | None
    rsi_prev: float | None
    macd_hist: float | None
    macd_hist_prev: float | None
    ma20: float | None
    ma50: float | None
    ma50_prev: float | None
    ma200: float | None
    bb_upper: float | None
    bb_lower: float | None
    bb_width_now: float | None
    bb_width_min: float | None
    atr: float | None
    vol_ratio: float
    supports: list[float]
    resistances: list[float]
    gap: dict


def _build_ctx(symbol: str) -> Ctx | None:
    from analysis import (
        _atr,
        _bollinger,
        _fetch_daily,
        _find_sr,
        _macd,
        _rsi,
        _sma,
    )
    from agent.market_data import analyze_gap

    try:
        df = _fetch_daily(symbol, outputsize=260, force=True)
    except Exception as exc:
        log.warning("recommender: fetch %s failed: %s", symbol, exc)
        return None
    if df is None or df.empty or len(df) < 40:
        return None

    closes = df["Close"]
    price = float(closes.iloc[-1])
    prev_close = float(closes.iloc[-2])

    rsi = _rsi(closes) if len(closes) >= 15 else None
    rsi_prev = _rsi(closes.iloc[:-1]) if len(closes) >= 16 else None

    if len(closes) >= 35:
        _m, _s, hist, prev_hist = _macd(closes)
    else:
        hist = prev_hist = None

    ma20, _ = _sma(closes, 20)
    ma50, ma50_prev = _sma(closes, 50)
    ma200, _ = _sma(closes, 200)
    bb_u, _bmid, bb_l = _bollinger(closes)

    mid = closes.rolling(20).mean()
    sd = closes.rolling(20).std()
    width = (4 * sd) / mid
    bb_width_now = float(width.iloc[-1]) if not pd.isna(width.iloc[-1]) else None
    tail = width.iloc[-60:].dropna()
    bb_width_min = float(tail.min()) if not tail.empty else None

    atr = _atr(df)
    vols = df["Volume"].to_numpy(dtype=float)
    avg_vol = float(np.mean(vols[-21:-1])) if len(vols) >= 21 else float(np.mean(vols[:-1]) or 1.0)
    vol_ratio = round(vols[-1] / avg_vol, 2) if avg_vol else 1.0

    supports, resistances = _find_sr(df)

    try:
        gap = analyze_gap(symbol=symbol, df=df)
    except Exception as exc:
        log.debug("analyze_gap %s failed: %s", symbol, exc)
        gap = {"error": str(exc)}

    return Ctx(
        symbol=symbol,
        df=df,
        price=price,
        prev_close=prev_close,
        rsi=round(rsi, 2) if rsi is not None else None,
        rsi_prev=round(rsi_prev, 2) if rsi_prev is not None else None,
        macd_hist=round(hist, 4) if hist is not None else None,
        macd_hist_prev=round(prev_hist, 4) if prev_hist is not None else None,
        ma20=round(ma20, 2) if ma20 is not None else None,
        ma50=round(ma50, 2) if ma50 is not None else None,
        ma50_prev=round(ma50_prev, 2) if ma50_prev is not None else None,
        ma200=round(ma200, 2) if ma200 is not None else None,
        bb_upper=round(bb_u, 2) if bb_u is not None else None,
        bb_lower=round(bb_l, 2) if bb_l is not None else None,
        bb_width_now=round(bb_width_now, 4) if bb_width_now is not None else None,
        bb_width_min=round(bb_width_min, 4) if bb_width_min is not None else None,
        atr=round(atr, 3) if atr is not None else None,
        vol_ratio=vol_ratio,
        supports=[round(s, 2) for s in supports[:4]],
        resistances=[round(r, 2) for r in resistances[:4]],
        gap=gap,
    )


# ─────────────────────────────────────────────────────────────
#  Detectors  (each returns a list[Signal], long side only)
# ─────────────────────────────────────────────────────────────

def _d_classic_patterns(ctx: Ctx) -> list[Signal]:
    from analysis import _patterns

    sup = next((s for s in ctx.supports if s < ctx.price), None)
    res = next((r for r in ctx.resistances if r > ctx.price), None)
    out: list[Signal] = []
    try:
        alerts = _patterns(ctx.symbol, ctx.df, ctx.price, sup, res)
    except Exception as exc:
        log.debug("_patterns %s failed: %s", ctx.symbol, exc)
        return out
    bullish_markers = ("🟢", "🚀", "▲", "☕", "Bottom", "Breakout", "Ascending", "Cup")
    seen: set = set()
    for a in alerts:
        title = a.title.strip()
        if not any(m in title for m in bullish_markers) or title in seen:
            continue
        seen.add(title)
        if any(m in title for m in ("Bottom", "☕", "Cup")):
            family, strength = "reversal", 0.72
        elif any(m in title for m in ("🚀", "Breakout", "Ascending", "▲")):
            family, strength = "breakout", 0.78
        else:
            family, strength = "reversal", 0.68
        out.append(Signal(name=title, family=family, strength=strength, detail=a.description))
    return out


def _d_downtrend_break(ctx: Ctx) -> list[Signal]:
    h = ctx.df["High"].to_numpy(dtype=float)
    c = ctx.df["Close"].to_numpy(dtype=float)
    if len(c) < 30:
        return []
    win = 30
    seg = h[-win:]
    idx = [i for i in range(2, win - 2) if seg[i] == max(seg[i - 2 : i + 3])]
    if len(idx) < 2:
        return []
    x = np.array(idx, dtype=float)
    y = seg[idx]
    slope, intercept = np.polyfit(x, y, 1)
    if slope >= 0:
        return []
    line_now = slope * (win - 1) + intercept
    line_prev = slope * (win - 2) + intercept
    if c[-1] > line_now and c[-2] <= line_prev * 1.001:
        return [
            Signal(
                name="פריצת קו־מגמה יורד",
                family="breakout",
                strength=0.75,
                detail=f"נסגר מעל קו השיאים היורדים (~{line_now:.2f})",
            )
        ]
    return []


def _d_52w_high(ctx: Ctx) -> list[Signal]:
    h = ctx.df["High"].to_numpy(dtype=float)
    if len(h) < 60:
        return []
    window = h[-252:] if len(h) >= 252 else h
    hi = float(np.max(window[:-1]))
    if ctx.price >= hi * 0.998 and ctx.price > hi:
        label = "פריצת שיא 52 שבועות" if len(h) >= 252 else "פריצת שיא כל־הזמנים בטווח הנתונים"
        return [
            Signal(
                name=label,
                family="breakout",
                strength=0.76,
                detail=f"מחיר {ctx.price:.2f} מעל שיא קודם {hi:.2f}",
            )
        ]
    return []


def _d_support_reclaim(ctx: Ctx) -> list[Signal]:
    o = ctx.df["Open"].to_numpy(dtype=float)
    h = ctx.df["High"].to_numpy(dtype=float)
    l = ctx.df["Low"].to_numpy(dtype=float)
    c = ctx.df["Close"].to_numpy(dtype=float)
    if len(c) < 5:
        return []
    sup = next((s for s in ctx.supports if s <= ctx.price * 1.001), None)
    if sup is None or (ctx.price - sup) / ctx.price > 0.025:
        return []
    rng = h[-1] - l[-1]
    bullish_candle = c[-1] > o[-1] and rng > 0 and (c[-1] - l[-1]) / rng > 0.6
    prev_red = c[-2] < o[-2]
    rsi_ok = ctx.rsi is not None and ctx.rsi_prev is not None and 30 <= ctx.rsi <= 55 and ctx.rsi > ctx.rsi_prev
    if bullish_candle and prev_red and rsi_ok:
        return [
            Signal(
                name="נגיעת תמיכה + ריבאונד",
                family="reversal",
                strength=0.72,
                detail=f"ריבאונד מתמיכה ~{sup:.2f} בנר שורי",
            )
        ]
    return []


def _d_ma_pullback(ctx: Ctx) -> list[Signal]:
    if not (ctx.ma50 and ctx.ma200 and ctx.ma50 > ctx.ma200):
        return []
    o = ctx.df["Open"].to_numpy(dtype=float)
    l = ctx.df["Low"].to_numpy(dtype=float)
    c = ctx.df["Close"].to_numpy(dtype=float)
    if l[-1] <= ctx.ma50 * 1.02 and c[-1] > ctx.ma50 and c[-1] > o[-1]:
        return [
            Signal(
                name="פולבק לממוצע נע 50 עולה",
                family="trend",
                strength=0.7,
                detail=f"מגע ב-MA50 ({ctx.ma50:.2f}) וריבאונד, MA50>MA200",
            )
        ]
    return []


def _d_golden_cross(ctx: Ctx) -> list[Signal]:
    if not (ctx.ma50 and ctx.ma200 and ctx.ma50_prev):
        return []
    if ctx.ma50_prev <= ctx.ma200 < ctx.ma50:
        return [
            Signal(
                name="גולדן קרוס טרי",
                family="trend",
                strength=0.68,
                detail=f"MA50 ({ctx.ma50:.2f}) חצה מעל MA200 ({ctx.ma200:.2f})",
            )
        ]
    return []


def _d_rsi_reversal(ctx: Ctx) -> list[Signal]:
    if ctx.rsi is None or ctx.rsi_prev is None:
        return []
    crossed_up = ctx.rsi_prev < 32 <= ctx.rsi
    recovering = ctx.rsi < 45 and ctx.rsi - ctx.rsi_prev >= 3
    if crossed_up or recovering:
        return [
            Signal(
                name="RSI יוצא ממכירת יתר",
                family="momentum",
                strength=0.66 if crossed_up else 0.6,
                detail=f"RSI {ctx.rsi_prev:.1f} → {ctx.rsi:.1f}",
            )
        ]
    return []


def _d_macd_cross(ctx: Ctx) -> list[Signal]:
    if ctx.macd_hist is None or ctx.macd_hist_prev is None:
        return []
    if ctx.macd_hist_prev < 0 <= ctx.macd_hist:
        return [
            Signal(
                name="MACD חיתוך שורי",
                family="momentum",
                strength=0.64,
                detail=f"היסטוגרמה {ctx.macd_hist_prev:.3f} → {ctx.macd_hist:.3f}",
            )
        ]
    return []


def _d_bb_squeeze_break(ctx: Ctx) -> list[Signal]:
    if ctx.bb_width_now is None or ctx.bb_width_min is None:
        return []
    squeezed = ctx.bb_width_now <= ctx.bb_width_min * 1.15
    if not squeezed:
        return []
    breaking = ctx.bb_upper is not None and ctx.price >= ctx.bb_upper
    return [
        Signal(
            name="סקוויז בולינג'ר" + (" + פריצה" if breaking else ""),
            family="volatility",
            strength=0.72 if breaking else 0.58,
            detail=f"רוחב הבנד {ctx.bb_width_now:.3f} קרוב למינימום {ctx.bb_width_min:.3f}",
        )
    ]


def _d_volume_breakout(ctx: Ctx) -> list[Signal]:
    h = ctx.df["High"].to_numpy(dtype=float)
    c = ctx.df["Close"].to_numpy(dtype=float)
    if len(c) < 22:
        return []
    recent_high = float(np.max(h[-21:-1]))
    if c[-1] > recent_high and ctx.vol_ratio >= 1.5:
        return [
            Signal(
                name="פריצה עם נפח חריג",
                family="volume",
                strength=0.8,
                detail=f"סגירה מעל שיא 20 יום ({recent_high:.2f}) בנפח x{ctx.vol_ratio}",
            )
        ]
    return []


def _d_gap_breakaway(ctx: Ctx) -> list[Signal]:
    g = ctx.gap or {}
    if g.get("direction") == "up" and g.get("gap_type") in ("breakaway", "runaway") and not g.get("filled_same_day"):
        return [
            Signal(
                name="גאפ פריצה שהחזיק",
                family="breakout",
                strength=0.74 if g.get("gap_and_go") else 0.66,
                detail=g.get("note", ""),
            )
        ]
    return []


DETECTORS = [
    _d_classic_patterns,
    _d_downtrend_break,
    _d_52w_high,
    _d_support_reclaim,
    _d_ma_pullback,
    _d_golden_cross,
    _d_rsi_reversal,
    _d_macd_cross,
    _d_bb_squeeze_break,
    _d_volume_breakout,
    _d_gap_breakaway,
]


# ─────────────────────────────────────────────────────────────
#  Scoring / gate 1
# ─────────────────────────────────────────────────────────────

def _dedup(signals: list[Signal]) -> list[Signal]:
    best: dict[str, Signal] = {}
    for s in signals:
        cur = best.get(s.name)
        if cur is None or s.strength > cur.strength:
            best[s.name] = s
    return list(best.values())


def _has_primary(signals: list[Signal]) -> bool:
    return any(
        s.family in PRIMARY_FAMILIES and s.strength >= MIN_PRIMARY_STRENGTH for s in signals
    )


def score_symbol(ctx: Ctx) -> tuple[float, list[Signal]]:
    raw: list[Signal] = []
    for det in DETECTORS:
        try:
            raw.extend(det(ctx) or [])
        except Exception as exc:
            log.debug("detector %s on %s failed: %s", det.__name__, ctx.symbol, exc)
    signals = _dedup(raw)
    if not signals:
        return 0.0, []

    families = {s.family for s in signals}
    avg_strength = sum(s.strength for s in signals) / len(signals)

    base = 18 * len(families) + 8 * len(signals) + 24 * avg_strength
    if ctx.vol_ratio >= MIN_VOLUME_RATIO:
        base += 10
    if ctx.vol_ratio >= 2.0:
        base += 6
    if _has_primary(signals):
        base += 8
    if ctx.gap.get("gap_type") == "exhaustion":
        base -= 25
    return round(max(0.0, min(100.0, base)), 1), signals


def passes_gate1(ctx: Ctx, score: float, signals: list[Signal]) -> bool:
    families = {s.family for s in signals}
    if not (
        len(families) >= MIN_FAMILIES
        and len(signals) >= MIN_DISTINCT_SIGNALS
        and _has_primary(signals)
        and score >= GATE1_MIN_SCORE
        and ctx.gap.get("gap_type") != "exhaustion"
    ):
        return False
    # Volume is a scoring factor, not a blanket gate — a setup backed by several
    # independent families is real even on a quiet tape. But a setup whose ONLY
    # primary family is "breakout" needs volume behind it, or it's a fakeout risk.
    primary_fams = {s.family for s in signals if s.family in PRIMARY_FAMILIES}
    if primary_fams == {"breakout"} and ctx.vol_ratio < MIN_VOLUME_RATIO:
        return False
    return True


def propose_levels(ctx: Ctx) -> tuple[float, float, float, float | None]:
    entry = ctx.price
    atr = ctx.atr or (ctx.price * 0.03)
    sup_below = max([s for s in ctx.supports if s < entry], default=entry - 2 * atr)
    stop = min(sup_below - 0.3 * atr, entry - 1.5 * atr)
    stop = round(max(stop, entry * (1 - MAX_STOP_PCT)), 2)  # cap risk
    res_above = min([r for r in ctx.resistances if r > entry * 1.01], default=entry + 3 * atr)
    target = round(max(res_above, entry + 2.0 * atr), 2)
    risk = entry - stop
    if risk > 0 and (target - entry) / risk < 1.8:
        target = round(entry + 1.8 * risk, 2)
    rr = round((target - entry) / risk, 2) if risk > 0 else None
    return round(entry, 2), target, stop, rr


# ─────────────────────────────────────────────────────────────
#  Gate 2 — Claude
# ─────────────────────────────────────────────────────────────

def _gather_extra(symbol: str) -> dict:
    extra: dict = {}
    try:
        from agent.tools import get_market_overview

        extra["market"] = get_market_overview()
    except Exception as exc:
        log.debug("market overview failed: %s", exc)
    try:
        from agent.tools import check_earnings

        extra["earnings"] = check_earnings(symbol)
    except Exception as exc:
        log.debug("earnings check failed: %s", exc)
    try:
        from agent.tools import fetch_news

        extra["news"] = fetch_news(query=symbol, since_hours=48)[:4]
    except Exception as exc:
        log.debug("news fetch failed: %s", exc)
    return extra


def _parse_judge_json(raw: str) -> dict:
    if not raw:
        return {}
    txt = raw.strip()
    if "```" in txt:
        parts = txt.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                txt = p
                break
    start = txt.find("{")
    end = txt.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(txt[start : end + 1])
    except Exception as exc:
        log.warning("judge JSON parse failed: %s | raw=%.200s", exc, raw)
        return {}


def _judge(ctx: Ctx, score: float, signals: list[Signal], levels: tuple) -> dict:
    from agent.core import run_agent_single_shot

    entry, target, stop, rr = levels
    payload = {
        "symbol": ctx.symbol,
        "price": ctx.price,
        "prev_close": ctx.prev_close,
        "deterministic_score": score,
        "signals": [asdict(s) for s in signals],
        "indicators": {
            "rsi": ctx.rsi,
            "macd_hist": ctx.macd_hist,
            "ma20": ctx.ma20,
            "ma50": ctx.ma50,
            "ma200": ctx.ma200,
            "bb_upper": ctx.bb_upper,
            "bb_lower": ctx.bb_lower,
            "atr": ctx.atr,
            "volume_ratio": ctx.vol_ratio,
        },
        "levels": {"supports": ctx.supports, "resistances": ctx.resistances},
        "gap": ctx.gap,
        "proposed": {"entry": entry, "target": target, "stop": stop, "rr": rr},
        "last_5_bars": [
            {
                "date": str(idx.date()),
                "o": round(float(r.Open), 2),
                "h": round(float(r.High), 2),
                "l": round(float(r.Low), 2),
                "c": round(float(r.Close), 2),
                "v": int(r.Volume),
            }
            for idx, r in ctx.df.tail(5).iterrows()
        ],
        "context": _gather_extra(ctx.symbol),
    }
    instruction = (
        "לפניך מועמד שעבר את הסינון הדטרמיניסטי של הממליץ. שפוט אותו לפי הכללים "
        "במערכת והחזר JSON תקין אחד בלבד."
    )

    raw = run_agent_single_shot(
        payload, instruction, max_tokens=700, system=_JUDGE_SYSTEM, model=JUDGE_MODEL
    )
    parsed = _parse_judge_json(raw)
    if "confidence" not in parsed:
        log.warning(
            "הממליץ judge: primary model %s gave no verdict (raw=%.300s) — retrying fallback",
            JUDGE_MODEL, raw,
        )
        raw = run_agent_single_shot(
            payload, instruction, max_tokens=700, system=_JUDGE_SYSTEM
        )
        parsed = _parse_judge_json(raw)
        if "confidence" not in parsed:
            log.error("הממליץ judge: fallback model also failed (raw=%.300s)", raw)
    return parsed


def _snapshot(ctx: Ctx, signals: list[Signal], judge: dict) -> dict:
    return {
        "price": ctx.price,
        "rsi": ctx.rsi,
        "macd_hist": ctx.macd_hist,
        "ma50": ctx.ma50,
        "ma200": ctx.ma200,
        "atr": ctx.atr,
        "volume_ratio": ctx.vol_ratio,
        "supports": ctx.supports,
        "resistances": ctx.resistances,
        "gap": ctx.gap,
        "signals": [asdict(s) for s in signals],
        "judge": judge,
        "taken_at_utc": datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────

def run_recommender() -> list[dict]:
    """Full scan. Returns the list of accepted+stored recommendations (may be empty)."""
    from analysis import WATCHLIST
    from agent.memory import init_memory_db, save_recommendation, was_symbol_recommended_recently

    init_memory_db()
    symbols = list(WATCHLIST)
    log.info("הממליץ: scanning %d symbols", len(symbols))

    candidates: list[tuple[Ctx, float, list[Signal]]] = []
    for i, sym in enumerate(symbols):
        ctx = _build_ctx(sym)
        if ctx is not None:
            score, signals = score_symbol(ctx)
            if passes_gate1(ctx, score, signals):
                candidates.append((ctx, score, signals))
                log.info("הממליץ gate1 ✓ %s score=%.1f families=%d", sym, score,
                         len({s.family for s in signals}))
        if i < len(symbols) - 1:
            time.sleep(FETCH_PACING_SECONDS)

    log.info("הממליץ: %d/%d passed gate 1", len(candidates), len(symbols))
    if candidates:
        time.sleep(FETCH_PACING_SECONDS)

    accepted: list[dict] = []
    for ctx, score, signals in candidates:
        if was_symbol_recommended_recently(ctx.symbol, hours=REDUNDANCY_HOURS):
            log.info("הממליץ: skipping %s — recommended within %dh", ctx.symbol, REDUNDANCY_HOURS)
            continue
        levels = propose_levels(ctx)
        judge = _judge(ctx, score, signals, levels)
        conf = judge.get("confidence")
        verdict = judge.get("verdict")
        if not isinstance(conf, (int, float)):
            log.warning("הממליץ: %s — no usable judge verdict, skipping", ctx.symbol)
            continue
        if verdict != "accept" or conf < CONFIDENCE_THRESHOLD:
            log.info("הממליץ gate2 ✗ %s conf=%s verdict=%s", ctx.symbol, conf, verdict)
            continue

        entry = float(judge.get("entry") or levels[0])
        target = float(judge.get("target") or levels[1])
        stop = float(judge.get("stop") or levels[2])
        risk = entry - stop
        rr = round((target - entry) / risk, 2) if risk > 0 else None

        rec = {
            "symbol": ctx.symbol,
            "setup": " · ".join(dict.fromkeys(s.name for s in signals))[:240],
            "direction": "long",
            "confidence": round(float(conf), 1),
            "det_score": score,
            "entry": round(entry, 2),
            "target": round(target, 2),
            "stop": round(stop, 2),
            "rr": rr,
            "reasoning": (judge.get("reasoning") or "").strip(),
            "risks": (judge.get("risks") or "").strip(),
            "snapshot": json.dumps(_snapshot(ctx, signals, judge), ensure_ascii=False, default=str),
        }
        rec["id"] = save_recommendation(rec)
        accepted.append(rec)
        log.info("הממליץ gate2 ✓ %s conf=%.1f id=%s", ctx.symbol, conf, rec["id"])

    if accepted:
        _emit(format_batch(accepted))
    else:
        log.info("הממליץ: nothing cleared gate 2 — staying silent")
    return accepted


# ─────────────────────────────────────────────────────────────
#  Formatting
# ─────────────────────────────────────────────────────────────

def _fmt_rec(rec: dict) -> str:
    risk = rec["entry"] - rec["stop"]
    reward = rec["target"] - rec["entry"]
    lines = [
        f"📈 **{rec['symbol']}** — המלצת לונג",
        f"🧩 סטאפ: {rec['setup']}",
        f"🎯 כניסה ${rec['entry']} · יעד ${rec['target']} (+{reward / rec['entry'] * 100:.1f}%) "
        f"· סטופ ${rec['stop']} (−{risk / rec['entry'] * 100:.1f}%)",
        f"⚖️ סיכוי/סיכון: {rec['rr']}  |  ביטחון: {rec['confidence']}/100  (ציון סורק {rec['det_score']})",
        f"💬 {rec['reasoning']}",
    ]
    if rec.get("risks"):
        lines.append(f"⚠️ {rec['risks']}")
    lines.append(f"⏳ מעקב ל-{HOLD_DAYS} ימים · מזהה #{rec.get('id', '?')}")
    return "\n".join(lines)


def format_batch(recs: list[dict]) -> str:
    today = datetime.now(timezone.utc).strftime("%d/%m/%Y")
    head = f"🟢 **הממליץ — {today}**\n{len(recs)} סטאפ(ים) עברו את שני הסינונים:\n"
    return head + "\n\n".join(_fmt_rec(r) for r in recs)


def format_status() -> str:
    from agent.memory import get_open_recommendations, get_recommendation_stats

    open_recs = get_open_recommendations()
    st = get_recommendation_stats()

    lines = ["📋 **הממליץ — סטטוס**\n"]
    if open_recs:
        lines.append(f"**פתוחות ({len(open_recs)}):**")
        for r in open_recs:
            lines.append(
                f"• **{r['symbol']}** — כניסה ${r['entry']} | יעד ${r['target']} | סטופ ${r['stop']} "
                f"| ביטחון {r['confidence']} | נפתח {str(r['opened_at'])[:10]}"
            )
    else:
        lines.append("אין המלצות פתוחות כרגע.")

    lines.append("")
    lines.append("**סקורבורד (כל הזמנים):**")
    lines.append(
        f"• נסגרו: {st['closed']} · הצליחו: {st['wins']} · הפסידו: {st['losses']}"
    )
    if st["closed"]:
        lines.append(f"• אחוז הצלחה: {st['win_rate']}% · תשואה ממוצעת: {st['avg_pnl']}%")
        if st.get("by_status"):
            brk = ", ".join(f"{k}: {v}" for k, v in st["by_status"].items())
            lines.append(f"• פילוח סגירה: {brk}")
        if st.get("best"):
            lines.append(f"• הכי טובה: {st['best']['symbol']} {st['best']['pnl_pct']:+.1f}%")
        if st.get("worst"):
            lines.append(f"• הכי גרועה: {st['worst']['symbol']} {st['worst']['pnl_pct']:+.1f}%")
    return "\n".join(lines)
