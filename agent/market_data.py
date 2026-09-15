"""
Fresh-price + gap analysis layer.

Sits on top of analysis.py's Twelve Data helpers. Two jobs:

1. get_quote(symbol)  — real-time-ish last price via Twelve Data /quote,
   with only a 60-second micro-cache, so "what's the price now?" never
   returns a stale daily-bar close.
2. analyze_gap(...)   — overnight gap size/type, whether it filled, and any
   still-open gap zones acting as support/resistance. The interactive agent
   and "הממליץ" both under-weighted gaps before this existed.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd
import requests

from analysis import _api_symbol, _atr, _fetch_daily, TWELVE_DATA_KEY

log = logging.getLogger(__name__)

_QUOTE_URL = "https://api.twelvedata.com/quote"
_QUOTE_TTL = 60  # seconds
_quote_cache: dict[str, tuple[float, dict]] = {}


def _f(v) -> float | None:
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return None


def _api_key() -> str | None:
    return (
        TWELVE_DATA_KEY
        or os.environ.get("TWELVE_DATA_KEY")
        or os.environ.get("TWELVEDATAKEY")
        or os.environ.get("TWELVE_DATA_API_KEY")
    )


# ─────────────────────────────────────────────────────────────
#  Real-time quote
# ─────────────────────────────────────────────────────────────

def get_quote(symbol: str) -> dict:
    """
    Latest price for *symbol* with a 60-second micro-cache.
    Use this — not analyze_stock — whenever the question is about the
    *current* price, because analyze_stock reads the last (possibly still
    forming) daily bar and is cached for up to 55 minutes.
    """
    sym = symbol.upper().strip()
    now = time.monotonic()
    hit = _quote_cache.get(sym)
    if hit and now - hit[0] < _QUOTE_TTL:
        return hit[1]

    key = _api_key()
    if not key:
        return {"symbol": sym, "error": "TWELVE_DATA_KEY not set"}

    try:
        resp = requests.get(
            _QUOTE_URL,
            params={"symbol": _api_symbol(sym), "apikey": key},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") or data.get("status") == "error":
            return {"symbol": sym, "error": data.get("message", "quote error")}

        price = _f(data.get("close")) or _f(data.get("price"))
        prev_close = _f(data.get("previous_close"))
        change_pct = None
        if price is not None and prev_close:
            change_pct = round((price - prev_close) / prev_close * 100, 2)

        out = {
            "symbol": sym,
            "price": price,
            "open": _f(data.get("open")),
            "day_high": _f(data.get("high")),
            "day_low": _f(data.get("low")),
            "prev_close": prev_close,
            "change_pct": change_pct,
            "volume": _f(data.get("volume")),
            "is_market_open": data.get("is_market_open")
            if isinstance(data.get("is_market_open"), bool)
            else None,
            "as_of": data.get("datetime") or data.get("timestamp"),
        }
        _quote_cache[sym] = (now, out)
        return out
    except Exception as exc:
        log.warning("get_quote %s failed (%s) — trying Yahoo", sym, exc)
        return _quote_from_yahoo(sym) or {"symbol": sym, "error": str(exc)}


def _quote_from_yahoo(sym: str) -> dict | None:
    try:
        from analysis import _fetch_daily_yahoo

        df = _fetch_daily_yahoo(sym, 5)
        if df is None or df.empty:
            return None
        last = df.iloc[-1]
        prev = float(df["Close"].iloc[-2]) if len(df) > 1 else None
        price = round(float(last["Close"]), 4)
        return {
            "symbol": sym,
            "price": price,
            "open": round(float(last["Open"]), 4),
            "day_high": round(float(last["High"]), 4),
            "day_low": round(float(last["Low"]), 4),
            "prev_close": prev,
            "change_pct": round((price - prev) / prev * 100, 2) if prev else None,
            "volume": round(float(last["Volume"]), 0),
            "is_market_open": None,
            "as_of": str(df.index[-1].date()),
            "source": "yahoo (EOD fallback)",
        }
    except Exception as exc:
        log.warning("_quote_from_yahoo %s failed: %s", sym, exc)
        return None


# ─────────────────────────────────────────────────────────────
#  Gap analysis
# ─────────────────────────────────────────────────────────────

def _classify_gap(
    direction: str,
    gap_pct_abs: float,
    gap_atr_abs: float,
    run_20: float,
    vol_ratio: float,
    pre_range: float,
    filled: bool,
) -> str:
    if direction == "flat" or gap_pct_abs < 0.5:
        return "none"
    if gap_atr_abs < 0.5 or filled:
        return "common"
    if direction == "up" and run_20 > 25 and vol_ratio > 1.8:
        return "exhaustion"
    if direction == "down" and run_20 < -25 and vol_ratio > 1.8:
        return "exhaustion"
    if pre_range < 0.12 and vol_ratio > 1.3:
        return "breakaway"
    if (direction == "up" and run_20 > 3) or (direction == "down" and run_20 < -3):
        return "runaway"
    return "common"


def _unfilled_gap_zones(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    atr: float | None,
    lookback: int,
    current: float,
) -> list[dict]:
    zones: list[dict] = []
    atr = atr or (current * 0.02)
    n = len(c)
    start = max(1, n - lookback)
    for i in range(start, n):
        gap = o[i] - c[i - 1]
        if abs(gap) < 0.4 * atr:
            continue
        lo_z, hi_z = (c[i - 1], o[i]) if gap > 0 else (o[i], c[i - 1])
        mid = (lo_z + hi_z) / 2
        if any(l[j] <= mid <= h[j] for j in range(i + 1, n)):
            continue  # already filled
        zones.append(
            {
                "low": round(float(lo_z), 2),
                "high": round(float(hi_z), 2),
                "direction": "up" if gap > 0 else "down",
                "role": "support"
                if hi_z < current
                else "resistance"
                if lo_z > current
                else "inside",
                "age_bars": n - 1 - i,
            }
        )
    return zones[-4:]


def _gap_note(direction: str, gtype: str, filled: bool, held: bool) -> str:
    if direction == "flat":
        return "אין גאפ משמעותי"
    d = "למעלה" if direction == "up" else "למטה"
    label = {
        "breakaway": "פריצה (breakaway)",
        "runaway": "המשך מגמה (runaway)",
        "exhaustion": "חשד לתשישות (exhaustion)",
        "common": "רגיל",
        "none": "זניח",
    }.get(gtype, gtype)
    parts = [f"גאפ {d}", label, "נסגר באותו יום" if filled else "לא נסגר"]
    if held and not filled:
        parts.append("החזיק את הכיוון")
    return " · ".join(parts)


def analyze_gap(
    symbol: str | None = None,
    df: pd.DataFrame | None = None,
    lookback: int = 25,
) -> dict:
    """
    Gap picture for the most recent bar of *symbol* (or the supplied df).

    Returns overnight gap %, size in ATRs, a heuristic type
    (breakaway / runaway / exhaustion / common / none), whether it filled
    the same day, whether the close held the gap direction, and any
    still-open gap zones that now act as support/resistance.
    """
    if df is None:
        if not symbol:
            return {"error": "need symbol or df"}
        df = _fetch_daily(symbol, outputsize=max(60, lookback + 5), force=True)
    if df is None or len(df) < 5:
        return {"error": "insufficient data"}

    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    v = df["Volume"].to_numpy(dtype=float)

    atr = _atr(df)
    if not atr:
        atr = float(np.mean(h[-14:] - l[-14:])) or (c[-1] * 0.02)

    prev_close = c[-2]
    today_open = o[-1]
    gap_abs = today_open - prev_close
    gap_pct = (gap_abs / prev_close * 100) if prev_close else 0.0
    gap_atr = (gap_abs / atr) if atr else 0.0

    direction = "up" if gap_pct > 0.5 else "down" if gap_pct < -0.5 else "flat"
    if direction == "up":
        filled = l[-1] <= prev_close
        held = c[-1] >= today_open
    elif direction == "down":
        filled = h[-1] >= prev_close
        held = c[-1] <= today_open
    else:
        filled, held = True, False

    run_20 = ((c[-1] - c[-21]) / c[-21] * 100) if len(c) > 21 else 0.0
    pre_gap_vol = float(np.mean(v[-11:-1])) if len(v) > 11 else float(np.mean(v[:-1]) or 1.0)
    vol_ratio = (v[-1] / pre_gap_vol) if pre_gap_vol else 1.0
    pre_range = (
        (float(np.max(h[-11:-1])) - float(np.min(l[-11:-1]))) / c[-2]
        if len(h) > 11 and c[-2]
        else 1.0
    )

    gtype = _classify_gap(
        direction, abs(gap_pct), abs(gap_atr), run_20, vol_ratio, pre_range, filled
    )
    gap_and_go = bool(
        direction == "up"
        and held
        and not filled
        and c[-1] > prev_close + 0.5 * gap_abs
    )

    return {
        "symbol": symbol,
        "overnight_gap_pct": round(gap_pct, 2),
        "gap_atr": round(gap_atr, 2),
        "direction": direction,
        "gap_type": gtype,
        "filled_same_day": bool(filled),
        "held_direction": bool(held),
        "gap_and_go": gap_and_go,
        "volume_ratio_on_gap": round(vol_ratio, 2),
        "prior_20d_move_pct": round(run_20, 1),
        "unfilled_gap_zones": _unfilled_gap_zones(o, h, l, c, atr, lookback, float(c[-1])),
        "note": _gap_note(direction, gtype, filled, held),
    }
