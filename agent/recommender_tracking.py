"""
Tracking + verdict engine for "הממליץ".

Runs daily (Mon–Sat 08:00 Israel). For every open recommendation:

  • pull fresh daily bars since it was opened
  • gap-aware fill check per bar:
      - bar opens >= target  → closed "target" at the open  (gap through target)
      - bar opens <= stop    → closed "stop"   at the open  (gap through stop)
      - else low <= stop     → closed "stop"   at the stop   (stop assumed first
        when both hit in the same bar — conservative)
      - else high >= target  → closed "target" at the target
  • track MFE / MAE the whole way
  • after HOLD_DAYS calendar days still open → closed "time" at the last close

Every close is reported to #הממליץ with BOTH readings the user asked for:
what was hit first, and the net entry→exit return. On Saturdays a weekly
scoreboard is appended.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from agent.recommender import HOLD_DAYS, _emit

log = logging.getLogger(__name__)


def _parse_dt(value) -> datetime:
    dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def evaluate_open(now: datetime | None = None) -> list[dict]:
    from analysis import _fetch_daily
    from agent.memory import (
        close_recommendation,
        get_open_recommendations,
        update_recommendation_excursions,
    )

    now = now or datetime.now(timezone.utc)
    closed: list[dict] = []

    for rec in get_open_recommendations():
        opened = _parse_dt(rec["opened_at"])
        deadline = opened + timedelta(days=HOLD_DAYS)
        entry = float(rec["entry"])
        target = float(rec["target"])
        stop = float(rec["stop"])

        try:
            df = _fetch_daily(rec["symbol"], outputsize=60, force=True)
        except Exception as exc:
            log.warning("tracking: fetch %s failed: %s", rec["symbol"], exc)
            continue
        if df is None or df.empty:
            continue

        cutoff = pd.Timestamp(opened.date())
        bars = df[df.index > cutoff]
        if bars.empty:
            # nothing has traded since the rec was opened yet
            if now >= deadline:
                _close_on_time(rec, df, entry, closed, close_recommendation)
            continue

        mfe = float(rec.get("mfe_pct") or 0.0)
        mae = float(rec.get("mae_pct") or 0.0)
        hit = hit_price = hit_when = None

        for ts, bar in bars.iterrows():
            op, hi, lo = float(bar["Open"]), float(bar["High"]), float(bar["Low"])
            mfe = max(mfe, (hi - entry) / entry * 100)
            mae = min(mae, (lo - entry) / entry * 100)

            if op >= target:
                hit, hit_price, hit_when = "target", op, ts
                break
            if op <= stop:
                hit, hit_price, hit_when = "stop", op, ts
                break
            if lo <= stop:
                hit, hit_price, hit_when = "stop", stop, ts
                break
            if hi >= target:
                hit, hit_price, hit_when = "target", target, ts
                break

        if hit:
            pnl = (hit_price - entry) / entry * 100
            close_recommendation(
                rec["id"], hit, round(hit_price, 2), hit,
                round(pnl, 2), round(mfe, 2), round(mae, 2),
            )
            closed.append(
                {
                    **rec,
                    "status": hit,
                    "close_price": round(hit_price, 2),
                    "close_reason": hit,
                    "closed_on": str(pd.Timestamp(hit_when).date()),
                    "pnl_pct": round(pnl, 2),
                    "mfe_pct": round(mfe, 2),
                    "mae_pct": round(mae, 2),
                }
            )
        elif now >= deadline:
            last = float(df["Close"].iloc[-1])
            pnl = (last - entry) / entry * 100
            close_recommendation(
                rec["id"], "time", round(last, 2), "time",
                round(pnl, 2), round(mfe, 2), round(mae, 2),
            )
            closed.append(
                {
                    **rec,
                    "status": "time",
                    "close_price": round(last, 2),
                    "close_reason": "time",
                    "closed_on": str(df.index[-1].date()),
                    "pnl_pct": round(pnl, 2),
                    "mfe_pct": round(mfe, 2),
                    "mae_pct": round(mae, 2),
                }
            )
        else:
            update_recommendation_excursions(rec["id"], round(mfe, 2), round(mae, 2))

    return closed


def _close_on_time(rec, df, entry, closed, close_recommendation) -> None:
    last = float(df["Close"].iloc[-1])
    pnl = (last - entry) / entry * 100
    close_recommendation(rec["id"], "time", round(last, 2), "time", round(pnl, 2), 0.0, 0.0)
    closed.append(
        {
            **rec,
            "status": "time",
            "close_price": round(last, 2),
            "close_reason": "time",
            "closed_on": str(df.index[-1].date()),
            "pnl_pct": round(pnl, 2),
            "mfe_pct": 0.0,
            "mae_pct": 0.0,
        }
    )


# ─────────────────────────────────────────────────────────────
#  Reporting
# ─────────────────────────────────────────────────────────────

_VERDICT_ICON = {"target": "✅", "stop": "❌", "time": "⏳"}
_VERDICT_WORD = {"target": "היעד הושג", "stop": "נגע בסטופ", "time": "נסגר בזמן (14 יום)"}


def _fmt_verdict(rec: dict) -> str:
    icon = _VERDICT_ICON.get(rec["status"], "•")
    word = _VERDICT_WORD.get(rec["status"], rec["status"])
    pnl = rec["pnl_pct"]
    held_days = ""
    try:
        opened = _parse_dt(rec["opened_at"]).date()
        held_days = f" · {(datetime.fromisoformat(rec['closed_on']).date() - opened).days} ימים"
    except Exception:
        pass
    lines = [
        f"{icon} **{rec['symbol']}** — {word}",
        f"מה נגע קודם: **{word}**{held_days}",
        f"תשואה נטו כניסה→סגירה: **{pnl:+.2f}%**  (${rec['entry']} → ${rec['close_price']})",
        f"שיא רווח בדרך (MFE): {rec['mfe_pct']:+.1f}% · שיא הפסד (MAE): {rec['mae_pct']:+.1f}%",
        f"מזהה #{rec.get('id', '?')}",
    ]
    return "\n".join(lines)


def build_weekly_summary() -> str:
    from agent.memory import get_recommendation_stats

    st7 = get_recommendation_stats(since_days=7)
    st_all = get_recommendation_stats()
    lines = ["📊 **הממליץ — סיכום שבועי**\n"]
    lines.append(
        f"השבוע נסגרו {st7['closed']} המלצות · הצליחו {st7['wins']} · הפסידו {st7['losses']}"
    )
    if st7["closed"]:
        lines.append(f"תשואה ממוצעת השבוע: {st7['avg_pnl']:+.2f}% · אחוז הצלחה: {st7['win_rate']}%")
    lines.append("")
    lines.append(
        f"מצטבר: {st_all['closed']} נסגרו · אחוז הצלחה {st_all['win_rate']}% "
        f"· תשואה ממוצעת {st_all['avg_pnl']}% · {st_all['open']} פתוחות"
    )
    if st_all.get("best"):
        lines.append(f"הכי טובה עד כה: {st_all['best']['symbol']} {st_all['best']['pnl_pct']:+.1f}%")
    if st_all.get("worst"):
        lines.append(f"הכי גרועה עד כה: {st_all['worst']['symbol']} {st_all['worst']['pnl_pct']:+.1f}%")
    return "\n".join(lines)


def run_tracking(weekly: bool = False) -> list[dict]:
    from agent.memory import init_memory_db

    init_memory_db()
    closed = evaluate_open()
    for rec in closed:
        _emit(_fmt_verdict(rec))
    if weekly:
        _emit(build_weekly_summary())
    if not closed and not weekly:
        log.info("הממליץ tracking: no positions closed today")
    return closed
