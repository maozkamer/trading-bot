"""
Technical analysis engine for the swing-trading bot.
All public functions return Hebrew-friendly Alert objects or formatted strings.
Data source: Twelve Data API.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import requests

TWELVE_DATA_KEY = os.environ.get("TWELVE_DATA_KEY")
TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  Twelve Data helpers + in-process cache
# ─────────────────────────────────────────────────────────────

# Cache: symbol → (timestamp, DataFrame)
# TTL = 55 minutes — avoid re-fetching in the same hourly scan.
_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_CACHE_TTL   = 55 * 60    # seconds
_RETRY_COUNT = 3
_RETRY_WAIT  = 5          # seconds between retries


def _fetch_daily(symbol: str) -> pd.DataFrame:
    """
    Fetch 60 days of OHLCV for *symbol* via Twelve Data.
    Retries up to 3 times with a 5-second wait on failure.
    Caches results for 55 minutes.
    Returns a DataFrame with DatetimeIndex and columns:
      Open, High, Low, Close, Volume
    """
    if not TWELVE_DATA_KEY:
        raise RuntimeError("TWELVE_DATA_KEY environment variable is not set")

    now = time.monotonic()
    if symbol in _CACHE:
        ts, df = _CACHE[symbol]
        if now - ts < _CACHE_TTL:
            return df

    last_exc: Exception = RuntimeError("unknown error")
    for attempt in range(1, _RETRY_COUNT + 1):
        try:
            resp = requests.get(
                TWELVE_DATA_URL,
                params={
                    "symbol":     symbol,
                    "interval":   "1day",
                    "outputsize": 60,
                    "apikey":     TWELVE_DATA_KEY,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            if "code" in data:
                raise ValueError(f"Twelve Data error for {symbol}: {data.get('message', data)}")

            values = data.get("values")
            if not values:
                raise ValueError(f"Twelve Data returned no values for {symbol}")

            df = pd.DataFrame(values)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = (
                df.rename(columns={"datetime": "Date"})
                .set_index("Date")
                .sort_index()
                .assign(
                    Open=lambda x: x["open"].astype(float),
                    High=lambda x: x["high"].astype(float),
                    Low=lambda x: x["low"].astype(float),
                    Close=lambda x: x["close"].astype(float),
                    Volume=lambda x: x["volume"].astype(float),
                )
                [["Open", "High", "Low", "Close", "Volume"]]
            )

            _CACHE[symbol] = (time.monotonic(), df)
            return df

        except Exception as exc:
            last_exc = exc
            log.warning("_fetch_daily %s attempt %d/%d failed: %s",
                        symbol, attempt, _RETRY_COUNT, exc)
            if attempt < _RETRY_COUNT:
                time.sleep(_RETRY_WAIT)

    raise last_exc

WATCHLIST = [
    "NNE", "MARA", "PLTR", "IREN", "SOFI", "AAPL", "NVDA", "TSLA",
    "CIFR", "HOOD", "MSFT", "OKLO", "SMR", "RKLB", "COIN", "RIOT",
    "AMD", "META", "GOOGL", "AMZN",
    "BTC-USD", "ETH-USD",
]


@dataclass
class Alert:
    symbol: str
    title: str
    price: float
    support: Optional[float]
    resistance: Optional[float]
    description: str
    recommendation: str
    key: str
    cooldown_hours: int = 6


# ─────────────────────────────────────────────────────────────
#  Core indicators
# ─────────────────────────────────────────────────────────────

def _rsi(closes: pd.Series, period: int = 14) -> float:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def _macd(closes: pd.Series) -> tuple[float, float, float, float]:
    """Returns: macd, signal, histogram, prev_histogram"""
    ema12  = closes.ewm(span=12, adjust=False).mean()
    ema26  = closes.ewm(span=26, adjust=False).mean()
    line   = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    hist   = line - signal
    return float(line.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1]), float(hist.iloc[-2])


def _sma(closes: pd.Series, period: int) -> tuple[float, float] | tuple[None, None]:
    if len(closes) < period + 1:
        return None, None
    s = closes.rolling(period).mean()
    return float(s.iloc[-1]), float(s.iloc[-2])


def _bollinger(closes: pd.Series, period: int = 20, std_dev: int = 2):
    """Returns (upper, middle, lower) Bollinger Band values for the last bar."""
    middle = closes.rolling(period).mean()
    std    = closes.rolling(period).std()
    upper  = middle + std_dev * std
    lower  = middle - std_dev * std
    return float(upper.iloc[-1]), float(middle.iloc[-1]), float(lower.iloc[-1])


def _vwap(df: pd.DataFrame):
    """
    Compute VWAP on the last 20 bars (current) and the 20 bars ending one bar back (prev).
    Returns (current_vwap, prev_vwap).
    """
    def _calc(slice_df):
        tp  = (slice_df["High"] + slice_df["Low"] + slice_df["Close"]) / 3
        vol = slice_df["Volume"]
        total_vol = vol.sum()
        if total_vol == 0:
            return None
        return float((tp * vol).sum() / total_vol)

    curr_vwap = _calc(df.iloc[-20:])
    prev_vwap = _calc(df.iloc[-21:-1])
    return curr_vwap, prev_vwap


# ─────────────────────────────────────────────────────────────
#  Support / Resistance
# ─────────────────────────────────────────────────────────────

def _find_sr(df: pd.DataFrame, window: int = 5) -> tuple[list[float], list[float]]:
    highs   = df["High"].values
    lows    = df["Low"].values
    current = float(df["Close"].iloc[-1])

    res_pts: list[float] = []
    sup_pts: list[float] = []

    for i in range(window, len(df) - window):
        if highs[i] == max(highs[i - window: i + window + 1]):
            res_pts.append(float(highs[i]))
        if lows[i] == min(lows[i - window: i + window + 1]):
            sup_pts.append(float(lows[i]))

    def cluster(pts: list[float], thr: float = 0.02) -> list[float]:
        if not pts:
            return []
        pts = sorted(pts)
        groups: list[list[float]] = [[pts[0]]]
        for p in pts[1:]:
            if p / groups[-1][-1] - 1 < thr:
                groups[-1].append(p)
            else:
                groups.append([p])
        return [sum(g) / len(g) for g in groups]

    supports    = sorted([s for s in cluster(sup_pts) if s < current * 1.03], reverse=True)[:4]
    resistances = sorted([r for r in cluster(res_pts) if r > current * 0.97])[:4]
    return supports, resistances


# ─────────────────────────────────────────────────────────────
#  Candlestick patterns (last 1-3 candles)
# ─────────────────────────────────────────────────────────────

def _candles(symbol: str, df: pd.DataFrame, price: float,
             sup: Optional[float], res: Optional[float]) -> list[Alert]:
    alerts: list[Alert] = []
    if len(df) < 3:
        return alerts

    def add(title: str, desc: str, rec: str, key: str, cooldown: int = 12) -> None:
        alerts.append(Alert(symbol=symbol, title=title, price=price,
                            support=sup, resistance=res,
                            description=desc, recommendation=rec,
                            key=key, cooldown_hours=cooldown))

    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values

    o1, h1, l1, c1 = float(o[-1]), float(h[-1]), float(l[-1]), float(c[-1])
    o2, h2, l2, c2 = float(o[-2]), float(h[-2]), float(l[-2]), float(c[-2])
    o3, _,  _,  c3 = float(o[-3]), float(h[-3]), float(l[-3]), float(c[-3])

    body1  = abs(c1 - o1)
    range1 = h1 - l1

    # Doji
    if range1 > 0 and body1 / range1 < 0.1:
        add("Doji 🕯️",
            "נר דוג'י — גוף קטן, חוסר החלטיות בשוק",
            "המתן לאישור כיוון בנר הבא",
            "doji")

    # Hammer  (small body at top, long lower wick, after downtrend)
    if range1 > 0 and body1 > 0:
        lower_wick = min(o1, c1) - l1
        upper_wick = h1 - max(o1, c1)
        if lower_wick > 2 * body1 and upper_wick < 0.3 * range1:
            if len(c) >= 5 and float(c[-5]) > c1:   # downtrend
                add("Hammer 🔨 (עולה)",
                    "נר פטיש — זנב תחתון ארוך, סיגנל היפוך עולה אפשרי",
                    "כדאי לבדוק כניסה אם הנר הבא מאשר עלייה",
                    "hammer")

        # Shooting Star (small body at bottom, long upper wick, after uptrend)
        elif upper_wick > 2 * body1 and lower_wick < 0.3 * range1:
            if len(c) >= 5 and float(c[-5]) < c1:   # uptrend
                add("Shooting Star 💫 (יורד)",
                    "נר כוכב נופל — זנב עליון ארוך, סיגנל היפוך יורד אפשרי",
                    "שקול יציאה אם הנר הבא מאשר ירידה",
                    "shooting_star")

    # Bullish Engulfing
    if c2 < o2 and c1 > o1 and c1 > o2 and o1 < c2:
        add("Bullish Engulfing 📈",
            "בליעה עולה — נר ירוק בלע לחלוטין את הנר האדום הקודם",
            "סיגנל עולה — כדאי לבדוק כניסה",
            "bull_engulfing")

    # Bearish Engulfing
    elif c2 > o2 and c1 < o1 and c1 < o2 and o1 > c2:
        add("Bearish Engulfing 📉",
            "בליעה יורדת — נר אדום בלע לחלוטין את הנר הירוק הקודם",
            "סיגנל יורד — שקול יציאה או פתיחת שורט",
            "bear_engulfing")

    # Morning Star (3 candles: bearish → small star → bullish)
    body3  = abs(c3 - o3)
    range3 = float(h[-3]) - float(l[-3])
    if (c3 < o3 and body3 > range3 * 0.4
            and abs(c2 - o2) < (h2 - l2) * 0.3
            and c1 > o1 and c1 > (o3 + c3) / 2):
        add("Morning Star 🌅 (עולה)",
            "פטרן כוכב הבוקר — היפוך עולה אחרי ירידה",
            "סיגנל עולה חזק — כדאי לבדוק כניסה",
            "morning_star")

    # Evening Star (3 candles: bullish → small star → bearish)
    if (c3 > o3 and body3 > range3 * 0.4
            and abs(c2 - o2) < (h2 - l2) * 0.3
            and c1 < o1 and c1 < (o3 + c3) / 2):
        add("Evening Star 🌆 (יורד)",
            "פטרן כוכב הערב — היפוך יורד אחרי עלייה",
            "סיגנל יורד — שקול יציאה",
            "evening_star")

    return alerts


# ─────────────────────────────────────────────────────────────
#  Chart patterns
# ─────────────────────────────────────────────────────────────

def _patterns(symbol: str, df: pd.DataFrame, price: float,
              sup: Optional[float], res: Optional[float]) -> list[Alert]:
    alerts: list[Alert] = []

    def add(title: str, desc: str, rec: str, key: str, cooldown: int = 24) -> None:
        alerts.append(Alert(symbol=symbol, title=title, price=price,
                            support=sup, resistance=res,
                            description=desc, recommendation=rec,
                            key=key, cooldown_hours=cooldown))

    closes  = df["Close"].values.astype(float)
    highs   = df["High"].values.astype(float)
    lows    = df["Low"].values.astype(float)
    volumes = df["Volume"].values.astype(float)
    n       = len(closes)

    # ── Double Top / Double Bottom (last 30 bars) ─────────────
    win = min(30, n)
    wh  = highs[-win:]
    wl  = lows[-win:]

    peaks   = [(i, wh[i]) for i in range(2, win - 2) if wh[i] == max(wh[i-2:i+3])]
    troughs = [(i, wl[i]) for i in range(2, win - 2) if wl[i] == min(wl[i-2:i+3])]

    for i in range(len(peaks) - 1):
        for j in range(i + 1, len(peaks)):
            pi, pv = peaks[i]; pj, p2v = peaks[j]
            if pj - pi < 5:
                continue
            if abs(pv - p2v) / max(pv, p2v) < 0.03:
                neckline = min(wl[pi:pj + 1])
                if (max(pv, p2v) - neckline) / max(pv, p2v) > 0.04:
                    if price < neckline * 1.02:
                        add("Double Top 🔴",
                            f"שתי פסגות ב-${max(pv,p2v):.2f}, צוואר ב-${neckline:.2f}",
                            "סיגנל יורד — שקול יציאה אם המחיר שובר מתחת לצוואר",
                            "double_top")

    for i in range(len(troughs) - 1):
        for j in range(i + 1, len(troughs)):
            ti, tv = troughs[i]; tj, t2v = troughs[j]
            if tj - ti < 5:
                continue
            if abs(tv - t2v) / min(tv, t2v) < 0.03:
                neckline = max(wh[ti:tj + 1])
                if (neckline - min(tv, t2v)) / min(tv, t2v) > 0.04:
                    if price > neckline * 0.98:
                        add("Double Bottom 🟢",
                            f"שתי תחתיות ב-${min(tv,t2v):.2f}, צוואר ב-${neckline:.2f}",
                            "סיגנל עולה — כדאי לבדוק כניסה מעל קו הצוואר",
                            "double_bottom")

    # ── Head and Shoulders ────────────────────────────────────
    for i in range(len(peaks) - 2):
        ls_i, ls_v = peaks[i]
        h_i,  h_v  = peaks[i + 1]
        rs_i, rs_v = peaks[i + 2]
        if h_v > ls_v * 1.04 and h_v > rs_v * 1.04:
            if abs(ls_v - rs_v) / max(ls_v, rs_v) < 0.08:
                neckline = (wl[ls_i] + wl[rs_i]) / 2
                if price < neckline * 1.02:
                    add("Head and Shoulders 🔴",
                        f"ראש ב-${h_v:.2f}, כתפיים ב-${ls_v:.2f}/${rs_v:.2f}, צוואר ב-${neckline:.2f}",
                        "סיגנל יורד חזק — שקול יציאה",
                        "head_shoulders")
                break

    # ── Bull Flag ─────────────────────────────────────────────
    if n >= 13:
        pole_gain = (closes[-7] - closes[-13]) / closes[-13]
        flag_move = (closes[-1] - closes[-7])  / closes[-7]
        if pole_gain > 0.12 and -0.08 < flag_move < 0.02:
            add("Bull Flag 🚀",
                f"עמוד עלייה של {pole_gain*100:.1f}% ואיחוד צדדי — דגל שורי",
                "פריצה אפשרית — כדאי לבדוק כניסה עם סטופ מתחת לתחתית הדגל",
                "bull_flag")

    # ── Bear Flag ─────────────────────────────────────────────
    if n >= 13:
        pole_drop = (closes[-13] - closes[-7]) / closes[-13]
        flag_move = (closes[-1]  - closes[-7]) / closes[-7]
        if pole_drop > 0.12 and -0.02 < flag_move < 0.08:
            add("Bear Flag 🐻",
                f"עמוד ירידה של {pole_drop*100:.1f}% ואיחוד צדדי — דגל דובי",
                "פריצה יורדת אפשרית — שקול שורט עם סטופ מעל ראש הדגל",
                "bear_flag")

    # ── Breakout from Consolidation ───────────────────────────
    if n >= 10:
        consol = closes[-9:-1]
        consol_range = (max(consol) - min(consol)) / float(closes[-9])
        if consol_range < 0.05:
            avg_vol  = float(np.mean(volumes[-21:-1])) if n >= 21 else float(np.mean(volumes[:-1]))
            curr_vol = float(volumes[-1])
            vol_spike = avg_vol > 0 and curr_vol > avg_vol * 1.5
            if price > max(consol) and vol_spike:
                add("Breakout מאיחוד 🚀",
                    f"פריצה מעל אזור איחוד ({consol_range*100:.1f}% טווח) עם נפח גבוה",
                    "פריצה חיובית — כדאי לבדוק כניסה",
                    "breakout_up", 12)
            elif price < min(consol) and vol_spike:
                add("שבירה מאיחוד 📉",
                    f"שבירה מתחת לאזור איחוד ({consol_range*100:.1f}% טווח) עם נפח גבוה",
                    "שבירה שלילית — שקול יציאה",
                    "breakout_down", 12)

    # ── Ascending / Descending Triangle (last 15 bars) ───────
    if n >= 15:
        rh = highs[-15:]
        rl = lows[-15:]
        x  = np.arange(15, dtype=float)

        high_range  = (max(rh) - min(rh)) / max(rh)
        lows_slope  = float(np.polyfit(x, rl, 1)[0])
        highs_slope = float(np.polyfit(x, rh, 1)[0])

        if high_range < 0.03 and lows_slope > 0:
            add("Ascending Triangle ▲",
                "התנגדות שטוחה עם תמיכה עולה — לחץ עולה",
                "פריצה כלפי מעלה אפשרית — כדאי לעקוב",
                "ascending_triangle")

        low_range = (max(rl) - min(rl)) / max(rl)
        if low_range < 0.03 and highs_slope < 0:
            add("Descending Triangle ▽",
                "תמיכה שטוחה עם התנגדות יורדת — לחץ יורד",
                "פריצה כלפי מטה אפשרית — שקול יציאה",
                "descending_triangle")

    # ── Cup and Handle (simplified, last 30 bars) ────────────
    if n >= 25:
        cup = closes[-25:-5]
        mid_idx = len(cup) // 2
        left  = float(cup[0])
        bot   = float(min(cup))
        right = float(cup[-1])
        handle = closes[-5:]
        depth  = (max(left, right) - bot) / max(left, right)

        if (depth > 0.08
                and abs(left - right) / max(left, right) < 0.05
                and float(bot) == float(cup[mid_idx - 1: mid_idx + 2].min())
                and max(handle) < right * 1.03
                and min(handle) > bot * 1.02
                and price > right * 0.99):
            add("Cup and Handle ☕",
                f"פטרן כוס וידית — עומק כוס {depth*100:.1f}%",
                "סיגנל עולה אפשרי — כדאי לבדוק כניסה מעל ימין הכוס",
                "cup_handle")

    return alerts


# ─────────────────────────────────────────────────────────────
#  Swing-specific alerts
# ─────────────────────────────────────────────────────────────

def _swing(symbol: str, df: pd.DataFrame, price: float,
           sup: Optional[float], res: Optional[float]) -> list[Alert]:
    alerts: list[Alert] = []
    if len(df) < 2:
        return alerts

    def add(title: str, desc: str, rec: str, key: str, cooldown: int = 12) -> None:
        alerts.append(Alert(symbol=symbol, title=title, price=price,
                            support=sup, resistance=res,
                            description=desc, recommendation=rec,
                            key=key, cooldown_hours=cooldown))

    prev_close = float(df["Close"].iloc[-2])
    curr_open  = float(df["Open"].iloc[-1])

    daily_chg = (price - prev_close) / prev_close * 100

    if daily_chg <= -10:
        add(f"ירידה חדה {daily_chg:.1f}% 💥",
            f"המנייה ירדה {abs(daily_chg):.1f}% — הזדמנות כניסה אפשרית?",
            "בדוק אם יש סיבה בסיסית לירידה לפני כניסה",
            "big_drop")

    elif daily_chg >= 15:
        add(f"עלייה חדה +{daily_chg:.1f}% 🚀",
            f"המנייה עלתה {daily_chg:.1f}% — שקול יציאה חלקית",
            "נעל רווחים חלקית, הגדר סטופ על שיא היום",
            "big_rise")

    gap_pct = (curr_open - prev_close) / prev_close * 100
    if gap_pct > 3:
        add(f"Gap Up +{gap_pct:.1f}% ⬆️",
            f"פתיחה בפער של {gap_pct:.1f}% מעל סגירת אתמול (${prev_close:.2f})",
            "שקול כניסה אם הפער נשמר, או המתן לסגירת הפער",
            "gap_up")
    elif gap_pct < -3:
        add(f"Gap Down {gap_pct:.1f}% ⬇️",
            f"פתיחה בפער של {abs(gap_pct):.1f}% מתחת לסגירת אתמול (${prev_close:.2f})",
            "המתן לייצוב לפני כניסה — בדוק אם יש סיבה לפער",
            "gap_down")

    return alerts


# ─────────────────────────────────────────────────────────────
#  Main analyze entry-point
# ─────────────────────────────────────────────────────────────

def analyze_symbol(symbol: str) -> list[Alert]:
    alerts: list[Alert] = []
    try:
        df = _fetch_daily(symbol)
        if df.empty or len(df) < 20:
            log.warning("Not enough data for %s", symbol)
            return []
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

        closes  = df["Close"]
        volumes = df["Volume"]
        price   = float(closes.iloc[-1])
        prev    = float(closes.iloc[-2])

        supports, resistances = _find_sr(df)
        sup = supports[0]    if supports    else None
        res = resistances[0] if resistances else None

        def add(title: str, desc: str, rec: str, key: str, cooldown: int = 6) -> None:
            alerts.append(Alert(symbol=symbol, title=title, price=price,
                                support=sup, resistance=res,
                                description=desc, recommendation=rec,
                                key=key, cooldown_hours=cooldown))

        # ── RSI ──────────────────────────────────────────────
        if len(closes) >= 15:
            rsi = _rsi(closes)
            if rsi < 30:
                add("RSI Oversold 📉",
                    f"RSI ירד ל-{rsi:.1f} — אזור מכירת יתר (oversold)",
                    "כדאי לבדוק כניסה לונג עם סטופ מתחת לתמיכה",
                    "rsi_oversold", 8)
            elif rsi > 70:
                add("RSI Overbought 📈",
                    f"RSI עלה ל-{rsi:.1f} — אזור קנייה יתר (overbought)",
                    "שקול יציאה חלקית או הגדרת סטופ",
                    "rsi_overbought", 8)

        # ── MACD ─────────────────────────────────────────────
        if len(closes) >= 30:
            macd_v, sig_v, hist_v, prev_hist = _macd(closes)
            if prev_hist < 0 < hist_v:
                add("MACD Crossover חיובי ✅",
                    f"MACD ({macd_v:.3f}) חצה מעל קו הסיגנל ({sig_v:.3f})",
                    "מומנטום עולה — כדאי לבדוק כניסה לונג",
                    "macd_bullish", 12)
            elif prev_hist > 0 > hist_v:
                add("MACD Crossover שלילי ❌",
                    f"MACD ({macd_v:.3f}) חצה מתחת לקו הסיגנל ({sig_v:.3f})",
                    "מומנטום יורד — שקול יציאה",
                    "macd_bearish", 12)

        # ── SMA 50 ───────────────────────────────────────────
        s50, s50_p = _sma(closes, 50)
        if s50 is not None and s50_p is not None:
            if prev < s50_p and price > s50:
                add("פריצה מעל MA50 🟢",
                    f"המחיר עבר מעל ממוצע 50 ימים (${s50:.2f})",
                    "סיגנל חיובי — כדאי לבדוק כניסה",
                    "above_ma50", 24)
            elif prev > s50_p and price < s50:
                add("שבירה מתחת MA50 🔴",
                    f"המחיר שבר מתחת לממוצע 50 ימים (${s50:.2f})",
                    "סיגנל שלילי — שקול יציאה",
                    "below_ma50", 24)

        # ── SMA 150 ──────────────────────────────────────────
        s150, s150_p = _sma(closes, 150)
        if s150 is not None and s150_p is not None:
            # פריצה / שבירה
            if prev < s150_p and price > s150:
                add("🚀 פריצה מעל MA150",
                    f"המחיר פרץ מעל ממוצע 150 ימים (${s150:.2f})",
                    "סיגנל עולה — כדאי לבדוק כניסה",
                    "break_above_ma150", 8)
            elif prev > s150_p and price < s150:
                add("💥 שבירה מתחת ל-MA150",
                    f"המחיר שבר מתחת לממוצע 150 ימים (${s150:.2f})",
                    "סיגנל יורד — שקול יציאה",
                    "break_below_ma150", 8)
            else:
                dist_pct = (s150 - price) / price * 100
                # מתקרב מלמטה (מחיר מתחת ל-MA150 ומרחק <= 2%)
                if 0 < dist_pct <= 2.0:
                    add("📊 מתקרב ל-MA150 מלמטה",
                        f"מחיר: ${price:.2f} | MA150: ${s150:.2f} | מרחק: {dist_pct:.1f}%",
                        "פריצה אפשרית — עקוב מקרוב",
                        "approach_ma150_below", 8)
                # מתקרב מלמעלה (מחיר מעל MA150 ומרחק <= 2%)
                elif -2.0 <= dist_pct < 0:
                    add("📊 מתקרב ל-MA150 מלמעלה",
                        f"מחיר: ${price:.2f} | MA150: ${s150:.2f} | מרחק: {abs(dist_pct):.1f}%",
                        "תמיכה אפשרית — שים לב",
                        "approach_ma150_above", 8)

        # ── SMA 200 + Golden/Death Cross ─────────────────────
        s200, s200_p = _sma(closes, 200)
        if s200 is not None and s200_p is not None:
            if prev < s200_p and price > s200:
                add("פריצה מעל MA200 🟢",
                    f"המחיר עבר מעל ממוצע 200 ימים (${s200:.2f})",
                    "סיגנל חיובי חזק — כדאי לבדוק כניסה",
                    "above_ma200", 48)

            if s50 is not None and s50_p is not None:
                if s50_p < s200_p and s50 > s200:
                    add("Golden Cross 🌟",
                        f"MA50 (${s50:.2f}) חצה מעל MA200 (${s200:.2f})",
                        "סיגנל עולה מאוד חזק — כדאי לבדוק כניסה",
                        "golden_cross", 72)
                elif s50_p > s200_p and s50 < s200:
                    add("Death Cross ☠️",
                        f"MA50 (${s50:.2f}) חצה מתחת ל-MA200 (${s200:.2f})",
                        "סיגנל יורד חזק — שקול יציאה",
                        "death_cross", 72)

        # ── Volume spike ─────────────────────────────────────
        if len(volumes) >= 21:
            avg_vol  = float(volumes.iloc[-21:-1].mean())
            curr_vol = float(volumes.iloc[-1])
            if avg_vol > 0 and curr_vol > avg_vol * 3:
                add("נפח גבוה חריג 📊",
                    f"נפח היום: {curr_vol/1e6:.1f}M | ממוצע: {avg_vol/1e6:.1f}M (פי {curr_vol/avg_vol:.1f})",
                    "נפח חריג לרוב מעיד על כניסת כסף גדול או חדשות קרובות — כדאי לעקוב מקרוב",
                    "high_volume_3x", 4)

        # ── Bollinger Bands ───────────────────────────────────
        if len(closes) >= 20:
            bb_upper, bb_mid, bb_lower = _bollinger(closes)
            if price > bb_upper:
                add("📈 פריצה מעל Bollinger Band עליון",
                    f"מחיר: ${price:.2f} | Band עליון: ${bb_upper:.2f}",
                    "פריצה מעל הפס העליון — מומנטום חזק אבל גם overbought. בסווינג זה יכול להיות המשך עלייה או נקודת יציאה",
                    "bb_upper_break", 6)
            elif price < bb_lower:
                add("📉 ירידה מתחת ל-Bollinger Band תחתון",
                    f"מחיר: ${price:.2f} | Band תחתון: ${bb_lower:.2f}",
                    "נגיעה בפס התחתון לרוב מעידה על oversold — הזדמנות כניסה פוטנציאלית לסווינג",
                    "bb_lower_break", 6)

        # ── Fibonacci proximity alerts ────────────────────────
        if len(df) >= 30:
            _fib_window = df.iloc[-30:]
            _fib_high = float(_fib_window["High"].max())
            _fib_low  = float(_fib_window["Low"].min())
            _fib_range = _fib_high - _fib_low
            if _fib_range > 0:
                _fib_levels = [
                    (0,    _fib_high),
                    (23.6, _fib_high - 0.236 * _fib_range),
                    (38.2, _fib_high - 0.382 * _fib_range),
                    (50,   _fib_high - 0.5   * _fib_range),
                    (61.8, _fib_high - 0.618 * _fib_range),
                    (78.6, _fib_high - 0.786 * _fib_range),
                    (100,  _fib_low),
                ]
                for pct, level in _fib_levels:
                    if level > 0 and abs(price - level) / level <= 0.01:
                        if pct == 61.8:
                            _fib_rec = "רמת הזהב (Golden Ratio) — רמת היפוך חשובה מאוד. שים לב לכיוון הבא"
                        elif price > level:
                            _fib_rec = f"המחיר מעל רמת פיבונאצ'י {pct}% — עשוי לשמש תמיכה"
                        else:
                            _fib_rec = f"המחיר מתחת לרמת פיבונאצ'י {pct}% — עשוי לשמש התנגדות"
                        add(f"📐 קרוב לרמת פיבונאצ'י {pct}%",
                            f"מחיר: ${price:.2f} | רמה: ${level:.2f} ({pct}%)",
                            _fib_rec,
                            f"fib_{pct}_{round(level,1)}", 8)

        # ── VWAP crossover alerts ─────────────────────────────
        if len(df) >= 22:
            _vwap_curr, _vwap_prev_val = _vwap(df)
            _prev_close = float(closes.iloc[-2])
            if _vwap_curr is not None and _vwap_prev_val is not None:
                if _prev_close < _vwap_prev_val and price > _vwap_curr:
                    add("⚡ חציית VWAP כלפי מעלה",
                        f"מחיר: ${price:.2f} | VWAP: ${_vwap_curr:.2f}",
                        "חציית VWAP כלפי מעלה — סיגנל שורי. מומנטום עולה",
                        "vwap_cross_up", 4)
                elif _prev_close > _vwap_prev_val and price < _vwap_curr:
                    add("⚡ חציית VWAP כלפי מטה",
                        f"מחיר: ${price:.2f} | VWAP: ${_vwap_curr:.2f}",
                        "חציית VWAP כלפי מטה — סיגנל דובי. מומנטום יורד",
                        "vwap_cross_down", 4)

        # ── Support / Resistance touch ────────────────────────
        for s in supports[:2]:
            if abs(price - s) / s < 0.015:
                add("הגעה לאזור תמיכה 🛡️",
                    f"המחיר נוגע באזור תמיכה חזק ב-${s:.2f}",
                    "כדאי לבדוק כניסה עם סטופ מתחת לתמיכה",
                    f"sup_touch_{round(s, 1)}", 12)
                break

        avg_vol_v = float(volumes.iloc[-21:-1].mean()) if len(volumes) >= 21 else 0
        curr_vol_v = float(volumes.iloc[-1])
        for r in resistances[:2]:
            dist = (price - r) / r
            if -0.015 < dist < 0.025:
                if dist > 0 and avg_vol_v > 0 and curr_vol_v > avg_vol_v * 1.3:
                    add("פריצה מעל התנגדות 🚀",
                        f"פריצה מעל התנגדות ב-${r:.2f} עם נפח גבוה",
                        "פריצה חיובית — כדאי לבדוק כניסה",
                        f"res_break_{round(r, 1)}", 12)
                elif dist <= 0:
                    add("הגעה לאזור התנגדות 🔺",
                        f"המחיר נוגע באזור התנגדות ב-${r:.2f}",
                        "שקול יציאה חלקית או הגדרת סטופ",
                        f"res_touch_{round(r, 1)}", 12)
                break

        # ── Candlestick + Chart patterns + Swing ─────────────
        alerts.extend(_candles(symbol, df, price, sup, res))
        if len(df) >= 20:
            alerts.extend(_patterns(symbol, df, price, sup, res))
        alerts.extend(_swing(symbol, df, price, sup, res))

    except Exception as exc:
        log.error("Error analyzing %s: %s", symbol, exc, exc_info=True)

    return alerts


# ─────────────────────────────────────────────────────────────
#  Status + formatted output helpers
# ─────────────────────────────────────────────────────────────

def get_quick_status() -> list[dict]:
    """Fetch price + change + RSI for all watchlist symbols."""
    results: list[dict] = []
    for symbol in WATCHLIST:
        try:
            df = _fetch_daily(symbol)
            if df.empty or len(df) < 2:
                results.append({"symbol": symbol, "price": None, "change": None, "rsi": None})
                continue
            closes = df["Close"]
            price  = float(closes.iloc[-1])
            chg    = (price - float(closes.iloc[-2])) / float(closes.iloc[-2]) * 100
            rsi    = _rsi(closes) if len(closes) >= 15 else None
            results.append({"symbol": symbol, "price": price, "change": chg, "rsi": rsi})
        except Exception as exc:
            log.warning("status error %s: %s", symbol, exc)
            results.append({"symbol": symbol, "price": None, "change": None, "rsi": None})
    return results


def get_full_analysis(symbol: str) -> str:
    try:
        df = _fetch_daily(symbol)
        if df.empty or len(df) < 20:
            return f"❌ אין מספיק נתונים עבור {symbol}"
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

        closes  = df["Close"]
        volumes = df["Volume"]
        price   = float(closes.iloc[-1])
        chg     = (price - float(closes.iloc[-2])) / float(closes.iloc[-2]) * 100

        supports, resistances = _find_sr(df)
        rsi  = _rsi(closes) if len(closes) >= 15 else None
        s50, _  = _sma(closes, 50)
        s200, _ = _sma(closes, 200)

        macd_v = sig_v = hist_v = None
        if len(closes) >= 30:
            macd_v, sig_v, hist_v, _ = _macd(closes)

        avg_vol  = float(volumes.iloc[-21:-1].mean()) if len(volumes) >= 21 else None
        curr_vol = float(volumes.iloc[-1])

        lines = [
            f"📊 *ניתוח מלא — {symbol}*",
            f"💵 מחיר: ${price:.2f}  ({'+'if chg>=0 else ''}{chg:.2f}%)",
            "",
            "🔢 *אינדיקטורים:*",
        ]

        if rsi is not None:
            emoji = "🟢" if rsi < 30 else "🔴" if rsi > 70 else "⚪"
            lines.append(f"  RSI(14): {rsi:.1f} {emoji}")

        if macd_v is not None:
            emoji = "🟢" if hist_v > 0 else "🔴"
            lines.append(f"  MACD: {macd_v:.3f} | Signal: {sig_v:.3f} | Hist: {hist_v:.3f} {emoji}")

        if s50:
            pos = "מעל ✅" if price > s50 else "מתחת ❌"
            lines.append(f"  MA50:  ${s50:.2f}  ({pos})")
        if s200:
            pos = "מעל ✅" if price > s200 else "מתחת ❌"
            lines.append(f"  MA200: ${s200:.2f}  ({pos})")

        if avg_vol:
            ratio = curr_vol / avg_vol
            lines.append(f"  נפח: {ratio:.1f}x ממוצע {'⚡' if ratio > 2 else ''}")

        lines += ["", "🛡️ *תמיכות:*"]
        for s in supports[:3]:
            dist = (price - s) / price * 100
            lines.append(f"  ${s:.2f}  ({dist:.1f}% מתחת)")
        if not supports:
            lines.append("  לא נמצאו")

        lines.append("🔺 *התנגדויות:*")
        for r in resistances[:3]:
            dist = (r - price) / price * 100
            lines.append(f"  ${r:.2f}  ({dist:.1f}% מעל)")
        if not resistances:
            lines.append("  לא נמצאו")

        # List active alerts
        active = analyze_symbol(symbol)
        if active:
            lines += ["", "🚨 *התראות פעילות:*"]
            for a in active[:5]:
                lines.append(f"  • {a.title}")

        return "\n".join(lines)

    except Exception as exc:
        return f"❌ שגיאה בניתוח {symbol}: {exc}"


def get_levels(symbol: str) -> str:
    try:
        df = _fetch_daily(symbol)
        if df.empty or len(df) < 20:
            return f"❌ אין מספיק נתונים עבור {symbol}"
        df = df.dropna(subset=["High", "Low", "Close"])
        price = float(df["Close"].iloc[-1])
        supports, resistances = _find_sr(df)

        lines = [
            f"🎯 *רמות תמיכה והתנגדות — {symbol}*",
            f"💵 מחיר נוכחי: ${price:.2f}",
            "",
            "🛡️ *תמיכות:*",
        ]
        for s in supports:
            dist = (price - s) / price * 100
            lines.append(f"  ${s:.2f}   ({dist:.1f}% מתחת)")
        if not supports:
            lines.append("  לא נמצאו")

        lines += ["", "🔺 *התנגדויות:*"]
        for r in resistances:
            dist = (r - price) / price * 100
            lines.append(f"  ${r:.2f}   (+{dist:.1f}% מעל)")
        if not resistances:
            lines.append("  לא נמצאו")

        return "\n".join(lines)

    except Exception as exc:
        return f"❌ שגיאה ב-{symbol}: {exc}"


def get_bollinger_levels(symbol: str) -> str:
    try:
        df = _fetch_daily(symbol)
        if df.empty or len(df) < 20:
            return f"❌ אין מספיק נתונים עבור {symbol}"
        df = df.dropna(subset=["Close"])
        closes = df["Close"]
        price  = float(closes.iloc[-1])
        bb_upper, bb_mid, bb_lower = _bollinger(closes)
        band_width_pct = (bb_upper - bb_lower) / bb_mid * 100
        if band_width_pct < 5:
            width_interp = "פס צר מאוד — צפוי תנועה חדה בקרוב (squeeze)"
        elif band_width_pct < 10:
            width_interp = "פס צר — שוק בצד, המתן לפריצה"
        elif band_width_pct < 20:
            width_interp = "פס בינוני — תנודתיות רגילה"
        else:
            width_interp = "פס רחב — תנודתיות גבוהה, זהירות"
        return (
            f"📊 *Bollinger Bands — {symbol}*\n"
            f"💵 מחיר נוכחי: ${price:.2f}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📈 Band עליון:  ${bb_upper:.2f}\n"
            f"➖ Band אמצעי: ${bb_mid:.2f} (MA20)\n"
            f"📉 Band תחתון: ${bb_lower:.2f}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📌 רוחב הפס: {band_width_pct:.1f}% — {width_interp}"
        )
    except Exception as exc:
        return f"❌ שגיאה ב-{symbol}: {exc}"


def get_fibonacci_levels(symbol: str) -> str:
    try:
        df = _fetch_daily(symbol)
        if df.empty or len(df) < 10:
            return f"❌ אין מספיק נתונים עבור {symbol}"
        df = df.dropna(subset=["High", "Low", "Close"])
        window = df.iloc[-30:]
        fib_high  = float(window["High"].max())
        fib_low   = float(window["Low"].min())
        fib_range = fib_high - fib_low
        price     = float(df["Close"].iloc[-1])

        pcts   = [0, 23.6, 38.2, 50, 61.8, 78.6, 100]
        levels = [fib_high - (p / 100) * fib_range for p in pcts]

        # Find which two levels price is between
        between_low_pct  = None
        between_high_pct = None
        sorted_pairs = sorted(zip(levels, pcts), key=lambda x: x[0])
        for i in range(len(sorted_pairs) - 1):
            lvl_a, pct_a = sorted_pairs[i]
            lvl_b, pct_b = sorted_pairs[i + 1]
            if lvl_a <= price <= lvl_b:
                between_low_pct  = pct_a
                between_high_pct = pct_b
                break

        lines = [
            f"📐 *רמות פיבונאצ'י — {symbol}*",
            f"💵 מחיר נוכחי: ${price:.2f}",
            f"📊 טווח: ${fib_low:.2f} — ${fib_high:.2f} (30 ימים)",
            "━━━━━━━━━━━━━━━",
        ]
        for pct, level in zip(pcts, levels):
            marker = " ◄ מחיר כאן" if (between_low_pct is not None
                                        and pct in (between_low_pct, between_high_pct)
                                        and abs(price - level) / max(level, 0.01) < 0.03) else ""
            lines.append(f"  {pct:5.1f}%  ${level:.2f}{marker}")
        lines.append("━━━━━━━━━━━━━━━")
        if between_low_pct is not None:
            lines.append(f"📌 המחיר בין {between_low_pct}% ל-{between_high_pct}%")
        return "\n".join(lines)
    except Exception as exc:
        return f"❌ שגיאה ב-{symbol}: {exc}"


def get_vwap(symbol: str) -> str:
    try:
        df = _fetch_daily(symbol)
        if df.empty or len(df) < 21:
            return f"❌ אין מספיק נתונים עבור {symbol}"
        df = df.dropna(subset=["High", "Low", "Close", "Volume"])
        price = float(df["Close"].iloc[-1])
        curr_vwap, _ = _vwap(df)
        if curr_vwap is None:
            return f"❌ לא ניתן לחשב VWAP עבור {symbol}"
        if price > curr_vwap:
            position = "מעל"
            interp   = "המחיר מעל ה-VWAP — מצב שורי, הביקוש דומיננטי"
        else:
            position = "מתחת"
            interp   = "המחיר מתחת ל-VWAP — מצב דובי, ההיצע דומיננטי"
        return (
            f"⚡ *VWAP — {symbol}*\n"
            f"💵 מחיר נוכחי: ${price:.2f}\n"
            f"📊 VWAP (20 ימים): ${curr_vwap:.2f}\n"
            f"📌 המחיר {position} ל-VWAP — {interp}"
        )
    except Exception as exc:
        return f"❌ שגיאה ב-{symbol}: {exc}"


# ─────────────────────────────────────────────────────────────
#  S/R proximity scan (used by bot.py every hour)
# ─────────────────────────────────────────────────────────────

SR_PROXIMITY_PCT = 2.0   # alert when within this % of a level


def check_sr_proximity(symbol: str) -> list[Alert]:
    """
    Returns Alert objects for every S/R level within SR_PROXIMITY_PCT of
    the current price.  Separate from analyze_symbol so it can use a
    different cooldown without flooding existing alert channels.
    """
    alerts: list[Alert] = []
    try:
        df = _fetch_daily(symbol)
        if df.empty or len(df) < 20:
            return []
        df = df.dropna(subset=["High", "Low", "Close"])

        price = float(df["Close"].iloc[-1])
        supports, resistances = _find_sr(df)

        for s in supports:
            dist_pct = (price - s) / price * 100
            if 0 < dist_pct <= SR_PROXIMITY_PCT:
                alerts.append(Alert(
                    symbol=symbol,
                    title=f"מתקרב לתמיכה ב-${s:.2f} 📊",
                    price=price,
                    support=s,
                    resistance=resistances[0] if resistances else None,
                    description=(
                        f"המחיר נמצא {dist_pct:.1f}% מעל אזור תמיכה חזק ב-${s:.2f}"
                    ),
                    recommendation="שים לב — אזור כניסה פוטנציאלי",
                    key=f"sr_prox_sup_{round(s, 1)}",
                    cooldown_hours=4,
                ))

        for r in resistances:
            dist_pct = (r - price) / price * 100
            if 0 < dist_pct <= SR_PROXIMITY_PCT:
                alerts.append(Alert(
                    symbol=symbol,
                    title=f"מתקרב להתנגדות ב-${r:.2f} 📊",
                    price=price,
                    support=supports[0] if supports else None,
                    resistance=r,
                    description=(
                        f"המחיר נמצא {dist_pct:.1f}% מתחת לאזור התנגדות חזק ב-${r:.2f}"
                    ),
                    recommendation="שים לב — אזור יציאה / זהירות פוטנציאלי",
                    key=f"sr_prox_res_{round(r, 1)}",
                    cooldown_hours=4,
                ))

    except Exception as exc:
        log.error("SR proximity error %s: %s", symbol, exc)

    return alerts
