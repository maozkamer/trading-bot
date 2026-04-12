"""
Chart generation — candlestick + S/R lines + MA50/200 + volume.
Returns a BytesIO PNG ready to send via Telegram.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mplfinance as mpf
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  Style
# ─────────────────────────────────────────────────────────────

_STYLE = mpf.make_mpf_style(
    base_mpf_style="nightclouds",
    facecolor="#0d1117",
    edgecolor="#30363d",
    figcolor="#0d1117",
    gridcolor="#21262d",
    gridstyle="--",
    gridaxis="both",
    y_on_right=True,
    rc={
        "axes.labelcolor":  "#c9d1d9",
        "xtick.color":      "#c9d1d9",
        "ytick.color":      "#c9d1d9",
        "font.family":      "DejaVu Sans",
    },
)

_UP_COLOR   = "#26a641"
_DOWN_COLOR = "#f85149"
_SUP_COLOR  = "#26a641"   # green  — support
_RES_COLOR  = "#f85149"   # red    — resistance
_MA50_COLOR = "#d29922"   # yellow
_MA200_COLOR= "#58a6ff"   # blue


# ─────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────

def build_chart(
    symbol: str,
    df: pd.DataFrame,
    supports: list[float],
    resistances: list[float],
) -> io.BytesIO:
    """
    Build a candlestick chart for the last 30 rows of *df*.
    Returns a BytesIO PNG buffer.
    """
    # ── Slice to last 30 trading days ────────────────────────
    df = df.tail(30).copy()
    if len(df) < 5:
        raise ValueError(f"Not enough data to chart {symbol} ({len(df)} rows)")

    closes = df["Close"]
    price  = float(closes.iloc[-1])

    # ── Moving averages (calculated on the full df, sliced to plot window) ──
    add_plots = []

    # MA50 — use the full history passed in for correct calculation
    if len(closes) >= 10:
        ma50 = closes.rolling(50, min_periods=10).mean()
        add_plots.append(
            mpf.make_addplot(ma50, color=_MA50_COLOR, width=1.4,
                             label="MA50", secondary_y=False)
        )

    if len(closes) >= 20:
        ma200 = closes.rolling(200, min_periods=20).mean()
        add_plots.append(
            mpf.make_addplot(ma200, color=_MA200_COLOR, width=1.4,
                             label="MA200", secondary_y=False)
        )

    # ── S/R horizontal lines (only those within ±15% of price) ──
    price_low  = price * 0.85
    price_high = price * 1.15

    vis_sup = [s for s in supports    if price_low  < s < price * 1.03][:3]
    vis_res = [r for r in resistances if price * 0.97 < r < price_high][:3]

    hlines_vals   = []
    hlines_colors = []
    hlines_lw     = []

    for s in vis_sup:
        hlines_vals.append(s)
        hlines_colors.append(_SUP_COLOR)
        hlines_lw.append(1.2)

    for r in vis_res:
        hlines_vals.append(r)
        hlines_colors.append(_RES_COLOR)
        hlines_lw.append(1.2)

    # ── Plot ─────────────────────────────────────────────────
    mc = mpf.make_marketcolors(
        up="green", down="red",
        edge={"up": "green", "down": "red"},
        wick={"up": "green", "down": "red"},
        volume={"up": "green", "down": "red"},
    )
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mc,
        facecolor="#0d1117",
        edgecolor="#30363d",
        figcolor="#0d1117",
        gridcolor="#21262d",
        gridstyle="--",
        y_on_right=True,
        rc={
            "axes.labelcolor": "#c9d1d9",
            "xtick.color":     "#8b949e",
            "ytick.color":     "#c9d1d9",
        },
    )

    fig, axes = mpf.plot(
        df,
        type="candle",
        style=style,
        volume=True,
        addplot=add_plots if add_plots else None,
        figsize=(12, 7),
        title=f"\n  {symbol}  —  ${price:.2f}   (30 ימים אחרונים)",
        tight_layout=True,
        returnfig=True,
    )

    # ── Draw S/R lines directly on the axis (avoids mplfinance kwarg issues) ──
    ax = axes[0]
    for s, color, lw in zip(hlines_vals, hlines_colors, hlines_lw):
        ax.axhline(y=s, color=color, linewidth=lw, linestyle="--", alpha=0.75)

    # ── Legend ───────────────────────────────────────────────
    legend_patches = []
    if any(mpf_p.get_label() == "MA50"  for mpf_p in add_plots):
        legend_patches.append(mpatches.Patch(color=_MA50_COLOR,  label="MA 50"))
    if any(mpf_p.get_label() == "MA200" for mpf_p in add_plots):
        legend_patches.append(mpatches.Patch(color=_MA200_COLOR, label="MA 200"))
    if vis_sup:
        legend_patches.append(mpatches.Patch(color=_SUP_COLOR, linestyle="--",
                                              label=f"תמיכה ({len(vis_sup)})"))
    if vis_res:
        legend_patches.append(mpatches.Patch(color=_RES_COLOR, linestyle="--",
                                              label=f"התנגדות ({len(vis_res)})"))

    if legend_patches:
        axes[0].legend(
            handles=legend_patches,
            loc="upper left",
            fontsize=8,
            facecolor="#161b22",
            edgecolor="#30363d",
            labelcolor="#c9d1d9",
        )

    # S/R price labels on the right y-axis
    for s in vis_sup:
        ax.annotate(f"  ${s:.2f}", xy=(1, s), xycoords=("axes fraction", "data"),
                    fontsize=7, color=_SUP_COLOR, va="center")
    for r in vis_res:
        ax.annotate(f"  ${r:.2f}", xy=(1, r), xycoords=("axes fraction", "data"),
                    fontsize=7, color=_RES_COLOR, va="center")


    # ── Encode to PNG buffer ──────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    return buf
