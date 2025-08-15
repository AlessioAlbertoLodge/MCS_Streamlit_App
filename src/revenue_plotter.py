# src/revenue_plotter.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.dates import DateFormatter


def plot_daily_and_cumulative_revenue(
    revenue_df: pd.DataFrame,
    *,
    bar_color: str = "black",          # thin black bars
    line_color: str = "#80bfff",       # light-blue cumulative line
    bar_width: float = 0.20,           # MUCH thinner bars (≈ 5 h wide on a daily axis)
    bar_alpha: float = 0.95,           # match the 75 % opacity
    fig_size: tuple[int, int] = (12, 6),
    dpi: int = 300,
    font_size: int = 14,
    title_suffix: str = "",
):
    """
    Bar (daily €) + line (cumulative €) chart with dual y-axes,
    using the same look-and-feel as plot_day_ahead_prices.

    Parameters
    ----------
    revenue_df : DataFrame with columns 'date' and 'revenue_eur'.
    bar_color  : colour for the daily-revenue columns.
    line_color : colour for the cumulative-revenue line.
    bar_width  : bar width in matplotlib date units (≈ days).
    bar_alpha  : opacity of the bars (0-1).
    fig_size   : (width, height) in inches.
    dpi        : figure resolution.
    font_size  : base font size for labels & ticks.
    title_suffix : optional string appended to the automatic title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ── prepare data ───────────────────────────────────────────────
    df = revenue_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df["cum_revenue"] = df["revenue_eur"].cumsum()

    # ── matplotlib styling ────────────────────────────────────────
    plt.rcParams.update({
        "font.size": font_size,
        "axes.labelsize": font_size + 1,
        "axes.titlesize": font_size + 2,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
    })

    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    # Bars: daily revenue (thin, semi-transparent)
    ax.bar(
        df["date"],
        df["revenue_eur"],
        color=bar_color,
        width=bar_width,
        alpha=bar_alpha,
        label="Daily revenue",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily revenue [€]")
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

    # Symmetric y-axis around 0 *if* we have losses as well
    rev_max = df["revenue_eur"].max()
    rev_min = df["revenue_eur"].min()
    if rev_min < 0:
        y_lim = max(abs(rev_max), abs(rev_min)) * 1.1
        ax.set_ylim(-y_lim, y_lim)
        ax.axhline(0, linewidth=0.8, linestyle="--", color="grey")

    # Line: cumulative revenue
    ax2 = ax.twinx()
    ax2.plot(
        df["date"],
        df["cum_revenue"],
        color=line_color,
        linewidth=2.5,
        label="Cumulative revenue",
    )
    ax2.set_ylabel("Cumulative revenue [€]")
    ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

    # X-axis formatting — show “DD Mon” and rotate like the day-ahead plot
    ax.xaxis.set_major_formatter(DateFormatter("%d %b"))
    ax.tick_params(axis="x", rotation=-45)

    # Title in the same style: “Daily & Cumulative Revenue – 01 Jan 2025 – 31 Jan 2025”
    start, end = df["date"].iloc[0], df["date"].iloc[-1]
    ax.set_title(
        f"Daily & Cumulative Revenue – {start:%d %b %Y} – {end:%d %b %Y} {title_suffix}"
    )

    # Combined legend (upper-left, frame-free to match price plot)
    handles, labels = [], []
    for axis in (ax, ax2):
        h, l = axis.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    ax.legend(handles, labels, loc="upper left", frameon=False)

    fig.tight_layout()
    return fig
