"""
Utility helpers for building a Day‑Ahead‑price dataframe
and plotting it.

Dependencies:
    pandas
    matplotlib
"""

from __future__ import annotations
import datetime as dt
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter


def create_day_ahead_price_df(
    country: str,
    date: dt.date,
    hourly_prices: list[float],
) -> pd.DataFrame:
    """
    Build a tidy dataframe with one row per hour.

    Columns
    -------
    time      : datetime64[ns] – hours 0‑23 on *date*
    price     : float          – €/MWh
    country   : string
    """

    if len(hourly_prices) != 24:
        raise ValueError("hourly_prices must contain exactly 24 values")

    times = [dt.datetime.combine(date, dt.time(hour=h)) for h in range(24)]

    df = pd.DataFrame(
        {"time": times, "price": hourly_prices}
    ).assign(country=country)

    return df


def plot_day_ahead_prices(
    price_df: pd.DataFrame,
    country: str | None = None,
    date: dt.date | None = None,
    *,
    figsize: tuple[int, int] = (11, 6),
    title_suffix: str = "",
    bar_width: float = 0.02,  # thinner default columns
):
    """
    Bar‑plot of Day‑Ahead prices.

    Supply either *country* and *date* explicitly or leave them as
    None – they will be inferred from the dataframe if present.

    Extra Parameters
    ----------------
    bar_width : float, default 0.02
        Width of the hourly bars (useful for making columns visually thinner).
    """
    print(price_df)
    # ── infer labels ──────────────────────────────────────────────────────────
    if country is None:
        country = price_df.get("country", ["Unknown"]).iloc[0]
    if date is None:
        # take the date part of the first timestamp
        date = price_df["time"].dt.date.iloc[0]

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(price_df["time"], price_df["price"], width=bar_width, alpha=0.75, color="black")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Price (€/MWh)")

    ax.set_title(
        f"Day‑Ahead Market Prices – {country} – {date:%d %b %Y} {title_suffix}"
    )

    # centre the y‑axis around 0 so negative hours stand out
    price_range = max(abs(price_df["price"].max()), abs(price_df["price"].min())) * 1.1
    ax.set_ylim(-price_range, price_range)
    ax.axhline(0, linewidth=0.8, linestyle="--", color="grey")

    # show only HH:MM on the x‑axis (no date)
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    ax.tick_params(axis="x", rotation=-45)

    plt.tight_layout()
    plt.show()

    return fig


def plot_day_ahead_prices_normal_and_forecasted(
    price_known_df: pd.DataFrame,
    price_forecast_df: pd.DataFrame,
    country: str | None = None,
    date: dt.date | None = None,
    *,
    figsize: tuple[int, int] = (11, 6),
    title_suffix: str = "",
    bar_width: float = 0.02,  # thinner default columns
):
    """
    Bar‑plot of Day‑Ahead prices.

    Supply either *country* and *date* explicitly or leave them as
    None – they will be inferred from the dataframe if present.

    Extra Parameters
    ----------------
    bar_width : float, default 0.02
        Width of the hourly bars (useful for making columns visually thinner).
    """
    print(price_known_df)
    # ── infer labels ──────────────────────────────────────────────────────────
    if country is None:
        country = price_known_df.get("country", ["Unknown"]).iloc[0]
    if date is None:
        # take the date part of the first timestamp
        date = price_known_df["time"].dt.date.iloc[0]

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(price_known_df["time"], price_known_df["price"],
        width=bar_width, alpha=0.6, color="lightgrey")
    ax.bar(price_forecast_df["time"], price_forecast_df["price"],
       width=bar_width, alpha=0.8, color="crimson")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Price (€/MWh)")

    ax.set_title(
        f"Day‑Ahead Market Prices – {country} – {date:%d %b %Y} {title_suffix}"
    )

    # centre the y‑axis around 0 so negative hours stand out
    price_range = max(abs(price_known_df["price"].max()), abs(price_known_df["price"].min())) * 1.1
    ax.set_ylim(-price_range, price_range)
    ax.axhline(0, linewidth=0.8, linestyle="--", color="grey")

    # show only HH:MM on the x‑axis (no date)
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    ax.tick_params(axis="x", rotation=-45)

    plt.tight_layout()
    plt.show()

    return fig
