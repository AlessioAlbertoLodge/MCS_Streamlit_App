from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt

def plot_bess_schedule(
    schedule: pd.DataFrame,
    country: str,
    day: dt.date,
    e_nom: float,
    title_prefix: str = ""
):
    # Unpack and scale
    t = pd.to_datetime(schedule["time"])
    price = schedule["price €/kWh"] * 1000  # €/MWh
    power = schedule["power kW"]
    soc = schedule["SOC start"] * e_nom / 100  # → kWh

    charge = np.where(power < 0, power, 0)  # negative bars
    discharge = np.where(power > 0, power, 0)  # positive bars

    revenue_per_hour = price * power * 1e-3  # €/MWh × kW = €/h

    fig, (ax_soc, ax_price) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    full_title = f"{title_prefix}Battery schedule – {country} {day}"
    ax_soc.set_title(full_title)

    # --- Upper panel: SOC and power ---
    ax_soc.plot(t, soc, label="Energy (kWh)", color="black")
    ax_soc.set_ylabel("Energy (kWh)")
    ax_soc.set_ylim(-0.1 * e_nom, 1.1 * e_nom)

    axp = ax_soc.twinx()
    axp.bar(t, charge, width=0.03, alpha=0.6, label="Charge kW")
    axp.bar(t, discharge, width=0.03, alpha=0.6, label="Discharge kW")
    axp.set_ylabel("Power (kW)")
    rng = max(abs(power)) * 1.1
    axp.set_ylim(-rng, rng)

    lines1, labs1 = ax_soc.get_legend_handles_labels()
    lines2, labs2 = axp.get_legend_handles_labels()
    ax_soc.legend(lines1 + lines2, labs1 + labs2, loc="upper right")

    # --- Lower panel: price and revenue ---
    # --- Lower panel: price and revenue (shared axis) ---
    ax_price.axhline(0, color="grey", linestyle="--", linewidth=1)
    
    # Transparent price bars for context
    ax_price.bar(t, price, width=0.03, color="black", alpha=0.2, label="Market Price (€/MWh)")

    # Overlay revenue (€/h) using the same y-axis
    ax_price.bar(t, revenue_per_hour, width=0.03, color="green", alpha=0.7, label="Revenue per Hour (€)")
    
    # Shared label and autoscale
    ax_price.set_ylabel("Price (€/MWh) + Revenue (€)")
    y_max = max(price.max(), revenue_per_hour.max(), abs(revenue_per_hour.min())) * 1.1
    y_min = min(price.min(), revenue_per_hour.min()) * 1.1
    ax_price.set_ylim(y_min, y_max)

    ax_price.legend(loc="upper left")
    ax_price.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    ax_price.set_xlabel("Hour of Day")

    fig.tight_layout()
    return fig
