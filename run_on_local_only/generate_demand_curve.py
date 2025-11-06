# generate_fake_cs_demand_minutes.py
# Python 3.9+ compatible

import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Config ----------
OUT_DIR = r"C:\Users\lodgeaa\Desktop\FlexMCS_Streamlit\MCS_Streamlit_App\run_on_local_only"
total_duration_minutes = 24 * 60      # 24h
resolution_minutes = 15               # <-- default resolution; change if desired
baseline_kw = 2000
seed = None                           # set e.g. 42 for reproducibility, or None for random
noise_std_kw = 120
# --------------------------------

def make_oscillatory_series(minutes_array, baseline=2000, noise_std=120, rng=None):
    """
    Build a 'very wiggly' signal around baseline using multiple sinusoids + noise.
    minutes_array: float minutes from 0 to 1440.
    Internally we convert to hours for oscillation periods.
    """
    if rng is None:
        rng = np.random.default_rng()

    t_min = minutes_array.astype(float)
    t_hr = t_min / 60.0

    # Multi-scale oscillations (defined over hours)
    daily = 350 * np.sin(2 * np.pi * t_hr / 24)                 # 1/day
    mid   = 280 * np.sin(2 * np.pi * t_hr / (24/6))             # 6/day
    high  = 180 * np.sin(2 * np.pi * t_hr / (24/18) + 0.7)      # 18/day
    spike = 120 * np.sin(2 * np.pi * t_hr / (24/36) + 1.3)      # 36/day

    # Subtle daytime bump
    daytime_bump = 120 * np.maximum(0, np.sin(2 * np.pi * (t_hr - 6) / 24))

    # Random noise
    noise = rng.normal(0, noise_std, size=t_min.shape)

    power = baseline + daily + mid + high + spike + daytime_bump + noise
    power = np.clip(power, 0, None)
    return power

def main():
    rng = np.random.default_rng(seed)

    # Time axis in MINUTES with chosen resolution
    n_steps = int(total_duration_minutes / resolution_minutes) + 1
    minutes = np.linspace(0, total_duration_minutes, n_steps)

    power_kw = make_oscillatory_series(minutes, baseline_kw, noise_std_kw, rng=rng)

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(OUT_DIR, f"fake_cs_demand_minutes_{ts}.csv")

    # Save CSV (minutes, power_in_kW)
    df = pd.DataFrame({"minutes": minutes, "power_in_kW": power_kw})
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Plot (x-axis in hours for readability)
    hours = minutes / 60.0
    plt.figure(figsize=(10, 4.5))
    plt.plot(hours, power_kw, linewidth=1.2)
    plt.title(f"Fictitious Charging-Station Power Demand (24h, {resolution_minutes}-min resolution)")
    plt.xlabel("Time [hours]")
    plt.ylabel("Power [kW]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
