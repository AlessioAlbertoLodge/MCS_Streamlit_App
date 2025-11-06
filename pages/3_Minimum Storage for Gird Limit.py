# pages/2_Minimum Storage for Grid Limit.py
from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd

from helper_functions import SystemParams, generate_step_demand
from helper_functions_2 import (
    make_demand_figure,
    compute_max_demand,
    derive_discharge_cap_from_grid_limit,
    find_min_storage_energy_bisect,
    storage_duration_hours,
)
from helper_functions_3 import load_demand_csv

# ───────────────────────── Page setup ─────────────────────────
st.set_page_config(page_title="Min Storage for Grid Limit (15-min)", page_icon="📐", layout="wide")
st.title("📐 Minimum Storage Sizing — Grid Limit (15-minute resolution)")

st.markdown(
    "Compute the **minimum storage energy** required to satisfy demand under a **grid import limit**. "
    "Internally we use hours (solver), but we infer and display **Δt in minutes** and plot **time in hours**."
)

# ───────────────────────── Helpers ─────────────────────────
def _snap_dt_hours(dt_h_raw: float) -> float:
    """Snap a raw dt (hours) to common intervals to avoid 0.025h-type artifacts."""
    if dt_h_raw <= 0:
        return 0.25
    common = np.array([1.0, 0.5, 0.25, 1/6, 0.1, 1/12, 1/30, 1/60], dtype=float)  # 60,30,15,10,6,5,2,1 min
    idx = np.argmin(np.abs(common - dt_h_raw))
    return float(common[idx])

def _infer_dt_and_duration(time_vec_hours: np.ndarray | None, n_points: int) -> tuple[float, float]:
    """
    Return (dt_hours, total_hours). Prefer diffs from time vector; else assume uniform bins over 24h.
    """
    if time_vec_hours is not None and len(time_vec_hours) >= 2:
        diffs = np.diff(time_vec_hours.astype(float))
        diffs = diffs[diffs > 0]
        if diffs.size:
            dt_h = float(np.median(diffs))
            dt_h = _snap_dt_hours(dt_h)
            total_h = dt_h * n_points
            return dt_h, float(total_h)
    # Fallback: assume 24h horizon if not provided
    dt_h = _snap_dt_hours(24.0 / max(1, n_points))
    return dt_h, float(dt_h * n_points)

def _hour_tick_labels(n: int, dt_hours: float, num_ticks: int = 6):
    """Build tick positions (index space) & labels (HH:MM) for a Plotly figure over indices 0..n-1."""
    total_h = dt_hours * n
    hours = np.linspace(0, total_h, num=num_ticks)
    # positions in index space
    idx_pos = np.clip((hours / dt_hours).round().astype(int), 0, max(0, n-1))
    labels = [f"{int(h):02d}:{int(round((h - int(h))*60)):02d}" for h in hours]
    return idx_pos.tolist(), labels

# ───────────────────────── Inputs — Demand section ─────────────────────────
st.subheader("Demand setup")

mode = st.radio(
    "Choose demand source:",
    ["Generate synthetic demand", "Load from CSV"],
    help=(
        "Generate a step-like demand, or upload a CSV.\n"
        "CSV accepted formats:\n"
        "• Two columns: time, power[kW] (this page treats time as **minutes**)\n"
        "• Per-location avg-load: Time, AvgLoadNorm_<year> (0..1) or AvgLoad_<year>_kW"
    ),
)

demand = None                  # ndarray/list
time_vec_hours = None          # ndarray or None (used for dt inference)
using_normalized = False       # True if series is per-unit (0..1)

if mode == "Generate synthetic demand":
    c0a, c0b = st.columns([1, 1])
    with c0a:
        horizon_hours = st.number_input("Horizon [hours]", min_value=1, max_value=168, value=24, step=1)
    with c0b:
        st.text_input("Time step", value="15 minutes", disabled=True)

    c1a, c1b, c1c = st.columns(3)
    with c1a:
        peak_kw = st.number_input("Peak demand [kW]", min_value=0.0, value=1200.0, step=50.0)
    with c1b:
        avg_to_peak_ratio = st.slider("Avg / Peak ratio", 0.1, 1.0, 0.70, 0.01)
    with c1c:
        seed = st.number_input("Random seed", min_value=0, value=7, step=1)

    steps = int(horizon_hours * 4)  # 15-min steps
    p_for_demand = SystemParams(
        hours=steps,
        grid_limit_kw=0.0,
        storage_max_discharge_kw=0.0,
        storage_max_charge_kw=0.0,
        usable_nominal_energy_kwh=0.0,
        eta_charge=1.0,
        eta_discharge=1.0,
        initial_soe=1.0,
        final_soe=1.0,
        peak_kw=peak_kw,
        avg_to_peak_ratio=avg_to_peak_ratio,
        seed=seed,
    )
    demand = np.asarray(generate_step_demand(p_for_demand), dtype=float)  # kW
    # synth time vector in hours for nicer dt inference/plot labels
    time_vec_hours = np.arange(steps, dtype=float) * 0.25
    using_normalized = False

else:
    uploaded_file = st.file_uploader(
        "Upload demand CSV",
        type=["csv"],
        help=(
            "Two-column CSV: time, power[kW] (time assumed **minutes** here), "
            "or per-location avg-load with Time + AvgLoadNorm_<year> or AvgLoad_<year>_kW."
        ),
    )

    if uploaded_file is not None:
        try:
            # Treat generic two-column time as **minutes**; prefer normalized if present
            time_vec_h, vals = load_demand_csv(
                uploaded_file,
                time_unit="minutes",
                prefer_normalized=True,
                prefer_years=[2021, 2020],
            )
            time_vec_hours = np.asarray(time_vec_h, dtype=float)
            demand = np.asarray(vals, dtype=float)

            dmax = float(np.nanmax(demand)) if demand.size else 0.0
            dmin = float(np.nanmin(demand)) if demand.size else 0.0
            using_normalized = (dmax > 0) and (abs(dmax - 1.0) <= 1e-3) and (dmin >= -1e-6)

            if using_normalized:
                st.success(f"Loaded normalized series: {len(demand)} points.")
            else:
                st.success(f"Loaded absolute series: {len(demand)} points.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# ───────────────────────── Normalize → Absolute toggle (if applicable) ───────
grid_limit_kw = None   # absolute limit
grid_limit_pu = None   # per-unit limit
energy_units = "kWh"   # or "PUh"
demand_for_calc = None # list (PU or kW)

if demand is not None and len(demand) > 0:
    if using_normalized:
        st.info("Detected normalized input (values in 0..1).")
        convert_to_abs = st.checkbox(
            "Convert normalized series to absolute kW (enter site max power)",
            value=False,
            help="If checked, scale 0..1 to kW by specifying the site's max (peak) power.",
        )
        if convert_to_abs:
            site_max_kw = st.number_input("Site max power [kW]", min_value=0.0, value=1000.0, step=10.0)
            demand_for_calc = (demand * site_max_kw).tolist()
            using_normalized = False
            energy_units = "kWh"
        else:
            demand_for_calc = demand.tolist()
            energy_units = "PUh"
    else:
        demand_for_calc = demand.tolist()
        energy_units = "kWh"

# ───────────────────────── Efficiencies & SoE policy ─────────────────────────
st.markdown("---")
st.subheader("Storage and efficiency parameters")

c2a, c2b, c2c = st.columns(3)
with c2a:
    eta_ch = st.number_input("η_charge", min_value=0.5, max_value=1.0, value=1.0, step=0.01)
with c2b:
    eta_dis = st.number_input("η_discharge", min_value=0.5, max_value=1.0, value=1.0, step=0.01)
with c2c:
    init_soe_pct = st.slider("Initial SoE assumed for sizing [%]", 0, 100, 100, 1)

# ───────────────────────── Demand preview ─────────────────────────
if demand_for_calc is not None and len(demand_for_calc) > 0:
    # Infer dt and duration (in hours); display Δt in minutes
    dt_hours, total_hours = _infer_dt_and_duration(time_vec_hours, len(demand_for_calc))
    dt_minutes = dt_hours * 60.0

    max_dem = compute_max_demand(demand_for_calc)
    st.markdown("#### Demand preview")
    fig = make_demand_figure(demand_for_calc)  # returns a Plotly fig; x-axis currently in index (steps)

    # Relabel x-axis to hours
    tickvals, ticktext = _hour_tick_labels(len(demand_for_calc), dt_hours, num_ticks=6)
    try:
        fig.update_xaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            title_text="Time [h]"
        )
    except Exception:
        pass

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Points: **{len(demand_for_calc)}**  •  Δt: **{dt_minutes:.0f} min**  •  Total: **{total_hours:.1f} h**  •  "
        f"Max: **{max_dem:,.3f} {'PU' if energy_units=='PUh' else 'kW'}**"
    )
else:
    st.info("Provide parameters or upload a CSV to display the demand curve.")

# ───────────────────────── Grid limit input ──────────────────────────────────
st.markdown("---")
st.subheader("Storage sizing under a grid limit")

if demand_for_calc is not None and len(demand_for_calc) > 0:
    # Reuse the same dt
    dt_hours, total_hours = _infer_dt_and_duration(time_vec_hours, len(demand_for_calc))

    if energy_units == "PUh":
        mode_grid = st.radio(
            "Grid limit input (normalized):",
            ["Percentage of max (0–100%)", "Per-unit (0–1)"],
            horizontal=True,
        )
        if mode_grid.startswith("Percentage"):
            pct = st.slider("Grid limit [% of max]", 0.0, 100.0, 70.0, 0.5) / 100.0
            grid_limit_pu = float(pct)
        else:
            grid_limit_pu = st.number_input("Grid limit [PU]", min_value=0.0, max_value=1.0, value=0.70, step=0.01)

        grid_limit_val = grid_limit_pu
        grid_label = "PU"
    else:
        mode_grid = st.radio(
            "Grid limit input (absolute):",
            ["Absolute [kW]", "Percentage of peak [%]"],
            horizontal=True,
        )
        if mode_grid.startswith("Absolute"):
            grid_limit_kw = st.number_input("Grid limit [kW]", min_value=0.0, value=900.0, step=50.0)
        else:
            pct = st.slider("Grid limit [% of peak]", 0.0, 100.0, 70.0, 0.5) / 100.0
            grid_limit_kw = float(pct * compute_max_demand(demand_for_calc))

        grid_limit_val = grid_limit_kw
        grid_label = "kW"

    # ── Compute discharge cap and run sizing ─────────────────────────────────
    p_dis_max = derive_discharge_cap_from_grid_limit(demand_for_calc, grid_limit_val)
    st.caption(
        f"Prescribed discharge cap = max(demand) − grid limit = "
        f"**{p_dis_max:,.3f} {grid_label}** (clamped ≥ 0)."
    )

    run = st.button("▶️ Size minimum storage (prioritise zero unmet)")

    if run:
        min_energy, sol, unmet = find_min_storage_energy_bisect(
            demand_kw=demand_for_calc,
            grid_limit_kw=grid_limit_val,
            p_dis_max=p_dis_max,
            p_ch_max=p_dis_max,
            eta_ch=eta_ch,
            eta_dis=eta_dis,
            init_soe_frac=init_soe_pct / 100.0,
            final_soe_frac=None,
            dt_hours=dt_hours,              # ← robust, snapped to minutes
            tol_kwh=0.05,
            max_iter=60,
            unmet_penalty=1e12,
            fill_bias_weight=1e-4,
            move_penalty=1e-4,
            add_safety_buffer=True,
        )

        duration_h = storage_duration_hours(min_energy, p_dis_max)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric(f"Max demand ({grid_label})", f"{compute_max_demand(demand_for_calc):,.3f}")
        with k2:
            st.metric(f"Grid limit ({grid_label})", f"{grid_limit_val:,.3f}")
        with k3:
            st.metric(f"Discharge cap ({grid_label})", f"{p_dis_max:,.3f}")
        with k4:
            st.metric(f"Min energy ({'PUh' if energy_units=='PUh' else 'kWh'})", f"{min_energy:,.4f}")

        st.write(
            f"<small><b>Storage duration:</b> {duration_h:,.2f} h (energy / discharge)</small>",
            unsafe_allow_html=True,
        )
        st.write(
            f"<small><b>Unmet energy:</b> {unmet:,.6f} {'PUh' if energy_units=='PUh' else 'kWh'}</small>",
            unsafe_allow_html=True,
        )

        if p_dis_max > 0:
            st.markdown("#### Dispatch check at sized energy")
            from helper_functions import make_main_dispatch_figure
            fig2 = make_main_dispatch_figure(
                demand=sol["demand"],
                grid=sol["grid"],
                storage_discharge=sol["p_dis"],
                storage_charge=sol["p_ch"],
                unmet=sol["unmet"],
                grid_limit_kw=sol["grid_limit_kw"],
                title=f"Dispatch with minimal storage ({'PU' if energy_units=='PUh' else 'kW'}, {int(round(dt_hours*60))}-min steps)",
            )
            # Relabel x to hours here too
            try:
                tickvals2, ticktext2 = _hour_tick_labels(len(sol["demand"]), dt_hours, num_ticks=6)
                fig2.update_xaxes(
                    tickmode="array",
                    tickvals=tickvals2,
                    ticktext=ticktext2,
                    title_text="Time [h]"
                )
            except Exception:
                pass
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Grid limit ≥ max demand → no storage required (0 units).")
else:
    st.warning("No valid demand data available for sizing.")
