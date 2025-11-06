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
    "This page computes the **minimum storage energy** required to fully meet demand "
    "under a prescribed **grid import limit**. Resolution: **Δt = 15 minutes**."
)

# ───────────────────────── Inputs — Demand section ─────────────────────────
st.subheader("Demand setup (15-minute horizon)")

mode = st.radio(
    "Choose demand source:",
    ["Generate synthetic demand", "Load from CSV"],
    help=(
        "You can generate a step-like demand from parameters, "
        "or upload a CSV (supports two-column [time, kW] or the per-location avg-load format)."
    ),
)

demand = None
dt_hours = 0.25
using_normalized = False   # True → values are per-unit (0..1), energy unit is PUh

if mode == "Generate synthetic demand":
    c0a, c0b = st.columns([1, 1])
    with c0a:
        horizon_hours = st.number_input("Horizon [hours]", min_value=1, max_value=168, value=24, step=1)
    with c0b:
        st.text_input("Time step", value="15 minutes (Δt = 0.25 h)", disabled=True)

    c1a, c1b, c1c = st.columns(3)
    with c1a:
        peak_kw = st.number_input("Peak demand [kW]", min_value=0.0, value=1200.0, step=50.0)
    with c1b:
        avg_to_peak_ratio = st.slider("Avg / Peak ratio", 0.1, 1.0, 0.70, 0.01)
    with c1c:
        seed = st.number_input("Random seed", min_value=0, value=7, step=1)

    steps = int(horizon_hours * 4)  # 15-min steps
    dt_hours = 0.25

    p_for_demand = SystemParams(
        hours=steps,
        grid_limit_kw=0.0,
        storage_max_discharge_kw=0.0,
        storage_max_charge_kw=0.0,
        usable_nominal_energy_kwh=0.0,
        eta_charge=1.0,
        eta_discharge=1.0,
        initial_soe=1.0,
        final_soe=None,
        peak_kw=peak_kw,
        avg_to_peak_ratio=avg_to_peak_ratio,
        seed=seed,
    )
    demand = generate_step_demand(p_for_demand)  # kW
    using_normalized = False

else:
    uploaded_file = st.file_uploader(
        "Upload demand CSV",
        type=["csv"],
        help=(
            "Accepted formats:\n"
            "• Two columns: time, power[kW] (time assumed in **minutes** here)\n"
            "• Per-location avg-load: Time, AvgLoadNorm_<year> (0..1) or AvgLoad_<year>_kW"
        ),
    )

    # Always interpret generic two-column time as **minutes** for this page
    time_unit_for_generic = "minutes"

    if uploaded_file is not None:
        try:
            # Prefer normalized if available; if not, will fall back to kW
            time_vec_h, vals = load_demand_csv(
                uploaded_file,
                time_unit=time_unit_for_generic,
                prefer_normalized=True,
                prefer_years=[2021, 2020],
            )
            # Estimate Δt
            diffs = np.diff(time_vec_h)
            dt_est = float(np.median(diffs[diffs > 0])) if diffs.size else 0.25
            dt_hours = dt_est if dt_est > 0 else 0.25

            demand = np.asarray(vals, dtype=float)

            # Detect normalization: max ≈ 1 and min ≥ 0
            dmax = np.nanmax(demand) if demand.size else 0.0
            dmin = np.nanmin(demand) if demand.size else 0.0
            using_normalized = (dmax > 0) and (abs(dmax - 1.0) <= 1e-3) and (dmin >= -1e-6)

            if using_normalized:
                st.success(f"Loaded normalized series ({len(demand)} pts, Δt ≈ {dt_hours:.3f} h). Max ≈ 1 detected.")
            else:
                st.success(f"Loaded absolute series ({len(demand)} pts, Δt ≈ {dt_hours:.3f} h).")

        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# ───────────────────────── Normalize → Absolute toggle (if applicable) ───────
grid_limit_kw = None              # absolute kW grid limit (if applicable)
grid_limit_pu = None              # per-unit 0..1 grid limit (if normalized)
energy_units = "kWh"              # or "PUh" for normalized mode
max_dem_display = None            # for the preview caption
demand_for_calc = None            # the series we pass to the solver (PU or kW)

if demand is not None and len(demand) > 0:
    if using_normalized:
        st.info("Detected normalized input (values in 0..1).")

        convert_to_abs = st.checkbox(
            "Convert normalized series to absolute kW (enter site max power)",
            value=False,
            help="If checked, you can scale 0..1 to kW by specifying the site's max power.",
        )

        if convert_to_abs:
            site_max_kw = st.number_input("Site max power [kW]", min_value=0.0, value=1000.0, step=10.0)
            demand_for_calc = (demand * site_max_kw).tolist()
            using_normalized = False
            energy_units = "kWh"
            max_dem_display = float(np.max(demand_for_calc))
        else:
            # keep normalized
            demand_for_calc = demand.tolist()
            energy_units = "PUh"
            max_dem_display = float(np.max(demand))

    else:
        # already absolute (kW)
        demand_for_calc = demand.tolist()
        energy_units = "kWh"
        max_dem_display = float(np.max(demand))

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
    max_dem = compute_max_demand(demand_for_calc)
    st.markdown("#### Demand preview")
    st.plotly_chart(make_demand_figure(demand_for_calc), use_container_width=True)
    st.caption(f"Max demand: **{max_dem:,.3f} {'PU' if energy_units=='PUh' else 'kW'}**  •  "
               f"Steps: **{len(demand_for_calc)}**  •  Δt: **{dt_hours:.3f} h**")
else:
    st.info("Provide parameters or upload a CSV to display the demand curve.")

# ───────────────────────── Grid limit input ──────────────────────────────────
st.markdown("---")
st.subheader("Storage sizing under a grid limit")

if demand_for_calc is not None and len(demand_for_calc) > 0:
    if energy_units == "PUh":
        # normalized mode: grid limit as % or per-unit
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
        # absolute mode: grid in kW or % of peak
        mode_grid = st.radio(
            "Grid limit input (absolute):",
            ["Absolute [kW]", "Percentage of peak [%]"],
            horizontal=True,
        )
        if mode_grid.startswith("Absolute"):
            grid_limit_kw = st.number_input("Grid limit [kW]", min_value=0.0, value=900.0, step=50.0)
        else:
            pct = st.slider("Grid limit [% of peak]", 0.0, 100.0, 70.0, 0.5) / 100.0
            grid_limit_kw = float(pct * max_dem)

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
            demand_kw=demand_for_calc,            # PU or kW
            grid_limit_kw=grid_limit_val,         # PU or kW
            p_dis_max=p_dis_max,                  # PU or kW
            p_ch_max=p_dis_max,                   # symmetric cap
            eta_ch=eta_ch,
            eta_dis=eta_dis,
            init_soe_frac=init_soe_pct / 100.0,
            final_soe_frac=None,
            dt_hours=dt_hours,
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
            st.metric(f"Max demand ({grid_label})", f"{max_dem:,.3f}")
        with k2:
            st.metric(f"Grid limit ({grid_label})", f"{grid_limit_val:,.3f}")
        with k3:
            st.metric(f"Discharge cap ({grid_label})", f"{p_dis_max:,.3f}")
        with k4:
            st.metric(f"Min energy ({energy_units})", f"{min_energy:,.4f}")

        st.write(
            f"<small><b>Storage duration:</b> {duration_h:,.2f} h (energy / discharge)</small>",
            unsafe_allow_html=True,
        )
        st.write(
            f"<small><b>Unmet energy:</b> {unmet:,.6f} {'PUh' if energy_units=='PUh' else 'kWh'}</small>",
            unsafe_allow_html=True,
        )

        # Dispatch plot uses the same units as the calculation (PU or kW)
        if p_dis_max > 0:
            st.markdown("#### Dispatch check at sized energy")
            from helper_functions import make_main_dispatch_figure
            fig = make_main_dispatch_figure(
                demand=sol["demand"],
                grid=sol["grid"],
                storage_discharge=sol["p_dis"],
                storage_charge=sol["p_ch"],
                unmet=sol["unmet"],
                grid_limit_kw=sol["grid_limit_kw"],
                title=f"Dispatch with minimal storage ({'PU' if energy_units=='PUh' else 'kW'}, 15-min steps)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Grid limit ≥ max demand → no storage required (0 units).")
else:
    st.warning("No valid demand data available for sizing.")
