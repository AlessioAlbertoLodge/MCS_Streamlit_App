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
    "This page computes the **minimum storage energy (kWh)** required to fully meet demand "
    "under a prescribed **grid import limit**. Resolution: **Δt = 15 minutes**."
)

# ───────────────────────── Inputs — Demand section ─────────────────────────
st.subheader("Demand setup (15-minute horizon)")

# Choice: generate vs load CSV
mode = st.radio(
    "Choose demand source:",
    ["Generate synthetic demand", "Load from CSV"],
    help=(
        "You can either generate a randomised step-like demand using the peak/average parameters, "
        "or upload a CSV file containing two columns: **time** and **power (kW)**. "
        "The time column can be expressed in hours or minutes."
    ),
)

if mode == "Generate synthetic demand":
    # Duration/shape parameters
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

    # Dummy params for generator
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

    demand = generate_step_demand(p_for_demand)

else:
    # Load CSV demand
    uploaded_file = st.file_uploader(
        "Upload demand CSV (two columns: time, power [kW])",
        type=["csv"],
        help="Expected format: two columns → time (in hours or minutes) and power (kW).",
    )
    time_unit = st.selectbox("Time unit in first column:", ["hours", "minutes"])

    demand = None
    dt_hours = 0.25  # default assumption
    if uploaded_file is not None:
        try:
            time_vec, demand = load_demand_csv(uploaded_file, time_unit)
            dt_est = np.median(np.diff(time_vec))
            dt_hours = dt_est if dt_est > 0 else 0.25
            st.success(f"Demand file loaded successfully ({len(demand)} points, Δt ≈ {dt_hours:.3f} h).")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

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
if demand is not None and len(demand) > 0:
    max_dem = compute_max_demand(demand)
    st.markdown("#### Demand preview")
    st.plotly_chart(make_demand_figure(demand), use_container_width=True)
    st.caption(f"Max demand: **{max_dem:,.1f} kW**  •  Steps: **{len(demand)}**  •  Δt: **{dt_hours:.3f} h**")
else:
    st.info("Provide parameters or upload a CSV to display the demand curve.")

# ───────────────────────── Sizing section ─────────────────────────
st.markdown("---")
st.subheader("Storage sizing under a grid limit")

grid_limit_kw = st.number_input("Grid limit [kW]", min_value=0.0, value=900.0, step=50.0)

if demand is not None and len(demand) > 0:
    p_dis_max = derive_discharge_cap_from_grid_limit(demand, grid_limit_kw)
    st.caption(
        f"Prescribed discharge cap = max(demand) − grid limit = **{p_dis_max:,.1f} kW** (clamped ≥ 0)."
    )

    run = st.button("▶️ Size minimum storage (prioritise zero unmet)")

    if run:
        min_energy_kwh, sol, unmet_kwh = find_min_storage_energy_bisect(
            demand_kw=demand,
            grid_limit_kw=grid_limit_kw,
            p_dis_max=p_dis_max,
            p_ch_max=p_dis_max,
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

        duration_h = storage_duration_hours(min_energy_kwh, p_dis_max)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Max demand (kW)", f"{max_dem:,.1f}")
        with k2:
            st.metric("Grid limit (kW)", f"{grid_limit_kw:,.1f}")
        with k3:
            st.metric("Discharge cap (kW)", f"{p_dis_max:,.1f}")
        with k4:
            st.metric("Min energy (kWh)", f"{min_energy_kwh:,.2f}")

        st.write(
            f"<small><b>Storage duration:</b> {duration_h:,.2f} h (energy / discharge)</small>",
            unsafe_allow_html=True,
        )
        st.write(
            f"<small><b>Unmet energy:</b> {unmet_kwh:,.6f} kWh</small>",
            unsafe_allow_html=True,
        )

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
                title="Dispatch with minimal storage (15-min steps)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Grid limit ≥ max demand → no storage required (0 kWh).")
else:
    st.warning("No valid demand data available for sizing.")
