# pages/Main_Dispatch.py
from __future__ import annotations

import streamlit as st
from helper_functions import (
    SystemParams,
    generate_step_demand,
    build_and_solve_lp,
    make_main_dispatch_figure,
    make_dashboard_figure,
)

# ───────────────────────── Page setup ─────────────────────────
st.set_page_config(page_title="Grid + Storage (15-min)", page_icon="⏱️", layout="wide")
st.title("⏱️ Grid + Storage — 15-minute Resolution")

st.markdown(
    "This page solves the **same storage dispatch LP** as the main app, but with a "
    "**15-minute time step (Δt = 0.25 h)**. Inputs are on this page (no sidebar)."
)

# ───────────────────────── Inputs (on page) ─────────────────────────
st.subheader("Inputs")

# Duration/Resolution
c0a, c0b = st.columns([1, 1])
with c0a:
    horizon_hours = st.number_input("Horizon [hours]", min_value=1, max_value=168, value=24, step=1)
with c0b:
    st.text_input("Time step", value="15 minutes (fixed)", disabled=True)

# Grid + Storage limits
c1a, c1b, c1c = st.columns(3)
with c1a:
    grid_limit_kw = st.number_input("Grid limit [kW]", min_value=0.0, value=900.0, step=50.0)
with c1b:
    p_dis_max = st.number_input("Storage max discharge [kW]", min_value=0.0, value=500.0, step=50.0)
with c1c:
    p_ch_max = st.number_input("Storage max charge [kW]", min_value=0.0, value=500.0, step=50.0)

# Energy + efficiencies
c2a, c2b, c2c = st.columns(3)
with c2a:
    usable_energy = st.number_input("Usable nominal energy [kWh]", min_value=0.0, value=3000.0, step=100.0)
with c2b:
    eta_ch = st.number_input("η_charge", min_value=0.5, max_value=1.0, value=1.0, step=0.01)
with c2c:
    eta_dis = st.number_input("η_discharge", min_value=0.5, max_value=1.0, value=1.0, step=0.01)

# SoE
c3a, c3b = st.columns([1, 2])
with c3a:
    init_soe_pct = st.slider("Initial SoE [%]", 0, 100, 100, 1)
with c3b:
    enforce_final = st.checkbox("Enforce Final SoE?", value=False)
    final_soe_pct = st.slider(
        "Final SoE [%]", 0, 100, 100, 1, disabled=not enforce_final
    )

# Demand shape
st.markdown("---")
st.subheader("Demand shape")
c4a, c4b, c4c = st.columns(3)
with c4a:
    peak_kw = st.number_input("Peak demand [kW]", min_value=0.0, value=1200.0, step=50.0)
with c4b:
    avg_to_peak_ratio = st.slider("Avg / Peak ratio", 0.1, 1.0, 0.70, 0.01)
with c4c:
    seed = st.number_input("Random seed", min_value=0, value=7, step=1)

# Note about hidden objective weights (kept at defaults)
st.caption(
    "Objective weights are not shown here. Defaults in the LP are used: "
    "unmet_penalty = 1e8, fill_bias_weight = 1e-2, move_penalty = 0.1."
)

# Run
run = st.button("▶️ Solve (15-min)")

# ───────────────────────── Run model ─────────────────────────
if run:
    # Convert to 15-minute resolution
    steps = int(horizon_hours * 4)          # number of intervals
    dt_hours = 0.25                         # 15 minutes

    # Build params: IMPORTANT — use steps for SystemParams.hours
    p = SystemParams(
        hours=steps,
        grid_limit_kw=grid_limit_kw,
        storage_max_discharge_kw=p_dis_max,
        storage_max_charge_kw=p_ch_max,
        usable_nominal_energy_kwh=usable_energy,
        eta_charge=eta_ch,
        eta_discharge=eta_dis,
        initial_soe=init_soe_pct / 100.0,
        final_soe=(final_soe_pct / 100.0) if enforce_final else None,
        unmet_penalty=1e8,
        fill_bias_weight=1e-2,
        move_penalty=0.1,
        peak_kw=peak_kw,
        avg_to_peak_ratio=avg_to_peak_ratio,
        seed=seed,
    )

    # Generate demand at 15-min step count
    demand = generate_step_demand(p)

    # Solve LP at 15-min Δt
    grid, p_dis, p_ch, unmet, soe = build_and_solve_lp(
        demand_kw=demand,
        grid_limit_kw=p.grid_limit_kw,
        storage_max_discharge_kw=p.storage_max_discharge_kw,
        storage_max_charge_kw=p.storage_max_charge_kw,
        usable_nominal_energy_kwh=p.usable_nominal_energy_kwh,
        eta_charge=p.eta_charge,
        eta_discharge=p.eta_discharge,
        initial_soe=p.initial_soe,
        final_soe=p.final_soe,
        dt_hours=dt_hours,                      # ← 15-minute time step
        unmet_penalty=p.unmet_penalty,
        fill_bias_weight=p.fill_bias_weight,
        move_penalty=p.move_penalty,
    )

    # ── Plots
    st.subheader("📊 Dispatch vs Demand (Stacked) — 15-min")
    fig_dispatch = make_main_dispatch_figure(
        demand, grid, p_dis, p_ch, unmet, p.grid_limit_kw,
        title="Main Dispatch (Grid first, then Storage) — 15-minute",
    )
    st.plotly_chart(fig_dispatch, use_container_width=True)

    with st.expander("More diagnostics (duration curves, histogram, SoE)"):
        fig_dash = make_dashboard_figure(
            demand, grid, p_dis, p_ch, unmet, soe, p.grid_limit_kw,
            title="Grid Connection with Storage — Dashboard (15-minute)",
        )
        st.plotly_chart(fig_dash, use_container_width=True)
