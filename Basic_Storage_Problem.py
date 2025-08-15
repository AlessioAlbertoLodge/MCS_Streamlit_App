# app.py
from __future__ import annotations

import streamlit as st
from helper_functions import (
    SystemParams,
    generate_step_demand,
    build_and_solve_lp,
    make_main_dispatch_figure,
    make_dashboard_figure,
)


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Grid + Storage Optimizer", page_icon="⚡", layout="wide")
st.title("⚡ Grid + Storage Optimizer (Day Profile)")

with st.sidebar:
    st.header("Inputs")
    hours = st.number_input("Hours", min_value=1, max_value=168, value=24, step=1)
    grid_limit_kw = st.number_input("Grid limit [kW]", min_value=0.0, value=900.0, step=50.0)

    colA, colB = st.columns(2)
    with colA:
        p_dis_max = st.number_input("Storage max discharge [kW]", min_value=0.0, value=500.0, step=50.0)
    with colB:
        p_ch_max = st.number_input("Storage max charge [kW]", min_value=0.0, value=500.0, step=50.0)

    usable_energy = st.number_input("Usable nominal energy [kWh]", min_value=0.0, value=3000.0, step=100.0)

    colC, colD = st.columns(2)
    with colC:
        eta_ch = st.number_input("η_charge", min_value=0.5, max_value=1.0, value=1.0, step=0.01)
    with colD:
        eta_dis = st.number_input("η_discharge", min_value=0.5, max_value=1.0, value=1.0, step=0.01)

    init_soe_pct = st.slider("Initial SoE [%]", 0, 100, 100, 1)

    enforce_final = st.checkbox("Enforce final SoE?", value=False)
    final_soe_pct = st.slider("Final SoE [%]", 0, 100, 100, 1, disabled=not enforce_final)

    st.markdown("---")
    st.subheader("Demand shape")
    peak_kw = st.number_input("Peak demand [kW]", min_value=0.0, value=1200.0, step=50.0)
    avg_to_peak_ratio = st.slider("Avg / Peak ratio", 0.1, 1.0, 0.70, 0.01)
    seed = st.number_input("Random seed", min_value=0, value=7, step=1)

    st.markdown("---")
    st.subheader("Objective weights")
    unmet_penalty = st.number_input("Unmet penalty", min_value=0.0, value=1e8, step=1e6, format="%.0f")
    fill_bias = st.number_input("Fill bias weight", min_value=0.0, value=1e-2, step=1e-3, format="%.4f")
    move_pen = st.number_input("Move penalty", min_value=0.0, value=0.1, step=0.05, format="%.2f")

# Build params object
p = SystemParams(
    hours=hours,
    grid_limit_kw=grid_limit_kw,
    storage_max_discharge_kw=p_dis_max,
    storage_max_charge_kw=p_ch_max,
    usable_nominal_energy_kwh=usable_energy,
    eta_charge=eta_ch,
    eta_discharge=eta_dis,
    initial_soe=init_soe_pct / 100.0,
    final_soe=(final_soe_pct / 100.0) if enforce_final else None,
    unmet_penalty=unmet_penalty,
    fill_bias_weight=fill_bias,
    move_penalty=move_pen,
    peak_kw=peak_kw,
    avg_to_peak_ratio=avg_to_peak_ratio,
    seed=seed,
)

# Generate demand & solve LP
demand = generate_step_demand(p)

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
    dt_hours=1.0,
    unmet_penalty=p.unmet_penalty,
    fill_bias_weight=p.fill_bias_weight,
    move_penalty=p.move_penalty,
)

# Main stacked dispatch plot
st.subheader("📊 Dispatch vs Demand (Stacked)")
fig_dispatch = make_main_dispatch_figure(
    demand, grid, p_dis, p_ch, unmet, p.grid_limit_kw,
    title="Main Dispatch (Grid first, then Storage)",
)
st.plotly_chart(fig_dispatch, use_container_width=True)

# Diagnostics dashboard (optional)
with st.expander("More diagnostics (duration curves, histogram, SoE)"):
    fig_dash = make_dashboard_figure(
        demand, grid, p_dis, p_ch, unmet, soe, p.grid_limit_kw,
        title="Grid Connection with Storage — Dashboard",
    )
    st.plotly_chart(fig_dash, use_container_width=True)
