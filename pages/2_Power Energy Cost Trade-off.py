# pages/power-energy-cost-tradeoff.py
from __future__ import annotations

import numpy as np
import streamlit as st

from helper_functions import SystemParams, generate_step_demand, make_main_dispatch_figure
from helper_functions_2 import (
    make_demand_figure,
    compute_max_demand,
    evaluate_grid_share_tradeoff,
)

# ───────────────────────── Page setup ─────────────────────────
st.set_page_config(page_title="Power/Energy Cost Trade-off (15-min)", page_icon="💰", layout="wide")
st.title("💰 Power–Energy Cost Trade-off — 15-minute Resolution")

st.markdown(
    "Choose **grid share** (grid limit as a fraction of the demand peak). For each grid share `r`, "
    "we set `grid_limit = r · max_demand` and the **discharge cap** becomes `max_demand − grid_limit`. "
    "We then find the **minimal energy** that yields **zero unmet** and compute the total cost:\n\n"
    "**Cost = (USD/kW · discharge_cap) + (USD/kWh · energy)**.\n\n"
    "The app scans a range of `r` values and selects the **cheapest** configuration."
)

# ───────────────────────── Demand first (15-min) ─────────────────────────
st.subheader("Demand setup")

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

# Efficiencies / initial SoE (used by LP during sizing)
c2a, c2b, c2c = st.columns(3)
with c2a:
    eta_ch = st.number_input("η_charge", min_value=0.5, max_value=1.0, value=1.0, step=0.01)
with c2b:
    eta_dis = st.number_input("η_discharge", min_value=0.5, max_value=1.0, value=1.0, step=0.01)
with c2c:
    init_soe_pct = st.slider("Initial SoE assumed for sizing [%]", 0, 100, 100, 1)

# Build demand now
steps = int(horizon_hours * 4)
dt_hours = 0.25
p_for_demand = SystemParams(
    hours=steps,
    grid_limit_kw=0.0,
    storage_max_discharge_kw=0.0,
    storage_max_charge_kw=0.0,
    usable_nominal_energy_kwh=0.0,
    eta_charge=eta_ch,
    eta_discharge=eta_dis,
    initial_soe=init_soe_pct / 100.0,
    final_soe=None,
    peak_kw=peak_kw,
    avg_to_peak_ratio=avg_to_peak_ratio,
    seed=seed,
)
demand = generate_step_demand(p_for_demand)
max_dem = compute_max_demand(demand)

st.markdown("#### Demand preview")
st.plotly_chart(make_demand_figure(demand), use_container_width=True)
st.caption(f"Max demand: **{max_dem:,.1f} kW**  •  Steps: **{steps}**  •  Δt: **{dt_hours} h**")

# ───────────────────────── Cost model & grid-share sweep ─────────────────────────
st.markdown("---")
st.subheader("Cost model and grid-share sweep")

c3a, c3b, c3c = st.columns(3)
with c3a:
    c_power = st.number_input("Specific power cost (USD/kW)", min_value=0.0, value=100.0, step=10.0)
with c3b:
    c_energy = st.number_input("Specific energy cost (USD/kWh)", min_value=0.0, value=100.0, step=10.0)
with c3c:
    st.text_input("Unmet penalty (LP)", value="1e12 (fixed, very large)", disabled=True)

c4a, c4b, c4c = st.columns(3)
with c4a:
    grid_share_max = st.slider("Max grid share r_max", 0.0, 1.0, 0.80, 0.01)
with c4b:
    grid_share_min = st.slider("Min grid share r_min", 0.0, 1.0, 0.30, 0.01)
with c4c:
    step_pct = st.slider("Grid share step (%)", 1, 20, 5, 1)

if grid_share_min > grid_share_max:
    st.error("r_min must be ≤ r_max.")
    st.stop()

grid_shares = list(np.round(np.arange(grid_share_min, grid_share_max + 1e-9, step_pct / 100.0), 4))

run = st.button("▶️ Evaluate trade-off and pick cheapest")

# ───────────────────────── Run sweep ─────────────────────────
if run:
    best, rows = evaluate_grid_share_tradeoff(
        demand_kw=demand,
        grid_share_values=grid_shares,
        eta_ch=eta_ch,
        eta_dis=eta_dis,
        init_soe_frac=init_soe_pct / 100.0,
        dt_hours=dt_hours,
        c_power_usd_per_kw=c_power,
        c_energy_usd_per_kwh=c_energy,
        # unmet penalty / weights left as defaults (very conservative)
    )

    # KPIs for the best design
    st.subheader("Optimal configuration (minimum cost)")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Grid share r*", f"{best['grid_share']*100:,.1f}%")
    with k2:
        st.metric("Grid limit (kW)", f"{best['grid_limit_kw']:,.1f}")
    with k3:
        st.metric("Discharge cap (kW)", f"{best['p_dis_max_kw']:,.1f}")
    with k4:
        st.metric("Min energy (kWh)", f"{best['energy_kwh']:,.2f}")

    k5, k6, k7 = st.columns(3)
    with k5:
        st.metric("Duration (h)", f"{best['duration_h']:,.2f}")
    with k6:
        st.metric("Unmet energy (kWh)", f"{best['unmet_kwh']:,.6f}")
    with k7:
        st.metric("Total cost (USD)", f"{best['total_cost_usd']:,.2f}")

    # Stacked dispatch for the best
    st.markdown("#### Dispatch for the chosen design")
    sol = best["sol"]
    fig = make_main_dispatch_figure(
        demand=sol["demand"],
        grid=sol["grid"],
        storage_discharge=sol["p_dis"],
        storage_charge=sol["p_ch"],
        unmet=sol["unmet"],
        grid_limit_kw=sol["grid_limit_kw"],
        title="Dispatch at optimal power–energy sizing (15-min)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Optional: simple table of candidates
    with st.expander("See all evaluated candidates"):
        import pandas as pd
        tbl = pd.DataFrame([
            {
                "grid_share": r["grid_share"],
                "grid_limit_kw": r["grid_limit_kw"],
                "p_dis_max_kw": r["p_dis_max_kw"],
                "energy_kwh": r["energy_kwh"],
                "duration_h": r["duration_h"],
                "unmet_kwh": r["unmet_kwh"],
                "total_cost_usd": r["total_cost_usd"],
            }
            for r in rows
        ]).sort_values("total_cost_usd")
        st.dataframe(tbl, use_container_width=True)
