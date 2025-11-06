# helper_functions.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

"""
helper_functions.py
───────────────────────────────────────────────────────────────────────────────
Core pieces for the Grid + Storage Optimizer: parameters, demand generator,
LP-based dispatch solver, and Plotly visualizations.

What’s here
- SystemParams: dataclass holding all model inputs and weights.
  Units: kW, kWh, hours (time step = dt_hours).
- generate_step_demand(): builds a simple two-level (low/peak) demand profile
  matching a target average (avg_to_peak_ratio × peak_kw).
- build_and_solve_lp(): linear program that dispatches grid + battery to meet
  demand under a grid import limit.
  • Decision vars per step h:
      grid[h]   ≥ 0   (kW)
      p_dis[h]  ≥ 0   (kW)
      p_ch[h]   ≥ 0   (kW)
      unmet[h]  ≥ 0   (kW)
      E[h] ∈ [0, usable_nominal_energy_kwh] (kWh), with E[0] fixed; E[H] optional
  • Power balance: grid + p_dis − p_ch + unmet = demand
  • Energy update: E[h+1] = E[h] + η_ch·p_ch·dt − (p_dis/η_dis)·dt
  • Limits: grid ≤ grid_limit_kw; p_dis ≤ storage_max_discharge_kw;
            p_ch ≤ storage_max_charge_kw
  • Objective (to minimize):
        unmet_penalty · Σ unmet
      + fill_bias_weight · Σ (usable_energy − E[h+1])
      + move_penalty · (Σ p_dis + Σ p_ch)
    Large unmet_penalty strongly discourages unmet demand. Small fill_bias_weight
    nudges the solution to keep the battery reasonably filled. move_penalty
    discourages unnecessary cycling.
  • Returns: (grid, p_dis, p_ch, unmet, soe_fraction)

- make_main_dispatch_figure(): stacked bars of grid/charge/discharge/unmet with
  demand and grid limit overlaid.
- make_dashboard_figure(): 2×2 panel with dispatch, duration curves, demand
  histogram, and state-of-energy trace.

Notes
- Efficiencies are applied as constants (η_ch, η_dis). Use dt_hours to control
  the step size. SoE is returned as a fraction of usable energy [0..1].
- Solver: PuLP with CBC by default; raise if status is not Optimal/Feasible.
"""



@dataclass(frozen=True)
class SystemParams:
    """Container for all model inputs and objective weights.

    All units are in kW/kWh and hours where applicable.
    """
    hours: int = 24
    grid_limit_kw: float = 900.0
    storage_max_discharge_kw: float = 500.0
    storage_max_charge_kw: float = 500.0
    usable_nominal_energy_kwh: float = 1000.0
    eta_charge: float = 1.0
    eta_discharge: float = 1.0
    initial_soe: float = 1.0
    final_soe: Optional[float] = None
    unmet_penalty: float = 1e8
    fill_bias_weight: float = 1e-2
    move_penalty: float = 0.1
    peak_kw: float = 1200.0
    avg_to_peak_ratio: float = 0.70
    seed: int = 7
    exact_square: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Demand generator
# ──────────────────────────────────────────────────────────────────────────────
def generate_step_demand(p: SystemParams) -> List[float]:
    """Generate a two-level (low/peak) hourly demand profile with a given average."""
    rnd = random.Random(p.seed)
    H = p.hours
    peak = float(p.peak_kw)
    target_avg = float(p.avg_to_peak_ratio) * peak

    # Choose a plausible "low" level
    low_min = 0.15 * peak
    low_max = min(0.9 * target_avg, 0.85 * peak)
    low = rnd.uniform(low_min, max(low_min + 1e-6, low_max))

    # Number of peak hours to match the target average
    denom = (peak - low)
    frac_peak = (target_avg - low) / denom if denom > 0 else 0.0
    frac_peak = max(0.0, min(1.0, frac_peak))
    n_peak = int(round(frac_peak * H))
    peak_hours = set(rnd.sample(range(H), n_peak))

    demand = [peak if h in peak_hours else low for h in range(H)]

    # Final normalization to hit the exact average
    current_avg = float(np.mean(demand))
    if current_avg > 0:
        demand = [d * (target_avg / current_avg) for d in demand]
    return demand


# ──────────────────────────────────────────────────────────────────────────────
# Optimization model (LP)
# ──────────────────────────────────────────────────────────────────────────────
def build_and_solve_lp(
    demand_kw: List[float],
    grid_limit_kw: float,
    storage_max_discharge_kw: float,
    storage_max_charge_kw: float,
    usable_nominal_energy_kwh: float,
    eta_charge: float = 1.0,
    eta_discharge: float = 1.0,
    initial_soe: float = 1.0,
    final_soe: Optional[float] = None,
    dt_hours: float = 1.0,
    unmet_penalty: float = 1e8,
    fill_bias_weight: float = 1e-2,
    move_penalty: float = 0.1,
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """Solve the dispatch problem with grid-first policy plus storage.

    Decision variables per hour:
      - grid[h]  ≥ 0 : grid import (kW)
      - p_dis[h] ≥ 0 : storage discharge power (kW)
      - p_ch[h]  ≥ 0 : storage charge power (kW)
      - unmet[h] ≥ 0 : unmet demand (kW)

    Storage dynamics:
      E[h+1] = E[h] + η_charge * p_ch[h] * dt - (p_dis[h] / η_discharge) * dt
      with 0 ≤ E[h] ≤ usable_nominal_energy_kwh
      and E[0] fixed (initial_soe), E[H] optional (final_soe).

    Objective:
      Minimize:
        unmet_penalty * Σ unmet[h]
        + fill_bias_weight * Σ (usable_nominal_energy_kwh - E[h+1])
        + move_penalty * (Σ p_dis[h] + Σ p_ch[h])

    Returns the optimal time series for grid, p_dis, p_ch, unmet, and SoE (E/usable_energy).
    """
    # Ensure pure floats and avoid division on LpVariables (use reciprocal)
    dt_hours = float(dt_hours)
    eta_charge = float(eta_charge)
    inv_eta_discharge = 1.0 / float(eta_discharge)

    H = len(demand_kw)
    model = pulp.LpProblem("GridPlusStorageDispatch", pulp.LpMinimize)

    # Variables
    grid = pulp.LpVariable.dicts("grid_kw", range(H), lowBound=0)
    p_dis = pulp.LpVariable.dicts("storage_discharge_kw", range(H), lowBound=0)
    p_ch = pulp.LpVariable.dicts("storage_charge_kw", range(H), lowBound=0)
    unmet = pulp.LpVariable.dicts("unmet_kw", range(H), lowBound=0)
    E = pulp.LpVariable.dicts(
        "E_kwh", range(H + 1), lowBound=0, upBound=float(usable_nominal_energy_kwh)
    )

    # Initial and terminal energy constraints
    model += E[0] == float(initial_soe) * float(usable_nominal_energy_kwh), "init_energy"
    if final_soe is not None:
        model += E[H] == float(final_soe) * float(usable_nominal_energy_kwh), "terminal_energy"

    # Per-hour constraints
    for h in range(H):
        # Power balance: grid + discharge - charge + unmet = demand
        model += grid[h] + p_dis[h] - p_ch[h] + unmet[h] == float(demand_kw[h]), f"balance_{h}"

        # Limits
        model += grid[h] <= float(grid_limit_kw), f"grid_limit_{h}"
        model += p_dis[h] <= float(storage_max_discharge_kw), f"dis_limit_{h}"
        model += p_ch[h] <= float(storage_max_charge_kw), f"ch_limit_{h}"

        # Energy dynamics (NO division on PuLP expressions)
        model += (
            E[h + 1]
            == E[h]
               + eta_charge * p_ch[h] * dt_hours
               - (inv_eta_discharge * p_dis[h] * dt_hours)
        ), f"E_dyn_{h}"

    # Objective
    model += (
        float(unmet_penalty) * pulp.lpSum(unmet[h] for h in range(H))
        + float(fill_bias_weight) * pulp.lpSum((float(usable_nominal_energy_kwh) - E[h + 1]) for h in range(H))
        + float(move_penalty) * (pulp.lpSum(p_dis[h] for h in range(H)) + pulp.lpSum(p_ch[h] for h in range(H)))
    )

    # Solve with default CBC solver
    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] not in ("Optimal", "Feasible"):
        raise RuntimeError(f"Solver failed: {pulp.LpStatus[status]}")

    # Extract solution
    grid_sol = [pulp.value(grid[h]) for h in range(H)]
    p_dis_sol = [pulp.value(p_dis[h]) for h in range(H)]
    p_ch_sol = [pulp.value(p_ch[h]) for h in range(H)]
    unmet_sol = [pulp.value(unmet[h]) for h in range(H)]
    E_sol = [pulp.value(E[h]) for h in range(H + 1)]

    # Convert to SoE fraction [0..1]
    usable = float(usable_nominal_energy_kwh)
    soe = [e / usable if usable > 0 else float("nan") for e in E_sol]
    return grid_sol, p_dis_sol, p_ch_sol, unmet_sol, soe


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers (Plotly)
# ──────────────────────────────────────────────────────────────────────────────
def make_main_dispatch_figure(
    demand: List[float],
    grid: List[float],
    storage_discharge: List[float],
    storage_charge: List[float],
    unmet: List[float],
    grid_limit_kw: float,
    title: str = "Dispatch vs Demand",
) -> go.Figure:
    """Stacked bar chart of Grid, Storage (±) and Demand line with Grid limit."""
    H = len(demand)
    hours = list(range(H))
    demand_np = np.array(demand)
    grid_np = np.array(grid)
    dis_np = np.array(storage_discharge)
    ch_np = -np.array(storage_charge)  # negative to show charge below zero in stacked bars
    unmet_np = np.array(unmet)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Grid", x=hours, y=grid_np, hovertemplate="%{y:.1f} kW"))
    fig.add_trace(go.Bar(name="Storage (discharge)", x=hours, y=dis_np, hovertemplate="%{y:.1f} kW"))
    fig.add_trace(go.Bar(name="Storage (charge)", x=hours, y=ch_np, hovertemplate="%{y:.1f} kW"))
    if unmet_np.sum() > 1e-9:
        fig.add_trace(go.Bar(name="Unmet", x=hours, y=unmet_np, hovertemplate="%{y:.2f} kW"))

    fig.add_trace(go.Scatter(
        name="Demand",
        x=hours, y=demand_np, mode="lines+markers",
        line=dict(width=2), hovertemplate="%{y:.1f} kW"
    ))
    fig.add_trace(go.Scatter(
        name="Grid limit",
        x=hours, y=[grid_limit_kw] * H, mode="lines",
        line=dict(dash="dashdot"), hovertemplate="%{y:.1f} kW"
    ))

    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Hour",
        yaxis_title="Power (kW)  (charge < 0, discharge > 0)",
        legend_title="Legend",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def make_dashboard_figure(
    demand: List[float],
    grid: List[float],
    storage_discharge: List[float],
    storage_charge: List[float],
    unmet: List[float],
    soe: List[float],
    grid_limit_kw: float,
    title: str = "Grid Connection — Overview",
) -> go.Figure:
    """2×2 dashboard: stacked dispatch, duration curves, demand histogram, SoE."""
    H = len(demand)
    hours = list(range(H))
    demand_np = np.array(demand)
    grid_np = np.array(grid)
    dis_np = np.array(storage_discharge)
    ch_np = np.array(storage_charge)
    unmet_np = np.array(unmet)

    # Supply to the load for duration curve (grid + discharge only)
    supplied_np = grid_np + dis_np

    # SoE has H+1 points (state at boundaries)
    soe_hours = list(range(H + 1))

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Dispatch (stacked)", "Duration Curves", "Demand Histogram", "State of Energy"),
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # A) Dispatch (stacked)
    fig.add_trace(go.Bar(name="Grid", x=hours, y=grid_np), row=1, col=1)
    fig.add_trace(go.Bar(name="Storage (discharge)", x=hours, y=dis_np), row=1, col=1)
    fig.add_trace(go.Bar(name="Storage (charge)", x=hours, y=-ch_np), row=1, col=1)
    if unmet_np.sum() > 1e-9:
        fig.add_trace(go.Bar(name="Unmet", x=hours, y=unmet_np), row=1, col=1)
    fig.add_trace(go.Scatter(name="Demand", x=hours, y=demand_np, mode="lines+markers"), row=1, col=1)
    fig.add_trace(
        go.Scatter(name="Grid limit", x=hours, y=[grid_limit_kw] * H, mode="lines", line=dict(dash="dashdot")),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="kW (charge < 0)", row=1, col=1)
    fig.update_xaxes(title_text="Hour", row=1, col=1)

    # B) Duration curves — Demand (solid) vs Supply (dashed)
    fig.add_trace(
        go.Scatter(name="Demand (sorted)", x=list(range(H)), y=np.sort(demand_np)[::-1], mode="lines"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            name="Supply (grid + discharge, sorted)",
            x=list(range(H)),
            y=np.sort(supplied_np)[::-1],
            mode="lines",
            line=dict(dash="dash"),
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="kW", row=1, col=2)
    fig.update_xaxes(title_text="Ranked hour", row=1, col=2)

    # C) Histogram of demand
    fig.add_trace(go.Histogram(name="Demand histogram", x=demand_np, nbinsx=12, showlegend=True), row=2, col=1)
    fig.update_yaxes(title_text="Hours", row=2, col=1)
    fig.update_xaxes(title_text="Demand (kW)", row=2, col=1)

    # D) State of Energy (0..1)
    fig.add_trace(go.Scatter(name="SoE", x=soe_hours, y=soe, mode="lines+markers"), row=2, col=2)
    fig.update_yaxes(title_text="SoE (0–1)", row=2, col=2)
    fig.update_xaxes(title_text="Hour (state index)", row=2, col=2)

    # Layout
    fig.update_layout(
        title=title,
        barmode="stack",
        legend_title="Legend",
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02),
        margin=dict(l=60, r=240, t=80, b=60),
    )
    return fig
