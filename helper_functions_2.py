# helper_functions_2.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

# Reuse core model & plots from your main helpers
from helper_functions import (
    SystemParams,
    generate_step_demand,
    build_and_solve_lp,
    make_main_dispatch_figure,
)

# ──────────────────────────────────────────────────────────────────────────────
# Demand utilities
# ──────────────────────────────────────────────────────────────────────────────

def compute_max_demand(demand_kw: List[float]) -> float:
    """Return the maximum power of a demand series (kW)."""
    if not demand_kw:
        return 0.0
    return float(np.max(demand_kw))


def make_demand_figure(
    demand_kw: List[float],
    grid_limit_kw: Optional[float] = None,
    title: str = "Demand (kW)"
) -> go.Figure:
    """Simple line plot of demand, optionally overlaying a horizontal grid limit."""
    H = len(demand_kw)
    x = list(range(H))
    y = np.asarray(demand_kw, float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(name="Demand", x=x, y=y, mode="lines+markers"))
    if grid_limit_kw is not None:
        fig.add_trace(go.Scatter(
            name="Grid limit",
            x=x, y=[grid_limit_kw]*H, mode="lines",
            line=dict(dash="dashdot")
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Step",
        yaxis_title="Power (kW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Sizing logic: prescribe discharge cap from grid limit, then find minimal energy
# ──────────────────────────────────────────────────────────────────────────────

def derive_discharge_cap_from_grid_limit(
    demand_kw: List[float],
    grid_limit_kw: float,
) -> float:
    """
    Discharge cap is: P_dis_max = max(demand) - grid_limit.
    If grid_limit >= max(demand), clamp at zero (no storage needed).
    """
    max_dem = compute_max_demand(demand_kw)
    return max(0.0, float(max_dem) - float(grid_limit_kw))


def unmet_total(unmet_kw: List[float], dt_hours: float) -> float:
    """Total unmet energy (kWh) over the horizon."""
    if not unmet_kw:
        return 0.0
    return float(np.sum(np.maximum(0.0, np.array(unmet_kw))) * dt_hours)


def _solve_with_energy(
    demand_kw: List[float],
    grid_limit_kw: float,
    p_dis_max: float,
    p_ch_max: float,
    energy_kwh: float,
    *,
    eta_ch: float,
    eta_dis: float,
    init_soe_frac: float,
    final_soe_frac: Optional[float],
    dt_hours: float,
    unmet_penalty: float = 1e12,      # VERY LARGE → prefer zero unmet
    fill_bias_weight: float = 1e-4,   # tiny
    move_penalty: float = 1e-4,       # tiny
) -> Dict[str, List[float]]:
    """
    Run the LP with a given usable energy (kWh) and return solution dict with series.
    """
    H = len(demand_kw)
    p = SystemParams(
        hours=H,
        grid_limit_kw=grid_limit_kw,
        storage_max_discharge_kw=p_dis_max,
        storage_max_charge_kw=p_ch_max,
        usable_nominal_energy_kwh=energy_kwh,
        eta_charge=eta_ch,
        eta_discharge=eta_dis,
        initial_soe=init_soe_frac,
        final_soe=final_soe_frac,
        unmet_penalty=unmet_penalty,
        fill_bias_weight=fill_bias_weight,
        move_penalty=move_penalty,
    )

    grid, p_dis, p_ch, unmet, soe = build_and_solve_lp(
        demand_kw=demand_kw,
        grid_limit_kw=p.grid_limit_kw,
        storage_max_discharge_kw=p.storage_max_discharge_kw,
        storage_max_charge_kw=p.storage_max_charge_kw,
        usable_nominal_energy_kwh=p.usable_nominal_energy_kwh,
        eta_charge=p.eta_charge,
        eta_discharge=p.eta_discharge,
        initial_soe=p.initial_soe,
        final_soe=p.final_soe,
        dt_hours=dt_hours,
        unmet_penalty=p.unmet_penalty,
        fill_bias_weight=p.fill_bias_weight,
        move_penalty=p.move_penalty,
    )

    return {
        "grid": grid,
        "p_dis": p_dis,
        "p_ch": p_ch,
        "unmet": unmet,
        "soe": soe,
        "energy_kwh": energy_kwh,
        "grid_limit_kw": grid_limit_kw,
        "demand": demand_kw,
    }


def find_min_storage_energy_bisect(
    demand_kw: List[float],
    grid_limit_kw: float,
    p_dis_max: float,
    *,
    eta_ch: float = 1.0,
    eta_dis: float = 1.0,
    init_soe_frac: float = 1.0,
    final_soe_frac: Optional[float] = None,
    dt_hours: float = 0.25,
    tol_kwh: float = 0.05,
    max_iter: int = 50,
    p_ch_max: Optional[float] = None,
    unmet_penalty: float = 1e12,
    fill_bias_weight: float = 1e-4,
    move_penalty: float = 1e-4,
    add_safety_buffer: bool = True,
) -> Tuple[float, Dict[str, List[float]], float]:
    """
    Find the *smallest* storage energy (kWh) that achieves effectively **zero unmet**.
    Preference is enforced via a huge unmet penalty. Optionally adds a small safety buffer.
    Returns (energy_kwh, solution_dict, unmet_kwh).
    """
    H = len(demand_kw)
    if H == 0:
        sol = _solve_with_energy(
            demand_kw, grid_limit_kw, 0.0, 0.0, 0.0,
            eta_ch=eta_ch, eta_dis=eta_dis, init_soe_frac=init_soe_frac,
            final_soe_frac=final_soe_frac, dt_hours=dt_hours,
            unmet_penalty=unmet_penalty, fill_bias_weight=fill_bias_weight, move_penalty=move_penalty,
        )
        return 0.0, sol, 0.0

    if p_ch_max is None:
        p_ch_max = p_dis_max

    # If grid limit already >= peak, zero energy needed
    if derive_discharge_cap_from_grid_limit(demand_kw, grid_limit_kw) <= 0.0:
        sol = _solve_with_energy(
            demand_kw, grid_limit_kw, 0.0, 0.0, 0.0,
            eta_ch=eta_ch, eta_dis=eta_dis, init_soe_frac=init_soe_frac,
            final_soe_frac=final_soe_frac, dt_hours=dt_hours,
            unmet_penalty=unmet_penalty, fill_bias_weight=fill_bias_weight, move_penalty=move_penalty,
        )
        return 0.0, sol, 0.0

    lo = 0.0
    hi = H * p_dis_max * dt_hours  # conservative upper bound
    best_sol = None
    best_energy = hi

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        sol = _solve_with_energy(
            demand_kw, grid_limit_kw, p_dis_max, p_ch_max, mid,
            eta_ch=eta_ch, eta_dis=eta_dis, init_soe_frac=init_soe_frac,
            final_soe_frac=final_soe_frac, dt_hours=dt_hours,
            unmet_penalty=unmet_penalty, fill_bias_weight=fill_bias_weight, move_penalty=move_penalty,
        )
        unmet_kwh = unmet_total(sol["unmet"], dt_hours)

        if unmet_kwh <= 1e-6:
            best_sol = sol
            best_energy = mid
            hi = mid
        else:
            lo = mid

        if (hi - lo) <= tol_kwh:
            break

    if best_sol is None:
        best_sol = _solve_with_energy(
            demand_kw, grid_limit_kw, p_dis_max, p_ch_max, hi,
            eta_ch=eta_ch, eta_dis=eta_dis, init_soe_frac=init_soe_frac,
            final_soe_frac=final_soe_frac, dt_hours=dt_hours,
            unmet_penalty=unmet_penalty, fill_bias_weight=fill_bias_weight, move_penalty=move_penalty,
        )
        best_energy = hi

    if add_safety_buffer and p_dis_max > 0:
        buffer_kwh = max(0.005 * best_energy, dt_hours * p_dis_max)  # 0.5% or one slot at full discharge
        best_energy += buffer_kwh
        best_sol = _solve_with_energy(
            demand_kw, grid_limit_kw, p_dis_max, p_ch_max, best_energy,
            eta_ch=eta_ch, eta_dis=eta_dis, init_soe_frac=init_soe_frac,
            final_soe_frac=final_soe_frac, dt_hours=dt_hours,
            unmet_penalty=unmet_penalty, fill_bias_weight=fill_bias_weight, move_penalty=move_penalty,
        )

    return float(best_energy), best_sol, unmet_total(best_sol["unmet"], dt_hours)


def storage_duration_hours(energy_kwh: float, discharge_cap_kw: float) -> float:
    """Duration = energy / discharge (h)."""
    if discharge_cap_kw <= 0:
        return 0.0
    return float(energy_kwh / discharge_cap_kw)


# ──────────────────────────────────────────────────────────────────────────────
# Grid-share sweep with cost model
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_grid_share_tradeoff(
    demand_kw: List[float],
    *,
    grid_share_values: List[float],
    eta_ch: float,
    eta_dis: float,
    init_soe_frac: float,
    dt_hours: float,
    c_power_usd_per_kw: float,
    c_energy_usd_per_kwh: float,
    unmet_penalty: float = 1e12,
    fill_bias_weight: float = 1e-4,
    move_penalty: float = 1e-4,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    For each grid share r (0..1), set grid_limit = r * max(demand),
    prescribe p_dis_max = max(demand) - grid_limit, find minimal energy for zero unmet,
    and compute total CAPEX cost = c_power * p_dis_max + c_energy * energy_kwh.

    Returns
    -------
    best : dict
        Keys: grid_share, grid_limit_kw, p_dis_max_kw, energy_kwh, unmet_kwh, duration_h, total_cost_usd
    rows : list of dict
        One row per grid_share with same keys (useful to plot tables/charts).
    """
    max_dem = compute_max_demand(demand_kw)
    rows: List[Dict[str, float]] = []
    best: Optional[Dict[str, float]] = None

    for r in grid_share_values:
        r_clamped = float(np.clip(r, 0.0, 1.0))
        grid_limit = r_clamped * max_dem
        p_dis_max = max(0.0, max_dem - grid_limit)

        energy_kwh, sol, unmet_kwh = find_min_storage_energy_bisect(
            demand_kw=demand_kw,
            grid_limit_kw=grid_limit,
            p_dis_max=p_dis_max,
            p_ch_max=p_dis_max,
            eta_ch=eta_ch,
            eta_dis=eta_dis,
            init_soe_frac=init_soe_frac,
            final_soe_frac=None,
            dt_hours=dt_hours,
            tol_kwh=0.05,
            max_iter=60,
            unmet_penalty=unmet_penalty,
            fill_bias_weight=fill_bias_weight,
            move_penalty=move_penalty,
            add_safety_buffer=True,
        )

        duration_h = storage_duration_hours(energy_kwh, p_dis_max) if p_dis_max > 0 else 0.0
        total_cost = c_power_usd_per_kw * p_dis_max + c_energy_usd_per_kwh * energy_kwh

        row = dict(
            grid_share=r_clamped,
            grid_limit_kw=grid_limit,
            p_dis_max_kw=p_dis_max,
            energy_kwh=energy_kwh,
            duration_h=duration_h,
            unmet_kwh=unmet_kwh,
            total_cost_usd=total_cost,
            sol=sol,  # keep solution object to plot later if this is best
        )
        rows.append(row)

        if best is None or total_cost < best["total_cost_usd"] - 1e-6:
            best = row

    # Remove the sol object from non-best rows if desired (left as-is; page can ignore)
    assert best is not None
    return best, rows
