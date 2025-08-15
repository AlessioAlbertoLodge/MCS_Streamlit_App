# optimize_battery_power_schedule.py
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize, Bounds, NonlinearConstraint

def _pick_solver():
    have = set(cp.installed_solvers())
    for s in ("ECOS", "HiGHS", "SCS", "OSQP", "GLPK"):
        if s in have:
            return s
    raise RuntimeError("No LP solver available for cvxpy")

def optimise_battery(
    prices: pd.DataFrame | pd.Series | np.ndarray,
    e_nom: float,
    soc0: float,
    p_charge_max: float,
    p_discharge_max: float,
    dt: float = 1.0,
) -> dict:
    """
    Solve the 24 h LP *and* give back a tidy schedule DataFrame.

    Parameters
    ----------
    prices : DataFrame with 'time' + 'price' columns, or Series/ndarray of prices.
    e_nom, soc0, p_charge_max, p_discharge_max, dt : see earlier version.

    Returns
    -------
    {
      "P"       : 1-D ndarray (kW)
      "SOC"     : 1-D ndarray (len = T+1, fraction)
      "revenue" : float (€)
      "status"  : solver status string
      "schedule": pandas.DataFrame  (time, price €/MWh, power MW, SOC start)
    }
    """
    # --- coerce price vector & keep the time column if present -------------
    if isinstance(prices, pd.DataFrame):
        time_col   = prices["time"].to_numpy()
        price_vec  = prices["price"].to_numpy(dtype=float)/1000
    elif isinstance(prices, pd.Series):
        time_col   = None
        price_vec  = prices.to_numpy(dtype=float)/1000
    else:  # ndarray
        time_col   = None
        price_vec  = np.asarray(prices, dtype=float).flatten()/1000

    T = price_vec.size

    # --- build LP ----------------------------------------------------------
    P   = cp.Variable(T)
    SOC = cp.Variable(T + 1)

    constraints = [
        SOC[0] == soc0,
        SOC[1:] == SOC[:-1] - 100*(P * dt) / e_nom,
        SOC >= 0, SOC <= 100,
        P >= -p_charge_max,
        P <=  p_discharge_max,
    ]

    revenue = cp.sum(cp.multiply(price_vec, P * dt))
    prob    = cp.Problem(cp.Maximize(revenue), constraints)
    prob.solve(solver=_pick_solver())

    P_opt   = np.asarray(P.value,  dtype=float).ravel()
    SOC_opt = np.asarray(SOC.value, dtype=float).ravel()

    # --- build schedule DataFrame -----------------------------------------
    sched = pd.DataFrame({
        "time":        time_col if time_col is not None else np.arange(T),
        "price €/kWh": price_vec.round(2),
        "power kW":    P_opt.round(3),
        "SOC start":   SOC_opt[:-1].round(3)
    })
    #print(P_opt)
    #print(SOC_opt)
    return dict(P=P_opt, SOC=SOC_opt, revenue=float(prob.value),
                status=prob.status, schedule=sched)

def optimise_battery_with_ageing_dp(
    prices,                       # DataFrame | Series | ndarray (€/MWh)
    e_nom: float,
    soc0: float,                  # %  0‥100
    p_charge_max: float,          # kW  (positive)
    p_discharge_max: float,       # kW  (positive)
    T_amb,                        # Kelvin  scalar or vector
    degradation_parameters,       # [a,b,c,d,e,f,g,h,z]
    full_cost_of_battery: float,  # €
    dt: float = 1.0,              # h   step length
    soc_step: float = 0.5,        # %   SOC grid
    p_step: float  = 0.25,        # kW  power grid
):
    # ── prices → vector (€/kWh) and optional time column ──────────────────
    if isinstance(prices, pd.DataFrame):
        time_col  = prices["time"].to_numpy()
        price_vec = prices["price"].to_numpy(dtype=float) / 1000.0
    elif isinstance(prices, pd.Series):
        time_col, price_vec = None, prices.to_numpy(dtype=float) / 1000.0
    else:
        time_col, price_vec = None, np.asarray(prices, float).flatten() / 1000.0

    T = price_vec.size                                   # horizon length

    # ── ambient temperature vector (Kelvin) ───────────────────────────────
    T_amb = np.full(T, float(T_amb)) if np.isscalar(T_amb) \
            else np.asarray(T_amb, float).flatten()

    # ── pre-compute coefficients ──────────────────────────────────────────
    a,b,c,d,e,f,g,h,z = degradation_parameters
    k1    = a*T_amb**2 + b*T_amb + c
    k2    = f*np.exp(h / T_amb) * (dt/24.0)**z          # calendar term
    kT    = d*T_amb + e
    rev_k = price_vec * dt                              # €/kW per step
    alpha = dt * 1000.0 / 1500.0                        # kWh → Ah

    # ── SOC grid & DP tables ──────────────────────────────────────────────
    soc_grid = np.arange(0.0, 100.0 + 1e-12, soc_step)
    G        = soc_grid.size
    value    = -np.inf * np.ones((T+1, G))
    policy   = np.zeros((T,   G))
    value[T] = 0.0                                      # terminal profit

    # ── backward dynamic programme ────────────────────────────────────────
    for t in range(T-1, -1, -1):
        for i, soc in enumerate(soc_grid):
            # power range that keeps SOC in 0‥100 % next step
            p_min_soc = -(soc       ) * e_nom / (100*dt)
            p_max_soc = (100.0-soc) * e_nom / (100*dt)
            p_min = max(-p_charge_max, p_min_soc)
            p_max = min( p_discharge_max, p_max_soc)
            if p_min > p_max:
                continue

            P_choices = np.arange(p_min, p_max + 1e-12, p_step)
            if P_choices.size == 0:
                continue

            next_soc = soc - 100.0 * P_choices * dt / e_nom
            next_idx = np.round(next_soc / soc_step).astype(int)
            ok       = (next_idx >= 0) & (next_idx < G)
            P_choices, next_idx = P_choices[ok], next_idx[ok]
            if P_choices.size == 0:
                continue

            P_abs  = np.abs(P_choices)
            Ah     = alpha * P_abs
            Crate  = P_abs / e_nom
            soc_bar = 0.5*(soc + next_soc[ok])

            q_cyc = (k1[t] * np.exp(kT[t]*Crate) * Ah)        / 100.0
            q_cal = (k2[t] * np.exp(g * soc_bar / 100.0))     / 100.0
            step_profit = rev_k[t]*P_choices \
                          - full_cost_of_battery*(q_cyc+q_cal)/40.0
            tot = step_profit + value[t+1, next_idx]

            best = np.argmax(tot)
            value[t, i]  = tot[best]
            policy[t, i] = P_choices[best]

    # ── roll forward from soc0 ────────────────────────────────────────────
    def idx(s): return int(round(s / soc_step))

    P_opt   = np.zeros(T)
    SOC_opt = np.zeros(T+1)
    SOC_opt[0] = soc0

    revenue = ageing_cyc = ageing_cal = 0.0
    soc_i   = idx(soc0)

    for t in range(T):
        P = policy[t, soc_i]
        P_opt[t] = P
        soc_next = SOC_opt[t] - 100.0 * P * dt / e_nom
        SOC_opt[t+1] = soc_next

        P_abs  = abs(P)
        Ah     = alpha * P_abs
        Crate  = P_abs / e_nom
        soc_bar = 0.5*(SOC_opt[t] + soc_next)

        q_cyc = (k1[t]*np.exp(kT[t]*Crate)*Ah)          / 100.0
        q_cal = (k2[t]*np.exp(g*soc_bar/100.0))         / 100.0
        ageing_cyc += full_cost_of_battery*q_cyc / 40.0
        ageing_cal += full_cost_of_battery*q_cal / 40.0
        revenue    += rev_k[t]*P
        soc_i       = idx(soc_next)

    ageing_cost = ageing_cyc + ageing_cal
    profit      = revenue - ageing_cost

    # ── tidy schedule DataFrame (same format as your LP helper) ───────────
    sched = pd.DataFrame({
        "time":        time_col if time_col is not None else np.arange(T),
        "price €/kWh": price_vec.round(4),
        "power kW":    P_opt.round(3),
        "SOC start":   SOC_opt[:-1].round(3),
    })

    return dict(
        P                     = P_opt,
        SOC                   = SOC_opt,
        revenue               = revenue,
        ageing_cost           = ageing_cost,
        ageing_cost_cyclic    = ageing_cyc,
        ageing_cost_calendar  = ageing_cal,
        profit                = profit,
        status                = f"DP optimum on {G}-point SOC grid "
                                f"({soc_step:.2f} %); horizon {T} steps",
        schedule              = sched,
    )

def optimise_battery_with_ageing(
    prices: pd.DataFrame | pd.Series | np.ndarray,
    e_nom: float,                # kWh  ( = battery_capacity )
    soc0: float,                 # %    initial state-of-charge  [0‥100]
    p_charge_max: float,         # kW   (<0 in optimisation)
    p_discharge_max: float,      # kW
    T_amb: np.ndarray | float,   # K, length-T vector or scalar
    degradation_parameters,      # [a, b, c, d, e, f, g, h, z]
    full_cost_of_battery: float, # €    replacement value of whole BESS
    dt: float = 1.0,             # h    sampling interval
    max_iter: int = 5000,
) -> dict:
    """
    24-step non-linear revenue maximisation with cyclic + calendar ageing costs.

        Q_loss_cyclic[t]   = (a·T² + b·T + c) · exp((d·T + e)·C_rate[t]) · |Ah[t]|
        Q_loss_calendar[t] = f·exp(g·SOC̄[t]) · exp(h/T) · (1/24)^z

        cost[t] = full_cost_of_battery · (Q_loss_cyclic + Q_loss_calendar) / 40

    Returns the usual schedule dict plus:
        ageing_cost_cyclic   – € total cyclic cost
        ageing_cost_calendar – € total calendar cost
    """
    # ───── price vector & time column ──────────────────────────────────────
    if isinstance(prices, pd.DataFrame):
        time_col  = prices["time"].to_numpy()
        price_vec = prices["price"].to_numpy(dtype=float) / 1000.0   # €/kWh
    elif isinstance(prices, pd.Series):
        time_col, price_vec = None, prices.to_numpy(dtype=float) / 1000.0
    else:
        time_col, price_vec = None, np.asarray(prices, dtype=float).flatten() / 1000.0

    T = price_vec.size

    # ───── ambient temperature vector ─────────────────────────────────────
    if np.isscalar(T_amb):
        T_amb = np.full(T, float(T_amb))
    else:
        T_amb = np.asarray(T_amb, dtype=float).flatten()
        if T_amb.size != T:
            raise ValueError("T_amb must be scalar or length = len(prices)")

    # ───── pre-compute constants that never change during optimisation ────
    a, b, c, d, e, f, g, h, z = degradation_parameters
    k1  = a*T_amb**2 + b*T_amb + c                         # (T,)
    k2  = f*np.exp(h / T_amb) * (1/24.0)**z                # (T,)
    kT  = d*T_amb + e                                      # (T,)
    rev_coeff = price_vec * dt                             # (T,)

    alpha = dt * 1000.0 / 1500.0                                # kWh → Ah

    # ───── helper functions ───────────────────────────────────────────────
    def simulate_soc(P: np.ndarray) -> np.ndarray:
        """Vectorised SOC trajectory (%) given power in kW."""
        soc = np.empty(T + 1)
        soc[0] = soc0
        soc[1:] = soc0 - 100.0 * np.cumsum(P) * dt / e_nom
        return soc

    def profit_and_grad(P: np.ndarray):
        """
        Return objective (= negative profit) and its analytic gradient.
        Positive P = discharge (sell); negative P = charge (buy).
        """
        # revenue part (linear)
        revenue   = np.dot(rev_coeff, P)          # €
        grad_rev  = rev_coeff                     # (T,)

        # ---- ageing part -------------------------------------
        P_abs  = np.abs(P)
        Ah     = alpha * P_abs
        C_rate = P_abs / e_nom

        soc         = simulate_soc(P)
        soc_bar     = 0.5 * (soc[:-1] + soc[1:])

        exp1 = np.exp(kT * C_rate)
        exp2 = np.exp(g * soc_bar / 100.0)

        # ➊ divide by 100 *here*
        q_loss_cyclic   = (k1 * exp1 * Ah) / 100.0
        q_loss_calendar = (k2 * exp2)      / 100.0
        q_loss          = q_loss_cyclic + q_loss_calendar

        cost_cyclic_vec   = full_cost_of_battery * q_loss_cyclic   / 40.0
        cost_calendar_vec = full_cost_of_battery * q_loss_calendar / 40.0
        cost_total        = cost_cyclic_vec.sum() + cost_calendar_vec.sum()

        # ---- gradient wrt P ---------------------------------------------
        signP        = np.sign(P)
        dCrate_dP    = signP / e_nom
        dexp1_dP     = exp1 * kT * dCrate_dP

        # gradient of cyclic cost (vector, length T)
        dterm1_dP = (alpha/100.0) * (k1 * exp1 * signP + k1 * P_abs * dexp1_dP)

        # gradient of calendar cost needs SOC̄ Jacobian --------------------
        dsoc_dP  = -100.0 * dt / e_nom                    # scalar
        J        = np.zeros((T, T))
        idx      = np.arange(T)
        J[idx, idx] = 0.5 * dsoc_dP
        J[idx[:-1], idx[1:]] += 0.5 * dsoc_dP             # tridiagonal

        dexp2_dP   = (exp2 * (g/100.0))[:, None] * J      # (T,T)
        dterm2_dP = (k2/100.0)[:, None] * dexp2_dP

        grad_cost = (full_cost_of_battery / 40.0) * (
            dterm1_dP + dterm2_dP.sum(axis=0)
        )

        # final objective & gradient (negated because we minimise)
        profit = revenue - cost_total
        obj    = -profit
        grad   = -(grad_rev - grad_cost)
        return obj, grad

    # wrappers for SciPy
    def obj_only(P):  return profit_and_grad(P)[0]
    def obj_grad(P):  return profit_and_grad(P)[1]

    # ───── SOC constraints ────────────────────────────────────────────────
    def soc_lower_fn(P):  return simulate_soc(P)[1:]            # ≥ 0
    def soc_upper_fn(P):  return 100.0 - simulate_soc(P)[1:]    # ≥ 0

    ineq_low  = NonlinearConstraint(soc_lower_fn,  0.0,  np.inf)
    ineq_high = NonlinearConstraint(soc_upper_fn,  0.0,  np.inf)

    # ───── optimisation call ──────────────────────────────────────────────
    bounds = Bounds(-p_charge_max * np.ones(T), p_discharge_max * np.ones(T))
    P0     = np.zeros(T)

    res = minimize(
        obj_only, P0,
        jac=obj_grad,
        bounds=bounds,
        constraints=[ineq_low, ineq_high],
        method="SLSQP",
        options=dict(maxiter=max_iter, ftol=1e-6, disp=False),
    )
    if not res.success:
        raise RuntimeError(f"Non-linear optimisation failed: {res.message}")

    P_opt   = res.x
    SOC_opt = simulate_soc(P_opt)

    # ───── rebuild cost parts with the optimal P for reporting ────────────
    P_abs  = np.abs(P_opt)
    Ah     = alpha * P_abs
    C_rate = P_abs / e_nom
    soc_bar = 0.5 * (SOC_opt[:-1] + SOC_opt[1:])

    exp1 = np.exp(kT * C_rate)
    exp2 = np.exp(g * soc_bar / 100.0)

    q_loss_cyclic   = (k1 * exp1 * Ah) / 100.0     # keep /100 here as well
    q_loss_calendar = (k2 * exp2)      / 100.0
    ageing_cost_cyclic   = float((full_cost_of_battery / 40.0) * q_loss_cyclic.sum())
    ageing_cost_calendar = float((full_cost_of_battery / 40.0) * q_loss_calendar.sum())
    revenue              = float(np.dot(price_vec * dt, P_opt))
    profit               = revenue - (ageing_cost_cyclic + ageing_cost_calendar)

    # ───── schedule DataFrame ─────────────────────────────────────────────
    sched = pd.DataFrame({
        "time":        time_col if time_col is not None else np.arange(T),
        "price €/kWh": price_vec.round(4),
        "power kW":    P_opt.round(3),
        "SOC start":   SOC_opt[:-1].round(3),
    })

    # ───── return dict ────────────────────────────────────────────────────
    return dict(
        P=P_opt,
        SOC=SOC_opt,
        revenue=revenue,
        ageing_cost=ageing_cost_cyclic + ageing_cost_calendar,
        ageing_cost_cyclic=ageing_cost_cyclic,
        ageing_cost_calendar=ageing_cost_calendar,
        profit=profit,
        status=res.message,
        schedule=sched,
    )

# ---------------------------------------------------------------------------
# Utility: evaluate any given power schedule with cyclic + calendar ageing
# ---------------------------------------------------------------------------
def evaluate_schedule_with_ageing(
    schedule: pd.DataFrame | pd.Series | np.ndarray,
    prices:   pd.DataFrame | pd.Series | np.ndarray,
    e_nom:    float,                 # kWh   – battery capacity
    soc0:     float,                 # %     – initial SOC [0‥100]
    T_amb:    np.ndarray | float,    # K    – scalar or length-T vector
    degradation_parameters,          # [a, b, c, d, e, f, g, h, z]
    full_cost_of_battery: float,     # €     – replacement value of the BESS
    dt:       float = 1.0,           # h     – time step
) -> dict:
    """
    Financial assessment of *any* hourly power profile `schedule`
    using the ageing model from `optimise_battery_with_ageing`.

    Returns
    -------
    dict with keys: P, SOC, revenue, ageing_cost, ageing_cost_cyclic,
                    ageing_cost_calendar, profit, status, schedule
    """
    # ───── pull power vector P (kW) and a matching time column ───────────
    if isinstance(schedule, pd.DataFrame):
        if "power kW" in schedule:
            P = schedule["power kW"].to_numpy(dtype=float)
        else:                          # treat whole row as numeric
            P = schedule.to_numpy(dtype=float).flatten()
        time_col = schedule["time"].to_numpy() if "time" in schedule else None
    elif isinstance(schedule, pd.Series):
        P, time_col = schedule.to_numpy(dtype=float), None
    else:  # ndarray
        P, time_col = np.asarray(schedule, dtype=float).flatten(), None

    # ───── obtain price vector (€/kWh) ────────────────────────────────────
    if isinstance(prices, pd.DataFrame):
        price_vec = prices["price"].to_numpy(dtype=float) / 1000.0
        if time_col is None and "time" in prices:
            time_col = prices["time"].to_numpy()
    elif isinstance(prices, pd.Series):
        price_vec = prices.to_numpy(dtype=float) / 1000.0
    else:
        price_vec = np.asarray(prices, dtype=float).flatten() / 1000.0

    T = price_vec.size
    assert len(P) == T, "schedule and price series must have identical length"

    # ───── ambient temperature vector ─────────────────────────────────────
    if np.isscalar(T_amb):
        T_amb = np.full(T, float(T_amb))
    else:
        T_amb = np.asarray(T_amb, dtype=float).flatten()
        if T_amb.size != T:
            raise ValueError("T_amb must be scalar or length = len(prices)")

    # ───── constants shared with the optimisation routine ────────────────
    a, b, c, d, e, f, g, h, z = degradation_parameters
    k1 = a*T_amb**2 + b*T_amb + c                    # (T,)
    k2 = f*np.exp(h / T_amb) * (1/24.0)**z           # (T,)
    kT = d*T_amb + e                                 # (T,)

    alpha = 1000.0 / 1500.0                          # kWh → Ah conversion

    # ───── simulate SOC trajectory (%) ───────────────────────────────────
    SOC = np.empty(T + 1)
    SOC[0] = soc0
    SOC[1:] = soc0 - 100.0 * np.cumsum(P) * dt / e_nom

    # ───── revenue (positive = income) ───────────────────────────────────
    revenue = float(np.dot(price_vec * dt, P))       # €

    # ───── ageing losses and costs ───────────────────────────────────────
    P_abs   = np.abs(P)
    Ah      = alpha * P_abs
    C_rate  = P_abs / e_nom
    soc_bar = 0.5 * (SOC[:-1] + SOC[1:])             # hourly mean SOC %

    exp1 = np.exp(kT * C_rate)
    exp2 = np.exp(g * soc_bar / 100.0)

    # % capacity lost *per hour*
    q_loss_cyclic   = k1 * exp1 * Ah       / 100.0
    q_loss_calendar = k2 * exp2            / 100.0

    ageing_cost_cyclic   = float((full_cost_of_battery / 40.0) * q_loss_cyclic.sum())
    ageing_cost_calendar = float((full_cost_of_battery / 40.0) * q_loss_calendar.sum())
    ageing_cost_total    = ageing_cost_cyclic + ageing_cost_calendar

    profit = revenue - ageing_cost_total

    # ───── schedule DataFrame for reporting ──────────────────────────────
    sched = pd.DataFrame({
        "time":        time_col if time_col is not None else np.arange(T),
        "price €/kWh": price_vec.round(4),
        "power kW":    P.round(3),
        "SOC start":   SOC[:-1].round(3),
    })

    # ───── return everything in familiar format ──────────────────────────
    return dict(
        P=P,
        SOC=SOC,
        revenue=revenue,
        ageing_cost=ageing_cost_total,
        ageing_cost_cyclic=ageing_cost_cyclic,
        ageing_cost_calendar=ageing_cost_calendar,
        profit=profit,
        status="evaluated",
        schedule=sched,
    )

