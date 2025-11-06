# pages/3_Documentation.py
import streamlit as st

# ----------------------------------------------------------------------------
# 📚 Integrated Documentation — Grid + Storage Optimizer Suite
# ----------------------------------------------------------------------------
# This page summarizes the purpose, structure, and logic of the complete
# Grid + Storage Optimizer web application, including:
#   • Home — Main hourly dispatch model
#   • Page 1 — Main 15-minute dispatch model
#   • Page 2 — Minimum storage sizing for a grid limit
#   • Page 3 — Power/Energy cost trade-off and optimal sizing
#   • Page 4 — Documentation (this page)
# The underlying mathematical models and utilities are defined in
# helper_functions.py and helper_functions_2.py.
# ----------------------------------------------------------------------------

def main():  # pragma: no cover
    """Render the integrated documentation/help page."""

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    st.title("📚 Documentation — Grid + Storage Optimizer Suite")
    st.caption("Version 0.2 · last updated 2025-10-04")

    st.markdown(
        """
        The **Grid + Storage Optimizer** is an interactive web application built with
        **Streamlit** for exploring the operational coupling between an **electric grid
        connection** and an **energy-storage system** (battery or equivalent).
        
        It provides a unified framework to:
        * 🕒 **Simulate dispatch** at hourly or 15-minute resolution  
        * 📐 **Size storage energy** to satisfy a grid-import constraint  
        * 💰 **Optimise cost trade-offs** between power, energy, and grid-limit choices  

        The core engine solves **linear-programming (LP)** problems using `PuLP` and
        visualises results through `Plotly`.
        """
    )

    # ------------------------------------------------------------------
    # App architecture
    # ------------------------------------------------------------------
    st.header("🔧 App Architecture")
    st.markdown(
        """
        The app consists of **five Streamlit pages (including the home)** and two helper modules:

        | Page | File | Purpose |
        |------|------|----------|
        | ⚡ **Home — Grid + Storage Optimizer (Hourly)** | `Basic_Storage_Problem.py` | Entry point; solves the dispatch LP at hourly resolution (Δt = 1 h) with full sidebar inputs |
        | ⏱️ **Main Dispatch (15-min)** | `pages/1_Main Dispatch.py` | Same LP as home, but with Δt = 0.25 h and inputs placed on the main page |
        | 📐 **Min Storage for Grid Limit** | `pages/2_Minimum Storage for Grid Limit.py` | Finds smallest storage energy (kWh) required for zero unmet demand under a given grid-limit |
        | 💰 **Power/Energy Cost Trade-off** | `pages/3_Power Energy Cost Trade-off.py` | Scans grid-share ratios to identify cost-optimal combinations of power and energy |
        | 📚 **Documentation** | `pages/3_Documentation.py` | This help and explanation page |
        | ⚙️ **Helpers** | `helper_functions.py`, `helper_functions_2.py` | Contain LP formulation, demand generation, and plotting utilities |

        All pages share the same demand generator and LP solver, ensuring methodological
        consistency across analyses.
        """
    )

    # ------------------------------------------------------------------
    # Core model description
    # ------------------------------------------------------------------
    st.header("🧩 Core Dispatch Model")
    st.subheader("Mathematical Formulation")

    st.markdown(r"""
    **Objective**

    $$
    \min \Big(
    w_U \sum_h U_h \;+\;
    w_F \sum_h (E_{nom} - E_h) \;+\;
    w_M \sum_h (P^{dis}_h + P^{ch}_h)
    \Big)
    $$

    where:

    * \( w_U \): unmet-demand penalty (≫ others)  
    * \( w_F \): fill-bias weight (prefers high SoE)  
    * \( w_M \): movement penalty (discourages rapid cycling)

    **Constraints**

    $$
    \begin{aligned}
    G_h + P^{dis}_h - P^{ch}_h + U_h &= D_h && \text{(power balance)} \\\\
    0 \le G_h &\le G_{max} && \text{(grid limit)} \\\\
    0 \le P^{dis}_h &\le P^{dis}_{max}, \quad
    0 \le P^{ch}_h \le P^{ch}_{max} && \text{(storage limits)} \\\\
    E_{h+1} &= E_h + \eta_{ch} P^{ch}_h \Delta t -
    \frac{P^{dis}_h}{\eta_{dis}} \Delta t && \text{(energy dynamics)} \\\\
    0 \le E_h &\le E_{nom}, \quad
    E_0 = E_{init}, \; E_H = E_{final?}
    \end{aligned}
    $$
    """)

    # ------------------------------------------------------------------
    # Demand generation
    # ------------------------------------------------------------------
    st.header("📈 Synthetic Demand Generator")
    st.markdown(
        """
        Demand profiles are generated stochastically using `generate_step_demand()`:

        * Two-level ("peak/low") shape, randomized via `seed`
        * Average equals `avg_to_peak_ratio × peak_kw`
        * Adjustable duration (`hours`) and resolution (1 h or 0.25 h)

        This ensures reproducible yet flexible test cases for grid-storage studies.
        """
    )

    # ------------------------------------------------------------------
    # Page-by-page overview
    # ------------------------------------------------------------------
    st.header("🧭 Page-by-Page Overview")

    st.subheader("🏠 ⚡ Home — Grid + Storage Optimizer (Hourly)")
    st.markdown(
        """
        * **Purpose:** main entry point of the app — runs the dispatch LP at **hourly resolution (Δt = 1 h)**  
          to demonstrate the baseline grid + storage behavior.  
        * **Inputs:**  
            * Grid and storage limits (`grid_limit_kw`, `p_dis_max`, `p_ch_max`)  
            * Usable energy (`usable_nominal_energy_kwh`) and efficiencies (`η_ch`, `η_dis`)  
            * Initial and optional final SoE (%)  
            * Demand shape: `peak_kw`, `avg_to_peak_ratio`, and random `seed`  
            * Objective weights: `unmet_penalty`, `fill_bias_weight`, `move_penalty`  
        * **Outputs:**  
            * **Stacked dispatch plot** (grid, storage charge/discharge, unmet demand, demand, grid limit)  
            * **Diagnostics dashboard** (duration curves, histogram, and SoE trajectory)  
        * Acts as the **reference case** for all other pages, which extend or specialise the same model.
        """
    )

    st.subheader("1️⃣ ⏱ Main Dispatch (15-minute resolution)")
    st.markdown(
        """
        * **Purpose:** replicate the hourly model at finer granularity (Δt = 15 min).  
        * **Inputs:** grid/charge/discharge limits, usable energy, efficiencies, initial/final SoE, and demand shape.  
        * **Outputs:** stacked dispatch plot + optional dashboard (duration curves, histogram, SoE).  
        * **Default weights:** `unmet_penalty = 1e8`, `fill_bias = 1e-2`, `move_penalty = 0.1`.  
        """
    )

    st.subheader("2️⃣ 📐 Minimum Storage for Grid Limit")
    st.markdown(
        """
        * **Goal:** find the smallest storage capacity (kWh) that guarantees **no unmet demand**
          given a fixed grid limit.  
        * **Algorithm:**
            1. Generate demand (`generate_step_demand`)
            2. Compute discharge cap: `p_dis_max = max(demand) − grid_limit`
            3. **Bisection search** on storage energy using LP solves until unmet ≈ 0 kWh  
        * Returns minimal energy (kWh), discharge cap (kW), duration (h = E/P), and verification plot.  
        * Logic implemented in `find_min_storage_energy_bisect()` within `helper_functions_2.py`.
        """
    )

    st.subheader("3️⃣ 💰 Power / Energy Cost Trade-off")
    st.markdown(
        """
        * **Goal:** determine cost-optimal combination of grid limit, storage power, and energy.  
        * The app scans **grid-share ratios** \(r = G_{limit}/D_{peak}\) between `r_min` and `r_max`.  
        * For each r:
            * `grid_limit = r · max_demand`  
            * `discharge_cap = max_demand − grid_limit`  
            * Run minimal-energy sizing (as above)  
            * Compute total cost:  
              $$
              C = c_{power}\,P_{dis,max} + c_{energy}\,E_{nom}
              $$
        * The cheapest configuration is reported with KPIs and full dispatch visualisation.  
        * Function: `evaluate_grid_share_tradeoff()` in `helper_functions_2.py`.
        """
    )

    # ------------------------------------------------------------------
    # Helper modules
    # ------------------------------------------------------------------
    st.header("🧠 Helper Modules")
    st.markdown(
        """
        ### `helper_functions.py`
        * `SystemParams` dataclass — single source of all system inputs  
        * `build_and_solve_lp()` — main LP solver using PuLP  
        * `generate_step_demand()` — randomised two-level demand  
        * Plotly visualisers:
            * `make_main_dispatch_figure()` — stacked bars + demand
            * `make_dashboard_figure()` — 2×2 diagnostics dashboard

        ### `helper_functions_2.py`
        * `derive_discharge_cap_from_grid_limit()` — simple P-cap rule  
        * `find_min_storage_energy_bisect()` — iterative sizing via LP solves  
        * `storage_duration_hours()` — converts (E, P) → duration  
        * `evaluate_grid_share_tradeoff()` — scans grid-shares with cost model  
        * `make_demand_figure()` — compact preview of generated demand  
        """
    )

    # ------------------------------------------------------------------
    # Interpretation of results
    # ------------------------------------------------------------------
    st.header("📊 Interpreting Outputs")
    st.markdown(
        """
        * **Stacked dispatch plots** show how demand is met by grid and storage over time —
          charging appears **below zero**, discharging **above**.  
        * **Duration curves** compare demand vs supply (grid + discharge).  
        * **Histograms** reveal demand distribution (useful for sizing).  
        * **State-of-Energy plots** show how storage cycles during the horizon.  
        * **Key metrics:**
            * `duration_h = energy_kwh / p_dis_max_kw`
            * `unmet_kwh` — total energy shortfall (should be ≈ 0)
            * `total_cost_usd` — evaluated via chosen cost coefficients
        """
    )

    # ------------------------------------------------------------------
    # Mathematical note for sizing LP
    # ------------------------------------------------------------------
    with st.expander("🔍 Mathematical Formulation — Sizing LP (used in Pages 2 & 4)"):
        st.markdown(
            r"""
            The sizing problem iteratively calls the dispatch LP with increasing
            \(E_{nom}\) until:
            $$
            \sum_h U_h \,\Delta t \le 10^{-6}\text{ kWh.}
            $$
            The outer loop (bisection) converges to the minimal energy ensuring no unmet
            demand, optionally adding a safety margin ΔE.
            """
        )

    # ------------------------------------------------------------------
    # Practical guidance
    # ------------------------------------------------------------------
    st.header("💡 Practical Guidance & Tips")
    st.markdown(
        """
        * The app is deterministic given a `seed` → repeatable results.  
        * High `unmet_penalty` (> 1e8) enforces zero unmet energy but may increase solver time.  
        * Adjust `fill_bias_weight` if the model leaves the battery empty too early.  
        * Use 15-minute pages for more realistic short-term behavior.  
        * For sensitivity studies, sweep peak demand, avg/peak ratio, or efficiencies.  
        * The Home page is ideal for **quick validation**, while Pages 2–4 support
          **design and optimisation** workflows.
        """
    )

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    st.info(
        "Source modules: `helper_functions.py`, `helper_functions_2.py`  •  "
        "Solvers: PuLP / CBC  •  Visualization: Plotly  •  Framework: Streamlit",
        icon="🧩",
    )


if __name__ == "__main__":
    main()
