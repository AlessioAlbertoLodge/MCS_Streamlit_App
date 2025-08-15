import streamlit as st

# ----------------------------------------------------------------------------
# 📚 Documentation page for the Grid + Storage Optimizer
# ----------------------------------------------------------------------------
# This Streamlit page appears under `pages/` so that users can open it
# from the navigation sidebar and quickly understand:
#   • What the tool does
#   • Why it exists
#   • How to operate it effectively
# ----------------------------------------------------------------------------

def main():  # pragma: no cover
    """Render the documentation/help page for the Grid + Storage Optimizer app."""

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    st.title("📚 Documentation — Grid + Storage Optimizer")
    st.caption("Version 0.1 · last updated 2025-08-15")

    # ------------------------------------------------------------------
    # Context section
    # ------------------------------------------------------------------
    st.header("Context")
    st.markdown(
        """
        ### Purpose
        The Grid + Storage Optimizer models the hourly dispatch of a **grid connection**
        coupled with an **energy storage system** (e.g., a battery) over a given time horizon
        (default: 24 hours).

        The tool determines, for each hour:

        * How much power is drawn from the grid
        * How much the storage charges or discharges
        * Whether there is any **unmet demand**

        The dispatch is computed by solving a **linear program (LP)** that minimises
        unmet demand while respecting all grid and storage constraints.

        ### Why use it?
        This simulator allows planners, operators, and researchers to:
        * Test the effect of grid import limits
        * Explore different storage sizes and charge/discharge rates
        * Understand how storage shifts load and reduces grid peaks
        * See how objective weights (fill bias, movement penalty) affect schedules
        """
    )

    # ------------------------------------------------------------------
    # Usage section
    # ------------------------------------------------------------------
    st.header("Usage")

    st.markdown(
        """
        1. **Set Inputs in the Sidebar**  
           * **Hours** to simulate (default: 24)  
           * **Grid limit** (kW) — maximum allowable import from the grid  
           * **Storage limits** — maximum charge and discharge rates (kW)  
           * **Usable energy** — storage capacity (kWh)  
           * **Efficiencies** — charge and discharge efficiency (0–1)  
           * **Initial/Final SoE** — starting and (optional) enforced ending state-of-energy  
           * **Demand shape** — peak demand (kW), average-to-peak ratio, and random seed  
           * **Objective weights** — penalty for unmet demand, fill bias weight, move penalty

        2. **Run the model**  
           The app:
           * Generates a synthetic hourly demand profile using the peak, ratio, and seed
           * Builds and solves the LP dispatch model
           * Produces time series for grid power, storage charge/discharge, unmet demand, and SoE

        3. **View results**  
           * **Main stacked dispatch plot** — shows grid, storage charge/discharge, demand, and grid limit  
           * **Optional diagnostics dashboard** — duration curves, histogram, and SoE trajectory
        """
    )

    # ------------------------------------------------------------------
    # Mathematical formulation
    # ------------------------------------------------------------------
    with st.expander("Mathematical formulation of the dispatch model"):
        st.subheader("Objective")
        st.markdown(
            r"""
            \[
            \min \;
            \text{UnmetPenalty} \cdot \sum_h U_h \;+\;
            \text{FillBias} \cdot \sum_h (E_{\text{nom}} - E_h) \;+\;
            \text{MovePenalty} \cdot \sum_h (P^{\text{dis}}_h + P^{\text{ch}}_h)
            \]
            """
        )
        st.markdown(
            "* \(U_h\) — unmet demand in hour \(h\) (kW)  \n"
            "* \(E_h\) — energy stored at the end of hour \(h\) (kWh)  \n"
            "* \(P^{\\text{dis}}_h, P^{\\text{ch}}_h\) — discharge/charge power in hour \(h\) (kW)"
        )

        st.subheader("Constraints")
        st.markdown(
            r"""
            Power balance:
            \[
            G_h + P^{\text{dis}}_h - P^{\text{ch}}_h + U_h = D_h
            \]
            Grid limit:
            \[
            0 \leq G_h \leq G_{\max}
            \]
            Storage power limits:
            \[
            0 \leq P^{\text{dis}}_h \leq P^{\text{dis}}_{\max}, \quad
            0 \leq P^{\text{ch}}_h \leq P^{\text{ch}}_{\max}
            \]
            Energy dynamics:
            \[
            E_{h+1} = E_h + \eta_{\text{ch}} P^{\text{ch}}_h \Delta t -
                      \frac{P^{\text{dis}}_h}{\eta_{\text{dis}}} \Delta t
            \]
            SoE bounds:
            \[
            0 \leq E_h \leq E_{\text{nom}}
            \]
            """
        )

    # ------------------------------------------------------------------
    # Footer/help
    # ------------------------------------------------------------------
    st.info(
        "💡 Tip: Adjust the **Avg/Peak ratio** and **seed** to explore different demand shapes. "
        "The full optimisation logic and plotting utilities are in `helper_functions.py`.",
        icon="💡",
    )


if __name__ == "__main__":
    main()
