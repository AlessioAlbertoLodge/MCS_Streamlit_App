import streamlit as st

# ----------------------------------------------------------------------------
# 📚 Documentation page for the BESS Day-Ahead Market Optimiser
# ----------------------------------------------------------------------------
# This Streamlit page is displayed under ``pages/`` so that users can open it
# from the navbar and quickly understand what the tool is, why it exists and
# how to operate it.
# ----------------------------------------------------------------------------

def main():  # pragma: no cover
    """Render the documentation page."""

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    st.title("Documentation")
    st.caption("Version 0.3 · last updated 2025‑05‑25")

    # ------------------------------------------------------------------
    # Context section
    # ------------------------------------------------------------------
    st.header("Context")
    st.markdown(
        """
        ### Day‑Ahead Trading of Electricity
        *Day‑ahead trading* refers to buying and selling electricity **one day before**
        physical delivery. A daily auction held by the power exchange (e.g. **EPEX
        SPOT**, **Nord Pool**, **PJM**) produces **24 hourly prices** for the next day.

        Market participants – called **balancing‑responsible parties (BRPs)** – use the
        auction to keep the net position of their *balancing group* at zero. Surplus
        generation is sold; unexpected demand is bought.

        ### Why optimise a Battery Energy Storage System (BESS)?
        A BESS can **charge when prices are low (or negative)** and **discharge when they
        are high**, capturing the spread and supporting grid flexibility. The goal is to
        choose an hourly power profile (schedule) that maximises revenue while respecting
        battery *state‑of‑charge (SOC)* limits.

        ### Linear Programming (LP)
        The scheduling task is formulated as a **linear‑programming (LP) problem** – a
        linear objective subject to linear constraints. LP guarantees a globally
        optimal solution in milliseconds (solved here with
        `cvxpy`).

        ### Day-Ahead Market Data
        While its is not possible in reality to know the prices of the day-ahead market of a following day,
        this programme allows to run a ''backtest'', simulating the revenues made in the ideal scenario of being able to train arbitrary amounts of energy
        with a known prices spread. The data is provided via API from the entsoe transparency platform.

        ### Cost penalty for ageing
        Cyling the battery once a day or twice a day, while maintaing different average SOCs will have different effects of 
        the ageing of the BESS. We can incoporate a cost to include the burden of degrading more the BESS when it is used more.
        By modelling the capacity fade, we can compute a cost by attributing a capacity fade of 40% to a loss equal to the inital cost of the system.
        Each fraction of that capacity loss with therefore 'cost' a share of the initial CAPEX.
        """,
        unsafe_allow_html=False,
    )

    # ------------------------------------------------------------------
    # Usage section
    # ------------------------------------------------------------------
    st.header("Usage")

    st.markdown(
        """
        1. **Select Dates and Market**  
           Select the market (DE, FR, ES etc.) and date (i.e. 2025-05-25) to simulate revenues on day-ahead market. 
           This will load the day-ahead data  from the ENTSOE Transparency Platform**.

        2. **Define System conditions**  
           * Battery **energy capacity** (kWh) 
           * Starting **SOC** (0 – 100 %)   
           * Charge/Discharge **power limit** (kW)
           * Optional: add usage-aware revenue which adds a cost to abusive usage conditions**

        3. **Run optimisation**  
           The app solves the LP and returns: 
           * The **expected revenue** for the day 
           * An **optimal power schedule** (kW per hour)  
           * The resulting **SOC trajectory**  

        4. **Inspect results**  
           Interactive charts display price, power and SOC. Download data as CSV for
           further analysis.
        """,
        unsafe_allow_html=False,
    )

    # ------------------------------------------------------------------
    # Mathematical formulation (now in an expander with proper LaTeX)
    # ------------------------------------------------------------------
    with st.expander("Mathematical formulation of Optimal Power Dispatch (no-usage penalty)", expanded=False):
        st.subheader("Optimisation objective")
        st.latex(r"\max_{P_1,\dots,P_{24}} \; \sum_{h=1}^{24} P_h\,Price_h\,\Delta t")
        st.markdown(
            "* **Power**: $P_h < 0$ when charging, $P_h > 0$ when discharging (kW)  \n"
            "* **Price**: $Price_h$ is the day‑ahead price for hour $h$ (€/MWh)  \n"
            "* **Time‑step**: $\\Delta t = 1\\,\\text{h}$",
            unsafe_allow_html=False,
        )

        st.subheader("State‑of‑Charge (SOC) dynamics")
        st.latex(r"\text{SOC}_{h} = \text{SOC}_{h-1} + \frac{P_h\,\Delta t}{E_{\text{nom}}}")
        st.latex(r"0 \le \text{SOC}_{h} \le 1 \qquad \forall\,h")

        st.subheader("Decision variables")
        st.markdown(
            "Hourly powers $P_1,\dots,P_{24}$. All relationships above are linear, so the"
            " problem is solvable with standard LP solvers in milliseconds.",
            unsafe_allow_html=False,
        )

    # ------------------------------------------------------------------
    # Footer/help
    # ------------------------------------------------------------------
    st.info(
        "Need help? Reach out via the *Feedback* link in the sidebar or consult the"
        " README on GitHub.",
        icon="💡",
    )


if __name__ == "__main__":
    main()
