import streamlit as st
import datetime
from src import entsoe_surrogate as es
from src import entsoe_prices as ep
from src import plot_day_ahead_market_prices as pu
from src import optimize_battery_power_schedule as ob
from src import bess_schedule_plotter as ps
from src import degradation as dg

# Initialize session state flags for simulations
if 'run_ageing_sim' not in st.session_state:
    st.session_state.run_ageing_sim = False
if 'run_forecast_sim' not in st.session_state:
    st.session_state.run_forecast_sim = False

st.set_page_config(
    page_title="BESS Optimizer Dashboard",
    page_icon="⚡",
)
st.title("BESS Day-Ahead Market Trading")
st.write(
    "BESS Day-Ahead Market Trading is a tool written by Alessio Lodge (TNO) "
    "that aims at providing a platform to evaluate BESS scheduling on the day-ahead market."
)

# 1️⃣ Inputs
st.subheader("🌐📅 Choose Your Market and Date")
col1, col2 = st.columns(2)
with col1:
    market = st.text_input(
        "Market", "DE", max_chars=2,
        help="ISO 2-letter code (DE, NL, FR…)"
    )
with col2:
    trade_day = st.date_input(
        "Day", value=datetime.date.today(),
        max_value=datetime.date.today(),
        format="YYYY-MM-DD",
        help="Delivery date of the 24h block."
    )

st.subheader("🔧🔋 Define Your System")
c1, c2 = st.columns(2)
c3, c4 = st.columns(2)
with c1:
    capacity_kwh = st.number_input(
        "Battery Capacity [kWh]", 0.0, step=0.1, format="%.1f", value=1000.0,
        help="Usable energy the battery can store."
    )
with c2:
    start_soc = st.number_input(
        "Starting SOC [%]", 0, 100, value=50, step=1,
        help="State of charge at the beginning (0–100%)."
    )
with c3:
    p_discharge_max = st.number_input(
        "Max Discharge Power [kW]", 0.0, step=0.1, format="%.1f", value=500.0,
        help="Max discharging power (kW)."
    )
with c4:
    p_charge_max = st.number_input(
        "Max Charge Power [kW]", 0.0, step=0.1, format="%.1f", value=500.0,
        help="Max charging power (kW)."
    )

# 2️⃣ Advanced Settings
with st.expander("Advanced Settings"):
    # Degradation cost toggle
    add_deg_cost = st.toggle(
        "Add cost based on battery degradation?",
        value=False,
        help="Include a usage-cost penalty?"
    )
    if add_deg_cost:
        chemistry = st.selectbox(
            "Chemistry", ("LFP", "NMC"), index=0,
            help="Battery chemistry."
        )
        optimisation_routine = st.selectbox(
            "Select Optimization Routine",
            ("1: Non-linear LP", "2: Linear LP with best profit"),
            index=0,
            help="Solver variant (both identical for now)."
        )
        if st.button("🔄 Run simulation with ageing optimization", key="run_ageing"):
            st.session_state.run_ageing_sim = True

    # Imperfect knowledge toggle
    add_imperfect_knowledge = st.toggle(
        "Add imperfect knowledge of Day-ahead market",
        value=False,
        help=(
            "The current model evaluates a schedule using the data of the day-ahead market, "
            "known from 12:00 onwards. Selecting this method creates a schedule based on a forecast."
        )
    )
    if add_imperfect_knowledge:
        forecast_method = st.selectbox(
            "Select Forecasting Method",
            ("1: Statistical Method: LEAR", "2: Machine Learning Method: DNN"),
            index=0,
            help="Forecasting method for imperfect market knowledge."
        )
        if st.button("🔄 Run simulation with forecast", key="run_forecast"):
            st.session_state.run_forecast_sim = True

# 3️⃣ Fetch prices & run BASE simulation automatically
api_key_entsoe = st.secrets["api_keys"]["entsoe"]
prices = ep.get_day_ahead_prices_single_day(market, trade_day, api_key=api_key_entsoe)
result_base = ob.optimise_battery(
    prices, capacity_kwh, start_soc,
    p_charge_max, p_discharge_max,
    dt=1.0
)

# 4️⃣ Display BASE results when no advanced sim selected
if not st.session_state.run_ageing_sim and not st.session_state.run_forecast_sim:
    st.subheader("Day-Ahead Prices")
    st.pyplot(pu.plot_day_ahead_prices(prices), use_container_width=True)
    st.subheader("📈 Base Results (no ageing)")
    st.markdown(f"**Expected revenue:** {result_base['revenue']:,.2f} €")
    st.subheader("🕒 Schedule (no ageing penalty)")
    st.pyplot(ps.plot_bess_schedule(
        result_base["schedule"], country=market,
        day=trade_day, e_nom=capacity_kwh
    ), use_container_width=True)

# 5️⃣ Ageing simulation
if st.session_state.run_ageing_sim and add_deg_cost:
    deg_params = dg.load_emperical_degradation_parameters(chemistry)
    full_cost = capacity_kwh * 200
    result_age = ob.optimise_battery_with_ageing_dp(
        prices, capacity_kwh, start_soc,
        p_charge_max, p_discharge_max,
        298.15, deg_params, full_cost, dt=1.0
    )
    ref_act = ob.evaluate_schedule_with_ageing(
        result_base["P"], prices, capacity_kwh,
        start_soc, 298.15, deg_params,
        full_cost, dt=1.0
    )
    st.subheader("📈 Advanced Results with Ageing-Cost Penalty")
    st.pyplot(ps.plot_bess_schedule(
        result_age["schedule"], country=market,
        day=trade_day, e_nom=capacity_kwh
    ), use_container_width=True)
    ac, cc, cl = (
        result_age["ageing_cost"], result_age["ageing_cost_cyclic"], result_age["ageing_cost_calendar"]
    )
    st.write("** Cost-penalty Optimized Results::**")
    st.markdown(
        f"- Profit: **{result_age['profit']:,.2f} €**\n"
        f"- Revenue: **{result_age['revenue']:,.2f} €**\n"
        f"- Ageing cost: **{ac:,.2f} € (cyclic {cc:,.2f} €, calendar {cl:,.2f} €) €**"
    )

    st.write("** Un-optimized Results:**")
    st.markdown(
        f"- Profit: **{ref_act['profit']:,.2f} €**\n"
        f"- Revenue: **{ref_act['revenue']:,.2f} €**\n"
        f"- Ageing cost: **{ref_act['ageing_cost']:,.2f} €**"
    )
    # plot unoptimized
    st.pyplot(ps.plot_bess_schedule(
        ref_act["schedule"], country=market,
        day=trade_day, e_nom=capacity_kwh
    ), use_container_width=True)

# 6️⃣ Forecast-based simulation
if st.session_state.run_forecast_sim and add_imperfect_knowledge:
    # Placeholder for future forecast function
    known_prices = ep.get_day_ahead_prices_single_day(market, trade_day, api_key=api_key_entsoe)
    forecast_prices = known_prices # ep.get_forecasted_day_ahead_prices( market, trade_day, api_key=api_key_entsoe, method=forecast_method)
    result_fc = ob.optimise_battery(
        forecast_prices, capacity_kwh, start_soc,
        p_charge_max, p_discharge_max,
        dt=1.0)
    result_known_price = ob.optimise_battery(
        known_prices, capacity_kwh, start_soc,
        p_charge_max, p_discharge_max,
        dt=1.0)
    st.subheader("📈 Forecast-based Results (Imperfect Knowledge)") 
    st.pyplot(pu.plot_day_ahead_prices_normal_and_forecasted(known_prices, forecast_prices), use_container_width=True) 
    st.markdown(f"**Expected revenue (Imperfect Knowledge - Forecast Based):** {result_fc['revenue']:,.2f} €")
    st.markdown(f"**Expected revenue (Perfect Knowledge):** {result_known_price['revenue']:,.2f} €")
    
    st.write("** Schedule with Imperfect Knowledge of Prices:**")

    st.pyplot(ps.plot_bess_schedule(
        result_fc["schedule"], country=market,
        day=trade_day, e_nom=capacity_kwh
    ), use_container_width=True)

    st.write("** Schedule with Perfect Knowledge of Prices:**")

    st.pyplot(ps.plot_bess_schedule(
        result_known_price["schedule"], country=market,
        day=trade_day, e_nom=capacity_kwh
    ), use_container_width=True)