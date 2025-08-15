# pages/02_multi_day_revenue.py
import streamlit as st
import datetime as dt
import pandas as pd

from src import entsoe_prices as ep
from src import optimize_battery_power_schedule as ob
from src import degradation as dg
from src import revenue_plotter as rp  # put this import near the top of the file

# ──────────────────────────── Page setup ────────────────────────────
st.set_page_config(page_title="BESS Backtest on Day-Ahead Market Revenue New", page_icon="💶")
st.title("BESS Day-Ahead – Multi-Day Revenue New")

st.write(
    "Compute the total revenue a Battery-Energy-Storage System (BESS) would "
    "make on the day-ahead market over a user-defined date range."
)

# ──────────────────────────── Market & dates ────────────────────────
st.subheader("🌐📅 Market & Period")

row1_col1, row1_col2, row1_col3 = st.columns(3)

with row1_col1:
    market = st.text_input(
        "Market",
        value="DE",
        max_chars=3,
        help="ISO 2-letter code of the power exchange (DE, NL, FR, …)."
    )

today = dt.date.today()
with row1_col2:
    start_date = st.date_input(
        "Start date",
        value=today - dt.timedelta(days=7),
        max_value=today,
        format="YYYY-MM-DD"
    )
with row1_col3:
    end_date = st.date_input(
        "End date",
        value=today,
        min_value=start_date,
        max_value=today,
        format="YYYY-MM-DD"
    )

# ──────────────────────────── BESS definition ───────────────────────
st.subheader("🔧🔋 BESS Parameters")

row2_col1, row2_col2 = st.columns(2)
row3_col1, row3_col2 = st.columns(2)

with row2_col1:
    capacity_kwh = st.number_input(
        "Battery Capacity [kWh]",
        min_value=0.0, step=0.1, format="%.1f", value=1000.0
    )
with row2_col2:
    start_soc = st.number_input(
        "Starting SOC [%]",
        min_value=0, max_value=100, value=99, step=1
    )
with row3_col1:
    p_discharge_max = st.number_input(
        "Max Discharge Power [kW]",
        min_value=0.0, step=0.1, format="%.1f", value=500.0
    )
with row3_col2:
    p_charge_max = st.number_input(
        "Max Charge Power [kW]",
        min_value=0.0, step=0.1, format="%.1f", value=500.0
    )

# ──────────────────────────── Advanced settings ─────────────────────
with st.expander("Advanced Settings"):
    add_deg_cost = st.toggle(
        "Add cost based on battery degradation?",
        value=False
    )
    chemistry = None
    if add_deg_cost:
        chemistry = st.selectbox("Chemistry", options=("LFP", "NMC"), index=0)

# ──────────────────────────── Run optimisation ──────────────────────
    # ----------------------------------------------------------------
    # 1️⃣  FETCH ALL PRICES UP-FRONT  (show spinner while waiting)
    # ----------------------------------------------------------------
button_col, spinner_col = st.columns([1, 1])

with button_col:
    compute_btn = st.button("Compute revenue", use_container_width=True)

if compute_btn:
    if end_date < start_date:
        st.error("End date must be on or after start date.")
        st.stop()

    api_key_entsoe = st.secrets["api_keys"]["entsoe"]

    # ────────────── spinner (must use st.spinner) ──────────────
    with spinner_col:
        with st.spinner("Loading price data via ENTSO-E API…"):
            all_prices = ep.get_day_ahead_prices_range(
                country=market,
                start=start_date,
                end=end_date,
                api_key=api_key_entsoe,
            )
    all_prices["date"] = pd.to_datetime(all_prices["time"]).dt.date
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days     = len(date_range)

    daily_revenues = []
    progress_bar   = st.progress(0.0)

    # 2️⃣  DAY–BY–DAY REVENUE CALCULATION
    for i, (day, day_prices) in enumerate(all_prices.groupby("date"), start=1):
        prices = day_prices["price"].to_numpy()

        if add_deg_cost:
            _ = dg.load_emperical_degradation_parameters(chemistry)
            result = ob.optimise_battery(
                prices, capacity_kwh, start_soc,
                p_charge_max, p_discharge_max, dt=1.0
            )
        else:
            result = ob.optimise_battery(
                prices, capacity_kwh, start_soc,
                p_charge_max, p_discharge_max, dt=1.0
            )

        daily_revenues.append({"date": day, "revenue_eur": result["revenue"]})
        progress_bar.progress(i / n_days)

    revenue_df    = pd.DataFrame(daily_revenues)
    total_revenue = revenue_df["revenue_eur"].sum()

    st.markdown(
        f"<p style='margin-bottom:0.5em;'>"
        f"Total revenue from {start_date} to {end_date}: "
        f"<strong>{total_revenue:,.2f} €</strong>"
        f"</p>",
        unsafe_allow_html=True,
    )

    st.subheader("Daily revenue breakdown")
    st.dataframe(
        revenue_df.style.format({"revenue_eur": "€{:,.2f}"}),
        height=min(250, 30 + len(revenue_df) * 30),
        use_container_width=True
    )

    st.subheader("📊 Daily & cumulative revenue")
    fig_revenue = rp.plot_daily_and_cumulative_revenue(revenue_df)
    st.pyplot(fig_revenue, use_container_width=True)