# ⚡ BESS Day-Ahead Market Trading Dashboard

![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b?logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/python-%3E=3.9-blue?logo=python&logoColor=white)

> **Interactive optimiser & back-tester for trading a Battery-Energy-Storage System (BESS) on the European day-ahead power market.**

The dashboard lets you **optimise a single day** or **back-test multiple days** of day-ahead prices, automatically fetched from the **ENTSO-E Transparency Platform**.  
The optimiser is formulated as a *linear-programming* problem solved with **cvxpy** in milliseconds.  
An optional **ageing-cost penalty** captures the revenue impact of faster battery degradation.

---

## ✨ Features

| Page | What it does |
|------|--------------|
| **Home** | Computes the *optimal 24-hour charge/discharge schedule* that maximises revenue for a chosen market day. Shows price curve, schedule and KPIs. |
| **Multi-Day Revenue** | Loops the optimiser over a date range to create a *historic revenue back-test* with daily & cumulative plots. |
| **Documentation** | In-app guide covering market context, maths, usage instructions and ageing-cost model. |

* **Battery ageing cost**: toggle on/off and choose LFP or NMC empirical parameters.  
* **Full ENTSO-E API integration**: no manual CSVs.  
* **Beautiful charts** with Matplotlib & Streamlit native components.  
* **CSV export-ready** dataframes for further analysis.  

---
## 🚀 Quick start

### 1️⃣ Clone & install
```bash
git clone https://github.com/<your-username>/bess-dayahead-dashboard.git
cd bess-dayahead-dashboard
pip install -r requirements.txt
```

### 2️⃣ Configure the ENTSO-E API key
Create a file `.streamlit/secrets.toml`:
```toml
[api_keys]
entsoe = "YOUR_API_KEY_HERE"
```
—or export an environment variable `ENTSOE_API_KEY` instead.

### 3️⃣ Run the app
```bash
streamlit run Home.py
```
Open the printed URL (usually [http://localhost:8501](http://localhost:8501)).

---

## 🖥️ Repository layout
```
├── Home.py                         # Single-day optimiser page
├── pages/
│   ├── 02_multi_day_revenue.py    # Multi-day back-test page
│   └── 03_Documentation.py        # In-app documentation
├── src/                           # Core logic modules
│   ├── entsoe_prices.py
│   ├── optimize_battery_power_schedule.py
│   ├── optimize_battery_power_schedule_with_ageing.py
│   ├── plot_day_ahead_market_prices.py
│   ├── bess_schedule_plotter.py
│   ├── degradation.py
│   └── ...
├── requirements.txt
└── README.md
```

## ⚙️ Mathematical formulation (extended)

The optimiser is a 24-step **linear programming (LP)** model solved with `cvxpy`.

### Decision variables

| Symbol     | Description                                              | Units |
|------------|----------------------------------------------------------|--------|
| Pₕ         | Battery power in hour h (charging < 0, discharging > 0) | kW     |
| SOCₕ       | State-of-charge at end of hour h                        | –      |

### Parameters

| Symbol                    | Description                                      |
|---------------------------|--------------------------------------------------|
| Priceₕ                   | Day-ahead price for hour h (€/MWh)              |
| E_nom                    | Nominal usable energy capacity (kWh)            |
| P_ch,max, P_dis,max      | Charge / discharge power limits (kW)            |
| η_ch, η_dis              | Charge / discharge efficiencies (fraction)      |
| c_deg                    | (Optional) degradation cost coefficient (€/kWh) |

### Objective

Maximise net revenue **minus** optional degradation cost:

**Maximise:**
```
Σₕ [ Pₕ × Priceₕ - c_deg × |Pₕ| ] × Δt, for h = 1 to 24
```
If ageing-cost mode is OFF, set `c_deg = 0`.

### Constraints

1. **SOC dynamics**
```
SOCₕ = SOC₍ₕ₋₁₎ + [ η_ch × max(0, -Pₕ) - (1/η_dis) × max(0, Pₕ) ] / E_nom × Δt
```

2. **SOC bounds**
```
0 ≤ SOCₕ ≤ 1     for all h
```

3. **Power limits**
```
-P_ch,max ≤ Pₕ ≤ P_dis,max     for all h
```

4. **Initial SOC**
```
SOC₀ = SOC_start
```

Note: `max(0, x)` denotes the positive part of x, ensuring efficiencies only apply during charging or discharging as appropriate. All relationships are linear, so the model yields a **globally optimal** schedule in milliseconds.



## 📸 Screenshots
| Home – Single-day optimisation | Multi-day back-test |
|--------------------------------|---------------------|
| <img src="docs/screenshot_single_day.png" width="45%"> | <img src="docs/screenshot_multi_day.png" width="45%"> |
*(Add your own screenshots in `docs/` or remove this section.)*

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss your ideas.

1. Fork the project  
2. Create your feature branch (`git checkout -b feature/foo`)  
3. Commit your changes (`git commit -am 'Add some foo'`)  
4. Push to the branch (`git push origin feature/foo`)  
5. Open a Pull Request  

### Development tips  
&nbsp;&nbsp;&nbsp;&nbsp;```bash  
&nbsp;&nbsp;&nbsp;&nbsp;# optional: automatically reload when editing  
&nbsp;&nbsp;&nbsp;&nbsp;streamlit run Home.py --server.runOnSave true  
&nbsp;&nbsp;&nbsp;&nbsp;```  

---

## 📄 License
Distributed under the **MIT License**. See `LICENSE` for details.

---

## 💬 Contact
Alessio Lodge · <alessio.lodge@example.com>  
Project link: <https://github.com/<your-username>/bess-dayahead-dashboard>
