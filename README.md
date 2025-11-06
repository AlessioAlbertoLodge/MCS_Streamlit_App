# Grid + Storage Optimizer (Streamlit)

Interactive Streamlit app for exploring how a grid connection interacts with an energy-storage system (battery or equivalent).  
Solve dispatch problems as a linear program (LP), find minimum energy to meet a grid limit, and study cost trade-offs between power, energy, and grid size.  
Uses Streamlit + PuLP + Plotly.

------------------------------------------------------------
## Features

- Dispatch simulation at hourly or 15-minute resolution
- Minimum storage sizing for a fixed grid cap
- Power/Energy/Grid cost trade-off scan
- Deterministic synthetic demand generator (seeded)
- Unified LP formulation shared across all pages

Core stack: Streamlit UI, PuLP/CBC solver, Plotly charts.
------------------------------------------------------------
## App Pages

| Page | File | Purpose |
|------|------|----------|
| ⚡ Home – Grid + Storage Optimizer (Hourly) | Basic_Storage_Problem.py | Entry point; dispatch LP at 1-hour resolution with sidebar inputs |
| ⏱ Main Dispatch (15-min) | pages/1_Main Dispatch.py | Same LP, finer Δt = 0.25 h, inputs on main page |
| 📐 Minimum Storage for Grid Limit | pages/2_Minimum Storage for Grid Limit.py | Finds smallest storage energy (kWh) for zero unmet demand given grid cap |
| 💰 Power/Energy Cost Trade-off | pages/3_Power Energy Cost Trade-off.py | Scans grid-share ratios and reports cheapest power/energy/grid combo |
| 📚 Documentation | pages/3_Documentation.py | In-app help page |

Helpers: helper_functions.py, helper_functions_2.py
------------------------------------------------------------
## Quick Start

# 1) Clone the repo
git clone https://github.com/<your-username>/MCS_Streamlit_App.git
cd MCS_Streamlit_App

# 2) Create virtual environment (optional)
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run the app
streamlit run Basic_Storage_Problem.py

Open the printed URL (usually http://localhost:8501).
------------------------------------------------------------
## Inputs & Outputs

### Home / Main Dispatch
Inputs:
- grid_limit_kw, p_dis_max, p_ch_max
- usable_nominal_energy_kwh, eta_ch, eta_dis
- initial/final SoE
- demand shape: peak_kw, avg_to_peak_ratio, seed
- objective weights: unmet_penalty, fill_bias_weight, move_penalty

Outputs:
- Stacked dispatch plot (grid, charge/discharge, unmet, demand)
- Diagnostics: duration curves, histogram, SoE trajectory

Default weights (15-min page):
unmet_penalty = 1e8  
fill_bias_weight = 1e-2  
move_penalty = 0.1
### Minimum Storage for Grid Limit
Goal: smallest energy E_nom ensuring unmet ≈ 0
Method:
1) Generate demand
2) p_dis_max = max(demand) – grid_limit
3) Bisection on E_nom with LP solves
Outputs: E_nom, p_dis_max, duration (h = E/P), plot

### Power / Energy Cost Trade-off
Goal: minimize total cost
Scan grid-share ratio r = G_limit / D_peak in [r_min, r_max]
For each r:
- grid_limit = r * max_demand
- p_dis_max = max_demand – grid_limit
- Solve for minimal energy
Cost: C = c_power * P_dis,max + c_energy * E_nom

Outputs: cheapest configuration, KPIs, and dispatch.
------------------------------------------------------------
## Mathematical Model

Decision variables (per step h): G_h, P_dis_h, P_ch_h, U_h, E_h

Objective:
min ( w_U ΣU_h + w_F Σ(E_nom – E_h) + w_M Σ(P_dis_h + P_ch_h) )

Constraints:
1. Power balance: G_h + P_dis_h – P_ch_h + U_h = D_h
2. Grid limit: 0 ≤ G_h ≤ G_max
3. Storage power: 0 ≤ P_dis_h ≤ P_dis_max, 0 ≤ P_ch_h ≤ P_ch_max
4. Energy dynamics: E_{h+1} = E_h + η_ch P_ch_h Δt – (P_dis_h / η_dis) Δt
5. Bounds: 0 ≤ E_h ≤ E_nom, E_0 = E_init, optional E_H = E_final

Sizing LPs (pages 2 & 4) reuse this formulation while searching E_nom and grid ratio.
------------------------------------------------------------
## Demand Generator

Function: generate_step_demand()
- Two-level “low/peak” demand shape
- Controlled by peak_kw, avg_to_peak_ratio, duration, resolution (1h or 0.25h)
- Random seed for reproducibility

------------------------------------------------------------
## Tips

- Use 15-minute page for short-term dynamics.
- Keep w_U high (>1e8) to force zero unmet energy.
- Increase fill_bias_weight if SoE depletes too early.
- Sweep peak_kw, avg_to_peak_ratio, or efficiencies for sensitivity tests.
------------------------------------------------------------
## Repository Layout

MCS_Streamlit_App/
├─ Basic_Storage_Problem.py
├─ pages/
│  ├─ 1_Main Dispatch.py
│  ├─ 2_Minimum Storage for Grid Limit.py
│  ├─ 3_Power Energy Cost Trade-off.py
│  └─ 3_Documentation.py
├─ helper_functions.py
├─ helper_functions_2.py
├─ requirements.txt
├─ LICENSE
└─ README.md

------------------------------------------------------------
## Development

streamlit run Basic_Storage_Problem.py --server.runOnSave true

------------------------------------------------------------
## License

Licensed under the Apache-2.0 License.  
See LICENSE for details.

------------------------------------------------------------
## Contact

Maintainer: Alessio A. Lodge  
Repo: AlessioAlbertoLodge/MCS_Streamlit_App
