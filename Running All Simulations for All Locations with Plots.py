#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import math
from io import BytesIO
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Plotly (identical dispatch panel as in Streamlit)
import plotly.io as pio
from helper_functions import make_main_dispatch_figure

# ── CONFIG ────────────────────────────────────────────────────────────────────
LOADS_DIR        = r"C:\Users\lodgeaa\Downloads\28182251\Codes & Data Alessio\Data\Loads"
SAVE_FIG_PATH    = os.path.join(LOADS_DIR, "summary_streamlit_style_COMPOSITE.png")
SAVE_SUMMARY_CSV = os.path.join(LOADS_DIR, "summary_min_storage_PUh_and_duration.csv")
TMP_DIR          = os.path.join(LOADS_DIR, "_tmp_composite")
os.makedirs(TMP_DIR, exist_ok=True)

YEARS_PREF       = [2021, 2020]   # prefer 2021 then 2020
GRID_LIMIT_PCT   = 70.0           # grid limit as % of peak → PU

# Ridgeline aesthetics (match previous)
VERTICAL_SPACING = 1.2
LINEWIDTH        = 0.9
FILL_ALPHA       = 0.35
SMOOTH_WINDOW    = 3

# Composite layout (pixels)
LEFT_WIDTH       = 900     # left ridgeline width
CENTER_WIDTH     = 1100    # plotly dispatch width
RIGHT_COL_WIDTH  = 420     # two numeric columns region
ROW_HEIGHT       = 160     # per-row height
HEADER_HEIGHT    = 80
GAP_X            = 18
GAP_Y            = 12
FONT_NAME        = None    # or path to a .ttf font
FONT_SIZE_ROW = 70
FONT_SIZE_HEAD = 70

# ── IO helpers ────────────────────────────────────────────────────────────────
def _read_location_series(fp: str,
                          years_pref: List[int]) -> Tuple[str, int, np.ndarray, np.ndarray, float]:
    """
    Load a location CSV (avg_load_in_<loc>.csv) and pick one year series.
    Returns: (location_label, year, time_hours[N], demand_pu[N], dt_hours)
    Normalizes to PU if file has only kW columns.
    """
    base = os.path.basename(fp)
    m = re.match(r"avg_load_in_(.+)\.csv$", base, re.IGNORECASE)
    location = m.group(1).replace("_", " ") if m else base

    df = pd.read_csv(fp)

    norm_cols = {int(c.split("_")[1]): c for c in df.columns if re.fullmatch(r"AvgLoadNorm_\d{4}", c)}
    kw_cols   = {int(c.split("_")[1]): c for c in df.columns if re.fullmatch(r"AvgLoad_\d{4}_kW", c)}

    chosen_year = None
    chosen_col  = None
    is_norm     = False

    for y in years_pref:
        if y in norm_cols:
            chosen_year, chosen_col, is_norm = y, norm_cols[y], True
            break
        if y in kw_cols:
            chosen_year, chosen_col, is_norm = y, kw_cols[y], False
            break

    if chosen_year is None:
        if norm_cols:
            chosen_year = sorted(norm_cols.keys())[-1]
            chosen_col  = norm_cols[chosen_year]
            is_norm     = True
        elif kw_cols:
            chosen_year = sorted(kw_cols.keys())[-1]
            chosen_col  = kw_cols[chosen_year]
            is_norm     = False
        else:
            raise ValueError(f"No AvgLoadNorm_* or AvgLoad_*_kW columns in {fp}")

    # Time axis → hours
    if "Time" in df.columns:
        try:
            s = df["Time"].astype(str).str.strip()
            hh = s.str.extract(r"^(\d{1,2}):")[0].astype(float)
            mm = s.str.extract(r":(\d{2})$")[0].astype(float)
            time_h = (hh + mm/60.0).to_numpy()
        except Exception:
            num = pd.to_numeric(df["Time"], errors="coerce")
            if num.notna().all():
                if num.max() <= 95 and num.min() >= 0 and num.nunique() >= 10:
                    time_h = (num.to_numpy() * 15.0) / 60.0
                else:
                    time_h = num.to_numpy()
            else:
                raise ValueError(f"Cannot parse 'Time' in {fp}.")
    else:
        # fallback: assume full-day evenly spaced
        N = len(df)
        time_h = np.linspace(0, 24, N, endpoint=False)

    y = pd.to_numeric(df[chosen_col], errors="coerce").to_numpy(dtype=float)
    good = ~np.isnan(time_h) & ~np.isnan(y)
    time_h = time_h[good]
    y = y[good]

    order = np.argsort(time_h)
    time_h = time_h[order]
    y = y[order]
    time_h, uniq_idx = np.unique(time_h, return_index=True)
    y = y[uniq_idx]

    # Normalize if kW
    if not is_norm:
        maxv = np.nanmax(y) if y.size else 1.0
        y = y / max(maxv, 1e-9)

    # Optional smoothing
    if SMOOTH_WINDOW > 1:
        y = pd.Series(y).rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean().to_numpy()

    # dt (hours), snap to common minute steps
    diffs = np.diff(time_h)
    diffs = diffs[diffs > 0]
    if diffs.size:
        dt_h = float(np.median(diffs))
        common = np.array([1.0, 0.5, 0.25, 1/6, 1/12, 1/24, 1/48, 1/96])
        dt_h = float(common[np.argmin(np.abs(common - dt_h))])
    else:
        dt_h = 24.0 / max(1, len(time_h))

    return location, chosen_year, time_h, y, dt_h

# ── PU dispatch model (simulation + bisection) ───────────────────────────────
def _simulate_dispatch_PU(demand_pu: np.ndarray,
                          grid_limit_pu: float,
                          E_pu_h: float,
                          dt_h: float,
                          eta_ch: float = 1.0,
                          eta_dis: float = 1.0):
    n = len(demand_pu)
    grid   = np.zeros(n)
    p_dis  = np.zeros(n)
    p_ch   = np.zeros(n)
    soe    = np.zeros(n+1)
    Emax   = max(E_pu_h, 0.0)
    unmet  = 0.0

    for t in range(n):
        d = demand_pu[t]
        if d > grid_limit_pu:
            need = d - grid_limit_pu
            avail_pw = (soe[t] * eta_dis) / max(dt_h, 1e-12) if eta_dis > 0 else 0.0
            pd = min(need, max(avail_pw, 0.0))
            p_dis[t] = pd
            grid[t]  = d - pd
            soe[t+1] = soe[t] - (pd * dt_h) / max(eta_dis, 1e-12)
            if (d - pd) > grid_limit_pu + 1e-12:
                unmet += ((d - pd) - grid_limit_pu) * dt_h
        else:
            headroom = max(grid_limit_pu - d, 0.0)
            cap_pw = ((Emax - soe[t]) / max(dt_h, 1e-12)) * eta_ch if eta_ch > 0 else 0.0
            pc = min(headroom, max(cap_pw, 0.0))
            p_ch[t] = pc
            grid[t] = d + pc
            soe[t+1] = soe[t] + (pc * dt_h) / max(eta_ch, 1e-12)

        soe[t+1] = min(max(soe[t+1], 0.0), Emax)

    return grid, p_dis, p_ch, soe[:-1], unmet

def find_min_storage_energy_PU(demand_pu: np.ndarray,
                               grid_limit_pu: float,
                               dt_h: float,
                               eta_ch: float = 1.0,
                               eta_dis: float = 1.0,
                               tol_puh: float = 5e-4,
                               max_iter: int = 60):
    excess = np.maximum(demand_pu - grid_limit_pu, 0.0)
    E_hi = float(np.sum(excess) * dt_h) + 1e-6
    E_lo = 0.0
    best = (E_hi, None, math.inf)

    for _ in range(max_iter):
        E_mid = 0.5 * (E_lo + E_hi)
        grid, p_dis, p_ch, soe, unmet = _simulate_dispatch_PU(demand_pu, grid_limit_pu, E_mid, dt_h, eta_ch, eta_dis)
        if unmet <= tol_puh:
            best = (E_mid, (grid, p_dis, p_ch, soe), unmet)
            E_hi = E_mid
        else:
            E_lo = E_mid
        if (E_hi - E_lo) <= tol_puh * 0.5:
            break

    E_min = best[0]
    if best[1] is None:
        grid, p_dis, p_ch, soe, unmet = _simulate_dispatch_PU(demand_pu, grid_limit_pu, E_min, dt_h, eta_ch, eta_dis)
    else:
        grid, p_dis, p_ch, soe = best[1]
        unmet = best[2]
    sol = {"grid": grid, "p_dis": p_dis, "p_ch": p_ch, "soe": soe, "grid_limit_pu": grid_limit_pu}
    return E_min, sol, unmet

def storage_duration_hours(E_puh: float, p_dis_max_pu: float) -> float:
    return 0.0 if p_dis_max_pu <= 1e-12 else E_puh / p_dis_max_pu

# ── Ridgeline per-row image (Matplotlib) ─────────────────────────────────────
def _render_single_ridge_row(location: str, year: int, y_pu: np.ndarray, width_px: int, height_px: int) -> Image.Image:
    n = len(y_pu)
    fig_w = width_px / 100.0
    fig_h = height_px / 100.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)

    x = np.arange(n)
    offset = 0.0
    ax.fill_between(x, offset, offset + y_pu, alpha=FILL_ALPHA, linewidth=0)
    ax.plot(x, offset + y_pu, linewidth=LINEWIDTH)

    ax.text(-0.02 * n, offset + 0.55, f"{location} {year}", va='center', ha='right', fontsize=10)

    dt_h = 24.0 / n if n > 0 else 0.25
    ticks = np.linspace(0, n, 6)
    labels = []
    for tp in ticks:
        mins = int(round(tp * dt_h * 60))
        h, m = divmod(mins, 60)
        if h > 24: h, m = 24, 0
        labels.append(f"{h:02d}:{m:02d}")
    ax.set_xlim(0, n-1)
    ax.set_xticks(np.clip(ticks, 0, n-1))
    ax.set_xticklabels(labels)
    ax.set_yticks([])
    ax.spines[['left', 'right', 'top']].set_visible(False)
    ax.spines['bottom'].set_alpha(0.6)

    plt.tight_layout(pad=0.6)
    tmp_path = os.path.join(TMP_DIR, f"ridge_{re.sub(r'[^A-Za-z0-9]+','_',location)}_{year}.png")
    fig.savefig(tmp_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return Image.open(tmp_path)


# ── Dispatch per-row image (Plotly → in-memory PNG → PIL) ────────────────────
def _render_dispatch_row_plotly(
    demand_pu: np.ndarray,
    grid_pu: np.ndarray,
    p_dis_pu: np.ndarray,
    p_ch_pu: np.ndarray,
    grid_cap_pu: float,
    width_px: int,
    height_px: int,
    title: str = "",
) -> Image.Image:
    """Render dispatch chart identical to Streamlit version, but without axes or tick labels."""
    H = len(demand_pu)
    unmet = np.zeros(H)

    fig = make_main_dispatch_figure(
        demand=demand_pu.tolist(),
        grid=grid_pu.tolist(),
        storage_discharge=p_dis_pu.tolist(),
        storage_charge=p_ch_pu.tolist(),
        unmet=unmet.tolist(),
        grid_limit_kw=float(grid_cap_pu),
        title=title,
    )

    # Hide axis titles, ticks, and grid lines, but keep legends
    fig.update_layout(
        showlegend=True,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
        margin=dict(l=20, r=20, t=10, b=10),
        title="",
        plot_bgcolor="white",
        paper_bgcolor="white",
        barmode="stack",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=20)
        ),
    )

    try:
        png_bytes = fig.to_image(format="png", width=width_px, height=height_px, scale=2)
    except Exception as e:
        raise RuntimeError("Plotly static export failed. Install `kaleido` and restart.") from e

    return Image.open(BytesIO(png_bytes))

# ── Compose rows into a single canvas ────────────────────────────────────────
def _get_font(size):
    if FONT_NAME and os.path.exists(FONT_NAME):
        try:
            return ImageFont.truetype(FONT_NAME, size)
        except Exception:
            pass
    return ImageFont.load_default()

def build_composite(rows, save_path):
    """
    rows: list of dicts with keys:
        'ridge_img'     PIL.Image
        'dispatch_img'  PIL.Image
        'duration_h'    float
        'emin_puh'      float
    """
    n = len(rows)
    width = LEFT_WIDTH + GAP_X + CENTER_WIDTH + GAP_X + RIGHT_COL_WIDTH + 2*GAP_X
    height = HEADER_HEIGHT + GAP_Y + n*ROW_HEIGHT + GAP_Y

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    font_head = _get_font(FONT_SIZE_HEAD)
    font_row  = _get_font(FONT_SIZE_ROW)

    # Right column headers
    x_right = LEFT_WIDTH + GAP_X + CENTER_WIDTH + GAP_X
    draw.text((x_right + 20, 20), "Duration [h]",       fill=(0,0,0), font=font_head)
    draw.text((x_right + 220, 20), "Min Storage [PUh]", fill=(0,0,0), font=font_head)

    y0 = HEADER_HEIGHT
    for i, r in enumerate(rows):
        y = y0 + i*ROW_HEIGHT

        # paste ridge
        ridge = r['ridge_img']
        if ridge.mode not in ("RGB", "RGBA", "P"):
            ridge = ridge.convert("RGBA")
        ridge = ridge.resize((LEFT_WIDTH, ROW_HEIGHT - 10), Image.LANCZOS)
        canvas.paste(ridge, (GAP_X, y))

        # paste dispatch
        disp = r['dispatch_img']
        if disp.mode not in ("RGB", "RGBA", "P"):
            disp = disp.convert("RGBA")
        disp = disp.resize((CENTER_WIDTH, ROW_HEIGHT - 10), Image.LANCZOS)
        canvas.paste(disp, (LEFT_WIDTH + 2*GAP_X, y))

        # numbers
        dur_txt = f"{r['duration_h']:.2f}"
        e_txt   = f"{r['emin_puh']:.3f}"
        draw.text((x_right + 20,  y + (ROW_HEIGHT//2 - 10)), dur_txt, fill=(0,0,0), font=font_row)
        draw.text((x_right + 220, y + (ROW_HEIGHT//2 - 10)), e_txt,   fill=(0,0,0), font=font_row)

    canvas.save(save_path, "PNG")

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    pattern = os.path.join(LOADS_DIR, "avg_load_in_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files found at {pattern}")

    grid_limit_pu = GRID_LIMIT_PCT / 100.0

    summary_records = []
    row_images = []

    for fp in files:
        try:
            loc, year, time_h, y_pu, dt_h = _read_location_series(fp, YEARS_PREF)

            # Size min storage in PU (bisection)
            E_puh, sol, unmet = find_min_storage_energy_PU(
                demand_pu=y_pu,
                grid_limit_pu=grid_limit_pu,
                dt_h=dt_h,
                eta_ch=1.0,
                eta_dis=1.0,
                tol_puh=5e-4,
                max_iter=60,
            )

            # Add tiny cushion to guarantee strictly zero-unmet when exporting
            E_puh *= 1.002
            grid, p_dis, p_ch, soe, unmet2 = _simulate_dispatch_PU(
                y_pu, grid_limit_pu, E_puh, dt_h, eta_ch=1.0, eta_dis=1.0
            )
            sol = {"grid": grid, "p_dis": p_dis, "p_ch": p_ch, "soe": soe, "grid_limit_pu": grid_limit_pu}

            # Duration (PUh / PU)
            p_dis_max_pu = float(np.max(y_pu) - grid_limit_pu) if np.max(y_pu) > grid_limit_pu else 0.0
            duration_h = storage_duration_hours(E_puh, p_dis_max_pu)

            # Left ridgeline
            ridge_img = _render_single_ridge_row(loc, year, y_pu, LEFT_WIDTH, ROW_HEIGHT - 10)

            # Center dispatch (Plotly identical to Streamlit)
            disp_img = _render_dispatch_row_plotly(
                demand_pu=y_pu,
                grid_pu=sol["grid"],
                p_dis_pu=sol["p_dis"],
                p_ch_pu=sol["p_ch"],
                grid_cap_pu=grid_limit_pu,
                width_px=CENTER_WIDTH,
                height_px=ROW_HEIGHT - 10,
                title="",
            )

            row_images.append({
                "ridge_img": ridge_img,
                "dispatch_img": disp_img,
                "duration_h": duration_h,
                "emin_puh": E_puh,
            })

            print(f"[OK] {loc} {year} — E_min={E_puh:.4f} PUh, duration={duration_h:.3f} h, unmet≈{unmet2:.6f} PUh")

            summary_records.append((loc, year, E_puh, duration_h))

        except Exception as e:
            print(f"[SKIP] {os.path.basename(fp)} — {e}")

    # Save summary CSV
    df_sum = pd.DataFrame(summary_records, columns=["Location", "Year", "MinStorage_PUh", "Duration_h"])
    df_sum = df_sum.sort_values(by=["Location", "Year"])
    df_sum.to_csv(SAVE_SUMMARY_CSV, index=False)
    print(f"\nSaved summary CSV: {SAVE_SUMMARY_CSV}")

    # Build final composite
    build_composite(row_images, SAVE_FIG_PATH)
    print(f"Saved composite figure: {SAVE_FIG_PATH}")

if __name__ == "__main__":
    main()
