# helper_functions_3.py
from __future__ import annotations
import io
import re
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

def _to_hours_from_hhmm(time_col: pd.Series) -> np.ndarray:
    s = time_col.astype(str).str.strip()
    hh = s.str.extract(r'^(\d{1,2}):')[0].astype(float)
    mm = s.str.extract(r':(\d{2})$')[0].astype(float)
    return (hh + mm/60.0).to_numpy()

def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def load_demand_csv(
    file,
    time_unit: str = "hours",
    year: Optional[int] = None,
    prefer_years: Optional[List[int]] = None,
    prefer_normalized: bool = True,   # <— NEW
) -> Tuple[np.ndarray, List[float]]:
    """
    Load a CSV as time-vs-power.

    Supported formats:
    1) Generic 2-column: [time, power] (time in hours or minutes via `time_unit`)
    2) Location avg-load: 'Time' + one or more of:
         - 'AvgLoadNorm_<YYYY>'   (unitless, 0..1)  ← preferred when present
         - 'AvgLoad_<YYYY>_kW'    (absolute kW)

    Parameters
    ----------
    file : UploadedFile, BytesIO, or str
    time_unit : {"hours","minutes"} for generic 2-col
    year : desired year for avg-load format (optional)
    prefer_years : priority order if year is None; default [2021, 2020]
    prefer_normalized : if True, pick normalized columns when available

    Returns
    -------
    time_hours : np.ndarray
    demand_vals : list[float]   # normalized or kW depending on column chosen
    """
    # read file/buffer
    if isinstance(file, (io.BytesIO, io.BufferedReader)):
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(file)

    if df.empty:
        raise ValueError("CSV is empty.")

    # Detect avg-load formats
    norm_cols = [c for c in df.columns if re.fullmatch(r'AvgLoadNorm_\d{4}', c or "")]
    kw_cols   = [c for c in df.columns if re.fullmatch(r'AvgLoad_\d{4}_kW', c or "")]

    if "Time" in df.columns and (norm_cols or kw_cols):
        # choose pool based on preference and availability
        if prefer_normalized and norm_cols:
            pool = sorted(norm_cols, key=lambda c: int(c.split('_')[1]))
            make_col = lambda y: f"AvgLoadNorm_{y}"
        else:
            pool = sorted(kw_cols, key=lambda c: int(c.split('_')[1]))
            make_col = lambda y: f"AvgLoad_{y}_kW"

        if not pool:
            raise ValueError("No matching avg-load columns found.")

        years_available = sorted(int(c.split('_')[1]) for c in pool)

        if year is not None:
            col = make_col(year)
            if col not in df.columns:
                raise ValueError(f"Requested year {year} not in file. Available: {years_available}")
        else:
            prefs = prefer_years or [2021, 2020]
            chosen = next((y for y in prefs if make_col(y) in df.columns), None)
            if chosen is None:
                chosen = max(years_available)
            col = make_col(chosen)

        # parse time as HH:MM, fallback numeric
        try:
            time_hours = _to_hours_from_hhmm(df["Time"])
        except Exception:
            num = _coerce_numeric(df["Time"])
            if num.notna().all():
                if num.max() <= 95 and num.min() >= 0 and num.nunique() >= 10:
                    time_hours = (num.to_numpy() * 15.0) / 60.0
                else:
                    time_hours = num.to_numpy()
            else:
                raise ValueError("Cannot parse 'Time' as HH:MM or numeric.")

        vals = _coerce_numeric(df[col]).to_numpy()
        valid = ~np.isnan(time_hours) & ~np.isnan(vals)
        if not valid.any():
            raise ValueError("No valid numeric time/power pairs after parsing avg-load CSV.")

        order = np.argsort(time_hours[valid])
        time_hours = time_hours[valid][order]
        vals = vals[valid][order]

        # unique time bins
        _, idx = np.unique(time_hours, return_index=True)
        time_hours = time_hours[idx]
        vals = vals[idx]

        if not np.all(np.diff(time_hours) > 0):
            raise ValueError("Time column must be strictly increasing after parsing.")

        return time_hours, vals.tolist()

    # Fallback: generic 2-column
    if df.shape[1] < 2:
        raise ValueError(
            "CSV must contain either avg-load columns or at least 2 columns: time, power."
        )

    df2 = df.iloc[:, :2].copy()
    df2.columns = ["time", "power"]
    df2 = df2.dropna()
    df2["time"] = pd.to_numeric(df2["time"], errors="coerce")
    df2["power"] = pd.to_numeric(df2["power"], errors="coerce")
    df2 = df2.dropna()
    if df2.empty:
        raise ValueError("No valid numeric data found in CSV (generic format).")

    if time_unit.lower().startswith("min"):
        df2["time"] = df2["time"] / 60.0

    df2 = df2.sort_values("time").drop_duplicates("time")
    time_hours = df2["time"].to_numpy()
    power = df2["power"].to_numpy()
    if not np.all(np.diff(time_hours) > 0):
        raise ValueError("Time column must be strictly increasing.")
    return time_hours, power.tolist()
