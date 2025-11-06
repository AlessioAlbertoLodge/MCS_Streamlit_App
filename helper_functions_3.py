# helper_functions_3.py
from __future__ import annotations
import io
import re
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

def _to_hours_from_hhmm(time_col: pd.Series) -> np.ndarray:
    """Parse 'HH:MM' (or 'H:MM') to float hours."""
    # strip, ensure string
    s = time_col.astype(str).str.strip()
    # handle 'HH:MM'
    hh = s.str.extract(r'^(\d{1,2}):')[0].astype(float)
    mm = s.str.extract(r':(\d{2})$')[0].astype(float)
    hours = hh + mm/60.0
    return hours.to_numpy()

def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def load_demand_csv(
    file,
    time_unit: str = "hours",
    year: Optional[int] = None,
    prefer_years: Optional[List[int]] = None,
) -> Tuple[np.ndarray, List[float]]:
    """
    Load a CSV as time-vs-power[kW].

    Supports two formats:
    1) Generic 2-column: [time, power] where time is hours or minutes (use time_unit)
    2) Location avg-load: columns = 'Time', 'AvgLoad_<YYYY>_kW' (one or more years),
       where 'Time' is 'HH:MM'.

    Parameters
    ----------
    file : UploadedFile, BytesIO, or str
        Streamlit uploaded file, file-like buffer, or path.
    time_unit : {"hours", "minutes"}
        Unit of the first column for the generic 2-column format.
    year : int | None
        Desired year for the avg-load format. If None, choose using prefer_years or the latest available.
    prefer_years : list[int] | None
        Preference order when year is None, default [2021, 2020].

    Returns
    -------
    time_hours : np.ndarray
        Time values in hours (float).
    demand_kw : list[float]
        Corresponding power demand (kW).

    Raises
    ------
    ValueError
        If the CSV is invalid or the requested year is not present.
    """
    # Read file from buffer or path
    if isinstance(file, (io.BytesIO, io.BufferedReader)):
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(file)

    if df.empty:
        raise ValueError("CSV is empty.")

    # --- Detect avg-load (location) format -----------------------------------
    # Needs a 'Time' column and at least one 'AvgLoad_<YYYY>_kW' column
    avg_cols = [c for c in df.columns if re.fullmatch(r'AvgLoad_\d{4}_kW', c or "")]
    if "Time" in df.columns and avg_cols:
        # Choose year column
        years_available = sorted(int(c.split('_')[1]) for c in avg_cols)
        if year is not None:
            col = f"AvgLoad_{year}_kW"
            if col not in df.columns:
                raise ValueError(
                    f"Requested year {year} not found. Available: {years_available}"
                )
        else:
            prefs = prefer_years or [2021, 2020]
            chosen = next((y for y in prefs if f"AvgLoad_{y}_kW" in df.columns), None)
            if chosen is None:
                chosen = max(years_available)  # fall back to latest
            col = f"AvgLoad_{chosen}_kW"

        # Parse Time as HH:MM → hours
        try:
            time_hours = _to_hours_from_hhmm(df["Time"])
        except Exception:
            # if Time accidentally numeric (e.g., 0..95 or 0..24), try numeric
            num = _coerce_numeric(df["Time"])
            if num.notna().all():
                # if it looks like 0..95 bins, convert bins to hours (×15min)
                if num.max() <= 95 and num.min() >= 0 and num.nunique() >= 10:
                    time_hours = (num.to_numpy() * 15.0) / 60.0
                else:
                    time_hours = num.to_numpy()
            else:
                raise ValueError("Cannot parse 'Time' column to HH:MM or numeric hours.")

        demand_kw = _coerce_numeric(df[col]).to_numpy()
        valid = ~np.isnan(time_hours) & ~np.isnan(demand_kw)
        if not valid.any():
            raise ValueError("No valid numeric time/power pairs after parsing avg-load CSV.")

        # sort by time and ensure strictly increasing
        order = np.argsort(time_hours[valid])
        time_hours = time_hours[valid][order]
        demand_kw = demand_kw[valid][order]

        # de-duplicate time bins if needed
        _, idx = np.unique(time_hours, return_index=True)
        time_hours = time_hours[idx]
        demand_kw = demand_kw[idx]

        if not np.all(np.diff(time_hours) > 0):
            raise ValueError("Time column must be strictly increasing after parsing.")

        return time_hours, demand_kw.tolist()

    # --- Fallback: generic 2-column format -----------------------------------
    if df.shape[1] < 2:
        raise ValueError(
            "CSV must contain either:\n"
            " - 'Time' + 'AvgLoad_<YYYY>_kW' columns, or\n"
            " - at least two columns: time and power[kW]."
        )

    # Use first two columns (ignore headers if any)
    df2 = df.iloc[:, :2].copy()
    df2.columns = ["time", "power"]
    df2 = df2.dropna()

    df2["time"] = pd.to_numeric(df2["time"], errors="coerce")
    df2["power"] = pd.to_numeric(df2["power"], errors="coerce")
    df2 = df2.dropna()
    if df2.empty:
        raise ValueError("No valid numeric data found in CSV (generic format).")

    # Convert time to hours if needed
    if time_unit.lower().startswith("min"):
        df2["time"] = df2["time"] / 60.0

    # Sort and enforce strict increase
    df2 = df2.sort_values("time").drop_duplicates("time")
    time_hours = df2["time"].to_numpy()
    demand_kw = df2["power"].to_numpy()

    if not np.all(np.diff(time_hours) > 0):
        raise ValueError("Time column must be strictly increasing.")

    return time_hours, demand_kw.tolist()
