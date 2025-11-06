# helper_functions_3.py
from __future__ import annotations
import io
import numpy as np
import pandas as pd
from typing import Tuple, List

def load_demand_csv(file, time_unit: str = "hours") -> Tuple[np.ndarray, List[float]]:
    """
    Load a CSV containing time vs power[kW].

    Parameters
    ----------
    file : UploadedFile or str
        Streamlit uploaded file or path-like object.
    time_unit : {"hours", "minutes"}
        Unit of the first column; affects conversion to hours.

    Returns
    -------
    time_hours : np.ndarray
        Time values in hours (float).
    demand_kw : list[float]
        Corresponding power demand (kW).

    Raises
    ------
    ValueError
        If file has fewer than 2 columns or invalid data.
    """
    # Read file from buffer or path
    if isinstance(file, io.BytesIO):
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(file)

    if df.shape[1] < 2:
        raise ValueError("CSV must contain at least two columns: time and power[kW].")

    # Use first two columns (ignore headers if any)
    df = df.iloc[:, :2]
    df.columns = ["time", "power"]

    # Clean numeric
    df = df.dropna()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["power"] = pd.to_numeric(df["power"], errors="coerce")
    df = df.dropna()

    if df.empty:
        raise ValueError("No valid numeric data found in CSV.")

    # Convert time to hours
    if time_unit.lower().startswith("min"):
        df["time"] = df["time"] / 60.0

    # Sort by time, remove duplicates
    df = df.sort_values("time").drop_duplicates("time")

    time_hours = df["time"].to_numpy()
    demand_kw = df["power"].to_numpy().tolist()

    # Check monotonicity
    if not np.all(np.diff(time_hours) > 0):
        raise ValueError("Time column must be strictly increasing.")

    return time_hours, demand_kw
