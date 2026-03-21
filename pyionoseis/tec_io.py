"""Input loaders for TEC workflows."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    pd = None
    PANDAS_AVAILABLE = False

_log = logging.getLogger(__name__)

_CONS_MAP = {
    "G": "GPS",
    "R": "GLO",
    "E": "GAL",
    "C": "BDS",
    "J": "QZS",
    "I": "IRN",
    "T": "NNS",
}


def _require_pandas() -> None:
    if not PANDAS_AVAILABLE or pd is None:
        raise ImportError("pandas is required for TEC input loaders.")


def _parse_sat_id(sat_id: str) -> tuple[str, int]:
    sat_id = sat_id.strip().upper()
    if len(sat_id) < 2:
        raise ValueError("sat_id must include constellation prefix and PRN.")
    cons = _CONS_MAP.get(sat_id[0])
    if cons is None:
        raise ValueError(f"Unknown constellation prefix '{sat_id[0]}'.")
    prn = int(sat_id[1:])
    return cons, prn


def load_receiver_positions_csv(
    csv_path: str | Path,
    receiver_code: str | None = None,
) -> dict:
    """Load receiver positions from a CSV file.

    Parameters
    ----------
    csv_path : str or Path
        Path to the receiver CSV containing columns ``code``, ``lat``, ``lon``, ``hgt``.
    receiver_code : str, optional
        Receiver code to select. Defaults to the first row when omitted.

    Returns
    -------
    dict
        Mapping with ``latitude``, ``longitude``, ``height_km``, and ``code``.
    """
    _require_pandas()
    pd_local = cast(Any, pd)
    df = pd_local.read_csv(csv_path)
    if df.empty:
        raise ValueError("Receiver CSV is empty.")

    if receiver_code is None:
        receiver_code = str(df.iloc[0]["code"])
        _log.info("Using first receiver entry: %s", receiver_code)

    mask = df["code"].astype(str).str.upper() == receiver_code.strip().upper()
    if not mask.any():
        raise ValueError(f"Receiver code '{receiver_code}' not found in {csv_path}.")

    row = df[mask].iloc[0]
    lat = float(row["lat"])
    lon = float(row["lon"])
    hgt_km = float(row["hgt"]) / 1000.0

    return {
        "latitude": np.array([lat]),
        "longitude": np.array([lon]),
        "height_km": np.array([hgt_km]),
        "code": str(row["code"]),
    }


def _orbit_time_seconds(df: Any, event_time: datetime) -> np.ndarray:
    pd_local = cast(Any, pd)
    date = pd_local.to_datetime(
        df["year"].astype(int).astype(str) + df["doy"].astype(int).astype(str).str.zfill(3),
        format="%Y%j",
        utc=True,
    )
    time = date + pd_local.to_timedelta(df["sod"].astype(float), unit="s")
    event_ts = pd_local.Timestamp(event_time, tz="UTC")
    return (time - event_ts).dt.total_seconds().values.astype(float)


def _orbit_unit_check(x_km: np.ndarray, y_km: np.ndarray, z_km: np.ndarray) -> None:
    radius = np.sqrt(x_km**2 + y_km**2 + z_km**2)
    median_radius = float(np.nanmedian(radius))
    if median_radius < 1.0e3 or median_radius > 1.0e5:
        _log.warning(
            "Orbit radius %.2f km looks unusual; verify that x/y/z are in km.",
            median_radius,
        )


def load_orbits_hdf5(
    h5_path: str | Path,
    event_time: datetime,
    sat_id: str | None = None,
    constellation: str | None = None,
    prn: int | None = None,
    output_dt_s: float | None = None,
) -> dict:
    """Load satellite orbit positions from an HDF5 file.

    Parameters
    ----------
    h5_path : str or Path
        Path to the HDF5 orbit file.
    event_time : datetime
        Event time (UTC) used to compute relative seconds.
    sat_id : str, optional
        Satellite ID like ``"G21"``. Overrides ``constellation`` and ``prn``.
    constellation : str, optional
        Constellation code (e.g., ``"GPS"``).
    prn : int, optional
        Satellite PRN number.
    output_dt_s : float, optional
        Optional output cadence in seconds for resampling.

    Returns
    -------
    dict
        Mapping with ``time``, ``x_km``, ``y_km``, ``z_km`` arrays.
    """
    _require_pandas()

    pd_local = cast(Any, pd)
    df = pd_local.read_hdf(h5_path, key="orbs")
    if df.empty:
        raise ValueError("Orbit HDF5 file contains no data.")

    if sat_id is not None:
        constellation, prn = _parse_sat_id(sat_id)

    if constellation is None or prn is None:
        raise ValueError("Provide sat_id or both constellation and prn.")

    mask = (df["cons"].astype(str).str.upper() == str(constellation).upper()) & (
        df["prn"].astype(int) == int(prn)
    )
    if not mask.any():
        raise ValueError("No orbit entries found for requested satellite.")

    sat_df = df.loc[mask].copy()
    sat_df = sat_df.sort_values(by=["year", "doy", "sod"])

    time_s = _orbit_time_seconds(sat_df, event_time)
    x_km = sat_df["x"].astype(float).values
    y_km = sat_df["y"].astype(float).values
    z_km = sat_df["z"].astype(float).values

    _orbit_unit_check(x_km, y_km, z_km)

    if output_dt_s is not None:
        dt_s = float(output_dt_s)
        if dt_s <= 0.0:
            raise ValueError("output_dt_s must be > 0.")
        new_time = np.arange(time_s.min(), time_s.max() + 0.5 * dt_s, dt_s)
        x_km = np.interp(new_time, time_s, x_km)
        y_km = np.interp(new_time, time_s, y_km)
        z_km = np.interp(new_time, time_s, z_km)
        time_s = new_time

    return {
        "time": time_s,
        "x_km": x_km,
        "y_km": y_km,
        "z_km": z_km,
        "constellation": str(constellation),
        "prn": int(prn),
    }
