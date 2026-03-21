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

_VALID_CONSTELLATIONS = set(_CONS_MAP.values())


EARTH_RADIUS_KM = 6371.0


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


def _normalize_constellation(constellation: str) -> str:
    token = str(constellation).strip().upper()
    if token in _VALID_CONSTELLATIONS:
        return token
    if token in _CONS_MAP:
        return _CONS_MAP[token]
    raise ValueError(
        f"Unknown constellation '{constellation}'. Expected one of "
        f"{sorted(_VALID_CONSTELLATIONS)} or prefixes {sorted(_CONS_MAP.keys())}."
    )


def _seconds_of_day(time_utc: datetime) -> float:
    return (
        float(time_utc.hour) * 3600.0
        + float(time_utc.minute) * 60.0
        + float(time_utc.second)
        + float(time_utc.microsecond) * 1.0e-6
    )


def _ecef_to_spherical_latlon(
    x_km: np.ndarray,
    y_km: np.ndarray,
    z_km: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_xy = np.sqrt(x_km**2 + y_km**2)
    lat = np.rad2deg(np.arctan2(z_km, r_xy))
    lon = np.rad2deg(np.arctan2(y_km, x_km))
    r = np.sqrt(x_km**2 + y_km**2 + z_km**2)
    height_km = r - EARTH_RADIUS_KM
    return lat, lon, height_km


def load_sat_number_mapping(mapping_path: str | Path) -> dict[int, str]:
    """Load satellite number-to-ID mapping from a CSV file.

    Parameters
    ----------
    mapping_path : str or Path
        CSV file path with columns ``sat_number`` and ``sat_id``.

    Returns
    -------
    dict[int, str]
        Mapping of SATPOS satellite number to satellite ID (e.g., ``17 -> "G17"``).
    """
    _require_pandas()
    pd_local = cast(Any, pd)
    df = pd_local.read_csv(mapping_path)
    expected = {"sat_number", "sat_id"}
    if not expected.issubset(df.columns):
        raise ValueError(
            f"Mapping file {mapping_path} must contain columns {sorted(expected)}."
        )

    out: dict[int, str] = {}
    for _, row in df.iterrows():
        if pd_local.isna(row["sat_number"]) or pd_local.isna(row["sat_id"]):
            raise ValueError("Mapping rows must define both sat_number and sat_id.")
        sat_number = int(row["sat_number"])
        sat_id = str(row["sat_id"]).strip().upper()
        _parse_sat_id(sat_id)
        if sat_number in out:
            raise ValueError(
                f"Duplicate sat_number={sat_number} in mapping file {mapping_path}."
            )
        out[sat_number] = sat_id
    return out


def _resolve_satellite_identity(
    sat_id: str | None,
    constellation: str | None,
    prn: int | None,
    sat_number: int | None,
    sat_number_mapping: dict[int, str] | None,
) -> tuple[str, int]:
    if sat_id is not None:
        return _parse_sat_id(sat_id)

    if constellation is not None and prn is not None:
        return _normalize_constellation(constellation), int(prn)

    if sat_number is not None and sat_number_mapping is not None:
        mapped = sat_number_mapping.get(int(sat_number))
        if mapped is None:
            raise ValueError(
                f"No sat_id mapping found for sat_number={sat_number}."
            )
        return _parse_sat_id(mapped)

    raise ValueError(
        "Provide sat_id, or constellation+prn, or sat_number with mapping."
    )


def build_satpos_file_path(
    satpos_root: str | Path,
    satpos_date: str,
    sat_number: int,
) -> Path:
    """Build legacy SATPOS file path for a single satellite.

    Parameters
    ----------
    satpos_root : str or Path
        Root directory containing legacy SATPOS day folders.
    satpos_date : str
        Legacy date tag used in folder and file names (example: ``"16591"``).
    sat_number : int
        Satellite number used in file naming.

    Returns
    -------
    Path
        Full path of the expected SATPOS file.
    """
    return Path(satpos_root) / str(satpos_date) / f"sat{int(sat_number):02d}_{satpos_date}.pos"


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


def load_receiver_positions_listesta(
    listesta_path: str | Path,
    receiver_code: str | None = None,
) -> dict:
    """Load receiver positions from legacy listesta station files.

    Parameters
    ----------
    listesta_path : str or Path
        Path to whitespace-delimited station file with columns
        ``code x_m y_m z_m lat_deg lon_deg``.
    receiver_code : str, optional
        Receiver code to select. Defaults to the first row when omitted.

    Returns
    -------
    dict
        Mapping with ``latitude``, ``longitude``, ``height_km``, and ``code``.
        Latitude/longitude/height are recomputed from ECEF as authoritative.
    """
    _require_pandas()
    pd_local = cast(Any, pd)
    df = pd_local.read_csv(listesta_path, sep=r"\s+", header=None)

    if df.empty:
        raise ValueError("listesta station file is empty.")
    if df.shape[1] < 6:
        raise ValueError(
            "listesta station file must contain 6 columns: "
            "code x_m y_m z_m lat_deg lon_deg."
        )

    if receiver_code is None:
        receiver_code = str(df.iloc[0, 0])
        _log.info("Using first legacy receiver entry: %s", receiver_code)

    code_series = df.iloc[:, 0].astype(str).str.upper()
    mask = code_series == receiver_code.strip().upper()
    if not mask.any():
        raise ValueError(
            f"Receiver code '{receiver_code}' not found in legacy station file."
        )

    row = df.loc[mask].iloc[0]
    if str(row.iloc[0]).strip() == "":
        raise ValueError("Receiver code is empty in listesta station file.")

    x_km = float(row.iloc[1]) / 1000.0
    y_km = float(row.iloc[2]) / 1000.0
    z_km = float(row.iloc[3]) / 1000.0

    if not np.all(np.isfinite([x_km, y_km, z_km])):
        raise ValueError("listesta ECEF coordinates must be finite numeric values.")

    lat_deg, lon_deg, height_km = _ecef_to_spherical_latlon(
        np.array([x_km]),
        np.array([y_km]),
        np.array([z_km]),
    )

    if np.any(np.abs(lat_deg) > 90.0) or np.any(np.abs(lon_deg) > 180.0):
        raise ValueError("Recomputed listesta geodetic coordinates are out of range.")

    return {
        "latitude": lat_deg,
        "longitude": lon_deg,
        "height_km": height_km,
        "code": str(row.iloc[0]),
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

    constellation = _normalize_constellation(constellation)

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


def load_orbits_pos(
    pos_path: str | Path,
    event_time: datetime,
    sat_id: str | None = None,
    constellation: str | None = None,
    prn: int | None = None,
    sat_number: int | None = None,
    sat_number_mapping: dict[int, str] | None = None,
    sat_mapping_file: str | Path | None = None,
    start_offset_s: float = 0.0,
    output_dt_s: float | None = None,
) -> dict:
    """Load satellite orbits from legacy SATPOS ``.pos`` files.

    Parameters
    ----------
    pos_path : str or Path
        Path to a legacy SATPOS file.
    event_time : datetime
        Event time (UTC) used to convert SATPOS seconds-of-day to
        relative model time (seconds).
    sat_id : str, optional
        Satellite ID like ``"G21"``. Overrides ``constellation`` and ``prn``.
    constellation : str, optional
        Constellation code used with ``prn``.
    prn : int, optional
        Satellite PRN used with ``constellation``.
    sat_number : int, optional
        Legacy SATPOS satellite number used with mapping.
    sat_number_mapping : dict[int, str], optional
        In-memory mapping from SATPOS number to sat_id.
    sat_mapping_file : str or Path, optional
        CSV mapping file with columns ``sat_number`` and ``sat_id``.
    start_offset_s : float, optional
        Start offset in seconds relative to event origin for output.
    output_dt_s : float, optional
        Optional output cadence in seconds.

    Returns
    -------
    dict
        Mapping with ``time``, ``x_km``, ``y_km``, ``z_km``, ``constellation``,
        and ``prn``.
    """
    _require_pandas()
    pd_local = cast(Any, pd)
    df = pd_local.read_csv(pos_path, sep=r"\s+", header=None)
    if df.empty:
        raise ValueError("Legacy SATPOS file contains no data.")
    if df.shape[1] < 7:
        raise ValueError(
            "Legacy SATPOS file must contain 7 columns: "
            "seconds_of_day x_m y_m z_m lon_deg lat_deg ele_m."
        )

    mapping = sat_number_mapping
    if mapping is None and sat_mapping_file is not None:
        mapping = load_sat_number_mapping(sat_mapping_file)

    resolved_cons, resolved_prn = _resolve_satellite_identity(
        sat_id=sat_id,
        constellation=constellation,
        prn=prn,
        sat_number=sat_number,
        sat_number_mapping=mapping,
    )

    seconds_of_day = df.iloc[:, 0].astype(float).values
    x_km = df.iloc[:, 1].astype(float).values / 1000.0
    y_km = df.iloc[:, 2].astype(float).values / 1000.0
    z_km = df.iloc[:, 3].astype(float).values / 1000.0

    if not np.all(np.isfinite(seconds_of_day)):
        raise ValueError("SATPOS seconds_of_day values must be finite numbers.")
    if not np.all(np.isfinite(x_km)) or not np.all(np.isfinite(y_km)) or not np.all(np.isfinite(z_km)):
        raise ValueError("SATPOS x/y/z values must be finite numbers.")
    if np.any(np.diff(seconds_of_day) <= 0.0):
        raise ValueError("SATPOS seconds_of_day must be strictly increasing.")

    event_sod = _seconds_of_day(event_time)
    time_s = seconds_of_day - event_sod

    # Legacy SATPOS are frequently 1-second samples; keep native points unless
    # an output cadence is requested.
    if output_dt_s is None:
        if float(start_offset_s) > time_s.min():
            keep = time_s >= float(start_offset_s)
            if not np.any(keep):
                raise ValueError("No SATPOS records remain after applying start_offset_s.")
            time_s = time_s[keep]
            x_km = x_km[keep]
            y_km = y_km[keep]
            z_km = z_km[keep]
    else:
        dt_s = float(output_dt_s)
        if dt_s <= 0.0:
            raise ValueError("output_dt_s must be > 0.")

        t_start = max(float(start_offset_s), float(time_s.min()))
        t_end = float(time_s.max())
        if t_end < t_start:
            raise ValueError("SATPOS time span does not include requested start_offset_s.")

        new_time = np.arange(t_start, t_end + 0.5 * dt_s, dt_s)
        x_km = np.interp(new_time, time_s, x_km)
        y_km = np.interp(new_time, time_s, y_km)
        z_km = np.interp(new_time, time_s, z_km)
        time_s = new_time

    _orbit_unit_check(x_km, y_km, z_km)

    return {
        "time": time_s,
        "x_km": x_km,
        "y_km": y_km,
        "z_km": z_km,
        "constellation": str(resolved_cons),
        "prn": int(resolved_prn),
    }
