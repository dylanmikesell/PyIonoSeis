"""Line-of-sight TEC integration utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr

try:
    from scipy.integrate import trapezoid
    from scipy.interpolate import RegularGridInterpolator

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    RegularGridInterpolator = None
    trapezoid = None
    SCIPY_AVAILABLE = False

_log = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0
WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563


@dataclass
class TECConfig:
    """Configuration options for LOS TEC integration.

    Parameters
    ----------
    elevation_mask_deg : float
        Minimum elevation angle in degrees for valid LOS samples.
    output_dt_s : float or None
        Optional output cadence in seconds. When None, use model cadence.
    ipp_altitude_km : float
        Altitude in km at which to report IPP latitude/longitude.
    """

    elevation_mask_deg: float = 20.0
    output_dt_s: float | None = None
    ipp_altitude_km: float = 350.0


def _sorted_axis(coord: np.ndarray, data: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    coord = np.asarray(coord, dtype=float)
    if coord.size < 2:
        return coord, data
    diff = np.diff(coord)
    if np.all(diff > 0):
        return coord, data
    if np.all(diff < 0):
        return coord[::-1], np.flip(data, axis=axis)
    raise ValueError("Grid coordinates must be monotonic for interpolation.")


def _prepare_density_arrays(
    grid: xr.Dataset, dNe: xr.DataArray | None
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    lat = grid.coords["latitude"].values.astype(float)
    lon = grid.coords["longitude"].values.astype(float)
    alt = grid.coords["altitude"].values.astype(float)

    ne0 = grid["electron_density"].transpose("latitude", "longitude", "altitude").values

    if dNe is None:
        dNe_arr = None
    else:
        dNe_arr = dNe.transpose("latitude", "longitude", "altitude", "time").values

    lat, ne0 = _sorted_axis(lat, ne0, axis=0)
    lon, ne0 = _sorted_axis(lon, ne0, axis=1)
    alt, ne0 = _sorted_axis(alt, ne0, axis=2)

    if dNe_arr is not None:
        lat, dNe_arr = _sorted_axis(lat, dNe_arr, axis=0)
        lon, dNe_arr = _sorted_axis(lon, dNe_arr, axis=1)
        alt, dNe_arr = _sorted_axis(alt, dNe_arr, axis=2)

    return ne0, dNe_arr, lat, lon, alt


def geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, height_km: np.ndarray) -> np.ndarray:
    """Convert geodetic coordinates to ECEF (km).

    Parameters
    ----------
    lat_deg : np.ndarray
        Latitude in degrees.
    lon_deg : np.ndarray
        Longitude in degrees.
    height_km : np.ndarray
        Height above the ellipsoid in km.

    Returns
    -------
    np.ndarray
        ECEF positions in km with shape ``(n, 3)``.
    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    e2 = WGS84_F * (2.0 - WGS84_F)
    n = WGS84_A_KM / np.sqrt(1.0 - e2 * sin_lat**2)

    x = (n + height_km) * cos_lat * cos_lon
    y = (n + height_km) * cos_lat * sin_lon
    z = (n * (1.0 - e2) + height_km) * sin_lat

    return np.column_stack((x, y, z))


def ecef_to_spherical_latlon(ecef_km: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ECEF coordinates to spherical latitude/longitude/height.

    Parameters
    ----------
    ecef_km : np.ndarray
        ECEF positions in km with shape ``(n, 3)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Latitude (deg), longitude (deg), and height (km).
    """
    x = ecef_km[:, 0]
    y = ecef_km[:, 1]
    z = ecef_km[:, 2]

    r_xy = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, r_xy)
    lon = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2 + z**2)
    height = r - EARTH_RADIUS_KM

    return np.rad2deg(lat), np.rad2deg(lon), height


def _interpolate_series(time_in: np.ndarray, values: np.ndarray, time_out: np.ndarray) -> np.ndarray:
    return np.interp(time_out, time_in, values)


def _normalize_positions(
    positions: xr.Dataset | dict,
    time_s: np.ndarray,
    name: str,
) -> dict:
    if isinstance(positions, xr.Dataset):
        data = positions
        if "time" in data.coords:
            pos_time = np.asarray(data["time"].values, dtype=float)
        else:
            pos_time = time_s
        data_vars = {key: data[key].values for key in data.data_vars}
    elif isinstance(positions, dict):
        data_vars = positions
        pos_time = np.asarray(data_vars.get("time", time_s), dtype=float)
    else:
        raise TypeError(f"{name} positions must be a dict or xarray Dataset.")

    if {"x_km", "y_km", "z_km"}.issubset(data_vars):
        x = np.asarray(data_vars["x_km"], dtype=float)
        y = np.asarray(data_vars["y_km"], dtype=float)
        z = np.asarray(data_vars["z_km"], dtype=float)
        ecef = np.column_stack((x, y, z))
        lat, lon, height = ecef_to_spherical_latlon(ecef)
    elif {"latitude", "longitude", "height_km"}.issubset(data_vars):
        lat = np.asarray(data_vars["latitude"], dtype=float)
        lon = np.asarray(data_vars["longitude"], dtype=float)
        height = np.asarray(data_vars["height_km"], dtype=float)
        ecef = geodetic_to_ecef(lat, lon, height)
    else:
        raise KeyError(
            f"{name} positions require either (latitude, longitude, height_km) "
            "or (x_km, y_km, z_km)."
        )

    if lat.size == 1 and time_s.size > 1:
        lat = np.repeat(lat, time_s.size)
        lon = np.repeat(lon, time_s.size)
        height = np.repeat(height, time_s.size)
        ecef = np.repeat(ecef, time_s.size, axis=0)
        pos_time = time_s

    if pos_time.size != time_s.size:
        lat = _interpolate_series(pos_time, lat, time_s)
        lon = _interpolate_series(pos_time, lon, time_s)
        height = _interpolate_series(pos_time, height, time_s)
        ecef = np.column_stack(
            (
                _interpolate_series(pos_time, ecef[:, 0], time_s),
                _interpolate_series(pos_time, ecef[:, 1], time_s),
                _interpolate_series(pos_time, ecef[:, 2], time_s),
            )
        )

    return {
        "time_s": time_s,
        "latitude": lat,
        "longitude": lon,
        "height_km": height,
        "ecef_km": ecef,
    }


def _enu_from_ecef(
    receiver_lat_deg: np.ndarray,
    receiver_lon_deg: np.ndarray,
    receiver_ecef_km: np.ndarray,
    target_ecef_km: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = target_ecef_km[:, 0] - receiver_ecef_km[:, 0]
    dy = target_ecef_km[:, 1] - receiver_ecef_km[:, 1]
    dz = target_ecef_km[:, 2] - receiver_ecef_km[:, 2]

    lat = np.deg2rad(receiver_lat_deg)
    lon = np.deg2rad(receiver_lon_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return east, north, up


def _los_ipp_latlon(
    receiver_ecef_km: np.ndarray,
    los_unit: np.ndarray,
    altitudes_km: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r0 = receiver_ecef_km
    u = los_unit

    r0_dot_u = np.dot(r0, u)
    r0_norm2 = np.dot(r0, r0)

    radii_km = EARTH_RADIUS_KM + altitudes_km
    disc = r0_dot_u**2 - (r0_norm2 - radii_km**2)
    disc = np.where(disc < 0.0, np.nan, disc)

    s = -r0_dot_u + np.sqrt(disc)
    points = r0 + s[:, None] * u[None, :]

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    lat = np.arctan2(z, np.sqrt(x**2 + y**2))
    lon = np.arctan2(y, x)

    return np.rad2deg(lat), np.rad2deg(lon), s


def compute_los_tec(
    grid: xr.Dataset,
    receiver_positions: xr.Dataset | dict,
    satellite_positions: xr.Dataset | dict,
    dNe: xr.DataArray | None = None,
    tec_config: TECConfig | None = None,
    receiver_id: str | None = None,
    satellite_id: str | None = None,
) -> xr.Dataset:
    """Compute LOS TEC for a single satellite-receiver pair.

    Parameters
    ----------
    grid : xr.Dataset
        Model grid containing ``electron_density`` and coordinates
        ``latitude``, ``longitude``, and ``altitude``. If ``dNe`` is not
        provided, ``grid`` may include a ``dNe`` variable with a ``time``
        dimension.
    receiver_positions : xr.Dataset or dict
        Receiver positions with either ``(latitude, longitude, height_km)``
        or ``(x_km, y_km, z_km)`` variables. Provide ``time`` in seconds as
        a coordinate or key when positions vary with time.
    satellite_positions : xr.Dataset or dict
        Satellite positions with the same variable requirements as
        ``receiver_positions``.
    dNe : xr.DataArray, optional
        Electron density perturbation with dimensions
        ``(latitude, longitude, altitude, time)``.
    tec_config : TECConfig, optional
        TEC configuration options. Defaults are applied when omitted.
    receiver_id : str, optional
        Identifier used for logging.
    satellite_id : str, optional
        Identifier used for logging.

    Returns
    -------
    xr.Dataset
        Dataset with TEC time series and LOS metadata.

    Raises
    ------
    KeyError
        If required grid variables are missing.
    ImportError
        If SciPy is not available for interpolation.
    """
    if "electron_density" not in grid:
        raise KeyError("grid must contain 'electron_density'.")

    if not SCIPY_AVAILABLE or RegularGridInterpolator is None or trapezoid is None:
        raise ImportError("scipy is required for LOS TEC interpolation.")
    interp_3d = RegularGridInterpolator
    integrate = trapezoid

    if tec_config is None:
        tec_config = TECConfig()

    if dNe is None and "dNe" in grid:
        dNe = grid["dNe"]

    if dNe is not None and "time" in dNe.coords:
        base_time_s = np.asarray(dNe["time"].values, dtype=float)
    else:
        if "time" in grid.coords:
            base_time_s = np.asarray(grid["time"].values, dtype=float)
        else:
            raise KeyError("time coordinate required when dNe is not provided.")

    output_time_s = base_time_s
    if tec_config.output_dt_s is not None:
        dt_s = float(tec_config.output_dt_s)
        if dt_s <= 0.0:
            raise ValueError("output_dt_s must be > 0.")
        output_time_s = np.arange(base_time_s[0], base_time_s[-1] + 0.5 * dt_s, dt_s)

    receiver = _normalize_positions(receiver_positions, output_time_s, "receiver")
    satellite = _normalize_positions(satellite_positions, output_time_s, "satellite")

    ne0, dNe_arr, lat, lon, alt = _prepare_density_arrays(grid, dNe)

    ne0_interp = interp_3d(
        (lat, lon, alt), ne0, bounds_error=False, fill_value=np.nan
    )

    if dNe_arr is not None and dNe is not None and "time" in dNe.coords and not np.array_equal(
        output_time_s, base_time_s
    ):
        dNe = dNe.interp(time=output_time_s)
        dNe_arr = dNe.transpose("latitude", "longitude", "altitude", "time").values

    n_time = output_time_s.size

    east, north, up = _enu_from_ecef(
        receiver["latitude"],
        receiver["longitude"],
        receiver["ecef_km"],
        satellite["ecef_km"],
    )
    slant_range = np.sqrt(east**2 + north**2 + up**2)
    elevation = np.rad2deg(np.arcsin(np.where(slant_range > 0.0, up / slant_range, 0.0)))
    azimuth = np.rad2deg(np.arctan2(east, north))
    azimuth = np.mod(azimuth, 360.0)

    los_valid = elevation >= float(tec_config.elevation_mask_deg)

    ipp_lat = np.full(n_time, np.nan)
    ipp_lon = np.full(n_time, np.nan)

    tec_bg = np.full(n_time, np.nan)
    tec_pert = np.full(n_time, np.nan)
    tec_total = np.full(n_time, np.nan)

    missing_bg = 0
    missing_pert = 0
    missing_path = 0
    total_samples = 0

    for idx in range(n_time):
        if not los_valid[idx]:
            continue

        rec_ecef = receiver["ecef_km"][idx]
        sat_ecef = satellite["ecef_km"][idx]
        los_vec = sat_ecef - rec_ecef
        los_norm = np.linalg.norm(los_vec)
        if los_norm == 0.0:
            continue
        los_unit = los_vec / los_norm

        ipp_lat_i, ipp_lon_i, _ = _los_ipp_latlon(
            rec_ecef, los_unit, np.array([tec_config.ipp_altitude_km])
        )
        ipp_lat[idx] = ipp_lat_i[0]
        ipp_lon[idx] = ipp_lon_i[0]

        ipp_lat_alt, ipp_lon_alt, s_km = _los_ipp_latlon(rec_ecef, los_unit, alt)

        points = np.column_stack((ipp_lat_alt, ipp_lon_alt, alt))
        ne_bg = ne0_interp(points)

        missing_bg += np.count_nonzero(~np.isfinite(ne_bg))
        ne_bg = np.where(np.isfinite(ne_bg), ne_bg, 0.0)

        if dNe_arr is not None:
            dNe_interp = interp_3d(
                (lat, lon, alt),
                dNe_arr[:, :, :, idx],
                bounds_error=False,
                fill_value=np.nan,
            )
            ne_pert = dNe_interp(points)
            missing_pert += np.count_nonzero(~np.isfinite(ne_pert))
            ne_pert = np.where(np.isfinite(ne_pert), ne_pert, 0.0)
        else:
            ne_pert = np.zeros_like(ne_bg)

        s_m = s_km * 1e3
        valid = np.isfinite(s_m)
        missing_path += np.count_nonzero(~valid)
        if np.count_nonzero(valid) < 2:
            continue

        total_samples += np.count_nonzero(valid)

        s_m_valid = s_m[valid]
        tec_bg[idx] = integrate(ne_bg[valid], s_m_valid) * 1e-16
        tec_pert[idx] = integrate(ne_pert[valid], s_m_valid) * 1e-16
        tec_total[idx] = tec_bg[idx] + tec_pert[idx]

    if total_samples > 0:
        missing_total = missing_bg + missing_pert + missing_path
        denom = float(total_samples)
        sample_factor = 2.0 if dNe_arr is not None else 1.0
        _log.info(
            "LOS missing values for %s/%s: background=%d (%.2f%%), perturbation=%d "
            "(%.2f%%), path=%d (%.2f%%), total=%d (%.2f%%)",
            receiver_id or "receiver",
            satellite_id or "satellite",
            missing_bg,
            100.0 * missing_bg / denom,
            missing_pert,
            100.0 * missing_pert / denom,
            missing_path,
            100.0 * missing_path / denom,
            missing_total,
            100.0 * missing_total / (sample_factor * denom),
        )

    out = xr.Dataset(
        data_vars={
            "tec_background": ("time", tec_bg),
            "tec_perturbation": ("time", tec_pert),
            "tec_total": ("time", tec_total),
            "elevation_deg": ("time", elevation),
            "azimuth_deg": ("time", azimuth),
            "ipp_latitude_deg": ("time", ipp_lat),
            "ipp_longitude_deg": ("time", ipp_lon),
            "los_valid": ("time", los_valid.astype(bool)),
        },
        coords={"time": output_time_s},
    )

    out["tec_background"].attrs["units"] = "TECU"
    out["tec_perturbation"].attrs["units"] = "TECU"
    out["tec_total"].attrs["units"] = "TECU"
    out["elevation_deg"].attrs["units"] = "deg"
    out["azimuth_deg"].attrs["units"] = "deg"
    out["ipp_latitude_deg"].attrs["units"] = "deg"
    out["ipp_longitude_deg"].attrs["units"] = "deg"
    out["los_valid"].attrs["units"] = "boolean"

    out.attrs["tec_elevation_mask_deg"] = float(tec_config.elevation_mask_deg)
    out.attrs["tec_output_dt_s"] = (
        float(tec_config.output_dt_s)
        if tec_config.output_dt_s is not None
        else None
    )
    out.attrs["tec_ipp_altitude_km"] = float(tec_config.ipp_altitude_km)
    if receiver_id is not None:
        out.attrs["receiver_id"] = str(receiver_id)
    if satellite_id is not None:
        out.attrs["satellite_id"] = str(satellite_id)

    return out
