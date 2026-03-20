"""Continuity-equation time integration for electron density perturbations."""

from __future__ import annotations

import numpy as np
import xarray as xr

EARTH_RADIUS_KM = 6371.0


def _time_axis(t0_s: float, tmax_s: float, dt_s: float) -> np.ndarray:
    """Build the time axis for the continuity solver.

    Parameters
    ----------
    t0_s : float
        Start time in seconds.
    tmax_s : float
        End time in seconds.
    dt_s : float
        Time step in seconds.

    Returns
    -------
    np.ndarray
        Time samples in seconds, inclusive of ``tmax_s``.
    """
    if dt_s <= 0.0:
        raise ValueError("dt_s must be > 0.")
    if tmax_s < t0_s:
        raise ValueError("tmax_s must be >= t0_s.")
    steps = int(np.floor((tmax_s - t0_s) / dt_s)) + 1
    return t0_s + dt_s * np.arange(steps)


def spherical_divergence(
    ar: np.ndarray,
    at: np.ndarray,
    ap: np.ndarray,
    r_m: np.ndarray,
    lat_deg: np.ndarray,
    dr_m: float,
    dtheta_deg: float,
    dphi_deg: float,
    divergence_flag: int = 3,
) -> np.ndarray:
    """Compute spherical divergence with clamped boundary indices.

    Parameters
    ----------
    ar : np.ndarray
        Radial component of a vector field with shape
        ``(latitude, longitude, altitude)``.
    at : np.ndarray
        Theta (southward) component of a vector field with shape
        ``(latitude, longitude, altitude)``.
    ap : np.ndarray
        Phi (eastward) component of a vector field with shape
        ``(latitude, longitude, altitude)``.
    r_m : np.ndarray
        Radial coordinate in meters, shape ``(altitude,)``.
    lat_deg : np.ndarray
        Latitude coordinates in degrees, shape ``(latitude,)``.
    dr_m : float
        Altitude spacing in meters.
    dtheta_deg : float
        Latitude spacing in degrees.
    dphi_deg : float
        Longitude spacing in degrees.
    divergence_flag : int, optional
        ``1`` for radial-only divergence, ``3`` for full 3-D divergence.

    Returns
    -------
    np.ndarray
        Divergence of the input vector field, shape
        ``(latitude, longitude, altitude)``.
    """
    if divergence_flag not in (1, 3):
        raise ValueError("divergence_flag must be 1 or 3.")
    if dr_m <= 0.0:
        raise ValueError("dr_m must be > 0.")

    n_lat, n_lon, n_alt = ar.shape
    r = r_m.reshape(1, 1, n_alt)

    alt_idx = np.arange(n_alt)
    im1 = np.clip(alt_idx - 1, 0, n_alt - 1)
    ip1 = np.clip(alt_idx + 1, 0, n_alt - 1)

    ar_ip1 = np.take(ar, ip1, axis=2)
    ar_im1 = np.take(ar, im1, axis=2)

    radial_term = 2.0 * ar / r + (ar_ip1 - ar_im1) / (2.0 * dr_m)

    if divergence_flag == 1:
        return radial_term

    lat_idx = np.arange(n_lat)
    jm1 = np.clip(lat_idx - 1, 0, n_lat - 1)
    jp1 = np.clip(lat_idx + 1, 0, n_lat - 1)

    lon_idx = np.arange(n_lon)
    km1 = np.clip(lon_idx - 1, 0, n_lon - 1)
    kp1 = np.clip(lon_idx + 1, 0, n_lon - 1)

    at_jp1 = np.take(at, jp1, axis=0)
    at_jm1 = np.take(at, jm1, axis=0)
    ap_kp1 = np.take(ap, kp1, axis=1)
    ap_km1 = np.take(ap, km1, axis=1)

    lat_rad = np.deg2rad(lat_deg)
    zh = np.pi / 2.0 - lat_rad
    sin_zh = np.sin(zh)
    cos_zh = np.cos(zh)

    dtheta_rad = np.deg2rad(dtheta_deg)
    dphi_rad = np.deg2rad(dphi_deg)
    if dtheta_rad == 0.0 or dphi_rad == 0.0:
        raise ValueError("Grid spacing in degrees must be > 0.")

    sin_zh_safe = np.where(np.abs(sin_zh) < 1.0e-8, np.nan, sin_zh)
    sin_zh_safe = sin_zh_safe.reshape(n_lat, 1, 1)
    cos_zh = cos_zh.reshape(n_lat, 1, 1)

    theta_term = (at_jp1 - at_jm1) / (2.0 * (-dtheta_rad)) / r
    theta_curv = cos_zh / (sin_zh_safe * r) * at
    phi_term = (ap_kp1 - ap_km1) / (2.0 * dphi_rad) / (r * sin_zh_safe)

    return radial_term + theta_term + theta_curv + phi_term


def solve_continuity(
    grid: xr.Dataset,
    t0_s: float,
    tmax_s: float,
    dt_s: float,
    b: float = 1.0,
    divergence_flag: int = 3,
    geomag_flag: bool = True,
    amplitude_var: str = "infraga_amplitude",
    travel_time_var: str = "travel_time_s",
    store_neutral_velocity: bool = True,
) -> xr.Dataset:
    """Integrate the electron density continuity equation through time.

    Parameters
    ----------
    grid : xr.Dataset
        Model grid containing ``electron_density``, wavevector components
        (``kr``, ``kt``, ``kp``), and mapped infraGA scalars.
    t0_s : float
        Start time in seconds.
    tmax_s : float
        End time in seconds.
    dt_s : float
        Time step in seconds.
    b : float, optional
        Pulse broadening coefficient in seconds per second.
    divergence_flag : int, optional
        ``1`` for radial-only divergence, ``3`` for full 3-D divergence.
    geomag_flag : bool, optional
        If True, project the neutral velocity along the magnetic field.
    amplitude_var : str, optional
        Grid variable containing linear amplitude to scale the pulse.
    travel_time_var : str, optional
        Grid variable containing infraGA travel times in seconds.
    store_neutral_velocity : bool, optional
        If True, store neutral velocity components in the output dataset.

    Returns
    -------
    xr.Dataset
        Dataset with ``dNe`` and optional neutral velocity components.
    """
    required = ["electron_density", "kr", "kt", "kp", amplitude_var, travel_time_var]
    missing = [name for name in required if name not in grid]
    if missing:
        raise KeyError(f"Grid is missing required variables: {missing}")

    lat = grid.coords["latitude"].values.astype(float)
    lon = grid.coords["longitude"].values.astype(float)
    alt = grid.coords["altitude"].values.astype(float)

    if lat.size < 2 or lon.size < 2 or alt.size < 2:
        raise ValueError("Grid must have at least 2 points in each dimension.")

    dtheta_deg = float(np.mean(np.diff(lat)))
    dphi_deg = float(np.mean(np.diff(lon)))
    dr_m = float(np.mean(np.diff(alt))) * 1e3
    r_m = (EARTH_RADIUS_KM + alt) * 1e3

    time_s = _time_axis(t0_s=float(t0_s), tmax_s=float(tmax_s), dt_s=float(dt_s))
    n_time = time_s.size

    ne0 = grid["electron_density"].values.astype(float)
    kr = grid["kr"].values.astype(float)
    kt = grid["kt"].values.astype(float)
    kp = grid["kp"].values.astype(float)
    amp = grid[amplitude_var].values.astype(float)
    ti = grid[travel_time_var].values.astype(float)

    if geomag_flag:
        if "inclination" not in grid or "declination" not in grid:
            raise KeyError("Magnetic field inclination/declination missing from grid.")
        inc = np.deg2rad(grid["inclination"].values.astype(float))
        dec = np.deg2rad(grid["declination"].values.astype(float))
        b_r = -np.sin(inc)
        b_t = -np.cos(inc) * np.cos(dec)
        b_p = np.cos(inc) * np.sin(dec)
    else:
        b_r = b_t = b_p = None

    dNe = np.zeros_like(ne0, dtype=float)
    dNe_out = np.zeros((*dNe.shape, n_time), dtype=float)

    if store_neutral_velocity:
        vn_r_out = np.zeros((*dNe.shape, n_time), dtype=float)
        vn_t_out = np.zeros((*dNe.shape, n_time), dtype=float)
        vn_p_out = np.zeros((*dNe.shape, n_time), dtype=float)

    sigma = b * ti
    sqrt2 = np.sqrt(2.0)
    pi_quarter = np.pi ** 0.25

    for idx, t in enumerate(time_s):
        tp = t - sigma
        delta = ti - tp

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            stf = (
                sqrt2
                / (np.power(sigma, 1.5) * pi_quarter)
                * delta
                * np.exp(-((delta / (sqrt2 * sigma)) ** 2))
            )
        stf = np.where(np.isfinite(stf), stf, 0.0)

        vn_r = amp * stf * kr * 1e3
        vn_t = amp * stf * kt * 1e3
        vn_p = amp * stf * kp * 1e3

        if geomag_flag:
            ue = vn_r * b_r + vn_t * b_t + vn_p * b_p
            ue_r = ue * b_r
            ue_t = ue * b_t
            ue_p = ue * b_p
        else:
            ue_r = vn_r
            ue_t = vn_t
            ue_p = vn_p

        flux_r = ne0 * ue_r
        flux_t = ne0 * ue_t
        flux_p = ne0 * ue_p

        div_flux = spherical_divergence(
            flux_r,
            flux_t,
            flux_p,
            r_m=r_m,
            lat_deg=lat,
            dr_m=dr_m,
            dtheta_deg=dtheta_deg,
            dphi_deg=dphi_deg,
            divergence_flag=divergence_flag,
        )

        dNe = dNe - dt_s * div_flux
        dNe[:, :, -1] = dNe[:, :, -2]

        dNe_out[:, :, :, idx] = dNe

        if store_neutral_velocity:
            vn_r_out[:, :, :, idx] = vn_r
            vn_t_out[:, :, :, idx] = vn_t
            vn_p_out[:, :, :, idx] = vn_p

    coords = {
        "latitude": lat,
        "longitude": lon,
        "altitude": alt,
        "time": time_s,
    }

    data_vars = {
        "dNe": (("latitude", "longitude", "altitude", "time"), dNe_out),
    }

    if store_neutral_velocity:
        data_vars.update(
            {
                "neutral_velocity_r": (
                    ("latitude", "longitude", "altitude", "time"),
                    vn_r_out,
                ),
                "neutral_velocity_t": (
                    ("latitude", "longitude", "altitude", "time"),
                    vn_t_out,
                ),
                "neutral_velocity_p": (
                    ("latitude", "longitude", "altitude", "time"),
                    vn_p_out,
                ),
            }
        )

    out = xr.Dataset(data_vars=data_vars, coords=coords)
    out["dNe"].attrs["units"] = "m^-3"
    out["dNe"].attrs["long_name"] = "electron density perturbation"

    if store_neutral_velocity:
        out["neutral_velocity_r"].attrs["units"] = "m/s"
        out["neutral_velocity_t"].attrs["units"] = "m/s"
        out["neutral_velocity_p"].attrs["units"] = "m/s"

    out.attrs["continuity_t0_s"] = float(t0_s)
    out.attrs["continuity_tmax_s"] = float(tmax_s)
    out.attrs["continuity_dt_s"] = float(dt_s)
    out.attrs["continuity_b"] = float(b)
    out.attrs["continuity_divergence_flag"] = int(divergence_flag)
    out.attrs["continuity_geomag_flag"] = int(bool(geomag_flag))
    out.attrs["continuity_amplitude_var"] = str(amplitude_var)
    out.attrs["continuity_travel_time_var"] = str(travel_time_var)

    return out
