"""Internal helpers for ray tracing orchestration.

These functions hold geometry and warning utilities used by
``Model3D.trace_rays``. They are intentionally internal and preserve existing
behavior while enabling staged extraction from ``model.py``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from pyionoseis import model_io


def build_ray_signature_payload(
    source_lat_deg: float,
    source_lon_deg: float,
    source_time: str,
    run_type: str,
    ray_params: dict,
    profile: dict,
) -> dict:
    """Build canonical signature payload for ray cache identity."""
    return {
        "signature_version": 1,
        "run_type": str(run_type),
        "source": {
            "latitude_deg": float(source_lat_deg),
            "longitude_deg": float(source_lon_deg),
            "time": str(source_time),
        },
        "ray_params": ray_params,
        "profile": profile,
    }


def signature_payload_matches(signature_file: Path, signature_payload: dict) -> bool:
    """Return True when stored signature payload matches expected payload."""
    with signature_file.open("r", encoding="utf-8") as fh:
        saved_signature_raw = json.load(fh)
    saved_signature_payload = model_io.normalize_signature_payload(saved_signature_raw)
    return saved_signature_payload == signature_payload


def apply_raypaths_metadata(
    raypaths: xr.Dataset,
    run_type: str,
    use_accel: bool,
    accel_used: bool,
    command: list[str],
    source_lat_deg: float,
    source_lon_deg: float,
    source_time: str,
    signature_hash: str | None = None,
) -> None:
    """Apply consistent metadata attrs to parsed raypaths."""
    raypaths.attrs["raytrace_backend"] = "infraga_sph"
    raypaths.attrs["raytrace_type"] = run_type
    raypaths.attrs["raytrace_accel_requested"] = bool(use_accel)
    raypaths.attrs["raytrace_accel_used"] = bool(accel_used)
    raypaths.attrs["source_latitude_deg"] = float(source_lat_deg)
    raypaths.attrs["source_longitude_deg"] = float(source_lon_deg)
    raypaths.attrs["source_time"] = str(source_time)
    raypaths.attrs["prof_format"] = "zcuvd"
    raypaths.attrs["winds_assumed_zero"] = True
    raypaths.attrs["infraga_command"] = " ".join(command)
    if signature_hash is not None:
        raypaths.attrs["raytrace_signature_hash"] = signature_hash


def azimuth_sequence(az_min: float, az_max: float, az_step: float) -> np.ndarray:
    """Build normalized azimuth sequence in degrees."""
    if float(az_step) <= 0.0:
        raise ValueError("az_interp_step must be > 0.")
    if float(az_max) < float(az_min):
        raise ValueError("az_interp_max must be >= az_interp_min.")

    az_values = np.arange(
        float(az_min),
        float(az_max) + 0.5 * float(az_step),
        float(az_step),
    )
    az_values = np.mod(az_values, 360.0)
    # Avoid duplicate endpoint (e.g., 0 and 360).
    _, unique_idx = np.unique(np.round(az_values, 12), return_index=True)
    unique_idx = np.sort(unique_idx)
    return az_values[unique_idx]


def forward_geodesic_deg(
    src_lat_deg: float,
    src_lon_deg: float,
    angular_distance_rad: np.ndarray,
    azimuth_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project points from source by angular distance and azimuth on a sphere."""
    lat1 = np.deg2rad(float(src_lat_deg))
    lon1 = np.deg2rad(float(src_lon_deg))
    az = np.deg2rad(float(azimuth_deg))
    delta = np.asarray(angular_distance_rad, dtype=np.float64)

    sin_lat1 = np.sin(lat1)
    cos_lat1 = np.cos(lat1)
    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)

    lat2 = np.arcsin(
        np.clip(sin_lat1 * cos_delta + cos_lat1 * sin_delta * np.cos(az), -1.0, 1.0)
    )
    lon2 = lon1 + np.arctan2(
        np.sin(az) * sin_delta * cos_lat1,
        cos_delta - sin_lat1 * np.sin(lat2),
    )

    lat2_deg = np.rad2deg(lat2)
    lon2_deg = ((np.rad2deg(lon2) + 180.0) % 360.0) - 180.0
    return lat2_deg, lon2_deg


def great_circle_angular_distance_rad(
    src_lat_deg: float,
    src_lon_deg: float,
    dst_lat_deg: np.ndarray,
    dst_lon_deg: np.ndarray,
) -> np.ndarray:
    """Great-circle angular distance between source and destination points."""
    lat1 = np.deg2rad(float(src_lat_deg))
    lon1 = np.deg2rad(float(src_lon_deg))
    lat2 = np.deg2rad(np.asarray(dst_lat_deg, dtype=np.float64))
    lon2 = np.deg2rad(np.asarray(dst_lon_deg, dtype=np.float64))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * np.arctan2(
        np.sqrt(np.clip(a, 0.0, 1.0)),
        np.sqrt(np.clip(1.0 - a, 0.0, 1.0)),
    )


def project_2d_raypaths_to_azimuths(
    raypaths: xr.Dataset,
    src_lat_deg: float,
    src_lon_deg: float,
    az_min: float,
    az_max: float,
    az_step: float,
) -> xr.Dataset:
    """Expand 2D raypaths into synthetic 3D by azimuthal projection."""
    az_values = azimuth_sequence(az_min=az_min, az_max=az_max, az_step=az_step)
    if az_values.size <= 1:
        return raypaths

    lat_orig = raypaths["ray_lat_deg"].values
    lon_orig = raypaths["ray_lon_deg"].values
    angular_distance = great_circle_angular_distance_rad(
        src_lat_deg=src_lat_deg,
        src_lon_deg=src_lon_deg,
        dst_lat_deg=lat_orig,
        dst_lon_deg=lon_orig,
    )

    n_points = lat_orig.size
    az_count = az_values.size
    expanded = {}
    for name in raypaths.data_vars:
        expanded[name] = np.tile(raypaths[name].values, az_count)

    expanded_lat = np.empty(n_points * az_count, dtype=np.float64)
    expanded_lon = np.empty(n_points * az_count, dtype=np.float64)
    expanded_az = np.empty(n_points * az_count, dtype=np.float64)

    for idx, azimuth in enumerate(az_values):
        start = idx * n_points
        end = start + n_points
        lat_new, lon_new = forward_geodesic_deg(
            src_lat_deg=src_lat_deg,
            src_lon_deg=src_lon_deg,
            angular_distance_rad=angular_distance,
            azimuth_deg=float(azimuth),
        )
        expanded_lat[start:end] = lat_new
        expanded_lon[start:end] = lon_new
        expanded_az[start:end] = azimuth

    expanded["ray_lat_deg"] = expanded_lat
    expanded["ray_lon_deg"] = expanded_lon
    expanded["ray_azimuth_deg"] = expanded_az

    projected = xr.Dataset(
        {name: (["ray_point"], values) for name, values in expanded.items()},
        attrs=dict(raypaths.attrs),
    )
    projected.attrs["synthetic_3d_from_2d"] = True
    projected.attrs["synthetic_3d_az_min_deg"] = float(az_min)
    projected.attrs["synthetic_3d_az_max_deg"] = float(az_max)
    projected.attrs["synthetic_3d_az_step_deg"] = float(az_step)
    projected.attrs["synthetic_3d_az_count"] = int(az_count)
    projected.attrs["synthetic_3d_note"] = (
        "Projected from single-azimuth 2D rays assuming axisymmetry."
    )
    return projected


def warn_az_interp_approximation() -> None:
    """Emit runtime warnings when az_interp synthetic-3D remapping is active."""
    warnings.warn(
        "az_interp=True remaps a single-azimuth 2D ray solution into synthetic 3D. "
        "This assumes an axisymmetric medium around the source. In non-axisymmetric "
        "atmosphere/wind fields, remapped travel times and amplitudes are not physically valid.",
        RuntimeWarning,
        stacklevel=4,
    )
    warnings.warn(
        "Ray tracing currently writes zero winds into infraGA profile (u=v=0). "
        "If your model includes winds or azimuth-dependent structure, synthetic 3D remapping "
        "is an approximation intended for visualization/sensitivity only.",
        RuntimeWarning,
        stacklevel=4,
    )