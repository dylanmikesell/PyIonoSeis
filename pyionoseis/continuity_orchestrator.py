"""Internal orchestration helpers for continuity assignment.

These helpers preserve ``Model3D.assign_continuity`` behavior while reducing
method complexity in ``model.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import xarray as xr

from pyionoseis import model_io


def resolve_timing(
    t0_s: float | None,
    tmax_s: float | None,
    dt_s: float | None,
    default_t0_s: float | None,
    default_tmax_s: float | None,
    default_dt_s: float | None,
) -> tuple[float, float, float]:
    """Resolve continuity timing using explicit values or stored defaults."""
    if t0_s is None:
        t0_s = default_t0_s
    if tmax_s is None:
        tmax_s = default_tmax_s
    if dt_s is None:
        dt_s = default_dt_s

    if t0_s is None or tmax_s is None or dt_s is None:
        raise ValueError(
            "Continuity timing parameters are missing. Set [continuity] in the TOML "
            "or pass t0_s/tmax_s/dt_s explicitly."
        )

    return float(t0_s), float(tmax_s), float(dt_s)


def ensure_travel_time_on_grid(
    grid: xr.Dataset,
    raypaths: xr.Dataset,
    map_ray_scalar_to_grid: Callable,
    interpolation_radius_km: float,
    mapping_mode: str,
    use_kdtree: bool,
    altitude_window_km: float,
    min_points: int,
    weight_power: float,
    chunk_size: int,
) -> None:
    """Map travel time to the grid if missing."""
    if "travel_time_s" in grid:
        return

    travel_map = map_ray_scalar_to_grid(
        grid,
        raypaths,
        ray_var="travel_time_s",
        output_name="travel_time_s",
        interpolation_radius_km=float(interpolation_radius_km),
        mapping_mode=str(mapping_mode),
        use_kdtree=bool(use_kdtree),
        altitude_window_km=float(altitude_window_km),
        min_points=int(min_points),
        weight_power=float(weight_power),
        chunk_size=int(chunk_size),
    )
    grid["travel_time_s"] = travel_map["travel_time_s"]
    grid["travel_time_s_raypoint_count"] = travel_map["travel_time_s_raypoint_count"]
    grid["travel_time_s"].attrs["units"] = "s"


def ensure_amplitude_on_grid(
    grid: xr.Dataset,
    raypaths: xr.Dataset,
    map_ray_scalar_to_grid: Callable,
    interpolation_radius_km: float,
    mapping_mode: str,
    use_kdtree: bool,
    altitude_window_km: float,
    min_points: int,
    weight_power: float,
    chunk_size: int,
) -> None:
    """Map linear transport amplitude to the grid if missing."""
    if "infraga_amplitude" in grid:
        return

    if "transport_amplitude_db" not in raypaths:
        raise KeyError("Raypaths missing transport_amplitude_db.")

    amp_db = raypaths["transport_amplitude_db"].values.astype(float)
    amp_linear = np.power(10.0, amp_db / 20.0)
    amp_paths = raypaths.copy()
    amp_paths["transport_amplitude_linear"] = (["ray_point"], amp_linear)
    amp_paths["transport_amplitude_linear"].attrs["units"] = "dimensionless"

    amp_map = map_ray_scalar_to_grid(
        grid,
        amp_paths,
        ray_var="transport_amplitude_linear",
        output_name="infraga_amplitude",
        interpolation_radius_km=float(interpolation_radius_km),
        mapping_mode=str(mapping_mode),
        use_kdtree=bool(use_kdtree),
        altitude_window_km=float(altitude_window_km),
        min_points=int(min_points),
        weight_power=float(weight_power),
        chunk_size=int(chunk_size),
    )
    grid["infraga_amplitude"] = amp_map["infraga_amplitude"]
    grid["infraga_amplitude_raypoint_count"] = amp_map[
        "infraga_amplitude_raypoint_count"
    ]
    grid["infraga_amplitude"].attrs["units"] = "dimensionless"


def build_signature_payload(
    grid: xr.Dataset,
    ray_signature_hash: str | None,
    t0_s: float,
    tmax_s: float,
    dt_s: float,
    b: float,
    divergence_flag: int,
    geomag_flag: bool,
    interpolation_radius_km: float,
    mapping_mode: str,
    use_kdtree: bool,
    altitude_window_km: float,
    min_points: int,
    weight_power: float,
    chunk_size: int,
    store_neutral_velocity: bool,
) -> dict:
    """Build canonical continuity signature payload."""
    return {
        "signature_version": 1,
        "continuity": {
            "t0_s": float(t0_s),
            "tmax_s": float(tmax_s),
            "dt_s": float(dt_s),
            "b": float(b),
            "divergence_flag": int(divergence_flag),
            "geomag_flag": bool(geomag_flag),
            "store_neutral_velocity": bool(store_neutral_velocity),
        },
        "mapping": {
            "interpolation_radius_km": float(interpolation_radius_km),
            "mapping_mode": str(mapping_mode),
            "use_kdtree": bool(use_kdtree),
            "altitude_window_km": float(altitude_window_km),
            "min_points": int(min_points),
            "weight_power": float(weight_power),
            "chunk_size": int(chunk_size),
        },
        "ray_signature_hash": ray_signature_hash,
        "grid": {
            "latitude_sha256": model_io.array_sha256(grid.coords["latitude"].values),
            "longitude_sha256": model_io.array_sha256(grid.coords["longitude"].values),
            "altitude_sha256": model_io.array_sha256(grid.coords["altitude"].values),
        },
    }


def resolve_cache_paths(
    output_dir: str | None,
    signature_hash: str,
    cache_id: str | None,
) -> tuple[Path | None, Path | None]:
    """Resolve continuity output and signature file paths for cache usage."""
    if output_dir is None:
        return None, None

    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_token = model_io.cache_token(signature_hash=signature_hash, cache_id=cache_id)
    output_prefix = run_dir / f"continuity_{output_token}"
    output_file = Path(str(output_prefix) + ".nc")
    signature_file = model_io.signature_path_for_output_prefix(output_prefix)
    return output_file, signature_file


def load_cached_dataset_if_valid(
    output_file: Path | None,
    signature_file: Path | None,
    signature_payload: dict,
    reuse_existing: bool,
    force_recompute: bool,
    load_dataset: Callable,
) -> xr.Dataset | None:
    """Load cached continuity output when cache and signatures match."""
    if output_file is None or signature_file is None:
        return None

    if not reuse_existing or force_recompute:
        return None

    if not output_file.exists() or not signature_file.exists():
        return None

    with signature_file.open("r", encoding="utf-8") as fh:
        saved_raw = json.load(fh)
    saved_payload = model_io.normalize_signature_payload(saved_raw)
    if saved_payload != signature_payload:
        return None

    continuity = load_dataset(output_file)
    continuity.attrs["continuity_loaded_from_cache"] = 1
    return continuity


def annotate_continuity_result(
    continuity: xr.Dataset,
    grid: xr.Dataset,
    signature_hash: str,
    signature_payload: dict,
) -> xr.Dataset:
    """Attach mapped scalars and continuity metadata to solver output."""
    continuity["travel_time_s"] = grid["travel_time_s"]
    continuity["infraga_amplitude"] = grid["infraga_amplitude"]
    continuity.attrs["continuity_loaded_from_cache"] = 0
    continuity.attrs["continuity_signature_hash"] = signature_hash
    continuity.attrs["continuity_signature_payload"] = json.dumps(
        signature_payload,
        sort_keys=True,
    )
    return continuity


def persist_continuity_with_signature(
    continuity: xr.Dataset,
    output_file: Path | None,
    signature_file: Path | None,
    signature_hash: str,
    signature_payload: dict,
) -> None:
    """Persist continuity output and sidecar signature when output paths exist."""
    if output_file is None or signature_file is None:
        return

    continuity.to_netcdf(output_file)
    with signature_file.open("w", encoding="utf-8") as fh:
        json.dump(
            {"signature_hash": signature_hash, "signature": signature_payload},
            fh,
            indent=2,
            sort_keys=True,
        )
