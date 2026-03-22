"""Internal orchestration helpers for Model3D grid enrichment.

These helpers keep ``Model3D`` wrappers small while preserving existing
behavior for ionosphere and magnetic-field profile assignment.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np


def build_profile_arg_list(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    altitudes: np.ndarray,
    time,
    model_name: str,
) -> list[tuple[float, float, np.ndarray, object, str]]:
    """Build per-column argument tuples for profile workers."""
    return [
        (float(lat), float(lon), altitudes, time, model_name)
        for lat in latitudes
        for lon in longitudes
    ]


def run_profile_workers(
    arg_list: list[tuple[float, float, np.ndarray, object, str]],
    worker: Callable,
    fallback_factory: Callable[[int], object],
    warning_prefix: str,
    logger,
    max_workers: int | None = None,
) -> list[object]:
    """Run profile workers in parallel and map results to original columns."""
    if not arg_list:
        return []

    altitude_count = len(arg_list[0][2])
    results: list[object] = [None] * len(arg_list)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, a): k for k, a in enumerate(arg_list)}
        for fut in as_completed(futures):
            k = futures[fut]
            try:
                results[k] = fut.result()
            except Exception as exc:
                lat_k, lon_k = arg_list[k][0], arg_list[k][1]
                logger.warning(
                    "%s failed at lat=%.2f lon=%.2f; storing NaN profile.",
                    warning_prefix,
                    lat_k,
                    lon_k,
                    exc_info=exc,
                )
                results[k] = fallback_factory(altitude_count)

    return results


def reshape_scalar_profiles(
    results: list[np.ndarray],
    latitude_count: int,
    longitude_count: int,
    altitude_count: int,
) -> np.ndarray:
    """Reshape 1-D profile list into a 3-D ``(lat, lon, alt)`` array."""
    output = np.empty((latitude_count, longitude_count, altitude_count))
    for k, profile in enumerate(results):
        i, j = divmod(k, longitude_count)
        output[i, j, :] = profile
    return output


def reshape_vector_profiles(
    results: list[dict[str, np.ndarray]],
    variable_names: list[str],
    latitude_count: int,
    longitude_count: int,
    altitude_count: int,
) -> dict[str, np.ndarray]:
    """Reshape dict profile list into per-variable 3-D arrays."""
    arrays = {
        name: np.full((latitude_count, longitude_count, altitude_count), np.nan)
        for name in variable_names
    }
    for k, profile in enumerate(results):
        i, j = divmod(k, longitude_count)
        for name in variable_names:
            arrays[name][i, j, :] = profile[name]
    return arrays
