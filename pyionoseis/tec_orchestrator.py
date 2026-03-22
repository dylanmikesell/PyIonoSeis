"""Internal orchestration helpers for LOS TEC computation."""

from __future__ import annotations

from typing import Callable

import xarray as xr


def resolve_density_perturbation(
    dNe: xr.DataArray | None,
    continuity: xr.Dataset | None,
) -> xr.DataArray | None:
    """Resolve density perturbation from explicit input or model continuity state."""
    if dNe is not None:
        return dNe
    if continuity is not None and "dNe" in continuity:
        return continuity["dNe"]
    return None


def build_los_ids(
    receiver_positions,
    sat_id: str | None,
    constellation: str | None,
    prn: int | None,
) -> tuple[str | None, str | None]:
    """Build receiver/satellite identifiers used in LOS TEC metadata."""
    receiver_id = receiver_positions.get("code") if isinstance(receiver_positions, dict) else None

    satellite_id = sat_id
    if satellite_id is None and constellation is not None and prn is not None:
        satellite_id = f"{constellation}{int(prn):02d}"

    return receiver_id, satellite_id


def compute_los_tec(
    grid: xr.Dataset,
    receiver_positions,
    satellite_positions,
    dNe: xr.DataArray | None,
    tec_config,
    receiver_id: str | None,
    satellite_id: str | None,
    compute_los_tec_fn: Callable,
) -> xr.Dataset:
    """Compute LOS TEC by delegating to tec module implementation."""
    return compute_los_tec_fn(
        grid=grid,
        receiver_positions=receiver_positions,
        satellite_positions=satellite_positions,
        dNe=dNe,
        tec_config=tec_config,
        receiver_id=receiver_id,
        satellite_id=satellite_id,
    )
