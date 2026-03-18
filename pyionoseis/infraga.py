"""Utilities for running infraGA spherical ray tracing from pyionoseis."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import xarray as xr


class InfraGARayTracingError(RuntimeError):
    """Raised when infraGA ray tracing fails."""


def accel_sph_available() -> bool:
    """Check whether infraGA accelerated spherical backend is available.

    Returns True when both of the following are available:
    - `infraga-accel-sph` backend binary
    - `mpirun` launcher
    """
    has_mpirun = shutil.which("mpirun") is not None
    if not has_mpirun:
        return False

    # Most reliable path for installed infraGA Python wrapper.
    try:
        import infraga.run_infraga as run_infraga  # type: ignore

        accel_bin = Path(run_infraga.bin_path) / "infraga-accel-sph"
        if accel_bin.exists() and os.access(accel_bin, os.X_OK):
            return True
    except Exception:
        pass

    # Fallback for environments where binary is provided on PATH directly.
    return shutil.which("infraga-accel-sph") is not None


def resolve_infraga_command(executable: str | None = None) -> list[str]:
    """Resolve the infraGA CLI command prefix.

    Returns either:
    - ["infraga"] when available in PATH
    - [<venv>/bin/infraga] when available next to sys.executable
    - [sys.executable, "-m", "infraga.cli"] when module layout provides it
    - [sys.executable, "-m", "infraga"] as final fallback
    - [executable] when an explicit executable path/name is provided
    """
    if executable:
        return [executable]

    if shutil.which("infraga"):
        return ["infraga"]

    # If running from a venv/kernel where PATH is not activated, prefer sibling script.
    local_cli = Path(sys.executable).with_name("infraga")
    if local_cli.exists():
        return [str(local_cli)]

    # Support both historical and current infraGA module layouts.
    if importlib.util.find_spec("infraga.cli") is not None:
        return [sys.executable, "-m", "infraga.cli"]

    return [sys.executable, "-m", "infraga"]


def write_zcuvd_profile(atmosphere: xr.Dataset, file_path: Path) -> Path:
    """Write a zcuvd profile file for infraGA sph propagation.

    The column order is:
    z [km], c [km/s], u [m/s], v [m/s], d [kg/m^3]

    Winds are set to zero for now and tracked in metadata by caller.
    """
    alt_km = atmosphere["altitude"].values.astype(float)
    sound_speed_km_s = atmosphere["velocity"].values.astype(float)
    density_kg_m3 = atmosphere["density"].values.astype(float)

    meridional_wind_ms = np.zeros_like(alt_km)
    zonal_wind_ms = np.zeros_like(alt_km)

    profile = np.column_stack(
        [alt_km, sound_speed_km_s, meridional_wind_ms, zonal_wind_ms, density_kg_m3]
    )
    np.savetxt(file_path, profile, fmt="%.8e")
    return file_path


def build_sph_prop_command(
    cmd_prefix: list[str],
    atmo_file: Path,
    output_id: Path,
    src_lat: float,
    src_lon: float,
    run_type: str,
    azimuth_deg: float = 0.0,
    az_min: float = 0.0,
    az_max: float = 360.0,
    az_step: float = 10.0,
    incl_min: float = 45.0,
    incl_max: float = 90.0,
    incl_step: float = 1.0,
    bounces: int = 0,
    max_rng_km: float = 5000.0,
    frequency_hz: float = 0.005,
    cpu_cnt: int | None = None,
    prof_format: str = "zcuvd",
) -> list[str]:
    """Build the `infraga sph prop` command line."""
    command = [
        *cmd_prefix,
        "sph",
        "prop",
        "--atmo-file",
        str(atmo_file),
        "--src-lat",
        str(src_lat),
        "--src-lon",
        str(src_lon),
        "--incl-min",
        str(incl_min),
        "--incl-max",
        str(incl_max),
        "--incl-step",
        str(incl_step),
        "--bounces",
        str(bounces),
        "--max-rng",
        str(max_rng_km),
        "--freq",
        str(frequency_hz),
        "--prof-format",
        prof_format,
        "--output-id",
        str(output_id),
    ]

    if run_type == "2d":
        command.extend(["--azimuth", str(azimuth_deg)])
    else:
        command.extend(
            [
                "--az-min",
                str(az_min),
                "--az-max",
                str(az_max),
                "--az-step",
                str(az_step),
            ]
        )

    if cpu_cnt is not None:
        command.extend(["--cpu-cnt", str(cpu_cnt)])

    return command


def run_sph_trace(command: list[str]) -> subprocess.CompletedProcess:
    """Execute infraGA spherical propagation command."""
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        missing_backend = "FileNotFoundError" in (exc.stderr or "") and "infraga-sph" in (
            exc.stderr or ""
        )

        guidance = ""
        if missing_backend:
            guidance = (
                "\nDetected infraGA CLI without a spherical backend binary (`infraga-sph`).\n"
                "This installed infraGA CLI may not provide a `compile` subcommand.\n"
                "Install/build infraGA native binaries for this environment, then retry.\n"
                "If you already built binaries elsewhere, pass an explicit executable path via "
                "`trace_rays(executable=...)`."
            )

        message = (
            "infraGA spherical propagation failed. "
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
            f"{guidance}"
        )
        raise InfraGARayTracingError(message) from exc
    except FileNotFoundError as exc:
        message = (
            "infraGA executable not found. Install infraGA and compile methods, "
            "or provide an explicit executable to trace_rays()."
        )
        raise InfraGARayTracingError(message) from exc


def parse_sph_raypaths(raypaths_file: Path, run_type: str) -> xr.Dataset:
    """Parse infraGA spherical raypaths output into an xarray Dataset."""
    data = np.loadtxt(raypaths_file, comments="#", ndmin=2)
    if data.shape[1] < 6:
        raise InfraGARayTracingError(
            f"Unexpected raypaths format in {raypaths_file}. "
            f"Expected at least 6 columns, found {data.shape[1]}."
        )

    dataset = xr.Dataset(
        {
            "ray_lat_deg": (["ray_point"], data[:, 0]),
            "ray_lon_deg": (["ray_point"], data[:, 1]),
            "ray_alt_km": (["ray_point"], data[:, 2]),
            "transport_amplitude_db": (["ray_point"], data[:, 3]),
            "absorption_db": (["ray_point"], data[:, 4]),
            "travel_time_s": (["ray_point"], data[:, 5]),
            # These are placeholders for now and are populated later from arrivals.
            "ray_azimuth_deg": (["ray_point"], np.full(data.shape[0], np.nan)),
            "ray_inclination_deg": (["ray_point"], np.full(data.shape[0], np.nan)),
        }
    )

    if run_type == "2d":
        # For spherical single-azimuth mode, all rays share the launch azimuth.
        dataset["ray_azimuth_deg"] = xr.DataArray(
            np.full(data.shape[0], 0.0), dims=["ray_point"]
        )

    return dataset


def parse_arrivals(arrivals_file: Path) -> xr.Dataset:
    """Parse infraGA arrivals output into an xarray Dataset when available."""
    # Some infraGA runs create an empty arrivals file; skip loading to avoid
    # noisy warnings in notebooks/scripts.
    with arrivals_file.open("r", encoding="utf-8", errors="ignore") as fh:
        has_data = any(line.strip() and not line.lstrip().startswith("#") for line in fh)
    if not has_data:
        return xr.Dataset()

    data = np.loadtxt(arrivals_file, comments="#", ndmin=2)
    if data.size == 0:
        return xr.Dataset()

    dataset = xr.Dataset()

    # Column indices follow infraGA spherical arrivals headers.
    if data.shape[1] >= 1:
        dataset["ray_inclination_deg"] = (["arrival"], data[:, 0])
    if data.shape[1] >= 2:
        dataset["ray_azimuth_deg"] = (["arrival"], data[:, 1])
    if data.shape[1] >= 6:
        dataset["travel_time_s"] = (["arrival"], data[:, 5])
    if data.shape[1] >= 11:
        dataset["transport_amplitude_db"] = (["arrival"], data[:, 10])
    if data.shape[1] >= 12:
        dataset["absorption_db"] = (["arrival"], data[:, 11])

    return dataset
