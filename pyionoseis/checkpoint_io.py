"""Checkpoint IO helpers for Model3D pipeline state persistence.

This module implements versioned checkpoint manifests and signature sidecars
for saving/loading intermediate Model3D state to disk.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import xarray as xr

from pyionoseis import __version__ as package_version
from pyionoseis import model_io

_CHECKPOINT_SCHEMA_VERSION = 1
_SUPPORTED_SCHEMA_VERSIONS = {1}

_MANIFEST_FILE = "checkpoint_manifest.json"
_SIGNATURE_FILE = "checkpoint.signature.json"

_ARTIFACT_FILENAMES = {
    "grid": "grid.nc",
    "atmosphere": "atmosphere.nc",
    "raypaths": "raypaths.nc",
    "ray_arrivals": "ray_arrivals.nc",
    "continuity": "continuity.nc",
}


class LoadedAtmosphereProfile:
    """Container for atmosphere profiles restored from checkpoints.

    Parameters
    ----------
    atmosphere : xr.Dataset
        Atmosphere dataset with altitude coordinates.
    """

    def __init__(self, atmosphere: xr.Dataset):
        self.atmosphere = atmosphere


def checkpoint_manifest_path(checkpoint_dir: str | Path) -> Path:
    """Return manifest file path inside a checkpoint directory."""
    return Path(checkpoint_dir) / _MANIFEST_FILE


def checkpoint_signature_path(checkpoint_dir: str | Path) -> Path:
    """Return signature file path inside a checkpoint directory."""
    return Path(checkpoint_dir) / _SIGNATURE_FILE


def ensure_checkpoint_dir(output_dir: str | Path, overwrite: bool = False) -> Path:
    """Create/validate a checkpoint directory for writing.

    Parameters
    ----------
    output_dir : str or Path
        Target checkpoint directory.
    overwrite : bool, optional
        If False, raises when directory already contains files.

    Returns
    -------
    Path
        Normalized checkpoint path.
    """
    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(checkpoint_dir.iterdir()):
        raise FileExistsError(
            f"Checkpoint directory is not empty: {checkpoint_dir}. "
            "Use overwrite=True to replace existing contents."
        )
    return checkpoint_dir


def write_dataset_atomic(dataset: xr.Dataset, target_path: Path) -> None:
    """Persist dataset to NetCDF using an atomic rename."""
    tmp_path = target_path.with_name(target_path.name + ".tmp")
    dataset.to_netcdf(tmp_path)
    tmp_path.replace(target_path)


def write_json_atomic(payload: dict[str, Any], target_path: Path) -> None:
    """Write JSON to disk atomically."""
    tmp_path = target_path.with_name(target_path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    tmp_path.replace(target_path)


def file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hash for a file in chunks."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def detect_stage_from_artifacts(artifact_files: dict[str, str]) -> str:
    """Infer checkpoint stage label from available artifact files."""
    keys = set(artifact_files)
    if "continuity" in keys:
        return "pre_tec"
    if "raypaths" in keys:
        return "pre_continuity"
    if {"grid", "atmosphere"}.issubset(keys):
        return "pre_ray"
    return "partial"


def build_manifest(
    model_metadata: dict[str, Any],
    source_metadata: dict[str, Any] | None,
    artifact_files: dict[str, str],
    stage: str | None = None,
) -> dict[str, Any]:
    """Construct checkpoint manifest payload."""
    manifest = {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage or detect_stage_from_artifacts(artifact_files),
        "package_version": str(package_version),
        "model": model_metadata,
        "source": source_metadata,
        "artifacts": artifact_files,
    }
    return manifest


def build_signature_payload(
    manifest: dict[str, Any],
    artifact_hashes: dict[str, str],
) -> dict[str, Any]:
    """Build deterministic signature payload for checkpoint validation."""
    return {
        "signature_version": 1,
        "schema_version": int(manifest["schema_version"]),
        "stage": manifest["stage"],
        "package_version": manifest.get("package_version"),
        "model": manifest.get("model", {}),
        "source": manifest.get("source"),
        "artifacts": manifest.get("artifacts", {}),
        "artifact_hashes": artifact_hashes,
    }


def validate_schema_version(schema_version: int, allow_migration: bool = True) -> None:
    """Validate checkpoint schema version compatibility."""
    if schema_version in _SUPPORTED_SCHEMA_VERSIONS:
        return

    if allow_migration:
        raise ValueError(
            "Unsupported checkpoint schema version "
            f"{schema_version}. Supported versions: "
            f"{sorted(_SUPPORTED_SCHEMA_VERSIONS)}. "
            "Migration path is not available yet."
        )

    raise ValueError(
        "Checkpoint schema version mismatch: "
        f"{schema_version} is unsupported."
    )


def load_manifest(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Load and validate manifest presence from checkpoint directory."""
    manifest_path = checkpoint_manifest_path(checkpoint_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Checkpoint manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def list_expected_artifact_paths(
    checkpoint_dir: str | Path,
    artifacts: dict[str, str],
) -> dict[str, Path]:
    """Resolve artifact names in the manifest to absolute paths."""
    base_dir = Path(checkpoint_dir)
    resolved = {}
    for artifact_name, filename in artifacts.items():
        resolved[artifact_name] = base_dir / filename
    return resolved


def validate_artifact_presence(artifact_paths: dict[str, Path]) -> None:
    """Ensure all manifest-listed artifacts exist on disk."""
    missing = [str(path) for path in artifact_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Checkpoint is missing artifact files: " + ", ".join(missing)
        )


def validate_signature(checkpoint_dir: str | Path, manifest: dict[str, Any]) -> None:
    """Validate signature sidecar against current artifact bytes."""
    sig_path = checkpoint_signature_path(checkpoint_dir)
    if not sig_path.exists():
        raise FileNotFoundError(f"Checkpoint signature not found: {sig_path}")

    with sig_path.open("r", encoding="utf-8") as fh:
        sig_raw = json.load(fh)

    expected_signature_hash = sig_raw.get("signature_hash")
    expected_payload = model_io.normalize_signature_payload(sig_raw)

    artifact_paths = list_expected_artifact_paths(
        checkpoint_dir=checkpoint_dir,
        artifacts=manifest.get("artifacts", {}),
    )
    validate_artifact_presence(artifact_paths)
    actual_hashes = {
        name: file_sha256(path)
        for name, path in artifact_paths.items()
    }

    actual_payload = build_signature_payload(manifest=manifest, artifact_hashes=actual_hashes)
    actual_signature_hash = model_io.canonical_json_hash(actual_payload)

    if expected_payload != actual_payload or expected_signature_hash != actual_signature_hash:
        raise ValueError("Checkpoint signature mismatch: checkpoint artifacts are inconsistent.")


def model_metadata_from_model(model) -> dict[str, Any]:
    """Extract serializable model metadata fields from a Model3D instance."""
    return {
        "name": str(getattr(model, "name", "No-name model")),
        "radius_km": float(getattr(model, "radius", 100.0)),
        "height_km": float(getattr(model, "height", 500.0)),
        "winds": bool(getattr(model, "winds", False)),
        "grid_spacing_deg": float(getattr(model, "grid_spacing", 1.0)),
        "height_spacing_km": float(getattr(model, "height_spacing", 20.0)),
        "atmosphere_model": str(getattr(model, "atmosphere_model", "msise00")),
        "ionosphere_model": str(getattr(model, "ionosphere_model", "iri2020")),
    }


def source_metadata_from_source(source) -> dict[str, Any] | None:
    """Extract serializable source metadata from an EarthquakeSource."""
    if source is None:
        return None

    time_obj = source.get_time()
    time_str = time_obj.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "latitude_deg": float(source.get_latitude()),
        "longitude_deg": float(source.get_longitude()),
        "depth_km": float(source.get_depth()),
        "time_utc": time_str,
    }


def artifact_filename(artifact_name: str) -> str:
    """Return standard filename for a named checkpoint artifact."""
    return _ARTIFACT_FILENAMES[artifact_name]
