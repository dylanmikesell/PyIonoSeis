"""Caching and IO helpers for the Model3D ray-trace workflow.

These functions were extracted from ``Model3D`` to keep the orchestrator class
focused on physics and let this module own all hashing, cache-key, and
signature-file concerns.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def array_sha256(values) -> str:
    """Build a deterministic SHA-256 digest for a numeric array.

    Parameters
    ----------
    values : array-like
        Numeric array to hash.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    array = np.asarray(values, dtype=np.float64)
    hasher = hashlib.sha256()
    hasher.update(str(array.shape).encode("utf-8"))
    hasher.update(array.tobytes())
    return hasher.hexdigest()


def canonical_json_hash(payload) -> str:
    """Hash a JSON-serializable payload using canonical key ordering.

    Parameters
    ----------
    payload : dict
        JSON-serializable mapping.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def normalize_signature_payload(raw: dict) -> dict:
    """Return the inner signature dict when wrapped in a JSON envelope.

    Parameters
    ----------
    raw : dict
        Signature JSON as loaded from disk (may be wrapped or bare).

    Returns
    -------
    dict
        The bare signature payload dict.
    """
    if isinstance(raw, dict) and "signature" in raw and isinstance(raw["signature"], dict):
        return raw["signature"]
    return raw


def signature_path_for_output_prefix(output_prefix) -> Path:
    """Return the sidecar signature JSON path for an infraGA output prefix.

    Parameters
    ----------
    output_prefix : str or Path
        infraGA output prefix (e.g. ``run_dir/infraga_3d_sph_abc123``).

    Returns
    -------
    Path
        Path to the ``.signature.json`` sidecar file.
    """
    return Path(str(output_prefix) + ".signature.json")


def signature_path_for_raypaths(raypaths_file) -> Path:
    """Return the sidecar signature JSON path for a raypaths .dat file.

    Parameters
    ----------
    raypaths_file : str or Path
        Path to the ``*.raypaths.dat`` file.

    Returns
    -------
    Path
        Path to the companion ``.signature.json`` sidecar.
    """
    raypaths_file = Path(raypaths_file)
    name = raypaths_file.name
    if name.endswith(".raypaths.dat"):
        base = name[: -len(".raypaths.dat")]
        return raypaths_file.with_name(base + ".signature.json")
    return Path(str(raypaths_file) + ".signature.json")


def cache_token(signature_hash: str, cache_id=None) -> str:
    """Build a short, filesystem-safe cache token from a signature hash.

    Parameters
    ----------
    signature_hash : str
        Full hex SHA-256 digest.
    cache_id : str, optional
        User-supplied label to embed in the token.

    Returns
    -------
    str
        Cache token string safe for use in file names.
    """
    if cache_id is None:
        return signature_hash[:12]
    token = str(cache_id).strip()
    if not token:
        return signature_hash[:12]
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in token)
