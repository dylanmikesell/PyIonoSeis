#!/usr/bin/env python

"""Tests for Model3D checkpoint save/load workflow."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from typing import cast

import numpy as np
import xarray as xr

from pyionoseis import checkpoint_io
from pyionoseis.model import Model3D


class _DummySource:
    """Minimal source fixture with Model3D-required methods."""

    def __init__(self):
        self._lat = 38.322
        self._lon = 142.369
        self._depth = 24.0
        self._time = datetime(2011, 3, 11, 5, 46, 23, tzinfo=timezone.utc)

    def get_latitude(self):
        return self._lat

    def get_longitude(self):
        return self._lon

    def get_depth(self):
        return self._depth

    def get_time(self):
        return self._time


class TestModelCheckpoint(unittest.TestCase):
    """Tests for Model3D checkpoint persistence and restore."""

    def setUp(self):
        """Create compact fixtures for checkpoint tests."""
        self.lat = np.array([38.0, 38.5])
        self.lon = np.array([142.0, 142.5])
        self.alt = np.array([100.0, 120.0, 140.0])

    def _new_model_pre_ray(self):
        """Return a model populated through pre-ray stage artifacts."""
        model = Model3D()
        model.source = _DummySource()
        model.grid = xr.Dataset(
            {
                "electron_density": (
                    ("latitude", "longitude", "altitude"),
                    np.full((self.lat.size, self.lon.size, self.alt.size), 1.0e11),
                )
            },
            coords={
                "latitude": self.lat,
                "longitude": self.lon,
                "altitude": self.alt,
            },
        )
        atmosphere = xr.Dataset(
            {
                "velocity": (("altitude",), np.full(self.alt.size, 300.0)),
                "density": (("altitude",), np.linspace(1.2, 0.9, self.alt.size)),
                "temperature": (("altitude",), np.full(self.alt.size, 250.0)),
                "pressure": (("altitude",), np.full(self.alt.size, 9.0e4)),
            },
            coords={"altitude": self.alt},
        )
        model.atmosphere = checkpoint_io.LoadedAtmosphereProfile(atmosphere=atmosphere)
        return model

    def _write_temp_toml(self):
        """Create a minimal TOML config file and return its path."""
        content = """
[model]
name = "toml-model"
radius = 100.0
height = 500.0
grid_spacing = 1.0
height_spacing = 20.0
""".strip()
        fd, path = tempfile.mkstemp(suffix=".toml")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        return path

    @staticmethod
    def _mock_run_sph_trace(command):
        """Create deterministic mock infraGA output files for trace tests."""
        output_idx = command.index("--output-id") + 1
        output_prefix = command[output_idx]
        np.savetxt(
            f"{output_prefix}.raypaths.dat",
            np.array(
                [
                    [38.4, 142.8, 200.0, -4.0, 2.0, 1000.0],
                    [38.5, 142.9, 220.0, -3.0, 1.7, 1010.0],
                ]
            ),
        )
        np.savetxt(
            f"{output_prefix}.arrivals.dat",
            np.array(
                [[45.0, 10.0, 0.0, 38.4, 142.8, 1000.0, 0.3, 250.0, 44.0, 190.0, -4.0, 2.0]]
            ),
        )

    def test_save_load_checkpoint_pre_ray_roundtrip(self):
        """Checkpoint pre-ray artifacts and load them into a fresh model."""
        model = self._new_model_pre_ray()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "ckpt"
            model.save_checkpoint(checkpoint_dir)

            restored = Model3D()
            manifest = restored.load_checkpoint(checkpoint_dir)

        self.assertEqual(manifest["stage"], "pre_ray")
        self.assertIn("grid", manifest["artifacts"])
        self.assertIn("atmosphere", manifest["artifacts"])
        self.assertIn("electron_density", restored.grid.data_vars)
        atmosphere = restored.atmosphere
        if atmosphere is None:
            self.fail("Expected restored atmosphere dataset")
        source = restored.source
        if source is None:
            self.fail("Expected restored source")
        atmosphere_ds = cast(xr.Dataset, atmosphere.atmosphere)
        self.assertIn("velocity", atmosphere_ds.data_vars)
        self.assertAlmostEqual(source.get_latitude(), 38.322)
        self.assertAlmostEqual(source.get_longitude(), 142.369)

    def test_load_checkpoint_detects_signature_tamper(self):
        """Signature validation fails when checkpoint artifacts are modified."""
        model = self._new_model_pre_ray()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "ckpt"
            model.save_checkpoint(checkpoint_dir)

            with (checkpoint_dir / "grid.nc").open("ab") as fh:
                fh.write(b"tampered")

            restored = Model3D()
            with self.assertRaisesRegex(ValueError, "signature mismatch"):
                restored.load_checkpoint(checkpoint_dir, validate_signature=True)

    def test_load_checkpoint_unsupported_schema_raises(self):
        """Loader rejects unsupported checkpoint schema versions."""
        model = self._new_model_pre_ray()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "ckpt"
            model.save_checkpoint(checkpoint_dir)

            manifest_path = checkpoint_dir / "checkpoint_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = 999
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            restored = Model3D()
            with self.assertRaisesRegex(ValueError, "schema version"):
                restored.load_checkpoint(checkpoint_dir, allow_migration=False)

    def test_save_checkpoint_toml_initialized_model_pre_ray(self):
        """TOML-initialized model can save pre-ray checkpoints without AttributeError."""
        toml_path = self._write_temp_toml()
        try:
            model = Model3D(toml_file=toml_path)
            model.source = _DummySource()
            model.grid = xr.Dataset(
                {
                    "electron_density": (
                        ("latitude", "longitude", "altitude"),
                        np.full((self.lat.size, self.lon.size, self.alt.size), 1.0e11),
                    )
                },
                coords={
                    "latitude": self.lat,
                    "longitude": self.lon,
                    "altitude": self.alt,
                },
            )
            atmosphere = xr.Dataset(
                {
                    "velocity": (("altitude",), np.full(self.alt.size, 300.0)),
                    "density": (("altitude",), np.linspace(1.2, 0.9, self.alt.size)),
                },
                coords={"altitude": self.alt},
            )
            model.atmosphere = checkpoint_io.LoadedAtmosphereProfile(
                atmosphere=atmosphere
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_dir = Path(tmpdir) / "ckpt"
                model.save_checkpoint(checkpoint_dir)
                self.assertTrue((checkpoint_dir / "checkpoint_manifest.json").exists())
                self.assertTrue((checkpoint_dir / "grid.nc").exists())
                self.assertTrue((checkpoint_dir / "atmosphere.nc").exists())
        finally:
            Path(toml_path).unlink(missing_ok=True)

    def test_save_checkpoint_toml_initialized_empty_state_raises_value_error(self):
        """TOML-initialized model without artifacts raises ValueError, not AttributeError."""
        toml_path = self._write_temp_toml()
        try:
            model = Model3D(toml_file=toml_path)
            with tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaisesRegex(ValueError, "No checkpointable artifacts"):
                    model.save_checkpoint(Path(tmpdir) / "ckpt")
        finally:
            Path(toml_path).unlink(missing_ok=True)

    @mock.patch(
        "pyionoseis.model.infraga_tools.run_sph_trace",
        side_effect=_mock_run_sph_trace.__func__,
    )
    @mock.patch("pyionoseis.model.infraga_tools.resolve_infraga_command", return_value=["infraga"])
    def test_reload_pre_ray_checkpoint_supports_alternate_trace_options(
        self,
        _resolve_cmd,
        mock_run_trace,
    ):
        """Reloaded pre-ray checkpoint can run trace_rays with different options."""
        model = self._new_model_pre_ray()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "ckpt"
            model.save_checkpoint(checkpoint_dir)

            restored = Model3D()
            restored.load_checkpoint(checkpoint_dir)

            run_dir = Path(tmpdir) / "rays"
            restored.trace_rays(type="2d", output_dir=run_dir, az_step=20.0)
            restored.trace_rays(
                type="2d",
                output_dir=run_dir,
                reuse_existing=False,
                az_step=10.0,
                incl_step=0.5,
            )

        self.assertEqual(mock_run_trace.call_count, 2)
        raypaths = restored.raypaths
        if raypaths is None:
            self.fail("Expected raypaths after restored trace run")
        self.assertIn("ray_lat_deg", raypaths.data_vars)
