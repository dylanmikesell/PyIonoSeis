#!/usr/bin/env python

"""Tests for infraGA spherical ray tracing integration."""

import tempfile
import json
import unittest
from unittest.mock import patch
from pathlib import Path
from typing import Any, cast
import subprocess

import numpy as np
import xarray as xr

from pyionoseis.infraga import InfraGARayTracingError, parse_sph_raypaths, run_sph_trace
from pyionoseis.model import Model3D
from pyionoseis.source import EarthquakeSource


class _FakeAtmosphere:
    def __init__(self, altitudes):
        self.atmosphere = xr.Dataset(
            {
                "velocity": (["altitude"], np.full(len(altitudes), 0.3)),
                "density": (["altitude"], np.linspace(1.0, 0.1, len(altitudes))),
                "pressure": (["altitude"], np.full(len(altitudes), 1.0e5)),
                "temperature": (["altitude"], np.full(len(altitudes), 273.0)),
            },
            coords={"altitude": altitudes},
        )


class TestInfraGATools(unittest.TestCase):
    def test_parse_sph_raypaths_geometry_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ray_file = f"{tmpdir}/sample.raypaths.dat"
            np.savetxt(
                ray_file,
                np.array(
                    [
                        [38.1, 142.1, 250.0, -3.0, 1.2, 900.0],
                        [38.2, 142.2, 260.0, -2.5, 1.1, 910.0],
                    ]
                ),
            )

            ds = parse_sph_raypaths(Path(ray_file), run_type="3d")

            self.assertIn("ray_lat_deg", ds.data_vars)
            self.assertIn("ray_lon_deg", ds.data_vars)
            self.assertIn("ray_alt_km", ds.data_vars)
            self.assertIn("travel_time_s", ds.data_vars)
            self.assertEqual(ds.sizes["ray_point"], 2)

    @patch("pyionoseis.infraga.subprocess.run")
    def test_run_sph_trace_missing_backend_binary_has_compile_guidance(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["infraga", "sph", "prop"],
            output="",
            stderr=(
                "Traceback...\n"
                "FileNotFoundError: [Errno 2] No such file or directory: '"
                "/tmp/site-packages/bin/infraga-sph'"
            ),
        )

        with self.assertRaises(InfraGARayTracingError) as ctx:
            run_sph_trace(["infraga", "sph", "prop"])

        msg = str(ctx.exception)
        self.assertIn("may not provide a `compile` subcommand", msg)
        self.assertIn("trace_rays(executable=...)", msg)
        self.assertIn("infraga-sph", msg)


class TestModelTraceRays(unittest.TestCase):
    @staticmethod
    def _mock_run_sph_trace(command):
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
            np.array([[45.0, 10.0, 0.0, 38.4, 142.8, 1000.0, 0.3, 250.0, 44.0, 190.0, -4.0, 2.0]]),
        )

    @patch("pyionoseis.model.infraga_tools.run_sph_trace", side_effect=_mock_run_sph_trace.__func__)
    @patch("pyionoseis.model.infraga_tools.resolve_infraga_command", return_value=["infraga"])
    def test_trace_rays_3d_spherical_only(self, _resolve_cmd, _run_trace):
        model = Model3D()
        source = EarthquakeSource()
        model.assign_source(source)
        model.make_3Dgrid()

        model.atmosphere = cast(Any, _FakeAtmosphere(model.grid.coords["altitude"].values))

        ray_ds = model.trace_rays(type="3d", keep_files=False)

        self.assertIn("ray_lat_deg", ray_ds.data_vars)
        self.assertIn("ray_lon_deg", ray_ds.data_vars)
        self.assertIn("ray_alt_km", ray_ds.data_vars)
        self.assertEqual(ray_ds.attrs["raytrace_backend"], "infraga_sph")
        self.assertIn("sph prop", ray_ds.attrs["infraga_command"])
        self.assertIn("--az-min", ray_ds.attrs["infraga_command"])

    @patch("pyionoseis.model.infraga_tools.run_sph_trace", side_effect=_mock_run_sph_trace.__func__)
    @patch("pyionoseis.model.infraga_tools.resolve_infraga_command", return_value=["infraga"])
    def test_trace_rays_2d_maps_to_sph_single_azimuth(self, _resolve_cmd, _run_trace):
        model = Model3D()
        source = EarthquakeSource()
        model.assign_source(source)
        model.make_3Dgrid()

        model.atmosphere = cast(Any, _FakeAtmosphere(model.grid.coords["altitude"].values))

        ray_ds = model.trace_rays(type="2d", azimuth_deg=77.0, keep_files=False)

        # 2D mode should still run via infraga sph and force north azimuth.
        self.assertIn("sph prop", ray_ds.attrs["infraga_command"])
        self.assertIn("--azimuth 0.0", ray_ds.attrs["infraga_command"])
        self.assertTrue(np.allclose(ray_ds["ray_azimuth_deg"].values, 0.0))

    @patch("pyionoseis.model.infraga_tools.accel_sph_available", return_value=True)
    @patch("pyionoseis.model.os.cpu_count", return_value=8)
    @patch("pyionoseis.model.infraga_tools.run_sph_trace", side_effect=_mock_run_sph_trace.__func__)
    @patch("pyionoseis.model.infraga_tools.resolve_infraga_command", return_value=["infraga"])
    def test_trace_rays_2d_use_accel_defaults_cpu_cnt(
        self, _resolve_cmd, _run_trace, _cpu_count, _accel_available
    ):
        model = Model3D()
        source = EarthquakeSource()
        model.assign_source(source)
        model.make_3Dgrid()
        model.atmosphere = cast(Any, _FakeAtmosphere(model.grid.coords["altitude"].values))

        ray_ds = model.trace_rays(type="2d", use_accel=True, keep_files=False)

        self.assertIn("--cpu-cnt 8", ray_ds.attrs["infraga_command"])
        self.assertTrue(ray_ds.attrs["raytrace_accel_requested"])
        self.assertTrue(ray_ds.attrs["raytrace_accel_used"])

    @patch("pyionoseis.model.infraga_tools.accel_sph_available", return_value=True)
    @patch("pyionoseis.model.infraga_tools.run_sph_trace", side_effect=_mock_run_sph_trace.__func__)
    @patch("pyionoseis.model.infraga_tools.resolve_infraga_command", return_value=["infraga"])
    def test_trace_rays_3d_use_accel_respects_explicit_cpu_cnt(
        self, _resolve_cmd, _run_trace, _accel_available
    ):
        model = Model3D()
        source = EarthquakeSource()
        model.assign_source(source)
        model.make_3Dgrid()
        model.atmosphere = cast(Any, _FakeAtmosphere(model.grid.coords["altitude"].values))

        ray_ds = model.trace_rays(type="3d", use_accel=True, cpu_cnt=4, keep_files=False)

        self.assertIn("--cpu-cnt 4", ray_ds.attrs["infraga_command"])
        self.assertTrue(ray_ds.attrs["raytrace_accel_requested"])
        self.assertTrue(ray_ds.attrs["raytrace_accel_used"])

    @patch("pyionoseis.model.infraga_tools.accel_sph_available", return_value=False)
    @patch("pyionoseis.model.infraga_tools.run_sph_trace", side_effect=_mock_run_sph_trace.__func__)
    @patch("pyionoseis.model.infraga_tools.resolve_infraga_command", return_value=["infraga"])
    def test_trace_rays_use_accel_falls_back_when_unavailable(
        self, _resolve_cmd, _run_trace, _accel_available
    ):
        model = Model3D()
        source = EarthquakeSource()
        model.assign_source(source)
        model.make_3Dgrid()
        model.atmosphere = cast(Any, _FakeAtmosphere(model.grid.coords["altitude"].values))

        with self.assertWarns(RuntimeWarning):
            ray_ds = model.trace_rays(type="3d", use_accel=True, cpu_cnt=4, keep_files=False)

        self.assertNotIn("--cpu-cnt", ray_ds.attrs["infraga_command"])
        self.assertTrue(ray_ds.attrs["raytrace_accel_requested"])
        self.assertFalse(ray_ds.attrs["raytrace_accel_used"])

    @patch("pyionoseis.model.infraga_tools.run_sph_trace", side_effect=_mock_run_sph_trace.__func__)
    @patch("pyionoseis.model.infraga_tools.resolve_infraga_command", return_value=["infraga"])
    def test_trace_rays_cache_hit_skips_recompute(self, _resolve_cmd, mock_run_trace):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Model3D()
            source = EarthquakeSource()
            model.assign_source(source)
            model.make_3Dgrid()
            model.atmosphere = cast(Any, _FakeAtmosphere(model.grid.coords["altitude"].values))

            first = model.trace_rays(type="3d", output_dir=tmpdir, reuse_existing=True)
            second = model.trace_rays(type="3d", output_dir=tmpdir, reuse_existing=True)

            self.assertIn("ray_lat_deg", first.data_vars)
            self.assertIn("ray_lat_deg", second.data_vars)
            self.assertEqual(mock_run_trace.call_count, 1)
            self.assertTrue(second.attrs["raytrace_loaded_from_cache"])
            self.assertEqual(first.attrs["raytrace_backend"], "infraga_sph")
            self.assertEqual(second.attrs["raytrace_backend"], "infraga_sph")
            self.assertEqual(first.attrs["raytrace_type"], "3d")
            self.assertEqual(second.attrs["raytrace_type"], "3d")
            self.assertIn("raytrace_signature_hash", first.attrs)
            self.assertIn("raytrace_signature_hash", second.attrs)
            self.assertEqual(
                first.attrs["raytrace_signature_hash"],
                second.attrs["raytrace_signature_hash"],
            )
            self.assertIn("raytrace_signature_file", first.attrs)
            self.assertIn("raytrace_signature_file", second.attrs)
            self.assertTrue(first.attrs["raytrace_signature_file"].endswith(".signature.json"))
            self.assertTrue(second.attrs["raytrace_signature_file"].endswith(".signature.json"))

    @patch("pyionoseis.model.infraga_tools.run_sph_trace", side_effect=_mock_run_sph_trace.__func__)
    @patch("pyionoseis.model.infraga_tools.resolve_infraga_command", return_value=["infraga"])
    def test_trace_rays_cache_miss_on_signature_change_recomputes(self, _resolve_cmd, mock_run_trace):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Model3D()
            source = EarthquakeSource()
            model.assign_source(source)
            model.make_3Dgrid()
            model.atmosphere = cast(Any, _FakeAtmosphere(model.grid.coords["altitude"].values))

            model.trace_rays(type="3d", output_dir=tmpdir, reuse_existing=True, frequency_hz=0.005)
            model.trace_rays(type="3d", output_dir=tmpdir, reuse_existing=True, frequency_hz=0.010)

            self.assertEqual(mock_run_trace.call_count, 2)

    @patch("pyionoseis.model.infraga_tools.run_sph_trace", side_effect=_mock_run_sph_trace.__func__)
    @patch("pyionoseis.model.infraga_tools.resolve_infraga_command", return_value=["infraga"])
    def test_trace_rays_force_recompute_bypasses_cache(self, _resolve_cmd, mock_run_trace):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Model3D()
            source = EarthquakeSource()
            model.assign_source(source)
            model.make_3Dgrid()
            model.atmosphere = cast(Any, _FakeAtmosphere(model.grid.coords["altitude"].values))

            model.trace_rays(type="3d", output_dir=tmpdir, reuse_existing=True)
            ray_ds = model.trace_rays(
                type="3d",
                output_dir=tmpdir,
                reuse_existing=True,
                force_recompute=True,
            )

            self.assertEqual(mock_run_trace.call_count, 2)
            self.assertFalse(ray_ds.attrs["raytrace_loaded_from_cache"])
            self.assertEqual(ray_ds.attrs["raytrace_backend"], "infraga_sph")
            self.assertEqual(ray_ds.attrs["raytrace_type"], "3d")
            self.assertIn("raytrace_signature_hash", ray_ds.attrs)

    @patch("pyionoseis.model.infraga_tools.run_sph_trace", side_effect=_mock_run_sph_trace.__func__)
    @patch("pyionoseis.model.infraga_tools.resolve_infraga_command", return_value=["infraga"])
    def test_trace_rays_az_interp_sets_projection_attrs_and_deduplicates_wraparound(
        self,
        _resolve_cmd,
        _run_trace,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Model3D()
            source = EarthquakeSource()
            model.assign_source(source)
            model.make_3Dgrid()
            model.atmosphere = cast(Any, _FakeAtmosphere(model.grid.coords["altitude"].values))

            ray_ds = model.trace_rays(
                type="2d",
                output_dir=tmpdir,
                reuse_existing=True,
                az_interp=True,
                az_interp_min=0.0,
                az_interp_max=360.0,
                az_interp_step=120.0,
            )

            self.assertTrue(ray_ds.attrs["synthetic_3d_from_2d"])
            self.assertEqual(ray_ds.attrs["synthetic_3d_az_min_deg"], 0.0)
            self.assertEqual(ray_ds.attrs["synthetic_3d_az_max_deg"], 360.0)
            self.assertEqual(ray_ds.attrs["synthetic_3d_az_step_deg"], 120.0)
            self.assertEqual(ray_ds.attrs["synthetic_3d_az_count"], 3)
            self.assertEqual(ray_ds.sizes["ray_point"], 6)
            unique_az = np.unique(np.round(ray_ds["ray_azimuth_deg"].values, 6))
            np.testing.assert_allclose(unique_az, np.array([0.0, 120.0, 240.0]))

    @patch("pyionoseis.model.infraga_tools.run_sph_trace", side_effect=_mock_run_sph_trace.__func__)
    @patch("pyionoseis.model.infraga_tools.resolve_infraga_command", return_value=["infraga"])
    def test_trace_rays_reuse_existing_false_recomputes(self, _resolve_cmd, mock_run_trace):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Model3D()
            source = EarthquakeSource()
            model.assign_source(source)
            model.make_3Dgrid()
            model.atmosphere = cast(Any, _FakeAtmosphere(model.grid.coords["altitude"].values))

            model.trace_rays(type="3d", output_dir=tmpdir, reuse_existing=True)
            model.trace_rays(type="3d", output_dir=tmpdir, reuse_existing=False)

            self.assertEqual(mock_run_trace.call_count, 2)

    @patch("pyionoseis.model.infraga_tools.run_sph_trace", side_effect=_mock_run_sph_trace.__func__)
    @patch("pyionoseis.model.infraga_tools.resolve_infraga_command", return_value=["infraga"])
    def test_load_rays_loads_outputs_and_handles_missing_arrivals(self, _resolve_cmd, _run_trace):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Model3D()
            source = EarthquakeSource()
            model.assign_source(source)
            model.make_3Dgrid()
            model.atmosphere = cast(Any, _FakeAtmosphere(model.grid.coords["altitude"].values))
            model.trace_rays(type="3d", output_dir=tmpdir, reuse_existing=True)

            raypaths_files = list(Path(tmpdir).glob("*.raypaths.dat"))
            self.assertEqual(len(raypaths_files), 1)
            raypaths_file = raypaths_files[0]
            arrivals_file = Path(str(raypaths_file).replace(".raypaths.dat", ".arrivals.dat"))
            arrivals_file.unlink()

            model2 = Model3D()
            model2.assign_source(source)
            model2.make_3Dgrid()
            loaded = model2.load_rays(raypaths_file=raypaths_file, type="3d")

            self.assertIn("ray_lat_deg", loaded.data_vars)
            self.assertEqual(model2.ray_arrivals.sizes, {})

    @patch("pyionoseis.model.infraga_tools.run_sph_trace", side_effect=_mock_run_sph_trace.__func__)
    @patch("pyionoseis.model.infraga_tools.resolve_infraga_command", return_value=["infraga"])
    def test_load_rays_validate_signature_raises_on_mismatch(self, _resolve_cmd, _run_trace):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Model3D()
            source = EarthquakeSource()
            model.assign_source(source)
            model.make_3Dgrid()
            model.atmosphere = cast(Any, _FakeAtmosphere(model.grid.coords["altitude"].values))
            model.trace_rays(type="3d", output_dir=tmpdir, reuse_existing=True)

            raypaths_file = list(Path(tmpdir).glob("*.raypaths.dat"))[0]
            signature_file = list(Path(tmpdir).glob("*.signature.json"))[0]

            model3 = Model3D()
            changed_source = EarthquakeSource()
            changed_source.latitude = 1.0
            model3.assign_source(changed_source)
            model3.make_3Dgrid()
            model3.atmosphere = cast(Any, _FakeAtmosphere(model3.grid.coords["altitude"].values))

            with self.assertRaises(ValueError):
                model3.load_rays(
                    raypaths_file=raypaths_file,
                    type="3d",
                    validate_signature=True,
                    signature_file=signature_file,
                )

    @patch("pyionoseis.model.infraga_tools.run_sph_trace", side_effect=_mock_run_sph_trace.__func__)
    @patch("pyionoseis.model.infraga_tools.resolve_infraga_command", return_value=["infraga"])
    def test_trace_rays_signature_payload_has_expected_schema(self, _resolve_cmd, _run_trace):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Model3D()
            source = EarthquakeSource()
            model.assign_source(source)
            model.make_3Dgrid()
            model.atmosphere = cast(Any, _FakeAtmosphere(model.grid.coords["altitude"].values))

            model.trace_rays(type="3d", output_dir=tmpdir, reuse_existing=True)

            signature_files = list(Path(tmpdir).glob("*.signature.json"))
            self.assertEqual(len(signature_files), 1)
            with signature_files[0].open("r", encoding="utf-8") as fh:
                payload = json.load(fh)

            self.assertIn("signature_hash", payload)
            self.assertIn("signature", payload)
            self.assertIn("signature_version", payload["signature"])
            self.assertIn("ray_params", payload["signature"])
            self.assertIn("profile", payload["signature"])
            self.assertEqual(payload["signature"]["run_type"], "3d")

    def test_azimuth_sequence_normalizes_wrap_and_deduplicates_endpoints(self):
        values = Model3D._azimuth_sequence(az_min=0.0, az_max=360.0, az_step=90.0)
        expected = np.array([0.0, 90.0, 180.0, 270.0])
        np.testing.assert_allclose(values, expected)
