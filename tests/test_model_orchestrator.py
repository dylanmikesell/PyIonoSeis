#!/usr/bin/env python

"""Targeted orchestrator tests for Model3D fallback and cache behavior."""

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import xarray as xr

from pyionoseis.model import Model3D


class _DummySource:
    """Minimal source fixture with Model3D-required methods."""

    def __init__(self):
        self._lat = 38.322
        self._lon = 142.369
        self._time = datetime(2011, 3, 11, 5, 46, 23, tzinfo=timezone.utc)

    def get_latitude(self):
        return self._lat

    def get_longitude(self):
        return self._lon

    def get_time(self):
        return self._time


class TestModelOrchestrator(unittest.TestCase):
    """Tests for orchestration failure/fallback and continuity cache logic."""

    def setUp(self):
        """Create a tiny grid fixture for fast orchestrator tests."""
        self.lat = np.array([38.0, 38.5])
        self.lon = np.array([142.0, 142.5])
        self.alt = np.array([100.0, 110.0, 120.0])

    def _new_model_with_grid(self):
        """Return a Model3D with source and a minimal 3-D grid."""
        model = Model3D()
        model.source = _DummySource()
        model.grid = xr.Dataset(
            coords={
                "latitude": self.lat,
                "longitude": self.lon,
                "altitude": self.alt,
            }
        )
        return model

    def test_assign_ionosphere_worker_failure_stores_nan_profile(self):
        """Ionosphere worker exceptions are isolated and failed profiles are NaN."""
        model = self._new_model_with_grid()

        call_counter = {"n": 0}

        def _worker_side_effect(args):
            call_counter["n"] += 1
            if call_counter["n"] == 1:
                raise RuntimeError("synthetic worker failure")
            return np.full(len(self.alt), 1.23e11)

        with mock.patch(
            "pyionoseis.model._iono_profile_worker", side_effect=_worker_side_effect
        ):
            model.assign_ionosphere(max_workers=1)

        self.assertIn("electron_density", model.grid.data_vars)
        ne = model.grid["electron_density"].values
        self.assertEqual(ne.shape, (self.lat.size, self.lon.size, self.alt.size))
        self.assertTrue(np.isnan(ne[0, 0, :]).all())
        self.assertTrue(np.isfinite(ne[0, 1, :]).all())
        self.assertTrue(np.isfinite(ne[1, 0, :]).all())
        self.assertTrue(np.isfinite(ne[1, 1, :]).all())

    def test_assign_magnetic_field_worker_failure_stores_nan_profile(self):
        """Magnetic worker exceptions are isolated and failed profiles are NaN."""
        model = self._new_model_with_grid()

        call_counter = {"n": 0}

        def _worker_side_effect(args):
            call_counter["n"] += 1
            if call_counter["n"] == 1:
                raise RuntimeError("synthetic worker failure")
            return {
                "Be": np.full(len(self.alt), 10.0),
                "Bn": np.full(len(self.alt), 20.0),
                "Bu": np.full(len(self.alt), 30.0),
                "Br": np.full(len(self.alt), 40.0),
                "Btheta": np.full(len(self.alt), 50.0),
                "Bphi": np.full(len(self.alt), 60.0),
                "inclination": np.full(len(self.alt), 70.0),
                "declination": np.full(len(self.alt), 80.0),
                "total_field": np.full(len(self.alt), 90.0),
                "horizontal_intensity": np.full(len(self.alt), 100.0),
            }

        with mock.patch("pyionoseis.model.PPIGRF_AVAILABLE", True):
            with mock.patch(
                "pyionoseis.model._magfield_profile_worker",
                side_effect=_worker_side_effect,
            ):
                model.assign_magnetic_field(max_workers=1)

        self.assertIn("Be", model.grid.data_vars)
        self.assertIn("inclination", model.grid.data_vars)

        self.assertTrue(np.isnan(model.grid["Be"].values[0, 0, :]).all())
        self.assertTrue(np.isnan(model.grid["declination"].values[0, 0, :]).all())
        self.assertTrue(np.isfinite(model.grid["Be"].values[0, 1, :]).all())
        self.assertTrue(np.isfinite(model.grid["total_field"].values[1, 1, :]).all())

    def test_assign_magnetic_field_raises_when_ppigrf_unavailable(self):
        """Magnetic assignment fails fast with ImportError when ppigrf is missing."""
        model = self._new_model_with_grid()

        with mock.patch("pyionoseis.model.PPIGRF_AVAILABLE", False):
            with self.assertRaisesRegex(ImportError, "ppigrf is not available"):
                model.assign_magnetic_field()

    def test_assign_ionosphere_worker_importerror_stores_nan_profile(self):
        """Ionosphere worker ImportError is handled as NaN fallback in orchestrator."""
        model = self._new_model_with_grid()

        call_counter = {"n": 0}

        def _worker_side_effect(args):
            call_counter["n"] += 1
            if call_counter["n"] == 1:
                raise ImportError("iri2020 not installed")
            return np.full(len(self.alt), 2.34e11)

        with mock.patch(
            "pyionoseis.model._iono_profile_worker", side_effect=_worker_side_effect
        ):
            model.assign_ionosphere(max_workers=1)

        ne = model.grid["electron_density"].values
        self.assertTrue(np.isnan(ne[0, 0, :]).all())
        self.assertTrue(np.isfinite(ne[0, 1, :]).all())

    def test_assign_ionosphere_multiple_failures_map_to_expected_columns(self):
        """Multiple worker failures map to the expected (lat, lon) columns."""
        model = self._new_model_with_grid()

        call_counter = {"n": 0}
        failing_calls = {2, 3}

        def _worker_side_effect(args):
            call_counter["n"] += 1
            if call_counter["n"] in failing_calls:
                raise RuntimeError("synthetic worker failure")
            return np.full(len(self.alt), 3.21e11)

        with mock.patch(
            "pyionoseis.model._iono_profile_worker", side_effect=_worker_side_effect
        ):
            model.assign_ionosphere(max_workers=1)

        ne = model.grid["electron_density"].values
        self.assertTrue(np.isfinite(ne[0, 0, :]).all())
        self.assertTrue(np.isnan(ne[0, 1, :]).all())
        self.assertTrue(np.isnan(ne[1, 0, :]).all())
        self.assertTrue(np.isfinite(ne[1, 1, :]).all())

    def _seed_continuity_inputs(self, model):
        """Populate minimal required inputs for assign_continuity."""
        shape = (self.lat.size, self.lon.size, self.alt.size)
        model.grid["electron_density"] = (
            ("latitude", "longitude", "altitude"),
            np.full(shape, 1.0e11),
        )
        model.grid["kr"] = (("latitude", "longitude", "altitude"), np.zeros(shape))
        model.grid["kt"] = (("latitude", "longitude", "altitude"), np.zeros(shape))
        model.grid["kp"] = (("latitude", "longitude", "altitude"), np.zeros(shape))
        model.grid["travel_time_s"] = (
            ("latitude", "longitude", "altitude"),
            np.full(shape, 10.0),
        )
        model.grid["infraga_amplitude"] = (
            ("latitude", "longitude", "altitude"),
            np.ones(shape),
        )
        model.raypaths = xr.Dataset(attrs={"raytrace_signature_hash": "ray-hash-a"})

    def _fake_solve_continuity(self, grid, t0_s, tmax_s, dt_s, **kwargs):
        """Return a tiny deterministic continuity dataset for cache tests."""
        time = np.arange(float(t0_s), float(tmax_s) + 0.5 * float(dt_s), float(dt_s))
        dshape = (
            time.size,
            grid.sizes["latitude"],
            grid.sizes["longitude"],
            grid.sizes["altitude"],
        )
        return xr.Dataset(
            {
                "dNe": (
                    ("time", "latitude", "longitude", "altitude"),
                    np.zeros(dshape),
                )
            },
            coords={
                "time": time,
                "latitude": grid.coords["latitude"].values,
                "longitude": grid.coords["longitude"].values,
                "altitude": grid.coords["altitude"].values,
            },
        )

    def _fake_to_netcdf(self, _ds, path, *args, **kwargs):
        """Create a placeholder file to emulate netCDF persistence in tests."""
        Path(path).write_bytes(b"stub-netcdf")

    def _fake_load_dataset(self, _path):
        """Return a deterministic continuity dataset for cache-hit loads."""
        return self._fake_solve_continuity(
            self._new_model_with_grid().grid,
            t0_s=0.0,
            tmax_s=10.0,
            dt_s=5.0,
        )

    def test_assign_continuity_cache_hit_skips_solver(self):
        """Continuity cache hit reuses saved output and avoids solver rerun."""
        model = self._new_model_with_grid()
        self._seed_continuity_inputs(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "xarray.Dataset.to_netcdf",
                autospec=True,
                side_effect=self._fake_to_netcdf,
            ):
                with mock.patch(
                    "pyionoseis.model.xr.load_dataset",
                    side_effect=self._fake_load_dataset,
                ):
                    with mock.patch(
                        "pyionoseis.model.continuity_tools.solve_continuity",
                        side_effect=self._fake_solve_continuity,
                    ) as mocked_solver:
                        first = model.assign_continuity(
                            t0_s=0.0,
                            tmax_s=10.0,
                            dt_s=5.0,
                            output_dir=tmpdir,
                            reuse_existing=True,
                            use_kdtree=False,
                        )
                        second = model.assign_continuity(
                            t0_s=0.0,
                            tmax_s=10.0,
                            dt_s=5.0,
                            output_dir=tmpdir,
                            reuse_existing=True,
                            use_kdtree=False,
                        )

        self.assertIn("dNe", first.data_vars)
        self.assertIn("dNe", second.data_vars)
        self.assertEqual(mocked_solver.call_count, 1)
        self.assertEqual(int(second.attrs["continuity_loaded_from_cache"]), 1)

    def test_assign_continuity_signature_change_recomputes(self):
        """Changing signature-relevant continuity inputs triggers recomputation."""
        model = self._new_model_with_grid()
        self._seed_continuity_inputs(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "xarray.Dataset.to_netcdf",
                autospec=True,
                side_effect=self._fake_to_netcdf,
            ):
                with mock.patch(
                    "pyionoseis.model.continuity_tools.solve_continuity",
                    side_effect=self._fake_solve_continuity,
                ) as mocked_solver:
                    model.assign_continuity(
                        t0_s=0.0,
                        tmax_s=10.0,
                        dt_s=5.0,
                        b=1.0,
                        output_dir=tmpdir,
                        reuse_existing=True,
                        use_kdtree=False,
                    )
                    second = model.assign_continuity(
                        t0_s=0.0,
                        tmax_s=10.0,
                        dt_s=5.0,
                        b=2.0,
                        output_dir=tmpdir,
                        reuse_existing=True,
                        use_kdtree=False,
                    )

        self.assertEqual(mocked_solver.call_count, 2)
        self.assertEqual(int(second.attrs["continuity_loaded_from_cache"]), 0)

    def test_assign_continuity_force_recompute_bypasses_cache(self):
        """force_recompute=True reruns continuity solver despite cache availability."""
        model = self._new_model_with_grid()
        self._seed_continuity_inputs(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "xarray.Dataset.to_netcdf",
                autospec=True,
                side_effect=self._fake_to_netcdf,
            ):
                with mock.patch(
                    "pyionoseis.model.continuity_tools.solve_continuity",
                    side_effect=self._fake_solve_continuity,
                ) as mocked_solver:
                    model.assign_continuity(
                        t0_s=0.0,
                        tmax_s=10.0,
                        dt_s=5.0,
                        output_dir=tmpdir,
                        reuse_existing=True,
                        use_kdtree=False,
                    )
                    second = model.assign_continuity(
                        t0_s=0.0,
                        tmax_s=10.0,
                        dt_s=5.0,
                        output_dir=tmpdir,
                        reuse_existing=True,
                        force_recompute=True,
                        use_kdtree=False,
                    )

        self.assertEqual(mocked_solver.call_count, 2)
        self.assertEqual(int(second.attrs["continuity_loaded_from_cache"]), 0)

    def test_assign_continuity_ray_signature_change_recomputes(self):
        """Ray-signature changes invalidate continuity cache and recompute output."""
        model = self._new_model_with_grid()
        self._seed_continuity_inputs(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "xarray.Dataset.to_netcdf",
                autospec=True,
                side_effect=self._fake_to_netcdf,
            ):
                with mock.patch(
                    "pyionoseis.model.continuity_tools.solve_continuity",
                    side_effect=self._fake_solve_continuity,
                ) as mocked_solver:
                    model.assign_continuity(
                        t0_s=0.0,
                        tmax_s=10.0,
                        dt_s=5.0,
                        output_dir=tmpdir,
                        reuse_existing=True,
                        use_kdtree=False,
                    )

                    model.raypaths.attrs["raytrace_signature_hash"] = "ray-hash-b"
                    second = model.assign_continuity(
                        t0_s=0.0,
                        tmax_s=10.0,
                        dt_s=5.0,
                        output_dir=tmpdir,
                        reuse_existing=True,
                        use_kdtree=False,
                    )

        self.assertEqual(mocked_solver.call_count, 2)
        self.assertEqual(int(second.attrs["continuity_loaded_from_cache"]), 0)

    def test_assign_continuity_missing_then_present_ray_signature_recomputes(self):
        """Continuity cache invalidates when ray signature hash appears later."""
        model = self._new_model_with_grid()
        self._seed_continuity_inputs(model)
        model.raypaths.attrs.pop("raytrace_signature_hash", None)

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "xarray.Dataset.to_netcdf",
                autospec=True,
                side_effect=self._fake_to_netcdf,
            ):
                with mock.patch(
                    "pyionoseis.model.continuity_tools.solve_continuity",
                    side_effect=self._fake_solve_continuity,
                ) as mocked_solver:
                    model.assign_continuity(
                        t0_s=0.0,
                        tmax_s=10.0,
                        dt_s=5.0,
                        output_dir=tmpdir,
                        reuse_existing=True,
                        use_kdtree=False,
                    )

                    model.raypaths.attrs["raytrace_signature_hash"] = "ray-hash-late"
                    second = model.assign_continuity(
                        t0_s=0.0,
                        tmax_s=10.0,
                        dt_s=5.0,
                        output_dir=tmpdir,
                        reuse_existing=True,
                        use_kdtree=False,
                    )

        self.assertEqual(mocked_solver.call_count, 2)
        self.assertEqual(int(second.attrs["continuity_loaded_from_cache"]), 0)

    def test_assign_continuity_maps_scalars_when_missing_on_grid(self):
        """Continuity maps travel time and amplitude when grid scalars are absent."""
        model = self._new_model_with_grid()

        shape = (self.lat.size, self.lon.size, self.alt.size)
        model.grid["electron_density"] = (
            ("latitude", "longitude", "altitude"),
            np.full(shape, 1.0e11),
        )
        model.grid["kr"] = (("latitude", "longitude", "altitude"), np.zeros(shape))
        model.grid["kt"] = (("latitude", "longitude", "altitude"), np.zeros(shape))
        model.grid["kp"] = (("latitude", "longitude", "altitude"), np.zeros(shape))

        model.raypaths = xr.Dataset(
            {
                "travel_time_s": ("ray_point", np.full(4, 10.0)),
                "transport_amplitude_db": ("ray_point", np.zeros(4)),
                "ray_lat_deg": ("ray_point", np.array([38.0, 38.0, 38.5, 38.5])),
                "ray_lon_deg": ("ray_point", np.array([142.0, 142.5, 142.0, 142.5])),
                "ray_alt_km": ("ray_point", np.array([100.0, 110.0, 100.0, 110.0])),
            },
            attrs={"raytrace_signature_hash": "ray-hash-a"},
        )

        def _map_side_effect(grid, raypaths, ray_var, output_name=None, **kwargs):
            name = output_name or ray_var
            count_name = f"{name}_raypoint_count"
            values = np.full(
                (
                    grid.sizes["latitude"],
                    grid.sizes["longitude"],
                    grid.sizes["altitude"],
                ),
                1.0,
            )
            counts = np.full_like(values, 4.0)
            return xr.Dataset(
                {
                    name: (
                        ("latitude", "longitude", "altitude"),
                        values,
                    ),
                    count_name: (
                        ("latitude", "longitude", "altitude"),
                        counts,
                    ),
                },
                coords={
                    "latitude": grid.coords["latitude"].values,
                    "longitude": grid.coords["longitude"].values,
                    "altitude": grid.coords["altitude"].values,
                },
            )

        with mock.patch(
            "pyionoseis.model.wavevector_tools.map_ray_scalar_to_grid",
            side_effect=_map_side_effect,
        ) as mocked_map:
            with mock.patch(
                "pyionoseis.model.continuity_tools.solve_continuity",
                side_effect=self._fake_solve_continuity,
            ):
                continuity = model.assign_continuity(
                    t0_s=0.0,
                    tmax_s=10.0,
                    dt_s=5.0,
                    reuse_existing=False,
                    use_kdtree=False,
                )

        self.assertEqual(mocked_map.call_count, 2)
        self.assertIn("travel_time_s", model.grid.data_vars)
        self.assertIn("infraga_amplitude", model.grid.data_vars)
        self.assertIn("travel_time_s", continuity.data_vars)
        self.assertIn("infraga_amplitude", continuity.data_vars)

    def test_assign_continuity_coordinate_change_recomputes_cache(self):
        """Grid coordinate changes invalidate continuity cache signatures."""
        model = self._new_model_with_grid()
        self._seed_continuity_inputs(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "xarray.Dataset.to_netcdf",
                autospec=True,
                side_effect=self._fake_to_netcdf,
            ):
                with mock.patch(
                    "pyionoseis.model.continuity_tools.solve_continuity",
                    side_effect=self._fake_solve_continuity,
                ) as mocked_solver:
                    model.assign_continuity(
                        t0_s=0.0,
                        tmax_s=10.0,
                        dt_s=5.0,
                        output_dir=tmpdir,
                        reuse_existing=True,
                        use_kdtree=False,
                    )

                    shifted_lat = model.grid.coords["latitude"].values.copy()
                    shifted_lat[1] = shifted_lat[1] + 1e-9
                    model.grid = model.grid.assign_coords(latitude=shifted_lat)

                    second = model.assign_continuity(
                        t0_s=0.0,
                        tmax_s=10.0,
                        dt_s=5.0,
                        output_dir=tmpdir,
                        reuse_existing=True,
                        use_kdtree=False,
                    )

        self.assertEqual(mocked_solver.call_count, 2)
        self.assertEqual(int(second.attrs["continuity_loaded_from_cache"]), 0)


if __name__ == "__main__":
    unittest.main()
