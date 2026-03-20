#!/usr/bin/env python

"""Tests for wavevector approximation utilities."""

import unittest

import numpy as np
import xarray as xr

from pyionoseis import wavevector as wavevector_tools


class TestWavevectorTools(unittest.TestCase):
    """Tests for pyionoseis.wavevector utilities."""

    def test_compute_ray_wavevectors_straight_north(self):
        """Ray tangents follow a northward great-circle direction."""
        lat = np.linspace(0.0, 0.1, 5)
        lon = np.zeros_like(lat)
        alt = np.full_like(lat, 100.0)
        time = np.linspace(0.0, 40.0, lat.size)

        raypaths = xr.Dataset(
            {
                "ray_lat_deg": ("ray_point", lat),
                "ray_lon_deg": ("ray_point", lon),
                "ray_alt_km": ("ray_point", alt),
                "travel_time_s": ("ray_point", time),
            }
        )

        result = wavevector_tools.compute_ray_wavevectors(raypaths, smoothing_radius_km=0.0)

        k_r = result["ray_k_r"].values
        k_t = result["ray_k_t"].values
        k_p = result["ray_k_p"].values

        inner = slice(1, -1)
        self.assertTrue(np.all(np.isfinite(k_t[inner])))
        self.assertTrue(np.allclose(k_r[inner], 0.0, atol=1e-2))
        self.assertTrue(np.allclose(k_p[inner], 0.0, atol=1e-2))
        self.assertTrue(np.allclose(k_t[inner], 1.0, atol=5e-2))

    def test_map_wavevector_to_grid_single_cell(self):
        """Grid mapping produces expected direction cosines at a nearby cell."""
        lat = np.linspace(0.0, 0.1, 5)
        lon = np.zeros_like(lat)
        alt = np.full_like(lat, 100.0)
        time = np.linspace(0.0, 40.0, lat.size)

        raypaths = xr.Dataset(
            {
                "ray_lat_deg": ("ray_point", lat),
                "ray_lon_deg": ("ray_point", lon),
                "ray_alt_km": ("ray_point", alt),
                "travel_time_s": ("ray_point", time),
            }
        )
        ray_wavevectors = wavevector_tools.compute_ray_wavevectors(
            raypaths, smoothing_radius_km=0.0
        )

        grid = xr.Dataset(
            coords={"latitude": [0.05], "longitude": [0.0], "altitude": [100.0]}
        )
        mapped = wavevector_tools.map_wavevector_to_grid(
            grid,
            ray_wavevectors,
            interpolation_radius_km=200.0,
            mapping_mode="weighted",
            min_points=1,
            chunk_size=8,
        )

        self.assertTrue(np.allclose(mapped["kr"].values, 0.0, atol=1e-2))
        self.assertTrue(np.allclose(mapped["kp"].values, 0.0, atol=1e-2))
        self.assertTrue(np.allclose(mapped["kt"].values, 1.0, atol=5e-2))
        self.assertEqual(mapped["wavevector_raypoint_count"].values.item(), lat.size)

    def test_map_wavevector_to_grid_nearest_neighbor(self):
        """Nearest-neighbor mapping picks the closest ray tangent."""
        lat = np.linspace(0.0, 0.1, 5)
        lon = np.zeros_like(lat)
        alt = np.full_like(lat, 100.0)
        time = np.linspace(0.0, 40.0, lat.size)

        raypaths = xr.Dataset(
            {
                "ray_lat_deg": ("ray_point", lat),
                "ray_lon_deg": ("ray_point", lon),
                "ray_alt_km": ("ray_point", alt),
                "travel_time_s": ("ray_point", time),
            }
        )
        ray_wavevectors = wavevector_tools.compute_ray_wavevectors(
            raypaths, smoothing_radius_km=0.0
        )

        grid = xr.Dataset(
            coords={"latitude": [0.05], "longitude": [0.0], "altitude": [100.0]}
        )
        mapped = wavevector_tools.map_wavevector_to_grid(
            grid,
            ray_wavevectors,
            interpolation_radius_km=200.0,
            mapping_mode="nearest",
            use_kdtree=False,
            min_points=1,
            chunk_size=8,
        )

        self.assertTrue(np.allclose(mapped["kr"].values, 0.0, atol=1e-2))
        self.assertTrue(np.allclose(mapped["kp"].values, 0.0, atol=1e-2))
        self.assertTrue(np.allclose(mapped["kt"].values, 1.0, atol=5e-2))
        self.assertEqual(mapped["wavevector_raypoint_count"].values.item(), 1)

    def test_map_wavevector_altitude_window_blocks_assignment(self):
        """Altitude windowing rejects rays outside the vertical window."""
        raypaths = xr.Dataset(
            {
                "ray_lat_deg": ("ray_point", np.array([0.0, 0.1])),
                "ray_lon_deg": ("ray_point", np.array([0.0, 0.0])),
                "ray_alt_km": ("ray_point", np.array([200.0, 210.0])),
                "travel_time_s": ("ray_point", np.array([0.0, 10.0])),
            }
        )
        ray_wavevectors = wavevector_tools.compute_ray_wavevectors(
            raypaths, smoothing_radius_km=0.0
        )

        grid = xr.Dataset(
            coords={"latitude": [0.05], "longitude": [0.0], "altitude": [100.0]}
        )
        mapped = wavevector_tools.map_wavevector_to_grid(
            grid,
            ray_wavevectors,
            interpolation_radius_km=200.0,
            altitude_window_km=50.0,
            mapping_mode="nearest",
            use_kdtree=False,
            min_points=1,
            chunk_size=8,
        )

        self.assertTrue(np.isnan(mapped["kr"].values).all())
        self.assertEqual(mapped["wavevector_raypoint_count"].values.item(), 0)

    def test_map_ray_scalar_to_grid_nearest_value(self):
        """Scalar ray mapping assigns the nearest ray value."""
        lat = np.linspace(0.0, 0.1, 5)
        lon = np.zeros_like(lat)
        alt = np.full_like(lat, 100.0)
        time = np.linspace(0.0, 40.0, lat.size)

        raypaths = xr.Dataset(
            {
                "ray_lat_deg": ("ray_point", lat),
                "ray_lon_deg": ("ray_point", lon),
                "ray_alt_km": ("ray_point", alt),
                "travel_time_s": ("ray_point", time),
            }
        )

        grid = xr.Dataset(
            coords={"latitude": [0.05], "longitude": [0.0], "altitude": [100.0]}
        )
        mapped = wavevector_tools.map_ray_scalar_to_grid(
            grid,
            raypaths,
            ray_var="travel_time_s",
            output_name="travel_time_s",
            interpolation_radius_km=200.0,
            mapping_mode="nearest",
            use_kdtree=False,
            min_points=1,
            chunk_size=8,
        )

        self.assertAlmostEqual(mapped["travel_time_s"].values.item(), 20.0, places=6)
        self.assertEqual(mapped["travel_time_s_raypoint_count"].values.item(), 1)
