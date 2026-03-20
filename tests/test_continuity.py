#!/usr/bin/env python

"""Tests for continuity time integration utilities."""

import unittest

import numpy as np
import xarray as xr

from pyionoseis import continuity as continuity_tools


class TestContinuityTools(unittest.TestCase):
    """Tests for pyionoseis.continuity utilities."""

    def setUp(self):
        """Create a small grid fixture."""
        self.lat = np.linspace(0.0, 2.0, 3)
        self.lon = np.linspace(0.0, 2.0, 3)
        self.alt = np.linspace(100.0, 120.0, 3)

    def test_spherical_divergence_matches_radial_only_when_tangential_zero(self):
        """3-D divergence matches 1-D when tangential components are zero."""
        shape = (self.lat.size, self.lon.size, self.alt.size)
        ar = np.ones(shape)
        at = np.zeros(shape)
        ap = np.zeros(shape)
        r_m = (continuity_tools.EARTH_RADIUS_KM + self.alt) * 1e3
        dr_m = float(np.mean(np.diff(self.alt))) * 1e3
        dtheta = float(np.mean(np.diff(self.lat)))
        dphi = float(np.mean(np.diff(self.lon)))

        div_1d = continuity_tools.spherical_divergence(
            ar,
            at,
            ap,
            r_m=r_m,
            lat_deg=self.lat,
            dr_m=dr_m,
            dtheta_deg=dtheta,
            dphi_deg=dphi,
            divergence_flag=1,
        )
        div_3d = continuity_tools.spherical_divergence(
            ar,
            at,
            ap,
            r_m=r_m,
            lat_deg=self.lat,
            dr_m=dr_m,
            dtheta_deg=dtheta,
            dphi_deg=dphi,
            divergence_flag=3,
        )

        np.testing.assert_allclose(div_1d, div_3d, rtol=1e-6, atol=1e-8)

    def test_solve_continuity_zero_wavevector_returns_zero_dne(self):
        """Zero wavevector yields zero electron-density perturbations."""
        grid = xr.Dataset(
            coords={
                "latitude": self.lat,
                "longitude": self.lon,
                "altitude": self.alt,
            }
        )
        shape = (self.lat.size, self.lon.size, self.alt.size)
        grid["electron_density"] = (
            ("latitude", "longitude", "altitude"),
            np.full(shape, 1.0e11),
        )
        grid["kr"] = (("latitude", "longitude", "altitude"), np.zeros(shape))
        grid["kt"] = (("latitude", "longitude", "altitude"), np.zeros(shape))
        grid["kp"] = (("latitude", "longitude", "altitude"), np.zeros(shape))
        grid["travel_time_s"] = (
            ("latitude", "longitude", "altitude"),
            np.full(shape, 10.0),
        )
        grid["infraga_amplitude"] = (
            ("latitude", "longitude", "altitude"),
            np.ones(shape),
        )
        grid["inclination"] = (("latitude", "longitude", "altitude"), np.zeros(shape))
        grid["declination"] = (("latitude", "longitude", "altitude"), np.zeros(shape))

        out = continuity_tools.solve_continuity(
            grid=grid,
            t0_s=0.0,
            tmax_s=10.0,
            dt_s=5.0,
            divergence_flag=3,
            geomag_flag=True,
            amplitude_var="infraga_amplitude",
            travel_time_var="travel_time_s",
            store_neutral_velocity=True,
        )

        self.assertIn("dNe", out.data_vars)
        self.assertIn("neutral_velocity_r", out.data_vars)
        self.assertEqual(out.dims["time"], 3)
        np.testing.assert_allclose(out["dNe"].values, 0.0, atol=0.0)
        np.testing.assert_allclose(out["neutral_velocity_r"].values, 0.0, atol=0.0)
        np.testing.assert_allclose(out["neutral_velocity_t"].values, 0.0, atol=0.0)
        np.testing.assert_allclose(out["neutral_velocity_p"].values, 0.0, atol=0.0)
