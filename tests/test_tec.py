#!/usr/bin/env python

"""Tests for LOS TEC utilities."""

import unittest

import numpy as np
import xarray as xr

from pyionoseis import tec as tec_tools


@unittest.skipIf(not tec_tools.SCIPY_AVAILABLE, "scipy not available")
class TestTECTools(unittest.TestCase):
    """Tests for pyionoseis.tec utilities."""

    def setUp(self):
        """Create a small grid fixture."""
        self.lat = np.linspace(40.0, 41.0, 3)
        self.lon = np.linspace(-120.0, -119.0, 3)
        self.alt = np.linspace(100.0, 300.0, 3)
        self.time = np.array([0.0, 10.0])

        ne0 = np.full((self.lat.size, self.lon.size, self.alt.size), 1.0e11)
        self.grid = xr.Dataset(
            {
                "electron_density": (
                    ("latitude", "longitude", "altitude"),
                    ne0,
                )
            },
            coords={
                "latitude": self.lat,
                "longitude": self.lon,
                "altitude": self.alt,
            },
        )

        dNe = np.zeros((self.lat.size, self.lon.size, self.alt.size, self.time.size))
        self.dNe = xr.DataArray(
            dNe,
            coords={
                "latitude": self.lat,
                "longitude": self.lon,
                "altitude": self.alt,
                "time": self.time,
            },
            dims=("latitude", "longitude", "altitude", "time"),
        )

        self.receiver = {
            "latitude": np.array([40.5]),
            "longitude": np.array([-119.5]),
            "height_km": np.array([0.1]),
        }

        self.satellite = {
            "time": self.time,
            "latitude": np.array([40.5, 40.5]),
            "longitude": np.array([-119.5, -119.5]),
            "height_km": np.array([20200.0, 20200.0]),
        }

    def test_compute_los_tec_constant_density(self):
        """TEC totals match background when dNe is zero."""
        out = tec_tools.compute_los_tec(
            grid=self.grid,
            receiver_positions=self.receiver,
            satellite_positions=self.satellite,
            dNe=self.dNe,
        )

        self.assertIn("tec_total", out.data_vars)
        self.assertEqual(out.dims["time"], 2)
        np.testing.assert_allclose(out["tec_perturbation"].values, 0.0, atol=0.0)
        np.testing.assert_allclose(
            out["tec_total"].values,
            out["tec_background"].values,
            rtol=1e-6,
        )
        self.assertTrue(np.all(out["tec_total"].values > 0.0))

    def test_geometry_outputs(self):
        """Elevation, azimuth, and IPP outputs are finite for valid LOS."""
        out = tec_tools.compute_los_tec(
            grid=self.grid,
            receiver_positions=self.receiver,
            satellite_positions=self.satellite,
            dNe=self.dNe,
        )

        elev = out["elevation_deg"].values
        az = out["azimuth_deg"].values
        ipp_lat = out["ipp_latitude_deg"].values
        ipp_lon = out["ipp_longitude_deg"].values

        self.assertTrue(np.all((elev >= 0.0) & (elev <= 90.0)))
        self.assertTrue(np.all((az >= 0.0) & (az < 360.0)))
        self.assertTrue(np.all(np.isfinite(ipp_lat)))
        self.assertTrue(np.all(np.isfinite(ipp_lon)))
