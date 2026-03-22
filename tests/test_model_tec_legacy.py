#!/usr/bin/env python

"""Integration tests for legacy TEC input dispatch in Model3D."""

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import xarray as xr

from pyionoseis.model import Model3D


class _DummySource:
    def __init__(self, time_utc):
        self._time_utc = time_utc

    def get_time(self):
        return self._time_utc


class TestModelLegacyTECDispatch(unittest.TestCase):
    """Tests legacy receiver/orbit routing in Model3D.compute_los_tec."""

    @staticmethod
    def _minimal_model():
        model = Model3D()
        model.grid = xr.Dataset(
            coords={"latitude": [0.0], "longitude": [0.0], "altitude": [100.0]}
        )
        model.source = _DummySource(datetime(2011, 1, 1, 0, 1, 40, tzinfo=timezone.utc))
        model.tec_output_dt_s = 1.0
        return model

    def test_compute_los_tec_uses_listesta_and_satpos_root_mapping(self):
        """Model3D dispatches to legacy listesta and SATPOS loaders from config keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            listesta = tmp / "listesta_all.txt"
            sat_dir = tmp / "SATPOS" / "16591"
            sat_dir.mkdir(parents=True, exist_ok=True)
            satpos = sat_dir / "sat17_16591.pos"
            sat_map = tmp / "sat_number_mapping.csv"

            listesta.write_text(
                "AGRD 3592016.2185 3353464.5240 4054981.8590 39.72 43.03\n",
                encoding="utf-8",
            )
            satpos.write_text(
                "100 20000000 0 0 0 0 0\n"
                "101 20100000 0 0 0 0 0\n"
                "102 20200000 0 0 0 0 0\n",
                encoding="utf-8",
            )
            sat_map.write_text("sat_number,sat_id\n17,E11\n", encoding="utf-8")

            model = Model3D()
            model.grid = xr.Dataset(coords={"latitude": [0.0], "longitude": [0.0], "altitude": [100.0]})
            model.source = _DummySource(datetime(2011, 1, 1, 0, 1, 40, tzinfo=timezone.utc))

            model.tec_receiver_format = "listesta"
            model.tec_receiver_listesta = str(listesta)
            model.tec_receiver_code = "AGRD"

            model.tec_orbit_format = "pos"
            model.tec_orbit_pos = None
            model.tec_satpos_root = str(tmp / "SATPOS")
            model.tec_satpos_date = "16591"
            model.tec_sat_number = 17
            model.tec_sat_mapping_file = str(sat_map)
            model.tec_start_offset_s = 0.0
            model.tec_output_dt_s = 1.0

            with mock.patch("pyionoseis.model.tec_tools.compute_los_tec") as mocked_compute:
                mocked_compute.return_value = xr.Dataset()
                model.compute_los_tec()

                self.assertTrue(mocked_compute.called)
                kwargs = mocked_compute.call_args.kwargs
                self.assertEqual(kwargs["receiver_positions"]["code"], "AGRD")
                self.assertEqual(kwargs["satellite_positions"]["constellation"], "GAL")
                self.assertEqual(kwargs["satellite_positions"]["prn"], 11)

    def test_compute_los_tec_uses_direct_orbit_pos_path(self):
        """Model3D dispatches to direct orbit_pos without SATPOS folder lookup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            listesta = tmp / "listesta_all.txt"
            satpos = tmp / "sat17_16591.pos"

            listesta.write_text(
                "AGRD 3592016.2185 3353464.5240 4054981.8590 39.72 43.03\n",
                encoding="utf-8",
            )
            satpos.write_text(
                "100 20000000 0 0 0 0 0\n"
                "101 20100000 0 0 0 0 0\n"
                "102 20200000 0 0 0 0 0\n",
                encoding="utf-8",
            )

            model = Model3D()
            model.grid = xr.Dataset(
                coords={"latitude": [0.0], "longitude": [0.0], "altitude": [100.0]}
            )
            model.source = _DummySource(
                datetime(2011, 1, 1, 0, 1, 40, tzinfo=timezone.utc)
            )

            model.tec_receiver_format = "listesta"
            model.tec_receiver_listesta = str(listesta)
            model.tec_receiver_code = "AGRD"

            model.tec_orbit_format = "pos"
            model.tec_orbit_pos = str(satpos)
            model.tec_sat_id = "G17"
            model.tec_satpos_root = None
            model.tec_satpos_date = None
            model.tec_sat_number = None
            model.tec_sat_mapping_file = None
            model.tec_start_offset_s = 0.0
            model.tec_output_dt_s = 1.0

            with mock.patch("pyionoseis.model.tec_io.build_satpos_file_path") as mocked_builder:
                with mock.patch("pyionoseis.model.tec_tools.compute_los_tec") as mocked_compute:
                    mocked_compute.return_value = xr.Dataset()
                    model.compute_los_tec()

                    self.assertTrue(mocked_compute.called)
                    mocked_builder.assert_not_called()
                    kwargs = mocked_compute.call_args.kwargs
                    self.assertEqual(kwargs["receiver_positions"]["code"], "AGRD")
                    self.assertEqual(kwargs["satellite_positions"]["constellation"], "GPS")
                    self.assertEqual(kwargs["satellite_positions"]["prn"], 17)

    def test_compute_los_tec_direct_orbit_pos_uses_mapping_without_sat_id(self):
        """Direct orbit_pos resolves satellite identity from sat_number mapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            listesta = tmp / "listesta_all.txt"
            satpos = tmp / "sat17_16591.pos"
            sat_map = tmp / "sat_number_mapping.csv"

            listesta.write_text(
                "AGRD 3592016.2185 3353464.5240 4054981.8590 39.72 43.03\n",
                encoding="utf-8",
            )
            satpos.write_text(
                "100 20000000 0 0 0 0 0\n"
                "101 20100000 0 0 0 0 0\n"
                "102 20200000 0 0 0 0 0\n",
                encoding="utf-8",
            )
            sat_map.write_text("sat_number,sat_id\n17,R17\n", encoding="utf-8")

            model = Model3D()
            model.grid = xr.Dataset(
                coords={"latitude": [0.0], "longitude": [0.0], "altitude": [100.0]}
            )
            model.source = _DummySource(
                datetime(2011, 1, 1, 0, 1, 40, tzinfo=timezone.utc)
            )

            model.tec_receiver_format = "listesta"
            model.tec_receiver_listesta = str(listesta)
            model.tec_receiver_code = "AGRD"

            model.tec_orbit_format = "pos"
            model.tec_orbit_pos = str(satpos)
            model.tec_sat_id = None
            model.tec_constellation = None
            model.tec_prn = None
            model.tec_sat_number = 17
            model.tec_sat_mapping_file = str(sat_map)
            model.tec_start_offset_s = 0.0
            model.tec_output_dt_s = 1.0

            with mock.patch("pyionoseis.model.tec_tools.compute_los_tec") as mocked_compute:
                mocked_compute.return_value = xr.Dataset()
                model.compute_los_tec()

                self.assertTrue(mocked_compute.called)
                kwargs = mocked_compute.call_args.kwargs
                self.assertEqual(kwargs["receiver_positions"]["code"], "AGRD")
                self.assertEqual(kwargs["satellite_positions"]["constellation"], "GLO")
                self.assertEqual(kwargs["satellite_positions"]["prn"], 17)

    def test_compute_los_tec_direct_orbit_pos_missing_mapping_raises(self):
        """Direct orbit_pos raises when sat_number is not present in mapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            listesta = tmp / "listesta_all.txt"
            satpos = tmp / "sat17_16591.pos"
            sat_map = tmp / "sat_number_mapping.csv"

            listesta.write_text(
                "AGRD 3592016.2185 3353464.5240 4054981.8590 39.72 43.03\n",
                encoding="utf-8",
            )
            satpos.write_text(
                "100 20000000 0 0 0 0 0\n"
                "101 20100000 0 0 0 0 0\n"
                "102 20200000 0 0 0 0 0\n",
                encoding="utf-8",
            )
            sat_map.write_text("sat_number,sat_id\n21,G21\n", encoding="utf-8")

            model = Model3D()
            model.grid = xr.Dataset(
                coords={"latitude": [0.0], "longitude": [0.0], "altitude": [100.0]}
            )
            model.source = _DummySource(
                datetime(2011, 1, 1, 0, 1, 40, tzinfo=timezone.utc)
            )

            model.tec_receiver_format = "listesta"
            model.tec_receiver_listesta = str(listesta)
            model.tec_receiver_code = "AGRD"

            model.tec_orbit_format = "pos"
            model.tec_orbit_pos = str(satpos)
            model.tec_sat_id = None
            model.tec_constellation = None
            model.tec_prn = None
            model.tec_sat_number = 17
            model.tec_sat_mapping_file = str(sat_map)
            model.tec_start_offset_s = 0.0
            model.tec_output_dt_s = 1.0

            with self.assertRaisesRegex(ValueError, "No sat_id mapping found"):
                model.compute_los_tec()

    def test_compute_los_tec_prefers_csv_when_receiver_format_unspecified(self):
        """When receiver_format is None and both inputs exist, CSV branch is chosen."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "receivers.csv"
            listesta = tmp / "listesta_all.txt"
            h5_path = tmp / "sat_orbits.h5"

            csv_path.write_text("code,lat,lon,height_km\nAGRD,39.72,43.03,0.0\n", encoding="utf-8")
            listesta.write_text(
                "AGRD 3592016.2185 3353464.5240 4054981.8590 39.72 43.03\n",
                encoding="utf-8",
            )
            h5_path.write_text("placeholder", encoding="utf-8")

            model = self._minimal_model()
            model.tec_receiver_format = None
            model.tec_receiver_csv = str(csv_path)
            model.tec_receiver_listesta = str(listesta)
            model.tec_receiver_code = "AGRD"
            model.tec_orbit_format = "h5"
            model.tec_orbit_h5 = str(h5_path)
            model.tec_sat_id = "G17"

            with mock.patch("pyionoseis.model.tec_io.load_receiver_positions_csv") as mocked_csv:
                with mock.patch("pyionoseis.model.tec_io.load_receiver_positions_listesta") as mocked_listesta:
                    with mock.patch("pyionoseis.model.tec_io.load_orbits_hdf5") as mocked_h5:
                        with mock.patch("pyionoseis.model.tec_tools.compute_los_tec") as mocked_compute:
                            mocked_csv.return_value = {"code": "AGRD", "latitude": 39.72, "longitude": 43.03, "height_km": 0.0}
                            mocked_h5.return_value = {"x_km": [0.0], "y_km": [0.0], "z_km": [20200.0], "time": [0.0]}
                            mocked_compute.return_value = xr.Dataset()

                            model.compute_los_tec()

                            mocked_csv.assert_called_once()
                            mocked_listesta.assert_not_called()

    def test_compute_los_tec_prefers_h5_when_orbit_format_unspecified(self):
        """When orbit_format is None and h5+pos are present, H5 branch is chosen."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "receivers.csv"
            h5_path = tmp / "sat_orbits.h5"
            pos_path = tmp / "sat17_16591.pos"

            csv_path.write_text("code,lat,lon,height_km\nAGRD,39.72,43.03,0.0\n", encoding="utf-8")
            h5_path.write_text("placeholder", encoding="utf-8")
            pos_path.write_text("100 20000000 0 0 0 0 0\n", encoding="utf-8")

            model = self._minimal_model()
            model.tec_receiver_format = "csv"
            model.tec_receiver_csv = str(csv_path)
            model.tec_receiver_code = "AGRD"

            model.tec_orbit_format = None
            model.tec_orbit_h5 = str(h5_path)
            model.tec_orbit_pos = str(pos_path)
            model.tec_sat_id = "G17"

            with mock.patch("pyionoseis.model.tec_io.load_receiver_positions_csv") as mocked_csv:
                with mock.patch("pyionoseis.model.tec_io.load_orbits_hdf5") as mocked_h5:
                    with mock.patch("pyionoseis.model.tec_io.load_orbits_pos") as mocked_pos:
                        with mock.patch("pyionoseis.model.tec_tools.compute_los_tec") as mocked_compute:
                            mocked_csv.return_value = {"code": "AGRD", "latitude": 39.72, "longitude": 43.03, "height_km": 0.0}
                            mocked_h5.return_value = {"x_km": [0.0], "y_km": [0.0], "z_km": [20200.0], "time": [0.0]}
                            mocked_compute.return_value = xr.Dataset()

                            model.compute_los_tec()

                            mocked_h5.assert_called_once()
                            mocked_pos.assert_not_called()

    def test_compute_los_tec_builds_satellite_id_from_constellation_prn(self):
        """LOS metadata satellite_id is synthesized from constellation and prn."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "receivers.csv"
            h5_path = tmp / "sat_orbits.h5"

            csv_path.write_text(
                "code,lat,lon,height_km\nAGRD,39.72,43.03,0.0\n",
                encoding="utf-8",
            )
            h5_path.write_text("placeholder", encoding="utf-8")

            model = self._minimal_model()
            model.tec_receiver_format = "csv"
            model.tec_receiver_csv = str(csv_path)
            model.tec_receiver_code = "AGRD"
            model.tec_orbit_format = "h5"
            model.tec_orbit_h5 = str(h5_path)
            model.tec_sat_id = None
            model.tec_constellation = "GPS"
            model.tec_prn = 17

            with mock.patch("pyionoseis.model.tec_io.load_receiver_positions_csv") as mocked_csv:
                with mock.patch("pyionoseis.model.tec_io.load_orbits_hdf5") as mocked_h5:
                    with mock.patch("pyionoseis.model.tec_tools.compute_los_tec") as mocked_compute:
                        mocked_csv.return_value = {
                            "code": "AGRD",
                            "latitude": 39.72,
                            "longitude": 43.03,
                            "height_km": 0.0,
                        }
                        mocked_h5.return_value = {
                            "x_km": [0.0],
                            "y_km": [0.0],
                            "z_km": [20200.0],
                            "time": [0.0],
                        }
                        mocked_compute.return_value = xr.Dataset()

                        model.compute_los_tec()

                        kwargs = mocked_compute.call_args.kwargs
                        self.assertEqual(kwargs["satellite_id"], "GPS17")

    def test_compute_los_tec_uses_continuity_dne_when_not_provided(self):
        """compute_los_tec falls back to continuity dNe when explicit dNe is absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "receivers.csv"
            h5_path = tmp / "sat_orbits.h5"

            csv_path.write_text(
                "code,lat,lon,height_km\nAGRD,39.72,43.03,0.0\n",
                encoding="utf-8",
            )
            h5_path.write_text("placeholder", encoding="utf-8")

            model = self._minimal_model()
            model.tec_receiver_format = "csv"
            model.tec_receiver_csv = str(csv_path)
            model.tec_receiver_code = "AGRD"
            model.tec_orbit_format = "h5"
            model.tec_orbit_h5 = str(h5_path)
            model.tec_sat_id = "G17"
            model.continuity = xr.Dataset(
                data_vars={
                    "dNe": (
                        ["time", "latitude", "longitude", "altitude"],
                        [[[[0.0]]]],
                    )
                },
                coords={
                    "time": [0.0],
                    "latitude": [0.0],
                    "longitude": [0.0],
                    "altitude": [100.0],
                },
            )
            expected_dne = model.continuity["dNe"]

            with mock.patch("pyionoseis.model.tec_io.load_receiver_positions_csv") as mocked_csv:
                with mock.patch("pyionoseis.model.tec_io.load_orbits_hdf5") as mocked_h5:
                    with mock.patch("pyionoseis.model.tec_tools.compute_los_tec") as mocked_compute:
                        mocked_csv.return_value = {
                            "code": "AGRD",
                            "latitude": 39.72,
                            "longitude": 43.03,
                            "height_km": 0.0,
                        }
                        mocked_h5.return_value = {
                            "x_km": [0.0],
                            "y_km": [0.0],
                            "z_km": [20200.0],
                            "time": [0.0],
                        }
                        mocked_compute.return_value = xr.Dataset()

                        model.compute_los_tec()

                        kwargs = mocked_compute.call_args.kwargs
                        xr.testing.assert_identical(kwargs["dNe"], expected_dne)


if __name__ == "__main__":
    unittest.main()
