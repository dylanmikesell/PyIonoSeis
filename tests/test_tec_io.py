#!/usr/bin/env python

"""Tests for TEC input loaders."""

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pyionoseis import tec_io


@unittest.skipIf(not tec_io.PANDAS_AVAILABLE, "pandas not available")
class TestTECIO(unittest.TestCase):
    """Tests for pyionoseis.tec_io loaders."""

    def test_load_receiver_positions_listesta_recomputes_geodetic(self):
        """Legacy listesta loader reads ECEF meters and returns geodetic arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "listesta_all.txt"
            path.write_text("ABCD 6371000.0 0.0 0.0 0.0 0.0\n", encoding="utf-8")

            out = tec_io.load_receiver_positions_listesta(path, receiver_code="ABCD")

            self.assertEqual(out["code"], "ABCD")
            np.testing.assert_allclose(out["latitude"], [0.0], atol=1.0e-6)
            np.testing.assert_allclose(out["longitude"], [0.0], atol=1.0e-6)
            np.testing.assert_allclose(out["height_km"], [0.0], atol=1.0e-6)

    def test_load_orbits_pos_relative_time_and_resample(self):
        """Legacy SATPOS loader converts SOD to relative time and resamples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pos_path = Path(tmpdir) / "sat17_16591.pos"
            pos_path.write_text(
                "10 20000000 0 0 0 0 0\n"
                "11 21000000 0 0 0 0 0\n"
                "12 22000000 0 0 0 0 0\n",
                encoding="utf-8",
            )

            event_time = datetime(2011, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
            out = tec_io.load_orbits_pos(
                pos_path,
                event_time=event_time,
                sat_id="G17",
                output_dt_s=2.0,
            )

            np.testing.assert_allclose(out["time"], [0.0, 2.0])
            np.testing.assert_allclose(out["x_km"], [20000.0, 22000.0])
            np.testing.assert_allclose(out["y_km"], [0.0, 0.0])
            np.testing.assert_allclose(out["z_km"], [0.0, 0.0])
            self.assertEqual(out["constellation"], "GPS")
            self.assertEqual(out["prn"], 17)

    def test_load_orbits_pos_uses_sat_mapping_file(self):
        """Legacy SATPOS loader resolves sat_id from sat-number mapping file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pos_path = Path(tmpdir) / "sat17_16591.pos"
            map_path = Path(tmpdir) / "sat_map.csv"
            pos_path.write_text(
                "10 20000000 0 0 0 0 0\n"
                "11 20000000 0 0 0 0 0\n",
                encoding="utf-8",
            )
            map_path.write_text("sat_number,sat_id\n17,E11\n", encoding="utf-8")

            event_time = datetime(2011, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
            out = tec_io.load_orbits_pos(
                pos_path,
                event_time=event_time,
                sat_number=17,
                sat_mapping_file=map_path,
                output_dt_s=1.0,
            )

            self.assertEqual(out["constellation"], "GAL")
            self.assertEqual(out["prn"], 11)

    def test_build_satpos_file_path_formats_sat_number(self):
        """SATPOS path builder uses legacy satNN_date.pos naming."""
        path = tec_io.build_satpos_file_path("example/VAN/SATPOS", "16591", 7)
        self.assertEqual(
            str(path).replace("\\", "/"),
            "example/VAN/SATPOS/16591/sat07_16591.pos",
        )

    def test_load_receiver_positions_listesta_requires_six_columns(self):
        """Legacy listesta loader enforces expected 6-column schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "listesta_bad.txt"
            path.write_text("ABCD 1 2 3\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                tec_io.load_receiver_positions_listesta(path, receiver_code="ABCD")

    def test_load_orbits_pos_requires_seven_columns(self):
        """Legacy SATPOS loader enforces expected 7-column schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sat17_16591.pos"
            path.write_text("100 20000000 0 0\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                tec_io.load_orbits_pos(
                    path,
                    event_time=datetime(2011, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                    sat_id="G17",
                )

    def test_load_orbits_pos_requires_monotonic_seconds(self):
        """Legacy SATPOS loader rejects non-increasing seconds-of-day."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sat17_16591.pos"
            path.write_text(
                "101 20000000 0 0 0 0 0\n"
                "100 20010000 0 0 0 0 0\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                tec_io.load_orbits_pos(
                    path,
                    event_time=datetime(2011, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                    sat_id="G17",
                )

    def test_load_sat_number_mapping_rejects_duplicates(self):
        """SAT mapping file parser rejects duplicate sat_number rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            map_path = Path(tmpdir) / "sat_map.csv"
            map_path.write_text(
                "sat_number,sat_id\n"
                "17,G17\n"
                "17,E11\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                tec_io.load_sat_number_mapping(map_path)


if __name__ == "__main__":
    unittest.main()
