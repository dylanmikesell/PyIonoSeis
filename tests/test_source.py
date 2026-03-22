#!/usr/bin/env python

"""Tests for pyionoseis.source.EarthquakeSource."""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from pyionoseis.source import EarthquakeSource


class TestEarthquakeSource(unittest.TestCase):
    """Tests for pyionoseis.source.EarthquakeSource."""

    def _write_event_toml(self, time_value: str) -> Path:
        """Create a temporary TOML file for a single event timestamp."""
        tmp = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
        with tmp:
            tmp.write("[event]\n")
            tmp.write(f'time = "{time_value}"\n')
            tmp.write("latitude = 37.0\n")
            tmp.write("longitude = -122.0\n")
            tmp.write("depth = 10.0\n")
        return Path(tmp.name)

    def test_load_from_toml_parses_whole_second_timestamp(self):
        """Load event time when TOML timestamp is whole-second UTC format."""
        toml_path = self._write_event_toml("2011-10-23T10:21:22Z")

        try:
            source = EarthquakeSource(toml_path)
        finally:
            toml_path.unlink(missing_ok=True)

        self.assertEqual(source.get_time(), datetime(2011, 10, 23, 10, 21, 22))

    def test_load_from_toml_parses_fractional_second_timestamp(self):
        """Load event time when TOML timestamp includes fractional seconds."""
        toml_path = self._write_event_toml("2011-10-23T10:21:22.010Z")

        try:
            source = EarthquakeSource(toml_path)
        finally:
            toml_path.unlink(missing_ok=True)

        self.assertEqual(
            source.get_time(),
            datetime(2011, 10, 23, 10, 21, 22, 10000),
        )

    def test_load_from_toml_raises_for_invalid_timestamp(self):
        """Reject event time strings that are not supported UTC formats."""
        toml_path = self._write_event_toml("2011/10/23 10:21:22")

        try:
            with self.assertRaises(ValueError):
                EarthquakeSource(toml_path)
        finally:
            toml_path.unlink(missing_ok=True)
