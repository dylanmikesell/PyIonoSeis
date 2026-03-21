#!/usr/bin/env python

"""Tests for `pyionoseis` package."""


import unittest
import pyionoseis


class TestPyionoseis(unittest.TestCase):
    """Tests for `pyionoseis` package."""

    def test_package_version_is_defined(self):
        """Package exports a semantic version string."""
        self.assertTrue(hasattr(pyionoseis, "__version__"))
        self.assertRegex(pyionoseis.__version__, r"^\d+\.\d+\.\d+$")

    def test_package_author_metadata_is_defined(self):
        """Package exports author metadata for distribution."""
        self.assertTrue(hasattr(pyionoseis, "__author__"))
        self.assertTrue(hasattr(pyionoseis, "__email__"))
        self.assertIsInstance(pyionoseis.__author__, str)
        self.assertIsInstance(pyionoseis.__email__, str)
