---
applyTo: "tests/**/*.py"
description: "Testing conventions for PyIonoSeis unittest test files"
name: "Testing Guidelines"
---
# Testing Guidelines

These instructions apply to all files in `tests/`. They complement the universal testing rules in `.github/copilot-instructions.md`.

## Test Class Structure

Mirror the name of the class or module under test:

```python
class TestAtmosphere1D(unittest.TestCase):
    """Tests for pyionoseis.atmosphere.Atmosphere1D."""

    def setUp(self):
        """Create small, fast fixtures — 5 altitude points is enough."""
        self.alt_km = np.linspace(0, 100, 5)
        self.lat, self.lon = 65.1, -147.5
        self.time = datetime(2015, 11, 13, 10, 0, 0)

    def test_<what>_<condition>(self):
        """Docstring: one sentence describing what this test asserts."""
```

- Test method names follow `test_<what>_<condition>` (e.g., `test_returns_xr_dataset_with_altitude_coord`).
- Keep fixtures minimal: use 3–5 altitude points, not full 500-point arrays.

## Mocking Optional Dependencies

Patch `*_AVAILABLE` flags and modules at the package level, not at the import:

```python
# Patch the flag in the module that reads it
@patch("pyionoseis.igrf.PPIGRF_AVAILABLE", False)
def test_igrf_unavailable_raises(self):
    ...

# Patch the module object itself when calls must be intercepted
@patch("pyionoseis.ionosphere.iri2020")
def test_iri2020_called_with_correct_args(self, mock_iri):
    mock_iri.pro.return_value = ...
```

## Mocking subprocess / infraGA

Always mock subprocess calls; never invoke real infraGA binaries in tests:

```python
@patch("pyionoseis.infraga.subprocess.run")
def test_run_sph_trace_passes_profile(self, mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr=""
    )
    ...
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    self.assertIn("infraga-sph", cmd[0])
```

## xarray Assertions

When testing functions that return `xr.Dataset`:

```python
# Check structure
self.assertIn("electron_density", ds.data_vars)
self.assertIn("altitude", ds.coords)

# Check dimension order
self.assertEqual(list(ds.dims), ["altitude"])

# Check values with numpy
np.testing.assert_allclose(ds["electron_density"].values, expected, rtol=1e-5)
```

Do not compare `xr.Dataset` objects with `==`; always inspect `.data_vars`, `.coords`, and `.values`.

## What Not to Test

- Standard library or third-party behavior (scipy, numpy internals).
- TOML loading logic beyond the format validation already in `EarthquakeSource`.
- `plot()` methods — keep these out of the test suite unless testing for `AttributeError` guards.
