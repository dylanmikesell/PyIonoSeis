# PyIonoSeis

A Python scientific library for modeling and analysis of **coseismic ionospheric disturbances (CIDs)** — ionospheric perturbations triggered by earthquakes. Targets geophysics researchers and students. Version 0.0.1, early development.

## Tech Stack

- **Python ≥3.10** — type hints encouraged
- **numpy** — numerical array computation
- **xarray** (`xr.Dataset` / `xr.DataArray`) — primary data container for all model outputs; always use named dimensions and coordinates
- **toml** — configuration files for events and model parameters
- **click** — CLI entry point (`pyionoseis.cli`)
- **matplotlib + cartopy** — plotting and geographic visualization
- **msise00** — MSISE-00 neutral atmosphere model
- **iri2020** — IRI2020 ionospheric electron density model (optional, graceful fallback)
- **ppigrf** — IGRF geomagnetic field model (optional, graceful fallback)
- **infraGA** — external spherical infrasound ray-tracing binary, invoked via subprocess (optional)

## Architecture

`Model3D` (in `model.py`) is the central orchestrator:

```
Model3D
├── EarthquakeSource   (source.py)   — event lat/lon/depth/time from TOML
├── Atmosphere1D       (atmosphere.py) — MSISE-00 vertical profiles
├── Ionosphere1D       (ionosphere.py) — IRI2020 electron density profiles
├── MagneticField1D    (igrf.py)      — IGRF field components via ppigrf
└── infraga tools      (infraga.py)   — subprocess wrapper for infraga-sph binaries
```

- 1D profile classes compute vertical profiles at a single (lat, lon, time); `Model3D` assembles them into a 3D grid.
- All physical output is returned as `xr.Dataset` with explicit coordinates.
- Model and event parameters are loaded from **TOML files** (see `example/event.toml`).
- `infraga-sph` / `infraga-sph-rngdep` binaries must be manually copied into the package `bin/` path; do not assume they are on `PATH`.

## Conventions

### Code style
- **Line length**: 88 characters (flake8 + black-compatible).
- **Docstrings**: NumPy style — `Parameters`, `Attributes`, `Returns`, `Raises`, `Methods` sections.
- **Private helpers** inside classes use a single underscore prefix (e.g., `_cache_token`).

### Units — always document in docstrings and comments
- Altitude / depth: **km**
- Latitude / longitude: **degrees**
- Temperature: **Kelvin (K)**
- Pressure: **Pascals (Pa)**
- Density: **kg m⁻³** (mass) or **m⁻³** (number)
- Velocity: **km/s**

### Optional dependencies
Wrap optional imports in `try/except` and expose an `*_AVAILABLE` boolean flag:
```python
try:
    import ppigrf
    PPIGRF_AVAILABLE = True
except ImportError:
    ppigrf = None
    PPIGRF_AVAILABLE = False
```
Guard usage with the flag rather than raising at import time.

### Data containers
- Store all profile/grid results in `xr.Dataset` with meaningful variable names and coordinate labels.
- Avoid bare numpy arrays as return values from public methods.

## Testing

- Framework: `unittest` (standard library).
- Test files live in `tests/`; mirror module names (e.g., `test_infraga.py`).
- Use `unittest.mock.patch` to stub subprocess calls and optional-dependency code paths.
- Run tests with: `python -m pytest tests/` or `python -m unittest discover tests/`.

## Common Pitfalls

- **infraGA binaries**: Never call `infraga compile`. Use the build-and-copy workflow documented in `README.md`. Check binary existence with `shutil.which` or `Path.exists()` before running.
- **Optional imports**: Do not assume `iri2020` or `ppigrf` are installed; always check the `*_AVAILABLE` flag before use.
- **xarray dimensions**: Dimension names must be consistent across the codebase (`altitude`, `latitude`, `longitude`); mismatched names break model assembly.
- **TOML config**: Event time strings must follow `"%Y-%m-%dT%H:%M:%SZ"` format; validate on load.
- **Legacy code**: The `old/` directory is archived legacy code — do not import from or modify it.
