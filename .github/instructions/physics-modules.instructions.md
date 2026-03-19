---
applyTo: "pyionoseis/**/*.py"
description: "Coding conventions for PyIonoSeis physics module source files"
name: "Physics Module Guidelines"
---
# Physics Module Guidelines

These instructions apply to all source files in `pyionoseis/`. They complement the universal conventions in `.github/copilot-instructions.md`.

## Library-First Principle

Prefer existing optimized libraries over new implementations:

- **Numerical integration** → `scipy.integrate` (e.g., `quad`, `trapezoid`, `cumulative_trapezoid`)
- **Interpolation** → `scipy.interpolate` (e.g., `interp1d`, `RegularGridInterpolator`)
- **Signal processing** → `scipy.signal`
- **Special functions** → `scipy.special`
- **Linear algebra** → `numpy.linalg`, not custom solvers
- **Geodesic calculations** → `pyproj.Geod` or equivalent, not hand-rolled haversine
- **Array operations** → always vectorized numpy; never iterate over altitude arrays in Python loops

## Module Structure

Each physics domain gets its own module file. Do not mix domains (e.g., atmosphere + ionosphere in one file).

### 1D Profile Class Pattern

All `*1D` classes follow this structure:

```python
class DomainName1D:
    """NumPy-style docstring with Parameters, Attributes, Methods sections."""

    def __init__(self, lat, lon, alt_km, time, model="default"):
        self.lat = lat          # degrees
        self.lon = lon          # degrees
        self.alt_km = alt_km    # km
        self.time = time
        self.model = model
        self.<domain> = None    # populated by compute_*() call
        self.compute_<model>_model()

    def compute_<model>_model(self) -> xr.Dataset:
        """Compute and store as self.<domain>; also return the Dataset."""
        ...

    def plot(self) -> None:
        """Visualize the vertical profile."""
        ...
```

- Call `compute_*` from `__init__` so the object is always ready after construction.
- Store results on `self.<domain>` **and** return the `xr.Dataset`.
- `plot()` is a convenience method; it should never mutate state.

## xarray Conventions

- Canonical dimension names: `altitude` (km), `latitude` (deg), `longitude` (deg).
- Always attach units as coordinate/variable `attrs`: `ds["temperature"].attrs["units"] = "K"`.
- Variables names use snake_case and match their physical meaning (e.g., `electron_density`, `mass_density`, `pressure`).

```python
ds = xr.Dataset(
    {"electron_density": (["altitude"], ne_m3)},
    coords={"altitude": alt_km},
    attrs={"model": "iri2020"},
)
ds["electron_density"].attrs["units"] = "m^-3"
```

## Docstring Requirements

Every public function and class must have a complete NumPy-style docstring:

```python
def compute_pressure(density_kg_m3, temperature_K):
    """
    Compute atmospheric pressure from ideal gas law.

    Parameters
    ----------
    density_kg_m3 : np.ndarray
        Mass density in kg m^-3.
    temperature_K : np.ndarray
        Temperature in Kelvin.

    Returns
    -------
    xr.Dataset
        Dataset with variable ``pressure`` in Pascals.
    """
```

- Document physical units for **every** parameter and return value.
- List exceptions that callers must handle in a `Raises` section.

## Performance

- Profile arrays (altitude grids) are typically 100–1000 elements; all operations must be vectorized.
- Avoid copying large arrays unnecessarily; prefer in-place numpy operations when safe.
- Cache expensive external model calls (e.g., MSISE-00 runs) if the same inputs are reused.
