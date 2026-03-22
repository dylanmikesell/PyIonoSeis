# Usage

## Quick Import

```python
import pyionoseis
```

## Full Modelling Workflow

The standard pipeline takes an event TOML file and builds a fully-populated
3-D grid in five steps.

```python
import logging
logging.basicConfig(level=logging.INFO)  # show progress from assign_* methods

from pyionoseis.model import Model3D
from pyionoseis.source import EarthquakeSource

# 1. Load event and model parameters from TOML
source = EarthquakeSource("event.toml")
model  = Model3D("event.toml")

# 2. Assign the earthquake source
model.assign_source(source)

# 3. Build the lat / lon / altitude grid
model.make_3Dgrid()

# 4. Populate physical fields (parallel across grid columns)
model.assign_atmosphere()       # MSISE-00 → density, pressure, temperature, velocity
model.assign_ionosphere()       # IRI2020  → electron_density
model.assign_magnetic_field()   # IGRF     → Be, Bn, Bu, inclination, declination …
model.trace_rays(type="2d", az_interp=True, az_interp_step=1.0)
model.assign_wavevector(mapping_mode="nearest")
model.assign_continuity(output_dir="continuity_output")

print(model.grid)  # xr.Dataset with all computed variables
```

`model.grid` is always an `xr.Dataset` with named dimensions
`(latitude, longitude, altitude)`.

## Progress Logging

`assign_ionosphere` and `assign_magnetic_field` run 1-D profile computations
in parallel threads. Progress and warnings are emitted through Python's
standard `logging` module — nothing is printed to stdout by default.

```python
import logging

# Show INFO messages in the terminal
logging.basicConfig(level=logging.INFO)

# Suppress all pyionoseis messages
logging.getLogger("pyionoseis").setLevel(logging.WARNING)
```

Example output when `INFO` is enabled:

```text
INFO:pyionoseis.model:Computing ionospheric electron density (iri2020): 121 profiles (11 lat × 11 lon × 25 alt)
INFO:pyionoseis.model:Ionospheric computation completed.
```

## TOML Configuration

Both `EarthquakeSource` and `Model3D` accept the same TOML file.
See `example/event.toml` for a complete example.

```toml
[event]
time      = "2023-10-01T12:34:56Z"  # ISO 8601, always UTC
latitude  = 37.7749
longitude = -122.4194
depth     = 10.0

[model]
name           = "example"
radius         = 500.0      # km — epicentral distance
height         = 500.0      # km — top of the atmosphere domain
winds          = false
atmosphere     = "msise00"
ionosphere     = "iri2020"
grid_spacing   = 1.0        # degrees
height_spacing = 50.0       # km

[continuity]
# seconds
t0_s = 0.0
tmax_s = 3600.0
dt_s = 10.0
```

## Visualisation

```python
# Source location on a cartopy map
model.plot_source()

# Lat/lon grid points
model.plot_grid(show_gridlines=True)

# Vertical profile at source location
model.plot_variable(variable="electron_density")

# Horizontal map at a specific altitude
model.plot_variable(variable="electron_density", altitude_slice=300)
model.plot_variable(variable="inclination",       altitude_slice=250, cmap="coolwarm")

# Atmospheric profiles
model.plot_variable(variable="temperature")
model.plot_variable(variable="velocity")

# Wavevector components (see the wavevector guide for full examples)
model.plot_variable(variable="kr", altitude_slice=300)
```

## Line-of-Sight TEC

Compute LOS TEC using receiver and orbit inputs listed in the TOML file:

```python
from pyionoseis.model import Model3D
from pyionoseis.source import EarthquakeSource

source = EarthquakeSource("event.toml")
model = Model3D("event.toml")
model.assign_source(source)
model.make_3Dgrid()
model.assign_ionosphere()

# If you already computed continuity, keep `model.continuity` populated.
tec_ds = model.compute_los_tec()
print(tec_ds)
```

See the full mathematical description in [Line-of-Sight TEC](tec.md).

## Spherical Ray Tracing

`Model3D.trace_rays(...)` uses the infraGA spherical runner only.
The `infraga-sph` binary must be compiled and placed in the package `bin/`
path — see `README.md` for the build-and-copy workflow.

Set launch inclination defaults in TOML under `[rays]`:

```toml
[rays]
incl_min = 45.0
incl_max = 90.0
incl_step = 1.0
```

Use this physical convention for takeoff angle:

- `90°` is vertical upward launch.
- Smaller angles tilt toward horizontal propagation.
- `incl_min` and `incl_max` are inclusive bounds used with `incl_step`.

```python
# Spherical azimuth sweep (3-D)
raypaths = model.trace_rays(type="3d")

# Single-azimuth (2-D), then project to synthetic 3-D
raypaths = model.trace_rays(
    type="2d",
    az_interp=True,
    az_interp_step=10.0,
    output_dir="ray_tracing_output",
    reuse_existing=True,   # use cached result if hash matches
)
```

Raw outputs are stored on the model object:

- `model.raypaths` — `xr.Dataset` with ray geometry and timing.
- `model.ray_arrivals` — `xr.Dataset` with ground-level arrival metadata.
- `model.raytrace_run_dir` — path to the directory holding infraGA output files.

Key variables in `model.raypaths`:

- `ray_lat_deg`, `ray_lon_deg`, `ray_alt_km` — 3-D position
- `travel_time_s` — one-way travel time in seconds
- `transport_amplitude_db`, `absorption_db` — amplitude descriptors

## Wavevector Mapping

Map geometry-based wavevector components from raypaths to the grid. Use nearest
neighbor for speed, or weighted averaging for smoother fields.

```python
model.assign_wavevector(
    mapping_mode="nearest",
    interpolation_radius_km=50.0,
    altitude_window_km=50.0,
    smoothing_radius_km=0.0,
)
```

See the full workflow and plotting examples in
[Wavevector Mapping](wavevector.md).

## Loading Previously Computed Rays

```python
model.load_rays(
    raypaths_file="ray_tracing_output/infraga_2d_sph_abc123.raypaths.dat",
    validate_signature=True,  # checks hash against current source/atmosphere
)
```

Use `validate_signature=True` when you want strict source/atmosphere consistency checks.

