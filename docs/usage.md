# Usage

To use PyIonoSeis in a project:

```python
import pyionoseis
```

## Spherical Ray Tracing

`Model3D.trace_rays(...)` uses the infraGA spherical runner only.

```python
from pyionoseis.model import Model3D
from pyionoseis.source import EarthquakeSource

model = Model3D()
source = EarthquakeSource()
model.assign_source(source)
model.make_3Dgrid()

# Spherical azimuth sweep
raypaths_3d = model.trace_rays(type="3d")

# Spherical single-azimuth (north) mode
raypaths_2d = model.trace_rays(type="2d")
```

Raw outputs are stored on the model object:

- `model.raypaths`
- `model.ray_arrivals`

`model.raypaths` includes geometry and timing fields for later interpolation work:

- `ray_lat_deg`
- `ray_lon_deg`
- `ray_alt_km`
- `travel_time_s`
- `transport_amplitude_db`
- `absorption_db`
