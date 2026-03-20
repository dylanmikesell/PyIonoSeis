# Wasp3D Wave Vector Components

Use this note to understand how the legacy wasp3d solver computes the wave
vector components $k_r$, $k_t$, and $k_p$ (written as `kr`, `kt`, and `kf` in
legacy output). The formulas live in the Fortran ray tracer and are used when
writing NetCDF outputs.

## Coordinate System and Variables

Work in spherical coordinates $(\rho, \theta, \phi)$ where:

- $\rho$ is radial distance from Earth center in km.
- $\theta$ is colatitude in radians.
- $\phi$ is azimuth in radians.

The ray state uses:

- $q$ = position vector (`qsol` or `qsht`) in $(\rho, \theta, \phi)$.
- $p$ = slowness vector (`psol` or `psht`).

The code integrates the ray equations using the Hamiltonian formulation in
[pyionoseis/legacy/src/wasp3d/rk2_dyn.f](pyionoseis/legacy/src/wasp3d/rk2_dyn.f#L1-L36).

## Initialization Formula

At the source, wasp3d computes the slowness components and then normalizes them
using the spherical metric. The normalization factor is:

$$
\|p\| = \frac{1}{p_{\mathrm{mod}}} \sqrt{p_\rho^2 + \left(\frac{p_\theta}{\rho}\right)^2 + \left(\frac{p_\phi}{\rho \sin\theta}\right)^2}
$$

The wave vector components stored to `bufpts(4:6)` are:

$$
\begin{aligned}
 k_r &= \frac{p_\rho}{\|p\|} \\
 k_t &= \frac{p_\theta}{\rho\,\|p\|} \\
 k_p &= \frac{p_\phi}{\rho\,\sin\theta\,\|p\|}
\end{aligned}
$$

See the initialization in
[pyionoseis/legacy/src/wasp3d/ray3d_dyn.f](pyionoseis/legacy/src/wasp3d/ray3d_dyn.f#L152-L176).

## Per Step Update

During propagation, wasp3d repeats the same normalization at each ray point
using the updated slowness `psol_new` and position `qsol_new`:

$$
\|p\|_{\mathrm{new}} = \sqrt{p_\rho^2 + \left(\frac{p_\theta}{\rho}\right)^2 + \left(\frac{p_\phi}{\rho \sin\theta}\right)^2}
$$

and then updates:

$$
\begin{aligned}
 k_r &= \frac{p_\rho}{\|p\|_{\mathrm{new}}} \\
 k_t &= \frac{p_\theta}{\rho\,\|p\|_{\mathrm{new}}} \\
 k_p &= \frac{p_\phi}{\rho\,\sin\theta\,\|p\|_{\mathrm{new}}}
\end{aligned}
$$

See the propagation update in
[pyionoseis/legacy/src/wasp3d/ray3d_dyn.f](pyionoseis/legacy/src/wasp3d/ray3d_dyn.f#L328-L341).

## Units and Interpretation

Interpret these components as metric-corrected slowness directions, not raw
Cartesian components. The normalization accounts for the spherical coordinate
basis so that $k_r$, $k_t$, and $k_p$ are dimensionless direction cosines in the
local spherical frame. The NetCDF output stores them as `wasp_kr`, `wasp_kt`,
`wasp_kf`.

The NetCDF extraction that maps `bufpts` into `kr`, `kt`, and `kf` is shown in
[pyionoseis/legacy/src/wasp3d/outputw_netcdf.f](pyionoseis/legacy/src/wasp3d/outputw_netcdf.f#L165-L185).

## infraGA Outputs vs wasp3d Wave Vector

infraGA raypaths output currently writes geometry and timing only. When you
enable ray outputs (`write_ray` or `write_rays` in infraGA), the raypaths files
contain:

- latitude, longitude, altitude
- transport amplitude, absorption
- travel time

See the raypaths headers in infraGA source:

- https://github.com/LANL-Seismoacoustics/infraGA/blob/main/infraga/src/main.infraga.sph.cpp
- https://github.com/LANL-Seismoacoustics/infraGA/blob/main/infraga/src/main.infraga.sph.rngdep.cpp

What you can and cannot recover from those outputs:

- You can approximate direction cosines by differentiating the ray geometry.
- You cannot recover the exact wave vector components used by wasp3d because
    infraGA does not output the slowness components or local sound speed along
    the raypath.
- You cannot reconstruct the metric-normalized $k_r$, $k_t$, $k_p$ used in
    wasp3d without additional slowness information from infraGA.

Use a central-difference direction approximation in spherical coordinates if
you need a geometry-only proxy. Define the arc-length step $\Delta s$ using the
great-circle distance between successive points (in km). Then compute the local
direction cosines $(t_r, t_\theta, t_\phi)$ as:

$$
\begin{aligned}
 t_r &\approx \frac{\rho_{i+1} - \rho_{i-1}}{2\,\Delta s} \\
 t_\theta &\approx \frac{\rho_i\,(\theta_{i+1} - \theta_{i-1})}{2\,\Delta s} \\
 t_\phi &\approx \frac{\rho_i\,\sin\theta_i\,(\phi_{i+1} - \phi_{i-1})}{2\,\Delta s}
\end{aligned}
$$

Treat these as dimensionless direction cosines, not true wave number
components. Expect numerical noise near turning points where the path gradient
is small or changes sign.

To recover exact $k_r$, $k_t$, $k_p$ from infraGA, the C++ code would need to
output the internal slowness components (or a local sound-speed field) alongside
raypath geometry.

Summary: infraGA outputs allow direction-only approximations, but wasp3d’s
metric-normalized wave vector components require slowness data that infraGA
does not currently emit.

## Data Flow

```mermaid
flowchart LR
    A[ray3d_dyn.f
compute psht, psol] --> B[normalize with metric
kr, kt, kp]
    B --> C[bufpts(4:6)]
    C --> D[outputw_netcdf.f
wasp_kr, wasp_kt, wasp_kf]
```

## Example

Use this pseudo-code to match the legacy computation in a modern
implementation:

```python
import numpy as np

def wavevector_components(p_rho, p_theta, p_phi, rho, theta, pmod=None):
    metric_norm = np.sqrt(
        p_rho**2 + (p_theta / rho) ** 2 + (p_phi / (rho * np.sin(theta))) ** 2
    )
    if pmod is not None:
        metric_norm = metric_norm / pmod
    k_r = p_rho / metric_norm
    k_t = p_theta / (rho * metric_norm)
    k_p = p_phi / (rho * np.sin(theta) * metric_norm)
    return k_r, k_t, k_p
```

## TODO

- Implement a wave vector computation in the modern `pyionoseis` ray tracing
  stack that reproduces the legacy spherical metric normalization used for
  `kr`, `kt`, and `kp`.
- Add an infraGA flag or code change to emit slowness components along
    raypaths if exact wave vector components are required.
