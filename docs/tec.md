# Line-of-Sight TEC

Use the LOS TEC workflow to compute a time series of slant total electron content (TEC) between a GNSS receiver and a satellite. The computation combines the 3-D electron density grid with time-varying satellite geometry.

## Geometry

Define the receiver and satellite positions in ECEF coordinates (km):

- Receiver position: $\mathbf{r}_r$
- Satellite position: $\mathbf{r}_s$

The line-of-sight unit vector is:

$$
\hat{\mathbf{u}} = \frac{\mathbf{r}_s - \mathbf{r}_r}{\|\mathbf{r}_s - \mathbf{r}_r\|}
$$

For each altitude $h$ (km) in the model grid, intersect the LOS with a spherical shell of radius $R_E + h$:

$$
\|\mathbf{r}_r + s\,\hat{\mathbf{u}}\|^2 = (R_E + h)^2
$$

Solve for $s$ (km) along the LOS. The intersection latitude/longitude define the ionospheric pierce point (IPP) at each altitude.

## Electron Density Sampling

At each time and altitude, sample the 3-D electron density fields on the IPP path:

- Background density: $N_{e0}(\phi, \lambda, h)$
- Perturbation density: $\Delta N_e(\phi, \lambda, h, t)$

Missing values are treated as zero, and the number of missing samples per LOS is logged.

## TEC Integration

TEC is the line integral of electron density along the slant path:

$$
\mathrm{TEC}(t) = 10^{-16} \int N_e(s, t)\, ds
$$

The current implementation uses trapezoidal integration in meters along the sampled path:

$$
\mathrm{TEC}(t) \approx 10^{-16} \sum_i \frac{N_e(s_i, t) + N_e(s_{i+1}, t)}{2} (s_{i+1} - s_i)
$$

Outputs include background TEC, perturbation TEC, and total TEC.

## Elevation Mask

The elevation angle is computed from the local ENU frame at the receiver. Samples with elevation below the configured mask are skipped. The default mask is 20 degrees, configurable in the TOML file.

## Output Metadata

The LOS output dataset includes:

- `tec_background`, `tec_perturbation`, `tec_total` (TECU)
- `elevation_deg`, `azimuth_deg` (deg, azimuth 0–360 from north)
- `ipp_latitude_deg`, `ipp_longitude_deg`

## Configuration

Add LOS TEC settings under the `[tec]` section in the event TOML file:

```toml
[tec]
elevation_mask_deg = 20.0
output_dt_s = 10.0
ipp_altitude_km = 350.0
receiver_csv = "example/inputs/gnss_receivers_list_complete.csv"
receiver_code = "TUA1"
orbit_h5 = "example/inputs/SatsOrbs2010298.h5"
sat_id = "G21"
```
