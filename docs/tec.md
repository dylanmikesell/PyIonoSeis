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
receiver_format = "csv"
receiver_csv = "example/inputs/gnss_receivers_list_complete.csv"
receiver_code = "TUA1"
orbit_format = "h5"
orbit_h5 = "example/inputs/SatsOrbs2010298.h5"
sat_id = "G21"
```

## Legacy Input Compatibility

Run LOS TEC with historical VAN-style files by selecting legacy input formats.

### Legacy Receiver File (`listesta_all.txt`)

- Format: whitespace-delimited station rows with `code x_m y_m z_m lat lon`
- The loader uses ECEF (`x_m`, `y_m`, `z_m`) as authoritative and recomputes
  geodetic latitude/longitude/height.

### Legacy Orbit File (`.pos`)

- Format: whitespace-delimited rows with `seconds_of_day x_m y_m z_m lon lat ele`
- One satellite per file (`satNN_XXXXX.pos`)
- Satellite identity supports mixed constellations via mapping file.

```toml
[tec]
elevation_mask_deg = 20.0
output_dt_s = 10.0
ipp_altitude_km = 350.0

receiver_format = "listesta"
receiver_listesta = "example/VAN/listesta_all.txt"
receiver_code = "agrd"

orbit_format = "pos"
# Option 1: direct SATPOS file
orbit_pos = "example/VAN/SATPOS/16591/sat17_16591.pos"

# Option 2: legacy SATPOS root/date/sat lookup
# satpos_root = "example/VAN/SATPOS"
# satpos_date = "16591"
# sat_number = 17

# Mixed-constellation mapping (CSV columns: sat_number,sat_id)
sat_mapping_file = "example/VAN/sat_number_mapping.csv"

# Start offset (s) relative to event origin for legacy SATPOS alignment
start_offset_s = 100.0
```
