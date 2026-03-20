"""Wavevector approximation utilities based on infraGA ray geometry."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    cKDTree = None
    SCIPY_AVAILABLE = False
import numpy as np
import xarray as xr

EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class RaySegment:
    """Container for contiguous ray segment indices."""

    start: int
    end: int


def _lat_lon_alt_to_ecef(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    alt_km: np.ndarray,
) -> np.ndarray:
    """Convert spherical (lat, lon, alt) to ECEF Cartesian coordinates.

    Parameters
    ----------
    lat_deg : np.ndarray
        Latitude in degrees.
    lon_deg : np.ndarray
        Longitude in degrees.
    alt_km : np.ndarray
        Altitude in km above mean Earth radius.

    Returns
    -------
    np.ndarray
        ECEF positions with shape (N, 3) in km.
    """
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    radius = EARTH_RADIUS_KM + alt_km

    cos_lat = np.cos(lat_rad)
    sin_lat = np.sin(lat_rad)
    cos_lon = np.cos(lon_rad)
    sin_lon = np.sin(lon_rad)

    x = radius * cos_lat * cos_lon
    y = radius * cos_lat * sin_lon
    z = radius * sin_lat

    return np.column_stack((x, y, z))


def _spherical_basis(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local spherical basis vectors (e_r, e_theta, e_phi).

    Parameters
    ----------
    lat_deg : np.ndarray
        Latitude in degrees.
    lon_deg : np.ndarray
        Longitude in degrees.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Basis vectors with shape (N, 3) each.
    """
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)

    cos_lat = np.cos(lat_rad)
    sin_lat = np.sin(lat_rad)
    cos_lon = np.cos(lon_rad)
    sin_lon = np.sin(lon_rad)

    e_r = np.column_stack((cos_lat * cos_lon, cos_lat * sin_lon, sin_lat))
    e_theta = np.column_stack((-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat))
    e_phi = np.column_stack((-sin_lon, cos_lon, np.zeros_like(lat_rad)))

    return e_r, e_theta, e_phi


def _smooth_1d(values: np.ndarray, half_window: int) -> np.ndarray:
    """Apply a centered moving-average smoother with edge padding."""
    if half_window <= 0:
        return values

    window_len = 2 * half_window + 1
    kernel = np.ones(window_len, dtype=float) / float(window_len)
    padded = np.pad(values, (half_window, half_window), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _smooth_positions(positions: np.ndarray, half_window: int) -> np.ndarray:
    """Smooth positions along a ray segment using a moving average."""
    if half_window <= 0:
        return positions

    smoothed = np.empty_like(positions)
    for idx in range(3):
        smoothed[:, idx] = _smooth_1d(positions[:, idx], half_window)
    return smoothed


def _split_ray_segments(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    alt_km: np.ndarray,
    travel_time_s: np.ndarray | None,
) -> list[RaySegment]:
    """Split ray points into contiguous segments based on validity and time order."""
    valid = np.isfinite(lat_deg) & np.isfinite(lon_deg) & np.isfinite(alt_km)
    if travel_time_s is not None:
        valid &= np.isfinite(travel_time_s)

    segments: list[RaySegment] = []
    start = None
    for idx in range(lat_deg.size):
        if not valid[idx]:
            if start is not None:
                segments.append(RaySegment(start, idx))
                start = None
            continue

        if start is None:
            start = idx
            continue

        if travel_time_s is not None and travel_time_s[idx] <= travel_time_s[idx - 1]:
            segments.append(RaySegment(start, idx))
            start = idx

    if start is not None:
        segments.append(RaySegment(start, lat_deg.size))

    return segments


def _compute_segment_tangent(
    positions: np.ndarray,
    smoothing_radius_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute unit tangents and arc length for a single contiguous segment."""
    if positions.shape[0] == 1:
        return np.full_like(positions, np.nan), np.zeros(positions.shape[0])

    deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    median_step = np.median(deltas[deltas > 0]) if np.any(deltas > 0) else 0.0
    half_window = 0
    if smoothing_radius_km > 0.0 and median_step > 0.0:
        half_window = int(np.ceil(smoothing_radius_km / median_step))

    smooth_pos = _smooth_positions(positions, half_window)
    ds = np.linalg.norm(np.diff(smooth_pos, axis=0), axis=1)
    arc = np.zeros(smooth_pos.shape[0])
    if ds.size:
        arc[1:] = np.cumsum(ds)

    tangent = np.full_like(smooth_pos, np.nan)
    if smooth_pos.shape[0] == 2:
        delta = smooth_pos[1] - smooth_pos[0]
        step = arc[1] - arc[0]
        if step > 0.0:
            tangent[0] = delta / step
            tangent[1] = delta / step
    else:
        forward_step = arc[1] - arc[0]
        if forward_step > 0.0:
            tangent[0] = (smooth_pos[1] - smooth_pos[0]) / forward_step
        backward_step = arc[-1] - arc[-2]
        if backward_step > 0.0:
            tangent[-1] = (smooth_pos[-1] - smooth_pos[-2]) / backward_step

        delta = smooth_pos[2:] - smooth_pos[:-2]
        step = arc[2:] - arc[:-2]
        valid_step = step > 0.0
        tangent[1:-1][valid_step] = delta[valid_step] / step[valid_step, None]

    norms = np.linalg.norm(tangent, axis=1)
    valid_norm = norms > 0.0
    tangent[valid_norm] = tangent[valid_norm] / norms[valid_norm, None]

    return tangent, arc


def compute_ray_wavevectors(
    raypaths: xr.Dataset,
    smoothing_radius_km: float = 20.0,
    turning_point: bool = True,
) -> xr.Dataset:
    """Compute direction-cosine wavevector components along ray paths.

    Parameters
    ----------
    raypaths : xr.Dataset
        Dataset containing ``ray_lat_deg`` (degrees), ``ray_lon_deg`` (degrees),
        and ``ray_alt_km`` (km).
    smoothing_radius_km : float, optional
        Smoothing radius (km) applied along each ray before differentiation.
    turning_point : bool, optional
        If True, enforce ``k_r = 0`` at detected turning points and renormalize.

    Returns
    -------
    xr.Dataset
        Dataset with variables ``ray_k_r``, ``ray_k_t``, ``ray_k_p`` and
        tangent vectors in ECEF coordinates.
    """
    lat = raypaths["ray_lat_deg"].values.astype(float)
    lon = raypaths["ray_lon_deg"].values.astype(float)
    alt = raypaths["ray_alt_km"].values.astype(float)
    time = raypaths["travel_time_s"].values if "travel_time_s" in raypaths else None

    tangent = np.full((lat.size, 3), np.nan)
    arc_length = np.full(lat.size, np.nan)

    segments = _split_ray_segments(lat, lon, alt, time)
    for segment in segments:
        seg_slice = slice(segment.start, segment.end)
        positions = _lat_lon_alt_to_ecef(lat[seg_slice], lon[seg_slice], alt[seg_slice])
        seg_tangent, seg_arc = _compute_segment_tangent(
            positions, smoothing_radius_km=smoothing_radius_km
        )
        tangent[seg_slice] = seg_tangent
        arc_length[seg_slice] = seg_arc

    e_r, e_theta, e_phi = _spherical_basis(lat, lon)
    k_r = np.einsum("ij,ij->i", tangent, e_r)
    k_t = np.einsum("ij,ij->i", tangent, e_theta)
    k_p = np.einsum("ij,ij->i", tangent, e_phi)

    if turning_point:
        radius = EARTH_RADIUS_KM + alt
        for segment in segments:
            seg_slice = slice(segment.start, segment.end)
            seg_radius = radius[seg_slice]
            seg_arc = arc_length[seg_slice]
            if seg_radius.size < 3 or not np.all(np.isfinite(seg_arc)):
                continue

            dr_ds = np.gradient(seg_radius, seg_arc, edge_order=1)
            sign_change = np.sign(dr_ds[:-1]) * np.sign(dr_ds[1:]) < 0
            local_turning = np.where(sign_change)[0] + 1
            if local_turning.size == 0:
                continue

            turning_idx = local_turning + segment.start
            k_r[turning_idx] = 0.0
            kt_kp_norm = np.sqrt(k_t[turning_idx] ** 2 + k_p[turning_idx] ** 2)
            nonzero = kt_kp_norm > 0.0
            k_t[turning_idx[nonzero]] = k_t[turning_idx[nonzero]] / kt_kp_norm[nonzero]
            k_p[turning_idx[nonzero]] = k_p[turning_idx[nonzero]] / kt_kp_norm[nonzero]

            tangent[turning_idx] = (
                k_r[turning_idx, None] * e_r[turning_idx]
                + k_t[turning_idx, None] * e_theta[turning_idx]
                + k_p[turning_idx, None] * e_phi[turning_idx]
            )

    ds_out = xr.Dataset(
        {
            "ray_lat_deg": ("ray_point", lat),
            "ray_lon_deg": ("ray_point", lon),
            "ray_alt_km": ("ray_point", alt),
            "ray_k_r": ("ray_point", k_r),
            "ray_k_t": ("ray_point", k_t),
            "ray_k_p": ("ray_point", k_p),
            "ray_tangent_x": ("ray_point", tangent[:, 0]),
            "ray_tangent_y": ("ray_point", tangent[:, 1]),
            "ray_tangent_z": ("ray_point", tangent[:, 2]),
            "ray_arc_length_km": ("ray_point", arc_length),
        }
    )

    ds_out["ray_k_r"].attrs["units"] = "dimensionless"
    ds_out["ray_k_t"].attrs["units"] = "dimensionless"
    ds_out["ray_k_p"].attrs["units"] = "dimensionless"
    ds_out["ray_arc_length_km"].attrs["units"] = "km"
    ds_out.attrs["wavevector_method"] = "geometry_tangent"
    ds_out.attrs["smoothing_radius_km"] = float(smoothing_radius_km)
    ds_out.attrs["turning_point_enforced"] = bool(turning_point)

    return ds_out


def map_wavevector_to_grid(
    grid: xr.Dataset,
    ray_wavevectors: xr.Dataset,
    interpolation_radius_km: float = 50.0,
    mapping_mode: str = "nearest",
    use_kdtree: bool = True,
    altitude_window_km: float | None = None,
    min_points: int = 3,
    weight_power: float = 2.0,
    chunk_size: int = 1024,
) -> xr.Dataset:
    """Interpolate ray-based wavevectors to the 3-D model grid.

    Parameters
    ----------
    grid : xr.Dataset
        Grid dataset with coordinates ``latitude``, ``longitude``, ``altitude``.
    ray_wavevectors : xr.Dataset
        Output of :func:`compute_ray_wavevectors` with tangent vectors.
    interpolation_radius_km : float, optional
        Neighborhood radius (km) for local averaging.
    mapping_mode : str, optional
        Mapping mode, either ``"nearest"`` or ``"weighted"``.
    use_kdtree : bool, optional
        If True, use a KD-tree for nearest-neighbor lookups when available.
    altitude_window_km : float, optional
        Vertical window (km) to limit ray candidates. Defaults to
        ``interpolation_radius_km`` when omitted.
    min_points : int, optional
        Minimum number of ray points required to assign a grid value.
    weight_power : float, optional
        Power for the distance taper ``(1 - d / r) ** weight_power``.
    chunk_size : int, optional
        Grid points processed per chunk to control memory usage.

    Returns
    -------
    xr.Dataset
        Dataset with ``kr``, ``kt``, ``kp`` and ``wavevector_raypoint_count``.
    """
    if interpolation_radius_km <= 0.0:
        raise ValueError("interpolation_radius_km must be > 0.")
    if min_points < 1:
        raise ValueError("min_points must be >= 1.")
    if mapping_mode not in {"nearest", "weighted"}:
        raise ValueError("mapping_mode must be 'nearest' or 'weighted'.")
    if altitude_window_km is None:
        altitude_window_km = float(interpolation_radius_km)
    if altitude_window_km <= 0.0:
        raise ValueError("altitude_window_km must be > 0.")
    if mapping_mode == "nearest" and use_kdtree and not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for KD-tree nearest-neighbor mapping. "
            "Install with the optional dependency 'pyionoseis[wavevector]'."
        )

    lat = grid.coords["latitude"].values.astype(float)
    lon = grid.coords["longitude"].values.astype(float)
    alt = grid.coords["altitude"].values.astype(float)

    lat_grid, lon_grid, alt_grid = np.meshgrid(lat, lon, alt, indexing="ij")
    flat_lat = lat_grid.ravel()
    flat_lon = lon_grid.ravel()
    flat_alt = alt_grid.ravel()

    ray_lat = ray_wavevectors["ray_lat_deg"].values.astype(float)
    ray_lon = ray_wavevectors["ray_lon_deg"].values.astype(float)
    ray_alt = ray_wavevectors["ray_alt_km"].values.astype(float)
    ray_tangent = np.column_stack(
        (
            ray_wavevectors["ray_tangent_x"].values,
            ray_wavevectors["ray_tangent_y"].values,
            ray_wavevectors["ray_tangent_z"].values,
        )
    )

    valid = np.all(np.isfinite(ray_tangent), axis=1)
    valid &= np.isfinite(ray_lat) & np.isfinite(ray_lon) & np.isfinite(ray_alt)

    ray_lat = ray_lat[valid]
    ray_lon = ray_lon[valid]
    ray_alt = ray_alt[valid]
    ray_tangent = ray_tangent[valid]

    ray_pos = _lat_lon_alt_to_ecef(ray_lat, ray_lon, ray_alt)

    out_k_r = np.full(flat_lat.size, np.nan)
    out_k_t = np.full(flat_lat.size, np.nan)
    out_k_p = np.full(flat_lat.size, np.nan)
    out_count = np.zeros(flat_lat.size, dtype=int)

    for start in range(0, flat_lat.size, chunk_size):
        end = min(start + chunk_size, flat_lat.size)
        chunk_lat = flat_lat[start:end]
        chunk_lon = flat_lon[start:end]
        chunk_alt = flat_alt[start:end]

        alt_min = chunk_alt.min() - altitude_window_km
        alt_max = chunk_alt.max() + altitude_window_km
        alt_mask = (ray_alt >= alt_min) & (ray_alt <= alt_max)
        if not np.any(alt_mask):
            continue

        chunk_ray_pos = ray_pos[alt_mask]
        chunk_ray_tangent = ray_tangent[alt_mask]
        chunk_ray_alt = ray_alt[alt_mask]

        grid_pos = _lat_lon_alt_to_ecef(chunk_lat, chunk_lon, chunk_alt)

        if mapping_mode == "nearest":
            if use_kdtree:
                tree = cKDTree(chunk_ray_pos)
                dist, idx = tree.query(
                    grid_pos,
                    distance_upper_bound=float(interpolation_radius_km),
                )
                valid = np.isfinite(dist) & (dist <= float(interpolation_radius_km))

                if np.any(valid):
                    ray_alt_sel = chunk_ray_alt[idx[valid]]
                    alt_ok = np.abs(ray_alt_sel - chunk_alt[valid]) <= altitude_window_km
                    valid_idx = np.where(valid)[0][alt_ok]
                    if valid_idx.size:
                        chosen = idx[valid][alt_ok]
                        out_count[start:end][valid_idx] = 1
                        avg_vec = chunk_ray_tangent[chosen]
                        e_r, e_theta, e_phi = _spherical_basis(
                            chunk_lat[valid_idx],
                            chunk_lon[valid_idx],
                        )
                        out_k_r[start:end][valid_idx] = np.einsum(
                            "ij,ij->i", avg_vec, e_r
                        )
                        out_k_t[start:end][valid_idx] = np.einsum(
                            "ij,ij->i", avg_vec, e_theta
                        )
                        out_k_p[start:end][valid_idx] = np.einsum(
                            "ij,ij->i", avg_vec, e_phi
                        )
            else:
                diff = grid_pos[:, None, :] - chunk_ray_pos[None, :, :]
                dist = np.linalg.norm(diff, axis=2)
                alt_delta = np.abs(chunk_alt[:, None] - chunk_ray_alt[None, :])
                dist = np.where(alt_delta <= altitude_window_km, dist, np.inf)
                min_idx = np.argmin(dist, axis=1)
                min_dist = dist[np.arange(dist.shape[0]), min_idx]

                valid = np.isfinite(min_dist) & (min_dist <= float(interpolation_radius_km))
                if np.any(valid):
                    out_count[start:end][valid] = 1
                    avg_vec = chunk_ray_tangent[min_idx[valid]]
                    e_r, e_theta, e_phi = _spherical_basis(
                        chunk_lat[valid],
                        chunk_lon[valid],
                    )
                    out_k_r[start:end][valid] = np.einsum("ij,ij->i", avg_vec, e_r)
                    out_k_t[start:end][valid] = np.einsum("ij,ij->i", avg_vec, e_theta)
                    out_k_p[start:end][valid] = np.einsum("ij,ij->i", avg_vec, e_phi)
        else:
            diff = grid_pos[:, None, :] - chunk_ray_pos[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            alt_delta = np.abs(chunk_alt[:, None] - chunk_ray_alt[None, :])

            within = (dist <= interpolation_radius_km) & (alt_delta <= altitude_window_km)
            out_count[start:end] = np.count_nonzero(within, axis=1)

            weight = np.where(
                within,
                (1.0 - dist / float(interpolation_radius_km)) ** weight_power,
                0.0,
            )
            weight_sum = weight.sum(axis=1)
            has_points = out_count[start:end] >= int(min_points)

            if np.any(has_points):
                avg_vec = weight[has_points] @ chunk_ray_tangent
                avg_vec = avg_vec / weight_sum[has_points, None]

                norm = np.linalg.norm(avg_vec, axis=1)
                good = norm > 0.0
                avg_vec[good] = avg_vec[good] / norm[good, None]

                e_r, e_theta, e_phi = _spherical_basis(
                    chunk_lat[has_points],
                    chunk_lon[has_points],
                )
                out_k_r[start:end][has_points] = np.einsum("ij,ij->i", avg_vec, e_r)
                out_k_t[start:end][has_points] = np.einsum("ij,ij->i", avg_vec, e_theta)
                out_k_p[start:end][has_points] = np.einsum("ij,ij->i", avg_vec, e_phi)

    out = xr.Dataset(
        {
            "kr": (
                ("latitude", "longitude", "altitude"),
                out_k_r.reshape(lat.size, lon.size, alt.size),
            ),
            "kt": (
                ("latitude", "longitude", "altitude"),
                out_k_t.reshape(lat.size, lon.size, alt.size),
            ),
            "kp": (
                ("latitude", "longitude", "altitude"),
                out_k_p.reshape(lat.size, lon.size, alt.size),
            ),
            "wavevector_raypoint_count": (
                ("latitude", "longitude", "altitude"),
                out_count.reshape(lat.size, lon.size, alt.size),
            ),
        },
        coords={"latitude": lat, "longitude": lon, "altitude": alt},
    )

    out["kr"].attrs["units"] = "dimensionless"
    out["kt"].attrs["units"] = "dimensionless"
    out["kp"].attrs["units"] = "dimensionless"
    out["wavevector_raypoint_count"].attrs["units"] = "count"
    out.attrs["wavevector_interpolation_radius_km"] = float(interpolation_radius_km)
    out.attrs["wavevector_mapping_mode"] = str(mapping_mode)
    out.attrs["wavevector_use_kdtree"] = bool(use_kdtree)
    out.attrs["wavevector_altitude_window_km"] = float(altitude_window_km)
    out.attrs["wavevector_min_points"] = int(min_points)
    out.attrs["wavevector_weight_power"] = float(weight_power)

    return out


def map_ray_scalar_to_grid(
    grid: xr.Dataset,
    raypaths: xr.Dataset,
    ray_var: str,
    output_name: str | None = None,
    interpolation_radius_km: float = 50.0,
    mapping_mode: str = "nearest",
    use_kdtree: bool = True,
    altitude_window_km: float | None = None,
    min_points: int = 3,
    weight_power: float = 2.0,
    chunk_size: int = 1024,
    count_name: str | None = None,
) -> xr.Dataset:
    """Map a scalar raypath variable onto the 3-D grid.

    Parameters
    ----------
    grid : xr.Dataset
        Grid dataset with coordinates ``latitude``, ``longitude``, ``altitude``.
    raypaths : xr.Dataset
        Raypath dataset containing ``ray_lat_deg``, ``ray_lon_deg``, ``ray_alt_km``.
    ray_var : str
        Name of scalar variable in ``raypaths`` to map.
    output_name : str, optional
        Variable name to use in the output dataset. Defaults to ``ray_var``.
    interpolation_radius_km : float, optional
        Neighborhood radius (km) for local averaging.
    mapping_mode : str, optional
        Mapping mode, either ``"nearest"`` or ``"weighted"``.
    use_kdtree : bool, optional
        If True, use a KD-tree for nearest-neighbor lookups when available.
    altitude_window_km : float, optional
        Vertical window (km) to limit ray candidates. Defaults to
        ``interpolation_radius_km`` when omitted.
    min_points : int, optional
        Minimum number of ray points required to assign a grid value.
    weight_power : float, optional
        Power for the distance taper ``(1 - d / r) ** weight_power``.
    chunk_size : int, optional
        Grid points processed per chunk to control memory usage.
    count_name : str, optional
        Variable name to use for raypoint counts. Defaults to
        ``"<output_name>_raypoint_count"``.

    Returns
    -------
    xr.Dataset
        Dataset with the mapped scalar and a raypoint count variable.
    """
    if interpolation_radius_km <= 0.0:
        raise ValueError("interpolation_radius_km must be > 0.")
    if min_points < 1:
        raise ValueError("min_points must be >= 1.")
    if mapping_mode not in {"nearest", "weighted"}:
        raise ValueError("mapping_mode must be 'nearest' or 'weighted'.")
    if altitude_window_km is None:
        altitude_window_km = float(interpolation_radius_km)
    if altitude_window_km <= 0.0:
        raise ValueError("altitude_window_km must be > 0.")
    if mapping_mode == "nearest" and use_kdtree and not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for KD-tree nearest-neighbor mapping. "
            "Install with the optional dependency 'pyionoseis[wavevector]'."
        )
    if ray_var not in raypaths:
        raise KeyError(f"Raypaths variable '{ray_var}' not found.")

    output_name = output_name or ray_var
    count_name = count_name or f"{output_name}_raypoint_count"

    lat = grid.coords["latitude"].values.astype(float)
    lon = grid.coords["longitude"].values.astype(float)
    alt = grid.coords["altitude"].values.astype(float)

    lat_grid, lon_grid, alt_grid = np.meshgrid(lat, lon, alt, indexing="ij")
    flat_lat = lat_grid.ravel()
    flat_lon = lon_grid.ravel()
    flat_alt = alt_grid.ravel()

    ray_lat = raypaths["ray_lat_deg"].values.astype(float)
    ray_lon = raypaths["ray_lon_deg"].values.astype(float)
    ray_alt = raypaths["ray_alt_km"].values.astype(float)
    ray_values = raypaths[ray_var].values.astype(float)

    valid = (
        np.isfinite(ray_lat)
        & np.isfinite(ray_lon)
        & np.isfinite(ray_alt)
        & np.isfinite(ray_values)
    )
    ray_lat = ray_lat[valid]
    ray_lon = ray_lon[valid]
    ray_alt = ray_alt[valid]
    ray_values = ray_values[valid]

    ray_pos = _lat_lon_alt_to_ecef(ray_lat, ray_lon, ray_alt)

    out_scalar = np.full(flat_lat.size, np.nan)
    out_count = np.zeros(flat_lat.size, dtype=int)

    for start in range(0, flat_lat.size, chunk_size):
        end = min(start + chunk_size, flat_lat.size)
        chunk_lat = flat_lat[start:end]
        chunk_lon = flat_lon[start:end]
        chunk_alt = flat_alt[start:end]

        alt_min = chunk_alt.min() - altitude_window_km
        alt_max = chunk_alt.max() + altitude_window_km
        alt_mask = (ray_alt >= alt_min) & (ray_alt <= alt_max)
        if not np.any(alt_mask):
            continue

        chunk_ray_pos = ray_pos[alt_mask]
        chunk_ray_values = ray_values[alt_mask]
        chunk_ray_alt = ray_alt[alt_mask]

        grid_pos = _lat_lon_alt_to_ecef(chunk_lat, chunk_lon, chunk_alt)

        if mapping_mode == "nearest":
            if use_kdtree:
                tree = cKDTree(chunk_ray_pos)
                dist, idx = tree.query(
                    grid_pos,
                    distance_upper_bound=float(interpolation_radius_km),
                )
                valid = np.isfinite(dist) & (dist <= float(interpolation_radius_km))
                if np.any(valid):
                    ray_alt_sel = chunk_ray_alt[idx[valid]]
                    alt_ok = np.abs(ray_alt_sel - chunk_alt[valid]) <= altitude_window_km
                    valid_idx = np.where(valid)[0][alt_ok]
                    if valid_idx.size:
                        chosen = idx[valid][alt_ok]
                        out_count[start:end][valid_idx] = 1
                        out_scalar[start:end][valid_idx] = chunk_ray_values[chosen]
            else:
                diff = grid_pos[:, None, :] - chunk_ray_pos[None, :, :]
                dist = np.linalg.norm(diff, axis=2)
                alt_delta = np.abs(chunk_alt[:, None] - chunk_ray_alt[None, :])
                dist = np.where(alt_delta <= altitude_window_km, dist, np.inf)
                min_idx = np.argmin(dist, axis=1)
                min_dist = dist[np.arange(dist.shape[0]), min_idx]

                valid = np.isfinite(min_dist) & (min_dist <= float(interpolation_radius_km))
                if np.any(valid):
                    out_count[start:end][valid] = 1
                    out_scalar[start:end][valid] = chunk_ray_values[min_idx[valid]]
        else:
            diff = grid_pos[:, None, :] - chunk_ray_pos[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            alt_delta = np.abs(chunk_alt[:, None] - chunk_ray_alt[None, :])

            within = (dist <= interpolation_radius_km) & (alt_delta <= altitude_window_km)
            out_count[start:end] = np.count_nonzero(within, axis=1)

            weight = np.where(
                within,
                (1.0 - dist / float(interpolation_radius_km)) ** weight_power,
                0.0,
            )
            weight_sum = weight.sum(axis=1)
            has_points = out_count[start:end] >= int(min_points)

            if np.any(has_points):
                weighted = weight[has_points] @ chunk_ray_values
                out_scalar[start:end][has_points] = weighted / weight_sum[has_points]

    out = xr.Dataset(
        {
            output_name: (
                ("latitude", "longitude", "altitude"),
                out_scalar.reshape(lat.size, lon.size, alt.size),
            ),
            count_name: (
                ("latitude", "longitude", "altitude"),
                out_count.reshape(lat.size, lon.size, alt.size),
            ),
        },
        coords={"latitude": lat, "longitude": lon, "altitude": alt},
    )

    if "units" in raypaths[ray_var].attrs:
        out[output_name].attrs["units"] = raypaths[ray_var].attrs["units"]
    out[count_name].attrs["units"] = "count"
    out.attrs["ray_scalar_interpolation_radius_km"] = float(interpolation_radius_km)
    out.attrs["ray_scalar_mapping_mode"] = str(mapping_mode)
    out.attrs["ray_scalar_use_kdtree"] = bool(use_kdtree)
    out.attrs["ray_scalar_altitude_window_km"] = float(altitude_window_km)
    out.attrs["ray_scalar_min_points"] = int(min_points)
    out.attrs["ray_scalar_weight_power"] = float(weight_power)

    return out
