"""The model module contains common functions and classes used to generate the spatial modes."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import toml
import xarray as xr

from pyionoseis import infraga as infraga_tools
from pyionoseis import model_io
from pyionoseis import wavevector as wavevector_tools
from pyionoseis.atmosphere import Atmosphere1D
from pyionoseis.igrf import MagneticField1D, PPIGRF_AVAILABLE
from pyionoseis.ionosphere import Ionosphere1D
from pyionoseis.model_plot import ModelPlotMixin

_log = logging.getLogger(__name__)

# Magnetic field variable names produced by MagneticField1D
_MAG_VARS = [
    "Be", "Bn", "Bu",
    "Br", "Btheta", "Bphi",
    "inclination", "declination",
    "total_field", "horizontal_intensity",
]


def _iono_profile_worker(args):
    """Compute a single ionospheric electron-density profile (thread worker)."""
    lat, lon, altitudes, time, model_name = args
    return Ionosphere1D(lat, lon, altitudes, time, model_name).ionosphere["electron_density"].values


def _magfield_profile_worker(args):
    """Compute a single magnetic-field vertical profile (thread worker)."""
    lat, lon, altitudes, time, model_name = args
    mf = MagneticField1D(lat, lon, altitudes, time, model_name)
    return {v: mf.magnetic_field[v].values for v in _MAG_VARS}

class Model3D(ModelPlotMixin):
    """Central orchestrator for 3-D ionospheric CID modelling.

    ``Model3D`` assembles a geographic 3-D grid and populates it with outputs
    from multiple 1-D physical models (atmosphere, ionosphere, magnetic field)
    evaluated at every grid column. It also manages infraGA spherical ray
    tracing and caches results to avoid redundant computation.

    Plotting methods are provided through :class:`~pyionoseis.model_plot.ModelPlotMixin`.
    Caching and IO helpers are in :mod:`pyionoseis.model_io`.

    Parameters
    ----------
    toml_file : str or Path, optional
        Path to a TOML configuration file. When supplied all model parameters
        are read from the ``[model]`` section. When omitted sensible defaults
        are used.

    Attributes
    ----------
    name : str
        Human-readable model label.
    radius : float
        Epicentral radius of the model domain in km.
    height : float
        Maximum altitude of the model domain in km.
    winds : bool
        Whether wind fields are included in the infraGA profile.
    grid_spacing : float
        Horizontal grid spacing in degrees.
    height_spacing : float
        Vertical grid spacing in km.
    atmosphere_model : str
        Name of the neutral atmosphere model (e.g. ``"msise00"``).
    ionosphere_model : str
        Name of the ionosphere model (e.g. ``"iri2020"``).
    source : EarthquakeSource or None
        Earthquake source assigned via :meth:`assign_source`.
    grid : xr.Dataset
        3-D grid Dataset with dimensions ``(latitude, longitude, altitude)``.
        Always an ``xr.Dataset``; populated progressively by the ``assign_*``
        methods.
    atmosphere : Atmosphere1D or None
        1-D atmosphere profile at the source location.
    raypaths : xr.Dataset or None
        Ray-path output from the last :meth:`trace_rays` call.
    ray_arrivals : xr.Dataset or None
        Arrival metadata from the last :meth:`trace_rays` call.
    raytrace_run_dir : str or None
        Directory where infraGA output files reside.
    lat_extent : tuple[float, float]
        ``(min_lat, max_lat)`` of the grid in degrees.
    lon_extent : tuple[float, float]
        ``(min_lon, max_lon)`` of the grid in degrees.

    Notes
    -----
    Grid population (:meth:`assign_ionosphere`, :meth:`assign_magnetic_field`)
    runs parallel 1-D model evaluations via :class:`~concurrent.futures.ThreadPoolExecutor`.
    Progress is emitted through Python's standard ``logging`` module at the
    ``INFO`` level. Enable it with::

        import logging
        logging.basicConfig(level=logging.INFO)
    """

    def __init__(self, toml_file=None):
        if toml_file:
            self.load_from_toml(toml_file)
        else:
            self.name = "No-name model"
            self.radius = 100.0
            self.height = 500.0
            self.winds = False
            self.grid_spacing = 1.0
            self.height_spacing = 20.0 
            self.source = None
            self.atmosphere = None
            self.atmosphere_model = "msise00"
            self.ionosphere_model = "iri2020"
            self.raypaths = None
            self.ray_arrivals = None
            self.raytrace_run_dir = None

    def load_from_toml(self, toml_file):
        data = toml.load(toml_file)
        model = data.get('model', {})
        self.name = model.get('name', "No-name model")
        self.radius = model.get('radius', 100.0)
        self.height = model.get('height', 500.0)
        self.winds = model.get('winds', False)
        self.atmosphere_model = model.get('atmosphere', "msise00")
        self.ionosphere_model = model.get('ionosphere', "iri2020")
        self.grid_spacing = model.get('grid_spacing', 1.0)
        units = model.get('grid_units')
        if units == "km":
            self.grid_spacing = self.grid_spacing / 111.32  # 1 degree is approximately 111.32 km at the equator
        self.height_spacing = model.get('height_spacing', 20.0) 
        

    def assign_source(self, source):
        if hasattr(self, 'source') and self.source is not None:
            print("Source already exists and will be updated.")
        self.source = source

    def assign_atmosphere(self):
        if not hasattr(self, 'source'):
            raise AttributeError("Source not assigned to the model.")
        if not hasattr(self, 'grid'):
            raise AttributeError("3D grid not created. Call make_3Dgrid() first.")
        
        self.atmosphere = Atmosphere1D(self.source.get_latitude(),
                                  self.source.get_longitude(), 
                                  self.grid.coords['altitude'].values, 
                                  self.source.get_time(), 
                                  self.atmosphere_model)

    def _build_ray_signature_payload(self, run_type, ray_params, profile):
        return {
            "signature_version": 1,
            "run_type": str(run_type),
            "source": {
                "latitude_deg": float(self.source.get_latitude()),
                "longitude_deg": float(self.source.get_longitude()),
                "time": str(self.source.get_time()),
            },
            "ray_params": ray_params,
            "profile": profile,
        }

    def _apply_raypaths_metadata(
        self,
        raypaths,
        type,
        use_accel,
        accel_used,
        command,
        signature_hash=None,
    ):
        raypaths.attrs["raytrace_backend"] = "infraga_sph"
        raypaths.attrs["raytrace_type"] = type
        raypaths.attrs["raytrace_accel_requested"] = bool(use_accel)
        raypaths.attrs["raytrace_accel_used"] = bool(accel_used)
        raypaths.attrs["source_latitude_deg"] = float(self.source.get_latitude())
        raypaths.attrs["source_longitude_deg"] = float(self.source.get_longitude())
        raypaths.attrs["source_time"] = str(self.source.get_time())
        raypaths.attrs["prof_format"] = "zcuvd"
        raypaths.attrs["winds_assumed_zero"] = True
        raypaths.attrs["infraga_command"] = " ".join(command)
        if signature_hash is not None:
            raypaths.attrs["raytrace_signature_hash"] = signature_hash

    @staticmethod
    def _azimuth_sequence(az_min, az_max, az_step):
        """Build normalized azimuth sequence in degrees."""
        if float(az_step) <= 0.0:
            raise ValueError("az_interp_step must be > 0.")
        if float(az_max) < float(az_min):
            raise ValueError("az_interp_max must be >= az_interp_min.")

        az_values = np.arange(float(az_min), float(az_max) + 0.5 * float(az_step), float(az_step))
        az_values = np.mod(az_values, 360.0)
        # Avoid duplicate endpoint (e.g., 0 and 360).
        _, unique_idx = np.unique(np.round(az_values, 12), return_index=True)
        unique_idx = np.sort(unique_idx)
        return az_values[unique_idx]

    @staticmethod
    def _forward_geodesic_deg(src_lat_deg, src_lon_deg, angular_distance_rad, azimuth_deg):
        """Project points from source by angular distance and azimuth on a sphere."""
        lat1 = np.deg2rad(float(src_lat_deg))
        lon1 = np.deg2rad(float(src_lon_deg))
        az = np.deg2rad(float(azimuth_deg))
        delta = np.asarray(angular_distance_rad, dtype=np.float64)

        sin_lat1 = np.sin(lat1)
        cos_lat1 = np.cos(lat1)
        sin_delta = np.sin(delta)
        cos_delta = np.cos(delta)

        lat2 = np.arcsin(
            np.clip(sin_lat1 * cos_delta + cos_lat1 * sin_delta * np.cos(az), -1.0, 1.0)
        )
        lon2 = lon1 + np.arctan2(
            np.sin(az) * sin_delta * cos_lat1,
            cos_delta - sin_lat1 * np.sin(lat2),
        )

        lat2_deg = np.rad2deg(lat2)
        lon2_deg = ((np.rad2deg(lon2) + 180.0) % 360.0) - 180.0
        return lat2_deg, lon2_deg

    @staticmethod
    def _great_circle_angular_distance_rad(src_lat_deg, src_lon_deg, dst_lat_deg, dst_lon_deg):
        """Great-circle angular distance between source and destination points."""
        lat1 = np.deg2rad(float(src_lat_deg))
        lon1 = np.deg2rad(float(src_lon_deg))
        lat2 = np.deg2rad(np.asarray(dst_lat_deg, dtype=np.float64))
        lon2 = np.deg2rad(np.asarray(dst_lon_deg, dtype=np.float64))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        return 2.0 * np.arctan2(np.sqrt(np.clip(a, 0.0, 1.0)), np.sqrt(np.clip(1.0 - a, 0.0, 1.0)))

    def _project_2d_raypaths_to_azimuths(self, raypaths, az_min, az_max, az_step):
        """Expand 2D raypaths into synthetic 3D by azimuthal projection."""
        az_values = self._azimuth_sequence(az_min=az_min, az_max=az_max, az_step=az_step)
        if az_values.size <= 1:
            return raypaths

        src_lat = float(self.source.get_latitude())
        src_lon = float(self.source.get_longitude())
        lat_orig = raypaths["ray_lat_deg"].values
        lon_orig = raypaths["ray_lon_deg"].values
        angular_distance = self._great_circle_angular_distance_rad(
            src_lat_deg=src_lat,
            src_lon_deg=src_lon,
            dst_lat_deg=lat_orig,
            dst_lon_deg=lon_orig,
        )

        n_points = lat_orig.size
        az_count = az_values.size
        expanded = {}
        for name in raypaths.data_vars:
            expanded[name] = np.tile(raypaths[name].values, az_count)

        expanded_lat = np.empty(n_points * az_count, dtype=np.float64)
        expanded_lon = np.empty(n_points * az_count, dtype=np.float64)
        expanded_az = np.empty(n_points * az_count, dtype=np.float64)

        for idx, azimuth in enumerate(az_values):
            start = idx * n_points
            end = start + n_points
            lat_new, lon_new = self._forward_geodesic_deg(
                src_lat_deg=src_lat,
                src_lon_deg=src_lon,
                angular_distance_rad=angular_distance,
                azimuth_deg=float(azimuth),
            )
            expanded_lat[start:end] = lat_new
            expanded_lon[start:end] = lon_new
            expanded_az[start:end] = azimuth

        expanded["ray_lat_deg"] = expanded_lat
        expanded["ray_lon_deg"] = expanded_lon
        expanded["ray_azimuth_deg"] = expanded_az

        projected = xr.Dataset(
            {name: (["ray_point"], values) for name, values in expanded.items()},
            attrs=dict(raypaths.attrs),
        )
        projected.attrs["synthetic_3d_from_2d"] = True
        projected.attrs["synthetic_3d_az_min_deg"] = float(az_min)
        projected.attrs["synthetic_3d_az_max_deg"] = float(az_max)
        projected.attrs["synthetic_3d_az_step_deg"] = float(az_step)
        projected.attrs["synthetic_3d_az_count"] = int(az_count)
        projected.attrs["synthetic_3d_note"] = (
            "Projected from single-azimuth 2D rays assuming axisymmetry."
        )
        return projected

    def load_rays(
        self,
        raypaths_file,
        arrivals_file=None,
        type=None,
        validate_signature=False,
        signature_file=None,
    ):
        """Load previously generated ray files into the model.

        Parameters
        ----------
        raypaths_file : str or Path
            Path to ``*.raypaths.dat`` produced by infraGA.
        arrivals_file : str or Path, optional
            Optional path to ``*.arrivals.dat``.
        type : str, optional
            Run type ("2d" or "3d"). If omitted, inferred from file metadata/name.
        validate_signature : bool, optional
            Validate signature against the current model source/atmosphere.
        signature_file : str or Path, optional
            Path to signature JSON sidecar. Defaults to inferred sibling file.
        """
        raypaths_path = Path(raypaths_file)
        if not raypaths_path.exists():
            raise FileNotFoundError(f"Raypaths file does not exist: {raypaths_path}")

        run_type = type
        if run_type is None:
            if "infraga_2d_" in raypaths_path.name:
                run_type = "2d"
            elif "infraga_3d_" in raypaths_path.name:
                run_type = "3d"
            elif "infraga_2d_sph" in raypaths_path.name:
                run_type = "2d"
            else:
                run_type = "3d"
        if run_type not in ["2d", "3d"]:
            raise ValueError("type must be '2d' or '3d'.")

        arrivals_path = Path(arrivals_file) if arrivals_file is not None else Path(
            str(raypaths_path).replace(".raypaths.dat", ".arrivals.dat")
        )
        if arrivals_file is None and arrivals_path == raypaths_path:
            arrivals_path = Path(str(raypaths_path) + ".arrivals.dat")

        if validate_signature:
            sig_path = (
                Path(signature_file)
                if signature_file is not None
                else model_io.signature_path_for_raypaths(raypaths_path)
            )
            if not sig_path.exists():
                raise FileNotFoundError(f"Signature file does not exist: {sig_path}")
            import json
            with sig_path.open("r", encoding="utf-8") as fh:
                raw_sig_data = json.load(fh)
            expected_payload = model_io.normalize_signature_payload(raw_sig_data)
            if self.source is None:
                raise AttributeError("Source must be assigned for signature validation.")
            if self.atmosphere is None:
                if not hasattr(self, "grid"):
                    raise AttributeError(
                        "3D grid not created. Call make_3Dgrid() first for signature validation."
                    )
                self.assign_atmosphere()
            actual_payload = self._build_ray_signature_payload(
                run_type=run_type,
                ray_params=expected_payload.get("ray_params", {}),
                profile={
                    "prof_format": expected_payload.get("profile", {}).get("prof_format", "zcuvd"),
                    "winds_assumed_zero": bool(
                        expected_payload.get("profile", {}).get("winds_assumed_zero", True)
                    ),
                    "altitude_sha256": model_io.array_sha256(
                        self.atmosphere.atmosphere["altitude"].values
                    ),
                    "velocity_sha256": model_io.array_sha256(
                        self.atmosphere.atmosphere["velocity"].values
                    ),
                    "density_sha256": model_io.array_sha256(
                        self.atmosphere.atmosphere["density"].values
                    ),
                },
            )
            if actual_payload != expected_payload:
                raise ValueError(
                    "Ray signature mismatch: files do not match current model source/atmosphere."
                )

        self.raypaths = infraga_tools.parse_sph_raypaths(raypaths_path, run_type=run_type)
        self.ray_arrivals = (
            infraga_tools.parse_arrivals(arrivals_path)
            if arrivals_path.exists()
            else xr.Dataset()
        )
        self.raytrace_run_dir = str(raypaths_path.parent)
        return self.raypaths

    def _warn_az_interp_approximation(self):
        """Emit runtime warnings when az_interp synthetic-3D remapping is active."""
        warnings.warn(
            "az_interp=True remaps a single-azimuth 2D ray solution into synthetic 3D. "
            "This assumes an axisymmetric medium around the source. In non-axisymmetric "
            "atmosphere/wind fields, remapped travel times and amplitudes are not physically valid.",
            RuntimeWarning,
            stacklevel=3,
        )
        warnings.warn(
            "Ray tracing currently writes zero winds into infraGA profile (u=v=0). "
            "If your model includes winds or azimuth-dependent structure, synthetic 3D remapping "
            "is an approximation intended for visualization/sensitivity only.",
            RuntimeWarning,
            stacklevel=3,
        )

    def trace_rays(
        self,
        type="3d",
        output_dir=None,
        keep_files=True,
        reuse_existing=True,
        force_recompute=False,
        cache_id=None,
        executable=None,
        use_accel=False,
        cpu_cnt=None,
        az_interp=False,
        az_interp_min=0.0,
        az_interp_max=360.0,
        az_interp_step=1.0,
        azimuth_deg=0.0,
        az_min=0.0,
        az_max=360.0,
        az_step=10.0,
        incl_min=45.0,
        incl_max=90.0,
        incl_step=1.0,
        bounces=0,
        max_rng_km=5000.0,
        frequency_hz=0.005,
    ):
        """Trace rays with infraGA using spherical propagation only.

        Parameters
        ----------
        type : str, optional
            Run type selector. Supported values:
            - "3d": spherical azimuth sweep
            - "2d": spherical single-azimuth mode (north by default)
        output_dir : str or Path, optional
            Directory where infraGA files are written.
        keep_files : bool, optional
            Keep generated files on disk. Defaults to True.
        reuse_existing : bool, optional
            If True and ``output_dir`` is set, reuse matching cached results.
        force_recompute : bool, optional
            If True, bypass cache checks and always run infraGA.
        cache_id : str, optional
            Optional cache token to isolate output files in one ``output_dir``.
        executable : str, optional
            Optional explicit executable name/path for infraGA CLI.
        use_accel : bool, optional
            Enable infraGA accelerated spherical backend selection via
            ``--cpu-cnt``. Defaults to False.
        cpu_cnt : int, optional
            CPU count passed to infraGA when acceleration is enabled. If None
            and ``use_accel`` is True, a default is inferred from available
            CPUs and ray launch count.
        az_interp : bool, optional
            When ``type="2d"``, project the single-azimuth 2D result into
            multiple azimuths to create a synthetic 3D ray cloud.
        az_interp_min, az_interp_max, az_interp_step : float, optional
            Azimuth sampling used when ``az_interp=True``. Values are in
            degrees and endpoint duplicates (e.g., 0 and 360) are removed.

        Notes
        -----
        This method stores raw ray products only and does not interpolate to
        the model grid yet.

        ``az_interp=True`` is a geometric remapping convenience. It assumes
        axisymmetry around the source, so remapped travel times/amplitudes are
        approximate in non-axisymmetric atmosphere/wind fields.
        """
        if type not in ["2d", "3d"]:
            raise ValueError("type must be '2d' or '3d'.")
        if type != "2d" and az_interp:
            raise ValueError("az_interp is only supported when type='2d'.")

        if self.source is None:
            raise AttributeError("Source not assigned to the model.")

        if not hasattr(self, "grid"):
            raise AttributeError("3D grid not created. Call make_3Dgrid() first.")

        if self.atmosphere is None:
            self.assign_atmosphere()

        if output_dir is not None:
            run_dir = Path(output_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = Path(tempfile.mkdtemp(prefix="pyionoseis_infraga_"))

        cmd_prefix = infraga_tools.resolve_infraga_command(executable=executable)

        effective_azimuth = azimuth_deg
        if type == "2d":
            # 2D behavior is represented as a single spherical azimuth run.
            effective_azimuth = 0.0

        effective_cpu_cnt = None
        accel_used = False
        if use_accel:
            if infraga_tools.accel_sph_available():
                if cpu_cnt is not None:
                    effective_cpu_cnt = int(cpu_cnt)
                    if effective_cpu_cnt < 2:
                        raise ValueError("cpu_cnt must be >= 2 when use_accel=True.")
                else:
                    # Smart default: scale to launch count, bounded by machine cores.
                    available = os.cpu_count() or 1
                    if type == "2d":
                        az_count = 1
                    else:
                        az_count = int(
                            np.floor((float(az_max) - float(az_min)) / float(az_step))
                        ) + 1
                        az_count = max(1, az_count)
                    incl_count = int(
                        np.floor((float(incl_max) - float(incl_min)) / float(incl_step))
                    ) + 1
                    incl_count = max(1, incl_count)
                    launch_count = az_count * incl_count

                    if available >= 2:
                        effective_cpu_cnt = min(available, max(2, launch_count))
                accel_used = effective_cpu_cnt is not None
            else:
                warnings.warn(
                    "use_accel=True requested, but accelerated spherical backend "
                    "(`infraga-accel-sph` + `mpirun`) is unavailable. "
                    "Falling back to non-accelerated spherical run.",
                    RuntimeWarning,
                )

        ray_params = {
            "azimuth_deg": float(effective_azimuth),
            "az_min": float(az_min),
            "az_max": float(az_max),
            "az_step": float(az_step),
            "incl_min": float(incl_min),
            "incl_max": float(incl_max),
            "incl_step": float(incl_step),
            "bounces": int(bounces),
            "max_rng_km": float(max_rng_km),
            "frequency_hz": float(frequency_hz),
            "use_accel": bool(use_accel),
            "cpu_cnt_effective": (
                int(effective_cpu_cnt) if effective_cpu_cnt is not None else None
            ),
            "accel_used": bool(accel_used),
        }
        profile_info = {
            "prof_format": "zcuvd",
            "winds_assumed_zero": True,
            "altitude_sha256": model_io.array_sha256(self.atmosphere.atmosphere["altitude"].values),
            "velocity_sha256": model_io.array_sha256(self.atmosphere.atmosphere["velocity"].values),
            "density_sha256": model_io.array_sha256(self.atmosphere.atmosphere["density"].values),
        }
        signature_payload = self._build_ray_signature_payload(
            run_type=type, ray_params=ray_params, profile=profile_info
        )
        signature_hash = model_io.canonical_json_hash(signature_payload)

        profile_file = run_dir / "infraga_input_profile.zcuvd"
        if output_dir is not None:
            output_token = model_io.cache_token(signature_hash=signature_hash, cache_id=cache_id)
            output_prefix = run_dir / f"infraga_{type}_sph_{output_token}"
        else:
            output_prefix = run_dir / f"infraga_{type}_sph"

        raypaths_file = Path(str(output_prefix) + ".raypaths.dat")
        arrivals_file = Path(str(output_prefix) + ".arrivals.dat")
        signature_file = model_io.signature_path_for_output_prefix(output_prefix)

        command = infraga_tools.build_sph_prop_command(
            cmd_prefix=cmd_prefix,
            atmo_file=profile_file,
            output_id=output_prefix,
            src_lat=float(self.source.get_latitude()),
            src_lon=float(self.source.get_longitude()),
            run_type=type,
            azimuth_deg=float(effective_azimuth),
            az_min=float(az_min),
            az_max=float(az_max),
            az_step=float(az_step),
            incl_min=float(incl_min),
            incl_max=float(incl_max),
            incl_step=float(incl_step),
            bounces=int(bounces),
            max_rng_km=float(max_rng_km),
            frequency_hz=float(frequency_hz),
            cpu_cnt=effective_cpu_cnt,
            prof_format="zcuvd",
        )

        if (
            output_dir is not None
            and reuse_existing
            and not force_recompute
            and raypaths_file.exists()
            and signature_file.exists()
        ):
            import json
            with signature_file.open("r", encoding="utf-8") as fh:
                saved_signature_raw = json.load(fh)
            saved_signature_payload = model_io.normalize_signature_payload(saved_signature_raw)
            if saved_signature_payload == signature_payload:
                self.load_rays(raypaths_file=raypaths_file, arrivals_file=arrivals_file, type=type)
                self._apply_raypaths_metadata(
                    self.raypaths,
                    type=type,
                    use_accel=use_accel,
                    accel_used=accel_used,
                    command=command,
                    signature_hash=signature_hash,
                )
                self.raypaths.attrs["raytrace_loaded_from_cache"] = True
                self.raypaths.attrs["raytrace_signature_file"] = str(signature_file)
                self.raytrace_run_dir = str(run_dir)
                if type == "2d" and az_interp:
                    self._warn_az_interp_approximation()
                    self.raypaths = self._project_2d_raypaths_to_azimuths(
                        self.raypaths,
                        az_min=az_interp_min,
                        az_max=az_interp_max,
                        az_step=az_interp_step,
                    )
                return self.raypaths

        infraga_tools.write_zcuvd_profile(self.atmosphere.atmosphere, profile_file)
        infraga_tools.run_sph_trace(command)

        self.load_rays(
            raypaths_file=raypaths_file,
            arrivals_file=arrivals_file,
            type=type,
        )

        signature_payload_to_write = {
            "signature_hash": signature_hash,
            "signature": signature_payload,
        }
        import json
        with signature_file.open("w", encoding="utf-8") as fh:
            json.dump(signature_payload_to_write, fh, indent=2, sort_keys=True)

        self._apply_raypaths_metadata(
            self.raypaths,
            type=type,
            use_accel=use_accel,
            accel_used=accel_used,
            command=command,
            signature_hash=signature_hash,
        )
        self.raypaths.attrs["raytrace_loaded_from_cache"] = False
        self.raypaths.attrs["raytrace_signature_file"] = str(signature_file)
        self.raytrace_run_dir = str(run_dir)

        if type == "2d" and az_interp:
            self._warn_az_interp_approximation()
            self.raypaths = self._project_2d_raypaths_to_azimuths(
                self.raypaths,
                az_min=az_interp_min,
                az_max=az_interp_max,
                az_step=az_interp_step,
            )

        if not keep_files and output_dir is None:
            shutil.rmtree(run_dir, ignore_errors=True)
            self.raytrace_run_dir = None

        return self.raypaths

    def assign_ionosphere(self, ionosphere_model="iri2020", max_workers=None):
        """Assign ionospheric electron density to all grid points.

        Computes IRI2020 electron density profiles in parallel across every
        lat/lon column, then stores results in ``self.grid``.

        Parameters
        ----------
        ionosphere_model : str
            Model to use. Currently supports ``"iri2020"`` (default).
        max_workers : int, optional
            Maximum number of threads. Defaults to ``os.cpu_count()``.

        Raises
        ------
        AttributeError
            If source is not assigned or grid not created.
        ValueError
            If *ionosphere_model* is not supported.
        """
        if not hasattr(self, "source"):
            raise AttributeError("Source not assigned to the model.")
        if not hasattr(self, "grid"):
            raise AttributeError("3D grid not created. Call make_3Dgrid() first.")

        supported_models = ["iri2020"]
        if ionosphere_model not in supported_models:
            raise ValueError(
                f"Unsupported ionosphere model '{ionosphere_model}'. "
                f"Supported models: {supported_models}"
            )

        latitudes = self.grid.coords["latitude"].values
        longitudes = self.grid.coords["longitude"].values
        altitudes = self.grid.coords["altitude"].values
        time = self.source.get_time()

        arg_list = [
            (float(lat), float(lon), altitudes, time, ionosphere_model)
            for lat in latitudes
            for lon in longitudes
        ]

        _log.info(
            "Computing ionospheric electron density (%s): %d profiles "
            "(%d lat × %d lon × %d alt)",
            ionosphere_model, len(arg_list),
            len(latitudes), len(longitudes), len(altitudes),
        )

        results = [None] * len(arg_list)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_iono_profile_worker, a): k for k, a in enumerate(arg_list)}
            for fut in as_completed(futures):
                k = futures[fut]
                try:
                    results[k] = fut.result()
                except Exception as exc:
                    lat_k, lon_k = arg_list[k][0], arg_list[k][1]
                    _log.warning(
                        "Ionosphere failed at lat=%.2f lon=%.2f: %s", lat_k, lon_k, exc
                    )
                    results[k] = np.full(len(altitudes), np.nan)

        electron_density_3d = np.empty((len(latitudes), len(longitudes), len(altitudes)))
        for k, ne in enumerate(results):
            i, j = divmod(k, len(longitudes))
            electron_density_3d[i, j, :] = ne

        self.grid["electron_density"] = (
            ["latitude", "longitude", "altitude"],
            electron_density_3d,
        )
        self.grid["electron_density"].attrs["units"] = "m^-3"
        self.grid.attrs["ionosphere_model"] = ionosphere_model
        self.grid.attrs["electron_density_description"] = (
            f"Electron density from {ionosphere_model} model"
        )
        self.ionosphere_model = ionosphere_model
        _log.info("Ionospheric computation completed.")
        
    def assign_magnetic_field(self, magnetic_field_model="igrf", max_workers=None):
        """Assign magnetic field parameters to all grid points.

        Computes IGRF magnetic field components and derived parameters in
        parallel across every lat/lon column, then stores results in
        ``self.grid``.

        Parameters
        ----------
        magnetic_field_model : str
            Model to use. Currently only ``"igrf"`` (default) is supported.
        max_workers : int, optional
            Maximum number of threads. Defaults to ``os.cpu_count()``.

        Raises
        ------
        AttributeError
            If source is not assigned or grid not created.
        ImportError
            If ``ppigrf`` is not installed.
        ValueError
            If *magnetic_field_model* is not supported.

        Notes
        -----
        Variables added to ``self.grid``:

        Geodetic components (nT): ``Be``, ``Bn``, ``Bu``
        Geocentric components (nT): ``Br``, ``Btheta``, ``Bphi``
        Derived (degrees): ``inclination``, ``declination``
        Derived (nT): ``total_field``, ``horizontal_intensity``
        """
        if not hasattr(self, "source"):
            raise AttributeError("Source not assigned to the model.")
        if not hasattr(self, "grid"):
            raise AttributeError("3D grid not created. Call make_3Dgrid() first.")
        if magnetic_field_model.lower() != "igrf":
            raise ValueError(f"Unsupported magnetic field model: {magnetic_field_model}")
        if not PPIGRF_AVAILABLE:
            raise ImportError(
                "ppigrf is not available. Install with: pip install ppigrf"
            )

        latitudes = self.grid.coords["latitude"].values
        longitudes = self.grid.coords["longitude"].values
        altitudes = self.grid.coords["altitude"].values

        arg_list = [
            (float(lat), float(lon), altitudes, self.source.get_time(), magnetic_field_model)
            for lat in latitudes
            for lon in longitudes
        ]

        _log.info(
            "Computing magnetic field (%s): %d profiles (%d lat × %d lon × %d alt)",
            magnetic_field_model, len(arg_list),
            len(latitudes), len(longitudes), len(altitudes),
        )

        results = [None] * len(arg_list)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_magfield_profile_worker, a): k for k, a in enumerate(arg_list)}
            for fut in as_completed(futures):
                k = futures[fut]
                try:
                    results[k] = fut.result()
                except Exception as exc:
                    lat_k, lon_k = arg_list[k][0], arg_list[k][1]
                    _log.warning(
                        "Magnetic field failed at lat=%.2f lon=%.2f: %s", lat_k, lon_k, exc
                    )
                    results[k] = {v: np.full(len(altitudes), np.nan) for v in _MAG_VARS}

        arrays = {
            v: np.full((len(latitudes), len(longitudes), len(altitudes)), np.nan)
            for v in _MAG_VARS
        }
        for k, profile in enumerate(results):
            i, j = divmod(k, len(longitudes))
            for v in _MAG_VARS:
                arrays[v][i, j, :] = profile[v]

        dims = ["latitude", "longitude", "altitude"]
        for v in _MAG_VARS:
            self.grid[v] = (dims, arrays[v])

        _nT_vars = [
            "Be", "Bn", "Bu", "Br", "Btheta", "Bphi",
            "total_field", "horizontal_intensity",
        ]
        for v in _nT_vars:
            self.grid[v].attrs["units"] = "nT"
        for v in ["inclination", "declination"]:
            self.grid[v].attrs["units"] = "degrees"

        self.grid.attrs["magnetic_field_model"] = magnetic_field_model
        self.magnetic_field_model = magnetic_field_model
        _log.info("Magnetic field computation completed.")

    def assign_wavevector(
        self,
        interpolation_radius_km=50.0,
        mapping_mode="nearest",
        use_kdtree=True,
        altitude_window_km=None,
        smoothing_radius_km=20.0,
        min_points=3,
        weight_power=2.0,
        chunk_size=1024,
        turning_point=True,
        keep_ray_wavevectors=False,
    ):
        """Assign geometry-based wavevector components to the 3-D grid.

        Parameters
        ----------
        interpolation_radius_km : float, optional
            Neighborhood radius (km) used when mapping ray directions to the grid.
        mapping_mode : str, optional
            Mapping mode, either ``"nearest"`` or ``"weighted"``.
        use_kdtree : bool, optional
            If True, use a KD-tree for nearest-neighbor lookups when available.
        altitude_window_km : float, optional
            Vertical window (km) to limit ray candidates. Defaults to
            ``interpolation_radius_km`` when omitted.
        smoothing_radius_km : float, optional
            Smoothing radius (km) applied along rays before differentiation.
        min_points : int, optional
            Minimum number of ray points required to assign a grid cell.
        weight_power : float, optional
            Power for distance taper ``(1 - d / r) ** weight_power``.
        chunk_size : int, optional
            Grid points processed per chunk to limit memory usage.
        turning_point : bool, optional
            If True, enforce ``k_r = 0`` at detected turning points.
        keep_ray_wavevectors : bool, optional
            If True, retain cached ray wavevectors for reuse.

        Raises
        ------
        AttributeError
            If raypaths are missing or the 3-D grid is not created.
        """
        if self.raypaths is None:
            raise AttributeError("Raypaths not available. Call trace_rays() first.")
        if not hasattr(self, "grid"):
            raise AttributeError("3D grid not created. Call make_3Dgrid() first.")

        cache_key = (float(smoothing_radius_km), bool(turning_point))
        ray_wavevectors = None
        if hasattr(self, "_ray_wavevectors_cache"):
            cached = self._ray_wavevectors_cache
            if cached is not None and cached.get("key") == cache_key:
                ray_wavevectors = cached.get("data")

        if ray_wavevectors is None:
            ray_wavevectors = wavevector_tools.compute_ray_wavevectors(
                self.raypaths,
                smoothing_radius_km=float(smoothing_radius_km),
                turning_point=bool(turning_point),
            )
            self._ray_wavevectors_cache = {"key": cache_key, "data": ray_wavevectors}

        grid_wavevectors = wavevector_tools.map_wavevector_to_grid(
            self.grid,
            ray_wavevectors,
            interpolation_radius_km=float(interpolation_radius_km),
            mapping_mode=str(mapping_mode),
            use_kdtree=bool(use_kdtree),
            altitude_window_km=altitude_window_km,
            min_points=int(min_points),
            weight_power=float(weight_power),
            chunk_size=int(chunk_size),
        )

        if not keep_ray_wavevectors:
            self._ray_wavevectors_cache = None

        for var in ["kr", "kt", "kp", "wavevector_raypoint_count"]:
            self.grid[var] = grid_wavevectors[var]

        self.grid.attrs["wavevector_method"] = "geometry_tangent"
        self.grid.attrs["wavevector_interpolation_radius_km"] = float(
            interpolation_radius_km
        )
        self.grid.attrs["wavevector_mapping_mode"] = str(mapping_mode)
        self.grid.attrs["wavevector_use_kdtree"] = bool(use_kdtree)
        if altitude_window_km is None:
            altitude_window_km = float(interpolation_radius_km)
        self.grid.attrs["wavevector_altitude_window_km"] = float(altitude_window_km)
        self.grid.attrs["wavevector_smoothing_radius_km"] = float(smoothing_radius_km)
        self.grid.attrs["wavevector_min_points"] = int(min_points)
        self.grid.attrs["wavevector_turning_point_enforced"] = bool(turning_point)
        _log.info("Wavevector assignment completed.")


        
    def __str__(self):
        source_info = f"latitude = {self.source.get_latitude():.2f} (deg), longitude={self.source.get_longitude():.2f} (deg)" if hasattr(self, 'source') else "None"
        lat_extent_info = f"lat_extent = ({self.lat_extent[0]:.2f}, {self.lat_extent[1]:.2f})" if hasattr(self, 'lat_extent') else ""
        lon_extent_info = f"lon_extent = ({self.lon_extent[0]:.2f}, {self.lon_extent[1]:.2f})" if hasattr(self, 'lon_extent') else ""
        atmosphere_info = f"atmosphere_model = {self.atmosphere_model}" if hasattr(self, 'atmosphere_model') else ""
        ionosphere_info = f"ionosphere_model = {self.ionosphere_model}" if hasattr(self, 'ionosphere_model') else ""
        
        base_info = (f"Model3D: name = {self.name}\n radius = {self.radius} (km)\n height = {self.height} (km)\n"
                     f" winds = {self.winds}\n {atmosphere_info}\n {ionosphere_info}\n grid_spacing = {self.grid_spacing} (deg)\n height_spacing = {self.height_spacing} (km)\n"
                     f" source: {source_info}\n {lat_extent_info}\n {lon_extent_info}")
        
        return base_info

    def print_info(self):
        """
        Print comprehensive information about the Model3D object.
        
        This method provides detailed information about:
        - Basic model parameters
        - Source configuration
        - Grid dimensions and extent
        - Available data variables
        - Model states (atmosphere, ionosphere, magnetic field)
        """
        print("=" * 80)
        print("MODEL3D INFORMATION")
        print("=" * 80)
        
        # Basic Model Information
        print(f"Model Name: {self.name}")
        print(f"Model Radius: {self.radius} km")
        print(f"Model Height: {self.height} km")
        print(f"Include Winds: {self.winds}")
        print(f"Grid Spacing: {self.grid_spacing} degrees")
        print(f"Height Spacing: {self.height_spacing} km")
        
        # Source Information
        print("\nSOURCE INFORMATION:")
        if hasattr(self, 'source') and self.source is not None:
            print(f"  Latitude: {self.source.get_latitude():.4f}°")
            print(f"  Longitude: {self.source.get_longitude():.4f}°")
            print(f"  Time: {self.source.get_time()}")
            if hasattr(self.source, 'get_depth'):
                print(f"  Depth: {self.source.get_depth()} km")
        else:
            print("  No source assigned")
        
        # Grid Information
        print("\nGRID INFORMATION:")
        if hasattr(self, 'grid') and self.grid is not None:
            if hasattr(self, 'lat_extent'):
                print(f"  Latitude extent: {self.lat_extent[0]:.4f}° to {self.lat_extent[1]:.4f}°")
            if hasattr(self, 'lon_extent'):
                print(f"  Longitude extent: {self.lon_extent[0]:.4f}° to {self.lon_extent[1]:.4f}°")
            
            # Grid dimensions
            if hasattr(self.grid, 'sizes'):
                # This is an xarray Dataset/DataArray with sizes
                grid_shape = self.grid.sizes
                print(f"  Grid dimensions: {dict(grid_shape)}")
                total_points = 1
                for dim, size in grid_shape.items():
                    total_points *= size
                print(f"  Total grid points: {total_points:,}")
            elif hasattr(self.grid, 'shape'):
                # This is a numpy array or similar
                grid_shape = self.grid.shape
                print(f"  Grid shape: {grid_shape}")
                total_points = 1
                for dim in grid_shape:
                    total_points *= dim
                print(f"  Total grid points: {total_points:,}")
            else:
                print("  Grid shape: Unknown")
            
            # Coordinate ranges
            if hasattr(self.grid, 'coords'):
                coords = self.grid.coords
                if 'latitude' in coords:
                    lat_vals = coords['latitude'].values
                    print(f"  Latitude range: {lat_vals.min():.4f}° to {lat_vals.max():.4f}° ({len(lat_vals)} points)")
                if 'longitude' in coords:
                    lon_vals = coords['longitude'].values
                    print(f"  Longitude range: {lon_vals.min():.4f}° to {lon_vals.max():.4f}° ({len(lon_vals)} points)")
                if 'altitude' in coords:
                    alt_vals = coords['altitude'].values
                    print(f"  Altitude range: {alt_vals.min():.1f} to {alt_vals.max():.1f} km ({len(alt_vals)} points)")
        else:
            print("  No grid created yet")
        
        # Available Data Variables
        print("\nAVAILABLE DATA VARIABLES:")
        if hasattr(self, 'grid') and self.grid is not None and hasattr(self.grid, 'data_vars'):
            if len(self.grid.data_vars) > 0:
                # Group variables by type
                atmospheric_vars = []
                ionospheric_vars = []
                magnetic_geodetic_vars = []
                magnetic_geocentric_vars = []
                magnetic_derived_vars = []
                other_vars = []
                
                for var in self.grid.data_vars:
                    if var in ['density', 'pressure', 'temperature', 'velocity']:
                        atmospheric_vars.append(var)
                    elif var in ['electron_density']:
                        ionospheric_vars.append(var)
                    elif var in ['Be', 'Bn', 'Bu']:
                        magnetic_geodetic_vars.append(var)
                    elif var in ['Br', 'Btheta', 'Bphi']:
                        magnetic_geocentric_vars.append(var)
                    elif var in ['inclination', 'declination', 'total_field', 'horizontal_intensity']:
                        magnetic_derived_vars.append(var)
                    else:
                        other_vars.append(var)
                
                if atmospheric_vars:
                    print(f"  Atmospheric variables: {', '.join(atmospheric_vars)}")
                if ionospheric_vars:
                    print(f"  Ionospheric variables: {', '.join(ionospheric_vars)}")
                if magnetic_geodetic_vars:
                    print(f"  Magnetic field (geodetic): {', '.join(magnetic_geodetic_vars)}")
                if magnetic_geocentric_vars:
                    print(f"  Magnetic field (geocentric): {', '.join(magnetic_geocentric_vars)}")
                if magnetic_derived_vars:
                    print(f"  Magnetic field (derived): {', '.join(magnetic_derived_vars)}")
                if other_vars:
                    print(f"  Other variables: {', '.join(other_vars)}")
                
                print(f"  Total variables: {len(self.grid.data_vars)}")
                
                # Show variable ranges for numeric data
                print("\n  Variable Ranges:")
                for var in sorted(self.grid.data_vars):
                    if var != 'grid_points':  # Skip the grid_points variable
                        try:
                            data = self.grid[var]
                            if data.dtype.kind in ['f', 'i']:  # float or integer
                                min_val = float(data.min().values)
                                max_val = float(data.max().values)
                                units = ""
                                if 'nT' in str(data.attrs.get('units', '')) or var in ['Be', 'Bn', 'Bu', 'Br', 'Btheta', 'Bphi', 'total_field', 'horizontal_intensity']:
                                    units = " nT"
                                elif var in ['inclination', 'declination']:
                                    units = "°"
                                elif var in ['density']:
                                    units = " kg/m³"
                                elif var in ['pressure']:
                                    units = " Pa"
                                elif var in ['temperature']:
                                    units = " K"
                                elif var in ['electron_density']:
                                    units = " /m³"
                                print(f"    {var}: {min_val:.2f} to {max_val:.2f}{units}")
                        except Exception:
                            print(f"    {var}: [data present]")
            else:
                print("  No data variables computed yet")
        else:
            print("  No data variables (grid not created)")
        
        # Model States
        print("\nMODEL STATES:")
        
        # Atmosphere model
        if hasattr(self, 'atmosphere_model'):
            print(f"  Atmosphere model: {self.atmosphere_model}")
            if hasattr(self, 'grid') and self.grid is not None and hasattr(self.grid, 'data_vars'):
                atm_vars = [v for v in self.grid.data_vars if v in ['density', 'pressure', 'temperature', 'velocity']]
                print(f"    Status: {'Computed' if atm_vars else 'Not computed'}")
                if atm_vars:
                    print(f"    Variables: {', '.join(atm_vars)}")
        
        # Ionosphere model
        if hasattr(self, 'ionosphere_model'):
            print(f"  Ionosphere model: {self.ionosphere_model}")
            if hasattr(self, 'grid') and self.grid is not None and hasattr(self.grid, 'data_vars'):
                iono_computed = 'electron_density' in self.grid.data_vars
                print(f"    Status: {'Computed' if iono_computed else 'Not computed'}")
        
        # Magnetic field model
        if hasattr(self, 'magnetic_field_model'):
            print(f"  Magnetic field model: {self.magnetic_field_model}")
            if hasattr(self, 'grid') and self.grid is not None and hasattr(self.grid, 'data_vars'):
                mag_vars = [v for v in self.grid.data_vars if v in ['Be', 'Bn', 'Bu', 'Br', 'Btheta', 'Bphi', 'inclination', 'declination']]
                print(f"    Status: {'Computed' if mag_vars else 'Not computed'}")
                if mag_vars:
                    geodetic = [v for v in mag_vars if v in ['Be', 'Bn', 'Bu']]
                    geocentric = [v for v in mag_vars if v in ['Br', 'Btheta', 'Bphi']]
                    derived = [v for v in mag_vars if v in ['inclination', 'declination']]
                    if geodetic:
                        print(f"    Geodetic components: {', '.join(geodetic)}")
                    if geocentric:
                        print(f"    Geocentric components: {', '.join(geocentric)}")
                    if derived:
                        print(f"    Derived parameters: {', '.join(derived)}")
        
        # Grid attributes
        if hasattr(self, 'grid') and self.grid is not None and hasattr(self.grid, 'attrs') and self.grid.attrs:
            print("\nGRID ATTRIBUTES:")
            for key, value in self.grid.attrs.items():
                if not key.endswith('_description'):  # Skip long description attributes
                    print(f"  {key}: {value}")
        
        print("=" * 80)

    def make_3Dgrid(self):
        """
        Create a 3D grid with the given source and model parameters.
        """
        if not hasattr(self, 'source'):
            raise AttributeError("Source not assigned to the model.")

        lat0 = self.source.get_latitude()
        lon0 = self.source.get_longitude()
        # Convert radius from kilometers to degrees (approximation)
        radius_in_degrees = self.radius / 111.32  # 1 degree is approximately 111.32 km at the equator

        # Create latitude and longitude arrays
        latitudes = np.arange(lat0 - radius_in_degrees, lat0 + radius_in_degrees, self.grid_spacing)
        longitudes = np.arange(lon0 - radius_in_degrees, lon0 + radius_in_degrees, self.grid_spacing)
        altitudes = np.arange(0, self.height, self.height_spacing)

        self.grid = xr.Dataset(
            {
                "grid_points": (
                    ["latitude", "longitude", "altitude"],
                    np.zeros((len(latitudes), len(longitudes), len(altitudes))),
                )
            },
            coords={
                "latitude": latitudes,
                "longitude": longitudes,
                "altitude": altitudes,
            },
        )

        self.lat_extent = (latitudes[0], latitudes[-1])
        self.lon_extent = (longitudes[0], longitudes[-1])

