"""The model module contains common functions and classes used to generate the spatial modes."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import warnings
import json
from pathlib import Path

import numpy as np
import toml
import xarray as xr

from pyionoseis import infraga as infraga_tools
from pyionoseis import continuity as continuity_tools
from pyionoseis import continuity_orchestrator
from pyionoseis import grid_enrichment_orchestrator
from pyionoseis import model_io
from pyionoseis import ray_tracing_orchestrator
from pyionoseis import tec as tec_tools
from pyionoseis import tec_io
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
            self.continuity_t0_s = None
            self.continuity_tmax_s = None
            self.continuity_dt_s = None
            self.continuity = None
            self.tec_elevation_mask_deg = 20.0
            self.tec_output_dt_s = None
            self.tec_ipp_altitude_km = 350.0
            self.tec_receiver_csv = None
            self.tec_receiver_listesta = None
            self.tec_orbit_h5 = None
            self.tec_orbit_pos = None
            self.tec_receiver_code = None
            self.tec_sat_id = None
            self.tec_constellation = None
            self.tec_prn = None
            self.tec_receiver_format = None
            self.tec_orbit_format = None
            self.tec_satpos_root = None
            self.tec_satpos_date = None
            self.tec_sat_number = None
            self.tec_sat_mapping_file = None
            self.tec_start_offset_s = 0.0

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

        continuity = data.get("continuity", {})
        self.continuity_t0_s = continuity.get("t0_s")
        self.continuity_tmax_s = continuity.get("tmax_s")
        self.continuity_dt_s = continuity.get("dt_s")
        self.continuity = None

        tec = data.get("tec", {})
        self.tec_elevation_mask_deg = tec.get("elevation_mask_deg", 20.0)
        self.tec_output_dt_s = tec.get("output_dt_s")
        self.tec_ipp_altitude_km = tec.get("ipp_altitude_km", 350.0)
        self.tec_receiver_format = tec.get("receiver_format")
        self.tec_orbit_format = tec.get("orbit_format")
        self.tec_receiver_csv = tec.get("receiver_csv")
        self.tec_receiver_listesta = tec.get("receiver_listesta")
        self.tec_orbit_h5 = tec.get("orbit_h5")
        self.tec_orbit_pos = tec.get("orbit_pos")
        self.tec_receiver_code = tec.get("receiver_code")
        self.tec_sat_id = tec.get("sat_id")
        self.tec_constellation = tec.get("constellation")
        self.tec_prn = tec.get("prn")
        self.tec_satpos_root = tec.get("satpos_root")
        self.tec_satpos_date = tec.get("satpos_date")
        self.tec_sat_number = tec.get("sat_number")
        self.tec_sat_mapping_file = tec.get("sat_mapping_file")
        self.tec_start_offset_s = tec.get("start_offset_s", 0.0)
        

    def assign_source(self, source):
        if hasattr(self, 'source') and self.source is not None:
            _log.info("Source already exists and will be updated.")
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
        return ray_tracing_orchestrator.build_ray_signature_payload(
            source_lat_deg=float(self.source.get_latitude()),
            source_lon_deg=float(self.source.get_longitude()),
            source_time=str(self.source.get_time()),
            run_type=run_type,
            ray_params=ray_params,
            profile=profile,
        )

    def _apply_raypaths_metadata(
        self,
        raypaths,
        type,
        use_accel,
        accel_used,
        command,
        signature_hash=None,
    ):
        ray_tracing_orchestrator.apply_raypaths_metadata(
            raypaths=raypaths,
            run_type=type,
            use_accel=use_accel,
            accel_used=accel_used,
            command=command,
            source_lat_deg=float(self.source.get_latitude()),
            source_lon_deg=float(self.source.get_longitude()),
            source_time=str(self.source.get_time()),
            signature_hash=signature_hash,
        )

    @staticmethod
    def _azimuth_sequence(az_min, az_max, az_step):
        """Build normalized azimuth sequence in degrees."""
        return ray_tracing_orchestrator.azimuth_sequence(
            az_min=az_min,
            az_max=az_max,
            az_step=az_step,
        )

    @staticmethod
    def _forward_geodesic_deg(src_lat_deg, src_lon_deg, angular_distance_rad, azimuth_deg):
        """Project points from source by angular distance and azimuth on a sphere."""
        return ray_tracing_orchestrator.forward_geodesic_deg(
            src_lat_deg=src_lat_deg,
            src_lon_deg=src_lon_deg,
            angular_distance_rad=angular_distance_rad,
            azimuth_deg=azimuth_deg,
        )

    @staticmethod
    def _great_circle_angular_distance_rad(src_lat_deg, src_lon_deg, dst_lat_deg, dst_lon_deg):
        """Great-circle angular distance between source and destination points."""
        return ray_tracing_orchestrator.great_circle_angular_distance_rad(
            src_lat_deg=src_lat_deg,
            src_lon_deg=src_lon_deg,
            dst_lat_deg=dst_lat_deg,
            dst_lon_deg=dst_lon_deg,
        )

    def _project_2d_raypaths_to_azimuths(self, raypaths, az_min, az_max, az_step):
        """Expand 2D raypaths into synthetic 3D by azimuthal projection."""
        src_lat = float(self.source.get_latitude())
        src_lon = float(self.source.get_longitude())
        return ray_tracing_orchestrator.project_2d_raypaths_to_azimuths(
            raypaths=raypaths,
            src_lat_deg=src_lat,
            src_lon_deg=src_lon,
            az_min=az_min,
            az_max=az_max,
            az_step=az_step,
        )

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
        ray_tracing_orchestrator.warn_az_interp_approximation()

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
            if ray_tracing_orchestrator.signature_payload_matches(
                signature_file=signature_file,
                signature_payload=signature_payload,
            ):
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

        arg_list = grid_enrichment_orchestrator.build_profile_arg_list(
            latitudes=latitudes,
            longitudes=longitudes,
            altitudes=altitudes,
            time=time,
            model_name=ionosphere_model,
        )

        _log.info(
            "Computing ionospheric electron density (%s): %d profiles "
            "(%d lat × %d lon × %d alt)",
            ionosphere_model, len(arg_list),
            len(latitudes), len(longitudes), len(altitudes),
        )

        results = grid_enrichment_orchestrator.run_profile_workers(
            arg_list=arg_list,
            worker=_iono_profile_worker,
            fallback_factory=lambda altitude_count: np.full(altitude_count, np.nan),
            warning_prefix="Ionosphere",
            logger=_log,
            max_workers=max_workers,
        )

        electron_density_3d = grid_enrichment_orchestrator.reshape_scalar_profiles(
            results=results,
            latitude_count=len(latitudes),
            longitude_count=len(longitudes),
            altitude_count=len(altitudes),
        )

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

        arg_list = grid_enrichment_orchestrator.build_profile_arg_list(
            latitudes=latitudes,
            longitudes=longitudes,
            altitudes=altitudes,
            time=self.source.get_time(),
            model_name=magnetic_field_model,
        )

        _log.info(
            "Computing magnetic field (%s): %d profiles (%d lat × %d lon × %d alt)",
            magnetic_field_model, len(arg_list),
            len(latitudes), len(longitudes), len(altitudes),
        )

        results = grid_enrichment_orchestrator.run_profile_workers(
            arg_list=arg_list,
            worker=_magfield_profile_worker,
            fallback_factory=lambda altitude_count: {
                v: np.full(altitude_count, np.nan) for v in _MAG_VARS
            },
            warning_prefix="Magnetic field",
            logger=_log,
            max_workers=max_workers,
        )

        arrays = grid_enrichment_orchestrator.reshape_vector_profiles(
            results=results,
            variable_names=_MAG_VARS,
            latitude_count=len(latitudes),
            longitude_count=len(longitudes),
            altitude_count=len(altitudes),
        )

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

    def assign_continuity(
        self,
        t0_s=None,
        tmax_s=None,
        dt_s=None,
        b=1.0,
        divergence_flag=3,
        geomag_flag=True,
        interpolation_radius_km=50.0,
        mapping_mode="nearest",
        use_kdtree=True,
        altitude_window_km=None,
        min_points=3,
        weight_power=2.0,
        chunk_size=1024,
        output_dir=None,
        reuse_existing=True,
        force_recompute=False,
        cache_id=None,
        store_neutral_velocity=True,
    ):
        """Compute electron density perturbations via the continuity equation.

        Parameters
        ----------
        t0_s : float, optional
            Start time in seconds. Defaults to ``[continuity].t0_s``.
        tmax_s : float, optional
            End time in seconds. Defaults to ``[continuity].tmax_s``.
        dt_s : float, optional
            Time step in seconds. Defaults to ``[continuity].dt_s``.
        b : float, optional
            Pulse broadening coefficient in seconds per second.
        divergence_flag : int, optional
            ``1`` for radial-only divergence, ``3`` for full 3-D divergence.
        geomag_flag : bool, optional
            If True, project the neutral velocity along the magnetic field.
        interpolation_radius_km : float, optional
            Neighborhood radius (km) for ray scalar mapping.
        mapping_mode : str, optional
            Mapping mode, either ``"nearest"`` or ``"weighted"``.
        use_kdtree : bool, optional
            If True, use a KD-tree for nearest-neighbor lookups when available.
        altitude_window_km : float, optional
            Vertical window (km) to limit ray candidates.
        min_points : int, optional
            Minimum number of ray points required to assign a grid cell.
        weight_power : float, optional
            Power for distance taper ``(1 - d / r) ** weight_power``.
        chunk_size : int, optional
            Grid points processed per chunk to limit memory usage.
        output_dir : str or Path, optional
            Directory where continuity outputs are saved.
        reuse_existing : bool, optional
            If True, reuse cached results when signatures match.
        force_recompute : bool, optional
            If True, bypass cache checks and recompute.
        cache_id : str, optional
            Optional cache token to isolate output files.
        store_neutral_velocity : bool, optional
            If True, store neutral velocity components for visualization.

        Returns
        -------
        xr.Dataset
            Continuity output with ``dNe`` and optional neutral velocity.
        """
        if self.raypaths is None:
            raise AttributeError("Raypaths not available. Call trace_rays() first.")
        if not hasattr(self, "grid"):
            raise AttributeError("3D grid not created. Call make_3Dgrid() first.")
        for name in ["kr", "kt", "kp", "electron_density"]:
            if name not in self.grid:
                raise AttributeError(
                    f"Grid variable '{name}' is missing. Run assign_wavevector() and "
                    "assign_ionosphere() first."
                )

        t0_s, tmax_s, dt_s = continuity_orchestrator.resolve_timing(
            t0_s=t0_s,
            tmax_s=tmax_s,
            dt_s=dt_s,
            default_t0_s=self.continuity_t0_s,
            default_tmax_s=self.continuity_tmax_s,
            default_dt_s=self.continuity_dt_s,
        )

        if altitude_window_km is None:
            altitude_window_km = float(interpolation_radius_km)

        raypaths = self.raypaths
        continuity_orchestrator.ensure_travel_time_on_grid(
            grid=self.grid,
            raypaths=raypaths,
            map_ray_scalar_to_grid=wavevector_tools.map_ray_scalar_to_grid,
            interpolation_radius_km=float(interpolation_radius_km),
            mapping_mode=str(mapping_mode),
            use_kdtree=bool(use_kdtree),
            altitude_window_km=float(altitude_window_km),
            min_points=int(min_points),
            weight_power=float(weight_power),
            chunk_size=int(chunk_size),
        )

        continuity_orchestrator.ensure_amplitude_on_grid(
            grid=self.grid,
            raypaths=raypaths,
            map_ray_scalar_to_grid=wavevector_tools.map_ray_scalar_to_grid,
            interpolation_radius_km=float(interpolation_radius_km),
            mapping_mode=str(mapping_mode),
            use_kdtree=bool(use_kdtree),
            altitude_window_km=float(altitude_window_km),
            min_points=int(min_points),
            weight_power=float(weight_power),
            chunk_size=int(chunk_size),
        )

        signature_payload = continuity_orchestrator.build_signature_payload(
            grid=self.grid,
            ray_signature_hash=self.raypaths.attrs.get("raytrace_signature_hash"),
            t0_s=float(t0_s),
            tmax_s=float(tmax_s),
            dt_s=float(dt_s),
            b=float(b),
            divergence_flag=int(divergence_flag),
            geomag_flag=bool(geomag_flag),
            interpolation_radius_km=float(interpolation_radius_km),
            mapping_mode=str(mapping_mode),
            use_kdtree=bool(use_kdtree),
            altitude_window_km=float(altitude_window_km),
            min_points=int(min_points),
            weight_power=float(weight_power),
            chunk_size=int(chunk_size),
            store_neutral_velocity=bool(store_neutral_velocity),
        )
        signature_hash = model_io.canonical_json_hash(signature_payload)

        output_file, signature_file = continuity_orchestrator.resolve_cache_paths(
            output_dir=output_dir,
            signature_hash=signature_hash,
            cache_id=cache_id,
        )

        cached_continuity = continuity_orchestrator.load_cached_dataset_if_valid(
            output_file=output_file,
            signature_file=signature_file,
            signature_payload=signature_payload,
            reuse_existing=bool(reuse_existing),
            force_recompute=bool(force_recompute),
            load_dataset=xr.load_dataset,
        )
        if cached_continuity is not None:
            self.continuity = cached_continuity
            return self.continuity

        continuity = continuity_tools.solve_continuity(
            grid=self.grid,
            t0_s=float(t0_s),
            tmax_s=float(tmax_s),
            dt_s=float(dt_s),
            b=float(b),
            divergence_flag=int(divergence_flag),
            geomag_flag=bool(geomag_flag),
            store_neutral_velocity=bool(store_neutral_velocity),
        )

        continuity = continuity_orchestrator.annotate_continuity_result(
            continuity=continuity,
            grid=self.grid,
            signature_hash=signature_hash,
            signature_payload=signature_payload,
        )

        continuity_orchestrator.persist_continuity_with_signature(
            continuity=continuity,
            output_file=output_file,
            signature_file=signature_file,
            signature_hash=signature_hash,
            signature_payload=signature_payload,
        )

        self.continuity = continuity
        return continuity

    def compute_los_tec(
        self,
        receiver_positions=None,
        satellite_positions=None,
        receiver_code=None,
        receiver_format=None,
        sat_id=None,
        constellation=None,
        prn=None,
        sat_number=None,
        receiver_csv=None,
        receiver_listesta=None,
        orbit_h5=None,
        orbit_pos=None,
        orbit_format=None,
        satpos_root=None,
        satpos_date=None,
        sat_mapping_file=None,
        start_offset_s=None,
        dNe=None,
        tec_config=None,
    ):
        """Compute LOS TEC for a receiver-satellite pair.

        Parameters
        ----------
        receiver_positions : dict or xr.Dataset, optional
            Receiver positions with ``latitude``, ``longitude``, ``height_km``.
            When omitted, load from ``[tec].receiver_csv``.
        satellite_positions : dict or xr.Dataset, optional
            Satellite positions with ``x_km``, ``y_km``, ``z_km`` and ``time``.
            When omitted, load from ``[tec].orbit_h5``.
        receiver_code : str, optional
            Receiver code to select from the CSV.
        receiver_format : str, optional
            Receiver input format. Supported values: ``"csv"`` or ``"listesta"``.
        sat_id : str, optional
            Satellite ID (e.g., ``"G21"``).
        constellation : str, optional
            Constellation code (e.g., ``"GPS"``) used with ``prn``.
        prn : int, optional
            Satellite PRN used with ``constellation``.
        sat_number : int, optional
            Legacy SATPOS satellite number for mapping-file workflows.
        receiver_csv : str or Path, optional
            CSV path for receiver positions. Defaults to ``[tec].receiver_csv``.
        receiver_listesta : str or Path, optional
            Legacy listesta station file path. Defaults to ``[tec].receiver_listesta``.
        orbit_h5 : str or Path, optional
            HDF5 path for orbits. Defaults to ``[tec].orbit_h5``.
        orbit_pos : str or Path, optional
            Legacy SATPOS file path. Defaults to ``[tec].orbit_pos``.
        orbit_format : str, optional
            Orbit input format. Supported values: ``"h5"`` or ``"pos"``.
        satpos_root : str or Path, optional
            Legacy SATPOS root path for folder-based file lookup.
        satpos_date : str, optional
            Legacy SATPOS date token for folder-based lookup.
        sat_mapping_file : str or Path, optional
            Mapping CSV with columns ``sat_number`` and ``sat_id``.
        start_offset_s : float, optional
            Start offset in seconds relative to event origin for legacy SATPOS.
        dNe : xr.DataArray, optional
            Electron density perturbation (m^-3) with time dimension.
        tec_config : pyionoseis.tec.TECConfig, optional
            Override TEC configuration settings.

        Returns
        -------
        xr.Dataset
            Dataset with TEC time series and LOS metadata.
        """
        if not hasattr(self, "grid"):
            raise AttributeError("3D grid not created. Call make_3Dgrid() first.")
        if not hasattr(self, "source"):
            raise AttributeError("Source not assigned to the model.")

        if tec_config is None:
            tec_config = tec_tools.TECConfig(
                elevation_mask_deg=float(self.tec_elevation_mask_deg),
                output_dt_s=self.tec_output_dt_s,
                ipp_altitude_km=float(self.tec_ipp_altitude_km),
            )

        if receiver_positions is None:
            receiver_format = receiver_format or self.tec_receiver_format
            receiver_csv = receiver_csv or self.tec_receiver_csv
            receiver_listesta = receiver_listesta or self.tec_receiver_listesta
            receiver_code = receiver_code or self.tec_receiver_code
            if receiver_format is None:
                receiver_format = "csv" if receiver_csv is not None else "listesta"

            if str(receiver_format).lower() == "csv":
                if receiver_csv is None:
                    raise ValueError(
                        "receiver_csv must be provided when receiver_format='csv'."
                    )
                receiver_positions = tec_io.load_receiver_positions_csv(
                    receiver_csv,
                    receiver_code,
                )
            elif str(receiver_format).lower() == "listesta":
                if receiver_listesta is None:
                    raise ValueError(
                        "receiver_listesta must be provided when receiver_format='listesta'."
                    )
                receiver_positions = tec_io.load_receiver_positions_listesta(
                    receiver_listesta,
                    receiver_code,
                )
            else:
                raise ValueError(
                    "receiver_format must be one of: 'csv', 'listesta'."
                )

        if satellite_positions is None:
            orbit_format = orbit_format or self.tec_orbit_format
            orbit_h5 = orbit_h5 or self.tec_orbit_h5
            orbit_pos = orbit_pos or self.tec_orbit_pos
            sat_id = sat_id or self.tec_sat_id
            constellation = constellation or self.tec_constellation
            prn = prn or self.tec_prn
            sat_number = sat_number or self.tec_sat_number
            satpos_root = satpos_root or self.tec_satpos_root
            satpos_date = satpos_date or self.tec_satpos_date
            sat_mapping_file = sat_mapping_file or self.tec_sat_mapping_file
            start_offset_s = (
                self.tec_start_offset_s
                if start_offset_s is None
                else float(start_offset_s)
            )

            if orbit_format is None:
                orbit_format = "h5" if orbit_h5 is not None else "pos"

            if str(orbit_format).lower() == "h5":
                if orbit_h5 is None:
                    raise ValueError(
                        "orbit_h5 must be provided when orbit_format='h5'."
                    )
                satellite_positions = tec_io.load_orbits_hdf5(
                    orbit_h5,
                    event_time=self.source.get_time(),
                    sat_id=sat_id,
                    constellation=constellation,
                    prn=prn,
                    output_dt_s=tec_config.output_dt_s,
                )
            elif str(orbit_format).lower() == "pos":
                if orbit_pos is None:
                    if satpos_root is None or satpos_date is None or sat_number is None:
                        raise ValueError(
                            "orbit_pos or (satpos_root, satpos_date, sat_number) "
                            "must be provided when orbit_format='pos'."
                        )
                    orbit_pos = tec_io.build_satpos_file_path(
                        satpos_root,
                        str(satpos_date),
                        int(sat_number),
                    )

                satellite_positions = tec_io.load_orbits_pos(
                    orbit_pos,
                    event_time=self.source.get_time(),
                    sat_id=sat_id,
                    constellation=constellation,
                    prn=prn,
                    sat_number=sat_number,
                    sat_mapping_file=sat_mapping_file,
                    start_offset_s=float(start_offset_s),
                    output_dt_s=tec_config.output_dt_s,
                )
            else:
                raise ValueError("orbit_format must be one of: 'h5', 'pos'.")

        if dNe is None and self.continuity is not None and "dNe" in self.continuity:
            dNe = self.continuity["dNe"]

        receiver_id = receiver_positions.get("code") if isinstance(receiver_positions, dict) else None
        satellite_id = sat_id
        if satellite_id is None and constellation is not None and prn is not None:
            satellite_id = f"{constellation}{int(prn):02d}"

        return tec_tools.compute_los_tec(
            grid=self.grid,
            receiver_positions=receiver_positions,
            satellite_positions=satellite_positions,
            dNe=dNe,
            tec_config=tec_config,
            receiver_id=receiver_id,
            satellite_id=satellite_id,
        )


        
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
        Log comprehensive information about the Model3D object.
        
        This method provides detailed information about:
        - Basic model parameters
        - Source configuration
        - Grid dimensions and extent
        - Available data variables
        - Model states (atmosphere, ionosphere, magnetic field)
        """
        lines = []
        lines.append("=" * 80)
        lines.append("MODEL3D INFORMATION")
        lines.append("=" * 80)
        
        # Basic Model Information
        lines.append(f"Model Name: {self.name}")
        lines.append(f"Model Radius: {self.radius} km")
        lines.append(f"Model Height: {self.height} km")
        lines.append(f"Include Winds: {self.winds}")
        lines.append(f"Grid Spacing: {self.grid_spacing} degrees")
        lines.append(f"Height Spacing: {self.height_spacing} km")
        
        # Source Information
        lines.append("")
        lines.append("SOURCE INFORMATION:")
        if hasattr(self, 'source') and self.source is not None:
            lines.append(f"  Latitude: {self.source.get_latitude():.4f}°")
            lines.append(f"  Longitude: {self.source.get_longitude():.4f}°")
            lines.append(f"  Time: {self.source.get_time()}")
            if hasattr(self.source, 'get_depth'):
                lines.append(f"  Depth: {self.source.get_depth()} km")
        else:
            lines.append("  No source assigned")
        
        # Grid Information
        lines.append("")
        lines.append("GRID INFORMATION:")
        if hasattr(self, 'grid') and self.grid is not None:
            if hasattr(self, 'lat_extent'):
                lines.append(
                    f"  Latitude extent: {self.lat_extent[0]:.4f}° to {self.lat_extent[1]:.4f}°"
                )
            if hasattr(self, 'lon_extent'):
                lines.append(
                    f"  Longitude extent: {self.lon_extent[0]:.4f}° to {self.lon_extent[1]:.4f}°"
                )
            
            # Grid dimensions
            if hasattr(self.grid, 'sizes'):
                # This is an xarray Dataset/DataArray with sizes
                grid_shape = self.grid.sizes
                lines.append(f"  Grid dimensions: {dict(grid_shape)}")
                total_points = 1
                for dim, size in grid_shape.items():
                    total_points *= size
                lines.append(f"  Total grid points: {total_points:,}")
            elif hasattr(self.grid, 'shape'):
                # This is a numpy array or similar
                grid_shape = self.grid.shape
                lines.append(f"  Grid shape: {grid_shape}")
                total_points = 1
                for dim in grid_shape:
                    total_points *= dim
                lines.append(f"  Total grid points: {total_points:,}")
            else:
                lines.append("  Grid shape: Unknown")
            
            # Coordinate ranges
            if hasattr(self.grid, 'coords'):
                coords = self.grid.coords
                if 'latitude' in coords:
                    lat_vals = coords['latitude'].values
                    lines.append(
                        f"  Latitude range: {lat_vals.min():.4f}° to {lat_vals.max():.4f}° ({len(lat_vals)} points)"
                    )
                if 'longitude' in coords:
                    lon_vals = coords['longitude'].values
                    lines.append(
                        f"  Longitude range: {lon_vals.min():.4f}° to {lon_vals.max():.4f}° ({len(lon_vals)} points)"
                    )
                if 'altitude' in coords:
                    alt_vals = coords['altitude'].values
                    lines.append(
                        f"  Altitude range: {alt_vals.min():.1f} to {alt_vals.max():.1f} km ({len(alt_vals)} points)"
                    )
        else:
            lines.append("  No grid created yet")
        
        # Available Data Variables
        lines.append("")
        lines.append("AVAILABLE DATA VARIABLES:")
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
                    lines.append(f"  Atmospheric variables: {', '.join(atmospheric_vars)}")
                if ionospheric_vars:
                    lines.append(f"  Ionospheric variables: {', '.join(ionospheric_vars)}")
                if magnetic_geodetic_vars:
                    lines.append(
                        f"  Magnetic field (geodetic): {', '.join(magnetic_geodetic_vars)}"
                    )
                if magnetic_geocentric_vars:
                    lines.append(
                        f"  Magnetic field (geocentric): {', '.join(magnetic_geocentric_vars)}"
                    )
                if magnetic_derived_vars:
                    lines.append(
                        f"  Magnetic field (derived): {', '.join(magnetic_derived_vars)}"
                    )
                if other_vars:
                    lines.append(f"  Other variables: {', '.join(other_vars)}")
                
                lines.append(f"  Total variables: {len(self.grid.data_vars)}")
                
                # Show variable ranges for numeric data
                lines.append("")
                lines.append("  Variable Ranges:")
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
                                lines.append(f"    {var}: {min_val:.2f} to {max_val:.2f}{units}")
                        except (ValueError, TypeError, AttributeError):
                            lines.append(f"    {var}: [data present]")
            else:
                lines.append("  No data variables computed yet")
        else:
            lines.append("  No data variables (grid not created)")
        
        # Model States
        lines.append("")
        lines.append("MODEL STATES:")
        
        # Atmosphere model
        if hasattr(self, 'atmosphere_model'):
            lines.append(f"  Atmosphere model: {self.atmosphere_model}")
            if hasattr(self, 'grid') and self.grid is not None and hasattr(self.grid, 'data_vars'):
                atm_vars = [v for v in self.grid.data_vars if v in ['density', 'pressure', 'temperature', 'velocity']]
                lines.append(f"    Status: {'Computed' if atm_vars else 'Not computed'}")
                if atm_vars:
                    lines.append(f"    Variables: {', '.join(atm_vars)}")
        
        # Ionosphere model
        if hasattr(self, 'ionosphere_model'):
            lines.append(f"  Ionosphere model: {self.ionosphere_model}")
            if hasattr(self, 'grid') and self.grid is not None and hasattr(self.grid, 'data_vars'):
                iono_computed = 'electron_density' in self.grid.data_vars
                lines.append(f"    Status: {'Computed' if iono_computed else 'Not computed'}")
        
        # Magnetic field model
        if hasattr(self, 'magnetic_field_model'):
            lines.append(f"  Magnetic field model: {self.magnetic_field_model}")
            if hasattr(self, 'grid') and self.grid is not None and hasattr(self.grid, 'data_vars'):
                mag_vars = [v for v in self.grid.data_vars if v in ['Be', 'Bn', 'Bu', 'Br', 'Btheta', 'Bphi', 'inclination', 'declination']]
                lines.append(f"    Status: {'Computed' if mag_vars else 'Not computed'}")
                if mag_vars:
                    geodetic = [v for v in mag_vars if v in ['Be', 'Bn', 'Bu']]
                    geocentric = [v for v in mag_vars if v in ['Br', 'Btheta', 'Bphi']]
                    derived = [v for v in mag_vars if v in ['inclination', 'declination']]
                    if geodetic:
                        lines.append(f"    Geodetic components: {', '.join(geodetic)}")
                    if geocentric:
                        lines.append(f"    Geocentric components: {', '.join(geocentric)}")
                    if derived:
                        lines.append(f"    Derived parameters: {', '.join(derived)}")
        
        # Grid attributes
        if hasattr(self, 'grid') and self.grid is not None and hasattr(self.grid, 'attrs') and self.grid.attrs:
            lines.append("")
            lines.append("GRID ATTRIBUTES:")
            for key, value in self.grid.attrs.items():
                if not key.endswith('_description'):  # Skip long description attributes
                    lines.append(f"  {key}: {value}")
        
        lines.append("=" * 80)
        _log.info("\n%s", "\n".join(lines))

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

