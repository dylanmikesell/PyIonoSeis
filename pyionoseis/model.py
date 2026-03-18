"""The model module contains common functions and classes used to generate the spatial modes.
"""

import toml
import xarray as xr
import numpy as np
import os
import warnings
import json
import hashlib
from pathlib import Path
import shutil
import tempfile
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyionoseis.atmosphere import Atmosphere1D
from pyionoseis.ionosphere import Ionosphere1D
from pyionoseis.igrf import MagneticField1D, PPIGRF_AVAILABLE
from pyionoseis import infraga as infraga_tools

class Model3D:
    """
    A class to represent a 3D model for ionospheric studies.
    Attributes:
    -----------
    name : str
        The name of the model.
    radius : float
        The radius of the model in kilometers.
    height : float
        The height of the model in kilometers.
    winds : bool
        A flag indicating if winds are considered in the model.
    grid_spacing : float
        The spacing of the grid in degrees.
    height_spacing : float
        The spacing of the height levels in kilometers.
    source : object
        The source object associated with the model.
    grid : xarray.DataArray
        The 3D grid of the model.
    lat_extent : tuple
        The latitude extent of the grid.
    lon_extent : tuple
        The longitude extent of the grid.
    Methods:
    --------
    __init__(self, toml_file=None):
        Initialize the Model3D instance.
        Parameters:
        -----------
        toml_file : str, optional
            The path to a TOML file to load model parameters from.
    load_from_toml(self, toml_file):
        Load model parameters from a TOML file.
        Parameters:
        -----------
        toml_file : str
            The path to the TOML file.
    assign_source(self, source):
        Assign a source to the model.
        Parameters:
        -----------
        source : object
            The source object to be assigned to the model.
    __str__(self):
        Return a string representation of the model.
        Returns:
        --------
        str
            A string describing the model.
    print_info(self):
        Print the model information.
    make_3Dgrid(self):
        Create a 3D grid with the given source and model parameters.
        Raises:
        -------
        AttributeError
            If the source is not assigned to the model.
    plot(self):
        Plot the source location on a map.
        Raises:
        -------
        AttributeError
            If the source is not assigned to the model.
        ValueError
            If the source does not have 'latitude' and 'longitude' attributes.
    plot_grid(self, show_gridlines=False):
        Plot the 2D grid points on a map.
        Parameters:
        -----------
        show_gridlines : bool, optional
            Whether to show gridlines on the plot. Default is False.
        Raises:
        -------
        AttributeError
            If the 3D grid is not created.
    plot_grid_3d(self):
        Plot the 3D grid points.
        Raises:
        -------
        AttributeError
            If the 3D grid is not created.
    assign_ionosphere(self, ionosphere_model="iri2020"):
        Assign ionospheric electron density to all points in the 3D model domain.
        Parameters:
        -----------
        ionosphere_model : str
            The ionospheric model to use (default: "iri2020")
    plot_variable(self, variable='grid_points', altitude_slice=None):
        Plot different variables on the model grid.
        Parameters:
        -----------
        variable : str
            Variable to plot: 'grid_points', 'electron_density', 'density', 
            'pressure', 'velocity', 'temperature'
        altitude_slice : float, optional
            Altitude level (km) for 2D maps
    """
    pass
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

    @staticmethod
    def _array_sha256(values):
        """Build a deterministic SHA256 digest for a numeric array."""
        array = np.asarray(values, dtype=np.float64)
        hasher = hashlib.sha256()
        hasher.update(str(array.shape).encode("utf-8"))
        hasher.update(array.tobytes())
        return hasher.hexdigest()

    @staticmethod
    def _canonical_json_hash(payload):
        """Hash a JSON-serializable payload using canonical key ordering."""
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_signature_payload(raw):
        """Return signature payload when wrapped/unwrapped JSON is supplied."""
        if isinstance(raw, dict) and "signature" in raw and isinstance(raw["signature"], dict):
            return raw["signature"]
        return raw

    @staticmethod
    def _signature_path_for_output_prefix(output_prefix):
        return Path(str(output_prefix) + ".signature.json")

    @staticmethod
    def _signature_path_for_raypaths(raypaths_file):
        raypaths_file = Path(raypaths_file)
        raypaths_name = raypaths_file.name
        if raypaths_name.endswith(".raypaths.dat"):
            base_name = raypaths_name[: -len(".raypaths.dat")]
            return raypaths_file.with_name(base_name + ".signature.json")
        return Path(str(raypaths_file) + ".signature.json")

    def _cache_token(self, signature_hash, cache_id=None):
        if cache_id is None:
            return signature_hash[:12]
        token = str(cache_id).strip()
        if not token:
            return signature_hash[:12]
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in token)

    def _build_ray_signature_payload(self, type, ray_params, profile):
        return {
            "signature_version": 1,
            "run_type": str(type),
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
                else self._signature_path_for_raypaths(raypaths_path)
            )
            if not sig_path.exists():
                raise FileNotFoundError(f"Signature file does not exist: {sig_path}")
            with sig_path.open("r", encoding="utf-8") as fh:
                raw_sig_data = json.load(fh)
            expected_payload = self._normalize_signature_payload(raw_sig_data)
            if self.source is None:
                raise AttributeError("Source must be assigned for signature validation.")
            if self.atmosphere is None:
                if not hasattr(self, "grid"):
                    raise AttributeError(
                        "3D grid not created. Call make_3Dgrid() first for signature validation."
                    )
                self.assign_atmosphere()
            actual_payload = self._build_ray_signature_payload(
                type=run_type,
                ray_params=expected_payload.get("ray_params", {}),
                profile={
                    "prof_format": expected_payload.get("profile", {}).get("prof_format", "zcuvd"),
                    "winds_assumed_zero": bool(
                        expected_payload.get("profile", {}).get("winds_assumed_zero", True)
                    ),
                    "altitude_sha256": self._array_sha256(
                        self.atmosphere.atmosphere["altitude"].values
                    ),
                    "velocity_sha256": self._array_sha256(
                        self.atmosphere.atmosphere["velocity"].values
                    ),
                    "density_sha256": self._array_sha256(
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
            "altitude_sha256": self._array_sha256(self.atmosphere.atmosphere["altitude"].values),
            "velocity_sha256": self._array_sha256(self.atmosphere.atmosphere["velocity"].values),
            "density_sha256": self._array_sha256(self.atmosphere.atmosphere["density"].values),
        }
        signature_payload = self._build_ray_signature_payload(
            type=type, ray_params=ray_params, profile=profile_info
        )
        signature_hash = self._canonical_json_hash(signature_payload)

        profile_file = run_dir / "infraga_input_profile.zcuvd"
        if output_dir is not None:
            output_token = self._cache_token(signature_hash=signature_hash, cache_id=cache_id)
            output_prefix = run_dir / f"infraga_{type}_sph_{output_token}"
        else:
            output_prefix = run_dir / f"infraga_{type}_sph"

        raypaths_file = Path(str(output_prefix) + ".raypaths.dat")
        arrivals_file = Path(str(output_prefix) + ".arrivals.dat")
        signature_file = self._signature_path_for_output_prefix(output_prefix)

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
            with signature_file.open("r", encoding="utf-8") as fh:
                saved_signature_raw = json.load(fh)
            saved_signature_payload = self._normalize_signature_payload(saved_signature_raw)
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
                    warnings.warn(
                        "az_interp=True remaps a single-azimuth 2D ray solution into synthetic 3D. "
                        "This assumes an axisymmetric medium around the source. In non-axisymmetric "
                        "atmosphere/wind fields, remapped travel times and amplitudes are not physically valid.",
                        RuntimeWarning,
                    )
                    warnings.warn(
                        "Ray tracing currently writes zero winds into infraGA profile (u=v=0). "
                        "If your model includes winds or azimuth-dependent structure, synthetic 3D remapping "
                        "is an approximation intended for visualization/sensitivity only.",
                        RuntimeWarning,
                    )
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
            warnings.warn(
                "az_interp=True remaps a single-azimuth 2D ray solution into synthetic 3D. "
                "This assumes an axisymmetric medium around the source. In non-axisymmetric "
                "atmosphere/wind fields, remapped travel times and amplitudes are not physically valid.",
                RuntimeWarning,
            )
            warnings.warn(
                "Ray tracing currently writes zero winds into infraGA profile (u=v=0). "
                "If your model includes winds or azimuth-dependent structure, synthetic 3D remapping "
                "is an approximation intended for visualization/sensitivity only.",
                RuntimeWarning,
            )
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

    def assign_ionosphere(self, ionosphere_model="iri2020"):
        """
        Assign ionospheric electron density to all points in the 3D model domain.
        
        This method computes electron density using the specified ionospheric model
        for each latitude-longitude location in the 3D grid. The computation is done
        efficiently by computing vertical profiles for each lat-lon point.
        
        Parameters:
        -----------
        ionosphere_model : str
            The ionospheric model to use. Currently supports:
            - "iri2020": International Reference Ionosphere 2020 model (default)
            
        Raises:
        -------
        AttributeError
            If source is not assigned to the model or 3D grid is not created.
        ValueError
            If an unsupported ionosphere model is specified.
            
        Notes:
        ------
        The electron density is computed at each grid point and stored in the
        'electron_density' field of the grid DataArray. Units are m⁻³.
        
        The computation strategy:
        1. For each unique lat-lon pair in the 3D grid
        2. Compute a vertical electron density profile using the ionosphere model
        3. Interpolate/assign the profile to all altitude levels at that location
        4. Store results in the 3D grid structure
        """
        if not hasattr(self, 'source'):
            raise AttributeError("Source not assigned to the model.")
        if not hasattr(self, 'grid'):
            raise AttributeError("3D grid not created. Call make_3Dgrid() first.")
            
        # Supported ionosphere models
        supported_models = ["iri2020"]
        if ionosphere_model not in supported_models:
            raise ValueError(f"Unsupported ionosphere model '{ionosphere_model}'. "
                           f"Supported models: {supported_models}")
        
        # Get grid coordinates
        latitudes = self.grid.coords['latitude'].values
        longitudes = self.grid.coords['longitude'].values
        altitudes = self.grid.coords['altitude'].values
        
        # Get source time
        time = self.source.get_time()
        
        print(f"Computing ionospheric electron density using {ionosphere_model}...")
        print(f"Grid dimensions: {len(latitudes)} lat × {len(longitudes)} lon × {len(altitudes)} alt")
        
        # Initialize the electron density array
        electron_density_3d = np.zeros((len(latitudes), len(longitudes), len(altitudes)))
        
        # Compute electron density for each lat-lon location
        total_profiles = len(latitudes) * len(longitudes)
        profile_count = 0
        
        for i, lat in enumerate(latitudes):
            for j, lon in enumerate(longitudes):
                profile_count += 1
                
                # Show progress for large grids
                if total_profiles > 10 and profile_count % max(1, total_profiles // 10) == 0:
                    print(f"  Progress: {profile_count}/{total_profiles} "
                          f"({100*profile_count/total_profiles:.0f}%) profiles computed")
                
                try:
                    # Create 1D ionosphere model for this location
                    iono_1d = Ionosphere1D(lat, lon, altitudes, time, ionosphere_model)
                    
                    # Extract electron density profile
                    ne_profile = iono_1d.ionosphere.electron_density.values
                    
                    # Assign to 3D grid
                    electron_density_3d[i, j, :] = ne_profile
                    
                except Exception as e:
                    print(f"Warning: Failed to compute ionosphere at lat={lat:.2f}, lon={lon:.2f}: {e}")
                    # Fill with NaN for failed computations
                    electron_density_3d[i, j, :] = np.nan
        
        print("Ionospheric computation completed.")
        
        # Convert DataArray to Dataset and add electron density
        if isinstance(self.grid, xr.DataArray):
            # Convert to Dataset with the original data as 'grid_points'
            grid_dataset = self.grid.to_dataset(name='grid_points')
            # Add electron density as a new variable
            grid_dataset['electron_density'] = (["latitude", "longitude", "altitude"], electron_density_3d)
            self.grid = grid_dataset
        else:
            # If already a Dataset, use assign
            self.grid = self.grid.assign(
                electron_density=(["latitude", "longitude", "altitude"], electron_density_3d)
            )
        
        # Update grid attributes
        if not hasattr(self.grid, 'attrs'):
            self.grid.attrs = {}
        self.grid.attrs['ionosphere_model'] = ionosphere_model
        self.grid.attrs['electron_density_units'] = 'm⁻³'
        self.grid.attrs['electron_density_description'] = f'Electron density from {ionosphere_model} model'
        
        # Store ionosphere model info
        self.ionosphere_model = ionosphere_model
        
    def assign_magnetic_field(self, magnetic_field_model="igrf"):
        """
        Assign magnetic field parameters to all points in the 3D model domain.
        
        This method computes magnetic field components (Be, Bn, Bu) and derived
        parameters (inclination, declination) at each grid point using the
        specified magnetic field model.
        
        Parameters:
        -----------
        magnetic_field_model : str
            The magnetic field model to use. Currently supports:
            - "igrf": International Geomagnetic Reference Field model (default)
            
        Raises:
        -------
        AttributeError
            If source is not assigned to the model or 3D grid is not created.
        ValueError
            If an unsupported magnetic field model is specified.
            
        Notes:
        ------
        The magnetic field components are computed at each grid point and stored in the
        grid Dataset. The following variables are added:
        
        Geodetic components (tangent to Earth's ellipsoid):
        - 'Be': East component of magnetic field (nT)
        - 'Bn': North component of magnetic field (nT) 
        - 'Bu': Up component of magnetic field (nT)
        
        Geocentric components (spherical coordinates):
        - 'Br': Radial component of magnetic field (nT)
        - 'Btheta': Colatitude component of magnetic field (nT)
        - 'Bphi': Azimuth component of magnetic field (nT)
        
        Derived parameters:
        - 'inclination': Magnetic inclination angle (degrees)
        - 'declination': Magnetic declination angle (degrees)
        
        The computation strategy:
        1. For each unique lat-lon pair in the 3D grid
        2. Compute magnetic field profile using specified model
        3. Assign computed values to all altitude levels at that location
        """
        if not hasattr(self, 'source'):
            raise AttributeError("Source not assigned to the model.")
        if not hasattr(self, 'grid'):
            raise AttributeError("3D grid not created. Call make_3Dgrid() first.")
        
        if magnetic_field_model.lower() != "igrf":
            raise ValueError(f"Unsupported magnetic field model: {magnetic_field_model}")
            
        if not PPIGRF_AVAILABLE:
            raise ImportError("ppigrf package is not available. Please install it with: pip install ppigrf")
        
        print(f"Computing magnetic field using {magnetic_field_model}...")
        
        # Get grid dimensions
        latitudes = self.grid.coords['latitude'].values
        longitudes = self.grid.coords['longitude'].values
        altitudes = self.grid.coords['altitude'].values
        
        print(f"Grid dimensions: {len(latitudes)} lat × {len(longitudes)} lon × {len(altitudes)} alt")
        
        # Initialize 3D arrays for magnetic field parameters
        Be_3d = np.full((len(latitudes), len(longitudes), len(altitudes)), np.nan)
        Bn_3d = np.full((len(latitudes), len(longitudes), len(altitudes)), np.nan)
        Bu_3d = np.full((len(latitudes), len(longitudes), len(altitudes)), np.nan)
        Br_3d = np.full((len(latitudes), len(longitudes), len(altitudes)), np.nan)
        Btheta_3d = np.full((len(latitudes), len(longitudes), len(altitudes)), np.nan)
        Bphi_3d = np.full((len(latitudes), len(longitudes), len(altitudes)), np.nan)
        inclination_3d = np.full((len(latitudes), len(longitudes), len(altitudes)), np.nan)
        declination_3d = np.full((len(latitudes), len(longitudes), len(altitudes)), np.nan)
        total_field_3d = np.full((len(latitudes), len(longitudes), len(altitudes)), np.nan)
        horizontal_intensity_3d = np.full((len(latitudes), len(longitudes), len(altitudes)), np.nan)
        
        # Compute magnetic field for each lat-lon pair
        total_profiles = len(latitudes) * len(longitudes)
        computed_profiles = 0
        
        for i, lat in enumerate(latitudes):
            for j, lon in enumerate(longitudes):
                try:
                    # Create magnetic field model for this location
                    mag_field = MagneticField1D(lat, lon, altitudes, self.source.get_time(), magnetic_field_model)
                    
                    # Extract computed parameters
                    Be_profile = mag_field.magnetic_field['Be'].values
                    Bn_profile = mag_field.magnetic_field['Bn'].values
                    Bu_profile = mag_field.magnetic_field['Bu'].values
                    Br_profile = mag_field.magnetic_field['Br'].values
                    Btheta_profile = mag_field.magnetic_field['Btheta'].values
                    Bphi_profile = mag_field.magnetic_field['Bphi'].values
                    inc_profile = mag_field.magnetic_field['inclination'].values
                    dec_profile = mag_field.magnetic_field['declination'].values
                    total_field_profile = mag_field.magnetic_field['total_field'].values
                    horizontal_intensity_profile = mag_field.magnetic_field['horizontal_intensity'].values
                    
                    # Assign to 3D grids
                    Be_3d[i, j, :] = Be_profile
                    Bn_3d[i, j, :] = Bn_profile
                    Bu_3d[i, j, :] = Bu_profile
                    Br_3d[i, j, :] = Br_profile
                    Btheta_3d[i, j, :] = Btheta_profile
                    Bphi_3d[i, j, :] = Bphi_profile
                    inclination_3d[i, j, :] = inc_profile
                    declination_3d[i, j, :] = dec_profile
                    total_field_3d[i, j, :] = total_field_profile
                    horizontal_intensity_3d[i, j, :] = horizontal_intensity_profile
                    
                    computed_profiles += 1
                    
                except Exception as e:
                    print(f"Warning: Failed to compute magnetic field at lat={lat:.2f}, lon={lon:.2f}: {e}")
                    # NaN values are already assigned above
                    
                # Progress reporting
                if computed_profiles % max(1, total_profiles // 10) == 0:
                    progress = int(100 * computed_profiles / total_profiles)
                    print(f"  Progress: {computed_profiles}/{total_profiles} ({progress}%) profiles computed")
        
        print("Magnetic field computation completed.")
        
        # Add magnetic field parameters to the grid Dataset
        if isinstance(self.grid, xr.DataArray):
            # Convert to Dataset with the original data as 'grid_points'
            grid_dataset = self.grid.to_dataset(name='grid_points')
            # Add magnetic field variables
            grid_dataset['Be'] = (["latitude", "longitude", "altitude"], Be_3d)
            grid_dataset['Bn'] = (["latitude", "longitude", "altitude"], Bn_3d)
            grid_dataset['Bu'] = (["latitude", "longitude", "altitude"], Bu_3d)
            grid_dataset['Br'] = (["latitude", "longitude", "altitude"], Br_3d)
            grid_dataset['Btheta'] = (["latitude", "longitude", "altitude"], Btheta_3d)
            grid_dataset['Bphi'] = (["latitude", "longitude", "altitude"], Bphi_3d)
            grid_dataset['inclination'] = (["latitude", "longitude", "altitude"], inclination_3d)
            grid_dataset['declination'] = (["latitude", "longitude", "altitude"], declination_3d)
            grid_dataset['total_field'] = (["latitude", "longitude", "altitude"], total_field_3d)
            grid_dataset['horizontal_intensity'] = (["latitude", "longitude", "altitude"], horizontal_intensity_3d)
            self.grid = grid_dataset
        else:
            # If already a Dataset, add new variables
            self.grid['Be'] = (["latitude", "longitude", "altitude"], Be_3d)
            self.grid['Bn'] = (["latitude", "longitude", "altitude"], Bn_3d)
            self.grid['Bu'] = (["latitude", "longitude", "altitude"], Bu_3d)
            self.grid['Br'] = (["latitude", "longitude", "altitude"], Br_3d)
            self.grid['Btheta'] = (["latitude", "longitude", "altitude"], Btheta_3d)
            self.grid['Bphi'] = (["latitude", "longitude", "altitude"], Bphi_3d)
            self.grid['inclination'] = (["latitude", "longitude", "altitude"], inclination_3d)
            self.grid['declination'] = (["latitude", "longitude", "altitude"], declination_3d)
            self.grid['total_field'] = (["latitude", "longitude", "altitude"], total_field_3d)
            self.grid['horizontal_intensity'] = (["latitude", "longitude", "altitude"], horizontal_intensity_3d)
        
        # Update grid attributes
        if not hasattr(self.grid, 'attrs'):
            self.grid.attrs = {}
        self.grid.attrs['magnetic_field_model'] = magnetic_field_model
        self.grid.attrs['magnetic_field_units'] = 'nT'
        self.grid.attrs['magnetic_angle_units'] = 'degrees'
        self.grid.attrs['Be_description'] = f'East component of magnetic field from {magnetic_field_model} model'
        self.grid.attrs['Bn_description'] = f'North component of magnetic field from {magnetic_field_model} model'
        self.grid.attrs['Bu_description'] = f'Up component of magnetic field from {magnetic_field_model} model'
        self.grid.attrs['Br_description'] = f'Radial component of magnetic field from {magnetic_field_model} model (geocentric)'
        self.grid.attrs['Btheta_description'] = f'Colatitude component of magnetic field from {magnetic_field_model} model (geocentric)'
        self.grid.attrs['Bphi_description'] = f'Azimuth component of magnetic field from {magnetic_field_model} model (geocentric)'
        self.grid.attrs['inclination_description'] = f'Magnetic inclination from {magnetic_field_model} model'
        self.grid.attrs['declination_description'] = f'Magnetic declination from {magnetic_field_model} model'
        
        # Store magnetic field model info
        self.magnetic_field_model = magnetic_field_model


        
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

        self.grid = xr.DataArray(
            np.zeros((len(latitudes), len(longitudes), len(altitudes))),
            coords=[latitudes, longitudes, altitudes],
            dims=["latitude", "longitude", "altitude"]
        )

        self.lat_extent = (latitudes[0], latitudes[-1])
        self.lon_extent = (longitudes[0], longitudes[-1])


    def plot_source(self):
        if not hasattr(self, 'source'):
            raise AttributeError("Source not assigned to the model.")
        
        lat = self.source.get_latitude()
        lon = self.source.get_longitude()
        
        if lat is None or lon is None:
            raise ValueError("Source must have 'latitude' and 'longitude' attributes.")
        
        # Convert radius from kilometers to degrees (approximation)
        radius_in_degrees = self.radius / 111.32  # 1 degree is approximately 111.32 km at the equator
        
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([lon - radius_in_degrees, lon + radius_in_degrees, lat - radius_in_degrees, lat + radius_in_degrees], crs=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)
        
        ax.plot(lon, lat, 'k*', markersize=10, transform=ccrs.PlateCarree(), label='Source')
        # ax.text(lon, lat, ' Source', horizontalalignment='left', transform=ccrs.PlateCarree())
        
        plt.title(f"Source Location: ({lat:.2f}, {lon:.2f})")
        
        # Add scale bar
        # scale_bar(ax, (0.1, 0.05), 100)  # 100 km scale bar
        
        # Add legend
        ax.legend(loc='upper right')
        
        plt.show()

    # def scale_bar(ax, location, length):
    #     """
    #     Add a scale bar to the map.
        
    #     Parameters:
    #     ax : matplotlib axes object
    #     The axes to draw the scale bar on.
    #     location : tuple
    #     The location of the scale bar in axes coordinates (0 to 1).
    #     length : float
    #     The length of the scale bar in kilometers.
    #     """
    #     # Get the extent of the map in degrees
    #     extent = ax.get_extent(ccrs.PlateCarree())
    #     # Calculate the length of the scale bar in degrees
    #     length_in_degrees = length / 111.32  # 1 degree is approximately 111.32 km at the equator
        
    #     # Create a line for the scale bar
    #     line = plt.Line2D([location[0], location[0] + length_in_degrees], [location[1], location[1]], 
    #               transform=ax.transAxes, color='black', linewidth=2)
    #     ax.add_line(line)
        
    #     # Add text for the scale bar
    #     ax.text(location[0] + length_in_degrees / 2, location[1] - 0.02, f'{length} km', 
    #         transform=ax.transAxes, horizontalalignment='center', verticalalignment='top')

    def plot_grid(self, show_gridlines=False):
        if not hasattr(self, 'grid'):
            raise AttributeError("3D grid not created. Call make_3Dgrid() first.")
        
        latitudes = self.grid.coords['latitude'].values
        longitudes = self.grid.coords['longitude'].values
        
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([longitudes[0] - self.grid_spacing, longitudes[-1] + self.grid_spacing, 
                   latitudes[0] - self.grid_spacing, latitudes[-1] + self.grid_spacing], 
                  crs=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)
        
        for lat in latitudes:
            for lon in longitudes:
                ax.plot(lon, lat, 'ro', markersize=2, transform=ccrs.PlateCarree())

        # Define the scale bar
        fontprops = fm.FontProperties(size=12)
        scalebar = AnchoredSizeBar(ax.transData,
                                1,  # Length of the scale bar in data units
                                '100 km',  # Label for the scale bar
                                'lower right',  # Location of the scale bar
                                pad=0.1,
                                color='black',
                                frameon=False,
                                size_vertical=0.1,
                                fontproperties=fontprops)
        
        # Add the scale bar to the plot
        ax.add_artist(scalebar)
        if show_gridlines:
            ax.gridlines(draw_labels=True)
        
        plt.title("Lat/Lon Grid Points")
        
        plt.show()

    def plot_grid_3d(self):
        if not hasattr(self, 'grid'):
            raise AttributeError("3D grid not created. Call make_3Dgrid() first.")
        
        latitudes = self.grid.coords['latitude'].values
        longitudes = self.grid.coords['longitude'].values
        altitudes = self.grid.coords['altitude'].values
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        lat_grid, lon_grid, alt_grid = np.meshgrid(latitudes, longitudes, altitudes, indexing='ij')
        
        ax.scatter(lon_grid, lat_grid, alt_grid, c='k', marker='o', s=1)
        
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        ax.set_zlabel('Altitude (km)')
        
        plt.title("3D Grid Points")
        
        plt.show()
        
    def plot_variable(self, variable='grid_points', altitude_slice=None, **kwargs):
        """
        Plot different variables on the model grid.
        
        Parameters:
        -----------
        variable : str
            The variable to plot. Options:
            - 'grid_points': Plot grid points only (default)
            - 'electron_density': Plot electron density (requires assign_ionosphere)
            - 'Be': Plot east magnetic field component (requires assign_magnetic_field)
            - 'Bn': Plot north magnetic field component (requires assign_magnetic_field)
            - 'Bu': Plot up magnetic field component (requires assign_magnetic_field)
            - 'Br': Plot radial magnetic field component (requires assign_magnetic_field)
            - 'Btheta': Plot colatitude magnetic field component (requires assign_magnetic_field)
            - 'Bphi': Plot azimuth magnetic field component (requires assign_magnetic_field)
            - 'inclination': Plot magnetic inclination (requires assign_magnetic_field)
            - 'declination': Plot magnetic declination (requires assign_magnetic_field)
            - 'density': Plot atmospheric density (requires assign_atmosphere)
            - 'pressure': Plot atmospheric pressure (requires assign_atmosphere)
            - 'velocity': Plot atmospheric velocity (requires assign_atmosphere)
            - 'temperature': Plot atmospheric temperature (requires assign_atmosphere)
        altitude_slice : float, optional
            Altitude level (km) to plot for 2D maps. If None, plots all grid points or
            vertical profiles depending on the variable.
        **kwargs : dict
            Additional plotting parameters passed to matplotlib functions.
            
        Examples:
        ---------
        # Plot grid points
        model.plot_variable('grid_points')
        
        # Plot electron density at 300 km altitude
        model.plot_variable('electron_density', altitude_slice=300)
        
        # Plot magnetic inclination at 100 km altitude
        model.plot_variable('inclination', altitude_slice=100)
        
        # Plot vertical profile at source location
        model.plot_variable('electron_density')
        """
        if not hasattr(self, 'grid'):
            raise AttributeError("3D grid not created. Call make_3Dgrid() first.")
            
        if variable == 'grid_points':
            # Plot grid points (existing functionality)
            if altitude_slice is None:
                self.plot_grid_3d()
            else:
                self.plot_grid()
                
        elif variable == 'electron_density':
            self._plot_electron_density(altitude_slice, **kwargs)
            
        elif variable in ['Be', 'Bn', 'Bu', 'Br', 'Btheta', 'Bphi', 'inclination', 'declination']:
            self._plot_magnetic_field_variable(variable, altitude_slice, **kwargs)
            
        elif variable in ['density', 'pressure', 'velocity', 'temperature']:
            self._plot_atmospheric_variable(variable, altitude_slice, **kwargs)
            
        else:
            available_vars = ['grid_points', 'electron_density', 'Be', 'Bn', 'Bu', 'inclination', 'declination', 'density', 'pressure', 'velocity', 'temperature']
            raise ValueError(f"Unknown variable '{variable}'. Available variables: {available_vars}")
    
    def _plot_electron_density(self, altitude_slice=None, **kwargs):
        """Plot electron density."""
        if 'electron_density' not in self.grid.data_vars:
            raise AttributeError("Electron density not computed. Call assign_ionosphere() first.")
        
        if altitude_slice is not None:
            # Plot 2D map at specified altitude
            self._plot_2d_map('electron_density', altitude_slice, 
                            title_prefix='Electron Density', 
                            units='m⁻³', log_scale=True, **kwargs)
        else:
            # Plot vertical profile at source location
            self._plot_vertical_profile('electron_density', 
                                      title_prefix='Electron Density',
                                      units='m⁻³', log_scale=True, **kwargs)
    
    def _plot_atmospheric_variable(self, variable, altitude_slice=None, **kwargs):
        """Plot atmospheric variables."""
        if not hasattr(self, 'atmosphere'):
            raise AttributeError("Atmospheric data not computed. Call assign_atmosphere() first.")
        
        # For atmospheric variables, we only have 1D profiles at source location
        if altitude_slice is not None:
            print("Warning: altitude_slice not supported for atmospheric variables. "
                  "Atmospheric data is only available as 1D profile at source location.")
        
        # Plot 1D atmospheric profile
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        
        atmos_data = self.atmosphere.atmosphere[variable]
        altitudes = self.atmosphere.alt_km
        
        if variable in ['density', 'pressure']:
            ax.semilogx(atmos_data, altitudes, **kwargs)
        else:
            ax.plot(atmos_data, altitudes, **kwargs)
            
        var_labels = {
            'density': 'Density (kg/m³)',
            'pressure': 'Pressure (Pa)', 
            'velocity': 'Velocity (km/s)',
            'temperature': 'Temperature (K)'
        }
        
        ax.set_xlabel(var_labels.get(variable, variable))
        ax.set_ylabel('Altitude (km)')
        ax.set_title(f'Atmospheric {variable.title()} Profile\n'
                    f'Lat: {self.source.get_latitude():.2f}°, '
                    f'Lon: {self.source.get_longitude():.2f}°')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_magnetic_field_variable(self, variable, altitude_slice=None, **kwargs):
        """Plot magnetic field variables."""
        if variable not in self.grid.data_vars:
            raise AttributeError(f"Magnetic field variable '{variable}' not computed. Call assign_magnetic_field() first.")
        
        # Define variable properties
        var_properties = {
            'Be': {'title': 'East Magnetic Field Component', 'units': 'nT', 'log_scale': False},
            'Bn': {'title': 'North Magnetic Field Component', 'units': 'nT', 'log_scale': False},
            'Bu': {'title': 'Up Magnetic Field Component', 'units': 'nT', 'log_scale': False},
            'Br': {'title': 'Radial Magnetic Field Component', 'units': 'nT', 'log_scale': False},
            'Btheta': {'title': 'Colatitude Magnetic Field Component', 'units': 'nT', 'log_scale': False},
            'Bphi': {'title': 'Azimuth Magnetic Field Component', 'units': 'nT', 'log_scale': False},
            'inclination': {'title': 'Magnetic Inclination', 'units': 'degrees', 'log_scale': False},
            'declination': {'title': 'Magnetic Declination', 'units': 'degrees', 'log_scale': False}
        }
        
        props = var_properties.get(variable, {'title': variable.title(), 'units': '', 'log_scale': False})
        
        if altitude_slice is not None:
            # Plot 2D map at specified altitude
            self._plot_2d_map(variable, altitude_slice, 
                            title_prefix=props['title'], 
                            units=props['units'], 
                            log_scale=props['log_scale'], **kwargs)
        else:
            # Plot vertical profile at source location
            self._plot_vertical_profile(variable, 
                                      title_prefix=props['title'],
                                      units=props['units'], 
                                      log_scale=props['log_scale'], **kwargs)
    
    def _plot_2d_map(self, variable, altitude, title_prefix, units, log_scale=False, **kwargs):
        """Plot 2D map of a variable at specified altitude."""
        # Find closest altitude level
        altitudes = self.grid.coords['altitude'].values
        alt_idx = np.argmin(np.abs(altitudes - altitude))
        actual_altitude = altitudes[alt_idx]
        
        # Extract 2D slice
        data_2d = self.grid[variable].isel(altitude=alt_idx)
        
        # Create map
        latitudes = self.grid.coords['latitude'].values
        longitudes = self.grid.coords['longitude'].values
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), 
                              subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Set map extent
        extent = [longitudes.min(), longitudes.max(), 
                 latitudes.min(), latitudes.max()]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, alpha=0.3)
        ax.add_feature(cfeature.OCEAN, alpha=0.3)
        
        # Create meshgrid for plotting
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
        
        # Plot data
        if log_scale:
            im = ax.contourf(lon_grid, lat_grid, data_2d.values, 
                           levels=20, transform=ccrs.PlateCarree(), 
                           norm=plt.cm.colors.LogNorm(), **kwargs)
        else:
            im = ax.contourf(lon_grid, lat_grid, data_2d.values, 
                           levels=20, transform=ccrs.PlateCarree(), **kwargs)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(f'{title_prefix} ({units})')
        
        # Plot source location
        if hasattr(self, 'source'):
            ax.plot(self.source.get_longitude(), self.source.get_latitude(), 
                   'k*', markersize=15, transform=ccrs.PlateCarree(), 
                   label='Source')
            ax.legend()
        
        # Add gridlines
        ax.gridlines(draw_labels=True, alpha=0.5)
        
        plt.title(f'{title_prefix} at {actual_altitude:.1f} km\n'
                 f'Time: {self.source.get_time() if hasattr(self, "source") else "N/A"}')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_vertical_profile(self, variable, title_prefix, units, log_scale=False, **kwargs):
        """Plot vertical profile of a variable at source location."""
        if not hasattr(self, 'source'):
            raise AttributeError("Source not assigned to the model.")
        
        # Find closest grid point to source
        latitudes = self.grid.coords['latitude'].values
        longitudes = self.grid.coords['longitude'].values
        
        lat_idx = np.argmin(np.abs(latitudes - self.source.get_latitude()))
        lon_idx = np.argmin(np.abs(longitudes - self.source.get_longitude()))
        
        # Extract vertical profile
        profile = self.grid[variable].isel(latitude=lat_idx, longitude=lon_idx)
        altitudes = self.grid.coords['altitude'].values
        
        # Plot profile
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        
        if log_scale:
            ax.semilogx(profile.values, altitudes, **kwargs)
        else:
            ax.plot(profile.values, altitudes, **kwargs)
            
        ax.set_xlabel(f'{title_prefix} ({units})')
        ax.set_ylabel('Altitude (km)')
        ax.set_title(f'{title_prefix} Vertical Profile\n'
                    f'Lat: {self.source.get_latitude():.2f}°, '
                    f'Lon: {self.source.get_longitude():.2f}°, '
                    f'Time: {self.source.get_time()}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
