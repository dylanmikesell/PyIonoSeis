# Changelog

## [Unreleased]

### Changed

- `Model3D` now inherits from `ModelPlotMixin` — all plotting methods
  (`plot_source`, `plot_grid`, `plot_grid_3d`, `plot_variable`, and private
  helpers) have been moved to `pyionoseis/model_plot.py`. The public API is
  unchanged.
- `make_3Dgrid()` now always initialises `model.grid` as an `xr.Dataset`
  (previously a `DataArray`). The `grid_points` variable holds the zero array.
  All subsequent `assign_*` methods add variables directly to the dataset without
  type-checking.
- `assign_ionosphere()` and `assign_magnetic_field()` now use
  `concurrent.futures.ThreadPoolExecutor` to evaluate 1-D profiles in parallel
  across the lat/lon grid. Wall time scales sub-linearly with grid size on
  multi-core systems.
- `assign_ionosphere()` and `assign_magnetic_field()` accept an optional
  `max_workers` keyword argument (default `None` → `os.cpu_count()`).
- Progress output (formerly `print()` calls) now goes through Python's standard
  `logging` module at the `INFO` level. Enable with
  `logging.basicConfig(level=logging.INFO)`.
- Six private static methods (`_array_sha256`, `_canonical_json_hash`,
  `_normalize_signature_payload`, `_signature_path_for_output_prefix`,
  `_signature_path_for_raypaths`, `_cache_token`) have been extracted from
  `Model3D` and are now module-level functions in `pyionoseis/model_io.py`.
- Duplicate `warnings.warn` blocks in `trace_rays` consolidated into
  `_warn_az_interp_approximation()`.
- `_build_ray_signature_payload` parameter renamed from `type` to `run_type`
  to avoid shadowing the Python built-in.

### Added

- `pyionoseis/model_io.py` — new module with pure caching and IO helper
  functions: `array_sha256`, `canonical_json_hash`, `normalize_signature_payload`,
  `signature_path_for_output_prefix`, `signature_path_for_raypaths`, `cache_token`.
- `pyionoseis/model_plot.py` — new `ModelPlotMixin` class containing all
  `Model3D` visualisation methods.
- Module-level thread worker functions `_iono_profile_worker` and
  `_magfield_profile_worker` in `model.py` (required for `ThreadPoolExecutor`
  pickling safety).

## [0.0.1] - TBD

### Added

- Initial release.
- `Model3D` — 3-D grid orchestrator with atmosphere, ionosphere, magnetic
  field, and infraGA spherical ray-tracing support.
- `EarthquakeSource` — TOML-driven earthquake event loader.
- `Atmosphere1D` — MSISE-00 neutral atmosphere vertical profiles.
- `Ionosphere1D` — IRI2020 electron density vertical profiles (optional).
- `MagneticField1D` — IGRF geomagnetic field via ppigrf (optional).
- `infraga` module — subprocess wrapper for `infraga-sph` /
  `infraga-sph-rngdep` binaries with SHA-256 signature caching.
- CLI entry point: `pyionoseis`.
