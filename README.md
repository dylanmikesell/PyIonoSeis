# PyIonoSeis

[![image](https://img.shields.io/pypi/v/pyionoseis.svg)](https://pypi.python.org/pypi/pyionoseis)
[![image](https://img.shields.io/conda/vn/conda-forge/pyionoseis.svg)](https://anaconda.org/conda-forge/pyionoseis)

## Overview

A python package for modeling and anlaysis of coseismic ionospheric disturbances.

- Free software: MIT License
- Documentation: <https://dylanmikesell.github.io/pyionoseis>

## Features

- 3D ionosphere and magnetic field model building
- Optional spherical infraGA ray tracing via `Model3D.trace_rays(...)`

## Development Environment

- Canonical dev environment is `.venv` at the repo root.
- `pyionoseis-dev/` is treated as legacy; remove it if you no longer use it.

## Testing

From a repo checkout:

```bash
.venv/bin/python scripts/run_tests.py
```

This is equivalent to:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Post-install testing (tests are packaged):

```bash
pyionoseis-test
```

Dev install with tooling:

```bash
pip install -e .[dev]
```

## Optional infraGA Dependency

PyIonoSeis requires integration with infraGA.

You need linux libraries
- build-essential 
- libfftw3-dev
- openmpi-bin
- libopenmpi-dev

The later two are for the accelerated version of the code using MPI.

You need to clone the git repo for infraga and then go compile the code.

```bash
git clone https://github.com/LANL-Seismoacoustics/infraGA.git
cd infraGA/
make
```

`make accel` is optional.

Then you need to link the build binaries using simlinks so that the python wrappers in PyIonoSeis can find the infraga binaries. We only use spherical without (infraga-sph) and with (infraga-sph-rngdep) winds. 

```bash
BIN_PATH=$(.venv/bin/python - <<'PY'
import infraga.run_infraga as r
print(r.bin_path)
PY
)

mkdir -p "$BIN_PATH"
cp ~/GIT/infraGA/bin/infraga-sph "$BIN_PATH"/
cp ~/GIT/infraGA/bin/infraga-sph-rngdep "$BIN_PATH"/
chmod +x "$BIN_PATH"/infraga-*
```

Then you can verify that the links are correct.

```bash
$ python - <<'PY'
import os, infraga.run_infraga as r
print("bin_path:", r.bin_path)
print("sph exists:", os.path.exists(r.bin_path + "infraga-sph"))
PY
```

You should get something like 

  bin_path: /home/dmi/GIT/PyIonoSeis/.venv/lib/python3.10/site-packages/bin/
  sph exists: True

The old `infraga compile` command is not available in all infraGA versions.
For this project, use the build-and-copy workflow above as the source of truth.

## Ray Tracing Quick Start

Once your `Model3D` object has a source and grid, you can run spherical ray tracing:

```python
from pyionoseis.model import Model3D

model = Model3D()
# ... assign source and build grid ...

raypaths = model.trace_rays(type="3d")
print(raypaths)
```

Notes:

- `type="3d"` runs spherical azimuth sweep propagation using `infraga sph prop`.
- `type="2d"` also uses `infraga sph prop`, but with a single north azimuth.
- `type="2d"` can be expanded into synthetic 3D azimuth coverage with
  `az_interp=True`:

```python
raypaths = model.trace_rays(
    type="2d",
    az_interp=True,
    az_interp_step=1.0,  # synthetic azimuth sampling [deg]
)
```

- This first implementation stores raw ray outputs (including geometry fields)
  and does not yet interpolate rays onto the model grid.
- Automatic ray reuse is only enabled when `output_dir` is set.
- Synthetic 3D from `az_interp=True` is geometric remapping of one 2D solution.
  It assumes axisymmetry around source; travel times/amplitudes are approximate
  when atmosphere/winds vary with azimuth.

### Ray Reuse and Loading

You can cache/reuse ray files for repeated runs with identical inputs:

```python
raypaths = model.trace_rays(
    type="3d",
    output_dir="ray_cache",
    reuse_existing=True,   # default
)
```

To force a rerun even if cache files match:

```python
raypaths = model.trace_rays(
    type="3d",
    output_dir="ray_cache",
    force_recompute=True,
)
```

To load previously generated ray files directly:

```python
raypaths = model.load_rays(
    raypaths_file="ray_cache/infraga_3d_sph_abc123.raypaths.dat",
    arrivals_file="ray_cache/infraga_3d_sph_abc123.arrivals.dat",  # optional
    type="3d",  # optional if filename is standard
)
```
