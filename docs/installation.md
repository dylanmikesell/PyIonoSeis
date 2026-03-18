# Installation

## Stable release

To install PyIonoSeis, run this command in your terminal:

```bash
pip install pyionoseis
```

This is the preferred method to install PyIonoSeis, as it will always install the most recent stable release.

If you don't have [pip](https://pip.pypa.io) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide you through the process.

## From sources

To install PyIonoSeis from sources, run this command in your terminal:

```bash
pip install git+https://github.com/dylanmikesell/pyionoseis
```

## Optional infraGA support

To enable infraGA integration, install the optional extra:

```bash
pip install "pyionoseis[infraga]"
```

infraGA includes native C/C++ methods that should be compiled after installation:

```bash
infraga compile
```

If the `infraga` executable is not on your `PATH`, use:

```bash
python -m infraga.cli compile
```

System packages commonly required for infraGA compilation include make, a C++ compiler,
and FFTW development libraries (for example libfftw3-dev on Debian/Ubuntu).
