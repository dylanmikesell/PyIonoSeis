---
applyTo: "**/*.ipynb"
description: "Conventions for PyIonoSeis example and documentation notebooks"
name: "Notebook Guidelines"
---
# Notebook Guidelines

These instructions apply to all Jupyter notebooks in `example/` and `docs/examples/`. Notebooks serve as executable documentation and are rendered on the GitHub Pages site via `mkdocs-jupyter`.

## Notebook Purpose

Every notebook must have a clear single purpose:

- `example/` notebooks → end-to-end workflows (model setup → compute → visualize).
- `docs/examples/` notebooks → conceptual explanations with minimal code.

Do not mix multiple unrelated workflows in one notebook.

## Cell Structure

Follow this ordering within a notebook:

1. **Title cell (Markdown)**: `# Title` + one-paragraph explanation of what the notebook demonstrates.
2. **Imports cell**: all `import` statements together; no inline imports in later cells.
3. **Parameters cell**: physical constants, event parameters, altitude arrays — clearly commented with units.
4. **Computation cells**: one logical step per cell; use a Markdown header cell before each major step.
5. **Visualization cells**: one plot per cell; always label axes with units.

```python
# --- Parameters ---
glat = 37.7749        # degrees N
glon = -122.4194      # degrees E
alt_km = np.arange(0, 500, 1.0)   # km
time = datetime(2023, 10, 1, 12, 34, 56)
```

## Model3D Workflow Pattern

Notebooks that demonstrate the full pipeline must follow this order:

```python
from pyionoseis.model import Model3D
from pyionoseis.source import EarthquakeSource

source = EarthquakeSource("path/to/event.toml")
model = Model3D("path/to/event.toml")
model.assign_source(source)
model.make_3Dgrid()
model.assign_atmosphere()
model.assign_ionosphere()
# ... then trace_rays / plot_variable / etc.
```

## Visualization

- Always set axis labels including units: `ax.set_xlabel("Altitude (km)")`.
- Use `matplotlib` inline; end visualization cells with `plt.show()` (needed for mkdocs-jupyter execution).
- Save publication figures with `fig.savefig("filename.png", dpi=150, bbox_inches="tight")` only if the image is referenced in the docs.
- Prefer `xr.Dataset.plot()` or `xr.DataArray.plot()` for quick profile plots; switch to explicit `matplotlib` when multi-panel or cartopy maps are needed.

## Execution Requirements

Notebooks on the site are executed by `mkdocs-jupyter` during `mkdocs build`. Every notebook must:

- Run to completion without errors in the CI environment.
- Not depend on local file paths outside the repository.
- Not require infraGA binaries unless the cell is clearly marked with a comment warning and has a graceful skip.

```python
# This cell requires infraGA binaries to be installed. Skip if unavailable.
from pyionoseis.infraga import accel_sph_available
if not accel_sph_available():
    print("infraGA not available — skipping ray trace.")
else:
    model.trace_rays(...)
```
