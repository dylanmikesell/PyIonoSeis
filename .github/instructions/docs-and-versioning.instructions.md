---
applyTo: "docs/**/*.md"
description: "Documentation style, changelog format, and versioning workflow for PyIonoSeis"
name: "Documentation & Versioning Guidelines"
---
# Documentation & Versioning Guidelines

These instructions apply to all Markdown files in `docs/`. This is a published PyPI package with a GitHub Pages documentation site built by MkDocs Material + `mkdocstrings`.

## Site Structure

| File | Purpose |
|------|---------|
| `docs/index.md` | Landing page — overview and quick-start |
| `docs/installation.md` | Install instructions (pip + optional extras) |
| `docs/usage.md` | Usage guide mirroring the `Model3D` workflow |
| `docs/changelog.md` | Release history (all versions) |
| `docs/contributing.md` | Contribution workflow |
| `docs/examples/*.ipynb` | Executable example notebooks |

## Writing Style

- Use **second-person imperative**: "Run the command", "Assign a source", not "The user should…".
- Code blocks must specify the language: ` ```python `, ` ```bash `, ` ```toml `.
- Every code example must be copy-paste-runnable — include all necessary imports.
- Prefer short paragraphs (2–4 sentences); use bullet lists for steps and options.

## API Documentation

Public classes and functions are auto-documented via `mkdocstrings`. Do not manually duplicate API signatures in prose — instead, reference them:

```markdown
::: pyionoseis.model.Model3D
```

This renders the NumPy docstring from source. Keep docstrings as the single source of truth for API reference.

## Changelog Format

Follow [Keep a Changelog](https://keepachangelog.com) conventions in `docs/changelog.md`:

```markdown
## [0.1.0] - 2026-03-19

### Added
- `Model3D.assign_ionosphere()` now accepts `"iri2020"` as default model.

### Fixed
- TOML loading raised `KeyError` when `[model]` section was missing.

### Changed
- `Atmosphere1D` returns pressure in Pascals; previously was in hPa.
```

Allowed section headers: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`.

## Versioning Workflow

This project uses `bump-my-version` (configured in `pyproject.toml`). Follow this workflow for every release:

```bash
# 1. Update docs/changelog.md — move Unreleased items under new version heading
# 2. Bump the version (patch | minor | major)
bump-my-version bump patch   # or minor / major
# This auto-commits and tags in git.

# 3. Push the commit and tag
git push && git push --tags

# 4. Publish to PyPI (CI/CD or manual)
python -m build
twine upload dist/*
```

Version numbers follow **Semantic Versioning**: `MAJOR.MINOR.PATCH`.  
- `PATCH` — bug fixes only, no API changes.  
- `MINOR` — new backward-compatible features.  
- `MAJOR` — breaking API changes.

## MkDocs Notes

- Run `mkdocs serve` locally to preview changes before committing.
- Notebooks in `docs/examples/` are executed at build time; ensure they run clean.
- Do not edit `site/` — it is generated output.
