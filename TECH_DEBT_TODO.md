# Technical Debt Reduction TODO

Purpose: Track high-value maintainability work identified in codebase review.

## Priority Legend
- P0: Highest impact/risk reduction
- P1: High value, moderate effort
- P2: Nice-to-have cleanup

Status convention: [x] means implementation is complete in repository state.
The linked GitHub issue may still be open until administrative close-out.

## P0: Reduce Complexity in Model3D

- [x] Split `Model3D` responsibilities into focused helpers/services
  - Issue: #1
  - Status: implementation complete; issue open pending administrative close-out
  - Progress:
    - Phase 1 complete: extracted ray orchestration internals to `pyionoseis/ray_tracing_orchestrator.py`
      and delegated `Model3D.trace_rays(...)` cache/signature/metadata helpers.
    - Phase 2 complete: extracted grid enrichment orchestration to
      `pyionoseis/grid_enrichment_orchestrator.py` and delegated
      `assign_ionosphere(...)` / `assign_magnetic_field(...)`.
    - Phase 3 complete: extracted continuity orchestration to
      `pyionoseis/continuity_orchestrator.py` and delegated
      `assign_continuity(...)` preconditions/cache/signature/persistence flow.
    - Phase 4 complete: extracted TEC receiver/orbit dispatch resolution to
      `pyionoseis/tec_input_resolver.py` and delegated branch logic from
      `compute_los_tec(...)`.
    - Phase 5 complete: further slimmed TEC wrapper with
      `pyionoseis/tec_orchestrator.py` (dNe fallback, LOS ID synthesis, final
      compute delegation), plus compatibility hardening tests.
    - Validation:
      - Focused gates pass (`tests.test_infraga`, `tests.test_model_orchestrator`,
        `tests.test_model_tec_legacy`, `tests.test_continuity` as exercised per phase).
      - Full suite pass after final cut:
        `.venv/bin/python scripts/run_tests.py` (55 tests, OK).
  - Scope:
    - Extract ray tracing + cache/signature logic from `trace_rays`
    - Extract grid enrichment flows from `assign_ionosphere` and `assign_magnetic_field`
    - Extract TEC input resolution/dispatch from `compute_los_tec`
  - Why:
    - `Model3D` currently combines many concerns, increasing regression risk and review difficulty.
  - Done when:
    - `Model3D` public API remains compatible
    - Major methods are shorter and delegate to isolated units
    - New unit tests cover helper classes/functions

- [x] Add targeted orchestrator tests for failure and fallback behavior
  - Issue: #2
  - Status: implementation complete; issue open pending close-out
  - Progress:
    - Added orchestrator-focused test module: `tests/test_model_orchestrator.py`
    - Added fallback tests for failed ionosphere/magnetic worker profiles storing NaN
    - Added optional-dependency absence coverage for `PPIGRF_AVAILABLE=False`
    - Added continuity cache tests for hit, signature-parameter miss, force recompute, and ray-signature miss
    - Added continuity scalar-mapping-path test when travel time/amplitude are absent on grid
    - Validated with `.venv` unittest runs for focused and related suites
  - Scope:
    - Test fallback-to-NaN behavior for profile worker failures
    - Test optional dependency absence paths
    - Test cache/signature hit/miss/force logic for ray and continuity workflows
  - Why:
    - Complex orchestration currently has less direct coverage than utility modules.
  - Done when:
    - New tests exist under `tests/` and pass in `.venv`
    - Critical branches in `Model3D` are explicitly asserted

## P1: Standardize Error Handling and Logging

- [x] Replace broad `except Exception` blocks with narrower exceptions where feasible
  - Issue: #3
  - Status: implementation complete; issue open pending close-out
  - Progress:
    - Narrowed optional dependency catches in `wavevector.py` and `infraga.py`
    - Narrowed metadata extraction catches in `ionosphere.py`
    - Kept broad worker-loop catches in `model.py` with explicit rationale + contextual warning logs
  - Scope:
    - Review broad catches in `model.py`, `ionosphere.py`, and optional import areas
    - Keep fallback behavior intentional and observable
  - Why:
    - Broad catches can mask root causes and reduce debuggability.
  - Done when:
    - Catch blocks are specific or include clear rationale when broad catches remain
    - Logs include enough context to diagnose failures

- [x] Remove library `print(...)` side effects and use logging consistently
  - Issue: #4
  - Status: implementation complete; issue open pending close-out
  - Progress:
    - Converted helper `print()` methods in `source.py`, `atmosphere.py`, `ionosphere.py`, and `igrf.py` to logger output
    - Converted `Model3D.print_info()` to logger-backed multi-line output
  - Scope:
    - Replace print-based status/info output in core modules with logger-based output
    - Keep user-facing CLI output in CLI layer only
  - Why:
    - Print side effects make automation and diagnostics inconsistent.
  - Done when:
    - Core modules no longer print during normal workflows
    - Logging levels are used consistently (`info`, `warning`, `error`)

## P1: Align Module Conventions Across Old/New Code

- [x] Normalize style and patterns in legacy-style modules
  - Issue: #7
  - Status: implementation complete; issue open pending close-out
  - Progress:
    - Normalized module/class docstrings to consistent NumPy-style sections
    - Removed stale placeholder artifacts (e.g., stray `pass`, stale TODO/comment clutter)
    - Standardized logging helper docstrings and basic typing across source/atmosphere/ionosphere/igrf
    - Added explicit variable-level units attrs where missing in atmosphere/ionosphere/igrf datasets
    - Added `get_depth()` to `EarthquakeSource` for API consistency with model-side source introspection
    - Validated with related `.venv` unittest runs (24 tests, OK)
  - Scope:
    - Review `atmosphere.py`, `ionosphere.py`, `igrf.py`, `source.py`
    - Remove stale TODO/pass artifacts and improve docstring consistency
  - Why:
    - Mixed coding styles increase contributor friction and maintenance cost.
  - Done when:
    - Modules follow consistent typing/docstring/error-handling patterns
    - Dead placeholders are removed

## P2: Clean Public Surface and Template Artifacts

- [x] Resolve placeholder package surface files
  - Issue: #6
  - Status: implementation complete; issue open pending close-out
  - Progress:
    - Removed placeholder modules: `common.py`, `cli.py`, `pyionoseis.py`
    - Updated package/test/docs references to match intended public surface
  - Scope:
    - Decide whether to keep, replace, or remove placeholder behavior in `cli.py`, `common.py`, and `pyionoseis.py`
    - Update related tests accordingly
  - Why:
    - Template-like behavior creates confusion about supported entry points.
  - Done when:
    - Public package entry points reflect intended user workflows
    - Tests assert meaningful behavior instead of scaffolding text

## P2: Maintenance Workflow

- [x] Add a lightweight periodic debt triage
  - Issue: #5
  - Status: implementation complete; issue open pending close-out
  - Progress:
    - Added reusable release triage issue template: `.github/ISSUE_TEMPLATE/release_debt_triage.md`
    - Added explicit release triage checklist below for maintainers
  - Scope:
    - During releases, review this checklist and re-prioritize
    - Close or split items based on scope and impact
  - Done when:
    - Checklist stays current and actionable

### Release Debt Triage Checklist
Run this once per release candidate or tag cut.

1. Open a GitHub issue from template: `Release Debt Triage`.
2. Review all items in this file and verify status accuracy (`[x]` vs `[ ]`).
3. Re-prioritize open items based on current risk and upcoming release goals.
4. Close completed debt issues and update their corresponding TODO entries.
5. Split oversized debt items into smaller follow-up issues when needed.
6. Remove or archive debt items that are no longer relevant.
7. Update the Suggested Implementation Order section.
8. Record outcomes in the triage issue (closed issues, new issues, priority changes).

## Suggested Implementation Order
1. Administrative close-out for completed issues (#1, #2, #3, #4, #5, #6, #7)
