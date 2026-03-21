# Technical Debt Reduction TODO

Purpose: Track high-value maintainability work identified in codebase review.

## Priority Legend
- P0: Highest impact/risk reduction
- P1: High value, moderate effort
- P2: Nice-to-have cleanup

Status convention: [x] means implementation is complete in repository state.
The linked GitHub issue may still be open until administrative close-out.

## P0: Reduce Complexity in Model3D

- [ ] Split `Model3D` responsibilities into focused helpers/services
  - Issue: #1
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

- [ ] Add targeted orchestrator tests for failure and fallback behavior
  - Issue: #2
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

- [ ] Normalize style and patterns in legacy-style modules
  - Issue: #7
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

- [ ] Add a lightweight periodic debt triage
  - Issue: #5
  - Scope:
    - During releases, review this checklist and re-prioritize
    - Close or split items based on scope and impact
  - Done when:
    - Checklist stays current and actionable

## Suggested Implementation Order
1. P0 orchestrator tests (safety net first)
2. P0 `Model3D` decomposition
3. P1 legacy-style module normalization
4. P2 maintenance workflow (periodic triage cadence)
5. Administrative close-out for completed issues (#3, #4, #6)
