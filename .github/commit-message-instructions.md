<commit-message-guidelines>

<instruction>
Generate commit messages following the Conventional Commits specification (https://www.conventionalcommits.org/). Use clear, specific subjects that describe the change and intent.
</instruction>

<format>
type(optional scope): description

optional body

optional footer(s)
</format>

<types>
  <type name="feat">A new feature</type>
  <type name="fix">A bug fix</type>
  <type name="docs">Documentation only changes</type>
  <type name="style">Formatting only, no behavior change</type>
  <type name="refactor">Code change that is not a fix or feature</type>
  <type name="perf">Performance improvement</type>
  <type name="test">Adding or correcting tests</type>
  <type name="build">Build system or dependency changes</type>
  <type name="ci">CI configuration changes</type>
  <type name="chore">Other changes that do not affect src or tests</type>
  <type name="revert">Reverts a previous commit</type>
</types>

<rules>
1) Use the imperative mood in the subject line
2) Do not capitalize the first letter of the subject line
3) Do not end the subject line with a period
4) Keep the subject line under 72 characters when possible
5) Separate subject from body with a blank line if a body is present
6) Use the body to explain what and why, not how
7) Wrap body lines at 72 characters
8) Use a scope when it adds clarity (module, subsystem, or feature)
9) Write commit messages in English
</rules>

<breaking-changes>
Indicate breaking changes by adding ! after type or scope, or by adding a BREAKING CHANGE footer in the body.
</breaking-changes>

<examples>
  <example type="feat">
    feat(model): add caching for ray trace outputs
    feat: add plotting helpers for electron density
    feat(cli): add list of available profiles
  </example>

  <example type="fix">
    fix(model): handle empty grid in assign_ionosphere
    fix(igrf): correct sign of Btheta component
    fix: prevent crash when cache signature missing
  </example>

  <example type="docs">
    docs(model): update Model3D workflow description
    docs(readme): clarify infraGA build steps
    docs: add quick start for ray tracing
  </example>

  <example type="style">
    style(model): reformat imports to match lint rules
    style: remove trailing whitespace in docs
    style(cli): sort options alphabetically
  </example>

  <example type="refactor">
    refactor(model): extract plotting into mixin
    refactor(io): split hashing helpers into module
    refactor: simplify grid initialization logic
  </example>

  <example type="perf">
    perf(model): parallelize ionosphere profile evaluation
    perf: reduce dataset allocations in grid build
    perf(infraga): avoid repeated signature hashing
  </example>

  <example type="test">
    test(model): add coverage for assign_magnetic_field
    test: add regression for azimuth interpolation warning
    test(infraga): mock subprocess calls in ray tracing
  </example>

  <example type="build">
    build: pin numpy to avoid ABI issues
    build(deps): bump xarray to 2024.1
    build(ci): update Python matrix to 3.12
  </example>

  <example type="ci">
    ci: add docs build workflow
    ci: cache pip dependencies
    ci(lint): add flake8 job
  </example>

  <example type="chore">
    chore: update .gitignore for build artifacts
    chore: remove unused notebooks
    chore(deps): refresh dev requirements
  </example>

  <example type="revert">
    revert: revert "feat(model): add caching for ray trace outputs"
    revert(igrf): revert "fix(igrf): correct sign of Btheta component"
  </example>

  <example type="breaking-change">
    feat(model)!: change grid coordinate naming

    BREAKING CHANGE: grid dimensions renamed from lat, lon to latitude, longitude

    refactor!: drop support for Python 3.10
  </example>
</examples>

</commit-message-guidelines>
