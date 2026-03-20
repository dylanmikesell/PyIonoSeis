"""Run tests from a repo checkout with robust path handling."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


def _find_repo_root(start: Path) -> Path | None:
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def main() -> int:
    cwd = Path.cwd()
    repo_root = _find_repo_root(cwd)
    if repo_root is None:
        print(
            "Could not find pyproject.toml. Run this from a PyIonoSeis repo "
            "checkout.",
            file=sys.stderr,
        )
        return 2

    tests_dir = repo_root / "tests"
    if not tests_dir.exists():
        print("Tests directory not found at repo root.", file=sys.stderr)
        return 2

    suite = unittest.defaultTestLoader.discover(str(tests_dir))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
