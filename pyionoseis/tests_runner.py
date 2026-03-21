"""Run the PyIonoSeis test suite from an installed package."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


def _resolve_tests_dir() -> Path | None:
    try:
        import tests  # type: ignore
    except ImportError:
        return None

    tests_file = getattr(tests, "__file__", None)
    if not tests_file:
        return None

    return Path(tests_file).resolve().parent


def main() -> int:
    tests_dir = _resolve_tests_dir()
    if tests_dir is None or not tests_dir.exists():
        print(
            "Tests are not installed. Install from a source checkout or a "
            "distribution that includes tests.",
            file=sys.stderr,
        )
        return 2

    suite = unittest.defaultTestLoader.discover(
        start_dir=str(tests_dir),
        pattern="test_*.py",
        top_level_dir=str(tests_dir.parent),
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
