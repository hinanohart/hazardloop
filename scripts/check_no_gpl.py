#!/usr/bin/env python3
"""Fail the build if a GPL-licensed survival library is referenced anywhere.

hazardloop re-implements Kaplan-Meier / Nelson-Aalen / Aalen-Johansen / Weibull-AFT on
numpy specifically so it can stay Apache-2.0 clean. ``scikit-survival`` (import name
``sksurv``) is GPL-3.0 and must never appear in source, dependency metadata, or extras.

Scope: ``src/`` recursively and ``pyproject.toml``. (``scripts/`` is excluded so this
checker's own banned-token literals do not trip it.)

Exit 0 = clean, 1 = GPL reference found.
"""

from __future__ import annotations

import sys
from pathlib import Path

GPL_TOKENS: tuple[str, ...] = ("scikit-survival", "scikit_survival", "sksurv")


def find_gpl_references(root: Path) -> list[str]:
    hits: list[str] = []
    targets: list[Path] = []
    src = root / "src"
    if src.is_dir():
        targets.extend(sorted(src.rglob("*.py")))
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        targets.append(pyproject)

    for path in targets:
        try:
            text = path.read_text()
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            low = line.lower()
            for tok in GPL_TOKENS:
                if tok in low:
                    hits.append(f"{path}:{lineno}: GPL reference {tok!r}: {line.strip()}")
    return hits


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    if argv:
        root = Path(argv[0])
    hits = find_gpl_references(root)
    if not hits:
        print("no-gpl: OK (scikit-survival / sksurv absent from src and pyproject)")
        return 0
    for h in hits:
        print(f"no-gpl VIOLATION: {h}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
