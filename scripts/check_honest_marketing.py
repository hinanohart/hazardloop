#!/usr/bin/env python3
"""Honest-marketing enforcement for hazardloop documentation.

Implemented in Python (not shell ``grep``) deliberately: shell BRE/ERE alternation is a
well-known source of silently-dead checks. Every rule here is a plain substring or
compiled-regex test with a paired positive/negative unit test in
``tests/test_honest_marketing.py``.

Self-reference safety: this script lives in ``scripts/`` and is **never** scanned by
itself. Only ``README.md`` and ``docs/*.md`` are scanned, so the banned-phrase literals
defined below do not trip the check.

Exit code 0 = clean, 1 = at least one violation (printed to stderr).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --- Banned phrases (case-insensitive substring). Absent from scanned docs. ---------------
# NB: bare "first" is intentionally NOT banned (false-positives on "first split" etc.);
# only the over-claiming limited forms are. See bootstrap protocol BP6(b).
BANNED_HYPE: tuple[str, ...] = (
    "state-of-the-art",
    "fully automatic",
    "fully automated",
    "world's first",
    "first survival analysis",
    "first competing-risk",
    "first competing risk",
    "improve success rate",
    "improves success rate",
    "improved success rate",
    "increase pass@k",
    "boost pass@k",
    "完全自動",
    "永続",
)

# METR must never be asserted wrong/incorrect (R5: we only say estimators "can diverge").
BANNED_METR_ASSERTION: tuple[str, ...] = (
    "metr is wrong",
    "metr is incorrect",
    "proves metr",
    "metr methodology is wrong",
)

# Placeholder markers that must be gone by the README-finalisation step (S7).
PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "<!-- MEASURED@",
    "<!-- QUICKSTART@",
    "<!-- MOTIVATION@",
    "TODO",
    "FIXME",
    "XXX",
)

# Disclaimer strings that must be PRESENT somewhere in the scanned docs (union).
REQUIRED_DISCLAIMERS: tuple[str, ...] = (
    "UNVERIFIED",
    "not a trained controller",
)

# BP1: if any bench_results/*.json reports a synthetic typed-CIF, this exact disclaimer
# must be present in the docs.
SYNTHETIC_CIF_DISCLAIMER: str = (
    "typed competing-risk CIF in this release is synthetic-validation-only"
)

# Drift guard: a PEP 440 pre-release version literal (e.g. 0.1.0a3). Every such literal in
# the source tree and README must equal the canonical __version__ — otherwise a stale value
# (the kind a context compaction / mid-release edit leaves behind) silently ships in user-
# facing docstrings, CLI help, or the README status badge. CHANGELOG keeps historical tags.
VERSION_LITERAL_RE = re.compile(r"\b\d+\.\d+\.\d+(?:a|b|rc)\d+\b")


@dataclass
class Report:
    violations: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.violations.append(msg)

    @property
    def ok(self) -> bool:
        return not self.violations


def scan_docs(
    docs: dict[str, str],
    *,
    check_placeholders: bool,
    require_disclaimers: bool,
) -> Report:
    """Run the marketing rules over ``docs`` (mapping of filename -> text).

    Pure function over an in-memory mapping so unit tests can feed fixtures directly.
    """
    rep = Report()
    joined_lower = "\n".join(docs.values()).lower()

    for phrase in BANNED_HYPE:
        for name, text in docs.items():
            if phrase.lower() in text.lower():
                rep.fail(f"banned hype phrase {phrase!r} found in {name}")
    for phrase in BANNED_METR_ASSERTION:
        for name, text in docs.items():
            if phrase.lower() in text.lower():
                rep.fail(f"banned METR assertion {phrase!r} found in {name}")
    if check_placeholders:
        for marker in PLACEHOLDER_MARKERS:
            for name, text in docs.items():
                if marker in text:
                    rep.fail(f"unresolved placeholder {marker!r} found in {name}")
    if require_disclaimers:
        for disclaimer in REQUIRED_DISCLAIMERS:
            if disclaimer.lower() not in joined_lower:
                rep.fail(f"required disclaimer {disclaimer!r} missing from docs")
    return rep


def bench_results_have_synthetic_cif(bench_dir: Path) -> bool:
    """True if any bench_results/*.json declares ``cif_mode == 'synthetic'`` (BP1)."""
    if not bench_dir.is_dir():
        return False
    for jf in sorted(bench_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and data.get("cif_mode") == "synthetic":
            return True
    return False


def check_synthetic_disclaimer(docs: dict[str, str], bench_dir: Path) -> Report:
    """BP1: synthetic typed-CIF in any bench result requires the disclaimer in docs."""
    rep = Report()
    if bench_results_have_synthetic_cif(bench_dir):
        joined = "\n".join(docs.values()).lower()
        if SYNTHETIC_CIF_DISCLAIMER.lower() not in joined:
            rep.fail(
                "bench_results contain cif_mode=='synthetic' but the synthetic-CIF "
                f"disclaimer is absent from docs: {SYNTHETIC_CIF_DISCLAIMER!r}"
            )
    return rep


def _canonical_version(root: Path) -> str | None:
    """The single source of truth: ``__version__`` in ``src/hazardloop/__init__.py``."""
    init = root / "src" / "hazardloop" / "__init__.py"
    if not init.is_file():
        return None
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init.read_text())
    return m.group(1) if m else None


def check_version_consistency(root: Path) -> Report:
    """No source/README version literal may disagree with the canonical ``__version__``.

    Scans ``src/hazardloop/**/*.py`` and ``README.md`` for PEP 440 pre-release version
    literals; every one must equal ``__version__`` (the canonical assignment line is
    exempt). Catches stale leftovers in docstrings, CLI help, and the README status badge —
    the kind of drift a context compaction or partial bump leaves behind. CHANGELOG and
    other docs are not scanned (they legitimately reference historical tags).
    """
    rep = Report()
    version = _canonical_version(root)
    if version is None:
        return rep
    targets: list[Path] = []
    src = root / "src" / "hazardloop"
    if src.is_dir():
        targets.extend(sorted(src.rglob("*.py")))
    readme = root / "README.md"
    if readme.is_file():
        targets.append(readme)
    for path in targets:
        try:
            text = path.read_text()
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "__version__" in line and "=" in line:
                continue  # the canonical definition itself
            for literal in VERSION_LITERAL_RE.findall(line):
                if literal != version:
                    rep.fail(
                        f"stale version literal {literal!r} (canonical __version__ is "
                        f"{version!r}) in {path.relative_to(root)}:{lineno}"
                    )
    return rep


def load_docs(root: Path) -> dict[str, str]:
    docs: dict[str, str] = {}
    readme = root / "README.md"
    if readme.is_file():
        docs[str(readme.relative_to(root))] = readme.read_text()
    docs_dir = root / "docs"
    if docs_dir.is_dir():
        for md in sorted(docs_dir.glob("*.md")):
            docs[str(md.relative_to(root))] = md.read_text()
    return docs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="hazardloop honest-marketing check")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument(
        "--no-placeholders",
        action="store_true",
        help="fail on unresolved placeholder markers (enforced from S7)",
    )
    parser.add_argument(
        "--require-disclaimers",
        action="store_true",
        help="fail if required disclaimer strings are absent (enforced from S7)",
    )
    args = parser.parse_args(argv)

    root: Path = args.root
    docs = load_docs(root)
    if not docs:
        print(f"ERROR: no docs found under {root}", file=sys.stderr)
        return 1

    rep = scan_docs(
        docs,
        check_placeholders=args.no_placeholders,
        require_disclaimers=args.require_disclaimers,
    )
    rep.violations.extend(check_synthetic_disclaimer(docs, root / "bench_results").violations)
    rep.violations.extend(check_version_consistency(root).violations)

    if rep.ok:
        print(f"honest-marketing: OK ({len(docs)} docs scanned)")
        return 0
    for v in rep.violations:
        print(f"honest-marketing VIOLATION: {v}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
