"""S6 honest-marketing checker tests — paired positive/negative (BP1, BP6, DoD #5).

Imports the checker from ``scripts/`` (not a package) by path. Every banned-phrase /
disclaimer / placeholder / synthetic-CIF rule is exercised in both its passing and its
firing direction, so the check can never silently become a dead grep.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "check_honest_marketing", ROOT / "scripts" / "check_honest_marketing.py"
)
assert _spec and _spec.loader
chm = importlib.util.module_from_spec(_spec)
sys.modules["check_honest_marketing"] = chm
_spec.loader.exec_module(chm)


_CLEAN = {
    "README.md": "hazardloop is an estimator; it is not a trained controller. "
    "Any live benefit is UNVERIFIED in this release.",
}


def test_clean_docs_pass_all_rules() -> None:
    rep = chm.scan_docs(_CLEAN, check_placeholders=True, require_disclaimers=True)
    assert rep.ok, rep.violations


def test_banned_hype_fires() -> None:
    bad = {"README.md": _CLEAN["README.md"] + " This is state-of-the-art."}
    rep = chm.scan_docs(bad, check_placeholders=False, require_disclaimers=False)
    assert not rep.ok
    assert any("state-of-the-art" in v for v in rep.violations)


def test_limited_first_claim_fires() -> None:
    bad = {"README.md": _CLEAN["README.md"] + " The first competing-risk tool for agents."}
    rep = chm.scan_docs(bad, check_placeholders=False, require_disclaimers=False)
    assert not rep.ok


def test_metr_assertion_fires() -> None:
    bad = {"README.md": _CLEAN["README.md"] + " This proves METR is wrong."}
    rep = chm.scan_docs(bad, check_placeholders=False, require_disclaimers=False)
    assert not rep.ok
    assert any("METR" in v for v in rep.violations)


def test_placeholder_fires_only_when_strict() -> None:
    doc = {"README.md": _CLEAN["README.md"] + " <!-- MEASURED@S6 -->"}
    assert chm.scan_docs(doc, check_placeholders=False, require_disclaimers=False).ok
    assert not chm.scan_docs(doc, check_placeholders=True, require_disclaimers=False).ok


def test_missing_disclaimer_fires_only_when_required() -> None:
    doc = {"README.md": "hazardloop computes Kaplan-Meier survival curves."}
    assert chm.scan_docs(doc, check_placeholders=False, require_disclaimers=False).ok
    rep = chm.scan_docs(doc, check_placeholders=False, require_disclaimers=True)
    assert not rep.ok
    assert any("UNVERIFIED" in v for v in rep.violations)


# --- BP1: synthetic typed-CIF requires the disclaimer ------------------------------------
def _write_bench(tmp_path: Path, cif_mode: str) -> Path:
    bench = tmp_path / "bench_results"
    bench.mkdir()
    (bench / "r.json").write_text(json.dumps({"cif_mode": cif_mode, "n": 10}))
    return bench


def test_bp1_synthetic_without_disclaimer_fires(tmp_path: Path) -> None:
    bench = _write_bench(tmp_path, "synthetic")
    assert chm.bench_results_have_synthetic_cif(bench) is True
    rep = chm.check_synthetic_disclaimer(_CLEAN, bench)  # _CLEAN lacks the disclaimer
    assert not rep.ok


def test_bp1_synthetic_with_disclaimer_passes(tmp_path: Path) -> None:
    bench = _write_bench(tmp_path, "synthetic")
    docs = {"README.md": _CLEAN["README.md"] + " " + chm.SYNTHETIC_CIF_DISCLAIMER}
    rep = chm.check_synthetic_disclaimer(docs, bench)
    assert rep.ok, rep.violations


def test_bp1_live_does_not_require_disclaimer(tmp_path: Path) -> None:
    bench = _write_bench(tmp_path, "live")
    assert chm.bench_results_have_synthetic_cif(bench) is False
    assert chm.check_synthetic_disclaimer(_CLEAN, bench).ok


# --- rc-level (DoD #5): main() returns nonzero on a bad tree, zero on a good tree ---------
def _make_tree(tmp_path: Path, readme: str, cif_mode: str | None) -> Path:
    (tmp_path / "README.md").write_text(readme)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "CLAIMS.md").write_text("placeholder claims doc")
    if cif_mode is not None:
        bench = tmp_path / "bench_results"
        bench.mkdir()
        (bench / "r.json").write_text(json.dumps({"cif_mode": cif_mode}))
    return tmp_path


def test_main_rc_fires_on_synthetic_without_disclaimer(tmp_path: Path) -> None:
    root = _make_tree(tmp_path, _CLEAN["README.md"], cif_mode="synthetic")
    rc = chm.main(["--root", str(root), "--require-disclaimers"])
    assert rc == 1  # synthetic CIF but no disclaimer -> nonzero


def test_main_rc_passes_on_clean_tree(tmp_path: Path) -> None:
    readme = _CLEAN["README.md"] + " " + chm.SYNTHETIC_CIF_DISCLAIMER
    root = _make_tree(tmp_path, readme, cif_mode="synthetic")
    rc = chm.main(["--root", str(root), "--no-placeholders", "--require-disclaimers"])
    assert rc == 0
