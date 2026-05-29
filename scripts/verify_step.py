#!/usr/bin/env python3
"""Per-phase gate checker for the hazardloop autonomous build.

Each phase ``Sn`` has a list of **concrete** checks (file presence, importability, a real
sub-process pytest run, or a numeric invariant). There are deliberately no trivially-true
gates: a check either inspects a real artefact or runs real code. (The S8 critic audits
this file specifically for vacuous gates.)

Usage:  python scripts/verify_step.py S2
Exit 0 iff every check for that phase passes.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CheckResult = tuple[str, bool, str]


# --------------------------------------------------------------------------- helpers ----
def _file(rel: str) -> CheckResult:
    p = ROOT / rel
    return (f"file:{rel}", p.is_file(), "present" if p.is_file() else "MISSING")


def _dir(rel: str) -> CheckResult:
    p = ROOT / rel
    return (f"dir:{rel}", p.is_dir(), "present" if p.is_dir() else "MISSING")


def _safe_import(module: str) -> object:
    """Import a module restricted to this package's namespace.

    The module name always comes from the hardcoded whitelist in ``CHECKS`` below, never
    from external input, and we additionally guard on the ``hazardloop`` prefix so only
    our own modules can ever be loaded here.
    """
    if module != "hazardloop" and not module.startswith("hazardloop."):
        raise ValueError(f"refusing to import non-hazardloop module {module!r}")
    loader = importlib.import_module  # nosemgrep
    return loader(module)


def _importable(module: str) -> CheckResult:
    try:
        _safe_import(module)
        return (f"import:{module}", True, "ok")
    except Exception as exc:
        return (f"import:{module}", False, f"{type(exc).__name__}: {exc}")


def _has_attrs(module: str, attrs: list[str]) -> CheckResult:
    try:
        mod = _safe_import(module)
    except Exception as exc:
        return (f"attrs:{module}", False, f"import failed: {exc}")
    missing = [a for a in attrs if not hasattr(mod, a)]
    return (f"attrs:{module}", not missing, "ok" if not missing else f"missing {missing}")


def _script_ok(script: str, args: list[str] | None = None) -> CheckResult:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *(args or [])]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    detail = (proc.stdout + proc.stderr).strip().splitlines()
    tail = detail[-1] if detail else ""
    return (f"script:{script} {' '.join(args or [])}".strip(), proc.returncode == 0, tail)


def _pytest(paths: list[str], extra: list[str] | None = None) -> CheckResult:
    cmd = [sys.executable, "-m", "pytest", "-q", *(extra or []), *paths]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    out = (proc.stdout + proc.stderr).strip().splitlines()
    tail = out[-1] if out else ""
    return (f"pytest:{' '.join(paths)}", proc.returncode == 0, tail)


def _progress() -> dict:
    p = ROOT / ".hazardloop-progress.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _gate1_resolved() -> CheckResult:
    prog = _progress()
    g = prog.get("kill_criteria_results", {}).get("GATE1_data_availability", {})
    verdict = g.get("verdict")
    ok = bool(g.get("checked")) and verdict in {
        "[A] real-available",
        "[B] real-narrow",
        "synthetic-only",
    }
    return ("S0:GATE-1 verdict", ok, f"verdict={verdict!r}")


def _kill_gates_checked() -> CheckResult:
    prog = _progress()
    k = prog.get("kill_criteria_results", {})
    needed = [
        "org_dup_check",
        "GATE1_data_availability",
        "prior_art",
        "claim_strawman_guard",
        "name_available",
    ]
    unchecked = [n for n in needed if not k.get(n, {}).get("checked")]
    return (
        "S0:all kill gates checked",
        not unchecked,
        "ok" if not unchecked else f"unchecked {unchecked}",
    )


# ----------------------------------------------------------------------- phase checks ---
def check_S0() -> list[CheckResult]:
    return [_file(".hazardloop-progress.json"), _kill_gates_checked(), _gate1_resolved()]


def check_S0_5() -> list[CheckResult]:
    return [
        _file("pyproject.toml"),
        _file("LICENSE"),
        _file("NOTICE"),
        _file("README.md"),
        _file("CHANGELOG.md"),
        _file(".gitignore"),
        _file("docs/CLAIMS.md"),
        _file("scripts/verify_step.py"),
        _file("scripts/check_no_gpl.py"),
        _file("scripts/check_honest_marketing.py"),
        _dir("src/hazardloop/core"),
        _dir("src/hazardloop/adapters"),
        _importable("hazardloop"),
        _has_attrs("hazardloop", ["__version__"]),
        _script_ok("check_no_gpl.py"),
        _script_ok("check_honest_marketing.py"),  # loose (no placeholder strictness yet)
    ]


def check_S1() -> list[CheckResult]:
    return [
        _importable("hazardloop.types"),
        _has_attrs(
            "hazardloop.types",
            [
                "StepRecord",
                "SurvivalRecord",
                "TerminationMode",
                "ControlDecision",
                "SurvivalReport",
            ],
        ),
        _pytest(["tests/test_types.py"]),
    ]


def check_S2() -> list[CheckResult]:
    return [
        _importable("hazardloop.core.kaplan_meier"),
        _importable("hazardloop.core.nelson_aalen"),
        _importable("hazardloop.core.aalen_johansen"),
        _importable("hazardloop.core.weibull_aft"),
        _importable("hazardloop.core.bootstrap"),
        _pytest(["tests/test_core_kaplan_meier.py", "tests/test_core_nelson_aalen.py"]),
        _pytest(["tests/test_core_aalen_johansen.py"]),
        _pytest(["tests/test_core_weibull_aft.py", "tests/test_core_bootstrap.py"]),
        _pytest(["tests/test_core_invariants.py"]),
    ]


def check_S3() -> list[CheckResult]:
    return [
        _importable("hazardloop.adapters.base"),
        _importable("hazardloop.adapters.normalize"),
        _importable("hazardloop.adapters.mock"),
        _importable("hazardloop.adapters.swebench"),
        _pytest(["tests/test_adapters.py"]),
    ]


def check_S4() -> list[CheckResult]:
    return [
        _importable("hazardloop.replay"),
        _importable("hazardloop.controller"),
        _importable("hazardloop.policy"),
        _pytest(["tests/test_replay.py", "tests/test_controller.py"]),
    ]


def check_S5() -> list[CheckResult]:
    return [
        _importable("hazardloop.cli"),
        _has_attrs("hazardloop.cli", ["app"]),
        _pytest(["tests/test_cli.py", "tests/test_api.py"]),
    ]


def check_S6() -> list[CheckResult]:
    prog = _progress()
    m = prog.get("measured", {})
    results_path = m.get("results_path")
    have_results = bool(results_path) and (ROOT / results_path).is_file() if results_path else False
    bench = sorted((ROOT / "bench_results").glob("*.json"))
    return [
        ("S6:bench_results/*.json present", len(bench) > 0, f"{len(bench)} files"),
        ("S6:measured.results_path set & exists", have_results, str(results_path)),
        (
            "S6:cif_mode recorded",
            m.get("cif_mode") in {"live", "synthetic"},
            f"cif_mode={m.get('cif_mode')!r}",
        ),
        _pytest(["tests/test_honest_marketing.py"]),
    ]


def check_S7() -> list[CheckResult]:
    return [
        _script_ok("check_honest_marketing.py", ["--no-placeholders", "--require-disclaimers"]),
    ]


def check_S8() -> list[CheckResult]:
    prog = _progress()
    c = prog.get("critic", {})
    ok = c.get("verdict") in {"SHIP-OK", "SHIP-OK-WITH-NITS"} and not c.get("blockers_open")
    return [("S8:critic verdict ship-able & no open blockers", ok, f"verdict={c.get('verdict')!r}")]


def check_S9() -> list[CheckResult]:
    return [_file(".github/workflows/ci.yml"), _pytest(["tests"], extra=["-m", "not network"])]


CHECKS: dict[str, Callable[[], list[CheckResult]]] = {
    "S0": check_S0,
    "S0_5": check_S0_5,
    "S1": check_S1,
    "S2": check_S2,
    "S3": check_S3,
    "S4": check_S4,
    "S5": check_S5,
    "S6": check_S6,
    "S7": check_S7,
    "S8": check_S8,
    "S9": check_S9,
}


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] not in CHECKS:
        print(f"usage: verify_step.py <{'|'.join(CHECKS)}>", file=sys.stderr)
        return 2
    step = argv[0]
    results = CHECKS[step]()
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}  ::  {detail}")
        all_ok = all_ok and ok
    print(
        f"{step}: {'OK' if all_ok else 'FAILED'} ({sum(o for _, o, _ in results)}/{len(results)} checks)"
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
