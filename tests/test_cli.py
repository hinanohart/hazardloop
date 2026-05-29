"""S5 CLI tests via Typer's CliRunner (no network: mock backend only)."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from hazardloop.cli import app

runner = CliRunner()


def test_doctor_mock_reports_synthetic() -> None:
    result = runner.invoke(app, ["doctor", "--backend", "mock"])
    assert result.exit_code == 0
    assert "synthetic=True" in result.stdout


def test_fit_mock_text() -> None:
    result = runner.invoke(app, ["fit", "--backend", "mock", "--limit", "400", "--seed", "0"])
    assert result.exit_code == 0
    assert "cif_mode=synthetic" in result.stdout
    assert "CIF causes" in result.stdout


def test_fit_mock_json_is_valid() -> None:
    result = runner.invoke(app, ["fit", "--backend", "mock", "--limit", "300", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["n_runs"] == 300
    assert payload["cif_mode"] == "synthetic"
    assert payload["observed_events"] is True


def test_replay_mock() -> None:
    result = runner.invoke(app, ["replay", "--backend", "mock", "--limit", "500", "--seed", "0"])
    assert result.exit_code == 0
    assert "premature-abort rate" in result.stdout
    assert "evaluated on test" in result.stdout


def test_control_mock() -> None:
    result = runner.invoke(
        app, ["control", "--backend", "mock", "--limit", "400", "--abort-threshold", "1.0"]
    )
    assert result.exit_code == 0
    assert "abort_threshold=1.0" in result.stdout


def test_compare_is_stub_with_nonzero_exit() -> None:
    result = runner.invoke(app, ["compare"])
    assert result.exit_code == 2
    assert "deferred to v0.2" in result.output


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("fit", "replay", "control", "doctor", "compare"):
        assert cmd in result.stdout
