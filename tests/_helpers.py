"""Shared test fixtures and a hand-computed golden dataset."""

from __future__ import annotations

from hazardloop.types import SurvivalRecord, TerminationMode

# --- Golden dataset (5 runs, mode-A failure-as-event). Hand-computed expectations: -------
#   event times          : [1, 2, 4]
#   KM survival          : [0.8, 0.6, 0.0]
#   Nelson-Aalen cum-haz : [0.2, 0.45, 1.45]
#   CIF wrong_patch      : [0.2, 0.2, 0.8]
#   CIF tool_error       : [0.0, 0.2, 0.2]
#   total CIF == 1 - S   : [0.2, 0.4, 1.0]
GOLDEN: list[SurvivalRecord] = [
    SurvivalRecord(duration=1.0, terminal_mode=TerminationMode.WRONG_PATCH),
    SurvivalRecord(duration=2.0, terminal_mode=TerminationMode.SOLVED),  # censored under mode-A
    SurvivalRecord(duration=2.0, terminal_mode=TerminationMode.TOOL_ERROR),
    SurvivalRecord(duration=3.0, terminal_mode=TerminationMode.SOLVED),  # censored
    SurvivalRecord(duration=4.0, terminal_mode=TerminationMode.WRONG_PATCH),
]

GOLDEN_EVENT_TIMES = [1.0, 2.0, 4.0]
GOLDEN_KM_SURVIVAL = [0.8, 0.6, 0.0]
GOLDEN_NA_CUMHAZ = [0.2, 0.45, 1.45]
GOLDEN_CIF_WRONG = [0.2, 0.2, 0.8]
GOLDEN_CIF_TOOL = [0.0, 0.2, 0.2]
