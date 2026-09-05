#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BetSmart Goal Intelligence V1.0.8.8.13 - JSON REDUCED.

FULL is canonical. REDUCED is a strict projection of the same FULL result.
"""
from fichier_py.fonction_totatl_goal_2_full import (
    predict_from_user_input as _predict_full,
    get_valid_date,
    llm_client,
    GOAL_INTELLIGENCE_VERSION,
)

_GOAL_REDUCED_KEYS = [
    "home","away","lambda_home","lambda_away","lambda_total",
    "Over15","Over25","Over35","BTTS",
    "expected_goal_range","most_likely_score",
    "prediction_confidence","low_confidence",
    "explanation","rule_applied","_final_state_version"
]

def _goal_reduced_view(full_result):
    reduced = {k: full_result.get(k) for k in _GOAL_REDUCED_KEYS}
    assert reduced.get("explanation") == full_result.get("explanation")
    return reduced

def predict_from_user_input(*args, output_mode="reduced", **kwargs):
    kwargs.pop("output_mode", None)
    full_result = _predict_full(*args, output_mode="full", **kwargs)
    if str(output_mode).lower() == "full":
        return full_result
    return _goal_reduced_view(full_result)

def build_goal_output_views_v1(full_result):
    reduced = _goal_reduced_view(full_result)
    return {"version": GOAL_INTELLIGENCE_VERSION, "full": full_result, "reduced": reduced}
