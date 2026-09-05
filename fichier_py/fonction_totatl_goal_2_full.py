#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:59:03 2025

@author: bobunda
"""


import json
from joblib import load
from pydantic import BaseModel
from flask import Flask, jsonify, request
from typing import List
import numpy as np
import pandas as pd
import os
from numpy import floating, integer, ndarray
import datetime
import pathlib
from dateutil import parser
import json
from openai import OpenAI
from scipy.stats import poisson
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, Optional


# ======================================================================================
# GOAL-1.0.8.2-PARTIAL-WEB - explicit .env + resilient Web pipeline
# ======================================================================================
# Python does not read a .env file automatically. We load it before any os.getenv(...)
# used by the Goal Intelligence engine. Existing process variables keep priority.
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

_GOAL_ENV_CANDIDATES = []
try:
    _GOAL_ENV_CANDIDATES.append(pathlib.Path(__file__).resolve().parent / ".env")
except Exception:
    pass
try:
    _GOAL_ENV_CANDIDATES.append(pathlib.Path.cwd() / ".env")
except Exception:
    pass

_GOAL_ENV_LOADED_PATHS = []
_GOAL_ENV_SEEN = set()
for _env_path in _GOAL_ENV_CANDIDATES:
    try:
        _resolved = str(_env_path.resolve())
    except Exception:
        _resolved = str(_env_path)
    if _resolved in _GOAL_ENV_SEEN:
        continue
    _GOAL_ENV_SEEN.add(_resolved)
    try:
        if load_dotenv is not None and _env_path.exists():
            # override=False: exported shell variables remain authoritative.
            load_dotenv(dotenv_path=str(_env_path), override=False)
            _GOAL_ENV_LOADED_PATHS.append(_resolved)
    except Exception:
        # Never crash the prediction engine only because dotenv loading failed.
        pass


def _goal_environment_diagnostics():
    """Safe diagnostics: never expose API key values."""
    return {
        "dotenv_available": bool(load_dotenv is not None),
        "dotenv_found": bool(_GOAL_ENV_LOADED_PATHS),
        "dotenv_loaded_paths": list(_GOAL_ENV_LOADED_PATHS),
        "brave_key_loaded": bool(os.getenv("BRAVE_SEARCH_API_KEY", "").strip()),
        "openai_key_loaded": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "realtime_key_loaded": bool(os.getenv("REALTIME_API_KEY", "").strip()),
    }


out_dir="betsmart_goals_out_pl"
def add_rolling_mean(df_in: pd.DataFrame, window: int) -> pd.DataFrame:
    d = df_in.sort_values("Date").copy()
    d[f"home_gf_last{window}"] = d.groupby("HomeTeam")["FTHG"].transform(lambda x: x.shift(1).rolling(window).mean())
    d[f"home_ga_last{window}"] = d.groupby("HomeTeam")["FTAG"].transform(lambda x: x.shift(1).rolling(window).mean())
    d[f"away_gf_last{window}"] = d.groupby("AwayTeam")["FTAG"].transform(lambda x: x.shift(1).rolling(window).mean())
    d[f"away_ga_last{window}"] = d.groupby("AwayTeam")["FTHG"].transform(lambda x: x.shift(1).rolling(window).mean())
    return d

def add_rolling_std(df_in: pd.DataFrame, window: int) -> pd.DataFrame:
    d = df_in.sort_values("Date").copy()
    d[f"home_gf_std{window}"] = d.groupby("HomeTeam")["FTHG"].transform(lambda x: x.shift(1).rolling(window).std())
    d[f"home_ga_std{window}"] = d.groupby("HomeTeam")["FTAG"].transform(lambda x: x.shift(1).rolling(window).std())
    d[f"away_gf_std{window}"] = d.groupby("AwayTeam")["FTAG"].transform(lambda x: x.shift(1).rolling(window).std())
    d[f"away_ga_std{window}"] = d.groupby("AwayTeam")["FTHG"].transform(lambda x: x.shift(1).rolling(window).std())
    return d


def _clip01(p: float) -> float:
    return float(np.clip(p, 1e-6, 1-1e-6))

def _p_over(lam_total: float, line: float) -> float:
    k = int(line)
    return float(1 - poisson.cdf(k, lam_total))

def _p_btts(lam_h: float, lam_a: float) -> float:
    p_h0 = poisson.pmf(0, lam_h)
    p_a0 = poisson.pmf(0, lam_a)
    return float(1 - p_h0 - p_a0 + (p_h0 * p_a0))

def _extract_league_id(match_df: pd.DataFrame):
    for col in ["competition_id", "league", "comp"]:
        if col in match_df.columns:
            try:
                return int(match_df[col].iloc[0])
            except Exception:
                return None
    return None

def _get_filter(config: dict, match_df: pd.DataFrame, market: str):
    base = config.get("default_filter", {"gray_low":0.47,"gray_high":0.55,"disagree_thr":0.18,"require_history":True})
    lid = _extract_league_id(match_df)
    if lid is None:
        return base

    if market == "O25":
        by = config.get("thresholds_o25_by_league", {})
    elif market == "BTTS":
        by = config.get("thresholds_btts_by_league", {})
    else:
        by = {}

    if lid in by:
        m = base.copy()
        m.update(by[lid])
        return m

    return base

def _decision(p_final: float, hist_ok: bool, flt: dict, disagree: float | None = None):
    pred = int(p_final >= 0.5)
    gray = (p_final >= flt["gray_low"]) and (p_final <= flt["gray_high"])
    low = bool(gray or ((not hist_ok) if flt.get("require_history", True) else False) or ((disagree is not None) and (disagree > flt["disagree_thr"])))
    return pred, low


def enforce_llm_output(exp: dict, payload: dict, fallback_fn) -> dict:
    """
    Si le LLM viole les règles (lambda/value/EV/cote...), on fallback.
    """
    forbidden = ["lambda", "value", " ev", "cote", "rentabilité"]
    txt = " ".join([
        str(exp.get("explanation", "")),
        json.dumps(exp.get("recommended_markets", []), ensure_ascii=False)
    ]).lower()

    if any(w in txt for w in forbidden):
        fb = fallback_fn(payload)
        fb["risk_flags"] = list(set(fb.get("risk_flags", []) + ["llm_style_violation"]))
        return fb

    # Obliger mention des équipes si teams existe
    if "teams" in payload:
        home = str(payload["teams"].get("home", "")).lower()
        away = str(payload["teams"].get("away", "")).lower()
        expl_txt = str(exp.get("explanation", "")).lower()
        if home and away and (home not in expl_txt or away not in expl_txt):
            fb = fallback_fn(payload)
            fb["risk_flags"] = list(set(fb.get("risk_flags", []) + ["llm_missing_team_names"]))
            return fb

    return exp
def build_explanation_prompt(result_json: dict) -> str:
    return f"""
Tu es un parieur pro (risk manager). Tu dois produire un plan de mise BET/NO_BET à partir du JSON ci-dessous.

RÈGLES NON-NÉGOCIABLES:
- Pour chaque marché:
  * low_confidence=true => action="NO_BET" (même si pred=1)
  * sinon si pred=0 => action="NO_BET"
  * sinon => action="BET"
- N’invente rien: tu n’utilises QUE ce qui est dans le JSON.
- Interdits: ne dis jamais "lambda", "value", "EV", "cote", "rentabilité".
- Si teams.home et teams.away existent: tu DOIS citer les 2 équipes dans l'explication.
- Si "_debug.*.disagree" existe:
  * faible => signaux alignés
  * élevé => désaccord => risque
- Si Over25 et BTTS sont tous deux BET => ajoute risk_flag "corrélation".

COMMENT ÉCRIRE (pour éviter un texte figé):
- 1ère phrase: annonce le ticket en une ligne avec les équipes (socle + idée générale: match ouvert/fermé).
- 2e phrase: SOCLE = le pari le plus solide (proba la plus haute parmi ceux en BET) + pourquoi en 6-10 mots.
- 3e phrase: SECONDaires (0 à 2 max) + pourquoi (court).
- 4e phrase: EXCLUS (NO_BET) => cite 1 raison factuelle (pred=0 ou low_confidence=true).
- 5e phrase (optionnelle): risque principal (désaccord / corrélation / historique faible).
=> 4 à 6 phrases max, ton direct.

FORMAT DE SORTIE (STRICT JSON, aucune clé en plus):
{{
  "explanation": "<4-6 phrases, ton parieur pro, variable selon le match>",
  "key_points": ["<fait 1>", "<fait 2>", "<fait 3>"],
  "recommended_markets": [
    {{"market":"Over15","action":"BET|NO_BET","reason":"<très court, factuel>"}},
    {{"market":"Over25","action":"BET|NO_BET","reason":"<très court, factuel>"}},
    {{"market":"Over35","action":"BET|NO_BET","reason":"<très court, factuel>"}},
    {{"market":"BTTS","action":"BET|NO_BET","reason":"<très court, factuel>"}}
  ],
  "risk_flags": ["<flag 1>", "<flag 2>"]
}}

JSON À INTERPRÉTER:
{json.dumps(result_json, ensure_ascii=False)}
""".strip()

def enforce_risk_flags(expl: dict, pred_json: dict) -> dict:
    try:
        recs = expl.get("recommended_markets", [])
        bet_markets = {r["market"] for r in recs if r.get("action") == "BET"}

        flags = set(expl.get("risk_flags", []))

        # corrélation classique
        if ("Over25" in bet_markets) and ("BTTS" in bet_markets):
            flags.add("Corrélation: Over25 et BTTS exposés au même scénario")

        expl["risk_flags"] = list(flags)
    except Exception:
        pass
    return expl


def rule_based_explainer(payload: dict) -> dict:
    """
    Fallback sans LLM. Retourne exactement le format:
    {explanation, key_points, recommended_markets, risk_flags}
    En style parieur pro, court.
    """
    def _mk(market: str):
        m = payload.get(market, {})
        pred = int(m.get("pred", 0))
        low = bool(m.get("low_confidence", True))
        proba = float(m.get("proba", 0.0))

        if low or pred == 0:
            action = "NO_BET"
            reason = "low_confidence=true" if low else "pred=0"
        else:
            action = "BET"
            reason = f"pred=1 & low_confidence=false (p={proba:.2f})"
        return {"market": market, "action": action, "reason": reason}

    recs = [_mk("Over15"), _mk("Over25"), _mk("Over35"), _mk("BTTS")]

    # socle = market BET avec proba la plus haute
    bet_recs = []
    for r in recs:
        if r["action"] == "BET":
            p = float(payload[r["market"]]["proba"])
            bet_recs.append((p, r))
    bet_recs.sort(reverse=True, key=lambda x: x[0])

    risk_flags = []

    # corrélation simple: Over25 & BTTS tous les deux BET
    bet_markets = {r["market"] for r in recs if r["action"] == "BET"}
    if ("Over25" in bet_markets) and ("BTTS" in bet_markets):
        risk_flags.append("Corrélation: Over25 et BTTS exposés au même scénario")

    # disagree via _debug si dispo
    dbg = payload.get("_debug", {})
    dis25 = dbg.get("Over25", {}).get("disagree", None)
    disb = dbg.get("BTTS", {}).get("disagree", None)
    if isinstance(dis25, (int, float)) and dis25 > 0.18:
        risk_flags.append(f"O25: désaccord élevé ({dis25:.2f})")
    if isinstance(disb, (int, float)) and disb > 0.18:
        risk_flags.append(f"BTTS: désaccord élevé ({disb:.2f})")

    # phrase parieur pro
    if bet_recs:
        main = bet_recs[0][1]["market"]
        main_p = float(payload[main]["proba"])
        seconds = [x[1]["market"] for x in bet_recs[1:3]]
        sec_txt = ", ".join(seconds) if seconds else "aucun secondaire"
        explanation = (
            f"Socle: {main} (p≈{main_p:.0%}, confiance OK). "
            f"Secondaires: {sec_txt}. "
            f"Tout ce qui est low_confidence ou pred=0 reste NO_BET. "
            f"Surveille la corrélation si plusieurs marchés offensifs passent en BET."
        )
    else:
        explanation = (
            "Aucun marché ne passe les règles (pred=1 et low_confidence=false). "
            "Plan: NO_BET, on évite un ticket forcé."
        )

    key_points = [
        "Décision imposée par pred + low_confidence",
        "Un seul socle, max 2 secondaires",
        "Filtre risque: corrélation / désaccord si présent",
    ]

    return {
        "explanation": explanation,
        "key_points": key_points,
        "recommended_markets": recs,
        "risk_flags": risk_flags,
    }



def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY manquante. "
            "En local: mets-la dans un fichier .env. "
            "Sur Render: ajoute-la dans Environment Variables."
        )
    return OpenAI(api_key=api_key)



def llm_client(prompt: str) -> str:
    _client = get_openai_client()
    resp = _client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Tu réponds STRICTEMENT en JSON valide, sans texte avant/après."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

import re

ALLOWED_MARKETS = ["Over15", "Over25", "Over35", "BTTS"]

def _extract_first_json_object(text: str) -> dict | None:
    """Extrait le premier {...} et tente un json.loads."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def _action_from_rules(pred: int, low_conf: bool) -> str:
    if low_conf:
        return "NO_BET"
    if pred == 0:
        return "NO_BET"
    return "BET"

def _enforce_llm_output(exp: dict, payload: dict) -> dict:
    """
    Corrige le JSON du LLM si:
    - keys manquent
    - recommended_markets contradictoires avec pred/low_confidence
    - risk_flags pas une liste, etc.
    """
    # structure minimale
    if not isinstance(exp, dict):
        exp = {}

    exp.setdefault("explanation", "")
    exp.setdefault("key_points", [])
    exp.setdefault("recommended_markets", [])
    exp.setdefault("risk_flags", [])

    if not isinstance(exp["key_points"], list):
        exp["key_points"] = []
    if not isinstance(exp["recommended_markets"], list):
        exp["recommended_markets"] = []
    if not isinstance(exp["risk_flags"], list):
        exp["risk_flags"] = []

    # construire une map marché -> (pred, low_conf, proba)
    mp = {}
    for m in ALLOWED_MARKETS:
        if m in payload and isinstance(payload[m], dict):
            pred = int(payload[m].get("pred", 0))
            low = bool(payload[m].get("low_confidence", True))
            proba = float(payload[m].get("proba", 0.0))
            mp[m] = (pred, low, proba)
        else:
            mp[m] = (0, True, 0.0)

    # reconstruire recommended_markets de façon canonique (toujours 4 items)
    rec = []
    for m in ALLOWED_MARKETS:
        pred, low, proba = mp[m]
        action = _action_from_rules(pred, low)
        # reason courte & factuelle
        if low:
            reason = "low_confidence=true"
        elif pred == 0:
            reason = "pred=0"
        else:
            reason = f"pred=1 & low_confidence=false (p={proba:.2f})"

        rec.append({"market": m, "action": action, "reason": reason})

    exp["recommended_markets"] = rec

    # corrélation: Over25 & BTTS tous deux BET
    a25 = next(x for x in rec if x["market"] == "Over25")["action"]
    abt = next(x for x in rec if x["market"] == "BTTS")["action"]
    if a25 == "BET" and abt == "BET":
        if "corrélation" not in exp["risk_flags"]:
            exp["risk_flags"].append("corrélation")

    # si teams manquantes dans payload mais le LLM devait les citer
    teams = payload.get("teams")
    if not teams or not teams.get("home") or not teams.get("away"):
        if "llm_missing_team_names" not in exp["risk_flags"]:
            exp["risk_flags"].append("llm_missing_team_names")

    return exp

def llm_explainer(payload: dict, llm_client) -> dict:
    """
    llm_client(prompt:str)->str doit retourner du JSON (string).
    Parser robuste + enforcement anti-incohérences.
    """
    prompt = build_explanation_prompt(payload)

    raw = llm_client(prompt)

    # 1) parse direct
    try:
        exp = json.loads(raw)
    except Exception:
        # 2) parse tolérant: extrait le 1er {...}
        exp = _extract_first_json_object(raw)

    if exp is None:
        # fallback
        return {
            "explanation": "Explication indisponible (réponse LLM non JSON).",
            "key_points": [],
            "recommended_markets": [
                {"market": m, "action": "NO_BET", "reason": "llm_non_json"} for m in ALLOWED_MARKETS
            ],
            "risk_flags": ["llm_non_json"],
        }

    # 3) enforce structure + décisions (anti contradictions)
    return _enforce_llm_output(exp, payload)



def prepare_user_input_and_enrich(df_hist: pd.DataFrame, home: str, away: str, date: str, odds: dict):
    d = df_hist.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Rolling sur tout l'historique
    d = add_rolling_mean(d, 5)
    d = add_rolling_mean(d, 10)
    d = add_rolling_std(d, 10)

    home_cnt = d.groupby("HomeTeam").cumcount()
    away_cnt = d.groupby("AwayTeam").cumcount()
    d["has_min_history"] = ((home_cnt >= 10) & (away_cnt >= 10)).astype(int)

    date_dt = pd.to_datetime(date)

    row = {"Date": date_dt, "HomeTeam": home, "AwayTeam": away}
    for k, v in (odds or {}).items():
        row[k] = v
    match_df = pd.DataFrame([row])

    # dernière stats home
    sub_home = d[(d["HomeTeam"] == home) & (d["Date"] < date_dt)].tail(1)
    sub_away = d[(d["AwayTeam"] == away) & (d["Date"] < date_dt)].tail(1)

    needed = [
        "home_gf_last5","home_ga_last5","away_gf_last5","away_ga_last5",
        "home_gf_last10","home_ga_last10","away_gf_last10","away_ga_last10",
        "home_gf_std10","home_ga_std10","away_gf_std10","away_ga_std10",
        "has_min_history",
        # derived
        "attack_diff5","defense_diff5","tempo5","attack_diff10","defense_diff10","tempo10"
    ]
    for c in needed:
        match_df[c] = np.nan

    if len(sub_home) == 1:
        for c in ["home_gf_last5","home_ga_last5","home_gf_last10","home_ga_last10","home_gf_std10","home_ga_std10","has_min_history"]:
            if c in sub_home.columns:
                match_df.loc[0, c] = sub_home.iloc[0][c]

    if len(sub_away) == 1:
        for c in ["away_gf_last5","away_ga_last5","away_gf_last10","away_ga_last10","away_gf_std10","away_ga_std10","has_min_history"]:
            if c in sub_away.columns:
                match_df.loc[0, c] = sub_away.iloc[0][c]

    # derived (recalcul sur la ligne)
    match_df["attack_diff5"]  = match_df["home_gf_last5"]  - match_df["away_ga_last5"]
    match_df["defense_diff5"] = match_df["away_gf_last5"]  - match_df["home_ga_last5"]
    match_df["tempo5"]        = (match_df["home_gf_last5"] + match_df["away_gf_last5"] + match_df["home_ga_last5"] + match_df["away_ga_last5"]) / 2.0

    match_df["attack_diff10"]  = match_df["home_gf_last10"] - match_df["away_ga_last10"]
    match_df["defense_diff10"] = match_df["away_gf_last10"] - match_df["home_ga_last10"]
    match_df["tempo10"]        = (match_df["home_gf_last10"] + match_df["away_gf_last10"] + match_df["home_ga_last10"] + match_df["away_ga_last10"]) / 2.0

    if pd.isna(match_df.loc[0, "has_min_history"]):
        match_df.loc[0, "has_min_history"] = 0

    return match_df


def predict_goal_with_proba(
    match_df,
    lambda_home_model,
    lambda_away_model,
    btts_ml,                 # optionnel (non utilisé si btts_cal est un modèle calibré predict_proba)
    btts_cal,                # modèle calibré (predict_proba) OU None
    config: dict,
    explainer=None,
    use_llm: bool = False,
    llm_client=None
):
    """
    Retourne un JSON minimal par marché:
      {"low_confidence": bool, "pred": int, "proba": float}
    Tout en conservant un payload enrichi (_debug + team_profile) pour l'explication LLM / rule-based.

    Pré-requis attendus:
      - config["lambda_features"] : list[str]
      - config["btts_features"]   : list[str]
      - config["o25_features"]    : list[str] si O25 hybride activé
      - config.get("_o25_cal_model") : modèle O25 calibré (predict_proba) ou None
      - fonctions utilitaires: _clip01, _p_over, _p_btts, _decision, _get_filter
      - explainers: rule_based_explainer, llm_explainer, enforce_risk_flags, build_explanation_prompt (via llm_explainer)
    """

    if not isinstance(match_df, pd.DataFrame):
        raise TypeError("match_df doit être un DataFrame (1 ligne).")
    if len(match_df) != 1:
        # on supporte 1 ligne pour éviter ambiguïtés
        raise ValueError("match_df doit contenir exactement 1 ligne (un match).")

    # ---- history flag
    hist_ok = int(match_df.get("has_min_history", pd.Series([0])).iloc[0]) == 1

    # ---- lambdas (à partir des features lambda)
    if "lambda_features" not in config:
        raise KeyError("config doit contenir 'lambda_features'.")
    X = match_df[config["lambda_features"]]

    lam_h = float(np.clip(lambda_home_model.predict(X)[0], 0.05, 4.5))
    lam_a = float(np.clip(lambda_away_model.predict(X)[0], 0.05, 4.5))
    lam_t = lam_h + lam_a

    # ---- Poisson probs
    p_o15 = _clip01(_p_over(lam_t, 1.5))
    p_o25_pois = _clip01(_p_over(lam_t, 2.5))
    p_o35 = _clip01(_p_over(lam_t, 3.5))
    p_btts_pois = _clip01(_p_btts(lam_h, lam_a))

    # ---- O15 / O35 (Poisson-only + filtre commun)
    flt_common = config.get(
        "default_filter",
        {"gray_low": 0.47, "gray_high": 0.55, "disagree_thr": 0.18, "require_history": True}
    )

    o15_pred, o15_low = _decision(p_o15, hist_ok, flt_common, disagree=None)
    o35_pred, o35_low = _decision(p_o35, hist_ok, flt_common, disagree=None)

    # ------------------------------------------------------------------
    # tmp enrichi (features attendues par hybrides)
    # ------------------------------------------------------------------
    tmp = match_df.copy()
    tmp["lambda_home"] = lam_h
    tmp["lambda_away"] = lam_a
    tmp["lambda_total"] = lam_t
    tmp["p_o25_pois"] = p_o25_pois
    tmp["p_btts_pois"] = p_btts_pois

    # ------------------------------------------------------------------
    # O25 Hybride (Poisson + ML calibré) si _o25_cal_model présent
    # ------------------------------------------------------------------
    o25_cal_model = config.get("_o25_cal_model", None)
    flt_o25 = _get_filter(config, match_df, market="O25")

    if o25_cal_model is None:
        o25_pred, o25_low = _decision(p_o25_pois, hist_ok, flt_o25, disagree=None)
        o25_pack = {"pred": o25_pred, "proba": p_o25_pois, "low_confidence": o25_low}
        o25_debug = {"proba_poisson": p_o25_pois}
    else:
        if "o25_features" not in config:
            raise KeyError("O25 hybride: config doit contenir 'o25_features'.")
        feats = config["o25_features"]

        p_o25_cal = _clip01(float(o25_cal_model.predict_proba(tmp[feats])[:, 1][0]))
        w25 = float(config.get("o25_blend_w", 1.0))
        p_o25_final = _clip01(w25 * p_o25_cal + (1 - w25) * p_o25_pois)
        dis25 = float(abs(p_o25_cal - p_o25_pois))

        o25_pred, o25_low = _decision(p_o25_final, hist_ok, flt_o25, disagree=dis25)

        o25_pack = {"pred": o25_pred, "proba": p_o25_final, "low_confidence": o25_low}
        o25_debug = {
            "proba_poisson": p_o25_pois,
            "proba_ml_cal": p_o25_cal,
            "blend_w": w25,
            "disagree": dis25,
        }

    # ------------------------------------------------------------------
    # BTTS Hybride ACTIVÉ (Poisson + modèle calibré btts_cal)
    #   - si btts_cal est None => Poisson-only + filtre
    #   - sinon => blend (w) + disagree (cal vs poisson) + filtre
    # NB: btts_ml est laissé en paramètre pour compatibilité, mais ici
    #     on utilise btts_cal comme modèle calibré avec predict_proba.
    # ------------------------------------------------------------------
    flt_btts = _get_filter(config, match_df, market="BTTS")

    if btts_cal is None:
        btts_pred, btts_low = _decision(p_btts_pois, hist_ok, flt_btts, disagree=None)
        btts_pack = {"pred": btts_pred, "proba": p_btts_pois, "low_confidence": btts_low}
        btts_debug = {"proba_poisson": p_btts_pois}
    else:
        if "btts_features" not in config:
            raise KeyError("BTTS hybride: config doit contenir 'btts_features'.")
        feats = config["btts_features"]

        # modèle calibré (predict_proba)
        p_btts_cal = _clip01(float(btts_cal.predict_proba(tmp[feats])[:, 1][0]))
        wb = float(config.get("btts_blend_w", 1.0))
        p_btts_final = _clip01(wb * p_btts_cal + (1 - wb) * p_btts_pois)
        disb = float(abs(p_btts_cal - p_btts_pois))

        btts_pred, btts_low = _decision(p_btts_final, hist_ok, flt_btts, disagree=disb)
        btts_pack = {"pred": btts_pred, "proba": p_btts_final, "low_confidence": btts_low}
        btts_debug = {
            "proba_poisson": p_btts_pois,
            "proba_ml_cal": p_btts_cal,
            "blend_w": wb,
            "disagree": disb,
        }

    # ------------------------------------------------------------------
    # Profil attaque/défense (OPTIONNEL) si rolling disponibles
    #   - on n'invente rien: seulement si colonnes existent
    # ------------------------------------------------------------------
    team_profile = {
        "lambda_home": lam_h,
        "lambda_away": lam_a,
        "lambda_total": lam_t,
    }
    for c in ["home_gf_last5", "home_ga_last5", "away_gf_last5", "away_ga_last5"]:
        if c in match_df.columns:
            try:
                team_profile[c] = float(match_df[c].iloc[0])
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Sortie MINIMALE (format API)
    # ------------------------------------------------------------------
    res = {
        "lambda_home": lam_h,
        "lambda_away": lam_a,
        "lambda_total": lam_t,

        "Over15": {"low_confidence": bool(o15_low), "pred": int(o15_pred), "proba": float(p_o15)},
        "Over25": {"low_confidence": bool(o25_pack["low_confidence"]), "pred": int(o25_pack["pred"]), "proba": float(o25_pack["proba"])},
        "Over35": {"low_confidence": bool(o35_low), "pred": int(o35_pred), "proba": float(p_o35)},
        "BTTS": {"low_confidence": bool(btts_pack["low_confidence"]), "pred": int(btts_pack["pred"]), "proba": float(btts_pack["proba"])},
    }

    # ------------------------------------------------------------------
    # Payload enrichi pour explication (LLM / rules)
    # ------------------------------------------------------------------
    
    
    # ------------------------------------------------------------------
    # Payload enrichi pour explication (LLM / rules)
    # ------------------------------------------------------------------
    payload = dict(res)

    # ✅ Ajout des équipes (indispensable pour que le LLM parle "noms d'équipes")
    home_name = match_df["HomeTeam"].iloc[0] if "HomeTeam" in match_df.columns else "Home"
    away_name = match_df["AwayTeam"].iloc[0] if "AwayTeam" in match_df.columns else "Away"
    payload["teams"] = {"home": str(home_name), "away": str(away_name)}

    payload["_debug"] = {
        "Over25": o25_debug,
        "BTTS": btts_debug,
        "hist_ok": bool(hist_ok),
    }
    payload["team_profile"] = team_profile

    # ---- explainer fallback
    if explainer is None:
        explainer = rule_based_explainer

    # ---- explanation
    if use_llm and (llm_client is not None):
        exp = llm_explainer(payload, llm_client)
        exp = enforce_llm_output(exp, payload, fallback_fn=rule_based_explainer)
        res["explanation"] = enforce_risk_flags(exp, payload)
        #res["explanation"] = enforce_risk_flags(exp, payload)
    else:
        res["explanation"] = explainer(payload)

    return res



# ======================================================================================
# BETSMART GOAL INTELLIGENCE V1.0.8.2
# Multi-source Goal Engine:
# ML lambdas -> Poisson/calibrated markets -> history/form/H2H -> realtime/web -> AI
# -> Stability Layer -> Guaranteed Final State.
# ======================================================================================

GOAL_INTELLIGENCE_VERSION = "GOAL-1.0.8.8.14-EXPLANATION-SOURCE-STABILITY"
GOAL_AI_MODEL = os.getenv("BETSMART_GOAL_AI_MODEL", "gpt-5.4-mini")
GOAL_WEB_MODEL = os.getenv("BETSMART_GOAL_WEB_MODEL", GOAL_AI_MODEL)
GOAL_AI_ENABLED = os.getenv("BETSMART_GOAL_AI_ENABLED", "1").strip().lower() not in {"0","false","off","no"}
GOAL_WEB_ENABLED = os.getenv("BETSMART_GOAL_WEB_ENABLED", "1").strip().lower() not in {"0","false","off","no"}
GOAL_REALTIME_ENABLED = os.getenv("BETSMART_GOAL_REALTIME_ENABLED", "1").strip().lower() not in {"0","false","off","no"}

GOAL_CONTEXT_WORKERS = int(os.getenv("BETSMART_GOAL_CONTEXT_WORKERS", "2"))
GOAL_WEB_TIMEOUT_SECONDS = int(os.getenv("BETSMART_GOAL_WEB_TIMEOUT_SECONDS", "15"))
GOAL_WEB_MAX_OUTPUT_TOKENS = int(os.getenv("BETSMART_GOAL_WEB_MAX_OUTPUT_TOKENS", "2600"))
GOAL_WEB_MAX_TOOL_CALLS = int(os.getenv("BETSMART_GOAL_WEB_MAX_TOOL_CALLS", "2"))
GOAL_WEB_MAX_SOURCES = int(os.getenv("BETSMART_GOAL_WEB_MAX_SOURCES", "6"))
GOAL_OPENAI_REASONING_EFFORT = os.getenv("BETSMART_GOAL_OPENAI_REASONING_EFFORT", "none").strip() or "none"
GOAL_OPENAI_VERBOSITY = os.getenv("BETSMART_GOAL_OPENAI_VERBOSITY", "low").strip() or "low"
GOAL_OPENAI_WEB_ANALYSIS_MAX_OUTPUT_TOKENS = int(os.getenv("BETSMART_GOAL_OPENAI_WEB_ANALYSIS_MAX_OUTPUT_TOKENS", "500"))
GOAL_OPENAI_WEB_ANALYSIS_TIMEOUT_SECONDS = float(os.getenv("BETSMART_GOAL_OPENAI_WEB_ANALYSIS_TIMEOUT_SECONDS", "12.0"))
GOAL_OPENAI_WEB_MAX_RETRIES = int(os.getenv("BETSMART_GOAL_OPENAI_WEB_MAX_RETRIES", "0"))
GOAL_OPENAI_WEB_CHAT_FALLBACK_ENABLED = os.getenv("BETSMART_GOAL_OPENAI_WEB_CHAT_FALLBACK_ENABLED", "0").strip().lower() in {"1","true","on","yes"}
GOAL_OPENAI_ARBITRATION_MAX_OUTPUT_TOKENS = int(os.getenv("BETSMART_GOAL_OPENAI_ARBITRATION_MAX_OUTPUT_TOKENS", "550"))
GOAL_OPENAI_ARBITRATION_TIMEOUT_SECONDS = float(os.getenv("BETSMART_GOAL_OPENAI_ARBITRATION_TIMEOUT_SECONDS", "10.0"))
GOAL_OPENAI_ARBITRATION_MAX_RETRIES = int(os.getenv("BETSMART_GOAL_OPENAI_ARBITRATION_MAX_RETRIES", "0"))
GOAL_OPENAI_ARBITRATION_CHAT_FALLBACK_ENABLED = os.getenv("BETSMART_GOAL_OPENAI_ARBITRATION_CHAT_FALLBACK_ENABLED", "0").strip().lower() in {"1","true","on","yes"}
GOAL_FORCE_AI_ARBITRATION = os.getenv("BETSMART_GOAL_FORCE_AI_ARBITRATION", "0").strip().lower() in {"1","true","on","yes"}
GOAL_AI_MARKET_ADJUST_CAP = float(os.getenv("BETSMART_GOAL_AI_MARKET_ADJUST_CAP", "0.06"))
GOAL_AI_MARKET_MIN_EVIDENCE = float(os.getenv("BETSMART_GOAL_AI_MARKET_MIN_EVIDENCE", "0.45"))
GOAL_REALTIME_TIMEOUT_SECONDS = int(os.getenv("BETSMART_GOAL_REALTIME_TIMEOUT_SECONDS", "10"))

_GOAL_WEB_CACHE = {}
_GOAL_WEB_CACHE_LOCK = threading.Lock()


def _goal_jsonable(x):
    if isinstance(x, dict):
        return {str(k): _goal_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_goal_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (pd.Timestamp, datetime.datetime, datetime.date)):
        return x.isoformat()
    return x


def _goal_safe_float(v, default=0.0):
    try:
        f = float(v)
        return f if np.isfinite(f) else float(default)
    except Exception:
        return float(default)


def _goal_normalize_prob(p):
    return float(np.clip(_goal_safe_float(p), 1e-6, 1 - 1e-6))


def _goal_market_pack(prob, confidence, history_ok=True, gray_low=0.47, gray_high=0.55):
    prob = _goal_normalize_prob(prob)
    pred = int(prob >= 0.50)
    low = bool(
        (gray_low <= prob <= gray_high)
        or confidence < 0.52
        or (not history_ok)
    )
    return {"pred": pred, "proba": prob, "low_confidence": low}


def _goal_team_matches(df, team, before_date):
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d[d["Date"] < pd.to_datetime(before_date)]
    return d[(d["HomeTeam"] == team) | (d["AwayTeam"] == team)].sort_values("Date")


def _goal_last_form(df, team, before_date, n=5):
    rows = _goal_team_matches(df, team, before_date).tail(n)
    gf, ga, totals, btts, over25 = [], [], [], [], []
    for _, r in rows.iterrows():
        if r.get("HomeTeam") == team:
            g_for, g_against = _goal_safe_float(r.get("FTHG")), _goal_safe_float(r.get("FTAG"))
        else:
            g_for, g_against = _goal_safe_float(r.get("FTAG")), _goal_safe_float(r.get("FTHG"))
        total = g_for + g_against
        gf.append(g_for); ga.append(g_against); totals.append(total)
        btts.append(int(g_for > 0 and g_against > 0))
        over25.append(int(total >= 3))
    return {
        "matches": int(len(rows)),
        "gf_avg": round(float(np.mean(gf)), 3) if gf else None,
        "ga_avg": round(float(np.mean(ga)), 3) if ga else None,
        "total_goals_avg": round(float(np.mean(totals)), 3) if totals else None,
        "btts_rate": round(float(np.mean(btts)), 3) if btts else None,
        "over25_rate": round(float(np.mean(over25)), 3) if over25 else None,
        "reliability": round(min(1.0, len(rows) / 5.0), 3),
    }


def _goal_venue_history(df, home, away, before_date, max_matches=60):
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d[d["Date"] < pd.to_datetime(before_date)].sort_values("Date")

    h = d[d["HomeTeam"] == home].tail(max_matches)
    a = d[d["AwayTeam"] == away].tail(max_matches)

    def agg(rows, side):
        gf, ga, totals, btts, o15, o25, o35 = [], [], [], [], [], [], []
        for _, r in rows.iterrows():
            if side == "home":
                f, g = _goal_safe_float(r.get("FTHG")), _goal_safe_float(r.get("FTAG"))
            else:
                f, g = _goal_safe_float(r.get("FTAG")), _goal_safe_float(r.get("FTHG"))
            t = f + g
            gf.append(f); ga.append(g); totals.append(t)
            btts.append(int(f > 0 and g > 0))
            o15.append(int(t >= 2)); o25.append(int(t >= 3)); o35.append(int(t >= 4))
        return {
            "matches": int(len(rows)),
            "gf_avg": round(float(np.mean(gf)), 3) if gf else None,
            "ga_avg": round(float(np.mean(ga)), 3) if ga else None,
            "total_goals_avg": round(float(np.mean(totals)), 3) if totals else None,
            "btts_rate": round(float(np.mean(btts)), 3) if btts else None,
            "over15_rate": round(float(np.mean(o15)), 3) if o15 else None,
            "over25_rate": round(float(np.mean(o25)), 3) if o25 else None,
            "over35_rate": round(float(np.mean(o35)), 3) if o35 else None,
        }

    return {"home_at_home": agg(h, "home"), "away_at_away": agg(a, "away")}


def _goal_h2h(df, home, away, before_date, n=8):
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d[d["Date"] < pd.to_datetime(before_date)]
    mask = (
        ((d["HomeTeam"] == home) & (d["AwayTeam"] == away))
        | ((d["HomeTeam"] == away) & (d["AwayTeam"] == home))
    )
    rows = d[mask].sort_values("Date").tail(n)
    totals, btts, scores = [], [], []
    for _, r in rows.iterrows():
        hg, ag = _goal_safe_float(r.get("FTHG")), _goal_safe_float(r.get("FTAG"))
        totals.append(hg + ag)
        btts.append(int(hg > 0 and ag > 0))
        scores.append({
            "date": _goal_jsonable(r.get("Date")),
            "home": str(r.get("HomeTeam")),
            "away": str(r.get("AwayTeam")),
            "score": f"{int(hg)}-{int(ag)}",
            "total_goals": hg + ag,
        })
    return {
        "matches": int(len(rows)),
        "total_goals_avg": round(float(np.mean(totals)), 3) if totals else None,
        "btts_rate": round(float(np.mean(btts)), 3) if btts else None,
        "over25_rate": round(float(np.mean([t >= 3 for t in totals])), 3) if totals else None,
        "confidence": (
            "VERY_HIGH" if len(rows) >= 7 else
            "HIGH" if len(rows) >= 5 else
            "MEDIUM" if len(rows) >= 3 else
            "LOW"
        ),
        "recent_matches": scores,
    }


def build_goal_historical_context(df_hist, home, away, date):
    home_form = _goal_last_form(df_hist, home, date, 5)
    away_form = _goal_last_form(df_hist, away, date, 5)
    venue = _goal_venue_history(df_hist, home, away, date)
    h2h = _goal_h2h(df_hist, home, away, date)

    current_rel = min(
        _goal_safe_float(home_form.get("reliability")),
        _goal_safe_float(away_form.get("reliability")),
    )
    return {
        "current_form": {"home": home_form, "away": away_form},
        "current_form_reliability": round(current_rel, 3),
        "venue_history": venue,
        "h2h": h2h,
        "phase": "MATURE_FORM" if current_rel >= 1.0 else "EARLY_OR_PARTIAL_FORM",
    }


def build_goal_market_context(odds):
    odds = odds or {}
    out = {}
    mapping = {
        "Over15": "OU_O15",
        "Over25": "OU_O25",
        "Over35": "OU_O35",
        "BTTS": "BTTS_Yes",
    }
    for market, key in mapping.items():
        odd = _goal_safe_float(odds.get(key), 0.0)
        out[market] = {
            "bookmaker_odds": odd if odd > 1.0 else None,
            # There is no opposite-side price in the current API payload;
            # therefore this is RAW implied probability, not de-margined.
            "market_probability_raw": round(1.0 / odd, 4) if odd > 1.0 else None,
            "demarged_available": False,
            "decision_impact": "INFORMATION_ONLY",
        }
    return out


def _goal_realtime_headers():
    """Use the canonical BetSmart realtime key only."""
    key = os.getenv("REALTIME_API_KEY")
    if not key:
        return None
    return {"x-apisports-key": key}


def research_goal_realtime_context(home, away, match_date):
    """
    Optional API-Football context.

    V1.0.8.8.4 optimization:
    - fixture resolution remains the prerequisite;
    - once fixture_id is known, injuries and lineups are fetched concurrently;
    - trained ML and prediction logic are untouched.
    """
    empty = {
        "available": False,
        "fixture_id": None,
        "injuries_home": [],
        "injuries_away": [],
        "lineups_available": False,
        "status": "UNAVAILABLE",
        "diagnostics": {
            "provider": "API_FOOTBALL",
            "fixture_request_seconds": None,
            "injuries_request_seconds": None,
            "lineups_request_seconds": None,
            "parallel_detail_seconds": None,
            "failure_stage": None,
            "error": None,
        },
    }
    if not GOAL_REALTIME_ENABLED:
        return {**empty, "status": "DISABLED"}

    headers = _goal_realtime_headers()
    if not headers:
        return {**empty, "status": "NO_API_KEY"}

    base_url = os.getenv("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io").rstrip("/")
    try:
        _t_fixture = time.perf_counter()
        r = requests.get(
            f"{base_url}/fixtures",
            headers=headers,
            params={"date": str(match_date)[:10]},
            timeout=8,
        )
        fixture_seconds = round(time.perf_counter() - _t_fixture, 3)
        data = r.json().get("response", []) if r.ok else []
        chosen = None
        hn, an = str(home).lower(), str(away).lower()
        for fx in data:
            th = str(((fx.get("teams") or {}).get("home") or {}).get("name", "")).lower()
            ta = str(((fx.get("teams") or {}).get("away") or {}).get("name", "")).lower()
            if (hn in th or th in hn) and (an in ta or ta in an):
                chosen = fx
                break
        if not chosen:
            out_nf = {**empty, "status": "FIXTURE_NOT_FOUND"}
            out_nf["diagnostics"] = {
                **empty["diagnostics"],
                "fixture_request_seconds": fixture_seconds,
                "failure_stage": "FIXTURE_MATCHING",
            }
            return out_nf

        fixture_id = int(((chosen.get("fixture") or {}).get("id")))
        out = {**empty, "available": True, "fixture_id": fixture_id, "status": "OK"}
        out["diagnostics"] = {**empty["diagnostics"], "fixture_request_seconds": fixture_seconds}

        def _fetch_injuries():
            t0 = time.perf_counter()
            resp = requests.get(
                f"{base_url}/injuries",
                headers=headers,
                params={"fixture": fixture_id},
                timeout=6,
            )
            rows = resp.json().get("response", []) if resp.ok else []
            return rows, round(time.perf_counter() - t0, 3)

        def _fetch_lineups():
            t0 = time.perf_counter()
            resp = requests.get(
                f"{base_url}/fixtures/lineups",
                headers=headers,
                params={"fixture": fixture_id},
                timeout=6,
            )
            rows = resp.json().get("response", []) if resp.ok else []
            return rows, round(time.perf_counter() - t0, 3)

        detail_t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as detail_pool:
            fut_inj = detail_pool.submit(_fetch_injuries)
            fut_lin = detail_pool.submit(_fetch_lineups)

            try:
                injuries, sec = fut_inj.result()
                out["diagnostics"]["injuries_request_seconds"] = sec
                for item in injuries:
                    team = str(((item.get("team") or {}).get("name", ""))).lower()
                    player = item.get("player") or {}
                    row = {
                        "name": player.get("name"),
                        "reason": player.get("reason"),
                        "type": player.get("type"),
                    }
                    if hn in team or team in hn:
                        out["injuries_home"].append(row)
                    elif an in team or team in an:
                        out["injuries_away"].append(row)
            except Exception as exc:
                out["diagnostics"]["injuries_error"] = f"{type(exc).__name__}:{str(exc)[:300]}"

            try:
                lineups, sec = fut_lin.result()
                out["diagnostics"]["lineups_request_seconds"] = sec
                out["lineups_available"] = bool(lineups)
            except Exception as exc:
                out["diagnostics"]["lineups_error"] = f"{type(exc).__name__}:{str(exc)[:300]}"

        out["diagnostics"]["parallel_detail_seconds"] = round(time.perf_counter() - detail_t0, 3)
        return out
    except Exception as e:
        out_err = {**empty, "status": f"ERROR:{type(e).__name__}"}
        out_err["diagnostics"] = {
            **empty["diagnostics"],
            "failure_stage": "FIXTURE_REQUEST",
            "error": f"{type(e).__name__}:{str(e)[:400]}",
        }
        return out_err
def _goal_extract_json(text):
    if text is None:
        return None
    if isinstance(text, dict):
        return text
    s = str(text).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i+1])
                except Exception:
                    return None
    return None



def _goal_response_text(resp):
    """
    Extract assistant text across Responses API and Chat Completions SDK shapes.
    """
    if resp is None:
        return ""

    try:
        txt = getattr(resp, "output_text", None)
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
    except Exception:
        pass

    chunks = []

    # Native Responses API object traversal
    try:
        output = getattr(resp, "output", None)
        if output:
            for item in output:
                content = getattr(item, "content", None)
                if not content:
                    continue
                for part in content:
                    txt = getattr(part, "text", None)
                    if isinstance(txt, str) and txt.strip():
                        chunks.append(txt.strip())
                    elif txt is not None:
                        val = getattr(txt, "value", None)
                        if isinstance(val, str) and val.strip():
                            chunks.append(val.strip())
    except Exception:
        pass

    # Generic dump traversal
    try:
        raw = resp.model_dump()
    except Exception:
        raw = None

    def walk(obj):
        if isinstance(obj, dict):
            for key in ("text", "output_text", "value"):
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    chunks.append(val.strip())
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    if raw is not None:
        walk(raw)

    if chunks:
        return max(chunks, key=len)

    try:
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


def _goal_response_sources(resp):
    """
    Extract web URLs/titles from web_search results and citations.
    """
    sources = []
    seen = set()

    def add(url=None, title=None):
        url = str(url or "").strip()
        if not url or not url.startswith("http"):
            return
        key = url.lower()
        if key in seen:
            return
        seen.add(key)
        sources.append({
            "url": url,
            "title": (str(title).strip()[:300] if title else None),
        })

    # Native Responses API traversal
    try:
        output = getattr(resp, "output", None)
        if output:
            for item in output:
                action = getattr(item, "action", None)
                if action:
                    srcs = getattr(action, "sources", None)
                    if srcs:
                        for src in srcs:
                            add(
                                getattr(src, "url", None),
                                getattr(src, "title", None),
                            )

                content = getattr(item, "content", None)
                if content:
                    for part in content:
                        anns = getattr(part, "annotations", None)
                        if anns:
                            for ann in anns:
                                add(
                                    getattr(ann, "url", None),
                                    getattr(ann, "title", None),
                                )
    except Exception:
        pass

    # Generic dict traversal
    try:
        raw = resp.model_dump()
    except Exception:
        raw = None

    def walk(obj):
        if isinstance(obj, dict):
            url = obj.get("url")
            if isinstance(url, str):
                add(url, obj.get("title"))
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    if raw is not None:
        walk(raw)

    return sources



def _goal_structured_ai_call(client, model, prompt, schema_name="betsmart_goal_decision"):
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {"type": "string"},
            "lambda_home_delta": {"type": "number"},
            "lambda_away_delta": {"type": "number"},
            "prediction_confidence": {"type": "number"},
            "source_agreement": {"type": "string", "enum": ["LOW","MEDIUM","HIGH"]},
            "risk_level": {"type": "string", "enum": ["LOW","MEDIUM","HIGH"]},
            "reason_codes": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
            "rationale_short": {"type": "string"},
            "explanation": {"type": "string"},
        },
        "required": [
            "status","lambda_home_delta","lambda_away_delta",
            "prediction_confidence","source_agreement","risk_level",
            "reason_codes","rationale_short","explanation",
        ],
    }
    errors=[]
    c=client.with_options(max_retries=GOAL_OPENAI_ARBITRATION_MAX_RETRIES)
    try:
        resp=c.responses.create(
            model=model,input=prompt,
            reasoning={"effort": GOAL_OPENAI_REASONING_EFFORT},
            text={"verbosity": GOAL_OPENAI_VERBOSITY,
                  "format":{"type":"json_schema","name":schema_name,"schema":schema,"strict":True}},
            max_output_tokens=GOAL_OPENAI_ARBITRATION_MAX_OUTPUT_TOKENS,
            timeout=GOAL_OPENAI_ARBITRATION_TIMEOUT_SECONDS,
        )
        obj=_goal_extract_json(_goal_response_text(resp))
        if isinstance(obj,dict): return obj
        errors.append("RESPONSES_NON_JSON")
    except Exception as exc:
        errors.append(f"RESPONSES:{type(exc).__name__}:{str(exc)[:500]}")

    if GOAL_OPENAI_ARBITRATION_CHAT_FALLBACK_ENABLED:
        try:
            resp=c.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":"Tu réponds STRICTEMENT en JSON valide."},{"role":"user","content":prompt}],
                response_format={"type":"json_object"},
                max_tokens=GOAL_OPENAI_ARBITRATION_MAX_OUTPUT_TOKENS,
                timeout=GOAL_OPENAI_ARBITRATION_TIMEOUT_SECONDS,
            )
            obj=_goal_extract_json(_goal_response_text(resp))
            if isinstance(obj,dict): return obj
            errors.append("CHAT_NON_JSON")
        except Exception as exc:
            errors.append(f"CHAT:{type(exc).__name__}:{str(exc)[:500]}")
    raise ValueError(" | ".join(errors)[:1200])

def _goal_web_sources_count(resp) -> int:
    return len(_goal_response_sources(resp))



def _goal_web_reformat_to_json(client, model: str, raw_text: str):
    """
    Second-pass formatter: turns a useful web answer into strict JSON.
    No new facts may be invented.
    """
    if not raw_text or not raw_text.strip():
        return None

    prompt = f"""
Transforme STRICTEMENT le texte suivant en JSON, sans ajouter aucun fait.

FORMAT:
{{
  "summary":"...",
  "data_confidence":0.0,
  "signals":[
    {{
      "side":"HOME|AWAY|BOTH",
      "category":"ATTACK|DEFENSE|GOALKEEPER|INJURY|LINEUP|TACTICS|FATIGUE|WEATHER|FORM",
      "direction":"MORE_GOALS|FEWER_GOALS|NEUTRAL",
      "impact":0.0,
      "confidence":0.0,
      "summary":"..."
    }}
  ]
}}

RÈGLES:
- n'invente rien;
- si un élément n'est pas certain, confidence faible;
- si le texte ne contient rien d'actionnable, signals=[] et data_confidence faible.

TEXTE:
{raw_text[:12000]}
""".strip()

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Réponds STRICTEMENT en JSON valide."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return _goal_extract_json(_goal_response_text(resp))
    except Exception:
        return None





def _goal_brave_domain_tier(url):
    """Simple, auditable source hierarchy for football evidence."""
    try:
        from urllib.parse import urlparse
        domain = (urlparse(str(url)).netloc or "").lower()
        if domain.startswith("www."):
            domain = domain[4:]
    except Exception:
        domain = ""

    tier_a = (
        "premierleague.com", "uefa.com", "fifa.com",
        "bbc.co.uk", "bbc.com", "espn.com",
        "fbref.com", "statmuse.com",
    )
    tier_b = (
        "skysports.com", "theguardian.com", "reuters.com",
        "sportsmole.co.uk", "rotowire.com", "fourfourtwo.com",
        "yahoo.com", "soccerway.com",
    )
    tier_d = (
        "reddit.com", "wikipedia.org", "x.com", "twitter.com",
        "facebook.com", "instagram.com",
    )

    if any(domain == d or domain.endswith("." + d) for d in tier_a):
        return "A"
    if any(domain == d or domain.endswith("." + d) for d in tier_b):
        return "B"
    if any(domain == d or domain.endswith("." + d) for d in tier_d):
        return "D"
    return "C"


def _goal_brave_search(query, count=3):
    """
    Deterministic external retrieval.
    No OpenAI web_search tool is used here.
    """
    api_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
    if not api_key:
        raise ValueError("BRAVE_SEARCH_API_KEY_MISSING")

    timeout_s = float(os.getenv("BETSMART_GOAL_RETRIEVAL_TIMEOUT_SECONDS", "6.0"))
    max_retries = int(os.getenv("BETSMART_GOAL_RETRIEVAL_MAX_RETRIES", "1"))
    max_retries = max(0, min(max_retries, 1))
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    params = {
        "q": str(query)[:390],
        "count": max(1, min(int(count), 6)),
        "search_lang": "en",
        "safesearch": "moderate",
        "text_decorations": "false",
        "result_filter": "web",
    }
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }

    started = time.perf_counter()
    attempts = []
    resp = None
    last_exc = None

    for attempt in range(1, max_retries + 2):
        attempt_started = time.perf_counter()
        try:
            resp = requests.get(endpoint, params=params, headers=headers, timeout=timeout_s)
            attempt_elapsed = time.perf_counter() - attempt_started
            attempts.append({
                "attempt": attempt,
                "status": "HTTP",
                "http_status": int(resp.status_code),
                "elapsed_seconds": round(attempt_elapsed, 3),
            })
            break
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout,
                requests.exceptions.ConnectionError) as exc:
            attempt_elapsed = time.perf_counter() - attempt_started
            last_exc = exc
            attempts.append({
                "attempt": attempt,
                "status": "NETWORK_ERROR",
                "error_type": type(exc).__name__,
                "elapsed_seconds": round(attempt_elapsed, 3),
            })
            if attempt > max_retries:
                raise

    elapsed = time.perf_counter() - started
    if resp is None:
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("BRAVE_NO_RESPONSE")

    if resp.status_code != 200:
        raise ValueError(
            f"BRAVE_HTTP_{resp.status_code}:{resp.text[:300]}"
        )

    payload = resp.json()
    web = payload.get("web") or {}
    raw_results = web.get("results") or []

    cleaned = []
    seen = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)

        cleaned.append({
            "title": str(item.get("title") or "")[:220],
            "url": url,
            "description": str(item.get("description") or "")[:900],
            "age": item.get("age"),
            "language": item.get("language"),
            "source_tier": _goal_brave_domain_tier(url),
        })
        if len(cleaned) >= count:
            break

    return {
        "query": query,
        "elapsed_seconds": round(elapsed, 3),
        "attempt_count": len(attempts),
        "attempts": attempts,
        "retried": len(attempts) > 1,
        "results": cleaned,
        "result_count": len(cleaned),
    }


def _goal_retrieve_web_evidence(home, away, match_date):
    """
    Exactly two targeted retrieval queries per match, run concurrently.
    Maximum 3 results per query by default, 6 total.
    """
    per_query = int(os.getenv("BETSMART_GOAL_RETRIEVAL_RESULTS_PER_QUERY", "3"))
    per_query = max(1, min(per_query, 3))

    q_team = (
        f'"{home}" "{away}" {str(match_date)[:10]} '
        f'injuries suspensions team news probable lineup'
    )
    q_form = (
        f'"{home}" "{away}" recent form last 5 matches '
        f'goals scored conceded'
    )

    started = time.perf_counter()
    outputs = []
    errors = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            ("TEAM_NEWS", pool.submit(_goal_brave_search, q_team, per_query)),
            ("RECENT_FORM", pool.submit(_goal_brave_search, q_form, per_query)),
        ]
        for label, fut in futures:
            try:
                item = fut.result()
                item["category"] = label
                outputs.append(item)
            except Exception as exc:
                errors.append({
                    "category": label,
                    "error": f"{type(exc).__name__}:{str(exc)[:400]}",
                })

    merged = []
    seen = set()
    for group in outputs:
        for item in group.get("results") or []:
            url = item.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            x = dict(item)
            x["retrieval_category"] = group.get("category")
            merged.append(x)

    # Prefer stronger sources without hiding the retrieval origin.
    tier_rank = {"A": 0, "B": 1, "C": 2, "D": 3}
    merged.sort(key=lambda x: tier_rank.get(x.get("source_tier"), 9))
    merged = merged[:6]

    return {
        "provider": "BRAVE_SEARCH_API",
        "query_count": 2,
        "queries": outputs,
        "errors": errors,
        "results": merged,
        "result_count": len(merged),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _goal_analyze_retrieved_evidence(client, model, home, away, match_date, retrieval):
    """
    OpenAI is analyst only. No web tools are provided.

    GOAL-1.0.8.4 compact structured-output contract:
    - same compact schema that succeeded in the standalone benchmark;
    - max 3 factual statements and max 2 market signals;
    - no duplicated source metadata in the LLM response;
    - Brave source metadata is retained and enriched deterministically by Python;
    - hard request timeout and zero automatic retries remain active.
    """
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "facts": {
                "type": "array",
                "maxItems": 3,
                "items": {"type": "string"},
            },
            "signals": {
                "type": "array",
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "market": {
                            "type": "string",
                            "enum": ["BTTS", "Over15", "Over25", "Over35"],
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["UP", "DOWN", "NEUTRAL"],
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "impact": {
                            "type": "number",
                            "minimum": -0.20,
                            "maximum": 0.20,
                        },
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "market", "direction", "confidence", "impact", "reason"
                    ],
                },
            },
            "summary": {"type": "string"},
        },
        "required": ["facts", "signals", "summary"],
    }

    compact_evidence = []
    for item in (retrieval.get("results") or [])[:6]:
        if not isinstance(item, dict):
            continue
        compact_evidence.append({
            "category": item.get("retrieval_category"),
            "tier": item.get("source_tier"),
            "title": item.get("title"),
            "snippet": item.get("description") or item.get("snippet"),
        })

    evidence_json = json.dumps(
        compact_evidence,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    prompt = f"""
BETSMART GOAL INTELLIGENCE — COMPACT EVIDENCE ANALYSIS.

Match: {home} vs {away}
Date: {match_date}

Analyse UNIQUEMENT les preuves ci-dessous.
N'invente rien et n'utilise aucune connaissance externe.
Retourne au maximum 3 faits vérifiables et 2 signaux exploitables.
Un signal doit rester NEUTRAL si la preuve est faible ou ambiguë.
Si un snippet contient explicitement des chiffres (résultats des 5 derniers matchs, buts marqués/encaissés, taux ou scores), extrais-les fidèlement et utilise-les dans les faits. Ne conclus jamais qu'il n'y a aucun chiffre si des valeurs numériques sont visibles dans les preuves fournies.
RÈGLES DE VALIDATION DES PREUVES:
1. VALIDITÉ TEMPORELLE: compare toute date explicitement visible dans un titre ou snippet à la date du match demandé. Si une preuve décrit explicitement le même match à une autre date, considère-la comme temporellement incohérente: elle peut être mentionnée comme incohérence mais ne doit produire aucun signal directionnel.
2. ATTRIBUTION SÉMANTIQUE: ne déduis jamais la nature d'une statistique à partir du titre de la page. Une page intitulée "Head to Head" peut contenir la forme récente propre à chaque équipe. N'appelle une statistique H2H que si le snippet dit explicitement qu'elle concerne les confrontations entre les deux équipes.
3. COMPLÉTUDE MÉTRIQUE: un chiffre sans période, échantillon ou contexte suffisamment clair peut être cité comme information brute, mais ne doit pas justifier à lui seul un signal UP ou DOWN.
4. FORME RÉCENTE: "Last 5, [équipe] won/draw/lose... goals per match" doit être attribué à la forme récente de cette équipe, sauf indication explicite contraire.
5. BLESSURES/COMPOSITIONS: une absence ou composition prédite ne doit influencer un marché de buts que si le snippet permet d'établir un lien footballistique explicite et suffisamment contextualisé; n'invente jamais le poste, le rôle ou l'importance d'un joueur.
6. En cas d'ambiguïté d'attribution, de date ou de métrique, conserve le fait avec prudence mais force le signal correspondant à NEUTRAL avec impact 0.

Marchés autorisés: BTTS, Over15, Over25, Over35.
impact doit rester entre -0.20 et 0.20.
confidence doit rester entre 0 et 1.
Ne donne aucun conseil de pari.
Retourne uniquement le JSON conforme au schéma demandé.

PREUVES:
{evidence_json}
""".strip()

    errors = []
    analysis_client = client.with_options(max_retries=GOAL_OPENAI_WEB_MAX_RETRIES)

    try:
        resp = analysis_client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": GOAL_OPENAI_REASONING_EFFORT},
            text={
                "verbosity": GOAL_OPENAI_VERBOSITY,
                "format": {
                    "type": "json_schema",
                    "name": "betsmart_goal_web_evidence_v10883_validated",
                    "schema": schema,
                    "strict": True,
                },
            },
            max_output_tokens=GOAL_OPENAI_WEB_ANALYSIS_MAX_OUTPUT_TOKENS,
            timeout=GOAL_OPENAI_WEB_ANALYSIS_TIMEOUT_SECONDS,
        )
        raw_text = _goal_response_text(resp)
        obj = _goal_extract_json(raw_text)
        if isinstance(obj, dict):
            return obj, "RESPONSES_JSON_SCHEMA_COMPACT"
        errors.append(f"RESPONSES_NON_JSON:chars={len(raw_text or '')}")
    except Exception as exc:
        err = f"RESPONSES:{type(exc).__name__}:{str(exc)[:400]}"
        errors.append(err)
        if "timeout" in type(exc).__name__.lower() or "timeout" in str(exc).lower():
            raise TimeoutError(err) from exc

    if GOAL_OPENAI_WEB_CHAT_FALLBACK_ENABLED:
        try:
            resp = analysis_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Analyse uniquement les preuves football fournies. "
                            "Aucun accès web. Retourne uniquement un JSON valide."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=GOAL_OPENAI_WEB_ANALYSIS_MAX_OUTPUT_TOKENS,
                timeout=GOAL_OPENAI_WEB_ANALYSIS_TIMEOUT_SECONDS,
            )
            text = (resp.choices[0].message.content or "").strip()
            obj = _goal_extract_json(text)
            if isinstance(obj, dict):
                return obj, "CHAT_JSON_FALLBACK"
            errors.append(f"CHAT_NON_JSON:chars={len(text)}")
        except Exception as exc:
            errors.append(f"CHAT:{type(exc).__name__}:{str(exc)[:400]}")

    raise ValueError("|".join(errors) if errors else "OPENAI_ANALYSIS_FAILED")

def _goal_retrieval_sources(retrieval, include_snippet=False):
    """Build safe public Brave source objects without inventing facts."""
    out = []
    for item in (retrieval or {}).get("results") or []:
        row = {
            "title": item.get("title"),
            "url": item.get("url"),
            "source_tier": item.get("source_tier"),
            "retrieval_category": item.get("retrieval_category"),
        }
        if include_snippet:
            row["snippet"] = str(item.get("description") or "")[:700]
            row["age"] = item.get("age")
        out.append(row)
    return out[:GOAL_WEB_MAX_SOURCES]


def _goal_retrieval_query_diagnostics(retrieval):
    return [
        {
            "category": x.get("category"),
            "query": x.get("query"),
            "result_count": x.get("result_count"),
            "elapsed_seconds": x.get("elapsed_seconds"),
            "attempt_count": x.get("attempt_count"),
            "retried": bool(x.get("retried")),
            "attempts": x.get("attempts") or [],
        }
        for x in (retrieval or {}).get("queries") or []
    ]


def research_goal_web_context(home, away, match_date):
    """
    GOAL-1.0.8.2 — Brave retrieval survives OpenAI analysis failure.

    Pipeline:
      Brave (2 parallel targeted queries) -> retrieved evidence
      -> bounded OpenAI analysis (no web tool, no retry by default)

    If OpenAI times out, Brave sources/snippets are returned with status PARTIAL.
    No AI-derived fact/signal is created from unanalysed snippets, so downstream
    lambda adjustments remain blocked by web_evidence_verified=False.
    """
    empty = {
        "status": "UNAVAILABLE",
        "web_research_used": False,
        "web_evidence_verified": False,
        "web_evidence_tier": "NONE",
        "source_count": 0,
        "sources": [],
        "retrieved_evidence": [],
        "facts": [],
        "signals": [],
        "summary": "",
        "data_confidence": 0.0,
        "actionable_signal_count": 0,
        "parse_mode": "NONE",
        "diagnostics": {"environment": _goal_environment_diagnostics()},
    }

    if not GOAL_WEB_ENABLED:
        return {**empty, "status": "DISABLED"}

    key = f"{home}|{away}|{str(match_date)[:10]}"
    with _GOAL_WEB_CACHE_LOCK:
        cached = _GOAL_WEB_CACHE.get(key)
        if cached and (time.time() - cached[0] < 1800):
            return dict(cached[1], cache_hit=True)

    pipeline_started = time.perf_counter()

    try:
        retrieval = _goal_retrieve_web_evidence(home, away, match_date)
    except Exception as exc:
        return {
            **empty,
            "status": f"RETRIEVAL_ERROR:{type(exc).__name__}",
            "error": str(exc)[:800],
            "diagnostics": {
                "environment": _goal_environment_diagnostics(),
                "failure_stage": "BRAVE_RETRIEVAL",
                "provider": "BRAVE_SEARCH_API",
                "native_openai_web_search_used": False,
                "pipeline_seconds": round(time.perf_counter() - pipeline_started, 3),
            },
        }

    retrieval_seconds = float(retrieval.get("elapsed_seconds") or 0.0)
    sources = _goal_retrieval_sources(retrieval, include_snippet=False)
    retrieved_evidence = _goal_retrieval_sources(retrieval, include_snippet=True)
    base_diag = {
        "environment": _goal_environment_diagnostics(),
        "provider": retrieval.get("provider") or "BRAVE_SEARCH_API",
        "native_openai_web_search_used": False,
        "retrieval_completed": True,
        "retrieval_query_count": retrieval.get("query_count"),
        "retrieval_result_count": retrieval.get("result_count"),
        "retrieval_seconds": retrieval_seconds,
        "retrieval_errors": retrieval.get("errors"),
        "retrieval_timeout_seconds": float(os.getenv("BETSMART_GOAL_RETRIEVAL_TIMEOUT_SECONDS", "6.0")),
        "retrieval_max_retries": int(os.getenv("BETSMART_GOAL_RETRIEVAL_MAX_RETRIES", "1")),
        "queries": _goal_retrieval_query_diagnostics(retrieval),
        "openai_analysis_started": False,
        "openai_analysis_completed": False,
        "openai_analysis_timeout_seconds": GOAL_OPENAI_WEB_ANALYSIS_TIMEOUT_SECONDS,
        "openai_max_retries": GOAL_OPENAI_WEB_MAX_RETRIES,
        "openai_chat_fallback_enabled": GOAL_OPENAI_WEB_CHAT_FALLBACK_ENABLED,
        "openai_model": GOAL_WEB_MODEL,
        "openai_reasoning_effort": GOAL_OPENAI_REASONING_EFFORT,
        "openai_verbosity": GOAL_OPENAI_VERBOSITY,
        "openai_web_analysis_max_output_tokens": GOAL_OPENAI_WEB_ANALYSIS_MAX_OUTPUT_TOKENS,
    }

    if retrieval.get("result_count", 0) <= 0:
        out = {
            **empty,
            "status": "NO_RETRIEVAL_EVIDENCE",
            "diagnostics": {
                **base_diag,
                "failure_stage": "BRAVE_NO_RESULTS",
                "pipeline_seconds": round(time.perf_counter() - pipeline_started, 3),
            },
        }
        with _GOAL_WEB_CACHE_LOCK:
            _GOAL_WEB_CACHE[key] = (time.time(), out)
        return out

    # Brave succeeded. From this point on Web research HAS been used, even if
    # the semantic OpenAI analysis fails later.
    partial_base = {
        **empty,
        "web_research_used": True,
        "source_count": len(sources),
        "sources": sources,
        "retrieved_evidence": retrieved_evidence,
        "web_evidence_tier": "LOW",
        "summary": "Brave retrieval completed; AI evidence analysis unavailable.",
    }

    try:
        client = get_openai_client()
    except Exception as exc:
        out = {
            **partial_base,
            "status": "PARTIAL_OPENAI_UNAVAILABLE",
            "error": str(exc)[:800],
            "diagnostics": {
                **base_diag,
                "failure_stage": "OPENAI_CLIENT",
                "pipeline_seconds": round(time.perf_counter() - pipeline_started, 3),
            },
        }
        with _GOAL_WEB_CACHE_LOCK:
            _GOAL_WEB_CACHE[key] = (time.time(), out)
        return out

    analysis_started = time.perf_counter()
    base_diag["openai_analysis_started"] = True

    try:
        obj, parse_mode = _goal_analyze_retrieved_evidence(
            client, GOAL_WEB_MODEL, home, away, match_date, retrieval
        )
        analysis_seconds = round(time.perf_counter() - analysis_started, 3)
    except Exception as exc:
        analysis_seconds = round(time.perf_counter() - analysis_started, 3)
        is_timeout = (
            "timeout" in type(exc).__name__.lower()
            or "timeout" in str(exc).lower()
        )
        status = "PARTIAL_OPENAI_TIMEOUT" if is_timeout else "PARTIAL_OPENAI_ERROR"
        failure_stage = "OPENAI_ANALYSIS_TIMEOUT" if is_timeout else "OPENAI_ANALYSIS"
        out = {
            **partial_base,
            "status": status,
            "error": f"{type(exc).__name__}:{str(exc)[:700]}",
            "diagnostics": {
                **base_diag,
                "openai_analysis_completed": False,
                "openai_analysis_seconds": analysis_seconds,
                "failure_stage": failure_stage,
                "partial_brave_evidence_retained": True,
                "pipeline_seconds": round(time.perf_counter() - pipeline_started, 3),
            },
        }
        with _GOAL_WEB_CACHE_LOCK:
            _GOAL_WEB_CACHE[key] = (time.time(), out)
        return out

    # Deterministic enrichment of the compact LLM contract.
    # Brave/Python owns source metadata; OpenAI only extracts concise evidence.
    source_tier_weight = {"A": 0.95, "B": 0.80, "C": 0.55, "D": 0.20}
    source_quality_values = [
        source_tier_weight.get(str(s.get("source_tier") or "C").upper(), 0.45)
        for s in sources
        if isinstance(s, dict)
    ]
    source_quality = (
        sum(source_quality_values) / len(source_quality_values)
        if source_quality_values else 0.0
    )

    raw_facts = obj.get("facts") if isinstance(obj.get("facts"), list) else []
    facts = []
    fact_confidence = round(float(np.clip(source_quality, 0.0, 1.0)), 3)

    for raw_fact in raw_facts[:3]:
        if isinstance(raw_fact, str):
            claim = raw_fact.strip()
        elif isinstance(raw_fact, dict):
            claim = str(raw_fact.get("claim") or "").strip()
        else:
            continue

        if not claim:
            continue

        facts.append({
            "side": "BOTH",
            "category": "WEB_EVIDENCE",
            "claim": claim[:420],
            "metric": None,
            "value": None,
            "sample_size": 0,
            "confidence": fact_confidence,
            "source_support_count": 0,
            "source_urls": [],
        })

    raw_signals = obj.get("signals") if isinstance(obj.get("signals"), list) else []
    cleaned = []
    direction_map = {
        "UP": "MORE_GOALS",
        "DOWN": "FEWER_GOALS",
        "NEUTRAL": "NEUTRAL",
        "MORE_GOALS": "MORE_GOALS",
        "FEWER_GOALS": "FEWER_GOALS",
    }

    for sig in raw_signals[:2]:
        if not isinstance(sig, dict):
            continue

        market = str(sig.get("market") or "").strip()
        if market not in {"BTTS", "Over15", "Over25", "Over35"}:
            continue

        raw_direction = str(sig.get("direction") or "NEUTRAL").upper()
        direction = direction_map.get(raw_direction, "NEUTRAL")

        confidence = round(float(np.clip(
            _goal_safe_float(sig.get("confidence"), 0.0), 0.0, 1.0
        )), 3)
        impact = round(float(np.clip(
            _goal_safe_float(sig.get("impact"), 0.0), -0.20, 0.20
        )), 3)

        if direction == "NEUTRAL":
            impact = 0.0

        cleaned.append({
            "market": market,
            "side": "BOTH",
            "category": "MARKET_SIGNAL",
            "direction": direction,
            "impact": impact,
            "confidence": confidence,
            "summary": str(sig.get("reason") or "")[:420],
        })

    signal_confidences = [
        float(sig["confidence"])
        for sig in cleaned
        if sig["direction"] != "NEUTRAL"
    ]

    if signal_confidences:
        mean_signal_conf = sum(signal_confidences) / len(signal_confidences)
        conf = float(np.clip(
            0.55 * source_quality + 0.45 * mean_signal_conf,
            0.0, 1.0
        ))
    elif facts:
        conf = float(np.clip(0.70 * source_quality, 0.0, 1.0))
    else:
        conf = 0.0

    actionable = [
        sig for sig in cleaned
        if sig["direction"] != "NEUTRAL"
        and sig["confidence"] >= 0.45
        and abs(sig["impact"]) >= 0.10
    ]

    strong_sources = sum(1 for s in sources if s.get("source_tier") in {"A", "B"})
    useful_facts = sum(1 for f in facts if f["confidence"] >= 0.50)
    evidence_present = bool(sources)

    if evidence_present and strong_sources >= 2 and conf >= 0.72 and len(actionable) >= 2:
        tier = "HIGH"
    elif evidence_present and conf >= 0.48 and (len(actionable) >= 1 or useful_facts >= 2):
        tier = "MEDIUM"
    elif evidence_present:
        tier = "LOW"
    else:
        tier = "NONE"

    out = {
        "status": "OK",
        "web_research_used": True,
        "web_evidence_verified": evidence_present,
        "web_evidence_tier": tier,
        "source_count": len(sources),
        "sources": sources,
        "retrieved_evidence": retrieved_evidence,
        "data_confidence": round(conf, 3),
        "facts": facts,
        "fact_count": len(facts),
        "signals": cleaned,
        "actionable_signal_count": len(actionable),
        "summary": str(obj.get("summary") or "")[:700],
        "parse_mode": parse_mode,
        "diagnostics": {
            **base_diag,
            "openai_analysis_completed": True,
            "openai_analysis_seconds": analysis_seconds,
            "failure_stage": None,
            "partial_brave_evidence_retained": False,
            "pipeline_seconds": round(time.perf_counter() - pipeline_started, 3),
        },
        "researched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    with _GOAL_WEB_CACHE_LOCK:
        _GOAL_WEB_CACHE[key] = (time.time(), out)

    return out


def _goal_web_strategic_leverage(hist_ctx, web_ctx):
    """
    Diagnostic weight for arbitration, NOT a direct mathematical blend.

    The weaker BetSmart's internal current-season history is, the more room
    verified Web Intelligence may receive. If Web evidence is weak/unverified,
    the weight remains low regardless of internal weakness.
    """
    form_rel = float(np.clip(
        _goal_safe_float((hist_ctx or {}).get("current_form_reliability"), 0.0),
        0.0, 1.0
    ))

    venue = (hist_ctx or {}).get("venue_history") or {}
    home_n = int(((venue.get("home_at_home") or {}).get("matches", 0)) or 0)
    away_n = int(((venue.get("away_at_away") or {}).get("matches", 0)) or 0)
    venue_rel = min(1.0, min(home_n, away_n) / 8.0)

    h2h_n = int((((hist_ctx or {}).get("h2h") or {}).get("matches", 0)) or 0)
    h2h_rel = min(1.0, h2h_n / 6.0)

    internal_reliability = float(np.clip(
        0.60 * form_rel + 0.28 * venue_rel + 0.12 * h2h_rel,
        0.0, 1.0
    ))

    web_conf = float(np.clip(
        _goal_safe_float((web_ctx or {}).get("data_confidence"), 0.0),
        0.0, 1.0
    ))
    web_verified = bool((web_ctx or {}).get("web_evidence_verified", False))
    actionable = int((web_ctx or {}).get("actionable_signal_count", 0) or 0)
    fact_count = int((web_ctx or {}).get("fact_count", 0) or 0)

    evidence_factor = web_conf
    if not web_verified:
        evidence_factor *= 0.20
    if actionable == 0 and fact_count == 0:
        evidence_factor *= 0.25

    recommended_web_weight = float(np.clip(
        0.10 + 0.55 * (1.0 - internal_reliability) * evidence_factor,
        0.10, 0.60
    ))

    return {
        "internal_reliability": round(internal_reliability, 3),
        "web_data_confidence": round(web_conf, 3),
        "web_verified": web_verified,
        "web_actionable_signals": actionable,
        "web_fact_count": fact_count,
        "recommended_web_weight": round(recommended_web_weight, 3),
        "policy": "ARBITRATION_GUIDANCE_ONLY",
    }



def _goal_dedupe_injuries(items):
    seen = set()
    output = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        key = (
            str(item.get("name") or "").strip().lower(),
            str(item.get("reason") or "").strip().lower(),
            str(item.get("type") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _goal_shutdown_executor_now(executor):
    """
    Do not wait for timed-out external calls when returning the HTTP response.
    """
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        executor.shutdown(wait=False)
    except Exception:
        pass



def _goal_build_external_context_parallel(home, away, date):
    """
    V1.0.8 — true concurrent timeout orchestration.

    Real-Time and Web start at the same instant and are observed against their
    own deadlines. Their timeout budgets are NEVER added sequentially.

    Example with defaults:
      realtime deadline = 10 s
      web deadline      = 16 s
      global context    <= ~16 s (+ tiny orchestration overhead)

    A branch that finishes is preserved immediately even if the other branch
    later times out.
    """
    default_rt = {
        "available": False,
        "fixture_id": None,
        "injuries_home": [],
        "injuries_away": [],
        "lineups_available": False,
        "status": "UNAVAILABLE",
    }
    default_web = {
        "status": "UNAVAILABLE",
        "web_research_used": False,
        "web_evidence_verified": False,
        "web_evidence_tier": "NONE",
        "source_count": 0,
        "sources": [],
        "facts": [],
        "signals": [],
        "summary": "",
        "data_confidence": 0.0,
        "actionable_signal_count": 0,
        "parse_mode": "NONE",
        "diagnostics": {"environment": _goal_environment_diagnostics()},
    }

    started = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=max(2, GOAL_CONTEXT_WORKERS))

    futures = {
        "realtime": pool.submit(research_goal_realtime_context, home, away, date),
        "web": pool.submit(research_goal_web_context, home, away, date),
    }
    deadlines = {
        "realtime": started + max(0.1, float(GOAL_REALTIME_TIMEOUT_SECONDS)),
        "web": started + max(0.1, float(GOAL_WEB_TIMEOUT_SECONDS)),
    }
    defaults = {"realtime": default_rt, "web": default_web}
    results = {}
    branch_finished_at = {}

    pending = set(futures.keys())

    # Poll both futures together. This avoids the old:
    # wait RT 10 s -> then wait Web 16 s -> 26 s total.
    while pending:
        now = time.monotonic()

        for name in list(pending):
            fut = futures[name]

            if fut.done():
                try:
                    results[name] = fut.result()
                except Exception as exc:
                    results[name] = {
                        **defaults[name],
                        "status": f"ERROR:{type(exc).__name__}",
                        "error": str(exc)[:700],
                        "diagnostics": ({"environment": _goal_environment_diagnostics()} if name == "web" else defaults[name].get("diagnostics", {})),
                    }
                branch_finished_at[name] = time.monotonic()
                pending.remove(name)
                continue

            if now >= deadlines[name]:
                fut.cancel()
                results[name] = {
                    **defaults[name],
                    "status": "TIMEOUT",
                    "diagnostics": ({"environment": _goal_environment_diagnostics()} if name == "web" else defaults[name].get("diagnostics", {})),
                    "timeout_seconds": (
                        GOAL_REALTIME_TIMEOUT_SECONDS
                        if name == "realtime"
                        else GOAL_WEB_TIMEOUT_SECONDS
                    ),
                }
                branch_finished_at[name] = now
                pending.remove(name)

        if pending:
            # Sleep only until the nearest remaining branch deadline.
            nearest = min(deadlines[name] for name in pending)
            remaining = max(0.0, nearest - time.monotonic())
            time.sleep(min(0.05, remaining))

    _goal_shutdown_executor_now(pool)

    rt_ctx = dict(results.get("realtime") or default_rt)
    web_ctx = dict(results.get("web") or default_web)

    rt_ctx["injuries_home"] = _goal_dedupe_injuries(rt_ctx.get("injuries_home"))
    rt_ctx["injuries_away"] = _goal_dedupe_injuries(rt_ctx.get("injuries_away"))

    rt_branch = max(
        0.0,
        branch_finished_at.get("realtime", time.monotonic()) - started
    )
    web_branch = max(
        0.0,
        branch_finished_at.get("web", time.monotonic()) - started
    )
    elapsed = max(rt_branch, web_branch)

    rt_ctx["_branch_seconds"] = round(rt_branch, 3)
    web_ctx["_branch_seconds"] = round(web_branch, 3)
    rt_ctx["_parallel_context_seconds"] = round(elapsed, 3)
    web_ctx["_parallel_context_seconds"] = round(elapsed, 3)
    rt_ctx["_timeout_budget_seconds"] = GOAL_REALTIME_TIMEOUT_SECONDS
    web_ctx["_timeout_budget_seconds"] = GOAL_WEB_TIMEOUT_SECONDS

    return rt_ctx, web_ctx



def _goal_compact_arbitration_payload(payload):
    """Compact only the final arbitration input; Web evidence has already been validated."""
    payload = payload or {}
    web = payload.get("web_intelligence") or {}
    rt = payload.get("realtime_context") or {}

    facts = []
    for f in (web.get("facts") or [])[:3]:
        facts.append({
            "claim": str((f or {}).get("claim") or "")[:320] if isinstance(f, dict) else str(f)[:320],
            "confidence": (f or {}).get("confidence") if isinstance(f, dict) else None,
        })

    signals = []
    for s in (web.get("signals") or [])[:3]:
        if not isinstance(s, dict):
            continue
        signals.append({
            "market": s.get("market"),
            "direction": s.get("direction"),
            "confidence": s.get("confidence"),
            "impact": s.get("impact"),
            "summary": str(s.get("summary") or "")[:260],
        })

    return _goal_jsonable({
        "teams": payload.get("teams"),
        "match_date": payload.get("match_date"),
        "base_goal_model": payload.get("base_goal_model"),
        "historical_context": payload.get("historical_context"),
        "model_market_disagreement": payload.get("model_market_disagreement"),
        "market_context": payload.get("market_context"),
        "realtime_context": {
            "status": rt.get("status"),
            "available": rt.get("available"),
            "lineups_available": rt.get("lineups_available"),
            "injuries_home": (rt.get("injuries_home") or [])[:6],
            "injuries_away": (rt.get("injuries_away") or [])[:6],
        },
        "web_intelligence": {
            "status": web.get("status"),
            "web_evidence_verified": web.get("web_evidence_verified"),
            "web_evidence_tier": web.get("web_evidence_tier"),
            "data_confidence": web.get("data_confidence"),
            "actionable_signal_count": web.get("actionable_signal_count"),
            "summary": web.get("summary"),
            "facts": facts,
            "signals": signals,
        },
        "web_strategic_leverage": payload.get("web_strategic_leverage"),
    })

def _goal_ai_schema_prompt(payload):
    compact = _goal_compact_arbitration_payload(payload)
    return f"""
BETSMART GOAL INTELLIGENCE — FINAL ARBITRATION.

OBJECTIF: ajuster prudemment lambda_home/lambda_away selon les preuves validées.
Le marché est informatif seulement.

RÈGLES:
1. Deltas lambda entre -0.40 et +0.40 par équipe; zéro est une décision valide.
2. Ne jamais inventer poste, rôle, importance, blessure ou statistique.
3. Une information ambiguë, ancienne ou contradictoire doit réduire l'ajustement.
4. H2H est secondaire; données récentes quantifiées et vérifiées priment.
5. Blessure/composition n'influence les buts que si l'effet sportif est explicitement contextualisé.
6. Le marché ne décide jamais seul.
7. Tu ne modifies jamais directement BTTS, Over15, Over25 ou Over35.
8. Ton seul levier quantitatif est lambda_home_delta / lambda_away_delta; zéro est valide.
9. Les probabilités finales seront recalculées mathématiquement par Poisson à partir des lambdas finaux.
10. L'explication fournie ici doit rester très courte (1 à 2 phrases) et identifier seulement les preuves qui motivent tes deltas lambda.
11. N'essaie pas de justifier les probabilités finales: elles seront calculées ensuite par Python et expliquées à partir de l'état final réel.

DOSSIER COMPACT:
{json.dumps(compact, ensure_ascii=False, separators=(",",":"))}
""".strip()

def goal_ai_arbitrator(payload):
    if not GOAL_AI_ENABLED:
        return {
            "status":"DISABLED","lambda_home_delta":0.0,"lambda_away_delta":0.0,
            "market_adjustments":{"BTTS":0.0,"Over15":0.0,"Over25":0.0,"Over35":0.0},
            "prediction_confidence":0.0,"source_agreement":"LOW","risk_level":"UNKNOWN",
            "reason_codes":["AI_DISABLED"],"rationale_short":"","explanation":"",
            "arbitration_diagnostics":{"called":False,"completed":False,"reason":"AI_DISABLED"},
        }
    prompt=_goal_ai_schema_prompt(payload)
    t0=time.perf_counter()
    try:
        client=get_openai_client()
        obj=_goal_structured_ai_call(client,GOAL_AI_MODEL,prompt,schema_name="betsmart_goal_arbitration_v10883_stable")
        elapsed=time.perf_counter()-t0
        agreement=str(obj.get("source_agreement") or "LOW").upper()
        if agreement not in {"LOW","MEDIUM","HIGH"}: agreement="LOW"
        risk=str(obj.get("risk_level") or "UNKNOWN").upper()
        if risk not in {"LOW","MEDIUM","HIGH"}: risk="UNKNOWN"
        return {
            "status":"OK",
            "lambda_home_delta":float(np.clip(_goal_safe_float(obj.get("lambda_home_delta"),0.0),-0.40,0.40)),
            "lambda_away_delta":float(np.clip(_goal_safe_float(obj.get("lambda_away_delta"),0.0),-0.40,0.40)),
            "prediction_confidence":float(np.clip(_goal_safe_float(obj.get("prediction_confidence"),0.0),0.0,1.0)),
            "source_agreement":agreement,"risk_level":risk,
            "reason_codes":[str(x)[:120] for x in (obj.get("reason_codes") or [])][:8],
            "rationale_short":str(obj.get("rationale_short") or "")[:900],
            "explanation":str(obj.get("explanation") or "")[:2400],
            "arbitration_diagnostics":{
                "called":True,"completed":True,"seconds":round(elapsed,3),
                "model":GOAL_AI_MODEL,"reasoning_effort":GOAL_OPENAI_REASONING_EFFORT,
                "verbosity":GOAL_OPENAI_VERBOSITY,"timeout_seconds":GOAL_OPENAI_ARBITRATION_TIMEOUT_SECONDS,
                "max_output_tokens":GOAL_OPENAI_ARBITRATION_MAX_OUTPUT_TOKENS,
                "max_retries":GOAL_OPENAI_ARBITRATION_MAX_RETRIES,
                "chat_fallback_enabled":GOAL_OPENAI_ARBITRATION_CHAT_FALLBACK_ENABLED,
                "prompt_chars":len(prompt),
            },
        }
    except Exception as exc:
        elapsed=time.perf_counter()-t0
        return {
            "status":"ERROR","lambda_home_delta":0.0,"lambda_away_delta":0.0,
            "market_adjustments":{"BTTS":0.0,"Over15":0.0,"Over25":0.0,"Over35":0.0},
            "prediction_confidence":0.0,"source_agreement":"LOW","risk_level":"UNKNOWN",
            "reason_codes":[f"AI_ERROR:{type(exc).__name__}"],"rationale_short":"","explanation":"",
            "error":str(exc)[:1200],
            "arbitration_diagnostics":{
                "called":True,"completed":False,"seconds":round(elapsed,3),"model":GOAL_AI_MODEL,
                "timeout_seconds":GOAL_OPENAI_ARBITRATION_TIMEOUT_SECONDS,
                "max_output_tokens":GOAL_OPENAI_ARBITRATION_MAX_OUTPUT_TOKENS,
                "max_retries":GOAL_OPENAI_ARBITRATION_MAX_RETRIES,
                "chat_fallback_enabled":GOAL_OPENAI_ARBITRATION_CHAT_FALLBACK_ENABLED,
                "prompt_chars":len(prompt),"error_type":type(exc).__name__,
            },
        }

def _goal_evidence_strength(hist_ctx, web_ctx, rt_ctx, ai):
    form_rel = _goal_safe_float(hist_ctx.get("current_form_reliability"), 0.0)
    h2h_n = int((hist_ctx.get("h2h") or {}).get("matches", 0) or 0)
    web_conf = _goal_safe_float(web_ctx.get("data_confidence"), 0.0) if isinstance(web_ctx, dict) else 0.0
    web_verified = bool(web_ctx.get("web_evidence_verified")) if isinstance(web_ctx, dict) else False
    rt_available = bool(rt_ctx.get("available")) if isinstance(rt_ctx, dict) else False
    agreement = str(ai.get("source_agreement") or "LOW").upper()

    score = 0.25
    score += 0.25 * form_rel
    score += 0.12 * min(1.0, h2h_n / 6.0)
    score += 0.20 * web_conf * (1.0 if web_verified else 0.25)
    score += 0.08 if rt_available else 0.0
    score += {"LOW": 0.0, "MEDIUM": 0.05, "HIGH": 0.10}.get(agreement, 0.0)
    return float(np.clip(score, 0.20, 1.0))




def _goal_stabilize_deltas(hist_ctx, web_ctx, rt_ctx, ai):
    strength = _goal_evidence_strength(hist_ctx, web_ctx, rt_ctx, ai)
    cap = 0.12 + 0.23 * strength
    agreement = str(ai.get("source_agreement") or "LOW").upper()
    form_rel = _goal_safe_float(hist_ctx.get("current_form_reliability"), 0.0)

    if agreement == "LOW":
        cap = min(cap, 0.18)
    if form_rel < 0.40:
        cap = min(cap, 0.20)

    raw_h = float(np.clip(_goal_safe_float(ai.get("lambda_home_delta")), -0.40, 0.40))
    raw_a = float(np.clip(_goal_safe_float(ai.get("lambda_away_delta")), -0.40, 0.40))

    web_actionable = int((web_ctx or {}).get("actionable_signal_count", 0) or 0)
    web_verified = bool((web_ctx or {}).get("web_evidence_verified", False))
    h2h_matches = int(((hist_ctx.get("h2h") or {}).get("matches", 0)) or 0)
    rt_available = bool((rt_ctx or {}).get("available", False))

    thin_evidence = (
        form_rel < 0.40
        and h2h_matches < 3
        and (not web_verified or web_actionable == 0)
    )

    if thin_evidence:
        raw_total = raw_h + raw_a
        if raw_total > 0:
            thin_cap = 0.05
            scale = min(1.0, thin_cap / max(raw_total, 1e-9))
            raw_h *= scale
            raw_a *= scale

    single_source_injury_guard = (
        rt_available
        and web_actionable == 0
        and form_rel < 0.40
        and h2h_matches < 3
    )

    if single_source_injury_guard:
        raw_h = float(np.clip(raw_h, -0.12, 0.12))
        raw_a = float(np.clip(raw_a, -0.12, 0.12))
        cap = min(cap, 0.12)

    dh = float(np.clip(raw_h, -cap, cap))
    da = float(np.clip(raw_a, -cap, cap))

    return dh, da, {
        "evidence_strength": round(strength, 3),
        "delta_cap": round(cap, 3),
        "raw_lambda_home_delta": round(_goal_safe_float(ai.get("lambda_home_delta")), 3),
        "raw_lambda_away_delta": round(_goal_safe_float(ai.get("lambda_away_delta")), 3),
        "applied_lambda_home_delta": round(dh, 3),
        "applied_lambda_away_delta": round(da, 3),
        "thin_evidence_guard": bool(thin_evidence),
        "single_source_injury_guard": bool(single_source_injury_guard),
        "web_actionable_signal_count": web_actionable,
    }




def _goal_should_skip_ai(hist_ctx, web_ctx, rt_ctx):
    """
    V1.0.8.8.4 policy: always run the final AI arbitrator in lambda-only mode.

    The arbitrator may still return zero deltas. Stability guards, bounded
    lambda deltas remain active. Direct market probability adjustments are
    disabled by architecture: final BTTS/Over probabilities come only from the
    final lambdas through the Poisson recalculation.
    """
    web_actionable = int((web_ctx or {}).get("actionable_signal_count", 0) or 0)
    web_conf = float(np.clip(_goal_safe_float((web_ctx or {}).get("data_confidence"), 0.0), 0.0, 1.0))
    web_verified = bool((web_ctx or {}).get("web_evidence_verified", False))
    lineups = bool((rt_ctx or {}).get("lineups_available", False))
    return False, {
        "reason": "FORCE_AI_ARBITRATION_POLICY",
        "web_actionable_signal_count": web_actionable,
        "web_data_confidence": round(web_conf, 3),
        "web_verified": web_verified,
        "lineups_available": lineups,
    }


def _goal_apply_ai_market_adjustments(final_probs, ai, web_ctx, evidence_strength):
    """
    V1.0.8.8.3 LAMBDA-ONLY: direct probability adjustments are disabled.

    The returned probabilities are an unchanged copy of the Poisson values
    recalculated from the final lambdas. A zeroed diagnostic object is kept for
    backward compatibility with existing API consumers.
    """
    probs = {k: float(np.clip(v, 0.0, 1.0)) for k, v in final_probs.items()}

    # Non-negotiable hierarchy for cumulative total-goal markets.
    probs["Over25"] = min(probs["Over25"], probs["Over15"])
    probs["Over35"] = min(probs["Over35"], probs["Over25"])

    return probs, {
        "allowed": False,
        "policy": "LAMBDA_ONLY",
        "reason": "DIRECT_MARKET_ADJUSTMENTS_DISABLED",
        "cap": 0.0,
        "evidence_strength": round(float(evidence_strength or 0.0), 3),
        "web_actionable_signal_count": int((web_ctx or {}).get("actionable_signal_count", 0) or 0),
        "web_data_confidence": round(float(np.clip(_goal_safe_float((web_ctx or {}).get("data_confidence"), 0.0), 0.0, 1.0)), 3),
        "applied": {"BTTS":0.0,"Over15":0.0,"Over25":0.0,"Over35":0.0},
    }

def _goal_final_probs(base_res, lam_h_base, lam_a_base, lam_h_final, lam_a_final):
    """
    V1.0.8.8.5: the public final market state is derived exclusively from
    the final lambdas. The calibrated O2.5/BTTS models remain available in
    model_raw/base_res for diagnostics, but they do not alter the final state.
    """
    final_total = float(lam_h_final + lam_a_final)

    p15 = _clip01(_p_over(final_total, 1.5))
    p25 = _clip01(_p_over(final_total, 2.5))
    p35 = _clip01(_p_over(final_total, 3.5))
    pbtts = _clip01(_p_btts(lam_h_final, lam_a_final))

    # Numerical guard only; mathematically Poisson already satisfies this.
    p25 = min(p25, p15)
    p35 = min(p35, p25)

    return {
        "Over15": p15,
        "Over25": p25,
        "Over35": p35,
        "BTTS": pbtts,
    }


def _goal_distribution(lam_total, max_goals=6):
    probs = {}
    used = 0.0
    for k in range(max_goals):
        p = float(poisson.pmf(k, lam_total))
        probs[str(k)] = round(p, 4)
        used += p
    probs[f"{max_goals}+"] = round(max(0.0, 1.0 - used), 4)
    return probs


def _goal_most_likely_score(lam_h, lam_a, max_goals=5):
    best = (None, -1.0)
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = float(poisson.pmf(h, lam_h) * poisson.pmf(a, lam_a))
            if p > best[1]:
                best = ((h, a), p)
    (h, a), p = best
    return {"score": f"{h}-{a}", "probability": round(p, 4)}




def _goal_market_disagreement(base_res, market_ctx):
    diffs = {}
    vals = []
    for market in ["Over15","Over25","Over35","BTTS"]:
        bp = (base_res.get(market) or {}).get("proba")
        mp = (market_ctx.get(market) or {}).get("market_probability_raw")
        if bp is None or mp is None:
            continue
        d = abs(float(bp) - float(mp))
        diffs[market] = round(d, 4)
        vals.append(d)

    if not vals:
        return {
            "available": False,
            "level": "UNKNOWN",
            "mean_abs_diff": None,
            "max_abs_diff": None,
            "by_market": {},
            "decision_impact": "CONFIDENCE_ONLY",
        }

    mean_diff = float(np.mean(vals))
    max_diff = float(np.max(vals))
    if max_diff >= 0.20 or mean_diff >= 0.15:
        level = "VERY_HIGH"
    elif max_diff >= 0.14 or mean_diff >= 0.10:
        level = "HIGH"
    elif max_diff >= 0.08 or mean_diff >= 0.06:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "available": True,
        "level": level,
        "mean_abs_diff": round(mean_diff, 4),
        "max_abs_diff": round(max_diff, 4),
        "by_market": diffs,
        "decision_impact": "CONFIDENCE_ONLY",
    }


def _goal_quantitative_confidence(base_res, hist_ctx, market_ctx):
    probs = []
    for market in ["Over15","Over25","Over35","BTTS"]:
        p = _goal_safe_float((base_res.get(market) or {}).get("proba"), 0.5)
        probs.append(abs(p - 0.5) * 2.0)

    signal_strength = float(np.mean(probs)) if probs else 0.0
    form_rel = _goal_safe_float(hist_ctx.get("current_form_reliability"), 0.0)

    vh = hist_ctx.get("venue_history") or {}
    hm = int((vh.get("home_at_home") or {}).get("matches", 0) or 0)
    am = int((vh.get("away_at_away") or {}).get("matches", 0) or 0)
    hist_depth = min(1.0, min(hm, am) / 20.0)

    agreements = []
    for market in ["Over15","Over25","Over35","BTTS"]:
        mp = (market_ctx.get(market) or {}).get("market_probability_raw")
        bp = (base_res.get(market) or {}).get("proba")
        if mp is not None and bp is not None:
            agreements.append(max(0.0, 1.0 - abs(float(mp)-float(bp)) / 0.30))

    market_agreement = float(np.mean(agreements)) if agreements else 0.5

    conf = (
        0.30
        + 0.28 * signal_strength
        + 0.18 * form_rel
        + 0.14 * hist_depth
        + 0.10 * market_agreement
    )
    return float(np.clip(conf, 0.35, 0.78))



_GOAL_PLAYER_EXPLANATION_CACHE = {}
_GOAL_PLAYER_EXPLANATION_CACHE_LOCK = threading.RLock()
GOAL_PLAYER_EXPLANATION_TIMEOUT_SECONDS = float(
    os.getenv("BETSMART_GOAL_PLAYER_EXPLANATION_TIMEOUT_SECONDS", "6.0")
)


def _goal_missing_fixture_names(rt_ctx):
    rt_ctx = rt_ctx or {}

    def _collect(items):
        out = []
        for x in (items or []):
            if not isinstance(x, dict):
                continue
            if str(x.get("type") or "").strip().lower() != "missing fixture":
                continue
            name = str(x.get("name") or "").strip()
            if name:
                out.append(name)
        return out

    return {
        "HOME": _collect(rt_ctx.get("injuries_home")),
        "AWAY": _collect(rt_ctx.get("injuries_away")),
    }



def _goal_player_evidence_rank(tier):
    return {"A": 4, "B": 3, "C": 2, "D": 1}.get(str(tier or "").upper(), 0)


def _goal_player_role_patterns():
    return {
        "GOALKEEPER": [
            r"\bgoalkeeper\b", r"\bkeeper\b", r"\bgardien\b",
        ],
        "DEFENDER": [
            r"\bdefender\b", r"\bcentre[- ]back\b", r"\bcenter[- ]back\b",
            r"\bfull[- ]back\b", r"\bleft[- ]back\b", r"\bright[- ]back\b",
            r"\bd[ée]fenseur\b",
        ],
        "ATTACKER": [
            r"\bstriker\b", r"\bforward\b", r"\bwinger\b", r"\battacker\b",
            r"\bcentre[- ]forward\b", r"\bcenter[- ]forward\b",
            r"\battaquant\b", r"\bbuteur\b", r"\bailier\b",
        ],
        "MIDFIELDER": [
            r"\bmidfielder\b", r"\bmidfield\b", r"\bmilieu\b",
        ],
    }


def _goal_player_name_tokens(rt_name, title="", snippet=""):
    """
    Return stable identity tokens for local role attribution.

    We always keep the RT surname (e.g. Mateta). If a fuller name appears
    immediately before that surname in title/snippet, its tokens are also kept.
    """
    raw = str(rt_name or "").strip()
    tokens = [
        t.lower() for t in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]+", raw)
        if len(t) >= 3
    ]
    surname = tokens[-1] if tokens else ""

    text = f"{title} {snippet}"
    if surname:
        # Capture up to two proper-name tokens immediately before the surname.
        m = re.search(
            rf"\b((?:[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’-]+\s+){{0,2}}{re.escape(surname)})\b",
            text,
            flags=re.I,
        )
        if m:
            for t in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]+", m.group(1)):
                tl = t.lower()
                if len(tl) >= 3 and tl not in tokens:
                    tokens.append(tl)

    return list(dict.fromkeys(tokens))


def _goal_sentence_split(text):
    # Keep this deliberately cheap: no NLP dependency, no network, no model.
    return [
        x.strip()
        for x in re.split(r"(?<=[.!?])\s+|\s*[|;]\s*", str(text or ""))
        if x and x.strip()
    ]


def _goal_role_mentions(sentence, role_patterns):
    found = []
    for role, patterns in role_patterns.items():
        for p in patterns:
            for m in re.finditer(p, sentence, flags=re.I):
                found.append((role, m.start(), m.end(), m.group(0)))
    return found


def _goal_role_local_score(sentence, role_start, role_end, name_tokens):
    """
    Score whether a role word qualifies the target player rather than another
    footballer mentioned in the same sentence.
    """
    sent = str(sentence or "")
    low = sent.lower()
    name_positions = []

    for token in name_tokens:
        for m in re.finditer(rf"\b{re.escape(token)}\b", low, flags=re.I):
            name_positions.append((m.start(), m.end(), token))

    # Strong anaphoric descriptor only when the text before the role is itself
    # a compact noun phrase: "The French striker", "The defender",
    # "L'attaquant". Do not trigger on "Mateta passed to Dodo, a defender".
    prefix = low[:role_start].strip(" ,:-")
    anaphoric_prefix = re.fullmatch(
        r"(?:the|a|an|le|la|l['’]|un|une)"
        r"(?:\s+(?:french|english|spanish|portuguese|brazilian|belgian|german|"
        r"italian|dutch|senegalese|congolese|ivorian|ghanaian|nigerian|moroccan|"
        r"algerian|young|veteran|experienced|international)){0,2}",
        prefix,
        flags=re.I,
    )
    if role_start <= 35 and anaphoric_prefix:
        return 72

    if not name_positions:
        return 0

    best = 0
    for ns, ne, _ in name_positions:
        if role_start >= ne:
            distance = role_start - ne
        elif ns >= role_end:
            distance = ns - role_end
        else:
            distance = 0

        # Role close to the target name.
        if distance <= 12:
            score = 100
        elif distance <= 28:
            score = 88
        elif distance <= 50:
            score = 72
        elif distance <= 80:
            score = 52
        else:
            score = 0

        if not score:
            continue

        # Penalize constructions where the role clearly names another person:
        # "Fiorentina defender Dodo", "forward Smith", etc.
        tail = sent[role_end:role_end + 45]
        other_name = re.match(
            r"[\s,]*(?:for\s+\w+\s+)?([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’-]{2,})",
            tail,
        )
        if other_name:
            other = other_name.group(1).lower()
            if other not in name_tokens:
                score -= 70

        best = max(best, score)

    return max(0, best)


def _goal_detect_local_player_role(rt_name, title, snippet):
    """
    Deterministic local semantic attribution.

    Priority:
      1) role attached locally to player's own name;
      2) anaphoric role in the immediately following sentence;
      3) single unambiguous role in the whole evidence item.

    This prevents "Fiorentina defender Dodo" from making Mateta a defender.
    """
    role_patterns = _goal_player_role_patterns()
    name_tokens = _goal_player_name_tokens(rt_name, title, snippet)
    combined = f"{title}. {snippet}".strip()
    sentences = _goal_sentence_split(combined)

    candidates = []
    prior_mentions_player = False

    for sent_idx, sentence in enumerate(sentences):
        low = sentence.lower()
        has_player = any(
            re.search(rf"\b{re.escape(tok)}\b", low, flags=re.I)
            for tok in name_tokens
        )

        mentions = _goal_role_mentions(sentence, role_patterns)
        for role, rs, re_, matched in mentions:
            score = _goal_role_local_score(sentence, rs, re_, name_tokens)

            # If previous sentence names target and this starts with an anaphoric
            # descriptor ("The French striker..."), strengthen attribution.
            if prior_mentions_player and rs <= 35:
                prefix = low[:rs].strip(" ,:-")
                if re.fullmatch(
                    r"(?:the|a|an|le|la|l['’]|un|une)"
                    r"(?:\s+(?:french|english|spanish|portuguese|brazilian|belgian|"
                    r"german|italian|dutch|senegalese|congolese|ivorian|ghanaian|"
                    r"nigerian|moroccan|algerian|young|veteran|experienced|international)){0,2}",
                    prefix,
                    flags=re.I,
                ):
                    score = max(score, 94)

            if score > 0:
                candidates.append({
                    "role": role,
                    "score": score,
                    "sentence": sentence,
                    "matched": matched,
                    "sentence_index": sent_idx,
                })

        prior_mentions_player = has_player

    if candidates:
        candidates.sort(
            key=lambda x: (
                -x["score"],
                x["sentence_index"],
                {"ATTACKER": 0, "DEFENDER": 1, "MIDFIELDER": 2, "GOALKEEPER": 3}.get(x["role"], 9),
            )
        )
        best = candidates[0]

        # Require a meaningful local attribution.
        if best["score"] >= 52:
            return best["role"], best["sentence"], best["score"]

    # Conservative fallback: only if exactly one role category exists anywhere.
    global_roles = set()
    for role, patterns in role_patterns.items():
        if any(re.search(p, combined, flags=re.I) for p in patterns):
            global_roles.add(role)

    if len(global_roles) == 1:
        role = next(iter(global_roles))
        return role, combined[:220], 40

    return "UNKNOWN", "", 0


def _goal_deterministic_player_from_evidence(rt_name, side, items):
    """
    Deterministic EXPLANATION_ONLY player qualification.

    V1.0.8.8.13 fixes role attribution: a role must qualify the target player
    locally. Role words describing another named player in the same snippet are
    ignored. No LLM and no extra network request.
    """
    if not items:
        return None

    items = sorted(
        [x for x in items if isinstance(x, dict)],
        key=lambda x: (
            -_goal_player_evidence_rank(x.get("tier")),
            str(x.get("url") or ""),
            str(x.get("title") or ""),
        )
    )

    role = "UNKNOWN"
    role_score = -1
    importance = "UNKNOWN"
    goals = None
    appearances = None
    evidence_used = []
    best_tier = None

    key_patterns = [
        r"\btop scorer\b", r"\bleading scorer\b", r"\btop goalscorer\b",
        r"\bkey player\b", r"\bkey attacker\b", r"\bkey defender\b",
        r"\bmain striker\b", r"\bfirst[- ]choice\b", r"\bclub captain\b",
        r"\bcaptain\b", r"\btalisman\b",
        r"\bmeilleur buteur\b", r"\bjoueur cl[ée]\b", r"\bcapitaine\b",
        r"\btitulaire indiscutable\b",
    ]

    identity_tokens = _goal_player_name_tokens(rt_name)

    for item in items:
        title = str(item.get("title") or "")
        snippet = str(item.get("snippet") or "")
        text = f"{title}. {snippet}"
        low = text.lower()
        tier = str(item.get("tier") or "").upper()

        if best_tier is None or _goal_player_evidence_rank(tier) > _goal_player_evidence_rank(best_tier):
            best_tier = tier

        detected_role, role_evidence, score = _goal_detect_local_player_role(
            rt_name, title, snippet
        )
        if detected_role != "UNKNOWN" and score > role_score:
            role = detected_role
            role_score = score
            evidence_used = [
                e for e in evidence_used
                if not str(e).startswith("ROLE:")
            ]
            evidence_used.append(
                f"ROLE:{detected_role}:SCORE={score}:{role_evidence[:180]}"
            )

        # KEY evidence must be in an item that actually references the target
        # player identity, not merely another player/team member.
        item_mentions_target = any(
            re.search(rf"\b{re.escape(tok)}\b", low, flags=re.I)
            for tok in identity_tokens
        )
        if (
            importance != "KEY"
            and item_mentions_target
            and any(re.search(p, low, flags=re.I) for p in key_patterns)
        ):
            importance = "KEY"
            evidence_used.append(
                f"IMPORTANCE:KEY:{snippet[:180] or title[:180]}"
            )

        if goals is None and item_mentions_target:
            gm = re.search(
                r"\b(\d{1,2})\s+(?:league\s+)?goals?\b",
                low,
                flags=re.I,
            )
            if gm:
                goals = int(gm.group(1))

        if appearances is None and item_mentions_target:
            am = re.search(
                r"\b(\d{1,3})\s+appearances?\b",
                low,
                flags=re.I,
            )
            if am:
                appearances = int(am.group(1))

    if role == "UNKNOWN":
        return None

    if importance != "KEY":
        importance = "REGULAR"

    tier_conf = {"A": 0.96, "B": 0.90, "C": 0.82, "D": 0.68}.get(
        best_tier, 0.72
    )
    local_conf = 0.96 if role_score >= 90 else 0.90 if role_score >= 70 else 0.82
    confidence = min(tier_conf, local_conf)

    return {
        "rt_name": str(rt_name),
        "name": str(rt_name),
        "side": str(side).upper(),
        "role": role,
        "importance": importance,
        "goals": goals,
        "appearances": appearances,
        "confidence": round(confidence, 3),
        "evidence": " | ".join(evidence_used)[:350],
        "classification_mode": "PYTHON_DETERMINISTIC_LOCAL_ATTRIBUTION",
        "role_attribution_score": int(role_score),
    }


def _goal_deterministic_players_from_evidence(compact_evidence, missing):
    grouped = {}
    for item in (compact_evidence or []):
        if not isinstance(item, dict):
            continue
        side = str(item.get("side") or "").upper()
        rt_name = str(item.get("rt_name") or "").strip()
        if side not in {"HOME", "AWAY"} or not rt_name:
            continue
        if rt_name not in (missing or {}).get(side, []):
            continue
        grouped.setdefault((side, rt_name), []).append(item)

    players = []
    for side in ("HOME", "AWAY"):
        for rt_name in (missing or {}).get(side, []):
            p = _goal_deterministic_player_from_evidence(
                rt_name, side, grouped.get((side, rt_name), [])
            )
            if p:
                players.append(p)
    return players



def _goal_research_player_explanation_context(home, away, match_date, rt_ctx):
    """
    EXPLANATION-ONLY branch.

    Correction V1.0.8.8.10:
    - one Brave query per confirmed Missing Fixture player;
    - all player queries run concurrently;
    - each result keeps rt_name + side + team so qualification can be
      cross-checked deterministically;
    - player evidence NEVER enters predictive Web signals or arbitration.
    """
    missing = _goal_missing_fixture_names(rt_ctx)
    if not missing["HOME"] and not missing["AWAY"]:
        return {
            "status": "NO_CONFIRMED_ABSENCES",
            "players": [],
            "used_for": "EXPLANATION_ONLY",
            "affects_prediction": False,
        }

    cache_key = (
        str(home), str(away), str(match_date)[:10],
        tuple(missing["HOME"]), tuple(missing["AWAY"])
    )
    with _GOAL_PLAYER_EXPLANATION_CACHE_LOCK:
        cached = _GOAL_PLAYER_EXPLANATION_CACHE.get(cache_key)
        if cached and (time.time() - cached[0] < 1800):
            return dict(cached[1], cache_hit=True)

    started = time.perf_counter()

    # One independent query per confirmed absentee.
    player_jobs = []
    max_players = int(os.getenv("BETSMART_GOAL_PLAYER_EXPLANATION_MAX_PLAYERS", "6"))
    max_players = max(1, min(max_players, 8))

    for side, team in (("HOME", home), ("AWAY", away)):
        for rt_name in missing.get(side, []):
            player_jobs.append({
                "side": side,
                "team": team,
                "rt_name": rt_name,
            })

    player_jobs = player_jobs[:max_players]

    errors = []
    retrievals = []

    def _query_for(job):
        # Keep the RT abbreviation, team and football vocabulary together.
        # Do NOT quote several player names in the same query.
        q = (
            f'"{job["rt_name"]}" "{job["team"]}" football '
            f'position striker forward winger defender goalkeeper '
            f'top scorer key player goals appearances starter'
        )
        result = _goal_brave_search(q, 3)
        return {
            **job,
            "query": q,
            "search": result,
        }

    workers = max(1, min(len(player_jobs), 6))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [(job, pool.submit(_query_for, job)) for job in player_jobs]
        for job, fut in futures:
            try:
                retrievals.append(fut.result())
            except Exception as exc:
                errors.append(
                    f'{job["side"]}:{job["rt_name"]}:'
                    f'{type(exc).__name__}:{str(exc)[:250]}'
                )

    evidence = []
    for group in retrievals:
        search = group.get("search") or {}
        for item in (search.get("results") or []):
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            title = str(item.get("title") or "").strip()
            snippet = str(item.get("description") or item.get("snippet") or "").strip()
            if not (url or title or snippet):
                continue
            evidence.append({
                "rt_name": group.get("rt_name"),
                "side": group.get("side"),
                "team": group.get("team"),
                "query": group.get("query"),
                "title": title[:220],
                "snippet": snippet[:900],
                "tier": item.get("source_tier"),
                "url": url,
            })

    # Keep at most two results per player in the LLM payload.
    compact_evidence = []
    per_player_count = {}
    for item in evidence:
        key = (item.get("side"), item.get("rt_name"))
        if per_player_count.get(key, 0) >= 2:
            continue
        per_player_count[key] = per_player_count.get(key, 0) + 1
        compact_evidence.append(item)

    if not compact_evidence:
        out = {
            "status": "NO_PLAYER_EVIDENCE",
            "players": [],
            "used_for": "EXPLANATION_ONLY",
            "affects_prediction": False,
            "query_mode": "ONE_QUERY_PER_CONFIRMED_ABSENT",
            "query_count": len(player_jobs),
            "queries": [
                {
                    "side": r.get("side"),
                    "team": r.get("team"),
                    "rt_name": r.get("rt_name"),
                    "query": r.get("query"),
                    "result_count": (r.get("search") or {}).get("result_count", 0),
                }
                for r in retrievals
            ],
            "errors": errors,
            "seconds": round(time.perf_counter() - started, 3),
        }
        with _GOAL_PLAYER_EXPLANATION_CACHE_LOCK:
            _GOAL_PLAYER_EXPLANATION_CACHE[cache_key] = (time.time(), out)
        return out

    # V1.0.8.8.12: deterministic classification from evidence already retrieved.
    # No extra OpenAI call and no second-stage Brave search.
    players = _goal_deterministic_players_from_evidence(compact_evidence, missing)
    parse_mode = "PYTHON_DETERMINISTIC_LOCAL_ATTRIBUTION"

    out = {
        "status": "OK" if players else "NO_QUALIFIED_PLAYER",
        "players": players,
        "used_for": "EXPLANATION_ONLY",
        "affects_prediction": False,
        "query_mode": "ONE_QUERY_PER_CONFIRMED_ABSENT",
        "query_count": len(player_jobs),
        "queries": [
            {
                "side": r.get("side"),
                "team": r.get("team"),
                "rt_name": r.get("rt_name"),
                "query": r.get("query"),
                "result_count": (r.get("search") or {}).get("result_count", 0),
                "elapsed_seconds": (r.get("search") or {}).get("elapsed_seconds"),
            }
            for r in retrievals
        ],
        "parse_mode": parse_mode,
        "retrieved_evidence": compact_evidence,
        "errors": errors,
        "seconds": round(time.perf_counter() - started, 3),
    }

    with _GOAL_PLAYER_EXPLANATION_CACHE_LOCK:
        _GOAL_PLAYER_EXPLANATION_CACHE[cache_key] = (time.time(), out)

    return out




def _goal_normalize_player_name(value):
    import unicodedata
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text).split())


def _goal_player_surname(value):
    norm = _goal_normalize_player_name(value)
    return norm.split()[-1] if norm else ""


def _goal_validated_major_absences(home, away, rt_ctx, player_ctx):
    """
    Cross-check Web qualification against the Real-Time Missing Fixture list.
    No confirmed RT absence => no player can be called absent in explanation.
    """
    missing = _goal_missing_fixture_names(rt_ctx)
    validated = []

    for p in (player_ctx or {}).get("players") or []:
        if not isinstance(p, dict):
            continue
        confidence = _goal_safe_float(p.get("confidence"), 0.0)
        if confidence < 0.55:
            continue

        side = str(p.get("side") or "").upper()
        if side not in {"HOME","AWAY"}:
            continue

        emitted_rt_name = str(p.get("rt_name") or "").strip()
        rt_name = emitted_rt_name if emitted_rt_name in missing.get(side, []) else None

        if not rt_name:
            surname = _goal_player_surname(p.get("name"))
            if not surname:
                continue
            rt_name = next(
                (n for n in missing.get(side, []) if _goal_player_surname(n) == surname),
                None
            )

        if not rt_name:
            continue

        role = str(p.get("role") or "UNKNOWN").upper()
        importance = str(p.get("importance") or "UNKNOWN").upper()

        # To explain offensive/defensive impact, role must be known.
        if role not in {"ATTACKER","DEFENDER","GOALKEEPER"}:
            continue

        validated.append({
            **p,
            "rt_name": rt_name,
            "team_name": home if side == "HOME" else away,
            "role": role,
            "importance": importance,
        })

    # KEY first; then offensive/defensive relevance.
    role_rank = {"ATTACKER": 0, "DEFENDER": 1, "GOALKEEPER": 2}
    validated.sort(key=lambda p: (
        0 if p.get("importance") == "KEY" else 1,
        role_rank.get(p.get("role"), 9),
        -_goal_safe_float(p.get("confidence"), 0.0)
    ))
    return validated


def _goal_final_state_explanation(
    home, away, lam_h_base, lam_a_base, lam_h_final, lam_a_final,
    final_probs, ai, web_ctx, hist_ctx=None, rt_ctx=None, player_ctx=None
):
    """
    Explication utilisateur courte et football:
    1) forme offensive/défensive récente,
    2) absence de joueurs majeurs qualifiée offensivement/défensivement,
    3) conséquence sur les marchés de buts.

    player_ctx est EXPLANATION_ONLY et n'influence jamais la prédiction.
    """
    hist_ctx, rt_ctx, player_ctx = hist_ctx or {}, rt_ctx or {}, player_ctx or {}
    form = hist_ctx.get("current_form") or {}
    hf, af = form.get("home") or {}, form.get("away") or {}

    def _n(v, default=0.0):
        try:
            return float(v)
        except Exception:
            return float(default)

    # Phrase 1: exactly the user-facing style requested.
    hm, am = int(hf.get("matches", 0) or 0), int(af.get("matches", 0) or 0)
    # V1.0.8.8.14: do not present a 1-2 match sample as a meaningful recent-form trend.
    # This is presentation-only: prediction inputs and weights are untouched.
    min_form_matches = 3
    if hm >= min_form_matches and am >= min_form_matches:
        sample = min(hm, am)
        home_gf = _n(hf.get("gf_avg"))
        home_ga = _n(hf.get("ga_avg"))
        away_gf = _n(af.get("gf_avg"))
        away_ga = _n(af.get("ga_avg"))
        form_sentence = (
            f"Sur leurs {sample} derniers matchs disponibles, {home} marque {home_gf:.1f} but et en encaisse "
            f"{home_ga:.1f} en moyenne, contre {away_gf:.1f} marqués et {away_ga:.1f} encaissés "
            f"pour {away}."
        )
    else:
        venue = hist_ctx.get("venue_history") or {}
        home_venue = venue.get("home_at_home") or {}
        away_venue = venue.get("away_at_away") or {}
        hvm = int(home_venue.get("matches", 0) or 0)
        avm = int(away_venue.get("matches", 0) or 0)

        # Venue fallback also needs a minimally meaningful sample.
        if hvm >= min_form_matches and avm >= min_form_matches:
            hgf = _n(home_venue.get("gf_avg"))
            hga = _n(home_venue.get("ga_avg"))
            agf = _n(away_venue.get("gf_avg"))
            aga = _n(away_venue.get("ga_avg"))
            form_sentence = (
                f"À domicile, {home} marque {hgf:.1f} but et en encaisse {hga:.1f} en moyenne; "
                f"à l'extérieur, {away} en marque {agf:.1f} et en encaisse {aga:.1f}."
            )
        else:
            # Reuse already-retrieved predictive Web facts without any new network call.
            claims = [str(f.get("claim") or "").strip() for f in (web_ctx.get("facts") or []) if isinstance(f, dict)]
            recent_claims = [c for c in claims if c and re.search(r"last\s+(?:five|5)|across\s+(?:five|5)", c, re.I)]
            def _web_form_numbers(team_name):
                for claim in recent_claims:
                    if team_name.lower() not in claim.lower():
                        continue
                    # Only use explicit per-match GF/GA figures; never expose raw English snippets.
                    m = re.search(
                        r"([0-9]+(?:\.[0-9]+)?)\s+goals?\s+per\s+match.*?"
                        r"([0-9]+(?:\.[0-9]+)?)\s+goals?\s+conceded\s+per\s+match",
                        claim, re.I
                    )
                    if m:
                        return float(m.group(1)), float(m.group(2))
                return None

            hw = _web_form_numbers(home)
            aw = _web_form_numbers(away)
            if hw and aw:
                form_sentence = (
                    f"Sur la forme récente documentée, {home} marque {hw[0]:.1f} but et en encaisse {hw[1]:.1f} "
                    f"en moyenne, contre {aw[0]:.1f} marqués et {aw[1]:.1f} encaissés pour {away}."
                )
            elif hw:
                form_sentence = (
                    f"Les données récentes documentées indiquent que {home} marque {hw[0]:.1f} but et en encaisse "
                    f"{hw[1]:.1f} en moyenne; l'échantillon reste toutefois insuffisant pour comparer solidement les deux équipes."
                )
            elif aw:
                form_sentence = (
                    f"Les données récentes documentées indiquent que {away} marque {aw[0]:.1f} but et en encaisse "
                    f"{aw[1]:.1f} en moyenne; l'échantillon reste toutefois insuffisant pour comparer solidement les deux équipes."
                )
            else:
                form_sentence = (
                    "L'échantillon récent disponible est encore trop limité pour comparer solidement "
                    "l'efficacité offensive et défensive des deux équipes."
                )

    # Phrase 2: explain ONLY major offensive/defensive absences that are
    # both RT-confirmed and Web-qualified.
    validated = _goal_validated_major_absences(home, away, rt_ctx, player_ctx)
    attackers = [p for p in validated if p["role"] == "ATTACKER"]
    defenders = [p for p in validated if p["role"] in {"DEFENDER","GOALKEEPER"}]

    def _short_name(p):
        name = str(p.get("name") or p.get("rt_name") or "").strip()
        surname = name.split()[-1] if name else ""
        return surname or name

    if attackers and defenders:
        a, d = attackers[0], defenders[0]
        a_desc = (
            "l'un de ses principaux joueurs offensifs"
            if a.get("importance") == "KEY"
            else "un joueur offensif"
        )
        d_desc = (
            "un élément important de sa défense"
            if d.get("importance") == "KEY"
            else "un défenseur de son effectif"
        )
        if a["team_name"] == d["team_name"]:
            team = a["team_name"]
            absence_sentence = (
                f"{team} devra toutefois composer sans {_short_name(a)}, {a_desc}, ainsi que "
                f"sans {_short_name(d)}, {d_desc}, deux absences qui peuvent peser respectivement "
                f"sur son efficacité devant le but et sa solidité défensive."
            )
        else:
            absence_sentence = (
                f"{a['team_name']} devra toutefois composer sans {_short_name(a)}, {a_desc}, "
                f"ce qui peut peser sur son efficacité devant le but; {d['team_name']} sera aussi "
                f"privé de {_short_name(d)}, {d_desc}, avec un risque pour sa solidité défensive."
            )
    elif attackers:
        a = attackers[0]
        a_desc = (
            "l'un de ses principaux joueurs offensifs"
            if a.get("importance") == "KEY"
            else "un joueur offensif"
        )
        absence_sentence = (
            f"{a['team_name']} devra toutefois composer sans {_short_name(a)}, {a_desc}, "
            f"une absence susceptible de peser sur son efficacité devant le but."
        )
    elif defenders:
        d = defenders[0]
        d_desc = (
            "un élément important de sa défense"
            if d.get("importance") == "KEY"
            else "un défenseur de son effectif"
        )
        absence_sentence = (
            f"{d['team_name']} devra toutefois composer sans {_short_name(d)}, {d_desc}, "
            f"une absence susceptible de peser sur sa solidité défensive."
        )
    else:
        missing = _goal_missing_fixture_names(rt_ctx)
        if missing["HOME"] or missing["AWAY"]:
            absence_sentence = (
                "Des absences sont confirmées, mais les sources disponibles ne permettent pas "
                "de qualifier suffisamment leur poids offensif ou défensif."
            )
        else:
            absence_sentence = (
                "Aucune absence majeure suffisamment documentée n'est retenue comme facteur "
                "offensif ou défensif déterminant."
            )

    # Phrase 3: concise market consequence, no lambda/model jargon.
    p15 = _n(final_probs.get("Over15"))
    p25 = _n(final_probs.get("Over25"))
    pb = _n(final_probs.get("BTTS"))

    if p15 >= max(p25, pb):
        if p25 < 0.50 and pb < 0.50:
            market_sentence = (
                f"Dans ce contexte, Over 1.5 reste le scénario le plus solide à {p15*100:.0f} %, "
                f"tandis qu'Over 2.5 ({p25*100:.0f} %) et BTTS ({pb*100:.0f} %) restent plus incertains."
            )
        else:
            market_sentence = (
                f"Dans ce contexte, Over 1.5 reste le scénario le plus solide à {p15*100:.0f} %, "
                f"avec Over 2.5 à {p25*100:.0f} % et BTTS à {pb*100:.0f} %."
            )
    elif p25 >= pb:
        market_sentence = (
            f"Dans ce contexte, Over 2.5 ressort à {p25*100:.0f} %, contre {pb*100:.0f} % pour BTTS."
        )
    else:
        market_sentence = (
            f"Dans ce contexte, BTTS ressort à {pb*100:.0f} %, tandis qu'Over 2.5 se situe à {p25*100:.0f} %."
        )

    return f"{form_sentence} {absence_sentence} {market_sentence}"[:780]


def _goal_fallback_explanation(home, away, final_probs, lam_total, confidence):
    ordered = sorted(
        [(m, p) for m, p in final_probs.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    main = ordered[0]
    return (
        f"BetSmart projette un match {home} – {away} autour de {lam_total:.2f} buts attendus. "
        f"Le signal le plus fort est {main[0]} avec {main[1]*100:.1f}% de probabilité. "
        f"Les probabilités finales restent issues du modèle de buts, de l'historique disponible "
        f"et des ajustements contextuels contrôlés. "
        f"Confiance globale : {confidence:.2f}."
    )


def _goal_full_view(result):
    return _goal_jsonable(result)


_GOAL_REDUCED_KEYS = [
    "home","away","lambda_home","lambda_away","lambda_total",
    "Over15","Over25","Over35","BTTS",
    "expected_goal_range","most_likely_score",
    "prediction_confidence","low_confidence",
    "explanation","rule_applied","_final_state_version"
]


def _goal_reduced_view(result):
    return _goal_jsonable({k: result.get(k) for k in _GOAL_REDUCED_KEYS})


def build_goal_output_views_v1(result):
    return {
        "version": GOAL_INTELLIGENCE_VERSION,
        "full": _goal_full_view(result),
        "reduced": _goal_reduced_view(result),
    }


def _goal_guaranteed_final_state(result):
    out = dict(result or {})

    # Enforce probabilities and predictions.
    for market in ["Over15", "Over25", "Over35", "BTTS"]:
        m = dict(out.get(market) or {})
        p = _goal_normalize_prob(m.get("proba", 0.5))
        m["proba"] = p
        m["pred"] = int(p >= 0.50)
        out[market] = m

    # Monotonic Over probabilities one last time.
    p15 = out["Over15"]["proba"]
    p25 = out["Over25"]["proba"]
    p35 = out["Over35"]["proba"]
    p15 = max(p15, p25)
    p25 = min(p15, max(p25, p35))
    p35 = min(p35, p25)
    for market, p in [("Over15",p15),("Over25",p25),("Over35",p35)]:
        out[market]["proba"] = float(p)
        out[market]["pred"] = int(p >= 0.50)

    ra = str(out.get("rule_applied") or "")
    tag = "goal_v1_guaranteed_final_state"
    if tag not in ra:
        out["rule_applied"] = f"{ra}|{tag}" if ra else tag

    out["_final_state_guaranteed"] = True
    out["_final_state_version"] = "GOAL-1.0.8.8.14-EXPLANATION-SOURCE-STABILITY"
    return out


def predict_from_user_input(
    df_hist,
    home,
    away,
    date,
    odds,
    out_dir="betsmart_goals_out_pl",
    use_llm: bool = False,
    llm_client=None,
    explainer=None,
    config=None,
    lambda_home_model=None,
    lambda_away_model=None,
    o25_cal=None,
    btts_ml=None,
    btts_cal=None,
    output_mode="full",
):
    """
    BetSmart Goal Intelligence V1.0.8.
    Existing trained models are preserved; intelligence is added around them.
    """
    _goal_t0 = time.time()
    _goal_timing = {}

    cfg = config
    if cfg is None:
        raise ValueError("config manquant")

    _t = time.time()
    match_df = prepare_user_input_and_enrich(df_hist, home, away, date, odds)
    _goal_timing['feature_engineering_seconds'] = round(time.time() - _t, 3)

    for c in ["OU_O15","OU_O25","OU_O35","BTTS_Yes"]:
        feature_lists = (
            list(cfg.get("lambda_features", []))
            + list(cfg.get("o25_features", []))
            + list(cfg.get("btts_features", []))
        )
        if c in feature_lists and c not in match_df.columns:
            match_df[c] = np.nan

    cfg_runtime = cfg.copy()
    cfg_runtime["_o25_cal_model"] = o25_cal
    cfg_runtime["_btts_cal_model"] = btts_cal

    # V1.0.8.8.4: start slow external I/O immediately while local ML/history execute.
    # This overlaps Real-Time + Brave/OpenAI Web with deterministic local work.
    _parallel_stage_t0 = time.perf_counter()
    _external_pool = ThreadPoolExecutor(max_workers=1)
    _external_future = _external_pool.submit(
        _goal_build_external_context_parallel, home, away, date
    )

    try:
        # BASE quantitative engine only. No old explanation LLM here.
        _t = time.time()
        base_res = predict_goal_with_proba(
            match_df=match_df,
            lambda_home_model=lambda_home_model,
            lambda_away_model=lambda_away_model,
            btts_ml=btts_ml,
            btts_cal=btts_cal,
            config=cfg_runtime,
            explainer=rule_based_explainer,
            use_llm=False,
            llm_client=None,
        )
        _goal_timing["base_model_seconds"] = round(time.time() - _t, 3)
        lam_h_base = _goal_safe_float(base_res.get("lambda_home"), 1.2)
        lam_a_base = _goal_safe_float(base_res.get("lambda_away"), 1.0)

        _t = time.time()
        hist_ctx = build_goal_historical_context(df_hist, home, away, date)
        market_ctx = build_goal_market_context(odds)
        market_disagreement = _goal_market_disagreement(base_res, market_ctx)
        _goal_timing['history_market_seconds'] = round(time.time() - _t, 3)

        _t = time.time()
        rt_ctx, web_ctx = _external_future.result()
        _goal_timing['external_context_wait_seconds'] = round(time.time() - _t, 3)
    finally:
        _goal_shutdown_executor_now(_external_pool)

    _goal_timing['parallel_pre_arbitration_seconds'] = round(
        time.perf_counter() - _parallel_stage_t0, 3
    )
    _goal_timing['external_context_seconds'] = round(
        max(
            _goal_safe_float((rt_ctx or {}).get('_parallel_context_seconds'), 0.0),
            _goal_safe_float((web_ctx or {}).get('_parallel_context_seconds'), 0.0),
        ),
        3,
    )

    web_strategic_leverage = _goal_web_strategic_leverage(hist_ctx, web_ctx)

    arbitration_payload = {
        "version": "GOAL-1.0.8.8.14-EXPLANATION-SOURCE-STABILITY",
        "teams": {"home": home, "away": away},
        "match_date": date,
        "base_goal_model": {
            "lambda_home": lam_h_base,
            "lambda_away": lam_a_base,
            "lambda_total": lam_h_base + lam_a_base,
            "markets": {
                m: dict(base_res.get(m) or {})
                for m in ["Over15","Over25","Over35","BTTS"]
            },
        },
        "historical_context": hist_ctx,
        "market_context": market_ctx,
        "model_market_disagreement": market_disagreement,
        "realtime_context": rt_ctx,
        "web_intelligence": web_ctx,
        "web_strategic_leverage": web_strategic_leverage,
        "arbitration_policy": "ALWAYS_CALL_AI_ZERO_DELTA_ALLOWED",
    }

    # EXPLANATION-ONLY player qualification starts after Real-Time is known.
    # It runs in parallel with final arbitration and is NEVER included in arbitration_payload.
    _player_pool = ThreadPoolExecutor(max_workers=1)
    _player_future = _player_pool.submit(
        _goal_research_player_explanation_context, home, away, date, rt_ctx
    )

    _t = time.time()
    skip_ai, skip_diag = _goal_should_skip_ai(hist_ctx, web_ctx, rt_ctx)
    if skip_ai:
        ai = {
            "status": "SKIPPED",
            "lambda_home_delta": 0.0,
            "lambda_away_delta": 0.0,
            "market_adjustments": {"BTTS":0.0,"Over15":0.0,"Over25":0.0,"Over35":0.0},
            "prediction_confidence": 0.0,
            "source_agreement": "LOW",
            "risk_level": "MEDIUM",
            "reason_codes": ["SMART_AI_SKIP", "NO_ACTIONABLE_EXTERNAL_EVIDENCE"],
            "rationale_short": "Arbitrage LLM évité: aucune preuve externe exploitable suffisante pour justifier une modification.",
            "explanation": "",
            "skip_diagnostics": skip_diag,
            "arbitration_diagnostics": {
                "called": False, "completed": False, "reason": "SMART_AI_SKIP",
                "model": GOAL_AI_MODEL,
            },
        }
    else:
        ai = goal_ai_arbitrator(arbitration_payload)
    _goal_timing['ai_arbitration_seconds'] = round(time.time() - _t, 3)

    # Collect explanation-only context after arbitration so its latency overlaps.
    _player_t = time.perf_counter()
    try:
        player_explanation_ctx = _player_future.result(
            timeout=max(0.1, GOAL_PLAYER_EXPLANATION_TIMEOUT_SECONDS)
        )
    except Exception as exc:
        player_explanation_ctx = {
            "status": f"UNAVAILABLE:{type(exc).__name__}",
            "players": [],
            "used_for": "EXPLANATION_ONLY",
            "affects_prediction": False,
            "error": str(exc)[:300],
        }
    finally:
        _goal_shutdown_executor_now(_player_pool)
    _goal_timing["player_explanation_wait_seconds"] = round(
        time.perf_counter() - _player_t, 3
    )
    _goal_timing["player_explanation_seconds"] = round(
        _goal_safe_float((player_explanation_ctx or {}).get("seconds"), 0.0), 3
    )

    dh, da, stability = _goal_stabilize_deltas(hist_ctx, web_ctx, rt_ctx, ai)

    lam_h_final = float(np.clip(lam_h_base + dh, 0.05, 4.5))
    lam_a_final = float(np.clip(lam_a_base + da, 0.05, 4.5))
    lam_t_final = lam_h_final + lam_a_final

    lambda_recalc_probs = _goal_final_probs(
        base_res, lam_h_base, lam_a_base, lam_h_final, lam_a_final
    )
    evidence_strength = _goal_evidence_strength(hist_ctx, web_ctx, rt_ctx, ai)
    final_probs, final_market_adjustment = _goal_apply_ai_market_adjustments(
        lambda_recalc_probs, ai, web_ctx, evidence_strength
    )

    quantitative_confidence = _goal_quantitative_confidence(
        base_res, hist_ctx, market_ctx
    )

    ai_status = str(ai.get("status") or "ERROR").upper()
    if ai_status == "OK":
        ai_confidence = float(np.clip(
            _goal_safe_float(ai.get("prediction_confidence"), 0.0),
            0.0, 1.0
        ))
        confidence = 0.55 * quantitative_confidence + 0.45 * ai_confidence
    elif ai_status == "SKIPPED":
        confidence = quantitative_confidence * 0.92
    else:
        confidence = quantitative_confidence * 0.88

    form_rel = _goal_safe_float(hist_ctx.get("current_form_reliability"), 0.0)
    if form_rel < 0.40:
        confidence = min(confidence, 0.60)

    if ai_status == "OK" and str(ai.get("source_agreement") or "LOW").upper() == "LOW":
        confidence = min(confidence, 0.56)

    md_level = str(market_disagreement.get("level") or "UNKNOWN").upper()
    if md_level == "VERY_HIGH":
        confidence *= 0.78
    elif md_level == "HIGH":
        confidence *= 0.86
    elif md_level == "MEDIUM":
        confidence *= 0.93

    confidence = float(np.clip(confidence, 0.0, 1.0))

    history_ok = (
        int((hist_ctx.get("venue_history") or {}).get("home_at_home", {}).get("matches", 0) or 0) >= 5
        and int((hist_ctx.get("venue_history") or {}).get("away_at_away", {}).get("matches", 0) or 0) >= 5
    )

    markets = {}
    for market, p in final_probs.items():
        markets[market] = _goal_market_pack(
            p, confidence, history_ok=history_ok
        )

    # If very strong probability, do not mark low confidence only because it is outside gray zone.
    for market, pack in markets.items():
        if pack["proba"] >= 0.67 and confidence >= 0.50 and history_ok:
            pack["low_confidence"] = False

    explanation = _goal_final_state_explanation(
        home, away, lam_h_base, lam_a_base, lam_h_final, lam_a_final,
        final_probs, ai, web_ctx, hist_ctx=hist_ctx, rt_ctx=rt_ctx,
        player_ctx=player_explanation_ctx
    )

    likely_total = int(round(lam_t_final))
    expected_range = (
        "0-1" if lam_t_final < 1.75 else
        "2-3" if lam_t_final < 3.55 else
        "4+"
    )

    result = {
        "version": "GOAL-1.0.8.8.14-EXPLANATION-SOURCE-STABILITY",
        "home": str(home),
        "away": str(away),

        "model_raw": {
            "lambda_home": round(lam_h_base, 4),
            "lambda_away": round(lam_a_base, 4),
            "lambda_total": round(lam_h_base + lam_a_base, 4),
            "markets": {
                m: dict(base_res.get(m) or {})
                for m in ["Over15","Over25","Over35","BTTS"]
            },
        },

        "lambda_home": round(lam_h_final, 4),
        "lambda_away": round(lam_a_final, 4),
        "lambda_total": round(lam_t_final, 4),

        "Over15": markets["Over15"],
        "Over25": markets["Over25"],
        "Over35": markets["Over35"],
        "BTTS": markets["BTTS"],

        "goal_distribution": _goal_distribution(lam_t_final),
        "expected_goal_range": expected_range,
        "expected_total_goals_rounded": likely_total,
        "most_likely_score": _goal_most_likely_score(lam_h_final, lam_a_final),

        "historical_context": hist_ctx,
        "market_context": market_ctx,
        "model_market_disagreement": market_disagreement,
        "final_market_state": {
            "lambda_recalculated_probabilities": lambda_recalc_probs,
            "ai_market_adjustment": final_market_adjustment,
            "probability_policy": "POISSON_FROM_FINAL_LAMBDAS_ONLY",
            "final_probabilities": final_probs,
        },
        "realtime_intelligence_context": rt_ctx,
        "realtime_web_intelligence": web_ctx,
        "player_explanation_context": player_explanation_ctx,
        "web_strategic_leverage": web_strategic_leverage,

        "ai_decision": {
            "status": ai.get("status"),
            "decision_origin": (
                "OPENAI_LLM" if ai_status == "OK"
                else "SMART_AI_SKIP" if ai_status == "SKIPPED"
                else "QUANTITATIVE_STABLE_FALLBACK"
            ),
            "prediction_confidence": round(confidence, 3),
            "quantitative_confidence": round(quantitative_confidence, 3),
            "ai_reported_confidence": round(_goal_safe_float(ai.get("prediction_confidence"), 0.0), 3),
            "source_agreement": str(ai.get("source_agreement") or "LOW"),
            "risk_level": str(ai.get("risk_level") or "UNKNOWN"),
            "reason_codes": list(ai.get("reason_codes") or [])[:12],
            "error": str(ai.get("error") or "")[:900] if ai.get("error") else None,
            "rationale_short": str(ai.get("rationale_short") or "")[:1500],
            "base_lambdas": {
                "home": round(lam_h_base, 4),
                "away": round(lam_a_base, 4),
                "total": round(lam_h_base + lam_a_base, 4),
            },
            "final_lambdas": {
                "home": round(lam_h_final, 4),
                "away": round(lam_a_final, 4),
                "total": round(lam_t_final, 4),
            },
            "stability": stability,
            "market_adjustments": {"BTTS":0.0,"Over15":0.0,"Over25":0.0,"Over35":0.0},
            "arbitration_mode": "LAMBDA_ONLY",
            "final_market_adjustment": final_market_adjustment,
            "skip_diagnostics": ai.get("skip_diagnostics"),
            "arbitration_diagnostics": ai.get("arbitration_diagnostics"),
            "public_state_synchronized": True,
        },

        "prediction_confidence": round(confidence, 3),
        "low_confidence": bool(confidence < 0.60),
        "explanation": explanation,
        "rule_applied": "goal_base_ml|poisson_final_lambdas_only|goal_web_retrieval_brave_partial_v10883|goal_ai_analysis_evidence_validation_gpt54mini_timeout12_v10883|force_ai_arbitration|lambda_only_arbitration_v108814|parallel_timeout_fix|goal_stability_layer|pure_poisson_from_final_lambdas_v108814|player_absence_individual_search_explanation_only_v108814|deterministic_local_role_attribution_v108814|explanation_source_stability_v108814",
    }

    accounted = sum(v for v in _goal_timing.values() if isinstance(v, (int, float)))
    _goal_timing["other_finalization_seconds"] = round(
        max(0.0, (time.time() - _goal_t0) - accounted), 3
    )
    result["runtime"] = {
        **_goal_timing,
        "total_seconds": round(time.time() - _goal_t0, 3),
        "context_parallel": True,
        "context_workers": GOAL_CONTEXT_WORKERS,
        "pre_arbitration_parallel": True,
        "realtime_details_parallel": True,
    }

    result = _goal_guaranteed_final_state(result)

    if str(output_mode).lower() == "reduced":
        return _goal_reduced_view(result)
    return _goal_full_view(result)

def get_valid_date(user_input):
    """
    Convertit différentes représentations de date en format 'YYYY-MM-DD'.
    """
    try:
        # Parse intelligent (fonctionne avec des formats très variés)
        date_obj = parser.parse(user_input)
        return date_obj.strftime("%Y-%m-%d")
    except Exception:
        raise ValueError("⛔ Format de date non reconnu. Essayez par exemple : '2025-02-14' ou '14/02/2025'")
