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
# BETSMART GOAL INTELLIGENCE V1.0.6
# Multi-source Goal Engine:
# ML lambdas -> Poisson/calibrated markets -> history/form/H2H -> realtime/web -> AI
# -> Stability Layer -> Guaranteed Final State.
# ======================================================================================

GOAL_INTELLIGENCE_VERSION = "1.0.6"
GOAL_AI_MODEL = os.getenv("BETSMART_GOAL_AI_MODEL", "gpt-5-mini")
GOAL_WEB_MODEL = os.getenv("BETSMART_GOAL_WEB_MODEL", GOAL_AI_MODEL)
GOAL_AI_ENABLED = os.getenv("BETSMART_GOAL_AI_ENABLED", "1").strip().lower() not in {"0","false","off","no"}
GOAL_WEB_ENABLED = os.getenv("BETSMART_GOAL_WEB_ENABLED", "1").strip().lower() not in {"0","false","off","no"}
GOAL_REALTIME_ENABLED = os.getenv("BETSMART_GOAL_REALTIME_ENABLED", "1").strip().lower() not in {"0","false","off","no"}

GOAL_CONTEXT_WORKERS = int(os.getenv("BETSMART_GOAL_CONTEXT_WORKERS", "2"))
GOAL_WEB_TIMEOUT_SECONDS = int(os.getenv("BETSMART_GOAL_WEB_TIMEOUT_SECONDS", "18"))
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
    Optional API-Football context. It never blocks prediction.
    """
    empty = {
        "available": False,
        "fixture_id": None,
        "injuries_home": [],
        "injuries_away": [],
        "lineups_available": False,
        "status": "UNAVAILABLE",
    }
    if not GOAL_REALTIME_ENABLED:
        return {**empty, "status": "DISABLED"}

    headers = _goal_realtime_headers()
    if not headers:
        return {**empty, "status": "NO_API_KEY"}

    base_url = os.getenv("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io").rstrip("/")
    try:
        r = requests.get(
            f"{base_url}/fixtures",
            headers=headers,
            params={"date": str(match_date)[:10]},
            timeout=8,
        )
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
            return {**empty, "status": "FIXTURE_NOT_FOUND"}

        fixture_id = int(((chosen.get("fixture") or {}).get("id")))
        out = {**empty, "available": True, "fixture_id": fixture_id, "status": "OK"}

        try:
            ri = requests.get(
                f"{base_url}/injuries",
                headers=headers,
                params={"fixture": fixture_id},
                timeout=6,
            )
            injuries = ri.json().get("response", []) if ri.ok else []
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
        except Exception:
            pass

        try:
            rl = requests.get(
                f"{base_url}/fixtures/lineups",
                headers=headers,
                params={"fixture": fixture_id},
                timeout=6,
            )
            lineups = rl.json().get("response", []) if rl.ok else []
            out["lineups_available"] = bool(lineups)
        except Exception:
            pass

        return out
    except Exception as e:
        return {**empty, "status": f"ERROR:{type(e).__name__}"}



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
            "source_agreement": {"type": "string"},
            "risk_level": {"type": "string"},
            "reason_codes": {"type": "array", "items": {"type": "string"}},
            "rationale_short": {"type": "string"},
            "explanation": {"type": "string"},
        },
        "required": [
            "status","lambda_home_delta","lambda_away_delta",
            "prediction_confidence","source_agreement","risk_level",
            "reason_codes","rationale_short","explanation",
        ],
    }

    errors = []

    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
            max_output_tokens=1200,
        )
        obj = _goal_extract_json(_goal_response_text(resp))
        if isinstance(obj, dict):
            return obj
        errors.append("RESPONSES_NON_JSON")
    except Exception as exc:
        errors.append(
            f"RESPONSES:{type(exc).__name__}:{str(exc)[:500]}"
        )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Tu réponds STRICTEMENT en JSON valide."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        obj = _goal_extract_json(_goal_response_text(resp))
        if isinstance(obj, dict):
            return obj
        errors.append("CHAT_NON_JSON")
    except Exception as exc:
        errors.append(
            f"CHAT:{type(exc).__name__}:{str(exc)[:500]}"
        )

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




def _goal_structured_web_call(client, model, prompt):
    """
    V1.0.5:
    one web_search call -> robust text/source extraction -> JSON parse.
    """
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            tools=[{"type": "web_search"}],
            max_output_tokens=1200,
        )
    except Exception as exc:
        raise ValueError(
            f"WEB_SEARCH_CALL_FAILED:{type(exc).__name__}:{str(exc)[:700]}"
        )

    sources = _goal_response_sources(resp)
    source_count = len(sources)
    raw_text = _goal_response_text(resp)

    obj = _goal_extract_json(raw_text)
    if isinstance(obj, dict):
        return obj, source_count, "DIRECT_JSON", sources

    if raw_text and len(raw_text.strip()) >= 20:
        obj = _goal_web_reformat_to_json(client, model, raw_text)
        if isinstance(obj, dict):
            return obj, source_count, "REFORMATTED_JSON", sources

    raise ValueError(
        f"WEB_RESPONSE_UNUSABLE:text_len={len(raw_text or '')}:sources={source_count}"
    )



def research_goal_web_context(home, away, match_date):
    empty = {
        "status": "UNAVAILABLE",
        "web_research_used": False,
        "web_evidence_verified": False,
        "web_evidence_tier": "NONE",
        "source_count": 0,
        "sources": [],
        "signals": [],
        "summary": "",
        "data_confidence": 0.0,
        "actionable_signal_count": 0,
        "parse_mode": "NONE",
    }

    if not GOAL_WEB_ENABLED:
        return {**empty, "status": "DISABLED"}

    key = f"{home}|{away}|{str(match_date)[:10]}"
    with _GOAL_WEB_CACHE_LOCK:
        cached = _GOAL_WEB_CACHE.get(key)
        if cached and (time.time() - cached[0] < 1800):
            return dict(cached[1], cache_hit=True)

    try:
        client = get_openai_client()

        prompt = f"""
Tu es BETSMART GOAL WEB INTELLIGENCE.

Recherche les informations ACTUELLES et utiles au NOMBRE DE BUTS pour
{home} vs {away}, match prévu le {match_date}.

Cherche uniquement:
- attaquants/créateurs absents ou de retour;
- gardien/défenseurs centraux absents ou de retour;
- compositions probables/officielles;
- forme offensive et défensive récente;
- changement tactique ou d'entraîneur;
- fatigue / calendrier rapproché;
- météo seulement si réellement significative.

RÈGLES:
- ignore les informations anciennes non pertinentes pour ce match;
- une équipe favorite ne signifie pas automatiquement plus de buts;
- indique MORE_GOALS ou FEWER_GOALS seulement si le fait soutient réellement ce sens;
- si incertain, baisse confidence;
- n'utilise aucun fait non vérifié.

Retourne idéalement STRICTEMENT ce JSON:
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
""".strip()

        obj, source_count, parse_mode, sources = _goal_structured_web_call(
            client, GOAL_WEB_MODEL, prompt
        )

        signals = obj.get("signals") if isinstance(obj.get("signals"), list) else []

        cleaned = []
        for sig in signals[:12]:
            if not isinstance(sig, dict):
                continue

            direction = str(sig.get("direction") or "NEUTRAL").upper()
            if direction not in {"MORE_GOALS","FEWER_GOALS","NEUTRAL"}:
                direction = "NEUTRAL"

            side = str(sig.get("side") or "BOTH").upper()
            if side not in {"HOME","AWAY","BOTH"}:
                side = "BOTH"

            category = str(sig.get("category") or "FORM").upper()

            cleaned.append({
                "side": side,
                "category": category,
                "direction": direction,
                "impact": round(float(np.clip(_goal_safe_float(sig.get("impact"), 0.0), -1.0, 1.0)), 3),
                "confidence": round(float(np.clip(_goal_safe_float(sig.get("confidence"), 0.0), 0.0, 1.0)), 3),
                "summary": str(sig.get("summary") or "")[:500],
            })

        conf = float(np.clip(
            _goal_safe_float(obj.get("data_confidence"), 0.0),
            0.0, 1.0
        ))

        actionable = [
            sig for sig in cleaned
            if sig["direction"] != "NEUTRAL"
            and sig["confidence"] >= 0.45
            and abs(sig["impact"]) >= 0.10
        ]

        verified = bool(source_count > 0)

        if verified and conf >= 0.72 and len(actionable) >= 2:
            tier = "HIGH"
        elif verified and conf >= 0.48 and len(actionable) >= 1:
            tier = "MEDIUM"
        elif verified:
            tier = "LOW"
        else:
            tier = "NONE"

        out = {
            "status": "OK",
            "web_research_used": True,
            "web_evidence_verified": verified,
            "web_evidence_tier": tier,
            "source_count": int(source_count),
            "sources": sources[:12],
            "data_confidence": round(conf, 3),
            "signals": cleaned,
            "actionable_signal_count": len(actionable),
            "summary": str(obj.get("summary") or "")[:1500],
            "parse_mode": parse_mode,
            "researched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        with _GOAL_WEB_CACHE_LOCK:
            _GOAL_WEB_CACHE[key] = (time.time(), out)

        return out

    except Exception as exc:
        return {
            **empty,
            "status": f"ERROR:{type(exc).__name__}",
            "error": str(exc)[:1200],
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
    Real-Time API + Web Intelligence concurrently.
    A timed-out external branch does not block executor shutdown.
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
        "signals": [],
        "summary": "",
        "data_confidence": 0.0,
        "actionable_signal_count": 0,
        "parse_mode": "NONE",
    }

    started = time.time()
    pool = ThreadPoolExecutor(max_workers=max(2, GOAL_CONTEXT_WORKERS))

    fut_rt = pool.submit(research_goal_realtime_context, home, away, date)
    fut_web = pool.submit(research_goal_web_context, home, away, date)

    try:
        rt_ctx = fut_rt.result(timeout=GOAL_REALTIME_TIMEOUT_SECONDS)
    except FuturesTimeoutError:
        fut_rt.cancel()
        rt_ctx = {**default_rt, "status": "TIMEOUT"}
    except Exception as exc:
        rt_ctx = {
            **default_rt,
            "status": f"ERROR:{type(exc).__name__}",
            "error": str(exc)[:500],
        }

    try:
        web_ctx = fut_web.result(timeout=GOAL_WEB_TIMEOUT_SECONDS)
    except FuturesTimeoutError:
        fut_web.cancel()
        web_ctx = {**default_web, "status": "TIMEOUT"}
    except Exception as exc:
        web_ctx = {
            **default_web,
            "status": f"ERROR:{type(exc).__name__}",
            "error": str(exc)[:500],
        }

    _goal_shutdown_executor_now(pool)

    rt_ctx = dict(rt_ctx or default_rt)
    web_ctx = dict(web_ctx or default_web)

    rt_ctx["injuries_home"] = _goal_dedupe_injuries(rt_ctx.get("injuries_home"))
    rt_ctx["injuries_away"] = _goal_dedupe_injuries(rt_ctx.get("injuries_away"))

    elapsed = round(time.time() - started, 3)
    rt_ctx["_parallel_context_seconds"] = elapsed
    web_ctx["_parallel_context_seconds"] = elapsed

    return rt_ctx, web_ctx



def _goal_ai_schema_prompt(payload):
    return f"""
Tu es l'arbitre final BETSMART GOAL INTELLIGENCE V1.0.6.

OBJECTIF:
Estimer le rythme de buts le plus plausible pour le match, pas la rentabilité du pari.

Tu reçois:
- lambdas bruts du modèle;
- probabilités Poisson et modèles calibrés;
- forme récente;
- historique domicile/extérieur;
- H2H;
- marché (information seulement);
- realtime API;
- Web Intelligence vérifiée.

RÈGLES:
1. Tu ajustes lambda_home et lambda_away, PAS directement Over15/25/35/BTTS.
2. Les deltas demandés doivent être compris entre -0.40 et +0.40 but par équipe.
3. L'absence de données = baisse de confiance, pas modification arbitraire. Ne transforme jamais une absence de données en affirmation qu'il n'y a aucune blessure ou absence.
4. Une blessure offensive peut réduire le lambda de l'équipe; une absence défensive/gardien
   peut augmenter le lambda adverse, MAIS uniquement si le rôle/poste/importance du joueur
   est explicitement présent dans les données fournies. Ne déduis jamais le poste, le statut
   de titulaire ou l'importance d'un joueur à partir de son nom ou de ta mémoire.
5. Une information ancienne/non actuelle = impact nul.
6. H2H est secondaire, surtout si moins de 5 matchs.
7. Le marché ne doit jamais décider seul.
8. Ne confonds pas équipe favorite et match riche en buts.
9. Si les sources se contredisent, réduis l'amplitude de tes deltas.
10. Si les blessures sont listées sans poste/importance, traite-les comme un signal générique d'effectif, pas comme un signal offensif/défensif spécifique.
11. ZERO AJUSTEMENT est une décision valide et souvent préférable lorsque les informations
    externes ne justifient pas un changement des lambdas du modèle.
12. N'ajuste jamais les lambdas uniquement pour "faire travailler" l'IA.
13. Donne une décision exploitable, mais baisse confidence si nécessaire.

SORTIE JSON STRICTE:
{{
 "status":"OK",
 "lambda_home_delta":0.0,
 "lambda_away_delta":0.0,
 "prediction_confidence":0.0,
 "source_agreement":"LOW|MEDIUM|HIGH",
 "risk_level":"LOW|MEDIUM|HIGH",
 "reason_codes":["..."],
 "rationale_short":"...",
 "explanation":"4 à 7 phrases en français, dynamique, citant les deux équipes et expliquant le scénario de buts."
}}

DOSSIER:
{json.dumps(_goal_jsonable(payload), ensure_ascii=False)}
""".strip()



def goal_ai_arbitrator(payload):
    if not GOAL_AI_ENABLED:
        return {
            "status": "DISABLED",
            "lambda_home_delta": 0.0,
            "lambda_away_delta": 0.0,
            "prediction_confidence": 0.0,
            "source_agreement": "LOW",
            "risk_level": "UNKNOWN",
            "reason_codes": ["AI_DISABLED"],
            "rationale_short": "",
            "explanation": "",
        }

    try:
        client = get_openai_client()
        obj = _goal_structured_ai_call(
            client,
            GOAL_AI_MODEL,
            _goal_ai_schema_prompt(payload),
        )

        agreement = str(obj.get("source_agreement") or "LOW").upper()
        if agreement not in {"LOW","MEDIUM","HIGH"}:
            agreement = "LOW"

        risk = str(obj.get("risk_level") or "UNKNOWN").upper()
        if risk not in {"LOW","MEDIUM","HIGH"}:
            risk = "UNKNOWN"

        return {
            "status": "OK",
            "lambda_home_delta": float(np.clip(_goal_safe_float(obj.get("lambda_home_delta"), 0.0), -0.40, 0.40)),
            "lambda_away_delta": float(np.clip(_goal_safe_float(obj.get("lambda_away_delta"), 0.0), -0.40, 0.40)),
            "prediction_confidence": float(np.clip(_goal_safe_float(obj.get("prediction_confidence"), 0.0), 0.0, 1.0)),
            "source_agreement": agreement,
            "risk_level": risk,
            "reason_codes": [str(x)[:120] for x in (obj.get("reason_codes") or [])][:12],
            "rationale_short": str(obj.get("rationale_short") or "")[:1500],
            "explanation": str(obj.get("explanation") or "")[:2500],
        }

    except Exception as exc:
        return {
            "status": "ERROR",
            "lambda_home_delta": 0.0,
            "lambda_away_delta": 0.0,
            "prediction_confidence": 0.0,
            "source_agreement": "LOW",
            "risk_level": "UNKNOWN",
            "reason_codes": [f"AI_ERROR:{type(exc).__name__}"],
            "rationale_short": "",
            "explanation": "",
            "error": str(exc)[:1200],
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



def _goal_final_probs(base_res, lam_h_base, lam_a_base, lam_h_final, lam_a_final):
    base_total = lam_h_base + lam_a_base
    final_total = lam_h_final + lam_a_final

    base_pois_o25 = _p_over(base_total, 2.5)
    final_pois_o25 = _p_over(final_total, 2.5)
    base_pois_btts = _p_btts(lam_h_base, lam_a_base)
    final_pois_btts = _p_btts(lam_h_final, lam_a_final)

    p15 = _clip01(_p_over(final_total, 1.5))
    p35 = _clip01(_p_over(final_total, 3.5))

    # Preserve calibration/hybrid intelligence:
    # shift calibrated probability by 70% of the Poisson movement.
    base_o25 = _goal_safe_float((base_res.get("Over25") or {}).get("proba"), base_pois_o25)
    base_btts = _goal_safe_float((base_res.get("BTTS") or {}).get("proba"), base_pois_btts)
    p25 = _clip01(base_o25 + 0.70 * (final_pois_o25 - base_pois_o25))
    pbtts = _clip01(base_btts + 0.70 * (final_pois_btts - base_pois_btts))

    # Guaranteed monotonic Over hierarchy.
    p15 = max(p15, p25)
    p25 = min(p15, max(p25, p35))
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
    out["_final_state_version"] = "GOAL-1.0.6"
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
    output_mode="reduced",
):
    """
    BetSmart Goal Intelligence V1.0.6.
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

    # V1.0.3: collect independent external contexts concurrently.
    _t = time.time()
    rt_ctx, web_ctx = _goal_build_external_context_parallel(home, away, date)
    _goal_timing['external_context_seconds'] = round(time.time() - _t, 3)

    arbitration_payload = {
        "version": "GOAL-1.0.6",
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
    }

    _t = time.time()
    ai = goal_ai_arbitrator(arbitration_payload)
    _goal_timing['ai_arbitration_seconds'] = round(time.time() - _t, 3)
    dh, da, stability = _goal_stabilize_deltas(hist_ctx, web_ctx, rt_ctx, ai)

    lam_h_final = float(np.clip(lam_h_base + dh, 0.05, 4.5))
    lam_a_final = float(np.clip(lam_a_base + da, 0.05, 4.5))
    lam_t_final = lam_h_final + lam_a_final

    final_probs = _goal_final_probs(
        base_res, lam_h_base, lam_a_base, lam_h_final, lam_a_final
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

    explanation = str(ai.get("explanation") or "").strip()
    if not explanation:
        explanation = _goal_fallback_explanation(
            home, away, final_probs, lam_t_final, confidence
        )

    likely_total = int(round(lam_t_final))
    expected_range = (
        "0-1" if lam_t_final < 1.75 else
        "2-3" if lam_t_final < 3.55 else
        "4+"
    )

    result = {
        "version": "GOAL-1.0.6",
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
        "realtime_intelligence_context": rt_ctx,
        "realtime_web_intelligence": web_ctx,

        "ai_decision": {
            "status": ai.get("status"),
            "decision_origin": "OPENAI_LLM" if ai_status == "OK" else "QUANTITATIVE_STABLE_FALLBACK",
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
            "public_state_synchronized": True,
        },

        "prediction_confidence": round(confidence, 3),
        "low_confidence": bool(confidence < 0.60),
        "explanation": explanation,
        "rule_applied": "goal_base_ml|poisson_hybrid|goal_web_intelligence_v106|goal_ai_structured_v106|goal_stability_layer",
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
