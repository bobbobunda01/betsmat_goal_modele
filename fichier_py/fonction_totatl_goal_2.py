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


def llm_explainer(payload: dict, llm_client) -> dict:
    """
    llm_client(prompt:str)->str doit retourner du JSON (string).
    """
    prompt = build_explanation_prompt(payload)
    raw = llm_client(prompt)

    # sécurité parse JSON
    try:
        exp = json.loads(raw)
    except Exception:
        # si le provider renvoie du texte, on fallback plus haut
        return {
            "explanation": "Explication indisponible (réponse LLM non JSON).",
            "key_points": [],
            "recommended_markets": [],
            "risk_flags": ["llm_non_json"]
        }

    return exp
""""" 
def llm_client(prompt: str) -> str:
    resp = _client.chat.completions.create(
        model="gpt-4.1-mini",  # tu peux changer
        messages=[
            {"role": "system", "content": "Tu réponds STRICTEMENT en JSON valide, sans texte avant/après."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content
"""
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
    btts_cal=None
):
    cfg, lambda_home, lambda_away, o25_cal_model, btts_ml_model, btts_cal_model = config, lambda_home_model, lambda_away_model, o25_cal, btts_ml, btts_cal

    match_df = prepare_user_input_and_enrich(df_hist, home, away, date, odds)

    # garantir odds attendues
    for c in ["OU_O15","OU_O25","OU_O35","BTTS_Yes"]:
        if (c in cfg["lambda_features"]) or (c in cfg["o25_features"]) or (c in cfg["btts_features"]):
            if c not in match_df.columns:
                match_df[c] = np.nan

    # runtime inject models dans config (propre, pas de global)
    cfg_runtime = cfg.copy()
    cfg_runtime["_o25_cal_model"] = o25_cal_model
    cfg_runtime["_btts_cal_model"] = btts_cal_model

    res = predict_goal_with_proba(
        match_df=match_df,
        lambda_home_model=lambda_home,
        lambda_away_model=lambda_away,
        btts_ml=btts_ml_model,
        btts_cal=btts_cal_model,
        config=cfg_runtime,
        explainer=explainer,
        use_llm=use_llm,
        llm_client=llm_client
    )
    return res

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
