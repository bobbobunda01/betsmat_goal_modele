# BetSmart Goal Intelligence V1.0.8.8.6 OPTIMIZED PARALLEL
# BetSmart Goal Intelligence V1.0.8.2 PARTIAL WEB
# API Match-by-Match - JSON FULL
# 4 matchs en parallèle + temps par match + temps total

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
import pathlib
import sys
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from fichier_py.fonction_totatl_goal_2 import predict_from_user_input, get_valid_date, llm_client


thread=0
app = Flask(__name__)
GOAL_MATCH_WORKERS = int(os.getenv("BETSMART_GOAL_MATCH_WORKERS", "4"))
GOAL_JSON_MODE = "REDUCED"



# Modèle Pydantic pour une entrée
class MatchInput(BaseModel):
    HomeTeam: str
    AwayTeam: str
    comp: int
    OU_O15:float
    OU_O25:float
    OU_O35:float
    BTTS_Yes:float
    match_Date:str
    

# Modèle pour recevoir un tableau d'entrées
class RequestBody(BaseModel):
    matches: List[MatchInput]  # Accepte un tableau de 4 entrées


#RACINE_PROJET = pathlib.Path().resolve().parent.parent
#RACINE_PROJET = pathlib.Path(__file__).resolve().parent.parent

RACINE_PROJET = pathlib.Path(__file__).resolve().parents[1]

# V1.0.8.8.6 resource cache: load CSV/models once per competition, then reuse.
_GOAL_RESOURCE_CACHE = {}
_GOAL_RESOURCE_CACHE_LOCK = threading.RLock()
_GOAL_RESOURCE_LOAD_LOCKS = {}

GOAL_COMPETITION_RESOURCES = {
    1: {'data_dir': 'mondiale', 'data_file': 'saison_encours.csv', 'model_dir': 'mondiale', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    2: {'data_dir': 'leagues_champions', 'data_file': 'saison_encours.csv', 'model_dir': 'leagues_champions', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    3: {'data_dir': 'leagues_europa', 'data_file': 'saison_encours.csv', 'model_dir': 'leagues_europa', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    6: {'data_dir': 'can', 'data_file': 'saison_encours.csv', 'model_dir': 'can', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    39: {'data_dir': 'pl', 'data_file': 'pl_24_25.csv', 'model_dir': 'pl', 'config': 'config2.joblib', 'lambda_home': 'lambda_home2.joblib', 'lambda_away': 'lambda_away2.joblib', 'o25_cal': 'o25_cal2.joblib', 'btts_ml': 'btts_ml2.joblib', 'btts_cal': 'btts_cal2.joblib'},
    40: {'data_dir': 'pl2', 'data_file': 'saison_encours.csv', 'model_dir': 'pl2', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    61: {'data_dir': 'fl', 'data_file': 'saison_encours.csv', 'model_dir': 'fl', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    62: {'data_dir': 'fl2', 'data_file': 'saison_encours.csv', 'model_dir': 'fl2', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    71: {'data_dir': 'bresil', 'data_file': 'saison_encours.csv', 'model_dir': 'bresil', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    78: {'data_dir': 'bl1', 'data_file': 'saison_encours.csv', 'model_dir': 'bl1', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    79: {'data_dir': 'bl2', 'data_file': 'saison_encours.csv', 'model_dir': 'bl2', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    88: {'data_dir': 'N1', 'data_file': 'saison_encours.csv', 'model_dir': 'N1', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    94: {'data_dir': 'port', 'data_file': 'saison_encours.csv', 'model_dir': 'port', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    98: {'data_dir': 'japon', 'data_file': 'saison_encours.csv', 'model_dir': 'japon', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    119: {'data_dir': 'danemark', 'data_file': 'saison_encours.csv', 'model_dir': 'danemark', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    128: {'data_dir': 'argentine', 'data_file': 'saison_encours.csv', 'model_dir': 'argentine', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    135: {'data_dir': 'sa1', 'data_file': 'saison_encours.csv', 'model_dir': 'sa1', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    136: {'data_dir': 'sa2', 'data_file': 'saison_encours.csv', 'model_dir': 'sa2', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    140: {'data_dir': 'lg1', 'data_file': 'saison_encours.csv', 'model_dir': 'lg1', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    141: {'data_dir': 'lg2', 'data_file': 'saison_encours.csv', 'model_dir': 'lg2', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    144: {'data_dir': 'belg', 'data_file': 'saison_encours.csv', 'model_dir': 'belg', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    179: {'data_dir': 'ecosse', 'data_file': 'saison_encours.csv', 'model_dir': 'ecosse', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    180: {'data_dir': 'ecosse_div_1', 'data_file': 'saison_encours.csv', 'model_dir': 'ecosse_div_1', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    197: {'data_dir': 'grece', 'data_file': 'saison_encours.csv', 'model_dir': 'grece', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    203: {'data_dir': 'turk', 'data_file': 'saison_encours.csv', 'model_dir': 'turk', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    207: {'data_dir': 'sui', 'data_file': 'saison_encours.csv', 'model_dir': 'sui', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    233: {'data_dir': 'egypte', 'data_file': 'saison_encours.csv', 'model_dir': 'egypte', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    235: {'data_dir': 'russie', 'data_file': 'saison_encours.csv', 'model_dir': 'russie', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    262: {'data_dir': 'mexique', 'data_file': 'saison_encours.csv', 'model_dir': 'mexique', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
    292: {'data_dir': 'coree_sud', 'data_file': 'saison_encours.csv', 'model_dir': 'coree_sud', 'config': 'config.joblib', 'lambda_home': 'lambda_home.joblib', 'lambda_away': 'lambda_away.joblib', 'o25_cal': 'o25_cal.joblib', 'btts_ml': 'btts_ml.joblib', 'btts_cal': 'btts_cal.joblib'},
}

def _goal_get_comp_load_lock(comp):
    with _GOAL_RESOURCE_CACHE_LOCK:
        lock = _GOAL_RESOURCE_LOAD_LOCKS.get(int(comp))
        if lock is None:
            lock = threading.Lock()
            _GOAL_RESOURCE_LOAD_LOCKS[int(comp)] = lock
        return lock

def _load_goal_comp_resources(comp):
    comp = int(comp)
    with _GOAL_RESOURCE_CACHE_LOCK:
        cached = _GOAL_RESOURCE_CACHE.get(comp)
    if cached is not None:
        return cached

    spec = GOAL_COMPETITION_RESOURCES.get(comp)
    if spec is None:
        raise ValueError(f"Compétition non supportée: {comp}")

    lock = _goal_get_comp_load_lock(comp)
    with lock:
        with _GOAL_RESOURCE_CACHE_LOCK:
            cached = _GOAL_RESOURCE_CACHE.get(comp)
        if cached is not None:
            return cached

        t0 = time.perf_counter()
        data_path = RACINE_PROJET / "data" / spec["data_dir"] / spec["data_file"]
        model_dir = RACINE_PROJET / "modele" / spec["model_dir"]
        hi = pd.read_csv(data_path)
        hi["Date"] = pd.to_datetime(hi["Date"])
        resources = {
            "df": hi,
            "config": load(model_dir / spec["config"]),
            "lambda_home_model": load(model_dir / spec["lambda_home"]),
            "lambda_away_model": load(model_dir / spec["lambda_away"]),
            "o25_cal": load(model_dir / spec["o25_cal"]),
            "btts_ml_model": load(model_dir / spec["btts_ml"]),
            "btts_cal_model": load(model_dir / spec["btts_cal"]),
            "load_seconds": round(time.perf_counter() - t0, 3),
        }
        with _GOAL_RESOURCE_CACHE_LOCK:
            _GOAL_RESOURCE_CACHE[comp] = resources
        return resources

@app.route('/', methods=["GET"])
def Accueil():
    return jsonify({'Message': 'Bienvenue sur l\'API de prédiction de matchs'})




def _process_goal_match(match):
    # Temps propre au traitement de CE match
    _match_t0 = time.perf_counter()

    # Traitement pour chaque match
    donnees_df = pd.DataFrame([match.dict()])

    home=np.array(donnees_df.HomeTeam.values).item()
    away=np.array(donnees_df.AwayTeam.values).item()
    #comp=np.array(donnees_df.comp.values).item()
    comp=donnees_df["comp"].values[0]
    odds_o15 = donnees_df["OU_O15"].values[0]
    odds_o25 = donnees_df["OU_O25"].values[0]
    odds_o35 = donnees_df["OU_O35"].values[0]
    odds_bbts= donnees_df["BTTS_Yes"].values[0]
    match_date=np.array(donnees_df.match_Date.values).item()
    # Premiere league ANGLETERRE
    resources = _load_goal_comp_resources(comp)
    df = resources["df"]
    config = resources["config"]
    lambda_home_model = resources["lambda_home_model"]
    lambda_away_model = resources["lambda_away_model"]
    o25_cal = resources["o25_cal"]
    btts_ml_model = resources["btts_ml_model"]
    btts_cal_model = resources["btts_cal_model"]

    date_match=get_valid_date(match_date)

    odds= {"OU_O15": odds_o15, "OU_O25": odds_o25, "OU_O35": odds_o35, "BTTS_Yes": odds_bbts}
    #odds= ""
    #pred["_use_realtime"] = True 
    pred_final = predict_from_user_input(
        df,
        home,
        away,
        date_match,
        odds,
        out_dir="betsmart_goals_out_pl",
        use_llm= True,
        llm_client=llm_client,
        explainer=None,
        config=config,
        lambda_home_model=lambda_home_model,
        lambda_away_model=lambda_away_model,
        o25_cal=o25_cal,
        btts_ml=btts_ml_model,
        btts_cal=btts_cal_model,
        output_mode="reduced"
        )
    #response_json = json.dumps(pred_final, ensure_ascii=False)

    # Log l'entrée + les prédictionsÒ
    #log_prediction(all_results)
            
    return {
        "prediction": pred_final,
        "_match_runtime_seconds": round(time.perf_counter() - _match_t0, 3)
    }



# ======================================================================================
# JSON FINAL COMPATIBILITY LAYER
# Présentation uniquement : ne modifie ni le moteur, ni les modèles, ni les probabilités.
# ======================================================================================

_GOAL_FINAL_MARKET_ORDER = ["Over15", "Over25", "Over35", "BTTS"]


def _goal_json_action(market_pack):
    """Reproduit la règle historique BET / NO_BET à partir du résultat final."""
    market_pack = market_pack if isinstance(market_pack, dict) else {}
    pred = int(market_pack.get("pred", 0))
    low = bool(market_pack.get("low_confidence", True))

    if low:
        return "NO_BET", "low_confidence=true"
    if pred == 0:
        return "NO_BET", "pred=0"

    proba = float(market_pack.get("proba", 0.0))
    return "BET", f"pred=1 & low_confidence=false (p={proba:.2f})"


def _goal_json_key_points(explanation_text):
    """
    Produit exactement 3 points à partir de l'explication déjà générée par le moteur.
    Aucun nouvel appel IA et aucune nouvelle interprétation prédictive.
    """
    text = str(explanation_text or "").strip()
    if not text:
        return ["", "", ""]

    parts = [
        p.strip()
        for p in re.split(r"(?<=[.!?])\s+", text)
        if p and p.strip()
    ]

    if not parts:
        parts = [text]

    parts = parts[:3]
    while len(parts) < 3:
        parts.append("")

    return parts


def _goal_json_explanation(prediction_result):
    """
    Convertit l'explication publique 8.14 (string) vers l'ancien contrat JSON objet.
    Cette fonction ne change aucune valeur de prédiction.
    """
    prediction_result = prediction_result if isinstance(prediction_result, dict) else {}
    explanation_text = str(prediction_result.get("explanation") or "")

    recommended = []
    bet_markets = set()

    for market in _GOAL_FINAL_MARKET_ORDER:
        pack = prediction_result.get(market, {})
        action, reason = _goal_json_action(pack)

        recommended.append({
            "action": action,
            "market": market,
            "reason": reason
        })

        if action == "BET":
            bet_markets.add(market)

    risk_flags = []
    if "Over25" in bet_markets and "BTTS" in bet_markets:
        risk_flags.append("corrélation")

    return {
        "explanation": explanation_text,
        "key_points": _goal_json_key_points(explanation_text),
        "recommended_markets": recommended,
        "risk_flags": risk_flags
    }


def _goal_json_final_view(prediction_result):
    """
    Contrat JSON final demandé.
    Seuls les champs de présentation sont filtrés/réorganisés.
    """
    prediction_result = prediction_result if isinstance(prediction_result, dict) else {}

    if prediction_result.get("status") == "ERROR":
        return {
            "status": "ERROR",
            "error": prediction_result.get("error")
        }

    return {
        "BTTS": prediction_result.get("BTTS"),
        "Over15": prediction_result.get("Over15"),
        "Over25": prediction_result.get("Over25"),
        "Over35": prediction_result.get("Over35"),
        "explanation": _goal_json_explanation(prediction_result),
        "lambda_away": prediction_result.get("lambda_away"),
        "lambda_home": prediction_result.get("lambda_home"),
        "lambda_total": prediction_result.get("lambda_total")
    }


@app.route('/predire/pred_goal', methods=["POST"])
def prediction():
    # Temps total de la requête API
    _request_t0 = time.perf_counter()

    if not request.json:
        return jsonify({
            "Erreur": "Aucun fichier JSON fourni"
        }), 400

    try:
        body = RequestBody(**request.json)
        matches = list(body.matches)

        if not matches:
            return jsonify({
                "Erreur": "Aucun match fourni"
            }), 400

        # L'ordre de sortie reste identique à l'ordre d'entrée,
        # même si les matchs se terminent dans un ordre différent.
        all_results = [None] * len(matches)

        max_workers = max(
            1,
            min(GOAL_MATCH_WORKERS, len(matches))
        )

        # Temps de soumission, utile aussi pour mesurer un match en erreur.
        submitted_at = {}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {}

            for idx, match in enumerate(matches):
                submitted_at[idx] = time.perf_counter()
                future = pool.submit(_process_goal_match, match)
                future_map[future] = idx

            for future in as_completed(future_map):
                idx = future_map[future]
                match = matches[idx]

                try:
                    worker_result = future.result()

                    prediction_result = worker_result["prediction"]
                    match_runtime = worker_result["_match_runtime_seconds"]

                    all_results[idx] = {
                        "json_mode": GOAL_JSON_MODE,
                        "match": {
                            "index": idx + 1,
                            "home": match.HomeTeam,
                            "away": match.AwayTeam,
                            "competition": match.comp,
                            "date": match.match_Date
                        },
                        "runtime_seconds": match_runtime,
                        "prediction": prediction_result
                    }

                except Exception as exc:
                    match_runtime = round(
                        time.perf_counter() - submitted_at[idx],
                        3
                    )

                    all_results[idx] = {
                        "json_mode": GOAL_JSON_MODE,
                        "match": {
                            "index": idx + 1,
                            "home": match.HomeTeam,
                            "away": match.AwayTeam,
                            "competition": match.comp,
                            "date": match.match_Date
                        },
                        "runtime_seconds": match_runtime,
                        "prediction": {
                            "status": "ERROR",
                            "error": f"{type(exc).__name__}: {exc}",
                            "_parallel_match_error": True
                        }
                    }

        total_seconds = round(
            time.perf_counter() - _request_t0,
            3
        )

        match_times = [
            r.get("runtime_seconds", 0.0)
            for r in all_results
            if isinstance(r, dict)
        ]

        sum_match_seconds = round(sum(match_times), 3)
        max_match_seconds = round(max(match_times), 3) if match_times else 0.0

        # Indicateur simple permettant d'observer le gain du parallélisme.
        # > 1 signifie que plusieurs traitements ont effectivement été recouverts.
        parallel_efficiency_ratio = (
            round(sum_match_seconds / total_seconds, 2)
            if total_seconds > 0
            else None
        )

        logging.basicConfig(level=logging.INFO)
        logging.info(
            "📊 %s matchs traités en %ss avec %s worker(s)",
            len(matches),
            total_seconds,
            max_workers
        )

        # JSON public final : compatibilité avec le contrat historique demandé.
        # Le traitement parallèle et les mesures restent internes à l'API.
        final_results = [
            _goal_json_final_view(
                item.get("prediction", {}) if isinstance(item, dict) else {}
            )
            for item in all_results
        ]

        return jsonify({
            "Resultats": final_results
        })

    except Exception as exc:
        return jsonify({
            "Erreur": f"{type(exc).__name__}: {exc}",
            "runtime_seconds": round(
                time.perf_counter() - _request_t0,
                3
            )
        }), 400


if __name__ == '__main__':
    app.run(debug=True)