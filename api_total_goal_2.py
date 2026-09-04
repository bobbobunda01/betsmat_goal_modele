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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from fonction_totatl_goal_2 import predict_from_user_input, get_valid_date, llm_client


thread=0
app = Flask(__name__)
GOAL_MATCH_WORKERS = int(os.getenv("BETSMART_GOAL_MATCH_WORKERS", "4"))
GOAL_JSON_MODE = "FULL"



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
    if comp==39:
        
        # Chargement des données de la Première league
        
        # Chargement des données historiques
        #chemin_csv = RACINE_PROJET / "data" / "pl" / "pl_24_25.csv"
        s_encours=RACINE_PROJET / "data" / "pl" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        config = load(RACINE_PROJET / "modele" / "pl" / "config2.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "pl" / "lambda_home2.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "pl" / "lambda_away2.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "pl" / "o25_cal2.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "pl" / "btts_ml2.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "pl" / "btts_cal2.joblib")
    # belgique
    elif comp==144:

        s_encours=RACINE_PROJET / "data" / "belg" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi

        config = load(RACINE_PROJET / "modele" / "belg" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "belg" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "belg" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "belg" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "belg" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "belg" / "btts_cal.joblib")

    # SERIE A
    elif comp==135:

        s_encours=RACINE_PROJET / "data" / "sa1" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "sa1" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "sa1" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "sa1" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "sa1" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "sa1" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "sa1" / "btts_cal.joblib")

    # ligA
    elif comp==140:

        s_encours=RACINE_PROJET / "data" / "lg1" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "lg1" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "lg1" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "lg1" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "lg1" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "lg1" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "lg1" / "btts_cal.joblib")


    # BUNSDESLIGA
    elif comp==78:

        s_encours=RACINE_PROJET / "data" / "bl1" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        config = load(RACINE_PROJET / "modele" / "bl1" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "bl1" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "bl1" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "bl1" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "bl1" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "bl1" / "btts_cal.joblib")
        
    # PREMIERE LEAGUE FRANCAISE, L1
    elif comp==61:

        s_encours=RACINE_PROJET / "data" / "fl" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        config = load(RACINE_PROJET / "modele" / "fl" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "fl" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "fl" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "fl" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "fl" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "fl" / "btts_cal.joblib")

     # NEDERLANDE N1, Pays bas
    elif comp==88:

        s_encours=RACINE_PROJET / "data" / "N1" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "N1" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "N1" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "N1" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "N1" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "N1" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "N1" / "btts_cal.joblib")

    # SUISSE
    elif comp==207:

        s_encours=RACINE_PROJET / "data" / "sui" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "sui" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "sui" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "sui" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "sui" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "sui" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "sui" / "btts_cal.joblib")

     # portugal
    elif comp==94:

        s_encours=RACINE_PROJET / "data" / "port" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "port" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "port" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "port" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "port" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "port" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "port" / "btts_cal.joblib")

     # Turquie
    elif comp==203:

        s_encours=RACINE_PROJET / "data" / "turk" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "turk" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "turk" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "turk" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "turk" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "turk" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "turk" / "btts_cal.joblib")
        
    # Japon
    elif comp==98:

        s_encours=RACINE_PROJET / "data" / "japon" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "japon" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "japon" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "japon" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "japon" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "japon" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "japon" / "btts_cal.joblib")

    # grèce 
    elif comp==197:

        s_encours=RACINE_PROJET / "data" / "grece" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "grece" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "grece" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "grece" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "grece" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "grece" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "grece" / "btts_cal.joblib")

    # bresil 
    elif comp==71:

        s_encours=RACINE_PROJET / "data" / "bresil" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "bresil" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "bresil" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "bresil" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "bresil" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "bresil" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "bresil" / "btts_cal.joblib")
        
    # ecosse premeiere league 
    elif comp==179:

        s_encours=RACINE_PROJET / "data" / "ecosse" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "ecosse" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "ecosse" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "ecosse" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "ecosse" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "ecosse" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "ecosse" / "btts_cal.joblib")

    # DANEMARK
    elif comp==119:

        s_encours=RACINE_PROJET / "data" / "danemark" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "danemark" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "danemark" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "danemark" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "danemark" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "danemark" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "danemark" / "btts_cal.joblib")

    # ecosse division 1
    elif comp==180:

        s_encours=RACINE_PROJET / "data" / "ecosse_div_1" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "ecosse_div_1" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "ecosse_div_1" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "ecosse_div_1" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "ecosse_div_1" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "ecosse_div_1" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "ecosse_div_1" / "btts_cal.joblib")
        
    # ecosse division 1
    elif comp==235:

        s_encours=RACINE_PROJET / "data" / "russie" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "russie" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "russie" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "russie" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "russie" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "russie" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "russie" / "btts_cal.joblib")

    # corée du sud
    elif comp==292:

        s_encours=RACINE_PROJET / "data" / "coree_sud" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "coree_sud" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "coree_sud" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "coree_sud" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "coree_sud" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "coree_sud" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "coree_sud" / "btts_cal.joblib")

    # Argentine
    elif comp==128:

        s_encours=RACINE_PROJET / "data" / "argentine" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "argentine" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "argentine" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "argentine" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "argentine" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "argentine" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "argentine" / "btts_cal.joblib")
        
    # league europa
    elif comp==3:

        s_encours=RACINE_PROJET / "data" / "leagues_europa" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "leagues_europa" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "leagues_europa" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "leagues_europa" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "leagues_europa" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "leagues_europa" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "leagues_europa" / "btts_cal.joblib")

    # champions league, à revenir
    elif comp==2:

        s_encours=RACINE_PROJET / "data" / "leagues_champions" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "leagues_champions" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "leagues_champions" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "leagues_champions" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "leagues_champions" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "leagues_champions" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "leagues_champions" / "btts_cal.joblib")

    # Egypte 
    elif comp==233:

        s_encours=RACINE_PROJET / "data" / "egypte" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "egypte" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "egypte" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "egypte" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "egypte" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "egypte" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "egypte" / "btts_cal.joblib")

    # MEXIQUE
    elif comp==262:

        s_encours=RACINE_PROJET / "data" / "mexique" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "mexique" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "mexique" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "mexique" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "mexique" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "mexique" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "mexique" / "btts_cal.joblib")
        
    # BUNDESLIGA 2
    elif comp==79:

        s_encours=RACINE_PROJET / "data" / "bl2" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "bl2" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "bl2" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "bl2" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "bl2" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "bl2" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "bl2" / "btts_cal.joblib")

    # SERIE b
    elif comp==136:

        s_encours=RACINE_PROJET / "data" / "sa2" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "sa2" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "sa2" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "sa2" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "sa2" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "sa2" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "sa2" / "btts_cal.joblib")

    # championShip  angleterre
    elif comp==40:

        s_encours=RACINE_PROJET / "data" / "pl2" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "pl2" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "pl2" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "pl2" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "pl2" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "pl2" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "pl2" / "btts_cal.joblib")

    # Ligue 2 française
    elif comp==62:

        s_encours=RACINE_PROJET / "data" / "fl2" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "fl2" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "fl2" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "fl2" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "fl2" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "fl2" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "fl2" / "btts_cal.joblib")

    # LIGA SECUNDA
    elif comp==141:

        s_encours=RACINE_PROJET / "data" / "lg2" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "lg2" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "lg2" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "lg2" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "lg2" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "lg2" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "lg2" / "btts_cal.joblib")

    # CAN, A COMPLETER
    elif comp==6:

        s_encours=RACINE_PROJET / "data" / "can" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "can" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "can" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "can" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "can" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "can" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "can" / "btts_cal.joblib")

     # CAN, A COMPLETER
    elif comp==1:

        s_encours=RACINE_PROJET / "data" / "mondiale" / "saison_encours.csv"
        hi=pd.read_csv(s_encours)
        #hi.drop('Unnamed: 0', axis=1, inplace=True)
        hi['Date']=pd.to_datetime(hi['Date'])
        df=hi
        
        config = load(RACINE_PROJET / "modele" / "mondiale" / "config.joblib")
        lambda_home_model = load(RACINE_PROJET / "modele" / "mondiale" / "lambda_home.joblib")
        lambda_away_model = load(RACINE_PROJET / "modele" / "mondiale" / "lambda_away.joblib")
        o25_cal = load(RACINE_PROJET / "modele" / "mondiale" / "o25_cal.joblib")
        btts_ml_model = load(RACINE_PROJET / "modele" / "mondiale" / "btts_ml.joblib")
        btts_cal_model = load(RACINE_PROJET / "modele" / "mondiale" / "btts_cal.joblib")
        
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
        btts_cal=btts_cal_model
        )
    #response_json = json.dumps(pred_final, ensure_ascii=False)

    # Log l'entrée + les prédictionsÒ
    #log_prediction(all_results)
            
    return {
        "prediction": pred_final,
        "_match_runtime_seconds": round(time.perf_counter() - _match_t0, 3)
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

        return jsonify({
            "json_mode": GOAL_JSON_MODE,
            "nombre_matchs": len(matches),

            "execution": {
                "parallel": True,
                "match_workers": max_workers,
                "total_seconds": total_seconds,
                "sum_individual_match_seconds": sum_match_seconds,
                "slowest_match_seconds": max_match_seconds,
                "parallel_efficiency_ratio": parallel_efficiency_ratio
            },

            "Resultats": all_results
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