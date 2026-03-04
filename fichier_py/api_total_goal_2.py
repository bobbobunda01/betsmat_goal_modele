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
from datetime import datetime
from  fichier_py.fonction_totatl_goal_2 import  predict_from_user_input,get_valid_date, llm_client


thread=0
app = Flask(__name__)



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



@app.route('/predire/pred_goal', methods=["POST"])
def prediction():
    if not request.json:
        return jsonify({'Erreur': 'Aucun fichier JSON fourni'}), 400
    
    try:
        # Extraction des 4 entrées
        body = RequestBody(**request.json)
        all_results = []

        for match in body.matches:
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
            
            all_results.append(pred_final)
            # Log l'entrée + les prédictionsÒ
            #log_prediction(all_results)
        
        logging.basicConfig(level=logging.INFO)

        logging.info(f"📊 Résultats all_results : {all_results}")
        return jsonify({'Resultats': all_results})
     
    

    except Exception as e:
        return jsonify({'Erreur': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)