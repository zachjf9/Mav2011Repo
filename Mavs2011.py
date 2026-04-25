import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


regularSeason = pd.read_csv('https://raw.githubusercontent.com/zachjf9/Mav2011Repo/refs/heads/main/MavsRegularSeason.csv')

playoffs = pd.read_csv('https://raw.githubusercontent.com/zachjf9/Mav2011Repo/refs/heads/main/MavsPlayOffs.csv')

# data preprocessing (used AI to clean the data)

def preprocess(df):
    df = df.copy()

    numeric_cols = [
        'TeamPts','OppPts','FG','FGA','FG_pct',
        '3P','3PA','3P_pct',
        '2P','2PA','2P_pct',
        'eFG_pct','FT','FTA','FT_pct',
        'ORB','DRB','TRB',
        'AST','STL','BLK','TOV','PF',
        'OppFG','OppFGA','OppFG_pct',
        'Opp3P','Opp3PA','Opp3P_pct',
        'Opp2P','Opp2PA','Opp2P_pct',
        'OppeFG_pct',
        'OppFT','OppFTA','OppFT_pct',
        'OppORB','OppDRB','OppTRB',
        'OppAST','OppSTL','OppBLK','OppTOV','OppPF'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["HomeAway"] = df["HomeAway"].astype(str).str.strip()

    df["win"] = df["Result"].astype(str).str.contains("W").astype(int)

    df["Point_Diff"] = df["TeamPts"] - df["OppPts"]
    df["Rebound_Diff"] = df["TRB"] - df["OppTRB"]
    df["Turnover_Diff"] = df["TOV"] - df["OppTOV"]

    df = df.dropna(subset=["Date", "TeamPts", "OppPts"])

    return df

regular_clean = preprocess(regularSeason)
playoffs_clean = preprocess(playoffs)

