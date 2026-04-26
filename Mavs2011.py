import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

regularSeason = pd.read_csv('https://raw.githubusercontent.com/zachjf9/Mav2011Repo/refs/heads/main/MavsRegularSeason.csv')

playoffs = pd.read_csv('https://raw.githubusercontent.com/zachjf9/Mav2011Repo/refs/heads/main/MavsPlayOffs.csv')

# data preprocessing (used AI to clean the data)

def preprocess(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    rename_map = {"Home/Away": "HomeAway","Home Away": "HomeAway","W/L": "Result","Outcome": "Result"}

    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

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

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if "Result" in df.columns:
        df["win"] = df["Result"].astype(str).str.contains("W").astype(int)
    else:
        df["win"] = np.nan  

    if "TeamPts" in df.columns and "OppPts" in df.columns:
        df["Point_Diff"] = df["TeamPts"] - df["OppPts"]

    if "TRB" in df.columns and "OppTRB" in df.columns:
        df["Rebound_Diff"] = df["TRB"] - df["OppTRB"]

    if "TOV" in df.columns and "OppTOV" in df.columns:
        df["Turnover_Diff"] = df["TOV"] - df["OppTOV"]

    required = ["TeamPts", "OppPts"]
    if "Date" in df.columns:
        required.append("Date")

    df = df.dropna(subset=[c for c in required if c in df.columns])

    return df

reg_clean = preprocess(regularSeason)
playoffs_clean = preprocess(playoffs)

# visualization and models

st.title("2011 Dallas Mavericks Season Analysis")

# the most important factors that contributed to the mavs wins during the season

feature_cols = [
    "FG_pct","3P_pct","FT_pct","TRB","AST",
    "STL","BLK","TOV","OppFG_pct","OppTOV",
    "Rebound_Diff","Turnover_Diff"
]

valid_features = [col for col in feature_cols if col in reg_clean.columns]
importantFactors_data = reg_clean.copy()
importantFactors_data = importantFactors_data.dropna(subset=["win"])
importantFactors_data[valid_features] = importantFactors_data[valid_features].fillna(
    importantFactors_data[valid_features].median()
)

st.write("Dataset size:", importantFactors_data.shape)

if len(importantFactors_data) < 10:
    st.error("Not enough data to train model.")
    st.stop()

X1 = importantFactors_data[valid_features]
y1 = importantFactors_data["win"]

X_train,X_test,y_train,y_test = train_test_split(
    X1, y1, test_size=.25, random_state=42
)

tree1 = DecisionTreeClassifier(max_depth=4, random_state=42)
tree1.fit(X_train,y_train)