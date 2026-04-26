import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

regularSeason = pd.read_csv("https://raw.githubusercontent.com/zachjf9/Mav2011Repo/refs/heads/main/MavsRegularSeason.csv")

playoffs = pd.read_csv("https://raw.githubusercontent.com/zachjf9/Mav2011Repo/refs/heads/main/MavsPlayOffs.csv")

regularSeason.columns = regularSeason.columns.str.strip()
playoffs.columns = playoffs.columns.str.strip()

regularSeason = regularSeason.loc[:, ~regularSeason.columns.duplicated()]
playoffs = playoffs.loc[:, ~playoffs.columns.duplicated()]

# data preprocessing

def preprocess(df):
    df = df.copy()

    if "Rslt" in df.columns:
        df["win"] = df["Rslt"].astype(str).str.contains("W").astype(int)
    else:
        df["win"] = np.nan

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

reg_clean = preprocess(regularSeason)
playoffs_clean = preprocess(playoffs)

# data visualization and modeling

st.title("2011 Dallas Mavericks ML Analysis")

# first ml model

feature_cols = [
    "FG","FGA","FG%",
    "3P","3PA","3P%",
    "TRB","AST","STL","BLK","TOV","FT%"
]

model1_data = reg_clean.copy()

model1_data = model1_data.dropna(subset=["win"])

valid_features = [c for c in feature_cols if c in model1_data.columns]

model1_data[valid_features] = model1_data[valid_features].fillna(
    model1_data[valid_features].median()
)

st.write("Model 1 dataset size:", model1_data.shape)

X1 = model1_data[valid_features]
y1 = model1_data["win"]

X_train, X_test, y_train, y_test = train_test_split(
    X1, y1, test_size=0.25, random_state=42
)

tree1 = DecisionTreeClassifier(max_depth=4, random_state=42)
tree1.fit(X_train, y_train)

importance = pd.DataFrame({
    "Feature": valid_features,
    "Importance": tree1.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.subheader("Model 1: What Stats Most Impact Wins?")
st.dataframe(importance)