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

    return df

reg_clean = preprocess(regularSeason)
playoffs_clean = preprocess(playoffs)

st.title("2011 Dallas Mavericks Season Analysis")

model1_data = reg_clean.copy()
model1_data = model1_data.dropna(subset=["win"])

feature_cols = [
    "FG", "FGA",
    "3P", "3PA",
    "2P", "2PA",
    "TRB", "AST",
    "STL", "BLK",
    "TOV", "PF"
]

feature_cols = [c for c in feature_cols if c in model1_data.columns]

for col in feature_cols:
    model1_data[col] = pd.to_numeric(model1_data[col], errors="coerce")

model1_data[feature_cols] = model1_data[feature_cols].fillna(
    model1_data[feature_cols].median()
)

st.write("Dataset shape:", model1_data.shape)
st.write("Features used:", feature_cols)

if len(model1_data) < 10 or len(feature_cols) == 0:
    st.error("Not enough usable data after cleaning.")
    st.stop()

X = model1_data[feature_cols]
y = model1_data["win"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# model
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)

importance = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": tree.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.subheader("Model 1: What Stats Most Impact Wins?")
st.dataframe(importance)