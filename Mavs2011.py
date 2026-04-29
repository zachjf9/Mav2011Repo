import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

regularSeason = pd.read_csv(
    "https://raw.githubusercontent.com/zachjf9/Mav2011Repo/refs/heads/main/MavsRegularSeason.csv",
    header=1
)

playoffs = pd.read_csv(
    "https://raw.githubusercontent.com/zachjf9/Mav2011Repo/refs/heads/main/MavsPlayOffs.csv",
    header=1
)
#four factors data
four_factors = pd.read_csv("MavsOppFourFactors.csv")

regularSeason = regularSeason.loc[:, ~regularSeason.columns.astype(str).str.contains("Unnamed")]
playoffs = playoffs.loc[:, ~playoffs.columns.astype(str).str.contains("Unnamed")]

regularSeason = regularSeason.loc[:, ~regularSeason.columns.duplicated()]
playoffs = playoffs.loc[:, ~playoffs.columns.duplicated()]

# data preprocessing (used AI to clean the data)
def preprocess(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "Rslt" in df.columns:
        df["win"] = df["Rslt"].astype(str).str.contains("W").astype(int)
    elif "Result" in df.columns:
        df["win"] = df["Result"].astype(str).str.contains("W").astype(int)
    else:
        df["win"] = np.nan

    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    return df

reg_clean = preprocess(regularSeason)
playoffs_clean = preprocess(playoffs)

st.title("2011 Dallas Mavericks Season Analysis")

# model 1

model1_data = reg_clean.copy()

model1_data["win"] = model1_data["Rslt"].astype(str).str.contains("W").astype(int)

feature_cols = [
    "FG%", "3P%", "FT%",
    "TRB", "AST", "STL", "BLK", "TOV"
]

feature_cols = [c for c in feature_cols if c in model1_data.columns]
model1_data = model1_data.dropna(subset=feature_cols + ["win"])

if len(model1_data) < 10:
    st.error("Not enough usable data after cleaning.")
    st.stop()

X = model1_data[feature_cols]
y = model1_data["win"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)

importance = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": tree.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.subheader("Model 1: What Stats Most Impact Wins?")
st.dataframe(importance)

# model 1 visualization

fig, ax = plt.subplots(figsize=(15, 10))
ax.barh(importance["Feature"], importance["Importance"], color="skyblue")
ax.set_xlabel("Importance")
ax.set_title("Feature Importance in Predicting Wins")
ax.invert_yaxis()

st.pyplot(fig)

st.title("Dallas Mavericks Four Factors")

cols = st.columns(2)

for i, row in four_factors.iterrows():
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    ax.bar(["Mavs", "Opp"], [row["team"], row["opponent"]])
    ax.set_title(row["metric"], fontsize=10)

    cols[i % 2].pyplot(fig, use_container_width=False)