import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Lead Intelligence",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("📊 AI Sales Lead Intelligence Dashboard")
st.write("Upload your leads CSV file to generate AI scoring and recommendations.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])


# ---------------- ML TRAINING FUNCTION ----------------
def train_model(df):
    X = df[["budget", "interactions"]]
    y = (df["budget"] > df["budget"].median()).astype(int)

    model = RandomForestClassifier()
    model.fit(X, y)
    return model


# ---------------- RECOMMENDATION FUNCTION ----------------
def recommend_action(score):
    if score >= 85:
        return "Call Immediately"
    elif score >= 70:
        return "Send Follow-up Email"
    elif score >= 50:
        return "Nurture Later"
    else:
        return "Low Priority"


# ---------------- MAIN LOGIC ----------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = ["budget", "interactions"]

    if not all(col in df.columns for col in required_cols):
        st.error("CSV must contain columns: budget and interactions")
    else:
        model = train_model(df)

        # Predict probability
        df["AI Score (%)"] = model.predict_proba(
            df[["budget", "interactions"]]
        )[:, 1] * 100

        df["AI Score (%)"] = df["AI Score (%)"].round(0).astype(int)

        df["Recommended Action"] = df["AI Score (%)"].apply(recommend_action)

        df = df.sort_values("AI Score (%)", ascending=False)

        # ---------------- KPI SECTION ----------------
        st.subheader("Key Insights")
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Leads", len(df))
        col2.metric("Top Lead Score", df["AI Score (%)"].max())
        col3.metric("Average Score", round(df["AI Score (%)"].mean(), 2))

        st.divider()

        # ---------------- TABLE + CHART ----------------
        colA, colB = st.columns(2)

        with colA:
            st.subheader("Lead Ranking")
            st.dataframe(df)

        with colB:
            st.subheader("Score Distribution")
            st.bar_chart(df["AI Score (%)"])