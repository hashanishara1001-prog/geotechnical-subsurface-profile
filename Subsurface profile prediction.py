import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Geotechnical Site Characterization", layout="wide")

st.title("🌍 Geotechnical Soil Profile App")

# File upload
uploaded_file = st.file_uploader("Upload Borehole Data (CSV)", type=["csv"])

if uploaded_file:
    # Load CSV
    df = pd.read_csv(uploaded_file)
    st.write("### Borehole Data Preview", df.head())

    # Check required columns
    required_cols = ["X", "Y", "TE", "BE", "Lithology"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain columns: {required_cols}")
    else:
        # Features and target
        X = df[["X", "Y", "TE", "BE"]]
        y = df["Lithology"]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        # Predict on all data
        df["Predicted_Lithology"] = model.predict(X)

        st.success("✅ Random Forest model trained and predictions generated!")

        st.write("### Prediction Results", df.head())

        # ---- 2D Cross-section Plot ----
        st.subheader("2D Cross-section")

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            df["X"], (df["TE"] + df["BE"]) / 2,  # midpoint depth
            c=pd.factorize(df["Predicted_Lithology"])[0],
            cmap="tab10",
            s=60,
            edgecolor="k"
        )
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Depth (Midpoint Elevation)")
        ax.invert_yaxis()  # Depth increases downward
        ax.set_title("Predicted Lithology Cross-section")
        cbar = plt.colorbar(scatter, ax=ax, ticks=np.arange(len(df["Predicted_Lithology"].unique())))
        cbar.ax.set_yticklabels(df["Predicted_Lithology"].unique())
        st.pyplot(fig)
