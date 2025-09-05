import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Geotechnical Site Characterization", layout="wide")
st.title("🌍 Subsurface-Profile")

# File upload
uploaded_file = st.file_uploader("Upload Borehole Data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Borehole Data Preview", df.head())

    required_cols = ["X", "Y", "TE", "BE", "Lithology"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain columns: {required_cols}")
    else:
        X = df[["X", "Y", "TE", "BE"]]
        y = df["Lithology"]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        # Predict on test set
        y_test_pred = model.predict(X_test)

        st.success("✅ Random Forest model trained and predictions generated!")

        # Plot test boreholes
        st.subheader("Test Boreholes Plot")

        # Compute midpoint for plotting
        X_test_plot = X_test.copy()
        X_test_plot["Midpoint"] = (X_test_plot["TE"] + X_test_plot["BE"]) / 2
        X_test_plot["Predicted_Lithology"] = y_test_pred

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for lith in X_test_plot["Predicted_Lithology"].unique():
            subset = X_test_plot[X_test_plot["Predicted_Lithology"] == lith]
            ax.scatter(subset["X"], subset["Midpoint"], label=lith, s=60)

        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Midpoint Depth (m)")
        ax.set_title("Test Boreholes - Predicted Lithology")
        ax.invert_yaxis()  # Depth increases downward
        ax.legend(title="Lithology")
        st.pyplot(fig)
