import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Geotechnical Site Characterization", layout="wide")
st.title("🌍 Geotechnical Test Boreholes")

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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train Random Forest
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        # Predict on test set
        y_test_pred = model.predict(X_test)

        st.success("✅ Random Forest model trained and predictions generated!")

        # Add predictions back to test set
        X_test_plot = X_test.copy()
        X_test_plot["Predicted_Lithology"] = y_test_pred

        # Plot continuous boreholes
        st.subheader("Test Boreholes (Predicted Lithology)")

        fig, ax = plt.subplots(figsize=(10, 6))
        unique_lith = X_test_plot["Predicted_Lithology"].unique()
        colors = plt.cm.tab10.colors
        color_map = {l: colors[i % len(colors)] for i, l in enumerate(unique_lith)}

        for i, row in X_test_plot.iterrows():
            rect = patches.Rectangle(
                (row["X"] - 1, row["BE"]),   # bottom-left corner
                2,                           # borehole width
                row["TE"] - row["BE"],       # height = TE - BE
                linewidth=1,
                edgecolor="black",
                facecolor=color_map[row["Predicted_Lithology"]],
                alpha=0.9
            )
            ax.add_patch(rect)

        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Depth (m)")
        ax.set_title("Predicted Lithology for Test Boreholes")
        ax.invert_yaxis()  # Depth increases downward

        # Legend
        handles = [patches.Patch(color=color_map[l], label=l) for l in unique_lith]
        ax.legend(handles=handles, title="Lithology", bbox_to_anchor=(1.05, 1), loc="upper left")

        st.pyplot(fig)
