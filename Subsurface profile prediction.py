import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, log_loss
)
from bayes_opt import BayesianOptimization
import joblib
import matplotlib.patches as patches

st.set_page_config(page_title="Geotechnical Soil Profile", layout="wide")

# App header
st.title("🌍 Subsurface - Profile")

# File uploader
uploaded_file = st.file_uploader("Upload Borehole Data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Borehole Data Preview", df.head())

    # Required columns
    required_cols = ["X", "Y", "TE", "BE", "formation"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain columns: {required_cols}")
    else:
        # Encode soil labels
        df['Soil_Label'] = df['formation'].astype('category').cat.codes
        features = ['X', 'Y', 'TE', 'BE']
        X = df[features]
        y = df['Soil_Label']

        # Stratified train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1, stratify=y
        )

        st.subheader("⚙️ Bayesian Optimization Parameters")
        n_init = st.number_input("Initial points (init_points)", min_value=1, max_value=20, value=5)
        n_iter = st.number_input("Iterations (n_iter)", min_value=1, max_value=50, value=10)

        if st.button("Run Bayesian Optimization & Train RF"):

            st.info("Running Bayesian Optimization... This may take a few minutes.")

            # Define RF cross-validation function
            def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf):
                model = RandomForestClassifier(
                    n_estimators=int(n_estimators),
                    max_depth=int(max_depth),
                    min_samples_split=int(min_samples_split),
                    min_samples_leaf=int(min_samples_leaf),
                    max_features='sqrt',
                    criterion='gini',
                    class_weight='balanced',
                    bootstrap=True,
                    random_state=1,
                    n_jobs=-1
                )
                f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
                return f1.mean()

            # Parameter bounds
            pbounds = {
                'n_estimators': (50, 300),
                'max_depth': (5, 30),
                'min_samples_split': (2, 10),
                'min_samples_leaf': (1, 5)
            }

            # Bayesian Optimization
            optimizer = BayesianOptimization(f=rf_cv, pbounds=pbounds, random_state=1)
            optimizer.maximize(init_points=n_init, n_iter=n_iter, acq='ei')

            best_params = {k: int(v) for k, v in optimizer.max['params'].items()}
            st.write("### Best Hyperparameters Found")
            st.json(best_params)

            # Train final model
            rf = RandomForestClassifier(
                **best_params,
                max_features='sqrt',
                criterion='gini',
                class_weight='balanced',
                bootstrap=True,
                random_state=1,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            joblib.dump(rf, "rf_bayes_model.pkl")

            # Evaluate on training set
            y_train_pred = rf.predict(X_train)
            y_train_proba = rf.predict_proba(X_train)
            train_acc = accuracy_score(y_train, y_train_pred)
            train_loss = log_loss(y_train, y_train_proba)
            print(f"Training Accuracy: {train_acc:.4f}")
            print(f"Training Log Loss: {train_loss:.4f}")

            # Evaluate on test set
            y_pred = rf.predict(X_test)
            y_proba = rf.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            loss = log_loss(y_test, y_proba)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            print(f"Accuracy: {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall: {rec:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Log Loss: {loss:.4f}")
            print("Classification Report:\n", classification_report(y_test, y_pred))

