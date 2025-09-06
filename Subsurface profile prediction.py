import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    classification_report, confusion_matrix, log_loss,
    accuracy_score, precision_score, recall_score, f1_score
)
from bayes_opt import BayesianOptimization

st.set_page_config(page_title="Geotechnical Site Characterization", layout="wide")
st.title("🌍 Subsurface - Profile")

# File upload
uploaded_file = st.file_uploader("Upload Borehole Data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Borehole Data Preview", df.head())

    required_cols = ["X", "Y", "TE", "BE", "Lithology"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain columns: {required_cols}")
    else:
        # Prepare data
        df = df.copy()
        df['Soil_Label'] = df['Lithology'].astype('category').cat.codes
        label_map = dict(enumerate(df['Lithology'].astype('category').cat.categories))

        features = ['X', 'Y', 'TE', 'BE']
        X = df[features]
        y = df['Soil_Label']

        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1, stratify=y
        )

        st.subheader("⚙️ Bayesian Optimization Parameters")
        n_init = st.number_input("Initial points (init_points)", min_value=0, max_value=20, value=5)
        n_iter = st.number_input("Iterations (n_iter)", min_value=0, max_value=50, value=10)
        do_opt = st.checkbox("Run Bayesian Optimization (uncheck to skip and use defaults)", value=True)

        # Default hyperparameters (used if skip optimization or it fails)
        default_params = {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }

        # Define Bayesian Optimization function (safe)
        def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf):
            # turn floats to ints
            n_estimators = int(round(n_estimators))
            max_depth = int(round(max_depth))
            min_samples_split = int(round(min_samples_split))
            min_samples_leaf = int(round(min_samples_leaf))

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth if max_depth > 0 else None,
                min_samples_split=max(2, min_samples_split),
                min_samples_leaf=max(1, min_samples_leaf),
                max_features='sqrt',
                criterion='gini',
                class_weight='balanced',
                bootstrap=True,
                random_state=1,
                n_jobs=-1
            )
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
            try:
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
                return float(np.mean(scores))
            except Exception as e:
                # Print to terminal/streamlit for debugging and return a safe low score
                st.warning(f"CV error during optimization: {e}")
                return 0.0

        # Parameter bounds for optimization
        pbounds = {
            'n_estimators': (10, 300),
            'max_depth': (3, 40),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10),
        }

        best_params = default_params.copy()

        if do_opt and (int(n_init) > 0 or int(n_iter) > 0):
            with st.spinner("Running Bayesian Optimization (this may take a while)..."):
                try:
                    optimizer = BayesianOptimization(
                        f=rf_cv,
                        pbounds=pbounds,
                        random_state=1,
                        verbose=2
                    )
                    # allow 0 for init or iter (skip accordingly)
                    if int(n_init) == 0 and int(n_iter) == 0:
                        st.info("Both init_points and n_iter are 0 — skipping optimization.")
                    else:
                        optimizer.maximize(init_points=int(n_init), n_iter=int(n_iter))
                        opt_res = optimizer.max
                        if opt_res and 'params' in opt_res:
                            params = opt_res['params']
                            # cast to ints
                            best_params = {
                                'n_estimators': int(round(params.get('n_estimators', default_params['n_estimators']))),
                                'max_depth': int(round(params.get('max_depth', default_params['max_depth']))),
                                'min_samples_split': int(round(params.get('min_samples_split', default_params['min_samples_split']))),
                                'min_samples_leaf': int(round(params.get('min_samples_leaf', default_params['min_samples_leaf'])))
                            }
                        else:
                            st.warning("BayesOpt returned no results — using default params.")
                except Exception as e:
                    st.error(f"Bayesian Optimization failed: {e}")
                    st.info("Falling back to default hyperparameters.")
                    best_params = default_params.copy()
        else:
            st.info("Skipping Bayesian Optimization — using default hyperparameters.")
            best_params = default_params.copy()

        st.subheader("Best Hyperparameters (used for final model)")
        for k, v in best_params.items():
            st.write(f"- {k}: {v}")

        # Train final model with best params
        rf = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'] if best_params['max_depth'] > 0 else None,
            min_samples_split=max(2, best_params['min_samples_split']),
            min_samples_leaf=max(1, best_params['min_samples_leaf']),
            max_features='sqrt',
            criterion='gini',
            class_weight='balanced',
            bootstrap=True,
            random_state=1,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        # Evaluate on training set
        y_train_pred = rf.predict(X_train)
        y_train_proba = rf.predict_proba(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_loss = log_loss(y_train, y_train_proba)
        st.write("### Training metrics")
        st.write(f"- Accuracy: {train_acc:.4f}")
        st.write(f"- Log loss: {train_loss:.4f}")

        # Evaluate on test set
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_proba)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        st.write("### Test metrics")
        st.write(f"- Accuracy: {acc:.4f}")
        st.write(f"- Log loss: {loss:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        im = ax_cm.imshow(cm, interpolation='nearest', aspect='auto')
        ax_cm.set_title("Confusion Matrix",fontsize=14)
        ax_cm.set_xlabel("Predicted",fontsize=12)
        ax_cm.set_ylabel("True",fontsize=12)
        ticks = list(range(len(label_map)))
        tick_names = [label_map[i] for i in ticks]
        ax_cm.set_xticks(ticks)
        ax_cm.set_yticks(ticks)
        ax_cm.set_xticklabels(tick_names, rotation=45, ha='right',fontsize=10)
        ax_cm.set_yticklabels(tick_names,fontsize=10)
        # annotate
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, cm[i, j], ha='center', va='center', fontsize=9, color='w' if cm[i, j] > cm.max()/2 else 'black')
        fig_cm.colorbar(im, ax=ax_cm)
        st.pyplot(fig_cm)






