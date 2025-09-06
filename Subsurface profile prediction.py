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

        st.subheader("Best Hyperparameters")
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
        ax_cm.set_title("Confusion Matrix",fontsize=10)
        ax_cm.set_xlabel("Predicted",fontsize=9)
        ax_cm.set_ylabel("True",fontsize=9)
        ticks = list(range(len(label_map)))
        tick_names = [label_map[i] for i in ticks]
        ax_cm.set_xticks(ticks)
        ax_cm.set_yticks(ticks)
        ax_cm.set_xticklabels(tick_names, rotation=45, ha='right',fontsize=9)
        ax_cm.set_yticklabels(tick_names,fontsize=9)
        # annotate
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, cm[i, j], ha='center', va='center', fontsize=8, color='w' if cm[i, j] > cm.max()/2 else 'black')
        fig_cm.colorbar(im, ax=ax_cm)
        st.pyplot(fig_cm)

        st.subheader("🌐 Predict 2D Cross-Section Between Two Boreholes")

        col1, col2 = st.columns(2)
        with col1:
            X1 = st.number_input("Borehole A: X", value=float(df["X"].min()))
            Y1 = st.number_input("Borehole A: Y", value=float(df["Y"].min()))
            TE1 = st.number_input("Borehole A: Top Elevation (TE)", value=float(df["TE"].max()))
            BE1 = st.number_input("Borehole A: Bottom Elevation (BE)", value=float(df["BE"].min()))
        
        with col2:
            X2 = st.number_input("Borehole B: X", value=float(df["X"].max()))
            Y2 = st.number_input("Borehole B: Y", value=float(df["Y"].max()))
            TE2 = st.number_input("Borehole B: Top Elevation (TE)", value=float(df["TE"].max()))
            BE2 = st.number_input("Borehole B: Bottom Elevation (BE)", value=float(df["BE"].min()))
        
        # Number of points along the cross-section
        n_points = st.number_input("Number of interpolated points along line", value=20, min_value=2)
        
        if st.button("Predict Cross-Section"):
            # Linear interpolation of coordinates and top/bottom elevations
            X_line = np.linspace(X1, X2, n_points)
            Y_line = np.linspace(Y1, Y2, n_points)
            TE_line = np.linspace(TE1, TE2, n_points)
            BE_line = np.linspace(BE1, BE2, n_points)
        
            # Depth step
            step = 1.0
            all_segments = []
        
            for xi, yi, te_i, be_i in zip(X_line, Y_line, TE_line, BE_line):
                depths = np.arange(max(te_i, be_i), min(te_i, be_i), -step)
                for j in range(len(depths)-1):
                    seg_te = depths[j]
                    seg_be = depths[j+1]
                    X_new = pd.DataFrame([[xi, yi, seg_te, seg_be]], columns=['X','Y','TE','BE'])
                    pred_label = rf.predict(X_new)[0]
                    pred_lith = label_map[pred_label]
                    all_segments.append({'X': xi, 'Y': yi, 'Midpoint': (seg_te+seg_be)/2, 'Lithology': pred_lith})
        
            df_cross = pd.DataFrame(all_segments)
        
            # Plot cross-section
            fig, ax = plt.subplots(figsize=(12,6))
            colors = plt.cm.tab20.colors
            unique_lith = df_cross['Lithology'].unique()
            color_map = {lith: colors[i % len(colors)] for i, lith in enumerate(unique_lith)}
        
            for lith in unique_lith:
                subset = df_cross[df_cross['Lithology']==lith]
                ax.scatter(subset['X'], subset['Midpoint'], color=color_map[lith], s=50, label=lith)
        
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Elevation (m)")
            ax.set_title("Predicted Cross-Section Lithology")
            ax.invert_yaxis()
            ax.legend(title="Lithology", bbox_to_anchor=(1.05,1), loc='upper left')
            st.pyplot(fig)

















