import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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
        df['Soil_Label'] = df['Lithology'].astype('category').cat.codes
        features = ['X', 'Y', 'TE', 'BE']
        X = df[features]
        y = df['Soil_Label']
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
        
        st.subheader("⚙️ Bayesian Optimization Parameters")
        n_init = st.number_input("Initial points (init_points)", min_value=1, max_value=20, value=5)
        n_iter = st.number_input("Iterations (n_iter)", min_value=1, max_value=50, value=10)
        
        # Define Bayesian Optimization function
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
        
        # Define parameter bounds
        pbounds = {
            'n_estimators': (10,20),
            'max_depth': (10, 30),
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 5),
        }
        optimizer = BayesianOptimization(f=rf_cv, pbounds=pbounds, random_state=1, verbose=2)
        optimizer.maximize(init_points= int(n_init), n_iter=int(n_iter))
        
        # Best parameters
        best_params = optimizer.max['params']
        # best_params = {k: int(v) for k, v in best_params.items()}
        best_params = {k: int(v) if v is not None else None for k, v in best_params.items()}
        st.subheader("Best Hyperparameters from Bayesian Optimization:")
        for key, value in best_params.items():
            st.write(f"  {key}: {value}")
        
        # Final model
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
        
        # Evaluate on training set
        y_train_pred = rf.predict(X_train)
        y_train_proba = rf.predict_proba(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_loss = log_loss(y_train, y_train_proba)
        st.write("### Training accuracy:", f"{train_acc:.4f}")
                       
        # Evaluate on test set
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_proba)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        st.write("### Testing accuracy:", f"{acc:.4f}")
        







