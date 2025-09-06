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
       
        # Train Random Forest
        model = RandomForestClassifier(n_estimators=200, random_state=1)
        model.fit(X_train, y_train)
        
        # Evaluate on training set
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_loss = log_loss(y_train, y_train_proba)
        st.write("### Training accuracy:", train_acc)
                       
        # Evaluate on test set
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_proba)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        st.write("### Testing accuracy:", acc)




