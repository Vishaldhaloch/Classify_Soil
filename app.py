import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Helper function: preprocessing & training the model
@st.cache_data
def load_and_train():
    # Load datasets
    df_chem = pd.read_csv("data/LUCAS-SOIL-2018.csv")
    df_texture = pd.read_csv("data/LUCAS_Text_All_10032025.csv")

    # Filter surface soil chemistry data
    df_chem_surface = df_chem[df_chem['Depth'] == '0-20 cm']
    chem_cols_to_keep = ['POINTID', 'pH_H2O', 'EC', 'OC', 'P', 'N', 'K']
    df_chem_surface = df_chem_surface[chem_cols_to_keep]

    # Keep relevant texture columns
    texture_cols_to_keep = ['POINTID', 'Clay', 'Sand', 'Silt', 'USDA']
    df_texture = df_texture[texture_cols_to_keep]

    # Merge on POINTID
    df_merged = pd.merge(df_texture, df_chem_surface, on='POINTID', how='inner')

    # Drop rows with missing texture and important chemistry columns
    cols_to_check = ['Clay', 'Sand', 'Silt', 'USDA', 'EC', 'P']
    df_cleaned = df_merged.dropna(subset=cols_to_check)

    # Replace '< LOD' with half of min detected value
    lod_cols = ['OC', 'P', 'N', 'K']
    for col in lod_cols:
        df_cleaned[col] = df_cleaned[col].replace('< LOD', np.nan)
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        min_val = df_cleaned[col].min(skipna=True)
        lod_proxy = min_val / 2 if min_val > 0 else 0
        df_cleaned[col] = df_cleaned[col].fillna(lod_proxy)
        df_cleaned.loc[df_cleaned[col] == 0, col] = lod_proxy

    # Drop highly correlated and high VIF features: 'Silt', 'N', 'pH_H2O'
    df_cleaned = df_cleaned.drop(columns=['Silt', 'N', 'pH_H2O'])

    # Group USDA soil types into binary target
    clayey_classes = ['clay', 'clay loam', 'silty clay', 'sandy clay', 'sandy clay loam', 'silty clay loam']
    sandy_classes = ['sand', 'loamy sand', 'sandy loam']

    def group_soil_type(usda_class):
        if usda_class.lower() in clayey_classes:
            return 'Clayey'
        elif usda_class.lower() in sandy_classes:
            return 'Sandy'
        else:
            return 'Other'

    df_cleaned['soil_group'] = df_cleaned['USDA'].apply(group_soil_type)

    # Filter only Clayey and Sandy groups
    df_binary = df_cleaned[df_cleaned['soil_group'].isin(['Clayey', 'Sandy'])]

    # Prepare features and target
    df_binary1 = df_binary.drop(columns=['USDA', 'POINTID'])
    X = df_binary1.drop(columns=['soil_group'])
    y = df_binary1['soil_group']

    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    return svm_model, scaler

# Load model and scaler once
svm_model, scaler = load_and_train()

st.title("Soil Texture Classification (Clayey vs Sandy)")
st.write("""
This app predicts if a soil sample is **Clayey** or **Sandy** based on soil features.
""")

# User input form
with st.form(key='input_form'):
    Clay = st.number_input("Clay (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    Sand = st.number_input("Sand (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    EC = st.number_input("Electrical Conductivity (EC)", min_value=0.0, value=20.0, step=0.1)
    OC = st.number_input("Organic Carbon (OC)", min_value=0.0, value=10.0, step=0.1)
    P = st.number_input("Phosphorus (P)", min_value=0.0, value=5.0, step=0.1)
    K = st.number_input("Potassium (K)", min_value=0.0, value=500.0, step=1.0)

    submit_button = st.form_submit_button(label='Predict Soil Group')

if submit_button:
    # Prepare input for prediction
    sample = np.array([[Clay, Sand, EC, OC, P, K]])
    sample_scaled = scaler.transform(sample)
    prediction = svm_model.predict(sample_scaled)[0]

    st.success(f"Predicted Soil Group: **{prediction}**")

    st.write("### Input Features:")
    st.write(f"- Clay: {Clay}%")
    st.write(f"- Sand: {Sand}%")
    st.write(f"- EC: {EC}")
    st.write(f"- OC: {OC}")
    st.write(f"- P: {P}")
    st.write(f"- K: {K}")
