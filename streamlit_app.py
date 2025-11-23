import pandas as pd
import streamlit as st
import joblib

model = joblib.load("saved_models/stacked_model.pkl")

st.title("🦠 MDR Prediction for a Patient")

# Saisie des caractéristiques
gender = st.selectbox("Gender", ["F", "M"])
strain = st.selectbox("Strain", [
    "Escherichia coli", "Enterobacteria spp.", "Klebsiella pneumoniae", 
    "Proteus mirabilis", "Citrobacter spp.", "Morganella morganii", 
    "Serratia marcescens", "Pseudomonas aeruginosa", "Acinetobacter baumannii"
])
infection_freq_label = st.selectbox("Infection Frequency", ["Never", "Rarely", "Often", "Very Often"])
age = st.number_input("Age", min_value=0, max_value=120, value=30)
ctx_cro_resistant = st.selectbox("CTX/CRO Resistant", [0, 1])
hospital_before = st.selectbox("Hospitalized Before", [0, 1])
diabetes = st.selectbox("Diabetes", [0, 1])
hypertension = st.selectbox("Hypertension", [0, 1])

# Calculate age_comorb
age_comorb = age * (diabetes + hypertension + hospital_before)

# infection_freq mapping
infection_freq_map = {
    "Never": 0,
    "Rarely": 1,
    "Often": 2,
    "Very Often": 3
}
infection_freq = infection_freq_map[infection_freq_label]

# age_bin
if age < 18:
    age_bin = "child"
elif age < 40:
    age_bin = "adult"
elif age < 65:
    age_bin = "senior"
else:
    age_bin = "elderly"

# Create the DataFrame with the exact columns expected by the model
input_data = pd.DataFrame([{
    "gender": gender,
    "strain": strain,
    "infection_freq": infection_freq,
    "age_comorb": age_comorb,
    "age_bin": age_bin,
    "ctx/cro_resistant": ctx_cro_resistant,
}])

if st.button("🔍 Predict MDR Status"):
    try:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]
        
        st.subheader("📊 Results")
        
        # Display prediction
        if prediction == 1:
            st.error(f"⚠️ **Prediction: MDR (Multi-Drug Resistant)**")
        else:
            st.success(f"✅ **Prediction: Non-MDR**")
        
        # Display probability with a progress bar
        st.write(f"**Probability of MDR:** {proba:.2%}")
        st.progress(proba)
        
        # Risk interpretation
        if proba > 0.7:
            st.error(f"🔴 High risk of MDR ({proba:.2%})")
        elif proba > 0.4:
            st.warning(f"🟠 Moderate risk of MDR ({proba:.2%})")
        else:
            st.success(f"🟢 Low risk of MDR ({proba:.2%})")
            
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.write("Please check that the model is correctly trained and saved.")