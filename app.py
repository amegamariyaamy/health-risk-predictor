import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Public Health Predictor", layout="centered")
st.title("🏥 Full Public Health Risk Predictor")
st.write("Provide your metrics below to get an accurate, dynamic health risk evaluation.")

# Load your full models
with open('linear_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)
with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

model_choice = st.sidebar.radio("Choose Model:", ("Linear Regression", "KNN"))

# Create input widgets for all data variables
st.header("📋 Enter Metrics")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (Years)", 18, 85, 50)
    sleep = st.slider("Daily Sleep Hours", 4.0, 10.0, 7.0, 0.1)
    stress = st.slider("Stress Level (1-10)", 1.0, 10.0, 5.0, 0.1)
    bmi = st.slider("Body Mass Index (BMI)", 15.0, 40.0, 25.0, 0.1)

with col2:
    oxygen = st.slider("Oxygen Level (%)", 85.0, 100.0, 95.0, 0.1)
    pollution = st.slider("Pollution Exposure Level", 10.0, 100.0, 50.0, 0.1)
    activity = st.slider("Activity Level", 1.0, 20.0, 10.0, 0.1)
    heart_rate = st.slider("Heart Rate (BPM)", 50, 110, 72)

if st.button("🔮 Calculate Risk Score", use_container_width=True):
    # Construct the exact order of features your model was trained on
    input_data = pd.DataFrame([[
        oxygen, pollution, stress, activity, sleep, age, bmi, heart_rate
    ]], columns=['Oxygen_Level', 'Pollution_Level', 'Stress_Level', 'Activity_Level', 'Sleep_Hours', 'Age', 'Body_Mass_Index', 'Heart_Rate_BPM'])
    
    if model_choice == "Linear Regression":
        prediction = lr_model.predict(input_data)[0]
    else:
        prediction = knn_model.predict(input_data)[0]
        
    prediction = max(0.0, min(100.0, prediction))
    
    st.markdown("---")
    st.subheader(f"Predicted Risk Score: {prediction:.1f}%")
    if prediction < 45:
        st.success("✅ Low Risk Profile")
    elif 45 <= prediction < 70:
        st.warning("⚠️ Moderate Risk Profile")
    else:
        st.error("🚨 High Risk Profile")