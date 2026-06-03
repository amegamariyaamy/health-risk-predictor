import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Public Health Risk Predictor", layout="centered")
st.title("🏥 Public Health Risk Predictor")
st.write("Enter your details below to estimate your health risk score based on Age, Sleep, and Stress.")

# 1. Load the updated models
with open('linear_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)
with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

# 2. Setup Sidebar Options and Main Input Sliders
model_choice = st.sidebar.radio("Choose Model:", ("Linear Regression", "KNN"))

st.header("📋 Input Your Metrics")
age = st.slider("Your Age (Years)", 18, 85, 45)
sleep = st.slider("Daily Sleep Hours", 4.0, 10.0, 7.0, 0.5)
stress = st.slider("Stress Level (Scale of 1 to 10)", 1.0, 10.0, 5.0, 0.5)

# 3. Predict button logic
if st.button("🔮 Calculate Risk Score", use_container_width=True):
    # Construct the dataframe matching the EXACT column names and order from training
    input_data = pd.DataFrame([[age, sleep, stress]], columns=['Age', 'Sleep_Hours', 'Stress_Level'])
    
    # Generate prediction based on chosen model
    if model_choice == "Linear Regression":
        prediction = lr_model.predict(input_data)[0]
    else:
        prediction = knn_model.predict(input_data)[0]
        
    # Boundary restriction (keep score between 0% and 100%)
    prediction = max(0.0, min(100.0, prediction))
    
    # 4. Display the results with color brackets
    st.markdown("---")
    st.subheader(f"Predicted Risk Score: {prediction:.1f}%")
    
    if prediction < 45:
        st.success("✅ Low Risk Profile")
    elif 45 <= prediction < 70:
        st.warning("⚠️ Moderate Risk Profile")
    else:
        st.error("🚨 High Risk Profile")