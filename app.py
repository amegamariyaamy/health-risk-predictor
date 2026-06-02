import streamlit as st
import pandas as pd
import pickle

st.title("🏥 Public Health Risk Predictor")
st.write("Enter your details below to estimate your health risk score.")

# 1. Load the models
with open('linear_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)
with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

# 2. Create the input choices for the user
model_choice = st.radio("Choose Model:", ("Linear Regression", "KNN"))
age = st.slider("Your Age", 1, 100, 45)
sleep = st.slider("Daily Sleep Hours", 0, 24, 7)

# 3. Predict button logic
if st.button("Calculate Risk Score"):
    # Format the inputs to match how the model was trained
    features = pd.DataFrame([[age, sleep]], columns=['Age', 'Sleep_Hours'])
    
    if model_choice == "Linear Regression":
        prediction = lr_model.predict(features)[0]
    else:
        prediction = knn_model.predict(features)[0]
        
    # Keep score inside a realistic 0% - 100% boundary
    prediction = max(0.0, min(100.0, prediction))
    
    # 4. Show the results with colored warnings
    st.subheader(f"Predicted Risk Score: {prediction:.1f}%")
    
    if prediction < 40:
        st.success("✅ Low Risk Profile")
    elif 40 <= prediction < 70:
        st.warning("⚠️ Moderate Risk Profile")
    else:
        st.error("🚨 High Risk Profile")