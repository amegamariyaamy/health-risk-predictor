import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Application Title and Description
st.set_page_config(page_title="Health Risk Predictor", layout="centered")
st.title("🏥 Public Health Risk Score Predictor")
st.write("Input your age and daily sleep hours below to predict your health risk score using optimized Machine Learning models.")

# Load the saved models with caching to make the app fast
@st.cache_resource
def load_models():
    with open('linear_model.pkl', 'rb') as f:
        lr = pickle.load(f)
    with open('knn_model.pkl', 'rb') as f:
        knn = pickle.load(f)
    return lr, knn

try:
    lr_model, knn_model = load_models()
except FileNotFoundError:
    st.error("Error: Model files not found. Make sure 'linear_model.pkl' and 'knn_model.pkl' are in your GitHub repository.")
    st.stop()

# Sidebar Layout for configuration
st.sidebar.header("🛠️ Model Settings")
model_choice = st.sidebar.radio(
    "Select Regression Model:",
    ("Linear Regression", "K-Nearest Neighbors (KNN)")
)

# Main Form Layout for inputs
st.header("👤 Enter User Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (Years)", min_value=1.0, max_value=100.0, value=45.0, step=0.1)

with col2:
    sleep_hours = st.slider("Daily Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.1)

# Format the inputs exactly like the training feature dataframes
input_df = pd.DataFrame([[age, sleep_hours]], columns=['Age', 'Sleep_Hours'])

# Predict Button
st.markdown("---")
if st.button("🔮 Predict Risk Score", use_container_width=True):
    if model_choice == "Linear Regression":
        prediction = lr_model.predict(input_df)[0]
    else:
        prediction = knn_model.predict(input_df)[0]
        
    # Boundary constraints handling (clamping scores if they slightly exceed realistic values)
    prediction = max(0.0, min(100.0, prediction))
    
    # Display Result
    st.subheader("Results:")
    st.metric(label=f"Predicted Risk Score ({model_choice})", value=f"{prediction:.2f}%")
    
    if prediction < 40:
        st.success("✅ Low Risk Profile")
    elif 40 <= prediction < 70:
        st.warning("⚠️ Moderate Risk Profile")
    else:
        st.error("🚨 High Risk Profile. Consider reviewing lifestyle routines.")