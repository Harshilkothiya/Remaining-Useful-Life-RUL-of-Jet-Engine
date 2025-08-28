import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import tensorflow as tf  # ADDED: Import TensorFlow
import warnings

# --- Local Imports ---
import database
import random_num

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# --- Feature Columns (No Change) ---
feature_columns = [
    'set1', 'set2', 'sensor2', 'sensor3', 'sensor4', 'sensor6', 'sensor7',
    'sensor8', 'sensor9', 'sensor11', 'sensor12', 'sensor13', 'sensor14',
    'sensor15', 'sensor17', 'sensor20', 'sensor21'
]

# --- Model Loading (CHANGED) ---
# Use Streamlit's caching to load models only once.
@st.cache_resource
def load_models():
    """Loads the Keras model and the scikit-learn scaler."""
    # Load the Keras model using TensorFlow's recommended function
    model = joblib.load('model.pkl')
    
    # The scaler is not a TF model, so joblib is still correct for it
    with open('scaler.pkl', 'rb') as f:
        scaler = joblib.load(f)
        
    print("Successfully loaded the model and scaler.")
    return model, scaler

# --- Prediction Function (No Change in Logic) ---
def predict_rul(model, scaler, data):
    # 1. Scale the data
    data_scaled = scaler.transform(data)

    # 2. Reshape for the model (e.g., for LSTM: [samples, timesteps, features])
    data_reshaped = data_scaled.reshape(1, data_scaled.shape[0], data_scaled.shape[1])

    # 3. Predict
    prediction = model.predict(data_reshaped)
    
    print(f"Successfully predicted the value: {prediction[0][0]}")
    return prediction[0][0]

# --- Main Application Logic ---
def main():
    st.title("Jet Engine RUL: Real-time Prediction")

    # Load the model and scaler
    model, scaler = load_models()

    # --- Session State Initialization (No Change) ---
    if 'run' not in st.session_state:
        st.session_state.run = False
    if 'time_points' not in st.session_state:
        st.session_state.time_points = []
    if 'predicted_rul_list' not in st.session_state:
        st.session_state.predicted_rul_list = []
    if 'iteration' not in st.session_state:
        st.session_state.iteration = 1

    # --- UI Controls (No Change) ---
    col1, col2 = st.columns(2)
    if col1.button("Start Prediction", type="primary"):
        st.session_state.run = True
    if col2.button("Stop Prediction"):
        st.session_state.run = False

    # --- Placeholders (No Change) ---
    rul_metric_placeholder = st.empty()
    chart_placeholder = st.empty()

    # --- Main Loop (CHANGED: Pass model and scaler to predict_rul) ---
    while st.session_state.run:
        # 1. Generate new data and store it
        sensor_data = random_num.generate_sensor_data()
        database.insert_sensor_data(sensor_data)
        
        # 2. Fetch the time-series window (last 10 records)
        df_data = database.fetch_last_10()
        df_data = database.pad_data(df_data, 10)
        
        # 3. Make a prediction
        prediction = predict_rul(model, scaler, df_data[feature_columns])
        
        # 4. Update session state for charting
        st.session_state.time_points.append(st.session_state.iteration)
        st.session_state.predicted_rul_list.append(prediction)
        st.session_state.iteration += 1
        
        # Create a DataFrame for the line chart
        df_chart = pd.DataFrame({
            "Time Step": st.session_state.time_points,
            "Predicted RUL": st.session_state.predicted_rul_list
        })

        # 5. Display the latest prediction and update the chart
        rul_metric_placeholder.metric(label="Predicted Remaining Useful Life (RUL)", value=f"{prediction:.2f} cycles")
        chart_placeholder.line_chart(df_chart.set_index("Time Step"))
        
        time.sleep(1) # Pause for 1 second
    else:
        st.write("Press the 'Start Prediction' button to begin.")

if __name__ == '__main__':
    main()