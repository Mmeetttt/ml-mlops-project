import streamlit as st
import joblib
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
import requests

# Prometheus Metrics
PREDICTION_COUNTER = Counter('ml_predictions_total', 'Total predictions made', ['status'])
PREDICTION_LATENCY = Histogram('ml_prediction_latency_seconds', 'Prediction latency')
ERROR_COUNTER = Counter('ml_errors_total', 'Total errors', ['error_type'])

# Load model
@st.cache_resource
def load_model():
    return joblib.load('models/iris_model.pkl')

model = load_model()

# Streamlit UI
st.title(" Iris Flower Classification")
st.markdown("### Enter flower measurements:")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)

with col2:
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.3)

if st.button("Predict", type="primary"):
    start_time = time.time()
    try:
        # Prepare input
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0]
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Update metrics
        PREDICTION_COUNTER.labels(status='success').inc()
        PREDICTION_LATENCY.observe(latency)
        
        # Display results
        iris_classes = ['Setosa', 'Versicolor', 'Virginica']
        predicted_class = iris_classes[prediction[0]]
        
        st.success(f"### Predicted Class: {predicted_class}")
        
        st.markdown("### Prediction Probabilities:")
        prob_df = pd.DataFrame({
            'Class': iris_classes,
            'Probability': probability
        })
        st.bar_chart(prob_df.set_index('Class'))
        
        st.info(f"Prediction Time: {latency:.4f} seconds")
        
    except Exception as e:
        latency = time.time() - start_time
        PREDICTION_COUNTER.labels(status='error').inc()
        ERROR_COUNTER.labels(error_type='prediction_error').inc()
        st.error(f"Error: {str(e)}")

# Metrics endpoint
st.sidebar.markdown("### Monitoring")
if st.sidebar.button("View Metrics"):
    st.sidebar.code(generate_latest().decode('utf-8'))