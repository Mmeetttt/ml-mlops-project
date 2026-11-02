import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, start_http_server, REGISTRY
import threading
from sklearn.datasets import load_iris
import os

# Use a custom registry to avoid duplicates
if 'registry' not in st.session_state:
    from prometheus_client import CollectorRegistry
    st.session_state.registry = CollectorRegistry()

# Initialize metrics only once
if 'metrics_initialized' not in st.session_state:
    st.session_state.metrics_initialized = True
    
    # Create metrics with the custom registry
    st.session_state.prediction_counter = Counter(
        'ml_predictions_total', 
        'Total predictions made', 
        ['status'],
        registry=st.session_state.registry
    )
    
    st.session_state.prediction_latency = Histogram(
        'ml_prediction_latency_seconds', 
        'Prediction latency',
        registry=st.session_state.registry
    )
    
    st.session_state.error_counter = Counter(
        'ml_errors_total', 
        'Total errors', 
        ['error_type'],
        registry=st.session_state.registry
    )

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load('models/iris_model.pkl')
    except:
        # If model doesn't exist, train it
        from sklearn.ensemble import RandomForestClassifier
        iris = load_iris()
        X, y = iris.data, iris.target
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model

model = load_model()

# Start Prometheus metrics server in a separate thread (only once)
def start_metrics_server():
    try:
        start_http_server(8000, registry=st.session_state.registry)
    except:
        pass  # Server might already be running

if 'metrics_server_started' not in st.session_state:
    start_metrics_server()
    st.session_state.metrics_server_started = True

# Streamlit UI
st.title("🌸 Iris Flower Classification")
st.markdown("### Enter flower measurements:")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)

with col2:
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.3, step=0.1)

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
        st.session_state.prediction_counter.labels(status='success').inc()
        st.session_state.prediction_latency.observe(latency)
        
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
        st.session_state.prediction_counter.labels(status='error').inc()
        st.session_state.error_counter.labels(error_type='prediction_error').inc()
        st.error(f"Error: {str(e)}")

# Display metrics info in sidebar
st.sidebar.markdown("### 📊 Monitoring")
st.sidebar.success("Metrics exposed on port 8000")
st.sidebar.markdown("Make predictions to see metrics update!")

# Simple endpoint to expose metrics
if st.sidebar.button("View Raw Metrics"):
    try:
        metrics_data = generate_latest(st.session_state.registry).decode('utf-8')
        st.sidebar.code(metrics_data[:1000])
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")