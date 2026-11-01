# app.py - Streamlit web app for the penguin species model
import time
import joblib
import pandas as pd
import streamlit as st
from prometheus_client import Counter, Histogram, start_http_server

st.set_page_config(page_title="Penguin Species Predictor", page_icon="🐧")


# Load the trained model only once
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


# Start the Prometheus metrics server only once (http://localhost:8001/metrics)
@st.cache_resource
def start_metrics():
    try:
        start_http_server(8001)
    except OSError:
        pass  # port already in use, the app still works without metrics
    return {
        "predictions": Counter("penguin_predictions_total",
                               "Number of predictions made", ["species"]),
        "latency": Histogram("penguin_prediction_latency_seconds",
                             "Time taken for one prediction"),
        "confidence": Histogram("penguin_prediction_confidence",
                                "Model confidence of each prediction",
                                buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    }


saved = load_model()
model = saved["model"]
metrics = start_metrics()

st.title("🐧 Penguin Species Predictor")
st.write("Enter a penguin's measurements and the Random Forest model "
         "predicts its species: **Adelie**, **Chinstrap** or **Gentoo**.")

# Input form
col1, col2 = st.columns(2)
with col1:
    island = st.selectbox("Island", ["Biscoe", "Dream", "Torgersen"])
    bill_length = st.slider("Bill length (mm)", 30.0, 60.0, 44.0, step=0.1)
    flipper_length = st.slider("Flipper length (mm)", 170, 235, 200, step=1)
with col2:
    sex = st.selectbox("Sex", ["male", "female"])
    bill_depth = st.slider("Bill depth (mm)", 13.0, 22.0, 17.0, step=0.1)
    body_mass = st.slider("Body mass (g)", 2700, 6300, 4200, step=50)

if st.button("Predict species", type="primary"):
    start = time.time()

    # Build one row in the same format the model was trained on
    row = pd.DataFrame([{
        "island": island, "bill_length_mm": bill_length,
        "bill_depth_mm": bill_depth, "flipper_length_mm": flipper_length,
        "body_mass_g": body_mass, "sex": sex,
    }])
    row = pd.get_dummies(row, columns=["island", "sex"])
    row = row.reindex(columns=saved["columns"], fill_value=0)

    species = model.predict(row)[0]
    confidence = model.predict_proba(row).max()

    # Record the metrics for Prometheus
    metrics["latency"].observe(time.time() - start)
    metrics["predictions"].labels(species=species).inc()
    metrics["confidence"].observe(confidence)

    st.success(f"Predicted species: **{species}**")
    st.write(f"Model confidence: {confidence:.0%}")

st.caption(f"Model: Random Forest (100 trees) trained on the Palmer Penguins "
           f"dataset, test accuracy {saved['accuracy']:.1%}")
