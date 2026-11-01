# Penguin Species Predictor - MLOps Experiment 8

End-to-end MLOps project on a small ML model: a Random Forest that predicts a
penguin's species (Adelie, Chinstrap or Gentoo) from its measurements, served
as a Streamlit web app.

**Krish Sachdev - E029 - BTI DS Sem 11 (E1 batch)**

## Pipeline

| Stage | Tool | Files |
|---|---|---|
| Model | scikit-learn Random Forest | `train.py`, `data/penguins.csv` |
| Version control | Git + GitHub | this repo |
| CI | GitHub Actions (train, test, Docker build, smoke test) | `.github/workflows/ci.yml`, `test_model.py` |
| Container | Docker | `Dockerfile` |
| Orchestration | Kubernetes (Docker Desktop) | `k8s/deployment.yaml`, `k8s/service.yaml` |
| Monitoring | Prometheus + Grafana | `docker-compose.yml`, `monitoring/` |
| Deployment | Streamlit Community Cloud | `app.py`, `requirements.txt` |

## Run locally

```bash
pip install -r requirements.txt
python train.py
pytest -v
streamlit run app.py
```

## Docker

```bash
docker build -t penguin-app:E029 .
docker run -d -p 8501:8501 -p 8001:8001 --name penguin-app penguin-app:E029
```

## Kubernetes

```bash
kubectl apply -f k8s/
kubectl get pods
kubectl port-forward service/penguin-service 8502:8501
```

## Monitoring

```bash
docker compose up -d --build
```

- App: http://localhost:8501
- Metrics: http://localhost:8001/metrics
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001 (admin / admin)

Metrics exposed by the app: `penguin_predictions_total` (by species),
`penguin_prediction_latency_seconds` and `penguin_prediction_confidence`.

## Dataset

Palmer Penguins (Gorman, Williams and Fraser, 2014), from the
`allisonhorst/palmerpenguins` repository. 344 penguins, 333 after dropping rows
with missing values.
