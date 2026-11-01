FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py model.pkl ./

# 8501 = Streamlit web app, 8001 = Prometheus metrics
EXPOSE 8501 8001
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
