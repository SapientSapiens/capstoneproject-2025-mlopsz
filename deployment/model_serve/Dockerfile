# Dockerfile inside deployment/model_serve
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y libgomp1

WORKDIR /app

COPY deployment/model_serve/predict.py .
COPY deployment/deployed_models/ ./deployed_models/
COPY deployment/model_serve/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8010

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8010"]