#!/bin/bash
cd /home/ubuntu/capstoneproject-2025-mlopsz

mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root s3://mlops-zoomcamp-bike-sharing-bucket/mlflow_artifacts \
    --host 0.0.0.0 \
    --port 5000

