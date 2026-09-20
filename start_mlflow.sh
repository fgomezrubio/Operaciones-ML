#!/bin/bash

# Obtener el directorio donde se encuentra este script
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

# Base central de MLflow
DB_PATH="$BASE_DIR/mlflow_tracking/mlflow.db"
TRACKING_URI="sqlite:///$DB_PATH"

echo "========================================"
echo " MLflow - Operaciones ML"
echo "========================================"
echo "Tracking Store:"
echo "$DB_PATH"
echo ""
echo "MLflow UI:"
echo "http://127.0.0.1:5000"
echo ""
echo "Presiona Ctrl+C para detener MLflow"
echo "========================================"

mlflow ui \
    --backend-store-uri "$TRACKING_URI" \
    --host 127.0.0.1 \
    --port 5000
