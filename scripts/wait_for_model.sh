#!/bin/bash

MODEL_PATH="/app/models/traffy_rf_model.joblib"
MAX_WAIT=1200  # 20 minutes
ELAPSED=0
CHECK_INTERVAL=10

echo "============================================"
echo "Waiting for ML model to be trained..."
echo "Model path: ${MODEL_PATH}"
echo "============================================"
echo ""

while [ ! -f "$MODEL_PATH" ]; do
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo ""
        echo "============================================"
        echo "ERROR: Model not found after ${MAX_WAIT} seconds"
        echo "============================================"
        echo ""
        echo "Please check:"
        echo "  1. Airflow UI: http://localhost:8080"
        echo "  2. Check DAG status: traffy_model_pipeline"
        echo "  3. Check Airflow logs: docker-compose logs airflow-scheduler"
        echo ""
        echo "The training pipeline should:"
        echo "  - Run automatically on first startup"
        echo "  - Take 10-15 minutes to complete"
        echo "  - Create model at: ${MODEL_PATH}"
        echo ""
        exit 1
    fi
    
    MINUTES=$((ELAPSED / 60))
    SECONDS=$((ELAPSED % 60))
    echo "[${MINUTES}m ${SECONDS}s] Model not ready yet... (checking every ${CHECK_INTERVAL}s)"
    sleep $CHECK_INTERVAL
    ELAPSED=$((ELAPSED + CHECK_INTERVAL))
done

echo ""
echo "============================================"
echo "Model found! Starting service..."
echo "============================================"
echo ""

# Execute the main command
exec "$@"
