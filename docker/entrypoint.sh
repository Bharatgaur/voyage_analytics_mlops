#!/bin/bash
# ============================================================
# entrypoint.sh — Docker container startup script
# Controls whether to start Flask API or Streamlit app
# ============================================================

set -e

echo "=============================================="
echo " Travel ML Project — Container Starting"
echo " APP_MODE = ${APP_MODE:-api}"
echo "=============================================="

case "${APP_MODE}" in
    api)
        echo "Starting Flask REST API on port ${PORT:-5000} …"
        exec gunicorn \
            --workers 4 \
            --bind "0.0.0.0:${PORT:-5000}" \
            --timeout 120 \
            --access-logfile - \
            --error-logfile - \
            api.app:app
        ;;
    streamlit)
        echo "Starting Streamlit Web App on port 8501 …"
        exec streamlit run streamlit_app/app.py \
            --server.port=8501 \
            --server.address=0.0.0.0 \
            --server.headless=true \
            --server.fileWatcherType=none
        ;;
    train)
        echo "Running model training pipeline …"
        exec python src/model_training.py
        ;;
    *)
        echo "ERROR: Unknown APP_MODE '${APP_MODE}'. Use: api | streamlit | train"
        exit 1
        ;;
esac
