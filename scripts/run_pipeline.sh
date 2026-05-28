#!/usr/bin/env bash
# Full ADAS Sound Detector pipeline — Steps 1 through 5
# Usage: bash scripts/run_pipeline.sh [config_path]

set -euo pipefail

CONFIG="${1:-config.yaml}"
echo "=== ADAS Sound Detector Pipeline ==="
echo "Config: $CONFIG"
echo ""

echo "[1/5] Data Pipeline"
python -m src.data_pipeline --config "$CONFIG"

echo "[2/5] Embedding Extraction"
python -m src.embedding --config "$CONFIG"

echo "[3/5] Classifier Training"
python -m src.classifier --config "$CONFIG"

echo "[4/5] Model Export (TFLite INT8)"
python -m src.model_export --config "$CONFIG"

echo ""
echo "=== Pipeline complete. Run inference with: ==="
echo "  python -m src.realtime_infer --config $CONFIG"
echo "  python -m src.realtime_infer --config $CONFIG --wav <path>.wav"
echo "  python -m src.realtime_infer --config $CONFIG --benchmark"
