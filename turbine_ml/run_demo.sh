#!/bin/bash
# Run Demo Script for Turbine ML Engine
# For hackathon presentation

set -e

echo "=============================================="
echo "     TURBINE ML ENGINE - DEMO MODE"
echo "=============================================="
echo ""

# Navigate to turbine_ml directory
cd "$(dirname "$0")"

# Check if models exist
if [ ! -f "saved_models/fault_classifier.joblib" ]; then
    echo "ERROR: Models not found. Run training first:"
    echo "  python training/train_all.py"
    exit 1
fi

echo "1. Testing on HEALTHY turbine data..."
echo "----------------------------------------------"
python inference/analyze.py --models-dir saved_models --csv examples/healthy.csv
echo ""

echo "2. Testing on FAULTY turbine data (Cyclic Vibration)..."
echo "----------------------------------------------"
python inference/analyze.py --models-dir saved_models --csv examples/demo_test.csv
echo ""

echo "=============================================="
echo "              DEMO COMPLETE"
echo "=============================================="
echo ""
echo "To start the HTTP API service:"
echo "  MODELS_DIR=./saved_models python api/ml_service.py"
echo ""
echo "Then test with:"
echo "  curl -X POST http://localhost:8000/analyze -F file=@examples/demo_test.csv"
