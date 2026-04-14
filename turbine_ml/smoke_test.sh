#!/bin/bash
# Smoke Test Script for Turbine ML Engine
# Tests the complete pipeline: simulate -> analyze -> validate

set -e

echo "=============================================="
echo "     TURBINE ML ENGINE - SMOKE TEST"
echo "=============================================="
echo ""

cd "$(dirname "$0")"

# 1. Generate fresh test data
echo "Step 1: Generating synthetic test data..."
python data_generator/generate_telemetry.py \
    --output /tmp/smoke_test_data \
    --n-healthy 5 \
    --n-faulty 5 \
    --duration 2 \
    --seed 12345
echo "✓ Data generated"
echo ""

# 2. Check if models exist
if [ ! -f "saved_models/fault_classifier.joblib" ]; then
    echo "Step 2: Training models (first time)..."
    python training/train_all.py \
        --data-dir /tmp/smoke_test_data \
        --output-dir /tmp/smoke_test_models
    MODELS_DIR=/tmp/smoke_test_models
else
    echo "Step 2: Using existing trained models"
    MODELS_DIR=saved_models
fi
echo "✓ Models ready"
echo ""

# 3. Run inference on healthy file
echo "Step 3: Analyzing healthy turbine..."
python inference/analyze.py \
    --models-dir $MODELS_DIR \
    --csv /tmp/smoke_test_data/train/healthy_000.csv 2>/dev/null | grep -E "Detected|Fault Type"
echo "✓ Healthy analysis complete"
echo ""

# 4. Run inference on faulty file
echo "Step 4: Analyzing faulty turbine..."
FAULTY_FILE=$(find /tmp/smoke_test_data/train -name "cyclic_vibration_*" | head -1)
if [ -z "$FAULTY_FILE" ]; then
    FAULTY_FILE=$(find /tmp/smoke_test_data/train -name "erosion_*" | head -1)
fi
python inference/analyze.py \
    --models-dir $MODELS_DIR \
    --csv "$FAULTY_FILE" 2>/dev/null | grep -E "Detected|Fault Type|Blade|Severity"
echo "✓ Faulty analysis complete"
echo ""

# 5. Cleanup
rm -rf /tmp/smoke_test_data /tmp/smoke_test_models 2>/dev/null || true

echo "=============================================="
echo "         SMOKE TEST PASSED ✓"
echo "=============================================="
