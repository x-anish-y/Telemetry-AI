# Turbine ML Engine

## Wind Turbine Blade-Level Predictive Maintenance

This ML engine provides real machine learning models for detecting, classifying, and assessing wind turbine blade faults from SCADA telemetry data.

---

## 🎯 ML Objectives

| Objective | Model | Output |
|-----------|-------|--------|
| Anomaly Detection | Isolation Forest | anomaly_score, anomaly_flag |
| Fault Classification | Random Forest | erosion, pitch_drift, cyclic_vibration, normal |
| Blade Attribution | Sector Energy Analysis | blade (1, 2, or 3) + confidence |
| Severity Estimation | Random Forest Regressor | severity (0.0 - 1.0) |
| What-If Forecasting | Gradient Boosting | fix_energy_mwh, nofix_energy_mwh, revenue_delta |

---

## 📁 Project Structure

```
turbine_ml/
├── data_generator/
│   └── generate_telemetry.py    # Synthetic SCADA data generator
├── feature_engineering/
│   └── features.py              # Feature extraction (40+ features)
├── models/
│   ├── anomaly.py              # Isolation Forest anomaly detector
│   ├── classifier.py           # Random Forest fault classifier
│   ├── severity.py             # Severity regression model
│   └── forecasting.py          # What-if forecaster + blade attribution
├── training/
│   └── train_all.py            # Complete training pipeline
├── inference/
│   └── analyze.py              # Inference module
├── api/
│   └── ml_service.py           # FastAPI HTTP endpoint
├── synthetic_data/             # Generated training data
│   ├── train/
│   ├── val/
│   └── test/
├── saved_models/               # Trained model files (.joblib)
├── examples/
│   └── demo_test.csv           # Demo file for presentations
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd turbine_ml
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
python data_generator/generate_telemetry.py --output synthetic_data
```

This generates:
- 30 healthy turbine files
- 20 files per fault type (erosion, pitch_drift, cyclic_vibration)
- Train/Val/Test split (70/15/15)
- `demo_test.csv` for live demos

### 3. Train Models

```bash
python training/train_all.py
```

Output:
- Trained models saved to `saved_models/`
- Metrics printed for each model
- `training_results.json` with all evaluation metrics

### 4. Run ML API

```bash
python api/ml_service.py
```

API available at `http://localhost:8000`

---

## 📊 Data Format

### Input CSV Columns (Required)

| Column | Description | Unit |
|--------|-------------|------|
| timestamp | Time of measurement | ISO datetime |
| wind_speed | Wind velocity | m/s |
| rotor_rpm | Rotor speed | RPM |
| power | Active power output | kW |
| pitch_cmd | Commanded pitch angle | degrees |
| pitch_meas | Measured pitch angle | degrees |
| torque | Generator torque | kNm |
| vibration | Nacelle vibration | m/s² |

### Optional Columns

| Column | Description |
|--------|-------------|
| rotor_phase | Rotor azimuth position (0-360°) |

---

## 🔬 How It Works

### 1. Synthetic Data Generation

The generator creates physics-guided fault signatures:

- **Erosion**: Gradual power curve degradation (5-20% loss)
- **Pitch Drift**: Increasing pitch error over time (2-8°)
- **Cyclic Vibration**: 1P frequency spikes aligned to blade phase

Each fault is parameterized by:
- `fault_type`: Type of fault
- `blade_id`: Affected blade (1, 2, or 3)
- `severity`: Intensity (0.3 - 0.9)
- `start_pct`: When fault begins in the file

### 2. Feature Engineering

40+ features extracted per sliding window:

**Power Residual Features**:
- `power_residual_mean/std/trend`: Deviation from theoretical power curve

**Pitch Error Features**:
- `pitch_error_mean/std/max`: Commanded vs measured pitch deviation

**FFT Vibration Features**:
- `fft_band_X_energy`: Energy in frequency bands (detects 1P excitation)

**Blade Sector Features**:
- `blade_X_energy/ratio`: Vibration energy per 120° rotor sector

### 3. Model Training

| Model | Training Data | Target |
|-------|--------------|--------|
| Anomaly Detector | Healthy only | Outlier detection |
| Fault Classifier | All labeled | Fault type (4 classes) |
| Severity Estimator | Faulty only | Severity (0-1) |
| What-If Forecaster | All | Capacity loss prediction |

### 4. Inference Pipeline

```
CSV Input
    ↓
Feature Extraction (sliding windows)
    ↓
Anomaly Detection → anomaly_count, detected
    ↓
Fault Classification → fault_type, confidence
    ↓
Severity Estimation → severity (0-1)
    ↓
Blade Attribution → blade (1/2/3)
    ↓
What-If Forecast → energy/revenue impact
    ↓
JSON Response
```

---

## 🌐 API Reference

### POST /analyze

Analyze SCADA CSV for faults.

**Request**: `multipart/form-data` with CSV file

**Response**:
```json
{
  "detected": true,
  "anomaly_count": 12,
  "blade": 2,
  "fault_type": "cyclic_vibration",
  "severity": 0.65,
  "confidence": 0.82,
  "residual_series": [-30.1, -28.3, -35.4, ...],
  "what_if_forecast": {
    "fix_energy_mwh": 612.5,
    "nofix_energy_mwh": 551.3,
    "revenue_delta": 214200.0,
    "currency": "INR"
  }
}
```

### GET /demo

Run analysis on pre-loaded demo file (for PPT demos).

### GET /health

Check service status and model availability.

---

## 📈 Expected Metrics

After training on synthetic data:

| Model | Metric | Expected |
|-------|--------|----------|
| Anomaly Detector | ROC-AUC | 0.85+ |
| Fault Classifier | Accuracy | 0.80+ |
| Fault Classifier | F1 Score | 0.75+ |
| Severity Estimator | RMSE | < 0.15 |
| Severity Estimator | R² | > 0.70 |

---

## 🎤 PPT Defense Notes

### For Judges

**Q: Is this real ML or rule-based?**
> Real ML. Isolation Forest for unsupervised anomaly detection, Random Forest for supervised classification. No hardcoded rules.

**Q: How do you attribute faults to specific blades?**
> We analyze vibration energy per 120° rotor sector. The blade sector with disproportionately high energy is flagged as the fault source.

**Q: How is severity calculated?**
> A Random Forest Regressor trained on labeled synthetic data with known severity levels predicts continuous scores from feature patterns.

**Q: What makes cyclic_vibration different from erosion?**
> Erosion shows steady power loss (trend feature). Cyclic vibration shows 1P frequency spikes in FFT analysis aligned to a specific blade sector.

**Q: Can this run offline?**
> Yes. All models are saved locally as .joblib files. No cloud dependency.

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| MODELS_DIR | ./saved_models | Path to trained models |
| ML_SERVICE_PORT | 8000 | API port |

---

## 📝 License

MIT License - Built for eDC Blueprint 6.0 Hackathon
