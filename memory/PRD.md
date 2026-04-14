# Turbine Fault Detect - Product Requirements Document

## Original Problem Statement
Create a wind turbine predictive maintenance system that:
- Ingests SCADA CSV telemetry
- Performs blade-level fault detection using ML
- Visualizes diagnosis and forecasting results
- Generates maintenance work orders
- Demo-safe for hackathon presentation

## User Choices
- **Backend**: FastAPI + MongoDB (fully supported)
- **ML Service**: Real Python ML endpoint with statistical anomaly detection
- **Authentication**: JWT-based custom auth
- **PDF Generation**: Server-side using ReportLab

## User Personas
1. **Wind Farm Operators**: Monitor turbine health from dashboard, identify faults early
2. **Maintenance Technicians**: Receive work orders, access diagnosis details
3. **Energy Sector Managers**: Review economic impact of faults

## Core Requirements (Static)
- [x] User authentication (register/login with JWT)
- [x] Dashboard with turbine cards and status indicators
- [x] Turbine detail page with blade SVG visualization
- [x] SCADA CSV upload and analysis
- [x] Real ML anomaly detection (z-score, trend analysis)
- [x] What-if forecast comparison charts
- [x] Work order PDF generation
- [x] SCADA data simulator
- [x] Demo mode fallback
- [x] Screenshot export (1920x1080)

## What's Been Implemented (Jan 8, 2025)

### Backend (FastAPI + MongoDB)
- JWT authentication (register, login, protected routes)
- Turbine CRUD operations
- Detection records storage
- ML analysis engine with:
  - Z-score anomaly detection
  - Trend analysis for erosion
  - Pitch drift detection
  - Blade attribution logic
- Work order PDF generation (ReportLab)
- SCADA data simulator (normal, erosion, cyclic_vibration, pitch_drift)
- Demo mode configuration

### Frontend (React + Tailwind)
- Login/Register page with dark industrial theme
- Dashboard with:
  - Stats summary (total, healthy, warning, critical)
  - Turbine cards grid with status badges
  - Seed demo data button
- Turbine Detail page with:
  - Custom SVG blade diagram (3 blades, color-coded)
  - Severity bar
  - Latest diagnosis summary
  - Residual analysis chart (Recharts)
  - What-if forecast comparison (bar chart)
  - CSV upload form
  - Work order generation button
  - Screenshot export
- Simulator page with:
  - Fault type selector
  - Row count input
  - CSV preview table
  - Download CSV button
- Admin page with:
  - Demo mode toggle
  - ML service info

### Test Results
- Backend: 93.3% pass rate
- Frontend: 100% pass rate
- Overall: 96.7% pass rate

## Prioritized Backlog

### P0 (Completed)
- [x] Core authentication flow
- [x] Turbine management
- [x] ML analysis pipeline
- [x] Visualization components
- [x] Work order generation

### P1 (Future Enhancements)
- [ ] Real-time SCADA streaming (WebSocket)
- [ ] Drone image analysis integration
- [ ] Multi-turbine batch analysis
- [ ] Maintenance scheduling calendar
- [ ] Alert notifications (email/SMS)

### P2 (Nice to Have)
- [ ] Historical trend analytics
- [ ] Comparative fleet analysis
- [ ] Cost tracking dashboard
- [ ] Mobile technician app
- [ ] ERP/CMMS integration

## Technical Architecture
```
Frontend (React + Tailwind)
    ↓ HTTP/REST
Backend (FastAPI + MongoDB)
    ↓ Internal / HTTP
ML Engine (/app/turbine_ml)
    ↓
PDF Generation (ReportLab)
```

---

## ML Engine (turbine_ml/)

### Implemented Models
| Model | Algorithm | Metric | Score |
|-------|-----------|--------|-------|
| Anomaly Detector | Isolation Forest | ROC-AUC | 0.932 |
| Fault Classifier | Random Forest | Accuracy | 91.0% |
| Severity Estimator | RF Regressor | RMSE | 0.107 |
| What-If Forecaster | Gradient Boosting | - | Trained |

### Top Features (by importance)
1. pitch_error_std (11.9%)
2. pitch_error_abs_mean (11.3%)
3. power_residual_norm_mean (10.6%)
4. pitch_error_max (6.6%)
5. blade_imbalance_max (5.7%)

### Synthetic Data Generated
- Train: 62 files (19 healthy, 43 faulty)
- Validation: 13 files
- Test: 15 files (+demo_test.csv)

### ML API Endpoints
- POST /analyze - Analyze SCADA CSV
- GET /health - Service health
- GET /demo - Run demo analysis
- GET /feature-importance - Model explainability

---

## Next Steps
1. Integrate ML service with main backend (call /analyze endpoint)
2. Add real-time WebSocket updates for turbine status
3. Implement batch analysis for multiple turbines
4. Add email notifications for critical faults
5. Build maintenance scheduling with calendar view
