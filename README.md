<p align="center">
  <img src="https://img.shields.io/badge/WIND-TURBINE%20AI-06b6d4?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PHBhdGggZD0iTTEyIDJ2MjAiLz48cGF0aCBkPSJNMiAxMmgyMCIvPjwvc3ZnPg==&logoColor=white" alt="Wind Turbine AI" />
  <img src="https://img.shields.io/badge/ML%20ENGINE-SCIKIT--LEARN-f97316?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="ML Engine" />
  <img src="https://img.shields.io/badge/BACKEND-FASTAPI-10b981?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/FRONTEND-REACT-61dafb?style=for-the-badge&logo=react&logoColor=black" alt="React" />
  <img src="https://img.shields.io/badge/DATABASE-MONGODB-47A248?style=for-the-badge&logo=mongodb&logoColor=white" alt="MongoDB" />
</p>

<br/>

<h1 align="center">
  ⚡ TELEMETRY-AI ⚡
</h1>

<h3 align="center">
  <em>Blade-Level Predictive Maintenance for Wind Turbines</em>
</h3>

<p align="center">
  <strong>🔬 Real ML. Real Predictions. Real Impact.</strong><br/>
  <sub>An end-to-end AI platform that detects, classifies, and forecasts wind turbine blade faults<br/>from raw SCADA telemetry — down to the <em>individual blade</em>.</sub>
</p>

<br/>

<p align="center">
  <a href="#-the-problem">The Problem</a> •
  <a href="#-what-telemetry-ai-does">What It Does</a> •
  <a href="#-ml-pipeline-deep-dive">ML Pipeline</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-api-reference">API Reference</a> •
  <a href="#-for-judges--evaluators">For Judges</a>
</p>

---

<br/>

## 🌪️ The Problem

> **$180 billion** — that's the projected global wind energy market by 2030. But unplanned turbine downtime costs operators **up to $500,000 per incident**, and blade failures account for **~30% of all turbine failures**.

Current maintenance? **Calendar-based**. Technicians climb 80-meter towers on a schedule, whether the turbine needs it or not. Meanwhile, micro-cracks propagate, pitch systems drift, and cyclic vibrations compound — invisible until catastrophic failure.

**We decided to fix that.**

<br/>

## 🧠 What Telemetry-AI Does

```
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   📤 Upload SCADA CSV  ──────►  🔬 ML Analysis Pipeline     ║
    ║                                    │                         ║
    ║                                    ├── Anomaly Detection     ║
    ║                                    ├── Fault Classification  ║
    ║                                    ├── Severity Estimation   ║
    ║                                    ├── Blade Attribution     ║
    ║                                    └── Revenue Forecasting   ║
    ║                                    │                         ║
    ║   📊 Dashboard  ◄───────────  📋 Results + Work Orders      ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
```

### Five ML Models. One Pipeline. Zero Guesswork.

| # | Model | Algorithm | What It Does | Output |
|---|-------|-----------|-------------|--------|
| 🔴 | **Anomaly Detector** | Isolation Forest | Finds abnormal telemetry windows | `anomaly_score`, `anomaly_flag` |
| 🟠 | **Fault Classifier** | Random Forest | Identifies *what* is wrong | `erosion` · `pitch_drift` · `cyclic_vibration` · `normal` |
| 🟡 | **Blade Attributor** | Sector Energy Analysis | Pinpoints *which blade* is failing | `blade (1, 2, or 3)` + `confidence` |
| 🟢 | **Severity Estimator** | RF Regressor | Quantifies *how bad* it is | `severity (0.0 → 1.0)` |
| 🔵 | **What-If Forecaster** | Gradient Boosting | Projects financial impact | `fix_energy_mwh`, `nofix_energy_mwh`, `revenue_delta` |

<br/>

---

## 🔬 ML Pipeline Deep Dive

### Phase 1 → Synthetic Data Generation

We don't wait for turbines to break. We simulate physics-guided fault signatures:

```
EROSION          ─────────────╲╲╲╲╲╲╲╲╲╲        Gradual power curve degradation (5-20% loss)
                               Power drops over time

PITCH DRIFT      ───────/‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾        Increasing pitch error (2-8° drift)
                        Cmd vs Measured diverges

CYCLIC VIBRATION ─╱╲─╱╲─╱╲─╱╲─╱╲─╱╲─╱╲─        1P frequency spikes aligned to blade phase
                  Periodic spikes every rotation
```

Each fault is parameterized by type, affected blade (1/2/3), severity (0.3–0.9), and onset point. We generate **90+ SCADA files** across train/val/test splits.

### Phase 2 → Feature Engineering (40+ Features)

```python
# Not just raw values — we extract the PHYSICS

Power Residual Features    →  Deviation from theoretical P = k·v³ power curve
Pitch Error Features       →  Commanded vs. measured pitch angle divergence
FFT Vibration Features     →  Frequency-domain energy bands (detects 1P excitation)
Blade Sector Features      →  Vibration energy per 120° rotor sector
Rolling Statistics         →  Windowed mean, std, trend across all channels
```

### Phase 3 → Model Training

```
                    ┌─────────────────────┐
                    │   HEALTHY DATA ONLY │
                    │   (30 files)        │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  ISOLATION FOREST   │  ← Learns "normal" boundary
                    │  (Anomaly Detector) │     Anything outside = anomaly
                    └─────────────────────┘

                    ┌─────────────────────┐
                    │  ALL LABELED DATA   │
                    │  (90 files)         │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
    ┌─────────────┐  ┌──────────────┐  ┌─────────────┐
    │ RANDOM      │  │ RF REGRESSOR │  │ GRADIENT    │
    │ FOREST      │  │ (Severity)   │  │ BOOSTING    │
    │ (Classifier)│  │              │  │ (Forecaster)│
    └─────────────┘  └──────────────┘  └─────────────┘
```

### Phase 4 → Inference Pipeline

```
CSV Upload → Feature Extraction → Anomaly Detection → Fault Classification
                                                            │
                        ┌───────────────────────────────────┘
                        ▼
              Severity Estimation → Blade Attribution → What-If Forecast
                                                            │
                        ┌───────────────────────────────────┘
                        ▼
                   JSON Response with full diagnostic + financial impact
```

<br/>

---

## 📈 Expected Model Performance

| Model | Metric | Target | Why It Matters |
|-------|--------|--------|---------------|
| Anomaly Detector | ROC-AUC | **≥ 0.85** | Catch faults before they escalate |
| Fault Classifier | Accuracy | **≥ 0.80** | Right diagnosis = right repair |
| Fault Classifier | F1 Score | **≥ 0.75** | Balance precision & recall across fault types |
| Severity Estimator | RMSE | **< 0.15** | Accurate severity = accurate cost estimates |
| Severity Estimator | R² | **> 0.70** | Model explains most variance in severity |

<br/>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TELEMETRY-AI                                │
│                                                                     │
│  ┌──────────────────────┐          ┌──────────────────────────────┐ │
│  │     REACT FRONTEND   │          │       FASTAPI BACKEND        │ │
│  │                      │  HTTP    │                              │ │
│  │  • Login / Auth      │◄────────►│  • JWT Authentication        │ │
│  │  • Dashboard         │          │  • Turbine CRUD              │ │
│  │  • Turbine Detail    │          │  • CSV Upload + Analysis     │ │
│  │  • 3-Blade SVG Viz   │          │  • Work Order + PDF Gen      │ │
│  │  • SCADA Simulator   │          │  • Synthetic Data Generator  │ │
│  │  • Admin Panel       │          │  • ML Status Monitoring      │ │
│  │                      │          │                              │ │
│  │  TailwindCSS         │          │         │                    │ │
│  │  Recharts            │          │         ▼                    │ │
│  │  Radix UI / Shadcn   │          │  ┌────────────────────┐     │ │
│  │  React Router        │          │  │   TURBINE ML ENGINE │     │ │
│  └──────────────────────┘          │  │                    │     │ │
│                                    │  │  Isolation Forest  │     │ │
│  ┌──────────────────────┐          │  │  Random Forest     │     │ │
│  │     MONGODB           │◄────────│  │  RF Regressor      │     │ │
│  │                      │          │  │  Gradient Boosting  │     │ │
│  │  • users             │          │  │                    │     │ │
│  │  • turbines          │          │  │  40+ Eng Features  │     │ │
│  │  • detections        │          │  │  Saved .joblib     │     │ │
│  │  • workorders        │          │  └────────────────────┘     │ │
│  └──────────────────────┘          └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

<br/>

---

## 📂 Project Structure

```
Telemetry-AI/
├── 📁 frontend/                 # React 19 + Tailwind + Shadcn UI
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Login.js          # JWT auth with bioluminescent UI
│   │   │   ├── Dashboard.js      # Fleet overview, health cards, charts
│   │   │   ├── TurbineDetail.js  # 3-blade SVG viz + fault analysis
│   │   │   ├── Simulator.js      # Generate & analyze synthetic SCADA
│   │   │   └── Admin.js          # ML status, config, seed data
│   │   ├── components/ui/        # Shadcn/Radix component library
│   │   ├── context/              # Auth & theme context providers
│   │   └── hooks/                # Custom React hooks
│   └── package.json
│
├── 📁 backend/
│   └── server.py                 # 775-line FastAPI powerhouse
│                                   • Auth (register/login/JWT)
│                                   • Turbine CRUD
│                                   • CSV analysis endpoint
│                                   • Work order + PDF generation
│                                   • Synthetic data generator
│                                   • ML engine integration
│
├── 📁 turbine_ml/               # 🧠 The ML Brain
│   ├── data_generator/           # Physics-guided synthetic SCADA
│   ├── feature_engineering/      # 40+ feature extraction
│   ├── models/                   # Anomaly, Classifier, Severity, Forecast
│   ├── training/                 # Complete training pipeline
│   ├── inference/                # Production inference module
│   ├── api/                      # Standalone ML microservice
│   ├── saved_models/             # Trained .joblib files
│   └── synthetic_data/           # Generated train/val/test splits
│
├── 📁 api/                      # Vercel serverless entry point
├── 📁 tests/                    # Backend + ML test suites
├── 📄 requirements.txt          # Python dependencies
├── 📄 vercel.json               # Deployment config
└── 📄 netlify.toml              # Alt deployment config
```

<br/>

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** with pip
- **Node.js 18+** with yarn or npm
- **MongoDB** (local or Atlas URI)

### 1️⃣ Clone & Setup Environment

```bash
git clone https://github.com/x-anish-y/Telemetry-AI.git
cd Telemetry-AI
```

### 2️⃣ Configure Environment

Create a `.env` file in the project root:

```env
MONGO_URL=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net
DB_NAME=telemetry-ai
JWT_SECRET=your-super-secret-key
DEMO_MODE=true
```

### 3️⃣ Start the Backend

```bash
pip install -r requirements.txt
cd backend
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

### 4️⃣ Train the ML Models (Optional — Demo mode works without this)

```bash
cd turbine_ml
pip install -r requirements.txt
python data_generator/generate_telemetry.py --output synthetic_data
python training/train_all.py
```

### 5️⃣ Start the Frontend

```bash
cd frontend
yarn install   # or npm install
yarn start     # or npm start
```

### 6️⃣ Open & Explore

```
Frontend  →  http://localhost:3000
Backend   →  http://localhost:8001
API Docs  →  http://localhost:8001/docs
```

<br/>

---

## 🌐 API Reference

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/register` | Create new account |
| `POST` | `/api/auth/login` | Get JWT token |
| `GET` | `/api/auth/me` | Get current user |

### Turbines

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/turbines` | Register a turbine |
| `GET` | `/api/turbines` | List all turbines |
| `GET` | `/api/turbines/:id` | Get turbine details |

### Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | **Upload CSV → Get ML diagnosis** |
| `GET` | `/api/detections/:turbine_id` | Detection history |
| `GET` | `/api/detection/:id` | Single detection |

### Work Orders

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/workorders` | Generate work order + PDF |
| `GET` | `/api/workorders/:turbine_id` | Work order history |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/ml-status` | ML engine status + feature importance |
| `POST` | `/api/simulate` | Generate synthetic SCADA data |
| `POST` | `/api/seed` | Seed demo turbines |

### Example: Analyze a CSV

```bash
curl -X POST http://localhost:8001/api/analyze \
  -H "Authorization: Bearer <your-jwt-token>" \
  -F "file=@scada_data.csv" \
  -F "turbine_id=<turbine-uuid>"
```

**Response:**
```json
{
  "detected": true,
  "anomaly_count": 12,
  "blade": 2,
  "fault_type": "cyclic_vibration",
  "severity": 0.65,
  "confidence": 0.82,
  "residual_series": [-30.1, -28.3, -35.4, "..."],
  "what_if_forecast": {
    "fix_energy_mwh": 612.5,
    "nofix_energy_mwh": 551.3,
    "revenue_delta": 214200.0,
    "currency": "INR"
  }
}
```

<br/>

---

## 📊 SCADA Data Format

### Required Columns

| Column | Description | Unit |
|--------|-------------|------|
| `timestamp` | Measurement time | ISO datetime |
| `wind_speed` | Wind velocity | m/s |
| `rotor_rpm` | Rotor speed | RPM |
| `power` | Active power output | kW |
| `pitch_cmd` | Commanded pitch angle | degrees |
| `pitch_meas` | Measured pitch angle | degrees |
| `torque` | Generator torque | kNm |
| `vibration` | Nacelle vibration | m/s² |

### Optional Columns

| Column | Description |
|--------|-------------|
| `rotor_phase` | Rotor azimuth position (0–360°) |
| `temperature` | Gearbox/nacelle temperature (°C) |

> **💡 Pro tip:** Column names are auto-normalized. `power_output` → `power`, `rotor_speed` → `rotor_rpm`, `pitch_angle` → `pitch_meas`. Just upload and go.

<br/>

---

## 🎨 Design Philosophy

> **"Bioluminescent Control Room"** — Data is art. Charts are heroes, not illustrations.

| Principle | Implementation |
|-----------|---------------|
| **Dark Mode First** | Energy monitoring happens in the dark. `#020617` backgrounds. |
| **SCADA × Swiss** | Industrial data density meets Swiss design clarity. |
| **Sharp Edges** | Industrial machinery has no soft corners. `rounded-sm` everywhere. |
| **Neon Accents** | Cyan (`#06b6d4`) primary. Orange (`#f97316`) accent. Glow effects. |
| **Typography** | Manrope for UI. JetBrains Mono for data values. |
| **Glass Panels** | `backdrop-blur-xl` + `bg-slate-950/70` for depth. |

<br/>

---

## 🎤 For Judges & Evaluators

<details>
<summary><strong>❓ "Is this real ML or rule-based?"</strong></summary>
<br/>

**Real ML.** Isolation Forest for unsupervised anomaly detection, Random Forest for supervised classification, RF Regressor for severity estimation, Gradient Boosting for revenue forecasting. Zero hardcoded rules. All models trained on physics-guided synthetic data and saved as `.joblib` artifacts.

</details>

<details>
<summary><strong>❓ "How do you attribute faults to specific blades?"</strong></summary>
<br/>

We analyze vibration energy per 120° rotor sector. Each blade sweeps through a distinct angular range. The sector with disproportionately high vibration energy is flagged as the fault source. This is a real technique used in wind industry condition monitoring.

</details>

<details>
<summary><strong>❓ "How is severity calculated?"</strong></summary>
<br/>

A Random Forest Regressor trained on labeled synthetic data with known severity levels. The model learns severity patterns from feature combinations — not just single thresholds, but the interaction between power residuals, vibration amplitudes, pitch errors, and their temporal trends.

</details>

<details>
<summary><strong>❓ "What's the 'What-If Forecaster'?"</strong></summary>
<br/>

Given a detected fault, a Gradient Boosting model predicts two energy output scenarios: **if repaired** vs. **if ignored**. The difference, multiplied by energy price, gives the projected revenue impact in INR. This transforms a maintenance decision into a **business decision**.

</details>

<details>
<summary><strong>❓ "Can this run fully offline?"</strong></summary>
<br/>

Yes. All ML models are saved locally as `.joblib` files. The backend includes a `DEMO_MODE` with fallback data. No cloud API dependencies whatsoever for core functionality.

</details>

<br/>

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Frontend** | React 19, Tailwind CSS, Shadcn/UI, Recharts | Bento-grid dashboard, interactive charts, glassmorphism |
| **Backend** | FastAPI, Motor (async MongoDB), Pydantic | Async-first, auto-generated docs, type-safe |
| **ML Engine** | scikit-learn, NumPy, Pandas, SciPy | Isolation Forest, Random Forest, Gradient Boosting |
| **Database** | MongoDB (Atlas or local) | Document-store for flexible telemetry schemas |
| **Auth** | JWT + bcrypt | Stateless, secure, token-based |
| **PDF Export** | ReportLab | Server-side work order PDF generation |
| **Deployment** | Vercel + Netlify | Serverless backend, static frontend |

<br/>

---

## 🔮 Roadmap

- [ ] 🌊 **Real-time streaming** — WebSocket live telemetry feed
- [ ] 🛰️ **MQTT integration** — Connect to actual SCADA gateways
- [ ] 📱 **Mobile app** — React Native companion for field technicians
- [ ] 🧪 **A/B model testing** — Deploy competing models, track performance
- [ ] 🤖 **Deep learning** — LSTM/Transformer models for temporal fault detection
- [ ] 🌍 **Multi-farm support** — Fleet-wide analytics across wind farms

<br/>

---

## 📜 License

**MIT License** — Built for the **eDC Blueprint 6.0 Hackathon**

<br/>

---

<p align="center">
  <sub>Built with ⚡ by the Telemetry-AI team</sub><br/>
  <sub>Because turbines shouldn't have to scream before we listen.</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/STATUS-OPERATIONAL-10b981?style=flat-square" />
  <img src="https://img.shields.io/badge/ML-TRAINED-06b6d4?style=flat-square" />
  <img src="https://img.shields.io/badge/BLADES-MONITORED-f97316?style=flat-square" />
</p>
