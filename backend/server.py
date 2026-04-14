from fastapi import FastAPI, APIRouter, HTTPException, Depends, UploadFile, File, Form, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import jwt
import bcrypt
import numpy as np
import pandas as pd
from scipy import stats
from io import StringIO
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import base64
from io import BytesIO

PROJECT_ROOT = Path(__file__).parent.parent

# Configure logging early (used during module import)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(PROJECT_ROOT / '.env')

# Demo mode
DEMO_MODE = os.environ.get('DEMO_MODE', 'true').lower() == 'true'

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL')
if not mongo_url and not DEMO_MODE:
    logger.error("MONGO_URL environment variable is not set (set DEMO_MODE=true to run without MongoDB)")

client = AsyncIOMotorClient(mongo_url) if mongo_url else None
db = client[os.environ.get('DB_NAME', 'telemetry-ai')] if client else None

# JWT Config
JWT_SECRET = os.environ.get('JWT_SECRET', 'telemetry-ai-secret-key-2024')
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

app = FastAPI(title="Telemetry AI API")
api_router = APIRouter(prefix="/api")
security = HTTPBearer()

# (logging configured above)

# ============== MODELS ==============

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    email: str
    name: str
    role: str
    created_at: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class TurbineCreate(BaseModel):
    name: str
    location: str
    rated_power_kw: int
    metadata: Optional[Dict[str, Any]] = {}

class TurbineResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    name: str
    location: str
    rated_power_kw: int
    metadata: Dict[str, Any]
    created_at: str
    last_detection: Optional[Dict[str, Any]] = None

class DetectionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    turbine_id: str
    blade: Optional[int]
    fault_type: str
    severity: float
    confidence: float
    anomaly_count: int
    ml_result: Dict[str, Any]
    created_at: str

class WorkOrderCreate(BaseModel):
    detection_id: str
    turbine_id: str
    parts: List[Dict[str, Any]] = []
    labor_hours: float
    cost_estimate: float

class WorkOrderResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    detection_id: str
    turbine_id: str
    parts: List[Dict[str, Any]]
    labor_hours: float
    cost_estimate: float
    status: str
    pdf_base64: Optional[str] = None
    created_at: str

class MLResult(BaseModel):
    detected: bool
    anomaly_count: int
    blade: Optional[int]
    fault_type: str
    severity: float
    confidence: float
    residual_series: List[float]
    what_if_forecast: Dict[str, Any]

class ConfigUpdate(BaseModel):
    demo_mode: Optional[bool] = None

# ============== AUTH HELPERS ==============

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.now(timezone.utc)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user = await db.users.find_one({"id": payload["sub"]}, {"_id": 0})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Import the real ML engine
import sys
sys.path.insert(0, str(PROJECT_ROOT / 'turbine_ml'))

# Global ML analyzer instance
_ml_analyzer = None

def get_ml_analyzer():
    """Lazy load the ML analyzer to avoid startup delays"""
    global _ml_analyzer
    if _ml_analyzer is None:
        try:
            from inference.analyze import TurbineAnalyzer
            models_dir = os.environ.get('ML_MODELS_DIR', str(PROJECT_ROOT / 'turbine_ml' / 'saved_models'))
            _ml_analyzer = TurbineAnalyzer(models_dir)
            logger.info(f"ML Analyzer loaded from {models_dir}")
        except Exception as e:
            logger.error(f"Failed to load ML analyzer: {e}")
            _ml_analyzer = None
    return _ml_analyzer

DEMO_FALLBACK = {
    "detected": True,
    "anomaly_count": 6,
    "blade": 2,
    "fault_type": "cyclic_vibration",
    "severity": 0.6,
    "confidence": 0.78,
    "residual_series": [-30.1, -28.3, -35.4, -22.1, -40.2, -33.8, -25.5, -38.9, -29.4, -36.7],
    "what_if_forecast": {
        "fix_energy_mwh": 220.0,
        "nofix_energy_mwh": 210.5,
        "revenue_delta": 18750.0,
        "currency": "INR"
    }
}

def analyze_scada_data(csv_content: str) -> Dict[str, Any]:
    """
    Analyze SCADA data using the trained ML models.
    
    Uses:
    - Isolation Forest for anomaly detection
    - Random Forest for fault classification
    - Random Forest Regressor for severity estimation
    - Gradient Boosting for what-if forecasting
    """
    try:
        # Try to use the real ML engine
        analyzer = get_ml_analyzer()
        
        if analyzer is not None:
            # Parse CSV to DataFrame
            df = pd.read_csv(StringIO(csv_content))
            
            # Normalize column names for ML engine compatibility
            df.columns = df.columns.str.lower().str.strip()
            
            # Map common column name variations to expected names
            col_mapping = {
                'power_output': 'power',
                'rotor_speed': 'rotor_rpm',
                'pitch_angle': 'pitch_meas',
                'pitch': 'pitch_meas',
                'pitch_command': 'pitch_cmd',
            }
            df = df.rename(columns=col_mapping)
            
            # Add missing columns with defaults if needed
            if 'pitch_cmd' not in df.columns and 'pitch_meas' in df.columns:
                df['pitch_cmd'] = df['pitch_meas']  # Assume no drift for missing cmd
            if 'torque' not in df.columns:
                df['torque'] = df.get('power', 0) / (df.get('rotor_rpm', 1) + 0.1) * 0.1
            if 'rotor_phase' not in df.columns:
                # Generate synthetic rotor phase from RPM
                if 'rotor_rpm' in df.columns:
                    df['rotor_phase'] = (df['rotor_rpm'].cumsum() / 60 * 360) % 360
                else:
                    df['rotor_phase'] = np.random.uniform(0, 360, len(df))
            
            # Run ML analysis
            result = analyzer.analyze_dataframe(df)
            
            logger.info(f"ML Analysis complete: {result['fault_type']}, severity={result['severity']}")
            
            return {
                "detected": result['detected'],
                "anomaly_count": result['anomaly_count'],
                "blade": result['blade'],
                "fault_type": result['fault_type'],
                "severity": result['severity'],
                "confidence": result['confidence'],
                "residual_series": result['residual_series'],
                "what_if_forecast": result['what_if_forecast']
            }
        else:
            # Fallback to demo mode if ML engine not available
            logger.warning("ML Analyzer not available, using demo fallback")
            if DEMO_MODE:
                return DEMO_FALLBACK
            raise HTTPException(status_code=503, detail="ML service unavailable")
            
    except Exception as e:
        logger.error(f"ML Analysis error: {e}")
        if DEMO_MODE:
            logger.info("Using demo fallback due to error")
            return DEMO_FALLBACK
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def generate_synthetic_csv(fault_type: str = "normal", rows: int = 100) -> str:
    """
    Generate synthetic SCADA data for testing.
    
    Output columns (ML-compatible):
    - timestamp: ISO datetime
    - wind_speed: Wind velocity (m/s)
    - rotor_rpm: Rotor speed (RPM)
    - power: Active power output (kW)
    - pitch_cmd: Commanded pitch angle (degrees)
    - pitch_meas: Measured pitch angle (degrees)
    - torque: Generator torque (kNm)
    - vibration: Nacelle vibration (m/s²)
    - temperature: Gearbox/Nacelle temperature (°C)
    """
    np.random.seed(42 + hash(fault_type) % 1000)
    
    timestamps = pd.date_range(start='2024-01-01', periods=rows, freq='1s')  # 1Hz sampling
    
    # Base values - physics-based simulation
    wind_speed = np.random.normal(12, 3, rows).clip(3, 25)
    rotor_rpm = wind_speed * 1.2 + np.random.normal(0, 0.3, rows)  # RPM proportional to wind
    rotor_rpm = rotor_rpm.clip(0, 15)  # Max 15 RPM
    
    # Power curve: P = k * v^3 (simplified)
    RATED_POWER = 2500  # kW
    power = np.where(
        wind_speed < 3.5, 0,
        np.where(
            wind_speed < 12.5,
            RATED_POWER * ((wind_speed - 3.5) / (12.5 - 3.5)) ** 3,
            RATED_POWER
        )
    ) + np.random.normal(0, 20, rows)
    power = power.clip(0, RATED_POWER * 1.05)
    
    # Pitch angles
    pitch_cmd = np.clip(15 - wind_speed, 0, 25)  # Simplified pitch schedule
    pitch_meas = pitch_cmd + np.random.normal(0, 0.2, rows)  # Small tracking error
    
    # Torque: proportional to power / rpm
    torque = np.where(rotor_rpm > 0.1, power / (rotor_rpm + 0.1) * 0.1, 0)
    torque = torque + np.random.normal(0, 2, rows)
    
    # Vibration baseline
    vibration = 0.5 + 0.1 * np.random.randn(rows)
    vibration = vibration.clip(0.1, 2.0)
    
    # Temperature: correlated with power output
    temperature = 35 + (power / RATED_POWER) * 25 + np.random.normal(0, 2, rows)
    temperature = temperature.clip(20, 80)
    
    # Inject faults based on type
    if fault_type == "erosion":
        # Gradual power decline (leading edge erosion)
        power_loss = np.linspace(0, 0.15, rows)  # Up to 15% loss
        power = power * (1 - power_loss)
        # Slight vibration increase
        vibration = vibration * (1 + np.linspace(0, 0.2, rows))
        
    elif fault_type == "cyclic_vibration":
        # Periodic vibration spikes (blade imbalance)
        # Simulate 1P frequency spikes every ~4 seconds (at 15 RPM)
        spike_period = int(60 / 15)  # seconds per rotation
        for i in range(0, rows, spike_period):
            end_idx = min(i + 2, rows)
            vibration[i:end_idx] += np.random.uniform(2, 4)
        # Small power fluctuation
        power = power * (1 - 0.02 * np.sin(2 * np.pi * np.arange(rows) / spike_period))
        
    elif fault_type == "pitch_drift":
        # Increasing pitch error over time
        drift = np.linspace(0, 5, rows)  # 5 degree drift
        pitch_meas = pitch_meas + drift + 0.5 * np.sin(2 * np.pi * np.arange(rows) / 100)
        # Power loss from pitch error
        pitch_error = np.abs(pitch_meas - pitch_cmd)
        power = power * (1 - np.clip(pitch_error / 15, 0, 0.12))
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'wind_speed': wind_speed.round(2),
        'rotor_rpm': rotor_rpm.round(2),
        'power': power.round(2),
        'pitch_cmd': pitch_cmd.round(2),
        'pitch_meas': pitch_meas.round(2),
        'torque': torque.round(2),
        'vibration': vibration.round(3),
        'temperature': temperature.round(1)
    })
    
    return df.to_csv(index=False)

def generate_work_order_pdf(work_order: dict, turbine: dict, detection: dict) -> str:
    """Generate PDF work order and return base64 encoded"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    elements.append(Paragraph(f"<b>WORK ORDER #{work_order['id'][:8].upper()}</b>", styles['Title']))
    elements.append(Spacer(1, 20))
    
    # Turbine Info
    elements.append(Paragraph(f"<b>Turbine:</b> {turbine['name']}", styles['Normal']))
    elements.append(Paragraph(f"<b>Location:</b> {turbine['location']}", styles['Normal']))
    elements.append(Paragraph(f"<b>Rated Power:</b> {turbine['rated_power_kw']} kW", styles['Normal']))
    elements.append(Spacer(1, 15))
    
    # Detection Info
    elements.append(Paragraph("<b>FAULT DIAGNOSIS</b>", styles['Heading2']))
    fault_data = [
        ['Fault Type', detection['fault_type'].replace('_', ' ').title()],
        ['Affected Blade', f"Blade {detection['blade']}" if detection['blade'] else 'N/A'],
        ['Severity', f"{detection['severity'] * 100:.0f}%"],
        ['Confidence', f"{detection['confidence'] * 100:.0f}%"],
        ['Anomalies Detected', str(detection['anomaly_count'])]
    ]
    fault_table = Table(fault_data, colWidths=[150, 200])
    fault_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(fault_table)
    elements.append(Spacer(1, 15))
    
    # Work Order Details
    elements.append(Paragraph("<b>WORK ORDER DETAILS</b>", styles['Heading2']))
    wo_data = [
        ['Labor Hours', f"{work_order['labor_hours']} hours"],
        ['Cost Estimate', f"INR {work_order['cost_estimate']:,.2f}"],
        ['Status', work_order['status'].upper()],
        ['Created', work_order['created_at'][:10]]
    ]
    wo_table = Table(wo_data, colWidths=[150, 200])
    wo_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(wo_table)
    elements.append(Spacer(1, 15))
    
    # Parts
    if work_order.get('parts'):
        elements.append(Paragraph("<b>REQUIRED PARTS</b>", styles['Heading2']))
        parts_data = [['Part Name', 'Quantity', 'Unit Cost']]
        for part in work_order['parts']:
            parts_data.append([part.get('name', 'N/A'), str(part.get('quantity', 1)), f"INR {part.get('cost', 0):,.2f}"])
        parts_table = Table(parts_data, colWidths=[200, 75, 100])
        parts_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(parts_table)
    
    # Forecast
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("<b>ECONOMIC IMPACT FORECAST</b>", styles['Heading2']))
    forecast = detection['ml_result'].get('what_if_forecast', {})
    forecast_data = [
        ['If Fixed - Energy Output', f"{forecast.get('fix_energy_mwh', 0)} MWh"],
        ['If Not Fixed - Energy Output', f"{forecast.get('nofix_energy_mwh', 0)} MWh"],
        ['Revenue Impact', f"INR {forecast.get('revenue_delta', 0):,.2f}"]
    ]
    forecast_table = Table(forecast_data, colWidths=[200, 150])
    forecast_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgreen),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(forecast_table)
    
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    return base64.b64encode(pdf_bytes).decode('utf-8')

# ============== AUTH ROUTES ==============

@api_router.post("/auth/register", response_model=TokenResponse)
async def register(user: UserCreate):
    existing = await db.users.find_one({"email": user.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_id = str(uuid.uuid4())
    user_doc = {
        "id": user_id,
        "email": user.email,
        "name": user.name,
        "password": hash_password(user.password),
        "role": "operator",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.users.insert_one(user_doc)
    
    token = create_token(user_id, user.email)
    return TokenResponse(
        access_token=token,
        user=UserResponse(
            id=user_id,
            email=user.email,
            name=user.name,
            role="operator",
            created_at=user_doc["created_at"]
        )
    )

@api_router.post("/auth/login", response_model=TokenResponse)
async def login(credentials: UserLogin):
    user = await db.users.find_one({"email": credentials.email}, {"_id": 0})
    if not user or not verify_password(credentials.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token(user["id"], user["email"])
    return TokenResponse(
        access_token=token,
        user=UserResponse(
            id=user["id"],
            email=user["email"],
            name=user["name"],
            role=user["role"],
            created_at=user["created_at"]
        )
    )

@api_router.get("/auth/me", response_model=UserResponse)
async def get_me(user: dict = Depends(get_current_user)):
    return UserResponse(
        id=user["id"],
        email=user["email"],
        name=user["name"],
        role=user["role"],
        created_at=user["created_at"]
    )

# ============== TURBINE ROUTES ==============

@api_router.post("/turbines", response_model=TurbineResponse)
async def create_turbine(turbine: TurbineCreate, user: dict = Depends(get_current_user)):
    turbine_id = str(uuid.uuid4())
    turbine_doc = {
        "id": turbine_id,
        "name": turbine.name,
        "location": turbine.location,
        "rated_power_kw": turbine.rated_power_kw,
        "metadata": turbine.metadata or {},
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.turbines.insert_one(turbine_doc)
    return TurbineResponse(**turbine_doc, last_detection=None)

@api_router.get("/turbines", response_model=List[TurbineResponse])
async def list_turbines(user: dict = Depends(get_current_user)):
    turbines = await db.turbines.find({}, {"_id": 0}).to_list(100)
    
    # Get last detection for each turbine
    result = []
    for t in turbines:
        last_det = await db.detections.find_one(
            {"turbine_id": t["id"]},
            {"_id": 0},
            sort=[("created_at", -1)]
        )
        result.append(TurbineResponse(**t, last_detection=last_det))
    
    return result

@api_router.get("/turbines/{turbine_id}", response_model=TurbineResponse)
async def get_turbine(turbine_id: str, user: dict = Depends(get_current_user)):
    turbine = await db.turbines.find_one({"id": turbine_id}, {"_id": 0})
    if not turbine:
        raise HTTPException(status_code=404, detail="Turbine not found")
    
    last_det = await db.detections.find_one(
        {"turbine_id": turbine_id},
        {"_id": 0},
        sort=[("created_at", -1)]
    )
    return TurbineResponse(**turbine, last_detection=last_det)

# ============== DETECTION/ANALYSIS ROUTES ==============

@api_router.post("/analyze", response_model=DetectionResponse)
async def analyze_csv(
    file: UploadFile = File(...),
    turbine_id: str = Form(...),
    user: dict = Depends(get_current_user)
):
    # Verify turbine exists
    turbine = await db.turbines.find_one({"id": turbine_id}, {"_id": 0})
    if not turbine:
        raise HTTPException(status_code=404, detail="Turbine not found")
    
    # Read CSV content
    content = await file.read()
    csv_content = content.decode('utf-8')
    
    # Run ML analysis
    ml_result = analyze_scada_data(csv_content)
    
    # Save detection
    detection_id = str(uuid.uuid4())
    detection_doc = {
        "id": detection_id,
        "turbine_id": turbine_id,
        "blade": ml_result["blade"],
        "fault_type": ml_result["fault_type"],
        "severity": ml_result["severity"],
        "confidence": ml_result["confidence"],
        "anomaly_count": ml_result["anomaly_count"],
        "ml_result": ml_result,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.detections.insert_one(detection_doc)
    
    return DetectionResponse(**detection_doc)

@api_router.get("/detections/{turbine_id}", response_model=List[DetectionResponse])
async def get_detections(turbine_id: str, user: dict = Depends(get_current_user)):
    detections = await db.detections.find(
        {"turbine_id": turbine_id},
        {"_id": 0}
    ).sort("created_at", -1).to_list(50)
    return [DetectionResponse(**d) for d in detections]

@api_router.get("/detection/{detection_id}", response_model=DetectionResponse)
async def get_detection(detection_id: str, user: dict = Depends(get_current_user)):
    detection = await db.detections.find_one({"id": detection_id}, {"_id": 0})
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    return DetectionResponse(**detection)

# ============== WORK ORDER ROUTES ==============

@api_router.post("/workorders", response_model=WorkOrderResponse)
async def create_work_order(wo: WorkOrderCreate, user: dict = Depends(get_current_user)):
    # Get detection and turbine
    detection = await db.detections.find_one({"id": wo.detection_id}, {"_id": 0})
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    turbine = await db.turbines.find_one({"id": wo.turbine_id}, {"_id": 0})
    if not turbine:
        raise HTTPException(status_code=404, detail="Turbine not found")
    
    wo_id = str(uuid.uuid4())
    wo_doc = {
        "id": wo_id,
        "detection_id": wo.detection_id,
        "turbine_id": wo.turbine_id,
        "parts": wo.parts,
        "labor_hours": wo.labor_hours,
        "cost_estimate": wo.cost_estimate,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Generate PDF
    pdf_base64 = generate_work_order_pdf(wo_doc, turbine, detection)
    wo_doc["pdf_base64"] = pdf_base64
    
    await db.workorders.insert_one(wo_doc)
    
    return WorkOrderResponse(**wo_doc)

@api_router.get("/workorders/{turbine_id}", response_model=List[WorkOrderResponse])
async def get_work_orders(turbine_id: str, user: dict = Depends(get_current_user)):
    workorders = await db.workorders.find(
        {"turbine_id": turbine_id},
        {"_id": 0}
    ).sort("created_at", -1).to_list(50)
    return [WorkOrderResponse(**wo) for wo in workorders]

# ============== SIMULATOR ROUTES ==============

@api_router.post("/simulate")
async def simulate_data(
    fault_type: str = Form("normal"),
    rows: int = Form(100),
    user: dict = Depends(get_current_user)
):
    csv_content = generate_synthetic_csv(fault_type, rows)
    return {
        "csv_content": csv_content,
        "rows": rows,
        "fault_type": fault_type
    }

# ============== CONFIG ROUTES ==============

@api_router.get("/config")
async def get_config(user: dict = Depends(get_current_user)):
    # Check ML analyzer status
    analyzer = get_ml_analyzer()
    ml_status = "loaded" if analyzer is not None else "unavailable"
    
    return {
        "demo_mode": DEMO_MODE,
        "ml_service_url": os.environ.get('ML_SERVICE_URL', 'internal'),
        "ml_status": ml_status,
        "ml_models_dir": os.environ.get('ML_MODELS_DIR', '/app/turbine_ml/saved_models')
    }

@api_router.put("/config")
async def update_config(config: ConfigUpdate, user: dict = Depends(get_current_user)):
    global DEMO_MODE
    if config.demo_mode is not None:
        DEMO_MODE = config.demo_mode
    return {"demo_mode": DEMO_MODE}

@api_router.get("/ml-status")
async def get_ml_status(user: dict = Depends(get_current_user)):
    """Get detailed ML engine status and feature importance."""
    analyzer = get_ml_analyzer()
    
    if analyzer is None:
        return {
            "status": "unavailable",
            "message": "ML models not loaded. Check ML_MODELS_DIR environment variable.",
            "models_dir": os.environ.get('ML_MODELS_DIR', '/app/turbine_ml/saved_models')
        }
    
    try:
        importance = analyzer.get_feature_importance()
        return {
            "status": "ready",
            "models_loaded": [
                "anomaly_detector (Isolation Forest)",
                "fault_classifier (Random Forest)",
                "severity_estimator (RF Regressor)",
                "whatif_forecaster (Gradient Boosting)"
            ],
            "feature_importance": importance,
            "models_dir": os.environ.get('ML_MODELS_DIR', '/app/turbine_ml/saved_models')
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ============== SEED DATA ==============

@api_router.post("/seed")
async def seed_data(user: dict = Depends(get_current_user)):
    """Seed demo turbines"""
    demo_turbines = [
        {"name": "WTG-001", "location": "Gujarat Wind Farm - Block A", "rated_power_kw": 2500},
        {"name": "WTG-002", "location": "Gujarat Wind Farm - Block A", "rated_power_kw": 2500},
        {"name": "WTG-003", "location": "Gujarat Wind Farm - Block B", "rated_power_kw": 3000},
        {"name": "WTG-004", "location": "Tamil Nadu Coast - Site 1", "rated_power_kw": 2000},
        {"name": "WTG-005", "location": "Rajasthan Desert Farm", "rated_power_kw": 3500},
    ]
    
    created = []
    for t in demo_turbines:
        existing = await db.turbines.find_one({"name": t["name"]})
        if not existing:
            turbine_id = str(uuid.uuid4())
            doc = {
                "id": turbine_id,
                **t,
                "metadata": {"manufacturer": "Suzlon", "model": "S120"},
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            await db.turbines.insert_one(doc)
            created.append(doc["name"])
    
    return {"message": f"Seeded {len(created)} turbines", "created": created}

# ============== HEALTH CHECK ==============

@api_router.get("/health")
async def health():
    return {"status": "healthy", "demo_mode": DEMO_MODE}

# Include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    if client is not None:
        client.close()
