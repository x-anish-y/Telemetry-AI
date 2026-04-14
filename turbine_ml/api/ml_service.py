"""
FastAPI ML Service for Wind Turbine Fault Detection

Exposes the trained ML models via HTTP API.
"""

import os
import sys
from pathlib import Path
import tempfile
import logging
from typing import Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import io

from inference.analyze import TurbineAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Turbine ML Service",
    description="Wind Turbine Blade Fault Detection ML API",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analyzer instance
analyzer: Optional[TurbineAnalyzer] = None

# Response models
class WhatIfForecast(BaseModel):
    fix_energy_mwh: float
    nofix_energy_mwh: float
    revenue_delta: float
    currency: str = "INR"

class AnalysisResponse(BaseModel):
    detected: bool
    anomaly_count: int
    blade: Optional[int]
    fault_type: str
    severity: float
    confidence: float
    residual_series: List[float]
    what_if_forecast: WhatIfForecast

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str

class FeatureImportanceResponse(BaseModel):
    features: List[str]
    importance: List[float]


def get_analyzer() -> TurbineAnalyzer:
    """Get or create the analyzer instance."""
    global analyzer
    
    if analyzer is None:
        # Determine models directory
        models_dir = os.environ.get('MODELS_DIR', '/app/turbine_ml/saved_models')
        
        if not Path(models_dir).exists():
            raise HTTPException(
                status_code=503,
                detail=f"Models directory not found: {models_dir}. Run training first."
            )
        
        try:
            analyzer = TurbineAnalyzer(models_dir)
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Failed to load ML models: {str(e)}"
            )
    
    return analyzer


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health and model status."""
    try:
        _ = get_analyzer()
        return HealthResponse(
            status="healthy",
            models_loaded=True,
            version="1.0.0"
        )
    except HTTPException:
        return HealthResponse(
            status="unhealthy",
            models_loaded=False,
            version="1.0.0"
        )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_csv(file: UploadFile = File(...)):
    """
    Analyze SCADA CSV file for wind turbine blade faults.
    
    Upload a CSV file with columns:
    - timestamp
    - wind_speed
    - rotor_rpm
    - power
    - pitch_cmd
    - pitch_meas
    - torque
    - vibration
    
    Returns fault detection results including:
    - Fault type (erosion, pitch_drift, cyclic_vibration, normal)
    - Affected blade (1, 2, or 3)
    - Severity score (0.0 - 1.0)
    - What-if energy forecast
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read CSV content
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Validate required columns
        required_cols = ['wind_speed', 'rotor_rpm', 'power', 'pitch_cmd', 
                        'pitch_meas', 'torque', 'vibration']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing}"
            )
        
        # Run analysis
        analyzer = get_analyzer()
        result = analyzer.analyze_dataframe(df)
        
        # Format response
        return AnalysisResponse(
            detected=result['detected'],
            anomaly_count=result['anomaly_count'],
            blade=result['blade'],
            fault_type=result['fault_type'],
            severity=result['severity'],
            confidence=result['confidence'],
            residual_series=result['residual_series'],
            what_if_forecast=WhatIfForecast(**result['what_if_forecast'])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance():
    """Get feature importance from the trained classifier."""
    try:
        analyzer = get_analyzer()
        importance = analyzer.get_feature_importance()
        return FeatureImportanceResponse(**importance)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """API root with documentation link."""
    return {
        "service": "Turbine ML Service",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "POST /analyze": "Analyze SCADA CSV for faults",
            "GET /health": "Health check",
            "GET /feature-importance": "Get model feature importance"
        }
    }


# Demo endpoint for testing without file upload
@app.get("/demo")
async def demo_analysis():
    """
    Run demo analysis on pre-loaded test data.
    Useful for quick testing and PPT demos.
    """
    try:
        analyzer = get_analyzer()
        
        # Look for demo_test.csv
        demo_paths = [
            '/app/turbine_ml/synthetic_data/test/demo_test.csv',
            '/app/turbine_ml/examples/demo_test.csv'
        ]
        
        demo_path = None
        for path in demo_paths:
            if Path(path).exists():
                demo_path = path
                break
        
        if demo_path is None:
            raise HTTPException(
                status_code=404,
                detail="Demo test file not found. Generate synthetic data first."
            )
        
        result = analyzer.analyze_csv(demo_path)
        
        return {
            "demo_file": demo_path,
            "result": AnalysisResponse(
                detected=result['detected'],
                anomaly_count=result['anomaly_count'],
                blade=result['blade'],
                fault_type=result['fault_type'],
                severity=result['severity'],
                confidence=result['confidence'],
                residual_series=result['residual_series'],
                what_if_forecast=WhatIfForecast(**result['what_if_forecast'])
            )
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    
    port = int(os.environ.get('ML_SERVICE_PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
