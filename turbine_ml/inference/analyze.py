"""
Inference Module for Wind Turbine Fault Detection

Loads trained models and performs complete analysis on new CSV data.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_engineering.features import (
    extract_features_from_file,
    extract_window_features,
    FeatureConfig,
    FEATURE_GROUPS
)
from models.anomaly import AnomalyDetector
from models.classifier import FaultClassifier
from models.severity import SeverityEstimator
from models.forecasting import WhatIfForecaster, BladeAttributor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TurbineAnalyzer:
    """
    Complete inference pipeline for wind turbine fault detection.
    
    Combines all trained models:
    - Anomaly detection
    - Fault classification
    - Severity estimation
    - Blade attribution
    - What-if forecasting
    """
    
    def __init__(self, models_dir: str):
        """
        Load all trained models from directory.
        
        Args:
            models_dir: Path to directory containing saved models
        """
        self.models_dir = Path(models_dir)
        self.config = FeatureConfig()
        
        # Load models
        logger.info(f"Loading models from {models_dir}")
        
        self.anomaly_detector = AnomalyDetector.load(
            str(self.models_dir / 'anomaly_detector.joblib')
        )
        self.classifier = FaultClassifier.load(
            str(self.models_dir / 'fault_classifier.joblib')
        )
        self.severity_estimator = SeverityEstimator.load(
            str(self.models_dir / 'severity_estimator.joblib')
        )
        self.forecaster = WhatIfForecaster.load(
            str(self.models_dir / 'whatif_forecaster.joblib')
        )
        self.blade_attributor = BladeAttributor()
        
        logger.info("All models loaded successfully")
    
    def analyze_csv(self, csv_path: str) -> Dict:
        """
        Perform complete analysis on a CSV file.
        
        Args:
            csv_path: Path to SCADA CSV file
            
        Returns:
            Analysis result matching the API contract
        """
        # Extract features from all windows
        features_df, windows = extract_features_from_file(
            csv_path, 
            self.config,
            return_windows=True
        )
        
        if len(features_df) == 0:
            return self._empty_result()
        
        # Anomaly detection
        anomaly_scores, anomaly_flags = self.anomaly_detector.predict(features_df)
        anomaly_count = int(anomaly_flags.sum())
        detected = anomaly_count > len(features_df) * 0.1  # >10% anomalous windows
        
        # Fault classification
        fault_types, confidences = self.classifier.predict(features_df)
        
        # Get dominant fault type (excluding normal)
        fault_counts = pd.Series(fault_types).value_counts()
        if 'normal' in fault_counts and len(fault_counts) > 1:
            fault_counts = fault_counts.drop('normal')
        
        primary_fault = fault_counts.index[0] if len(fault_counts) > 0 else 'normal'
        
        # If primary fault is normal but we detected anomalies, use most anomalous window
        if primary_fault == 'normal' and detected:
            most_anomalous_idx = np.argmax(anomaly_scores)
            primary_fault = fault_types[most_anomalous_idx]
        
        # Get mean confidence for the primary fault
        mask = fault_types == primary_fault
        confidence = float(confidences[mask].mean()) if mask.any() else 0.5
        
        # Severity estimation
        if primary_fault != 'normal' and detected:
            severities = self.severity_estimator.predict(features_df)
            severity = float(np.percentile(severities, 75))  # Use 75th percentile
        else:
            severity = 0.0
        
        # Blade attribution
        # Use most anomalous window for attribution
        most_anomalous_idx = np.argmax(anomaly_scores)
        window_features = features_df.iloc[most_anomalous_idx].to_dict()
        blade_id, blade_confidence = self.blade_attributor.attribute(window_features)
        
        # Adjust overall confidence with blade attribution
        if blade_id is not None:
            confidence = min(confidence, confidence * 0.7 + blade_confidence * 0.3)
        
        # Power residual series (from last few windows)
        residual_series = features_df['power_residual_mean'].tail(10).tolist()
        residual_series = [round(r, 1) for r in residual_series]
        
        # What-if forecast
        forecast = self.forecaster.forecast(
            window_features,
            primary_fault,
            severity,
            forecast_months=1
        )
        
        # Generate explanation
        explanation = self.blade_attributor.get_explanation(
            window_features, primary_fault, blade_id
        )
        
        return {
            'detected': detected,
            'anomaly_count': anomaly_count,
            'blade': blade_id,
            'fault_type': primary_fault,
            'severity': round(severity, 2),
            'confidence': round(confidence, 2),
            'residual_series': residual_series,
            'what_if_forecast': {
                'fix_energy_mwh': forecast['fix_energy_mwh'],
                'nofix_energy_mwh': forecast['nofix_energy_mwh'],
                'revenue_delta': forecast['revenue_delta'],
                'currency': forecast['currency']
            },
            'explanation': explanation,
            'n_windows_analyzed': len(features_df)
        }
    
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict:
        """
        Analyze a DataFrame directly (for API use).
        """
        # Save to temp file and analyze
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f, index=False)
            temp_path = f.name
        
        try:
            result = self.analyze_csv(temp_path)
        finally:
            os.unlink(temp_path)
        
        return result
    
    def _empty_result(self) -> Dict:
        """Return empty result structure."""
        return {
            'detected': False,
            'anomaly_count': 0,
            'blade': None,
            'fault_type': 'normal',
            'severity': 0.0,
            'confidence': 0.0,
            'residual_series': [],
            'what_if_forecast': {
                'fix_energy_mwh': 0.0,
                'nofix_energy_mwh': 0.0,
                'revenue_delta': 0.0,
                'currency': 'INR'
            },
            'explanation': 'Insufficient data for analysis.',
            'n_windows_analyzed': 0
        }
    
    def get_feature_importance(self) -> Dict[str, List]:
        """Get feature importance from classifier."""
        importance_df = self.classifier.get_feature_importance(top_n=15)
        return {
            'features': importance_df['feature'].tolist(),
            'importance': importance_df['importance'].tolist()
        }


def demo_analysis(models_dir: str, csv_path: str):
    """
    Run demo analysis and print results.
    """
    analyzer = TurbineAnalyzer(models_dir)
    result = analyzer.analyze_csv(csv_path)
    
    print("\n" + "=" * 60)
    print("TURBINE FAULT ANALYSIS RESULT")
    print("=" * 60)
    print(f"Fault Detected: {result['detected']}")
    print(f"Fault Type: {result['fault_type']}")
    print(f"Affected Blade: {result['blade']}")
    print(f"Severity: {result['severity']:.0%}")
    print(f"Confidence: {result['confidence']:.0%}")
    print(f"Anomalies Found: {result['anomaly_count']}")
    print(f"\nExplanation: {result['explanation']}")
    print(f"\nWhat-If Forecast (1 month):")
    print(f"  If Fixed: {result['what_if_forecast']['fix_energy_mwh']} MWh")
    print(f"  If Not Fixed: {result['what_if_forecast']['nofix_energy_mwh']} MWh")
    print(f"  Revenue Impact: ₹{result['what_if_forecast']['revenue_delta']:,.0f}")
    print("=" * 60)
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference on CSV')
    parser.add_argument('--models-dir', type=str, default='../saved_models')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    models_dir = script_dir / args.models_dir.lstrip('../')
    
    demo_analysis(str(models_dir), args.csv)
