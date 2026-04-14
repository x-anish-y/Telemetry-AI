"""
What-If Forecasting Model

Predicts energy output for two scenarios:
A) Immediate repair (optimistic trajectory)
B) No repair (degradation trajectory)

Also calculates revenue impact in INR.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


# Constants
RATED_POWER_KW = 2500
CAPACITY_FACTOR_HEALTHY = 0.35  # Typical wind farm capacity factor
HOURS_PER_MONTH = 720
ENERGY_PRICE_INR_PER_MWH = 3500  # Feed-in tariff


class WhatIfForecaster:
    """
    Forecasting model for what-if energy scenarios.
    
    Uses regression to predict:
    - Expected capacity factor reduction from fault
    - Recovery trajectory after repair
    - Degradation trajectory without repair
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        # Model predicts capacity factor loss
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.feature_names: list = []
        self.is_fitted: bool = False
    
    def fit(
        self,
        X: pd.DataFrame,
        severity: pd.Series,
        feature_names: Optional[list] = None
    ) -> 'WhatIfForecaster':
        """
        Fit forecaster using severity as proxy for capacity factor loss.
        
        Capacity factor loss is approximated as:
        loss = severity * 0.15 (up to 15% loss at max severity)
        """
        if feature_names is None:
            exclude = ['window_idx', 'start_idx', 'end_idx', 'fault_type', 'blade_id', 'severity', 'filename']
            feature_names = [c for c in X.columns if c not in exclude and X[c].dtype in ['float64', 'int64']]
        
        self.feature_names = feature_names
        X_features = X[feature_names].values
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Target: capacity factor loss
        y = severity * 0.15  # Scale severity to capacity factor loss
        
        X_scaled = self.scaler.fit_transform(X_features)
        
        logger.info(f"Fitting What-If Forecaster on {len(X)} samples")
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
    
    def predict_capacity_loss(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict capacity factor loss.
        
        Returns:
            loss: Array of capacity factor losses (0.0 - 0.2)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X_features = X[self.feature_names].values
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X_features)
        
        predictions = self.model.predict(X_scaled)
        return np.clip(predictions, 0.0, 0.2)
    
    def forecast(
        self,
        features: Dict[str, float],
        fault_type: str,
        severity: float,
        forecast_months: int = 1
    ) -> Dict[str, float]:
        """
        Generate what-if forecast for a single sample.
        
        Returns forecast with energy output and revenue for:
        - Scenario A: Immediate repair
        - Scenario B: No repair (continued degradation)
        """
        # Calculate base energy (healthy operation)
        base_energy_mwh = (
            RATED_POWER_KW / 1000 *  # Convert to MW
            CAPACITY_FACTOR_HEALTHY *
            HOURS_PER_MONTH *
            forecast_months
        )
        
        # Predict capacity loss from features
        df = pd.DataFrame([features])
        if self.is_fitted:
            capacity_loss = self.predict_capacity_loss(df)[0]
        else:
            # Fallback: use severity directly
            capacity_loss = severity * 0.15
        
        # Scenario A: Repair (recover to ~98% of baseline)
        repair_efficiency = 0.98
        fix_energy_mwh = base_energy_mwh * repair_efficiency
        
        # Scenario B: No repair (continued degradation)
        # Degradation accelerates over time
        degradation_factor = 1 - capacity_loss
        if fault_type == 'erosion':
            # Erosion: slow but steady degradation
            degradation_factor *= (1 - 0.02 * forecast_months)
        elif fault_type == 'pitch_drift':
            # Pitch drift: moderate degradation
            degradation_factor *= (1 - 0.05 * forecast_months)
        elif fault_type == 'cyclic_vibration':
            # Cyclic vibration: can worsen rapidly
            degradation_factor *= (1 - 0.08 * forecast_months)
        
        nofix_energy_mwh = base_energy_mwh * max(degradation_factor, 0.7)
        
        # Revenue calculations
        fix_revenue = fix_energy_mwh * ENERGY_PRICE_INR_PER_MWH
        nofix_revenue = nofix_energy_mwh * ENERGY_PRICE_INR_PER_MWH
        revenue_delta = fix_revenue - nofix_revenue
        
        return {
            'fix_energy_mwh': round(fix_energy_mwh, 1),
            'nofix_energy_mwh': round(nofix_energy_mwh, 1),
            'revenue_delta': round(revenue_delta, 2),
            'currency': 'INR',
            'forecast_months': forecast_months,
            'capacity_loss_pct': round(capacity_loss * 100, 1)
        }
    
    def save(self, filepath: str):
        """Save model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, filepath)
        logger.info(f"Saved forecaster to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'WhatIfForecaster':
        """Load model from disk."""
        data = joblib.load(filepath)
        forecaster = cls()
        forecaster.model = data['model']
        forecaster.scaler = data['scaler']
        forecaster.feature_names = data['feature_names']
        forecaster.is_fitted = True
        return forecaster


class BladeAttributor:
    """
    Blade-level attribution using sector vibration analysis.
    
    Determines which blade (1, 2, or 3) is most likely affected
    based on rotor-phase aligned vibration energy.
    """
    
    def __init__(self):
        self.sector_weights = {1: 1.0, 2: 1.0, 3: 1.0}
    
    def attribute(self, features: Dict[str, float]) -> Tuple[Optional[int], float]:
        """
        Determine affected blade from sector features.
        
        Returns:
            blade_id: 1, 2, or 3 (or None if no clear attribution)
            confidence: Confidence score (0.0 - 1.0)
        """
        # Get blade sector energies
        energies = {}
        for blade_id in [1, 2, 3]:
            key = f'blade_{blade_id}_energy'
            if key in features:
                energies[blade_id] = features[key] * self.sector_weights[blade_id]
            else:
                energies[blade_id] = 0.0
        
        total_energy = sum(energies.values()) + 1e-10
        ratios = {k: v / total_energy for k, v in energies.items()}
        
        # Find dominant blade
        max_blade = max(ratios, key=ratios.get)
        max_ratio = ratios[max_blade]
        
        # Check if there's a clear winner
        second_max = sorted(ratios.values(), reverse=True)[1]
        
        # Confidence based on separation from second
        separation = max_ratio - second_max
        confidence = min(separation * 3, 1.0)  # Scale to 0-1
        
        # Only attribute if one blade clearly dominates
        if max_ratio > 0.4 and confidence > 0.3:
            return max_blade, round(confidence, 2)
        
        return None, 0.0
    
    def get_explanation(
        self,
        features: Dict[str, float],
        fault_type: str,
        blade_id: Optional[int]
    ) -> str:
        """
        Generate human-readable explanation for attribution.
        """
        if blade_id is None:
            return "No specific blade attribution. Fault may affect multiple blades or hub."
        
        # Get relevant feature values
        blade_energy = features.get(f'blade_{blade_id}_energy', 0)
        blade_ratio = features.get(f'blade_{blade_id}_ratio', 0)
        
        explanations = {
            'erosion': f"Blade {blade_id} shows {blade_ratio*100:.0f}% of total power residual contribution, indicating leading-edge erosion affecting aerodynamic efficiency.",
            'pitch_drift': f"Blade {blade_id} pitch sector exhibits {blade_ratio*100:.0f}% of pitch error energy, suggesting actuator or sensor fault.",
            'cyclic_vibration': f"Blade {blade_id} sector shows {blade_ratio*100:.0f}% of vibration energy with {blade_energy:.2f} RMS, indicating structural imbalance or damage.",
            'normal': "No fault detected. All blades operating normally."
        }
        
        return explanations.get(fault_type, f"Fault detected on Blade {blade_id}.")


if __name__ == '__main__':
    # Test
    print("Testing What-If Forecaster...")
    
    forecaster = WhatIfForecaster()
    
    # Test forecast without training (uses fallback)
    features = {'power_residual_mean': -100, 'vibration_std': 0.5}
    result = forecaster.forecast(features, 'cyclic_vibration', severity=0.6)
    print(f"Forecast result: {result}")
    
    print("\nTesting Blade Attributor...")
    attributor = BladeAttributor()
    
    test_features = {
        'blade_1_energy': 0.3,
        'blade_2_energy': 0.8,
        'blade_3_energy': 0.25,
        'blade_1_ratio': 0.22,
        'blade_2_ratio': 0.59,
        'blade_3_ratio': 0.19
    }
    
    blade, conf = attributor.attribute(test_features)
    print(f"Attributed blade: {blade}, confidence: {conf}")
    print(f"Explanation: {attributor.get_explanation(test_features, 'cyclic_vibration', blade)}")
