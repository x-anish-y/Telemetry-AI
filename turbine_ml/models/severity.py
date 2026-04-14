"""
Severity Estimation Model using Random Forest Regressor

Predicts severity score (0.0 - 1.0) for detected faults.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SeverityEstimator:
    """
    Random Forest Regressor for fault severity estimation.
    
    Predicts severity score in range [0.0, 1.0]
    - 0.0: No fault / minimal impact
    - 0.5: Moderate severity
    - 1.0: Critical severity
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names: list = []
        self.is_fitted: bool = False
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[list] = None
    ) -> 'SeverityEstimator':
        """
        Fit regressor on labeled data.
        
        Args:
            X: Feature DataFrame
            y: Severity scores (0.0 - 1.0)
            feature_names: List of feature column names to use
        """
        if feature_names is None:
            exclude = ['window_idx', 'start_idx', 'end_idx', 'fault_type', 'blade_id', 'severity', 'filename']
            feature_names = [c for c in X.columns if c not in exclude and X[c].dtype in ['float64', 'int64']]
        
        self.feature_names = feature_names
        X_features = X[feature_names].values
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Fit regressor
        logger.info(f"Fitting Severity Regressor on {len(X)} samples")
        logger.info(f"Severity range: {y.min():.2f} - {y.max():.2f}")
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict severity scores.
        
        Returns:
            severity_scores: Array of scores clipped to [0.0, 1.0]
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X_features = X[self.feature_names].values
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X_features)
        
        predictions = self.model.predict(X_scaled)
        return np.clip(predictions, 0.0, 1.0)
    
    def predict_single(self, features: Dict[str, float]) -> float:
        """
        Predict for a single sample.
        
        Returns:
            severity: Score in [0.0, 1.0]
        """
        df = pd.DataFrame([features])
        predictions = self.predict(df)
        return float(predictions[0])
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Get top N important features."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance.head(top_n)
    
    def save(self, filepath: str):
        """Save model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, filepath)
        logger.info(f"Saved severity estimator to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SeverityEstimator':
        """Load model from disk."""
        data = joblib.load(filepath)
        estimator = cls()
        estimator.model = data['model']
        estimator.scaler = data['scaler']
        estimator.feature_names = data['feature_names']
        estimator.is_fitted = True
        return estimator


def evaluate_severity_estimator(
    estimator: SeverityEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict:
    """
    Evaluate severity estimator performance.
    """
    predictions = estimator.predict(X_test)
    
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
        'mae': float(mean_absolute_error(y_test, predictions)),
        'r2': float(r2_score(y_test, predictions)),
        'mean_prediction': float(predictions.mean()),
        'std_prediction': float(predictions.std()),
        'n_samples': len(y_test)
    }


if __name__ == '__main__':
    # Test
    print("Testing Severity Estimator...")
    
    np.random.seed(42)
    n_features = 10
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Create dummy data with correlation
    X_train = pd.DataFrame(np.random.randn(200, n_features), columns=feature_names)
    y_train = pd.Series(np.clip(X_train['feature_0'] * 0.3 + 0.5 + np.random.randn(200) * 0.1, 0, 1))
    
    X_test = pd.DataFrame(np.random.randn(50, n_features), columns=feature_names)
    y_test = pd.Series(np.clip(X_test['feature_0'] * 0.3 + 0.5 + np.random.randn(50) * 0.1, 0, 1))
    
    # Train and evaluate
    estimator = SeverityEstimator()
    estimator.fit(X_train, y_train, feature_names)
    
    metrics = evaluate_severity_estimator(estimator, X_test, y_test)
    print(f"RMSE: {metrics['rmse']:.3f}")
    print(f"R2: {metrics['r2']:.3f}")
