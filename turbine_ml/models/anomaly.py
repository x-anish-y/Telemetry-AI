"""
Anomaly Detection Model using Isolation Forest

Trained ONLY on healthy data to detect anomalies (unsupervised).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Isolation Forest-based anomaly detector.
    
    Trained on healthy turbine data only.
    Returns anomaly scores and flags for new data.
    """
    
    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names: list = []
        self.is_fitted: bool = False
    
    def fit(
        self,
        X: pd.DataFrame,
        feature_names: Optional[list] = None
    ) -> 'AnomalyDetector':
        """
        Fit on healthy data only.
        
        Args:
            X: Feature DataFrame (healthy samples only)
            feature_names: List of feature column names to use
        """
        if feature_names is None:
            # Use all numeric columns except metadata
            exclude = ['window_idx', 'start_idx', 'end_idx', 'fault_type', 'blade_id', 'severity', 'filename']
            feature_names = [c for c in X.columns if c not in exclude and X[c].dtype in ['float64', 'int64']]
        
        self.feature_names = feature_names
        X_features = X[feature_names].values
        
        # Handle NaN/Inf
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Fit Isolation Forest
        logger.info(f"Fitting Isolation Forest on {len(X)} healthy samples with {len(feature_names)} features")
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomaly scores and flags.
        
        Returns:
            anomaly_scores: Higher = more anomalous (inverted from sklearn)
            anomaly_flags: 1 = anomaly, 0 = normal
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X_features = X[self.feature_names].values
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X_features)
        
        # Get anomaly scores (sklearn returns negative for anomalies)
        raw_scores = self.model.decision_function(X_scaled)
        
        # Invert so higher = more anomalous, normalize to [0, 1]
        anomaly_scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-10)
        
        # Get predictions (-1 = anomaly, 1 = normal)
        predictions = self.model.predict(X_scaled)
        anomaly_flags = (predictions == -1).astype(int)
        
        return anomaly_scores, anomaly_flags
    
    def predict_single(self, features: Dict[str, float]) -> Tuple[float, bool]:
        """
        Predict for a single sample.
        
        Returns:
            anomaly_score: 0-1 (higher = more anomalous)
            is_anomaly: Boolean flag
        """
        df = pd.DataFrame([features])
        scores, flags = self.predict(df)
        return float(scores[0]), bool(flags[0])
    
    def save(self, filepath: str):
        """Save model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'contamination': self.contamination
        }, filepath)
        logger.info(f"Saved anomaly detector to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'AnomalyDetector':
        """Load model from disk."""
        data = joblib.load(filepath)
        detector = cls(contamination=data['contamination'])
        detector.model = data['model']
        detector.scaler = data['scaler']
        detector.feature_names = data['feature_names']
        detector.is_fitted = True
        return detector


def evaluate_anomaly_detector(
    detector: AnomalyDetector,
    X_healthy: pd.DataFrame,
    X_faulty: pd.DataFrame
) -> Dict[str, float]:
    """
    Evaluate anomaly detector performance.
    
    Returns metrics including ROC-AUC.
    """
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    
    # Combine data
    X_all = pd.concat([X_healthy, X_faulty], ignore_index=True)
    y_true = np.array([0] * len(X_healthy) + [1] * len(X_faulty))
    
    # Predict
    scores, flags = detector.predict(X_all)
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)
    
    # Detection rates
    healthy_detected = flags[:len(X_healthy)].sum()
    faulty_detected = flags[len(X_healthy):].sum()
    
    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'false_positive_rate': float(healthy_detected / len(X_healthy)),
        'true_positive_rate': float(faulty_detected / len(X_faulty)),
        'n_healthy': len(X_healthy),
        'n_faulty': len(X_faulty)
    }


if __name__ == '__main__':
    # Test
    print("Testing Anomaly Detector...")
    
    # Create dummy data
    np.random.seed(42)
    n_features = 10
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    healthy_data = pd.DataFrame(
        np.random.randn(100, n_features),
        columns=feature_names
    )
    
    faulty_data = pd.DataFrame(
        np.random.randn(20, n_features) + 2,  # Shifted distribution
        columns=feature_names
    )
    
    # Train and evaluate
    detector = AnomalyDetector()
    detector.fit(healthy_data, feature_names)
    
    metrics = evaluate_anomaly_detector(detector, healthy_data, faulty_data)
    print(f"Metrics: {metrics}")
