"""
Fault Classification Model using Random Forest

Supervised classifier for fault type prediction:
- normal
- erosion
- pitch_drift
- cyclic_vibration
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import joblib
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


FAULT_CLASSES = ['normal', 'erosion', 'pitch_drift', 'cyclic_vibration']


class FaultClassifier:
    """
    Random Forest classifier for fault type prediction.
    
    Classes:
    - normal: Healthy operation
    - erosion: Blade surface erosion (power loss)
    - pitch_drift: Pitch system malfunction
    - cyclic_vibration: Blade imbalance/damage
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 15,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names: list = []
        self.is_fitted: bool = False
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[list] = None
    ) -> 'FaultClassifier':
        """
        Fit classifier on labeled data.
        
        Args:
            X: Feature DataFrame
            y: Fault type labels (Series of strings)
            feature_names: List of feature column names to use
        """
        if feature_names is None:
            exclude = ['window_idx', 'start_idx', 'end_idx', 'fault_type', 'blade_id', 'severity', 'filename']
            feature_names = [c for c in X.columns if c not in exclude and X[c].dtype in ['float64', 'int64']]
        
        self.feature_names = feature_names
        X_features = X[feature_names].values
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Encode labels
        self.label_encoder.fit(FAULT_CLASSES)
        y_encoded = self.label_encoder.transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Fit classifier
        logger.info(f"Fitting Random Forest on {len(X)} samples with {len(feature_names)} features")
        logger.info(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
        
        self.model.fit(X_scaled, y_encoded)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict fault types and confidence scores.
        
        Returns:
            predictions: Array of fault type strings
            confidences: Array of confidence scores (max probability)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X_features = X[self.feature_names].values
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X_features)
        
        # Get predictions and probabilities
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)
        
        # Decode labels
        predictions = self.label_encoder.inverse_transform(y_pred)
        confidences = np.max(y_proba, axis=1)
        
        return predictions, confidences
    
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get probability distribution over all fault classes.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X_features = X[self.feature_names].values
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X_features)
        
        proba = self.model.predict_proba(X_scaled)
        return pd.DataFrame(proba, columns=self.label_encoder.classes_)
    
    def predict_single(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict for a single sample.
        
        Returns:
            fault_type: Predicted fault type string
            confidence: Confidence score
            probabilities: Dict of class probabilities
        """
        df = pd.DataFrame([features])
        predictions, confidences = self.predict(df)
        proba_df = self.predict_proba(df)
        
        return (
            predictions[0],
            float(confidences[0]),
            proba_df.iloc[0].to_dict()
        )
    
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
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }, filepath)
        logger.info(f"Saved classifier to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FaultClassifier':
        """Load model from disk."""
        data = joblib.load(filepath)
        classifier = cls()
        classifier.model = data['model']
        classifier.scaler = data['scaler']
        classifier.label_encoder = data['label_encoder']
        classifier.feature_names = data['feature_names']
        classifier.is_fitted = True
        return classifier


def evaluate_classifier(
    classifier: FaultClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict:
    """
    Evaluate classifier performance.
    
    Returns detailed metrics including per-class precision/recall.
    """
    predictions, confidences = classifier.predict(X_test)
    
    # Overall metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predictions, average='weighted'
    )
    
    # Per-class metrics
    class_report = classification_report(y_test, predictions, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions, labels=FAULT_CLASSES)
    
    return {
        'accuracy': float((predictions == y_test).mean()),
        'precision_weighted': float(precision),
        'recall_weighted': float(recall),
        'f1_weighted': float(f1),
        'classification_report': class_report,
        'confusion_matrix': cm.tolist(),
        'classes': FAULT_CLASSES,
        'mean_confidence': float(confidences.mean())
    }


if __name__ == '__main__':
    # Test
    print("Testing Fault Classifier...")
    
    np.random.seed(42)
    n_features = 10
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Create dummy data
    X_train = pd.DataFrame(np.random.randn(200, n_features), columns=feature_names)
    y_train = pd.Series(np.random.choice(FAULT_CLASSES, 200))
    
    X_test = pd.DataFrame(np.random.randn(50, n_features), columns=feature_names)
    y_test = pd.Series(np.random.choice(FAULT_CLASSES, 50))
    
    # Train and evaluate
    classifier = FaultClassifier()
    classifier.fit(X_train, y_train, feature_names)
    
    metrics = evaluate_classifier(classifier, X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1 Score: {metrics['f1_weighted']:.3f}")
