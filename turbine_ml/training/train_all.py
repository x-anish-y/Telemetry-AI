"""
Complete Training Pipeline for Wind Turbine ML Models

Trains all models:
1. Anomaly Detector (Isolation Forest) - on healthy data only
2. Fault Classifier (Random Forest) - on all labeled data
3. Severity Estimator (Random Forest Regressor) - on faulty data
4. What-If Forecaster (Gradient Boosting) - on all data

Evaluates on validation and test sets, prints metrics, saves models.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple
import json

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_engineering.features import (
    extract_features_from_file,
    FeatureConfig,
    get_feature_names
)
from models.anomaly import AnomalyDetector, evaluate_anomaly_detector
from models.classifier import FaultClassifier, evaluate_classifier
from models.severity import SeverityEstimator, evaluate_severity_estimator
from models.forecasting import WhatIfForecaster

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_split_data(
    data_dir: str,
    split: str,
    config: FeatureConfig = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and extract features from all CSV files in a split.
    
    Returns:
        features_df: DataFrame with features for all windows
        labels_df: Labels for each file
    """
    split_path = Path(data_dir) / split
    labels_path = split_path / 'labels.csv'
    
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    labels_df = pd.read_csv(labels_path)
    
    all_features = []
    
    for _, row in labels_df.iterrows():
        filepath = split_path / row['filename']
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            continue
        
        try:
            features_df, _ = extract_features_from_file(str(filepath), config)
            
            # Add labels to each window
            features_df['filename'] = row['filename']
            features_df['fault_type'] = row['fault_type']
            features_df['blade_id'] = row.get('blade_id', None)
            features_df['severity'] = row.get('severity', 0.0)
            
            all_features.append(features_df)
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
    
    if not all_features:
        raise ValueError(f"No valid files found in {split_path}")
    
    combined_df = pd.concat(all_features, ignore_index=True)
    logger.info(f"Loaded {split}: {len(combined_df)} windows from {len(labels_df)} files")
    
    return combined_df, labels_df


def train_all_models(
    data_dir: str,
    output_dir: str,
    config: FeatureConfig = None
) -> Dict:
    """
    Train all ML models and save them.
    
    Returns dict with training metrics.
    """
    if config is None:
        config = FeatureConfig()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Load data
    logger.info("=" * 60)
    logger.info("Loading training data...")
    train_df, train_labels = load_split_data(data_dir, 'train', config)
    
    logger.info("Loading validation data...")
    val_df, val_labels = load_split_data(data_dir, 'val', config)
    
    logger.info("Loading test data...")
    test_df, test_labels = load_split_data(data_dir, 'test', config)
    
    # Get feature names
    feature_names = get_feature_names()
    logger.info(f"Using {len(feature_names)} features")
    
    # Split by fault type
    train_healthy = train_df[train_df['fault_type'] == 'normal']
    train_faulty = train_df[train_df['fault_type'] != 'normal']
    val_healthy = val_df[val_df['fault_type'] == 'normal']
    val_faulty = val_df[val_df['fault_type'] != 'normal']
    test_healthy = test_df[test_df['fault_type'] == 'normal']
    test_faulty = test_df[test_df['fault_type'] != 'normal']
    
    logger.info(f"Train: {len(train_healthy)} healthy, {len(train_faulty)} faulty windows")
    logger.info(f"Val: {len(val_healthy)} healthy, {len(val_faulty)} faulty windows")
    logger.info(f"Test: {len(test_healthy)} healthy, {len(test_faulty)} faulty windows")
    
    # ============================================================
    # 1. Train Anomaly Detector (on healthy data only)
    # ============================================================
    logger.info("=" * 60)
    logger.info("Training Anomaly Detector (Isolation Forest)...")
    
    anomaly_detector = AnomalyDetector(contamination=0.05)
    anomaly_detector.fit(train_healthy, feature_names)
    
    # Evaluate on validation
    val_anomaly_metrics = evaluate_anomaly_detector(
        anomaly_detector, val_healthy, val_faulty
    )
    logger.info(f"Validation - ROC-AUC: {val_anomaly_metrics['roc_auc']:.3f}, "
                f"TPR: {val_anomaly_metrics['true_positive_rate']:.3f}, "
                f"FPR: {val_anomaly_metrics['false_positive_rate']:.3f}")
    
    # Evaluate on test
    test_anomaly_metrics = evaluate_anomaly_detector(
        anomaly_detector, test_healthy, test_faulty
    )
    logger.info(f"Test - ROC-AUC: {test_anomaly_metrics['roc_auc']:.3f}, "
                f"TPR: {test_anomaly_metrics['true_positive_rate']:.3f}, "
                f"FPR: {test_anomaly_metrics['false_positive_rate']:.3f}")
    
    anomaly_detector.save(str(output_path / 'anomaly_detector.joblib'))
    results['anomaly_detector'] = {
        'val': val_anomaly_metrics,
        'test': test_anomaly_metrics
    }
    
    # ============================================================
    # 2. Train Fault Classifier (on all data)
    # ============================================================
    logger.info("=" * 60)
    logger.info("Training Fault Classifier (Random Forest)...")
    
    classifier = FaultClassifier(n_estimators=100, max_depth=15)
    classifier.fit(train_df, train_df['fault_type'], feature_names)
    
    # Evaluate on validation
    val_class_metrics = evaluate_classifier(classifier, val_df, val_df['fault_type'])
    logger.info(f"Validation - Accuracy: {val_class_metrics['accuracy']:.3f}, "
                f"F1: {val_class_metrics['f1_weighted']:.3f}")
    
    # Evaluate on test
    test_class_metrics = evaluate_classifier(classifier, test_df, test_df['fault_type'])
    logger.info(f"Test - Accuracy: {test_class_metrics['accuracy']:.3f}, "
                f"F1: {test_class_metrics['f1_weighted']:.3f}")
    
    # Feature importance
    importance = classifier.get_feature_importance(top_n=10)
    logger.info(f"Top features:\n{importance.to_string()}")
    
    classifier.save(str(output_path / 'fault_classifier.joblib'))
    results['fault_classifier'] = {
        'val': {k: v for k, v in val_class_metrics.items() if k != 'confusion_matrix'},
        'test': {k: v for k, v in test_class_metrics.items() if k != 'confusion_matrix'},
        'feature_importance': importance.to_dict('records')
    }
    
    # ============================================================
    # 3. Train Severity Estimator (on faulty data)
    # ============================================================
    logger.info("=" * 60)
    logger.info("Training Severity Estimator (Random Forest Regressor)...")
    
    severity_estimator = SeverityEstimator(n_estimators=100, max_depth=10)
    severity_estimator.fit(train_faulty, train_faulty['severity'], feature_names)
    
    # Evaluate on validation
    val_sev_metrics = evaluate_severity_estimator(
        severity_estimator, val_faulty, val_faulty['severity']
    )
    logger.info(f"Validation - RMSE: {val_sev_metrics['rmse']:.3f}, "
                f"MAE: {val_sev_metrics['mae']:.3f}, "
                f"R2: {val_sev_metrics['r2']:.3f}")
    
    # Evaluate on test
    test_sev_metrics = evaluate_severity_estimator(
        severity_estimator, test_faulty, test_faulty['severity']
    )
    logger.info(f"Test - RMSE: {test_sev_metrics['rmse']:.3f}, "
                f"MAE: {test_sev_metrics['mae']:.3f}, "
                f"R2: {test_sev_metrics['r2']:.3f}")
    
    severity_estimator.save(str(output_path / 'severity_estimator.joblib'))
    results['severity_estimator'] = {
        'val': val_sev_metrics,
        'test': test_sev_metrics
    }
    
    # ============================================================
    # 4. Train What-If Forecaster
    # ============================================================
    logger.info("=" * 60)
    logger.info("Training What-If Forecaster (Gradient Boosting)...")
    
    forecaster = WhatIfForecaster(n_estimators=100, max_depth=6)
    forecaster.fit(train_df, train_df['severity'], feature_names)
    
    forecaster.save(str(output_path / 'whatif_forecaster.joblib'))
    results['whatif_forecaster'] = {'status': 'trained'}
    
    # ============================================================
    # Summary
    # ============================================================
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Models saved to: {output_path}")
    logger.info("")
    logger.info("SUMMARY METRICS:")
    logger.info(f"  Anomaly Detector ROC-AUC: {test_anomaly_metrics['roc_auc']:.3f}")
    logger.info(f"  Fault Classifier Accuracy: {test_class_metrics['accuracy']:.3f}")
    logger.info(f"  Severity Estimator RMSE: {test_sev_metrics['rmse']:.3f}")
    
    # Save results
    with open(output_path / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train all ML models')
    parser.add_argument('--data-dir', type=str, default='../synthetic_data',
                        help='Path to synthetic data directory')
    parser.add_argument('--output-dir', type=str, default='../saved_models',
                        help='Path to save trained models')
    parser.add_argument('--window-size', type=int, default=300,
                        help='Sliding window size in samples')
    parser.add_argument('--stride', type=int, default=60,
                        help='Sliding window stride')
    
    args = parser.parse_args()
    
    config = FeatureConfig(
        window_size=args.window_size,
        stride=args.stride
    )
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / args.data_dir.lstrip('../')
    output_dir = script_dir / args.output_dir.lstrip('../')
    
    train_all_models(str(data_dir), str(output_dir), config)


if __name__ == '__main__':
    main()
