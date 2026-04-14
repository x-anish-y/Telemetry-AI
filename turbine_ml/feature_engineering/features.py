"""
Feature Engineering Module for Wind Turbine SCADA Data

Extracts sliding window features for ML models:
- Power residual statistics
- Pitch error metrics
- FFT-based vibration energy bands
- Rotor-phase sector vibration (for blade attribution)
- Rolling statistics
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# Constants
RATED_POWER_KW = 2500
CUT_IN_WIND = 3.5
RATED_WIND = 12.5
BLADE_PHASES = {1: 0, 2: 120, 3: 240}


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    window_size: int = 300  # samples (5 min at 1Hz)
    stride: int = 60  # 1 minute stride
    fft_bands: List[Tuple[float, float]] = None
    sample_rate: float = 1.0
    
    def __post_init__(self):
        if self.fft_bands is None:
            # Frequency bands for vibration analysis
            self.fft_bands = [
                (0.01, 0.1),   # Very low frequency
                (0.1, 0.5),    # Low frequency (1P range)
                (0.5, 2.0),    # Mid frequency
                (2.0, 5.0),    # High frequency
            ]


def theoretical_power(wind_speed: np.ndarray) -> np.ndarray:
    """Calculate expected power from wind speed."""
    power = np.zeros_like(wind_speed, dtype=float)
    
    mask_cubic = (wind_speed >= CUT_IN_WIND) & (wind_speed < RATED_WIND)
    normalized = (wind_speed[mask_cubic] - CUT_IN_WIND) / (RATED_WIND - CUT_IN_WIND)
    power[mask_cubic] = RATED_POWER_KW * (normalized ** 3)
    
    mask_rated = (wind_speed >= RATED_WIND) & (wind_speed <= 25.0)
    power[mask_rated] = RATED_POWER_KW
    
    return power


def compute_power_residual_features(
    power: np.ndarray,
    wind_speed: np.ndarray
) -> Dict[str, float]:
    """
    Compute power residual statistics.
    Residual = Actual - Expected
    """
    expected = theoretical_power(wind_speed)
    residual = power - expected
    
    # Normalize by expected (avoid division by zero)
    valid_mask = expected > 10
    normalized_residual = np.zeros_like(residual)
    normalized_residual[valid_mask] = residual[valid_mask] / expected[valid_mask]
    
    return {
        'power_residual_mean': float(np.mean(residual)),
        'power_residual_std': float(np.std(residual)),
        'power_residual_min': float(np.min(residual)),
        'power_residual_max': float(np.max(residual)),
        'power_residual_norm_mean': float(np.mean(normalized_residual)),
        'power_residual_norm_std': float(np.std(normalized_residual)),
        'power_residual_trend': float(np.polyfit(np.arange(len(residual)), residual, 1)[0]),
    }


def compute_pitch_error_features(
    pitch_cmd: np.ndarray,
    pitch_meas: np.ndarray
) -> Dict[str, float]:
    """
    Compute pitch error statistics.
    """
    error = pitch_meas - pitch_cmd
    abs_error = np.abs(error)
    
    return {
        'pitch_error_mean': float(np.mean(error)),
        'pitch_error_std': float(np.std(error)),
        'pitch_error_abs_mean': float(np.mean(abs_error)),
        'pitch_error_max': float(np.max(abs_error)),
        'pitch_error_trend': float(np.polyfit(np.arange(len(error)), error, 1)[0]),
        'pitch_error_range': float(np.ptp(error)),
    }


def compute_fft_features(
    vibration: np.ndarray,
    sample_rate: float,
    bands: List[Tuple[float, float]]
) -> Dict[str, float]:
    """
    Compute FFT-based vibration energy in frequency bands.
    """
    n = len(vibration)
    if n < 4:
        return {f'fft_band_{i}_energy': 0.0 for i in range(len(bands))}
    
    # Compute FFT
    fft_vals = np.fft.rfft(vibration - np.mean(vibration))
    fft_freqs = np.fft.rfftfreq(n, d=1.0/sample_rate)
    power_spectrum = np.abs(fft_vals) ** 2
    
    features = {}
    total_energy = np.sum(power_spectrum) + 1e-10
    
    for i, (f_low, f_high) in enumerate(bands):
        mask = (fft_freqs >= f_low) & (fft_freqs < f_high)
        band_energy = np.sum(power_spectrum[mask])
        features[f'fft_band_{i}_energy'] = float(band_energy)
        features[f'fft_band_{i}_ratio'] = float(band_energy / total_energy)
    
    # Peak frequency
    peak_idx = np.argmax(power_spectrum[1:]) + 1  # Skip DC
    features['fft_peak_freq'] = float(fft_freqs[peak_idx])
    features['fft_total_energy'] = float(total_energy)
    
    return features


def compute_blade_sector_features(
    vibration: np.ndarray,
    rotor_phase: np.ndarray
) -> Dict[str, float]:
    """
    Compute vibration energy per blade sector.
    Used for blade-level attribution.
    """
    features = {}
    sector_energies = []
    
    for blade_id, center_phase in BLADE_PHASES.items():
        # Sector: ±60 degrees around blade phase
        phase_distance = np.abs((rotor_phase - center_phase + 180) % 360 - 180)
        sector_mask = phase_distance < 60
        
        if np.sum(sector_mask) > 0:
            sector_vib = vibration[sector_mask]
            energy = float(np.mean(sector_vib ** 2))
            peak = float(np.max(sector_vib))
            std = float(np.std(sector_vib))
        else:
            energy, peak, std = 0.0, 0.0, 0.0
        
        features[f'blade_{blade_id}_energy'] = energy
        features[f'blade_{blade_id}_peak'] = peak
        features[f'blade_{blade_id}_std'] = std
        sector_energies.append(energy)
    
    # Blade imbalance ratio
    total_energy = sum(sector_energies) + 1e-10
    for blade_id in [1, 2, 3]:
        features[f'blade_{blade_id}_ratio'] = features[f'blade_{blade_id}_energy'] / total_energy
    
    # Max imbalance
    features['blade_imbalance_max'] = float(max(sector_energies) / (np.mean(sector_energies) + 1e-10))
    
    return features


def compute_rolling_features(
    df: pd.DataFrame,
    window: int = 60
) -> Dict[str, float]:
    """
    Compute rolling statistics features.
    """
    features = {}
    
    for col in ['power', 'vibration', 'torque']:
        if col in df.columns:
            series = df[col].values
            rolling_mean = pd.Series(series).rolling(window, min_periods=1).mean()
            rolling_std = pd.Series(series).rolling(window, min_periods=1).std()
            
            features[f'{col}_rolling_mean'] = float(rolling_mean.iloc[-1])
            features[f'{col}_rolling_std'] = float(rolling_std.iloc[-1]) if not np.isnan(rolling_std.iloc[-1]) else 0.0
            features[f'{col}_trend_60'] = float(np.polyfit(np.arange(min(window, len(series))), series[-min(window, len(series)):], 1)[0])
    
    return features


def extract_window_features(
    df: pd.DataFrame,
    config: FeatureConfig = None
) -> Dict[str, float]:
    """
    Extract all features from a single window of data.
    """
    if config is None:
        config = FeatureConfig()
    
    features = {}
    
    # Auto-generate rotor_phase if missing
    if 'rotor_phase' not in df.columns:
        if 'rotor_rpm' in df.columns:
            # Generate synthetic rotor phase from RPM
            rotor_rpm = df['rotor_rpm'].values
            df = df.copy()
            df['rotor_phase'] = (np.cumsum(rotor_rpm / 60 * 360 / config.sample_rate)) % 360
        else:
            # Random phase as fallback
            df = df.copy()
            df['rotor_phase'] = np.random.uniform(0, 360, len(df))
    
    # Basic statistics
    features['n_samples'] = len(df)
    features['wind_speed_mean'] = float(df['wind_speed'].mean())
    features['wind_speed_std'] = float(df['wind_speed'].std())
    features['rotor_rpm_mean'] = float(df['rotor_rpm'].mean())
    features['power_mean'] = float(df['power'].mean())
    features['vibration_mean'] = float(df['vibration'].mean())
    features['vibration_std'] = float(df['vibration'].std())
    features['vibration_max'] = float(df['vibration'].max())
    features['torque_mean'] = float(df['torque'].mean())
    
    # Power residual features
    power_features = compute_power_residual_features(
        df['power'].values,
        df['wind_speed'].values
    )
    features.update(power_features)
    
    # Pitch error features
    pitch_features = compute_pitch_error_features(
        df['pitch_cmd'].values,
        df['pitch_meas'].values
    )
    features.update(pitch_features)
    
    # FFT features
    fft_features = compute_fft_features(
        df['vibration'].values,
        config.sample_rate,
        config.fft_bands
    )
    features.update(fft_features)
    
    # Blade sector features (always compute, rotor_phase is auto-generated if missing)
    blade_features = compute_blade_sector_features(
        df['vibration'].values,
        df['rotor_phase'].values
    )
    features.update(blade_features)
    
    # Rolling features
    rolling_features = compute_rolling_features(df)
    features.update(rolling_features)
    
    return features


def extract_features_from_file(
    filepath: str,
    config: FeatureConfig = None,
    return_windows: bool = False
) -> Tuple[pd.DataFrame, Optional[List[pd.DataFrame]]]:
    """
    Extract features from all sliding windows in a CSV file.
    
    Returns:
        features_df: DataFrame with one row per window
        windows: List of window DataFrames (if return_windows=True)
    """
    if config is None:
        config = FeatureConfig()
    
    df = pd.read_csv(filepath)
    
    # Ensure required columns
    required_cols = ['wind_speed', 'rotor_rpm', 'power', 'pitch_cmd', 'pitch_meas', 'torque', 'vibration']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Extract features from sliding windows
    all_features = []
    windows = []
    
    n_samples = len(df)
    n_windows = max(1, (n_samples - config.window_size) // config.stride + 1)
    
    for i in range(n_windows):
        start_idx = i * config.stride
        end_idx = min(start_idx + config.window_size, n_samples)
        
        window_df = df.iloc[start_idx:end_idx]
        
        if len(window_df) < config.window_size // 2:
            continue
        
        features = extract_window_features(window_df, config)
        features['window_idx'] = i
        features['start_idx'] = start_idx
        features['end_idx'] = end_idx
        
        all_features.append(features)
        
        if return_windows:
            windows.append(window_df)
    
    features_df = pd.DataFrame(all_features)
    
    if return_windows:
        return features_df, windows
    return features_df, None


def get_feature_names() -> List[str]:
    """Get list of all feature names for model input."""
    # Create dummy data to get feature names
    dummy_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=300, freq='1s'),
        'wind_speed': np.random.rand(300) * 15,
        'rotor_rpm': np.random.rand(300) * 15,
        'power': np.random.rand(300) * 2500,
        'pitch_cmd': np.random.rand(300) * 10,
        'pitch_meas': np.random.rand(300) * 10,
        'torque': np.random.rand(300) * 100,
        'vibration': np.random.rand(300) * 2,
        'rotor_phase': np.random.rand(300) * 360
    })
    
    features = extract_window_features(dummy_df)
    
    # Remove metadata features
    exclude = ['window_idx', 'start_idx', 'end_idx', 'n_samples']
    return [k for k in features.keys() if k not in exclude]


# Feature groups for explainability
FEATURE_GROUPS = {
    'power_residual': [
        'power_residual_mean', 'power_residual_std', 'power_residual_min',
        'power_residual_max', 'power_residual_norm_mean', 'power_residual_norm_std',
        'power_residual_trend'
    ],
    'pitch_error': [
        'pitch_error_mean', 'pitch_error_std', 'pitch_error_abs_mean',
        'pitch_error_max', 'pitch_error_trend', 'pitch_error_range'
    ],
    'vibration_fft': [
        'fft_band_0_energy', 'fft_band_1_energy', 'fft_band_2_energy',
        'fft_band_3_energy', 'fft_peak_freq', 'fft_total_energy'
    ],
    'blade_sector': [
        'blade_1_energy', 'blade_2_energy', 'blade_3_energy',
        'blade_1_ratio', 'blade_2_ratio', 'blade_3_ratio',
        'blade_imbalance_max'
    ],
    'basic_stats': [
        'wind_speed_mean', 'wind_speed_std', 'rotor_rpm_mean',
        'power_mean', 'vibration_mean', 'vibration_std', 'vibration_max'
    ]
}


if __name__ == '__main__':
    # Test feature extraction
    print("Testing feature extraction...")
    
    feature_names = get_feature_names()
    print(f"Total features: {len(feature_names)}")
    print(f"Feature names: {feature_names[:10]}...")
