"""
Synthetic SCADA Data Generator for Wind Turbine Blade Fault Detection

Generates physics-guided synthetic telemetry data with injected faults:
- Erosion: Sustained power residual loss
- Pitch Drift: Increasing pitch error over time
- Cyclic Vibration: Periodic vibration spikes aligned to blade phase

Each generated file includes ground-truth labels for supervised training.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
import os
from datetime import datetime, timedelta

# Constants
BLADE_PHASES = {1: 0, 2: 120, 3: 240}  # Degrees
RATED_POWER_KW = 2500
CUT_IN_WIND = 3.5
RATED_WIND = 12.5
CUT_OUT_WIND = 25.0


def power_curve(wind_speed: np.ndarray) -> np.ndarray:
    """
    Theoretical power curve for a wind turbine.
    P = 0.5 * rho * A * Cp * v^3 (simplified)
    """
    power = np.zeros_like(wind_speed, dtype=float)
    
    # Region 1: Below cut-in
    mask_off = wind_speed < CUT_IN_WIND
    power[mask_off] = 0
    
    # Region 2: Cubic region (cut-in to rated)
    mask_cubic = (wind_speed >= CUT_IN_WIND) & (wind_speed < RATED_WIND)
    normalized = (wind_speed[mask_cubic] - CUT_IN_WIND) / (RATED_WIND - CUT_IN_WIND)
    power[mask_cubic] = RATED_POWER_KW * (normalized ** 3)
    
    # Region 3: Rated power (rated to cut-out)
    mask_rated = (wind_speed >= RATED_WIND) & (wind_speed <= CUT_OUT_WIND)
    power[mask_rated] = RATED_POWER_KW
    
    # Region 4: Above cut-out (shutdown)
    mask_shutdown = wind_speed > CUT_OUT_WIND
    power[mask_shutdown] = 0
    
    return power


def generate_healthy_data(
    duration_hours: float = 24,
    sample_rate_hz: float = 1.0,
    noise_level: float = 0.05,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate healthy turbine SCADA data.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = int(duration_hours * 3600 * sample_rate_hz)
    timestamps = pd.date_range(
        start=datetime.now().replace(microsecond=0),
        periods=n_samples,
        freq=f'{int(1000/sample_rate_hz)}ms'
    )
    
    # Wind speed: slowly varying with gusts
    base_wind = 10 + 3 * np.sin(2 * np.pi * np.arange(n_samples) / (3600 * sample_rate_hz))
    gusts = np.random.normal(0, 1.5, n_samples)
    wind_speed = np.clip(base_wind + gusts, 0, 28)
    
    # Rotor RPM: proportional to wind in partial load
    rotor_rpm_base = np.clip(wind_speed * 1.2, 0, 15)  # Max 15 RPM
    rotor_rpm = rotor_rpm_base + np.random.normal(0, 0.1, n_samples)
    
    # Power: from power curve with noise
    expected_power = power_curve(wind_speed)
    power = expected_power * (1 + np.random.normal(0, noise_level, n_samples))
    power = np.clip(power, 0, RATED_POWER_KW * 1.05)
    
    # Pitch command and measured (should track closely for healthy)
    pitch_cmd = np.clip(15 - wind_speed, 0, 25)  # Simplified pitch schedule
    pitch_meas = pitch_cmd + np.random.normal(0, 0.2, n_samples)
    
    # Torque: proportional to power / rpm
    torque = np.where(rotor_rpm > 0.1, power / (rotor_rpm + 0.1) * 0.1, 0)
    torque += np.random.normal(0, torque.std() * noise_level, n_samples)
    
    # Vibration: baseline with small random variation
    vibration = 0.5 + 0.1 * np.random.randn(n_samples)
    vibration = np.clip(vibration, 0.1, 2.0)
    
    # Temperature: gearbox/nacelle temp correlated with power output
    base_temp = 35 + (power / RATED_POWER_KW) * 25  # 35-60°C based on load
    temperature = base_temp + np.random.normal(0, 2, n_samples)
    temperature = np.clip(temperature, 20, 80)
    
    # Rotor phase (for blade attribution)
    rotor_phase = (np.cumsum(rotor_rpm / 60 * 360 / sample_rate_hz)) % 360
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'wind_speed': np.round(wind_speed, 2),
        'rotor_rpm': np.round(rotor_rpm, 2),
        'power': np.round(power, 2),
        'pitch_cmd': np.round(pitch_cmd, 2),
        'pitch_meas': np.round(pitch_meas, 2),
        'torque': np.round(torque, 2),
        'vibration': np.round(vibration, 3),
        'temperature': np.round(temperature, 1),
        'rotor_phase': np.round(rotor_phase, 1)
    })
    
    return df


def inject_erosion_fault(
    df: pd.DataFrame,
    severity: float = 0.5,
    blade_id: int = 1,
    start_pct: float = 0.2
) -> Tuple[pd.DataFrame, dict]:
    """
    Inject blade erosion fault: gradual power loss.
    Severity 0-1 maps to 5-20% power loss.
    """
    df = df.copy()
    n = len(df)
    start_idx = int(n * start_pct)
    
    # Power degradation ramp
    power_loss_pct = 0.05 + severity * 0.15  # 5-20%
    degradation = np.zeros(n)
    ramp = np.linspace(0, power_loss_pct, n - start_idx)
    degradation[start_idx:] = ramp
    
    df['power'] = df['power'] * (1 - degradation)
    
    # Add slight vibration increase on affected blade sector
    blade_phase = BLADE_PHASES[blade_id]
    phase_mask = np.abs((df['rotor_phase'] - blade_phase + 180) % 360 - 180) < 30
    df.loc[phase_mask & (df.index >= start_idx), 'vibration'] *= (1 + severity * 0.3)
    
    labels = {
        'fault_type': 'erosion',
        'blade_id': blade_id,
        'severity': severity,
        'fault_start_idx': start_idx,
        'power_loss_pct': power_loss_pct
    }
    
    return df, labels


def inject_pitch_drift_fault(
    df: pd.DataFrame,
    severity: float = 0.5,
    blade_id: int = 2,
    start_pct: float = 0.2
) -> Tuple[pd.DataFrame, dict]:
    """
    Inject pitch drift fault: increasing pitch error.
    """
    df = df.copy()
    n = len(df)
    start_idx = int(n * start_pct)
    
    # Pitch error grows over time
    max_drift = 2 + severity * 6  # 2-8 degrees drift
    drift = np.zeros(n)
    ramp = np.linspace(0, max_drift, n - start_idx)
    drift[start_idx:] = ramp
    
    # Add oscillation
    oscillation = 0.5 * np.sin(2 * np.pi * np.arange(n) / 100)
    df['pitch_meas'] = df['pitch_meas'] + drift + oscillation * severity
    
    # Power loss from pitch error
    pitch_error = np.abs(df['pitch_meas'] - df['pitch_cmd'])
    power_penalty = np.clip(pitch_error / 10, 0, 0.15)
    df['power'] = df['power'] * (1 - power_penalty)
    
    labels = {
        'fault_type': 'pitch_drift',
        'blade_id': blade_id,
        'severity': severity,
        'fault_start_idx': start_idx,
        'max_drift_deg': max_drift
    }
    
    return df, labels


def inject_cyclic_vibration_fault(
    df: pd.DataFrame,
    severity: float = 0.5,
    blade_id: int = 3,
    start_pct: float = 0.2
) -> Tuple[pd.DataFrame, dict]:
    """
    Inject cyclic vibration fault: periodic spikes aligned to blade phase.
    """
    df = df.copy()
    n = len(df)
    start_idx = int(n * start_pct)
    
    # Vibration spikes when blade passes (1P frequency)
    blade_phase = BLADE_PHASES[blade_id]
    phase_distance = np.abs((df['rotor_phase'] - blade_phase + 180) % 360 - 180)
    
    # Create spike profile (narrow peak)
    spike_width = 15  # degrees
    spike_amplitude = 1.5 + severity * 4  # 1.5-5.5x baseline
    spike_profile = np.exp(-0.5 * (phase_distance / spike_width) ** 2)
    
    vibration_increase = np.zeros(n)
    vibration_increase[start_idx:] = spike_profile[start_idx:] * spike_amplitude
    
    df['vibration'] = df['vibration'] + vibration_increase
    
    # Slight power fluctuation
    power_fluctuation = spike_profile * severity * 0.02
    df.loc[start_idx:, 'power'] *= (1 - power_fluctuation[start_idx:])
    
    labels = {
        'fault_type': 'cyclic_vibration',
        'blade_id': blade_id,
        'severity': severity,
        'fault_start_idx': start_idx,
        'spike_amplitude': spike_amplitude
    }
    
    return df, labels


def generate_dataset(
    output_dir: str,
    n_healthy: int = 30,
    n_faulty_per_type: int = 20,
    duration_hours: float = 8,
    sample_rate_hz: float = 1.0,
    seed: int = 42
) -> dict:
    """
    Generate complete dataset with train/val/test split.
    
    Returns metadata about generated files.
    """
    np.random.seed(seed)
    output_path = Path(output_dir)
    
    fault_types = ['erosion', 'pitch_drift', 'cyclic_vibration']
    inject_functions = {
        'erosion': inject_erosion_fault,
        'pitch_drift': inject_pitch_drift_fault,
        'cyclic_vibration': inject_cyclic_vibration_fault
    }
    
    all_files = []
    metadata = []
    
    # Generate healthy data
    print("Generating healthy data...")
    for i in range(n_healthy):
        df = generate_healthy_data(
            duration_hours=duration_hours,
            sample_rate_hz=sample_rate_hz,
            noise_level=0.03 + np.random.rand() * 0.04,
            seed=seed + i
        )
        filename = f"healthy_{i:03d}.csv"
        all_files.append((filename, df, {
            'fault_type': 'normal',
            'blade_id': None,
            'severity': 0.0
        }))
    
    # Generate faulty data
    print("Generating faulty data...")
    file_idx = 0
    for fault_type in fault_types:
        inject_fn = inject_functions[fault_type]
        
        for i in range(n_faulty_per_type):
            # Vary parameters
            blade_id = np.random.choice([1, 2, 3])
            severity = 0.3 + np.random.rand() * 0.6  # 0.3-0.9
            start_pct = 0.1 + np.random.rand() * 0.3  # 10-40% into file
            
            df = generate_healthy_data(
                duration_hours=duration_hours,
                sample_rate_hz=sample_rate_hz,
                noise_level=0.03 + np.random.rand() * 0.04,
                seed=seed + 1000 + file_idx
            )
            
            df, labels = inject_fn(df, severity=severity, blade_id=blade_id, start_pct=start_pct)
            
            filename = f"{fault_type}_{i:03d}_b{blade_id}_s{int(severity*100)}.csv"
            all_files.append((filename, df, labels))
            file_idx += 1
    
    # Shuffle and split
    np.random.shuffle(all_files)
    n_total = len(all_files)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]
    
    # Save files
    splits = [
        ('train', train_files),
        ('val', val_files),
        ('test', test_files)
    ]
    
    for split_name, files in splits:
        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        labels_list = []
        for filename, df, labels in files:
            filepath = split_dir / filename
            df.to_csv(filepath, index=False)
            labels_list.append({
                'filename': filename,
                **labels
            })
        
        # Save labels
        labels_df = pd.DataFrame(labels_list)
        labels_df.to_csv(split_dir / 'labels.csv', index=False)
        
        print(f"  {split_name}: {len(files)} files")
        metadata.append({
            'split': split_name,
            'n_files': len(files),
            'n_healthy': sum(1 for _, _, l in files if l['fault_type'] == 'normal'),
            'n_faulty': sum(1 for _, _, l in files if l['fault_type'] != 'normal')
        })
    
    # Create demo_test.csv (guaranteed faulty for demo)
    print("Creating demo_test.csv...")
    demo_df = generate_healthy_data(duration_hours=4, sample_rate_hz=1.0, seed=99999)
    demo_df, demo_labels = inject_cyclic_vibration_fault(demo_df, severity=0.65, blade_id=2, start_pct=0.15)
    demo_df.to_csv(output_path / 'test' / 'demo_test.csv', index=False)
    
    # Update test labels
    test_labels = pd.read_csv(output_path / 'test' / 'labels.csv')
    demo_row = pd.DataFrame([{
        'filename': 'demo_test.csv',
        'fault_type': demo_labels['fault_type'],
        'blade_id': demo_labels['blade_id'],
        'severity': demo_labels['severity']
    }])
    test_labels = pd.concat([test_labels, demo_row], ignore_index=True)
    test_labels.to_csv(output_path / 'test' / 'labels.csv', index=False)
    
    return {'splits': metadata, 'demo_labels': demo_labels}


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic SCADA data')
    parser.add_argument('--output', type=str, default='../synthetic_data', help='Output directory')
    parser.add_argument('--n-healthy', type=int, default=30, help='Number of healthy files')
    parser.add_argument('--n-faulty', type=int, default=20, help='Faulty files per type')
    parser.add_argument('--duration', type=float, default=8, help='Duration in hours')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print(f"Generating dataset to {args.output}")
    result = generate_dataset(
        output_dir=args.output,
        n_healthy=args.n_healthy,
        n_faulty_per_type=args.n_faulty,
        duration_hours=args.duration,
        seed=args.seed
    )
    print("\nDataset generation complete!")
    print(f"Splits: {result['splits']}")
