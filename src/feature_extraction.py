"""
Feature Extraction Module for HMM Activity Recognition

This module contains functions for extracting time-domain and frequency-domain
features from accelerometer and gyroscope sensor data.
"""

import pandas as pd
import numpy as np
from scipy.fft import fft


def extract_time_features(window):
    """
    Extract time-domain features from a window of sensor data.
    
    Args:
        window: DataFrame with columns: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
        
    Returns:
        Dictionary of time-domain features
    """
    features = {}
    axes = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
    
    for axis in axes:
        features[f"{axis}_mean"] = np.mean(window[axis])
        features[f"{axis}_std"] = np.std(window[axis])
        features[f"{axis}_var"] = np.var(window[axis])
    
    # Signal Magnitude Area (SMA) for accelerometer
    features["accel_sma"] = np.sum(
        np.abs(window["accel_x"]) + np.abs(window["accel_y"]) + np.abs(window["accel_z"])
    ) / len(window)
    
    # Correlations between accelerometer axes
    features["accel_xy_corr"] = np.corrcoef(window["accel_x"], window["accel_y"])[0, 1]
    features["accel_xz_corr"] = np.corrcoef(window["accel_x"], window["accel_z"])[0, 1]
    features["accel_yz_corr"] = np.corrcoef(window["accel_y"], window["accel_z"])[0, 1]
    
    return features


def extract_frequency_features(window):
    """
    Extract frequency-domain features from a window of sensor data.
    
    Args:
        window: DataFrame with columns: accel_x, accel_y, accel_z
        
    Returns:
        Dictionary of frequency-domain features
    """
    features = {}
    for axis in ["accel_x", "accel_y", "accel_z"]:
        sig = window[axis].values
        fft_vals = np.abs(fft(sig))
        features[f"{axis}_dom_freq"] = np.argmax(fft_vals)
    return features


def create_windows(df, window_size, step_size):
    """
    Split dataframe into overlapping windows.
    
    Args:
        df: DataFrame with sensor data
        window_size: Number of samples per window
        step_size: Number of samples to slide the window
        
    Returns:
        List of window DataFrames
    """
    windows = []
    for start in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[start:start + window_size]
        windows.append(window)
    return windows


def extract_features_from_recording(accel_file, gyro_file, activity, recording_name,
                                   window_size=50, step_size=25):
    """
    Extract features from a single recording (pair of accelerometer and gyroscope files).
    
    Args:
        accel_file: Path to Accelerometer.csv file
        gyro_file: Path to Gyroscope.csv file
        activity: Activity label (e.g., 'walking', 'jumping')
        recording_name: Name of the recording
        window_size: Number of samples per window (default: 50)
        step_size: Window slide step size (default: 25)
        
    Returns:
        Dictionary of features for this recording (averaged across all windows)
    """
    # Load sensor data
    accel_df = pd.read_csv(accel_file)
    gyro_df = pd.read_csv(gyro_file)
    
    # Merge on time
    merged = pd.merge(accel_df, gyro_df, on='time', how='inner', suffixes=('', '_gyro'))
    
    # Rename columns to standard format
    # Handle both raw sensor files (x,y,z) and processed files (accel_x, accel_y, accel_z)
    if 'x' in merged.columns and 'x_gyro' in merged.columns:
        # Raw sensor files format
        merged = merged.rename(columns={
            'x': 'accel_x',
            'y': 'accel_y',
            'z': 'accel_z',
            'x_gyro': 'gyro_x',
            'y_gyro': 'gyro_y',
            'z_gyro': 'gyro_z'
        })
    elif 'accel_x' not in merged.columns:
        # Couldn't find expected columns
        raise ValueError(f"Could not find expected sensor columns in {accel_file}. Found: {merged.columns.tolist()}")
    
    # Verify we have all required columns
    required_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    missing_cols = [col for col in required_cols if col not in merged.columns]
    if missing_cols:
        raise ValueError(f"Missing columns after renaming: {missing_cols}. Available: {merged.columns.tolist()}")
    
    # Skip if too short
    if len(merged) < window_size:
        raise ValueError(f"Recording too short: {len(merged)} < {window_size}")
    
    # Create windows
    windows = create_windows(merged, window_size, step_size)
    
    # Extract features from all windows and average
    all_window_features = []
    for window in windows:
        time_feats = extract_time_features(window)
        freq_feats = extract_frequency_features(window)
        window_features = {**time_feats, **freq_feats}
        all_window_features.append(window_features)
    
    # Average features across all windows
    avg_features = {}
    if all_window_features:
        feature_keys = all_window_features[0].keys()
        for key in feature_keys:
            values = [wf[key] for wf in all_window_features]
            avg_features[key] = np.mean(values)
    
    # Add metadata
    avg_features['activity'] = activity
    avg_features['recording'] = recording_name
    
    return avg_features


def extract_features_batch(data, window_size=50, step_size=25):
    """
    Extract features from merged sensor data for multiple recordings.
    
    Args:
        data: DataFrame with columns: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, recording, activity
        window_size: Number of samples per window
        step_size: Window slide step size
        
    Returns:
        DataFrame with extracted features
    """
    all_features = []
    
    for recording_id, group in data.groupby('recording'):
        # Skip short recordings
        if len(group) < window_size:
            print(f"Skipping short recording: {recording_id}")
            continue
        
        windows = create_windows(group, window_size, step_size)
        
        for window in windows:
            time_feats = extract_time_features(window)
            freq_feats = extract_frequency_features(window)
            features = {**time_feats, **freq_feats}
            features['activity'] = window['activity'].iloc[0]
            features['recording'] = recording_id
            all_features.append(features)
    
    return pd.DataFrame(all_features)
