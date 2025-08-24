#!/usr/bin/env python3
"""
Test script for signal analysis functionality
"""

import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
from src.time_series_loader import load_sample_pitch_data
from src.biomechanical_signal_analyzer import SignalAnalyzer, analyze_pitch_signals

def main():
    print("="*60)
    print("TESTING SIGNAL ANALYSIS")
    print("="*60)
    
    # 1. Test basic signal analyzer
    print("\n1. TESTING SIGNAL ANALYZER BASICS")
    print("-" * 30)
    
    analyzer = SignalAnalyzer(sampling_rate=360.0)
    
    # Create test signal
    t = np.linspace(0, 2, 720)  # 2 seconds at 360 Hz
    test_signal = np.sin(2 * np.pi * 5 * t) + 0.3 * np.sin(2 * np.pi * 15 * t) + 0.1 * np.random.randn(len(t))
    
    # Test filtering
    filtered_signal = analyzer.apply_filter(test_signal, 'lowpass', cutoff_freq=10.0)
    print(f"Original signal length: {len(test_signal)}")
    print(f"Filtered signal length: {len(filtered_signal)}")
    
    # Test peak detection
    peak_results = analyzer.find_peaks_enhanced(test_signal)
    print(f"Found {len(peak_results.get('peaks', []))} peaks")
    print(f"Found {len(peak_results.get('valleys', []))} valleys")
    
    # Test feature extraction
    statistical_features = analyzer.extract_statistical_features(test_signal)
    print(f"Extracted {len(statistical_features)} statistical features")
    
    frequency_features = analyzer.extract_frequency_features(test_signal)
    print(f"Extracted {len(frequency_features)} frequency features")
    
    timing_features = analyzer.extract_timing_features(test_signal, t)
    print(f"Extracted {len(timing_features)} timing features")
    
    # 2. Test with real biomechanical data
    print("\n2. TESTING WITH REAL BIOMECHANICAL DATA")
    print("-" * 30)
    
    try:
        # Load sample pitch data
        session_pitch, pitch_data = load_sample_pitch_data()
        print(f"Loaded pitch: {session_pitch}")
        print(f"Available data types: {list(pitch_data.keys())}")
        
        # Test with landmarks data
        if 'landmarks' in pitch_data and len(pitch_data['landmarks']) > 0:
            landmarks_df = pitch_data['landmarks']
            print(f"Landmarks data shape: {landmarks_df.shape}")
            
            # Test on a specific signal (elbow position)
            if 'elbow_jc_x' in landmarks_df.columns:
                elbow_signal = landmarks_df['elbow_jc_x'].values
                time_array = landmarks_df['time'].values
                
                # Get timing events
                timing_events = {}
                timing_cols = ['pkh_time', 'fp_10_time', 'MER_time', 'BR_time']
                for col in timing_cols:
                    if col in landmarks_df.columns and not landmarks_df[col].isna().all():
                        timing_events[col] = landmarks_df[col].iloc[0]
                
                print(f"\nAnalyzing elbow X position:")
                print(f"  Signal length: {len(elbow_signal)}")
                print(f"  Time range: {time_array[0]:.3f} to {time_array[-1]:.3f} seconds")
                print(f"  Timing events: {list(timing_events.keys())}")
                
                # Comprehensive analysis
                elbow_features = analyzer.analyze_signal_comprehensive(
                    elbow_signal, time_array, timing_events, "elbow_x"
                )
                print(f"  Extracted {len(elbow_features)} total features")
                
                # Show some key features
                key_features = ['elbow_x_mean', 'elbow_x_std', 'elbow_x_max', 'elbow_x_min',
                               'elbow_x_duration', 'elbow_x_time_to_peak', 'elbow_x_num_peaks']
                print(f"\n  Key features:")
                for feature in key_features:
                    if feature in elbow_features:
                        print(f"    {feature}: {elbow_features[feature]:.4f}")
    
    except Exception as e:
        print(f"Error testing with real data: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Test comprehensive pitch analysis
    print("\n3. TESTING COMPREHENSIVE PITCH ANALYSIS")
    print("-" * 30)
    
    try:
        # Load sample data again
        session_pitch, pitch_data = load_sample_pitch_data()
        
        # Analyze subset of signals to avoid overwhelming output
        test_signals = ['elbow_jc_x', 'shoulder_jc_y', 'wrist_jc_z']
        
        all_features = analyze_pitch_signals(
            pitch_data, 
            signal_columns=test_signals,
            sampling_rate=360.0
        )
        
        print(f"Analyzed pitch {session_pitch}")
        print(f"Total features extracted: {len(all_features)}")
        
        # Group features by type
        feature_types = {}
        for feature_name in all_features.keys():
            # Extract feature type from name
            if '_mean' in feature_name:
                feature_type = 'statistical'
            elif '_frequency' in feature_name or '_power' in feature_name or '_spectral' in feature_name:
                feature_type = 'frequency'
            elif '_time' in feature_name or '_duration' in feature_name:
                feature_type = 'timing'
            elif '_peaks' in feature_name or '_valleys' in feature_name:
                feature_type = 'peak_analysis'
            else:
                feature_type = 'other'
            
            if feature_type not in feature_types:
                feature_types[feature_type] = 0
            feature_types[feature_type] += 1
        
        print(f"\nFeature breakdown:")
        for feature_type, count in feature_types.items():
            print(f"  {feature_type}: {count} features")
        
        # Show sample features from each category
        print(f"\nSample features:")
        for feature_type in ['statistical', 'timing', 'frequency']:
            matching_features = [f for f in all_features.keys() if feature_type in f or 
                               ('_mean' in f and feature_type == 'statistical') or
                               ('_time' in f and feature_type == 'timing') or
                               ('_frequency' in f and feature_type == 'frequency')]
            if matching_features:
                sample_feature = matching_features[0]
                print(f"  {sample_feature}: {all_features[sample_feature]:.4f}")
    
    except Exception as e:
        print(f"Error in comprehensive analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("SIGNAL ANALYSIS TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
