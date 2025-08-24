#!/usr/bin/env python3
"""
Test script for temporal feature extraction
"""

import sys
sys.path.append('src')

from src.temporal_feature_extractor import TemporalFeatureExtractor, extract_pitcher_temporal_features, batch_extract_temporal_features
from src.time_series_data_loader import load_sample_pitch_data

def main():
    print("="*60)
    print("TESTING TEMPORAL FEATURE EXTRACTION")
    print("="*60)
    
    # 1. Test basic temporal feature extractor
    print("\n1. TESTING TEMPORAL FEATURE EXTRACTOR")
    print("-" * 30)
    
    try:
        extractor = TemporalFeatureExtractor(sampling_rate=360.0)
        print("✓ TemporalFeatureExtractor initialized successfully")
        
        # Load sample data
        session_pitch, pitch_data = load_sample_pitch_data()
        print(f"✓ Loaded sample pitch: {session_pitch}")
        print(f"  Available data types: {list(pitch_data.keys())}")
        
        # Test individual feature extraction methods
        print("\n  Testing individual feature extractors:")
        
        # Phase timing features
        timing_features = extractor.extract_phase_timing_features(pitch_data)
        print(f"  ✓ Phase timing features: {len(timing_features)}")
        
        # Velocity sequencing features
        velocity_features = extractor.extract_velocity_sequencing_features(pitch_data)
        print(f"  ✓ Velocity sequencing features: {len(velocity_features)}")
        
        # Force development features
        force_features = extractor.extract_force_development_features(pitch_data)
        print(f"  ✓ Force development features: {len(force_features)}")
        
        # Movement efficiency features
        efficiency_features = extractor.extract_movement_efficiency_features(pitch_data)
        print(f"  ✓ Movement efficiency features: {len(efficiency_features)}")
        
        # Asymmetry features
        asymmetry_features = extractor.extract_asymmetry_features(pitch_data)
        print(f"  ✓ Asymmetry features: {len(asymmetry_features)}")
        
        # Comprehensive features
        all_features = extractor.extract_comprehensive_temporal_features(pitch_data)
        print(f"  ✓ Total temporal features: {len(all_features)}")
        
        # Show sample features
        print(f"\n  Sample temporal features:")
        feature_samples = list(all_features.items())[:10]
        for feature_name, value in feature_samples:
            print(f"    {feature_name}: {value:.4f}")
    
    except Exception as e:
        print(f"✗ Error testing temporal feature extractor: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. Test pitcher-specific feature extraction
    print("\n2. TESTING PITCHER-SPECIFIC EXTRACTION")
    print("-" * 30)
    
    try:
        temporal_features = extract_pitcher_temporal_features(session_pitch)
        print(f"✓ Extracted {len(temporal_features)} temporal features for pitch {session_pitch}")
        
        # Categorize features
        feature_categories = {
            'timing': [f for f in temporal_features.keys() if 'time' in f or 'duration' in f or 'delay' in f],
            'force': [f for f in temporal_features.keys() if 'force' in f or 'rfd' in f],
            'velocity': [f for f in temporal_features.keys() if 'velo' in f or 'velocity' in f],
            'asymmetry': [f for f in temporal_features.keys() if 'asymmetry' in f],
            'efficiency': [f for f in temporal_features.keys() if 'efficiency' in f or 'stability' in f],
            'other': [f for f in temporal_features.keys() if not any(keyword in f for keyword in ['time', 'duration', 'delay', 'force', 'rfd', 'velo', 'velocity', 'asymmetry', 'efficiency', 'stability'])]
        }
        
        print(f"\n  Feature breakdown:")
        total_categorized = 0
        for category, features in feature_categories.items():
            if features:
                print(f"    {category}: {len(features)} features")
                total_categorized += len(features)
        
        print(f"  Total categorized: {total_categorized}/{len(temporal_features)}")
        
        # Show key injury-relevant features
        injury_relevant_keywords = ['elbow', 'shoulder', 'sequenc', 'phase', 'asymmetry']
        injury_features = [f for f in temporal_features.keys() 
                          if any(keyword in f.lower() for keyword in injury_relevant_keywords)]
        
        print(f"\n  Key injury-relevant features ({len(injury_features)}):")
        for feature in injury_features[:8]:  # Show first 8
            print(f"    {feature}: {temporal_features[feature]:.4f}")
    
    except Exception as e:
        print(f"✗ Error in pitcher-specific extraction: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Test batch extraction (small sample)
    print("\n3. TESTING BATCH EXTRACTION")
    print("-" * 30)
    
    try:
        # Get a small sample of pitches for testing
        from src.time_series_loader import TimeSeriesLoader
        loader = TimeSeriesLoader()
        available_pitches = loader.get_available_pitches()
        
        # Test with first 3 pitches
        test_pitches = available_pitches[:3]
        print(f"Testing batch extraction with {len(test_pitches)} pitches")
        
        features_df = batch_extract_temporal_features(test_pitches, max_pitches=3)
        
        if not features_df.empty:
            print(f"✓ Batch extraction successful")
            print(f"  Result shape: {features_df.shape}")
            print(f"  Pitches processed: {features_df['session_pitch'].tolist()}")
            print(f"  Features per pitch: {features_df.shape[1] - 1}")  # Exclude session_pitch column
            
            # Check for missing values
            missing_count = features_df.isnull().sum().sum()
            print(f"  Missing values: {missing_count}")
            
            # Show feature summary
            numeric_cols = features_df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                print(f"  Feature value ranges:")
                for col in numeric_cols[:5]:  # Show first 5 numeric features
                    col_data = features_df[col]
                    print(f"    {col}: {col_data.min():.3f} to {col_data.max():.3f}")
        else:
            print("✗ Batch extraction returned empty DataFrame")
    
    except Exception as e:
        print(f"✗ Error in batch extraction: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("TEMPORAL FEATURE EXTRACTION TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
