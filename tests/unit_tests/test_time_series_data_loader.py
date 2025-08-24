#!/usr/bin/env python3
"""
Test script for TimeSeriesLoader functionality
"""

import sys
sys.path.append('src')

from src.time_series_data_loader import TimeSeriesLoader, get_dataset_overview, load_sample_pitch_data

def main():
    print("="*60)
    print("TESTING TIME-SERIES DATA LOADER")
    print("="*60)
    
    # 1. Test dataset overview
    print("\n1. DATASET OVERVIEW")
    print("-" * 30)
    
    try:
        overview = get_dataset_overview()
        print(f"Total pitches: {overview['total_pitches']}")
        print(f"Available data types: {overview['data_types']}")
        
        if overview.get('sample_pitch_summary'):
            print(f"\nSample pitch summaries:")
            for i, pitch_summary in enumerate(overview['sample_pitch_summary'][:3]):
                print(f"  Pitch {i+1}: {pitch_summary['session_pitch']} "
                      f"({pitch_summary['duration_seconds']:.2f}s, "
                      f"{pitch_summary['sample_count']} samples)")
    
    except Exception as e:
        print(f"Error getting overview: {e}")
        return
    
    # 2. Test TimeSeriesLoader directly
    print("\n2. TESTING TIME-SERIES LOADER")
    print("-" * 30)
    
    try:
        loader = TimeSeriesLoader()
        available_pitches = loader.get_available_pitches()
        print(f"Available pitches: {len(available_pitches)}")
        
        if available_pitches:
            # Test loading one pitch
            test_pitch = available_pitches[0]
            print(f"\nTesting with pitch: {test_pitch}")
            
            # Load landmarks data
            landmarks = loader.load_data_type('landmarks')
            print(f"Landmarks data shape: {landmarks.shape}")
            
            # Load specific pitch data
            pitch_data = loader.get_pitch_data(test_pitch, ['landmarks', 'joint_angles'])
            print(f"Loaded {len(pitch_data)} data types for pitch {test_pitch}")
            
            for data_type, df in pitch_data.items():
                print(f"  {data_type}: {df.shape}")
            
            # Test timing events
            timing_events = loader.get_timing_events(test_pitch)
            print(f"Timing events: {list(timing_events.keys())}")
    
    except Exception as e:
        print(f"Error testing loader: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Test sample data loading
    print("\n3. TESTING SAMPLE DATA LOADING")
    print("-" * 30)
    
    try:
        session_pitch, pitch_data = load_sample_pitch_data()
        print(f"Loaded sample pitch: {session_pitch}")
        print(f"Data types: {list(pitch_data.keys())}")
        
        for data_type, df in pitch_data.items():
            if len(df) > 0:
                print(f"  {data_type}: {df.shape[0]} time points over {df['time'].max() - df['time'].min():.2f}s")
            else:
                print(f"  {data_type}: No data")
    
    except Exception as e:
        print(f"Error loading sample data: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("TIME-SERIES LOADER TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
