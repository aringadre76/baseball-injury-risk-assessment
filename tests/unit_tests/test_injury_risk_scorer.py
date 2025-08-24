#!/usr/bin/env python3
"""
Test script for advanced risk scoring functionality
"""

import sys
sys.path.append('src')

from src.injury_risk_scorer import AdvancedRiskScorer, batch_analyze_pitcher_risks
from src.time_series_data_loader import load_sample_pitch_data

def main():
    print("="*60)
    print("TESTING ADVANCED RISK SCORING")
    print("="*60)
    
    # 1. Test basic risk scorer initialization
    print("\n1. TESTING RISK SCORER INITIALIZATION")
    print("-" * 30)
    
    try:
        scorer = AdvancedRiskScorer()
        print("✓ AdvancedRiskScorer initialized successfully")
        
        # Test individual scoring methods with sample data
        session_pitch, pitch_data = load_sample_pitch_data()
        print(f"✓ Loaded sample pitch: {session_pitch}")
        
        # Create sample features dictionary for testing
        sample_features = {
            'elbow_varus_moment': 45.2,
            'max_elbow_extension_velo': 1850.0,
            'shoulder_internal_rotation_moment': 67.3,
            'max_shoulder_internal_rotational_velo': 6200.0,
            'max_rotation_hip_shoulder_separation': 42.1,
            'proper_kinetic_sequence': 1.0,
            'sequence_interval_std': 0.05,
            'rear_force_z_peak': 850.0,
            'rear_ankle_jc_x_lead_ankle_jc_x_asymmetry_index': 12.5
        }
        
        print("  Testing individual scoring components:")
        
        # Test elbow stress composite
        elbow_score = scorer.create_elbow_stress_composite(sample_features)
        print(f"  ✓ Elbow stress score: {elbow_score:.1f}/100")
        
        # Test shoulder stress composite  
        shoulder_score = scorer.create_shoulder_stress_composite(sample_features)
        print(f"  ✓ Shoulder stress score: {shoulder_score:.1f}/100")
        
        # Test kinetic chain efficiency
        efficiency_score = scorer.create_kinetic_chain_efficiency_score(sample_features)
        print(f"  ✓ Kinetic chain efficiency: {efficiency_score:.1f}/100")
        
        # Test movement quality
        quality_score = scorer.create_movement_quality_score(sample_features)
        print(f"  ✓ Movement quality score: {quality_score:.1f}/100")
        
        # Test comprehensive risk profile
        risk_profile = scorer.create_comprehensive_risk_profile(sample_features)
        print(f"  ✓ Comprehensive risk profile generated: {len(risk_profile)} components")
        
        print(f"\n  Risk profile summary:")
        key_scores = ['elbow_stress_score', 'shoulder_stress_score', 'kinetic_chain_efficiency', 
                     'movement_quality', 'ucl_injury_risk', 'shoulder_injury_risk', 
                     'overall_injury_risk', 'risk_category']
        
        for score_name in key_scores:
            if score_name in risk_profile:
                value = risk_profile[score_name]
                if isinstance(value, (int, float)):
                    print(f"    {score_name}: {value:.1f}")
                else:
                    print(f"    {score_name}: {value}")
    
    except Exception as e:
        print(f"✗ Error testing risk scorer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. Test full pitcher risk analysis
    print("\n2. TESTING FULL PITCHER RISK ANALYSIS")
    print("-" * 30)
    
    try:
        risk_analysis = scorer.analyze_pitcher_risk(session_pitch)
        print(f"✓ Full risk analysis completed for pitch {session_pitch}")
        print(f"  Analysis components: {len(risk_analysis)}")
        
        # Display key risk metrics
        print(f"\n  Risk Analysis Results:")
        risk_metrics = ['elbow_stress_score', 'shoulder_stress_score', 'kinetic_chain_efficiency',
                       'movement_quality', 'ucl_injury_risk', 'shoulder_injury_risk', 
                       'overall_injury_risk', 'risk_category']
        
        for metric in risk_metrics:
            if metric in risk_analysis:
                value = risk_analysis[metric]
                if isinstance(value, (int, float)):
                    print(f"    {metric}: {value:.1f}")
                else:
                    print(f"    {metric}: {value}")
        
        # Interpret the results
        overall_risk = risk_analysis.get('overall_injury_risk', 0)
        print(f"\n  Risk Interpretation:")
        if overall_risk < 25:
            interpretation = "Low risk - Good biomechanics with minimal injury concerns"
        elif overall_risk < 50:
            interpretation = "Moderate risk - Some areas for improvement in mechanics"
        elif overall_risk < 75:
            interpretation = "High risk - Significant biomechanical concerns requiring attention"
        else:
            interpretation = "Very high risk - Major biomechanical issues requiring immediate intervention"
        
        print(f"    {interpretation}")
    
    except Exception as e:
        print(f"✗ Error in full pitcher analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Test batch analysis (small sample)
    print("\n3. TESTING BATCH RISK ANALYSIS")
    print("-" * 30)
    
    try:
        # Get a small sample for testing
        from src.time_series_loader import TimeSeriesLoader
        loader = TimeSeriesLoader()
        available_pitches = loader.get_available_pitches()
        test_pitches = available_pitches[:3]  # Test with first 3 pitches
        
        print(f"Testing batch analysis with {len(test_pitches)} pitches")
        
        risk_df = batch_analyze_pitcher_risks(test_pitches, max_pitches=3)
        
        if not risk_df.empty:
            print(f"✓ Batch analysis successful")
            print(f"  Result shape: {risk_df.shape}")
            print(f"  Pitches analyzed: {risk_df['session_pitch'].tolist()}")
            
            # Summary statistics
            print(f"\n  Risk Distribution Summary:")
            if 'risk_category' in risk_df.columns:
                risk_counts = risk_df['risk_category'].value_counts()
                for category, count in risk_counts.items():
                    print(f"    {category} risk: {count} pitchers")
            
            # Average risk scores
            risk_columns = ['elbow_stress_score', 'shoulder_stress_score', 'overall_injury_risk']
            for col in risk_columns:
                if col in risk_df.columns:
                    avg_score = risk_df[col].mean()
                    print(f"    Average {col}: {avg_score:.1f}")
            
            # Identify highest risk pitcher
            if 'overall_injury_risk' in risk_df.columns:
                highest_risk_idx = risk_df['overall_injury_risk'].idxmax()
                highest_risk_pitcher = risk_df.loc[highest_risk_idx, 'session_pitch']
                highest_risk_score = risk_df.loc[highest_risk_idx, 'overall_injury_risk']
                print(f"\n  Highest risk pitcher: {highest_risk_pitcher} (Risk: {highest_risk_score:.1f})")
        else:
            print("✗ Batch analysis returned empty DataFrame")
    
    except Exception as e:
        print(f"✗ Error in batch analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("ADVANCED RISK SCORING TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
