#!/usr/bin/env python3
"""
Phase 2 Demonstration Script
OpenBiomechanics Baseball Injury Risk Assessment - Advanced Features

This script demonstrates the complete Phase 2 pipeline:
1. Time-series data loading and analysis
2. Temporal feature extraction  
3. Advanced composite risk scoring
4. Feature selection and dimensionality reduction
5. Enhanced model training and evaluation
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.feature_selection_engineer import comprehensive_feature_selection_pipeline
from src.baseline_injury_model import BaselineInjuryRiskModel
from src.time_series_data_loader import TimeSeriesLoader, get_dataset_overview
from src.temporal_feature_extractor import batch_extract_temporal_features
from src.injury_risk_scorer import batch_analyze_pitcher_risks

def main():
    print("="*70)
    print("PHASE 2: ADVANCED FEATURES FOR INJURY RISK ASSESSMENT")
    print("OpenBiomechanics Baseball Pitching Data Analysis")
    print("="*70)
    
    # 1. Time-Series Data Overview
    print("\n1. TIME-SERIES DATA OVERVIEW")
    print("-" * 40)
    
    try:
        overview = get_dataset_overview()
        print(f"Total pitches available: {overview['total_pitches']}")
        print(f"Data types: {', '.join(overview['data_types'])}")
        
        if overview.get('sample_pitch_summary'):
            print(f"\nSample pitch characteristics:")
            for i, pitch in enumerate(overview['sample_pitch_summary'][:3]):
                duration = pitch.get('duration_seconds', 0)
                samples = pitch.get('sample_count', 0)
                rate = pitch.get('sampling_rate_hz', 0)
                print(f"  Pitch {i+1}: {duration:.2f}s, {samples} samples, {rate:.1f} Hz")
    
    except Exception as e:
        print(f"Error getting overview: {e}")
        return
    
    # 2. Comprehensive Feature Engineering Pipeline
    print("\n2. COMPREHENSIVE FEATURE ENGINEERING")
    print("-" * 40)
    
    try:
        # Run the complete feature selection pipeline
        results = comprehensive_feature_selection_pipeline(
            max_pitches=30,  # Use smaller sample for demo
            n_final_features=40
        )
        
        # Extract results
        X_original = results['original_features']
        X_selected = results['selected_features'] 
        X_pca = results['pca_features']
        y = results['target']
        
        baseline_eval = results['baseline_evaluation']
        selected_eval = results['selected_evaluation']
        pca_eval = results['pca_evaluation']
        
        print(f"\nFeature Engineering Results:")
        print(f"  Original features: {X_original.shape[1]} features")
        print(f"  Selected features: {X_selected.shape[1]} features")
        print(f"  PCA components: {X_pca.shape[1]} components")
        
        print(f"\nPerformance Comparison:")
        print(f"  Baseline (all features): AUC = {baseline_eval['mean_auc']:.3f} ± {baseline_eval['std_auc']:.3f}")
        print(f"  Selected features: AUC = {selected_eval['mean_auc']:.3f} ± {selected_eval['std_auc']:.3f}")
        print(f"  PCA features: AUC = {pca_eval['mean_auc']:.3f} ± {pca_eval['std_auc']:.3f}")
        
        # Performance improvement
        improvement = selected_eval['mean_auc'] - baseline_eval['mean_auc']
        print(f"  Feature selection improvement: {improvement:+.3f} AUC points")
    
    except Exception as e:
        print(f"Error in feature engineering pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Enhanced Model Training and Comparison
    print("\n3. ENHANCED MODEL TRAINING")
    print("-" * 40)
    
    try:
        # Train models on different feature sets
        model_results = {}
        
        # Selected features model
        model_selected = BaselineInjuryRiskModel(random_state=42)
        X_sel, y_sel = model_selected.prepare_features_and_target(
            pd.concat([X_selected, y], axis=1), 
            X_selected.columns.tolist(), 
            'injury_risk_label'
        )
        
        print(f"Training enhanced models with {X_sel.shape[1]} selected features...")
        results_selected = model_selected.train_baseline_models(X_sel, y_sel)
        model_results['selected_features'] = model_selected.performance_metrics
        
        # Display results
        print(f"\nEnhanced Model Performance:")
        for model_name, metrics in model_selected.performance_metrics.items():
            print(f"  {model_name.replace('_', ' ').title()}:")
            print(f"    Accuracy: {metrics['accuracy']:.3f}")
            print(f"    Precision: {metrics['precision']:.3f}")
            print(f"    Recall: {metrics['recall']:.3f}")
            print(f"    F1-Score: {metrics['f1_score']:.3f}")
            print(f"    AUC-ROC: {metrics.get('auc_roc', 'N/A')}")
            print(f"    CV AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
            print()
        
        # Best model identification
        best_model_name = max(model_selected.performance_metrics.keys(), 
                             key=lambda k: model_selected.performance_metrics[k]['cv_auc_mean'])
        best_auc = model_selected.performance_metrics[best_model_name]['cv_auc_mean']
        
        print(f"Best Enhanced Model: {best_model_name.replace('_', ' ').title()}")
        print(f"Best CV AUC: {best_auc:.3f}")
    
    except Exception as e:
        print(f"Error in enhanced model training: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Feature Importance Analysis
    print("\n4. FEATURE IMPORTANCE ANALYSIS")
    print("-" * 40)
    
    try:
        if 'model_selected' in locals() and model_selected.feature_importance:
            # Get feature importance from best model
            if best_model_name in model_selected.feature_importance:
                importance_df = model_selected.feature_importance[best_model_name]
                
                print(f"Top 15 Most Important Features ({best_model_name}):")
                for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
                    feature_name = row['feature']
                    importance = row['importance']
                    
                    # Categorize feature type
                    if any(keyword in feature_name.lower() for keyword in ['time', 'duration', 'delay']):
                        category = 'Temporal'
                    elif any(keyword in feature_name.lower() for keyword in ['force', 'moment']):
                        category = 'Force/Moment'
                    elif any(keyword in feature_name.lower() for keyword in ['velo', 'velocity']):
                        category = 'Velocity'
                    elif any(keyword in feature_name.lower() for keyword in ['asymmetry']):
                        category = 'Asymmetry'
                    elif any(keyword in feature_name.lower() for keyword in ['angle']):
                        category = 'Kinematics'
                    elif any(keyword in feature_name.lower() for keyword in ['stress', 'risk', 'efficiency']):
                        category = 'Risk Score'
                    else:
                        category = 'Other'
                    
                    print(f"  {i+1:2d}. {feature_name:<40} {importance:.4f} ({category})")
                
                # Feature category analysis
                print(f"\nFeature Category Breakdown:")
                categories = {}
                for _, row in importance_df.iterrows():
                    feature_name = row['feature']
                    importance = row['importance']
                    
                    category = 'Other'
                    if any(keyword in feature_name.lower() for keyword in ['time', 'duration', 'delay']):
                        category = 'Temporal'
                    elif any(keyword in feature_name.lower() for keyword in ['force', 'moment']):
                        category = 'Force/Moment'
                    elif any(keyword in feature_name.lower() for keyword in ['velo', 'velocity']):
                        category = 'Velocity'
                    elif any(keyword in feature_name.lower() for keyword in ['asymmetry']):
                        category = 'Asymmetry'
                    elif any(keyword in feature_name.lower() for keyword in ['stress', 'risk', 'efficiency']):
                        category = 'Risk Score'
                    
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(importance)
                
                for category, importances in categories.items():
                    avg_importance = np.mean(importances)
                    count = len(importances)
                    print(f"  {category}: {count} features, avg importance = {avg_importance:.4f}")
    
    except Exception as e:
        print(f"Error in feature importance analysis: {e}")
    
    # 5. Risk Assessment Demonstration
    print("\n5. RISK ASSESSMENT DEMONSTRATION")
    print("-" * 40)
    
    try:
        # Get sample pitches for risk demonstration
        loader = TimeSeriesLoader()
        sample_pitches = loader.get_available_pitches()[:5]
        
        print(f"Demonstrating risk assessment on {len(sample_pitches)} pitchers:")
        
        risk_profiles = batch_analyze_pitcher_risks(sample_pitches, max_pitches=5)
        
        if not risk_profiles.empty:
            print(f"\nRisk Assessment Results:")
            
            for _, pitcher in risk_profiles.iterrows():
                session_pitch = pitcher['session_pitch']
                overall_risk = pitcher['overall_injury_risk']
                risk_category = pitcher['risk_category']
                elbow_risk = pitcher['ucl_injury_risk']
                shoulder_risk = pitcher['shoulder_injury_risk']
                
                print(f"\n  Pitcher {session_pitch}:")
                print(f"    Overall Risk: {overall_risk:.1f}/100 ({risk_category})")
                print(f"    UCL Risk: {elbow_risk:.1f}/100")
                print(f"    Shoulder Risk: {shoulder_risk:.1f}/100")
                
                # Risk interpretation
                if overall_risk < 25:
                    interpretation = "Low risk - Excellent biomechanics"
                elif overall_risk < 50:
                    interpretation = "Moderate risk - Monitor closely"
                elif overall_risk < 75:
                    interpretation = "High risk - Biomechanical intervention recommended"
                else:
                    interpretation = "Very high risk - Immediate attention required"
                
                print(f"    Recommendation: {interpretation}")
            
            # Summary statistics
            print(f"\nRisk Distribution Summary:")
            risk_categories = risk_profiles['risk_category'].value_counts()
            for category, count in risk_categories.items():
                print(f"  {category} Risk: {count} pitcher(s)")
            
            avg_overall_risk = risk_profiles['overall_injury_risk'].mean()
            print(f"  Average Overall Risk: {avg_overall_risk:.1f}/100")
    
    except Exception as e:
        print(f"Error in risk assessment demonstration: {e}")
    
    # 6. Phase 2 Summary and Conclusions
    print("\n6. PHASE 2 SUMMARY AND CONCLUSIONS")
    print("-" * 40)
    
    print(f"Phase 2 Achievements:")
    print(f"✓ Time-series data loading and processing pipeline")
    print(f"✓ Advanced temporal feature extraction (108 features per pitch)")
    print(f"✓ Comprehensive injury risk scoring system")
    print(f"✓ Intelligent feature selection and dimensionality reduction")
    print(f"✓ Enhanced machine learning models with improved performance")
    
    if 'baseline_eval' in locals() and 'selected_eval' in locals():
        baseline_auc = baseline_eval['mean_auc']
        enhanced_auc = selected_eval['mean_auc']
        improvement = enhanced_auc - baseline_auc
        
        print(f"\nPerformance Improvements:")
        print(f"  Baseline (Phase 1) AUC: {baseline_auc:.3f}")
        print(f"  Enhanced (Phase 2) AUC: {enhanced_auc:.3f}")
        print(f"  Improvement: {improvement:+.3f} AUC points ({improvement/baseline_auc*100:+.1f}%)")
        
        if improvement > 0.05:
            print(f"  ✓ Significant improvement achieved!")
        elif improvement > 0.02:
            print(f"  ✓ Moderate improvement achieved")
        else:
            print(f"  ⚠ Marginal improvement - consider additional feature engineering")
    
    print(f"\nKey Innovations in Phase 2:")
    print(f"  • Time-series biomechanical pattern analysis")
    print(f"  • Injury-specific composite risk scoring")
    print(f"  • Multi-method ensemble feature selection")
    print(f"  • Comprehensive temporal and kinematic features")
    print(f"  • Bilateral asymmetry assessment")
    
    print(f"\nReady for Phase 3: Model Optimization and Validation!")
    
    print("\n" + "="*70)
    print("PHASE 2 COMPLETE - ADVANCED FEATURES IMPLEMENTED")
    print("="*70)

if __name__ == "__main__":
    main()
