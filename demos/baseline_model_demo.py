#!/usr/bin/env python3
"""
Phase 1 Demonstration Script
OpenBiomechanics Baseball Injury Risk Assessment

This script demonstrates the complete Phase 1 pipeline:
1. Data loading and validation
2. Feature engineering
3. Baseline model training
4. Results export
"""

import sys
sys.path.append('src')

from src import (
    OpenBiomechanicsLoader, 
    validate_data_quality, 
    print_data_summary,
    FeatureEngineer,
    BaselineInjuryRiskModel
)

import pandas as pd
import numpy as np
import os

def main():
    print("="*60)
    print("PHASE 1: INJURY RISK ASSESSMENT BASELINE")
    print("OpenBiomechanics Baseball Pitching Data")
    print("="*60)
    
    # 1. Data Loading and Validation
    print("\n1. LOADING AND VALIDATING DATA")
    print("-" * 30)
    
    loader = OpenBiomechanicsLoader()
    data = loader.load_and_merge_data()
    validation_results = validate_data_quality(data)
    print_data_summary(data, validation_results)
    
    # 2. Feature Engineering
    print("\n2. FEATURE ENGINEERING")
    print("-" * 30)
    
    feature_engineer = FeatureEngineer()
    
    print("Creating injury risk features...")
    data = feature_engineer.create_injury_risk_features(data)
    
    print("Creating performance features...")
    data = feature_engineer.create_performance_features(data)
    
    print("Encoding categorical variables...")
    data = feature_engineer.encode_categorical_variables(data)
    
    print("Creating risk labels...")
    data = feature_engineer.create_risk_labels(data, method='composite_score')
    
    print(f"\nEnhanced dataset shape: {data.shape}")
    print(f"Risk label distribution:")
    print(data['injury_risk_label'].value_counts())
    
    # 3. Feature Selection for Modeling
    print("\n3. FEATURE SELECTION")
    print("-" * 30)
    
    feature_categories = feature_engineer.get_feature_importance_data(data)
    
    model_features = (
        feature_categories['primary_risk_factors'] + 
        feature_categories['secondary_risk_factors'] + 
        feature_categories['engineered_features'] +
        ['age_yrs', 'pitch_speed_mph']
    )
    
    # Filter to available features
    model_features = [f for f in model_features if f in data.columns]
    
    print(f"Selected {len(model_features)} features for modeling:")
    for i, feature in enumerate(model_features[:10]):  # Show first 10
        print(f"  {i+1:2d}. {feature}")
    if len(model_features) > 10:
        print(f"  ... and {len(model_features) - 10} more")
    
    # 4. Baseline Model Training
    print("\n4. BASELINE MODEL TRAINING")
    print("-" * 30)
    
    model = BaselineInjuryRiskModel()
    X, y = model.prepare_features_and_target(data, model_features, 'injury_risk_label')
    results = model.train_baseline_models(X, y)
    
    # 5. Results Summary
    print("\n5. RESULTS SUMMARY")
    print("-" * 30)
    
    print(model.generate_model_report())
    
    # 6. Export Results
    print("\n6. EXPORTING RESULTS")
    print("-" * 30)
    
    # Create output directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Save processed dataset
    output_path = 'data/processed_pitching_data.csv'
    data.to_csv(output_path, index=False)
    print(f"✓ Processed dataset saved to: {output_path}")
    
    # Save model results
    model.save_results('results')
    print(f"✓ Model results saved to: results/")
    
    # Phase 1 Completion Summary
    print("\n" + "="*60)
    print("PHASE 1 COMPLETION SUMMARY")
    print("="*60)
    print(f"✓ Data Loading: {len(data)} pitches from {data['user'].nunique()} pitchers")
    print(f"✓ Feature Engineering: {len(model_features)} features prepared")
    print(f"✓ Risk Label Creation: {data['injury_risk_label'].sum()} high-risk cases ({data['injury_risk_label'].mean()*100:.1f}%)")
    print(f"✓ Baseline Models: 3 models trained and evaluated")
    print(f"✓ Data Export: Processed data and results saved")
    
    # Best model performance
    best_model_name = max(model.performance_metrics.keys(), 
                         key=lambda k: model.performance_metrics[k].get('cv_auc_mean', 0))
    best_auc = model.performance_metrics[best_model_name]['cv_auc_mean']
    
    print(f"\nBest Model: {best_model_name.replace('_', ' ').title()}")
    print(f"Cross-Validation AUC: {best_auc:.3f}")
    
    if best_auc > 0.70:
        print("✓ Baseline performance target achieved!")
    else:
        print("⚠ Consider additional feature engineering in Phase 2")
    
    print(f"\nReady to proceed to Phase 2: Advanced Features!")
    print("="*60)

if __name__ == "__main__":
    main()
