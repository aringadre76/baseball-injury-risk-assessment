#!/usr/bin/env python3
"""
COMPREHENSIVE SYSTEM VERIFICATION
Tests EVERY component to ensure the system actually works
"""

import sys
from pathlib import Path
import traceback

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.append(str(src_path))

import pandas as pd
import numpy as np
import pickle
import os

def test_1_data_loading():
    """Test 1: Data Loading - Can we actually load the real data?"""
    print("="*80)
    print("TEST 1: DATA LOADING VERIFICATION")
    print("="*80)
    
    try:
        from openbiomechanics_loader import OpenBiomechanicsLoader
        
        loader = OpenBiomechanicsLoader()
        
        # Test metadata loading
        metadata = loader.load_metadata()
        print(f"✓ Metadata loaded: {metadata.shape}")
        
        # Test POI loading
        poi_data = loader.load_poi_metrics()
        print(f"✓ POI data loaded: {poi_data.shape}")
        
        # Test merge
        merged_data = loader.load_and_merge_data()
        print(f"✓ Data merged: {merged_data.shape}")
        
        # Verify we have the key injury risk variables
        key_vars = ['elbow_varus_moment', 'shoulder_internal_rotation_moment']
        missing_vars = [var for var in key_vars if var not in merged_data.columns]
        
        if missing_vars:
            print(f"❌ MISSING KEY VARIABLES: {missing_vars}")
            return False, merged_data
        else:
            print(f"✓ All key injury risk variables present")
            return True, merged_data
            
    except Exception as e:
        print(f"❌ DATA LOADING FAILED: {e}")
        traceback.print_exc()
        return False, None

def test_2_feature_engineering(data):
    """Test 2: Feature Engineering - Can we create features from real data?"""
    print("\n" + "="*80)
    print("TEST 2: FEATURE ENGINEERING VERIFICATION")
    print("="*80)
    
    try:
        from biomechanical_feature_engineer import FeatureEngineer
        
        feature_engineer = FeatureEngineer()
        
        # Test feature creation
        engineered_data = feature_engineer.create_injury_risk_features(data)
        print(f"✓ Features created: {engineered_data.shape}")
        
        # Check if we have the expected features
        expected_features = ['elbow_stress_composite', 'kinetic_chain_ratio']
        missing_features = [f for f in expected_features if f not in engineered_data.columns]
        
        if missing_features:
            print(f"❌ MISSING ENGINEERED FEATURES: {missing_features}")
            return False, engineered_data
        else:
            print(f"✓ All expected engineered features present")
            return True, engineered_data
            
    except Exception as e:
        print(f"❌ FEATURE ENGINEERING FAILED: {e}")
        traceback.print_exc()
        return False, None

def test_3_model_training(data):
    """Test 3: Model Training - Can we actually train models on real data?"""
    print("\n" + "="*80)
    print("TEST 3: MODEL TRAINING VERIFICATION")
    print("="*80)
    
    try:
        from advanced_models import AdvancedInjuryRiskModel
        from sklearn.model_selection import train_test_split
        
        # Create injury risk labels
        key_vars = ['elbow_varus_moment', 'shoulder_internal_rotation_moment']
        available_vars = [var for var in key_vars if var in data.columns]
        
        if not available_vars:
            print("❌ No injury risk variables available for labeling")
            return False, None
        
        # Create risk score
        risk_score = np.zeros(len(data))
        for var in available_vars:
            values = data[var].fillna(0)
            if values.max() > values.min():
                normalized = (values - values.min()) / (values.max() - values.min())
                risk_score += normalized
        
        threshold = np.percentile(risk_score, 70)
        injury_risk_label = (risk_score > threshold).astype(int)
        
        print(f"✓ Risk labels created: {np.bincount(injury_risk_label)}")
        
        # Select features
        biomech_features = [col for col in data.columns if any(keyword in col.lower() 
                           for keyword in ['elbow', 'shoulder', 'torso', 'hip', 'force', 'moment', 'velo', 'pitch_speed', 'age', 'height', 'weight'])]
        
        biomech_features = [col for col in biomech_features if col not in ['injury_risk_label', 'injury_risk_score']]
        
        print(f"✓ Features selected: {len(biomech_features)}")
        
        # Prepare data
        X = data[biomech_features].copy()
        y = injury_risk_label.copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"✓ Data split: Train {X_train.shape[0]}, Test {X_test.shape[0]}")
        
        # Train model
        model = AdvancedInjuryRiskModel(random_state=42)
        
        # Train just one model for verification
        print("Training Random Forest for verification...")
        results = model.train_ensemble_models(
            X_train, y_train, optimize_hyperparameters=False
        )
        
        print(f"✓ Model training completed")
        print(f"✓ Models trained: {list(model.models.keys())}")
        
        return True, (model, X_test, y_test)
        
    except Exception as e:
        print(f"❌ MODEL TRAINING FAILED: {e}")
        traceback.print_exc()
        return False, None

def test_4_model_prediction(model_data):
    """Test 4: Model Prediction - Can the trained model actually make predictions?"""
    print("\n" + "="*80)
    print("TEST 4: MODEL PREDICTION VERIFICATION")
    print("="*80)
    
    try:
        model, X_test, y_test = model_data
        
        # Get best model
        best_model_name = max(model.performance_metrics.keys(), 
                             key=lambda x: model.performance_metrics[x]['cv_auc_mean'])
        best_model = model.models[best_model_name]
        
        print(f"✓ Best model: {best_model_name}")
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        print(f"✓ Predictions made: {len(y_pred)}")
        print(f"✓ Probabilities generated: {len(y_pred_proba)}")
        
        # Calculate basic metrics
        accuracy = (y_pred == y_test).mean()
        print(f"✓ Test accuracy: {accuracy:.3f}")
        
        # Check if predictions make sense
        if accuracy < 0.5:
            print(f"❌ SUSPICIOUSLY LOW ACCURACY: {accuracy:.3f}")
            return False
        else:
            print(f"✓ Accuracy is reasonable: {accuracy:.3f}")
            return True
            
    except Exception as e:
        print(f"❌ MODEL PREDICTION FAILED: {e}")
        traceback.print_exc()
        return False

def test_5_model_loading():
    """Test 5: Model Loading - Can we load the saved models?"""
    print("\n" + "="*80)
    print("TEST 5: MODEL LOADING VERIFICATION")
    print("="*80)
    
    try:
        # Check if saved models exist
        model_files = [
            'results/real_data_validation/validated_model.pkl',
            'results/phase3/voting_ensemble_advanced.pkl',
            'results/phase3/gradient_boosting_optimized_advanced.pkl'
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                print(f"✓ Model file exists: {model_file}")
                
                # Try to load it
                with open(model_file, 'rb') as f:
                    loaded_model = pickle.load(f)
                print(f"✓ Model loaded successfully: {type(loaded_model)}")
                
                # Check if it has predict method
                if hasattr(loaded_model, 'predict'):
                    print(f"✓ Model has predict method")
                else:
                    print(f"❌ Model missing predict method")
                    return False
            else:
                print(f"❌ Model file missing: {model_file}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ MODEL LOADING FAILED: {e}")
        traceback.print_exc()
        return False

def test_6_feature_importance():
    """Test 6: Feature Importance - Can we analyze feature importance?"""
    print("\n" + "="*80)
    print("TEST 6: FEATURE IMPORTANCE VERIFICATION")
    print("="*80)
    
    try:
        # Load the validated model
        with open('results/real_data_validation/validated_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Check if we can access feature importance
        if hasattr(model, 'feature_importance'):
            print(f"✓ Feature importance data available")
            
            # Check if we have feature importance for any model
            if model.feature_importance:
                print(f"✓ Feature importance for models: {list(model.feature_importance.keys())}")
                return True
            else:
                print(f"❌ No feature importance data found")
                return False
        else:
            print(f"❌ Model missing feature_importance attribute")
            return False
            
    except Exception as e:
        print(f"❌ FEATURE IMPORTANCE ANALYSIS FAILED: {e}")
        traceback.print_exc()
        return False

def test_7_end_to_end_pipeline():
    """Test 7: End-to-End Pipeline - Does the whole system work together?"""
    print("\n" + "="*80)
    print("TEST 7: END-TO-END PIPELINE VERIFICATION")
    print("="*80)
    
    try:
        # Load data
        from openbiomechanics_loader import OpenBiomechanicsLoader
        loader = OpenBiomechanicsLoader()
        data = loader.load_and_merge_data()
        print("✓ Step 1: Data loaded")
        
        # Engineer features
        from biomechanical_feature_engineer import FeatureEngineer
        feature_engineer = FeatureEngineer()
        engineered_data = feature_engineer.create_injury_risk_features(data)
        print("✓ Step 2: Features engineered")
        
        # Create labels
        key_vars = ['elbow_varus_moment', 'shoulder_internal_rotation_moment']
        risk_score = np.zeros(len(engineered_data))
        for var in key_vars:
            values = engineered_data[var].fillna(0)
            if values.max() > values.min():
                normalized = (values - values.min()) / (values.max() - values.min())
                risk_score += normalized
        
        threshold = np.percentile(risk_score, 70)
        injury_risk_label = (risk_score > threshold).astype(int)
        print("✓ Step 3: Labels created")
        
        # Select features
        biomech_features = [col for col in engineered_data.columns if any(keyword in col.lower() 
                           for keyword in ['elbow', 'shoulder', 'torso', 'hip', 'force', 'moment', 'velo'])]
        biomech_features = [col for col in biomech_features if col not in ['injury_risk_label', 'injury_risk_score']]
        
        X = engineered_data[biomech_features].fillna(engineered_data[biomech_features].median())
        y = injury_risk_label
        print("✓ Step 4: Features prepared")
        
        # Load trained model
        with open('results/real_data_validation/validated_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✓ Step 5: Model loaded")
        
        # Make prediction on new data
        sample_data = X.iloc[:5]  # First 5 samples
        best_model_name = max(model.performance_metrics.keys(), 
                             key=lambda x: model.performance_metrics[x]['cv_auc_mean'])
        best_model = model.models[best_model_name]
        
        predictions = best_model.predict(sample_data)
        probabilities = best_model.predict_proba(sample_data)[:, 1]
        
        print(f"✓ Step 6: Predictions made on sample data")
        print(f"  Sample predictions: {predictions}")
        print(f"  Sample probabilities: {probabilities}")
        
        return True
        
    except Exception as e:
        print(f"❌ END-TO-END PIPELINE FAILED: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive verification."""
    print("🚀 COMPREHENSIVE SYSTEM VERIFICATION STARTING")
    print("="*80)
    
    test_results = {}
    
    # Test 1: Data Loading
    success, data = test_1_data_loading()
    test_results['data_loading'] = success
    if not success:
        print("\n❌ CRITICAL FAILURE: Cannot load data. System is broken.")
        return
    
    # Test 2: Feature Engineering
    success, engineered_data = test_2_feature_engineering(data)
    test_results['feature_engineering'] = success
    if not success:
        print("\n❌ CRITICAL FAILURE: Cannot engineer features. System is broken.")
        return
    
    # Test 3: Model Training
    success, model_data = test_3_model_training(engineered_data)
    test_results['model_training'] = success
    if not success:
        print("\n❌ CRITICAL FAILURE: Cannot train models. System is broken.")
        return
    
    # Test 4: Model Prediction
    success = test_4_model_prediction(model_data)
    test_results['model_prediction'] = success
    
    # Test 5: Model Loading
    success = test_5_model_loading()
    test_results['model_loading'] = success
    
    # Test 6: Feature Importance
    success = test_6_feature_importance()
    test_results['feature_importance'] = success
    
    # Test 7: End-to-End Pipeline
    success = test_7_end_to_end_pipeline()
    test_results['end_to_end'] = success
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20}: {status}")
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! SYSTEM IS FULLY OPERATIONAL!")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} TESTS FAILED! SYSTEM HAS ISSUES!")
    
    print("="*80)

if __name__ == "__main__":
    main()
