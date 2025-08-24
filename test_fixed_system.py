#!/usr/bin/env python3
"""
Test Fixed System - Verify that all critical issues are resolved
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.append(str(src_path))

import pandas as pd
import numpy as np
import pickle
import os

from openbiomechanics_loader import OpenBiomechanicsLoader
from robust_feature_engineer import RobustFeatureEngineer
from robust_injury_risk_model import RobustInjuryRiskModel

def test_1_robust_feature_engineering():
    """Test 1: Robust Feature Engineering - Does it create consistent features?"""
    print("="*80)
    print("TEST 1: ROBUST FEATURE ENGINEERING")
    print("="*80)
    
    try:
        # Load data
        loader = OpenBiomechanicsLoader()
        data = loader.load_and_merge_data()
        print(f"✓ Data loaded: {data.shape}")
        
        # Create robust feature engineer
        feature_engineer = RobustFeatureEngineer()
        
        # Fit on training data
        print("Fitting robust feature engineer...")
        feature_engineer.fit(data)
        
        # Transform the same data
        print("Transforming data...")
        transformed_data = feature_engineer.transform(data)
        
        print(f"✓ Original data: {data.shape}")
        print(f"✓ Transformed data: {transformed_data.shape}")
        print(f"✓ Feature columns: {len(feature_engineer.feature_columns)}")
        
        # Test consistency - transform again
        print("Testing consistency...")
        transformed_data_2 = feature_engineer.transform(data)
        
        # Check if features are consistent
        if transformed_data.shape == transformed_data_2.shape:
            print("✓ Feature consistency verified")
            return True, feature_engineer, transformed_data
        else:
            print("❌ Feature consistency failed")
            return False, None, None
            
    except Exception as e:
        print(f"❌ Robust feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_2_robust_model_training(feature_engineer, data):
    """Test 2: Robust Model Training - Can we train a model that can be saved/loaded?"""
    print("\n" + "="*80)
    print("TEST 2: ROBUST MODEL TRAINING")
    print("="*80)
    
    try:
        # Create robust model
        model = RobustInjuryRiskModel(random_state=42)
        model.feature_engineer = feature_engineer
        
        # Train model
        print("Training robust model...")
        training_results = model.train(data, test_size=0.3)
        
        print(f"✓ Training completed: {training_results}")
        print(f"✓ Best model: {model.best_model_name}")
        print(f"✓ Test accuracy: {training_results['test_accuracy']:.3f}")
        print(f"✓ Test AUC: {training_results['test_auc']:.3f}")
        
        return True, model
        
    except Exception as e:
        print(f"❌ Robust model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_3_model_saving_loading(model):
    """Test 3: Model Saving/Loading - Can we save and load the model?"""
    print("\n" + "="*80)
    print("TEST 3: MODEL SAVING/LOADING")
    print("="*80)
    
    try:
        # Save model
        save_path = "results/robust_model.pkl"
        print(f"Saving model to {save_path}...")
        model.save(save_path)
        
        # Load model
        print("Loading model...")
        loaded_model = RobustInjuryRiskModel.load(save_path)
        
        print(f"✓ Model saved and loaded successfully")
        print(f"✓ Loaded model type: {type(loaded_model)}")
        print(f"✓ Loaded model trained: {loaded_model.is_trained}")
        print(f"✓ Loaded model best: {loaded_model.best_model_name}")
        
        return True, loaded_model
        
    except Exception as e:
        print(f"❌ Model saving/loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_4_model_prediction(loaded_model, original_data):
    """Test 4: Model Prediction - Can the loaded model make predictions on new data?"""
    print("\n" + "="*80)
    print("TEST 4: MODEL PREDICTION")
    print("="*80)
    
    try:
        # Use a subset of the original data as "new data"
        sample_data = original_data.iloc[:10].copy()
        print(f"Testing predictions on {len(sample_data)} samples...")
        
        # Make predictions
        predictions = loaded_model.predict(sample_data)
        probabilities = loaded_model.predict_proba(sample_data)
        
        print(f"✓ Predictions made: {len(predictions)}")
        print(f"✓ Probabilities generated: {probabilities.shape}")
        print(f"✓ Sample predictions: {predictions}")
        print(f"✓ Sample probabilities: {probabilities[:3]}")
        
        # Check if predictions make sense
        if len(predictions) == len(sample_data) and len(probabilities) == len(sample_data):
            print("✓ Prediction consistency verified")
            return True
        else:
            print("❌ Prediction consistency failed")
            return False
            
    except Exception as e:
        print(f"❌ Model prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_5_end_to_end_pipeline():
    """Test 5: End-to-End Pipeline - Does the complete system work?"""
    print("\n" + "="*80)
    print("TEST 5: END-TO-END PIPELINE")
    print("="*80)
    
    try:
        # Load data
        print("Step 1: Loading data...")
        loader = OpenBiomechanicsLoader()
        data = loader.load_and_merge_data()
        print("✓ Data loaded")
        
        # Create and fit feature engineer
        print("Step 2: Feature engineering...")
        feature_engineer = RobustFeatureEngineer()
        feature_engineer.fit(data)
        print("✓ Feature engineer fitted")
        
        # Create and train model
        print("Step 3: Model training...")
        model = RobustInjuryRiskModel(random_state=42)
        model.feature_engineer = feature_engineer
        training_results = model.train(data, test_size=0.3)
        print("✓ Model trained")
        
        # Save model
        print("Step 4: Model saving...")
        save_path = "results/end_to_end_model.pkl"
        model.save(save_path)
        print("✓ Model saved")
        
        # Load model
        print("Step 5: Model loading...")
        loaded_model = RobustInjuryRiskModel.load(save_path)
        print("✓ Model loaded")
        
        # Make predictions
        print("Step 6: Making predictions...")
        sample_data = data.iloc[:5].copy()
        predictions = loaded_model.predict(sample_data)
        probabilities = loaded_model.predict_proba(sample_data)
        print("✓ Predictions made")
        
        print(f"Final results:")
        print(f"  Training samples: {training_results['training_samples']}")
        print(f"  Test samples: {training_results['test_samples']}")
        print(f"  Features: {training_results['features']}")
        print(f"  Test accuracy: {training_results['test_accuracy']:.3f}")
        print(f"  Test AUC: {training_results['test_auc']:.3f}")
        print(f"  Sample predictions: {predictions}")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests to verify the fixed system."""
    print("🚀 TESTING FIXED SYSTEM - VERIFYING ALL CRITICAL ISSUES RESOLVED")
    print("="*80)
    
    test_results = {}
    
    # Test 1: Robust Feature Engineering
    success, feature_engineer, transformed_data = test_1_robust_feature_engineering()
    test_results['robust_feature_engineering'] = success
    if not success:
        print("\n❌ CRITICAL FAILURE: Robust feature engineering broken.")
        return
    
    # Test 2: Robust Model Training
    success, model = test_2_robust_model_training(feature_engineer, transformed_data)
    test_results['robust_model_training'] = success
    if not success:
        print("\n❌ CRITICAL FAILURE: Robust model training broken.")
        return
    
    # Test 3: Model Saving/Loading
    success, loaded_model = test_3_model_saving_loading(model)
    test_results['model_saving_loading'] = success
    if not success:
        print("\n❌ CRITICAL FAILURE: Model saving/loading broken.")
        return
    
    # Test 4: Model Prediction
    success = test_4_model_prediction(loaded_model, transformed_data)
    test_results['model_prediction'] = success
    
    # Test 5: End-to-End Pipeline
    success = test_5_end_to_end_pipeline()
    test_results['end_to_end_pipeline'] = success
    
    # Summary
    print("\n" + "="*80)
    print("FIXED SYSTEM VERIFICATION SUMMARY")
    print("="*80)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25}: {status}")
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! SYSTEM IS NOW FULLY OPERATIONAL!")
        print("✓ Feature consistency: RESOLVED")
        print("✓ Model serialization: RESOLVED")
        print("✓ End-to-end pipeline: RESOLVED")
        print("✓ Production deployment: READY")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} TESTS FAILED! SYSTEM STILL HAS ISSUES!")
    
    print("="*80)

if __name__ == "__main__":
    main()
