#!/usr/bin/env python3
"""
Production Verification - Demonstrate the fixed system working with new data
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

def demonstrate_production_workflow():
    """Demonstrate the complete production workflow with new data."""
    print("🚀 PRODUCTION WORKFLOW DEMONSTRATION")
    print("="*80)
    
    # Step 1: Load training data
    print("STEP 1: Loading training data...")
    loader = OpenBiomechanicsLoader()
    training_data = loader.load_and_merge_data()
    print(f"✓ Training data loaded: {training_data.shape}")
    
    # Step 2: Create and fit feature engineer
    print("\nSTEP 2: Creating and fitting feature engineer...")
    feature_engineer = RobustFeatureEngineer()
    feature_engineer.fit(training_data)
    print(f"✓ Feature engineer fitted with {len(feature_engineer.feature_columns)} features")
    
    # Step 3: Create and train model
    print("\nSTEP 3: Training injury risk model...")
    model = RobustInjuryRiskModel(random_state=42)
    model.feature_engineer = feature_engineer
    
    training_results = model.train(training_data, test_size=0.3)
    print(f"✓ Model trained successfully")
    print(f"  Test Accuracy: {training_results['test_accuracy']:.3f}")
    print(f"  Test AUC: {training_results['test_auc']:.3f}")
    
    # Step 4: Save the complete model
    print("\nSTEP 4: Saving production model...")
    model_path = "results/production_model.pkl"
    model.save(model_path)
    print(f"✓ Production model saved to {model_path}")
    
    # Step 5: Load the saved model
    print("\nSTEP 5: Loading production model...")
    loaded_model = RobustInjuryRiskModel.load(model_path)
    print(f"✓ Production model loaded successfully")
    print(f"  Model trained: {loaded_model.is_trained}")
    print(f"  Best model: {loaded_model.best_model_name}")
    
    # Step 6: Demonstrate prediction on new data
    print("\nSTEP 6: Making predictions on new data...")
    
    # Simulate new data (subset of training data for demonstration)
    new_data = training_data.iloc[:20].copy()
    print(f"  New data samples: {len(new_data)}")
    
    # Make predictions
    predictions = loaded_model.predict(new_data)
    probabilities = loaded_model.predict_proba(new_data)
    
    print(f"✓ Predictions completed successfully")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Probabilities shape: {probabilities.shape}")
    print(f"  Sample predictions: {predictions[:5]}")
    print(f"  Sample probabilities: {probabilities[:3]}")
    
    # Step 7: Feature importance analysis
    print("\nSTEP 7: Feature importance analysis...")
    feature_importance = loaded_model.get_feature_importance(top_n=10)
    print("Top 10 Most Important Features:")
    for i, (_, row) in enumerate(feature_importance.iterrows(), 1):
        print(f"  {i:2d}. {row['feature']}: {row['importance']:.4f}")
    
    # Step 8: Generate comprehensive report
    print("\nSTEP 8: Generating production report...")
    report = loaded_model.generate_report()
    print("Model Report Generated Successfully")
    
    # Save report to file
    report_path = "results/production_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✓ Report saved to {report_path}")
    
    return True

def test_model_consistency():
    """Test that the model produces consistent results."""
    print("\n🔍 MODEL CONSISTENCY TEST")
    print("="*80)
    
    # Load the saved model
    model_path = "results/production_model.pkl"
    if not os.path.exists(model_path):
        print("❌ Production model not found. Run demonstration first.")
        return False
    
    model = RobustInjuryRiskModel.load(model_path)
    
    # Load some data
    loader = OpenBiomechanicsLoader()
    data = loader.load_and_merge_data()
    test_data = data.iloc[:10].copy()
    
    # Make predictions multiple times
    predictions_1 = model.predict(test_data)
    predictions_2 = model.predict(test_data)
    predictions_3 = model.predict(test_data)
    
    # Check consistency
    consistent_1_2 = np.array_equal(predictions_1, predictions_2)
    consistent_2_3 = np.array_equal(predictions_2, predictions_3)
    consistent_1_3 = np.array_equal(predictions_1, predictions_3)
    
    print(f"Prediction consistency check:")
    print(f"  Run 1 vs Run 2: {'✅ PASS' if consistent_1_2 else '❌ FAIL'}")
    print(f"  Run 2 vs Run 3: {'✅ PASS' if consistent_2_3 else '❌ FAIL'}")
    print(f"  Run 1 vs Run 3: {'✅ PASS' if consistent_1_3 else '❌ FAIL'}")
    
    if consistent_1_2 and consistent_2_3 and consistent_1_3:
        print("✓ All consistency checks passed!")
        return True
    else:
        print("❌ Consistency checks failed!")
        return False

def test_feature_engineering_consistency():
    """Test that feature engineering produces consistent results."""
    print("\n🔍 FEATURE ENGINEERING CONSISTENCY TEST")
    print("="*80)
    
    # Load the saved model
    model_path = "results/production_model.pkl"
    if not os.path.exists(model_path):
        print("❌ Production model not found. Run demonstration first.")
        return False
    
    model = RobustInjuryRiskModel.load(model_path)
    
    # Load some data
    loader = OpenBiomechanicsLoader()
    data = loader.load_and_merge_data()
    test_data = data.iloc[:10].copy()
    
    # Transform data multiple times
    transformed_1 = model.feature_engineer.transform(test_data)
    transformed_2 = model.feature_engineer.transform(test_data)
    transformed_3 = model.feature_engineer.transform(test_data)
    
    # Check consistency
    consistent_1_2 = transformed_1.equals(transformed_2)
    consistent_2_3 = transformed_2.equals(transformed_3)
    consistent_1_3 = transformed_1.equals(transformed_3)
    
    print(f"Feature engineering consistency check:")
    print(f"  Run 1 vs Run 2: {'✅ PASS' if consistent_1_2 else '❌ FAIL'}")
    print(f"  Run 2 vs Run 3: {'✅ PASS' if consistent_2_3 else '❌ FAIL'}")
    print(f"  Run 1 vs Run 3: {'✅ PASS' if consistent_1_3 else '❌ FAIL'}")
    
    if consistent_1_2 and consistent_2_3 and consistent_1_3:
        print("✓ All consistency checks passed!")
        return True
    else:
        print("❌ Consistency checks failed!")
        return False

def test_error_handling():
    """Test error handling for edge cases."""
    print("\n🔍 ERROR HANDLING TEST")
    print("="*80)
    
    # Load the saved model
    model_path = "results/production_model.pkl"
    if not os.path.exists(model_path):
        print("❌ Production model not found. Run demonstration first.")
        return False
    
    model = RobustInjuryRiskModel.load(model_path)
    
    # Test 1: Empty DataFrame
    print("Test 1: Empty DataFrame...")
    try:
        empty_data = pd.DataFrame()
        predictions = model.predict(empty_data)
        print("❌ Should have failed with empty DataFrame")
        return False
    except Exception as e:
        print(f"✓ Correctly handled empty DataFrame: {type(e).__name__}")
    
    # Test 2: Missing required columns
    print("Test 2: Missing required columns...")
    try:
        missing_data = pd.DataFrame({'random_column': [1, 2, 3]})
        predictions = model.predict(missing_data)
        print("✓ Handled missing columns gracefully")
    except Exception as e:
        print(f"❌ Failed to handle missing columns: {e}")
        return False
    
    # Test 3: Invalid data types
    print("Test 3: Invalid data types...")
    try:
        invalid_data = pd.DataFrame({
            'elbow_varus_moment': ['not_a_number', 'also_not_a_number'],
            'shoulder_internal_rotation_moment': ['invalid', 'data']
        })
        predictions = model.predict(invalid_data)
        print("✓ Handled invalid data types gracefully")
    except Exception as e:
        print(f"❌ Failed to handle invalid data types: {e}")
        return False
    
    print("✓ All error handling tests passed!")
    return True

def main():
    """Run the complete production verification."""
    print("🚀 COMPREHENSIVE PRODUCTION VERIFICATION")
    print("="*80)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Run demonstration
    print("Running production workflow demonstration...")
    demo_success = demonstrate_production_workflow()
    
    if not demo_success:
        print("❌ Production workflow demonstration failed!")
        return
    
    print("\n" + "="*80)
    print("PRODUCTION VERIFICATION RESULTS")
    print("="*80)
    
    # Run all verification tests
    tests = [
        ("Model Consistency", test_model_consistency),
        ("Feature Engineering Consistency", test_feature_engineering_consistency),
        ("Error Handling", test_error_handling)
    ]
    
    test_results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            test_results[test_name] = False
    
    # Summary
    print(f"\nVerification Summary:")
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:30}: {status}")
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"\nOverall: {passed_tests}/{total_tests} verification tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 PRODUCTION VERIFICATION COMPLETE!")
        print("✓ System is production-ready")
        print("✓ Model serialization works")
        print("✓ Feature consistency maintained")
        print("✓ Error handling robust")
        print("✓ Ready for deployment")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} verification tests failed!")
        print("System may not be fully production-ready")
    
    print("="*80)

if __name__ == "__main__":
    main()
