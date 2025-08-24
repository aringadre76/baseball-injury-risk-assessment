"""
Phase 3 Demo: Advanced Model Optimization and Interpretability
Comprehensive demonstration of ensemble methods, neural networks, and SHAP analysis
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

# Import our modules
try:
    from openbiomechanics_loader import OpenBiomechanicsLoader
    from biomechanical_feature_engineer import FeatureEngineer
    REAL_DATA_AVAILABLE = True
except ImportError as e:
    print(f"Could not import data modules: {e}")
    REAL_DATA_AVAILABLE = False

from advanced_models import AdvancedInjuryRiskModel, run_advanced_model_analysis
from injury_risk_explainer import comprehensive_interpretability_analysis

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")


def load_and_prepare_data() -> tuple:
    """Load and prepare data for Phase 3 analysis."""
    print("Loading and preparing data for Phase 3...")
    
    if REAL_DATA_AVAILABLE:
        try:
            # Try to load real data first
            loader = OpenBiomechanicsLoader()
            poi_data = loader.load_poi_metrics()
            metadata = loader.load_metadata()
            
            # Merge data
            merged_data = pd.merge(poi_data, metadata, on='pitch_id', how='inner')
            print(f"Merged dataset shape: {merged_data.shape}")
            
            # Engineer features
            feature_engineer = FeatureEngineer()
            engineered_data = feature_engineer.create_injury_risk_features(merged_data)
            
            # Select relevant features for modeling
            biomech_features = [col for col in engineered_data.columns if any(keyword in col.lower() 
                               for keyword in ['elbow', 'shoulder', 'torso', 'hip', 'force', 'moment', 'velo'])]
            
            # Create injury risk labels (simplified)
            engineered_data['injury_risk_label'] = (
                (engineered_data.get('elbow_varus_moment', 0) > engineered_data.get('elbow_varus_moment', 0).quantile(0.7)) |
                (engineered_data.get('shoulder_internal_rotation_moment', 0) > engineered_data.get('shoulder_internal_rotation_moment', 0).quantile(0.7))
            ).astype(int)
            
            print(f"Final dataset shape: {engineered_data.shape}")
            print(f"Number of features: {len(biomech_features)}")
            
            return engineered_data, biomech_features
            
        except Exception as e:
            print(f"Could not load real data: {e}")
    
    print("Using synthetic data for Phase 3 demonstration...")
    return create_synthetic_demo_data()


def create_synthetic_demo_data() -> tuple:
    """Create synthetic data for demonstration purposes."""
    print("Creating synthetic biomechanical data for Phase 3 demo...")
    
    np.random.seed(42)
    n_samples = 300
    
    # Create synthetic biomechanical features
    feature_data = {
        # Elbow stress indicators
        'elbow_varus_moment': np.random.normal(50, 15, n_samples),
        'max_elbow_extension_velo': np.random.normal(2000, 300, n_samples),
        'elbow_flexion_fp': np.random.normal(25, 8, n_samples),
        
        # Shoulder stress indicators
        'shoulder_internal_rotation_moment': np.random.normal(80, 20, n_samples),
        'max_shoulder_internal_rotational_velo': np.random.normal(7000, 1000, n_samples),
        'shoulder_abduction_fp': np.random.normal(85, 15, n_samples),
        
        # Torso mechanics
        'max_torso_rotational_velo': np.random.normal(800, 150, n_samples),
        'torso_anterior_tilt_fp': np.random.normal(15, 5, n_samples),
        'torso_lateral_tilt_fp': np.random.normal(5, 3, n_samples),
        
        # Hip and pelvis
        'max_rotation_hip_shoulder_separation': np.random.normal(45, 10, n_samples),
        'pelvis_anterior_tilt_fp': np.random.normal(12, 4, n_samples),
        'pelvis_rotation_fp': np.random.normal(8, 3, n_samples),
        
        # Force plate data
        'max_force_z': np.random.normal(1200, 200, n_samples),
        'force_z_at_fp': np.random.normal(800, 150, n_samples),
        
        # Kinetic chain
        'lead_knee_extension_angular_velo_fp': np.random.normal(600, 100, n_samples),
        
        # Pitch characteristics
        'pitch_speed': np.random.normal(85, 8, n_samples),
        'age': np.random.randint(16, 25, n_samples),
        'height': np.random.normal(72, 3, n_samples),
        'weight': np.random.normal(180, 20, n_samples),
    }
    
    # Create DataFrame
    data = pd.DataFrame(feature_data)
    
    # Add some correlated features for realism
    data['elbow_stress_composite'] = (
        0.4 * (data['elbow_varus_moment'] - data['elbow_varus_moment'].mean()) / data['elbow_varus_moment'].std() +
        0.3 * (data['max_elbow_extension_velo'] - data['max_elbow_extension_velo'].mean()) / data['max_elbow_extension_velo'].std() +
        0.3 * (data['shoulder_internal_rotation_moment'] - data['shoulder_internal_rotation_moment'].mean()) / data['shoulder_internal_rotation_moment'].std()
    )
    
    data['kinetic_chain_efficiency'] = (
        0.5 * (data['max_rotation_hip_shoulder_separation'] - data['max_rotation_hip_shoulder_separation'].mean()) / data['max_rotation_hip_shoulder_separation'].std() +
        0.3 * (data['max_torso_rotational_velo'] - data['max_torso_rotational_velo'].mean()) / data['max_torso_rotational_velo'].std() +
        0.2 * (data['max_force_z'] - data['max_force_z'].mean()) / data['max_force_z'].std()
    )
    
    # Create injury risk labels based on biomechanical risk factors
    risk_score = (
        0.3 * (data['elbow_varus_moment'] - data['elbow_varus_moment'].min()) / (data['elbow_varus_moment'].max() - data['elbow_varus_moment'].min()) +
        0.25 * (data['shoulder_internal_rotation_moment'] - data['shoulder_internal_rotation_moment'].min()) / (data['shoulder_internal_rotation_moment'].max() - data['shoulder_internal_rotation_moment'].min()) +
        0.2 * (data['max_shoulder_internal_rotational_velo'] - data['max_shoulder_internal_rotational_velo'].min()) / (data['max_shoulder_internal_rotational_velo'].max() - data['max_shoulder_internal_rotational_velo'].min()) +
        0.15 * (data['max_elbow_extension_velo'] - data['max_elbow_extension_velo'].min()) / (data['max_elbow_extension_velo'].max() - data['max_elbow_extension_velo'].min()) +
        0.1 * np.random.random(n_samples)
    )
    
    # Create binary injury risk labels (top 30% as high risk)
    threshold = np.percentile(risk_score, 70)
    data['injury_risk_label'] = (risk_score > threshold).astype(int)
    
    # Add continuous risk score
    data['injury_risk_score'] = risk_score
    
    # Feature columns (exclude target variables)
    feature_columns = [col for col in data.columns if col not in ['injury_risk_label', 'injury_risk_score']]
    
    print(f"Synthetic dataset created: {data.shape}")
    print(f"High risk samples: {data['injury_risk_label'].sum()} ({data['injury_risk_label'].mean():.1%})")
    
    return data, feature_columns


def run_phase3_advanced_models(data: pd.DataFrame, feature_columns: List[str]) -> AdvancedInjuryRiskModel:
    """Run advanced ensemble models with hyperparameter optimization."""
    print("\n" + "="*60)
    print("PHASE 3: ADVANCED MODEL OPTIMIZATION")
    print("="*60)
    
    # Initialize advanced model
    advanced_model = AdvancedInjuryRiskModel(random_state=42)
    
    # Prepare features and target
    X = data[feature_columns].copy()
    y = data['injury_risk_label'].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    print(f"Training on {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Train ensemble models with hyperparameter optimization
    print("\nTraining advanced ensemble models...")
    results = advanced_model.train_ensemble_models(X, y, optimize_hyperparameters=True)
    
    # Generate comprehensive visualizations
    print("\nGenerating model comparison visualizations...")
    advanced_model.plot_model_comparison()
    
    print("\nGenerating feature importance comparison...")
    advanced_model.plot_feature_importance_comparison()
    
    # Print comprehensive report
    print("\n" + advanced_model.generate_comprehensive_report())
    
    # Save results
    advanced_model.save_advanced_results()
    
    return advanced_model


def run_phase3_interpretability(advanced_model: AdvancedInjuryRiskModel, 
                               data: pd.DataFrame, 
                               feature_columns: List[str]) -> Dict[str, Any]:
    """Run comprehensive model interpretability analysis."""
    print("\n" + "="*60)
    print("PHASE 3: MODEL INTERPRETABILITY & RISK FACTOR ANALYSIS")
    print("="*60)
    
    # Prepare data
    X = data[feature_columns].copy().fillna(data[feature_columns].median())
    y = data['injury_risk_label'].copy()
    
    # Split data (same as used in training)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Get best model for interpretability analysis
    best_model_name = max(
        advanced_model.performance_metrics.keys(),
        key=lambda k: advanced_model.performance_metrics[k].get('cv_auc_mean', 0)
    )
    
    print(f"Running interpretability analysis on best model: {best_model_name}")
    best_model = advanced_model.models[best_model_name]
    
    # Run comprehensive interpretability analysis
    analysis_results = comprehensive_interpretability_analysis(
        model=best_model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_columns
    )
    
    return analysis_results


def generate_phase3_summary(advanced_model: AdvancedInjuryRiskModel, 
                           interpretability_results: Dict[str, Any]) -> None:
    """Generate comprehensive Phase 3 summary report."""
    print("\n" + "="*60)
    print("PHASE 3: COMPREHENSIVE SUMMARY REPORT")
    print("="*60)
    
    # Model performance summary
    print("\nMODEL PERFORMANCE SUMMARY:")
    print("-" * 30)
    
    # Sort models by performance
    sorted_models = sorted(
        advanced_model.performance_metrics.items(),
        key=lambda x: x[1]['cv_auc_mean'],
        reverse=True
    )
    
    for rank, (model_name, metrics) in enumerate(sorted_models, 1):
        print(f"{rank}. {model_name.replace('_', ' ').title()}")
        print(f"   CV AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
        print(f"   Test AUC: {metrics.get('auc_roc', 'N/A')}")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall: {metrics['recall']:.3f}")
    
    # Key insights
    print("\nKEY INSIGHTS:")
    print("-" * 15)
    
    best_model = sorted_models[0][0]
    best_auc = sorted_models[0][1]['cv_auc_mean']
    
    print(f"• Best performing model: {best_model.replace('_', ' ').title()}")
    print(f"• Best CV AUC achieved: {best_auc:.3f}")
    
    if len(sorted_models) > 1:
        improvement = best_auc - sorted_models[1][1]['cv_auc_mean']
        print(f"• Performance improvement over 2nd best: {improvement:.3f}")
    
    # Ensemble performance
    if 'voting_ensemble' in advanced_model.performance_metrics:
        ensemble_auc = advanced_model.performance_metrics['voting_ensemble']['cv_auc_mean']
        print(f"• Voting ensemble AUC: {ensemble_auc:.3f}")
    
    # Top risk factors
    if 'top_risk_factors' in interpretability_results:
        print(f"\nTOP RISK FACTORS:")
        print("-" * 20)
        
        top_factors = list(interpretability_results['top_risk_factors'].items())[:10]
        for rank, (feature, score) in enumerate(top_factors, 1):
            print(f"{rank:2d}. {feature}: {score:.4f}")
    
    # Clinical implications
    if 'biomechanical_insights' in interpretability_results:
        insights = interpretability_results['biomechanical_insights']
        
        print(f"\nCLINICAL IMPLICATIONS:")
        print("-" * 25)
        
        if insights.get('stress_concentrations'):
            stress_areas = [item['category'] for item in insights['stress_concentrations']]
            unique_areas = list(set(stress_areas))
            print(f"• Primary stress concentrations: {', '.join(unique_areas)}")
        
        if insights.get('kinetic_chain_issues'):
            print(f"• Kinetic chain issues identified: {len(insights['kinetic_chain_issues'])} factors")
        
        if insights.get('asymmetry_patterns'):
            print(f"• Asymmetry patterns detected: {len(insights['asymmetry_patterns'])} factors")
    
    # Success metrics evaluation
    print(f"\nPHASE 3 SUCCESS METRICS EVALUATION:")
    print("-" * 40)
    
    # Technical metrics
    target_auc = 0.85
    target_precision = 0.80
    target_recall = 0.75
    
    achieved_auc = best_auc >= target_auc
    achieved_precision = sorted_models[0][1]['precision'] >= target_precision
    achieved_recall = sorted_models[0][1]['recall'] >= target_recall
    
    print(f"• AUC > 0.85: {'✓' if achieved_auc else '✗'} ({best_auc:.3f})")
    print(f"• Precision > 0.80: {'✓' if achieved_precision else '✗'} ({sorted_models[0][1]['precision']:.3f})")
    print(f"• Recall > 0.75: {'✓' if achieved_recall else '✗'} ({sorted_models[0][1]['recall']:.3f})")
    
    # Model interpretability
    interpretable = 'top_risk_factors' in interpretability_results and len(interpretability_results['top_risk_factors']) >= 10
    print(f"• Model interpretability: {'✓' if interpretable else '✗'}")
    
    # Risk factor identification
    risk_factors_identified = 'biomechanical_insights' in interpretability_results
    print(f"• Risk factors identified: {'✓' if risk_factors_identified else '✗'}")
    
    overall_success = sum([achieved_auc, achieved_precision, achieved_recall, interpretable, risk_factors_identified])
    print(f"\nOverall Phase 3 Success: {overall_success}/5 objectives met")
    
    # Next steps
    print(f"\nNEXT STEPS FOR PHASE 4:")
    print("-" * 25)
    print("• Model validation on holdout dataset")
    print("• Cross-validation performance analysis") 
    print("• Real-world applicability assessment")
    print("• Model serialization and API development")


def main():
    """Main Phase 3 demonstration function."""
    print("BASEBALL INJURY RISK ASSESSMENT - PHASE 3")
    print("Advanced Model Optimization & Interpretability")
    print("=" * 60)
    
    try:
        # Step 1: Load and prepare data
        data, feature_columns = load_and_prepare_data()
        
        # Step 2: Run advanced model optimization
        advanced_model = run_phase3_advanced_models(data, feature_columns)
        
        # Step 3: Run interpretability analysis
        interpretability_results = run_phase3_interpretability(advanced_model, data, feature_columns)
        
        # Step 4: Generate comprehensive summary
        generate_phase3_summary(advanced_model, interpretability_results)
        
        print(f"\n{'='*60}")
        print("PHASE 3 COMPLETED SUCCESSFULLY!")
        print("Advanced models trained and analyzed.")
        print("Results saved to ../results/phase3/")
        print("Ready to proceed to Phase 4: Validation & Deployment")
        print("="*60)
        
    except Exception as e:
        print(f"Error in Phase 3 demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
