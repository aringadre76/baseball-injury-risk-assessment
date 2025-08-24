"""
Model interpretability and explainability tools for injury risk assessment.
Phase 3: SHAP analysis and risk factor identification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("SHAP not available. Install with: pip install shap")
    SHAP_AVAILABLE = False

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
import pickle


class InjuryRiskExplainer:
    """Comprehensive model interpretability and explanation framework."""
    
    def __init__(self, model: BaseEstimator, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                 feature_names: List[str]):
        """
        Initialize the explainer.
        
        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features  
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.shap_explainer = None
        self.shap_values = None
        self.permutation_importance_results = None
        
    def initialize_shap_explainer(self, explainer_type: str = 'auto') -> None:
        """
        Initialize SHAP explainer for the model.
        
        Args:
            explainer_type: Type of SHAP explainer ('tree', 'linear', 'kernel', 'auto')
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available. Skipping SHAP analysis.")
            return
        
        print("Initializing SHAP explainer...")
        
        if explainer_type == 'auto':
            # Auto-detect best explainer type
            if hasattr(self.model, 'feature_importances_'):
                explainer_type = 'tree'
            elif hasattr(self.model, 'coef_'):
                explainer_type = 'linear'
            else:
                explainer_type = 'kernel'
        
        try:
            if explainer_type == 'tree':
                # For tree-based models
                if hasattr(self.model, 'named_steps'):
                    # Pipeline model - extract the classifier
                    classifier = self.model.named_steps.get('classifier')
                    if hasattr(classifier, 'feature_importances_'):
                        self.shap_explainer = shap.TreeExplainer(classifier)
                    else:
                        self.shap_explainer = shap.KernelExplainer(
                            self.model.predict_proba, 
                            shap.sample(self.X_train, 100)
                        )
                else:
                    self.shap_explainer = shap.TreeExplainer(self.model)
                    
            elif explainer_type == 'linear':
                # For linear models
                if hasattr(self.model, 'named_steps'):
                    classifier = self.model.named_steps.get('classifier')
                    self.shap_explainer = shap.LinearExplainer(classifier, self.X_train)
                else:
                    self.shap_explainer = shap.LinearExplainer(self.model, self.X_train)
                    
            elif explainer_type == 'kernel':
                # General kernel explainer (slower but works for any model)
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba, 
                    shap.sample(self.X_train, 100)
                )
            
            print(f"SHAP {explainer_type} explainer initialized successfully.")
            
        except Exception as e:
            print(f"Failed to initialize SHAP explainer: {e}")
            print("Falling back to kernel explainer...")
            try:
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba, 
                    shap.sample(self.X_train, 100)
                )
                print("Kernel explainer initialized successfully.")
            except Exception as e2:
                print(f"Failed to initialize any SHAP explainer: {e2}")
                self.shap_explainer = None
    
    def calculate_shap_values(self, sample_size: int = 100) -> None:
        """
        Calculate SHAP values for the test set.
        
        Args:
            sample_size: Number of samples to use for explanation
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            print("SHAP explainer not available.")
            return
        
        print(f"Calculating SHAP values for {sample_size} samples...")
        
        # Sample test data for efficiency
        if len(self.X_test) > sample_size:
            sample_indices = np.random.choice(len(self.X_test), sample_size, replace=False)
            X_sample = self.X_test.iloc[sample_indices]
        else:
            X_sample = self.X_test
        
        try:
            # Handle different explainer types
            if isinstance(self.shap_explainer, shap.TreeExplainer):
                if hasattr(self.model, 'named_steps'):
                    # For pipeline models, transform the data first
                    X_transformed = self.model.named_steps['scaler'].transform(X_sample)
                    self.shap_values = self.shap_explainer.shap_values(X_transformed)
                else:
                    self.shap_values = self.shap_explainer.shap_values(X_sample)
                    
                # For binary classification, use positive class
                if isinstance(self.shap_values, list):
                    self.shap_values = self.shap_values[1]
                    
            elif isinstance(self.shap_explainer, shap.LinearExplainer):
                if hasattr(self.model, 'named_steps'):
                    X_transformed = self.model.named_steps['scaler'].transform(X_sample)
                    self.shap_values = self.shap_explainer.shap_values(X_transformed)
                else:
                    self.shap_values = self.shap_explainer.shap_values(X_sample)
                    
            else:  # KernelExplainer
                self.shap_values = self.shap_explainer.shap_values(X_sample)
                if isinstance(self.shap_values, list):
                    self.shap_values = self.shap_values[1]
            
            print("SHAP values calculated successfully.")
            
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            self.shap_values = None
    
    def plot_shap_summary(self, save_path: Optional[str] = None) -> None:
        """Plot SHAP summary plot."""
        if not SHAP_AVAILABLE or self.shap_values is None:
            print("SHAP values not available.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Get corresponding X data
        sample_size = self.shap_values.shape[0]
        if len(self.X_test) > sample_size:
            sample_indices = np.random.choice(len(self.X_test), sample_size, replace=False)
            X_sample = self.X_test.iloc[sample_indices]
        else:
            X_sample = self.X_test
        
        shap.summary_plot(self.shap_values, X_sample, feature_names=self.feature_names, show=False)
        plt.title('SHAP Summary Plot - Injury Risk Factors', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_shap_waterfall(self, instance_idx: int = 0, save_path: Optional[str] = None) -> None:
        """Plot SHAP waterfall plot for a specific instance."""
        if not SHAP_AVAILABLE or self.shap_values is None:
            print("SHAP values not available.")
            return
        
        if instance_idx >= self.shap_values.shape[0]:
            print(f"Instance index {instance_idx} out of range.")
            return
        
        # Get the base value (expected value)
        if hasattr(self.shap_explainer, 'expected_value'):
            expected_value = self.shap_explainer.expected_value
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        else:
            expected_value = 0
        
        # Create explanation object
        explanation = shap.Explanation(
            values=self.shap_values[instance_idx],
            base_values=expected_value,
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explanation, show=False)
        plt.title(f'SHAP Waterfall Plot - Instance {instance_idx}', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_shap_feature_importance(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """Plot SHAP-based feature importance."""
        if not SHAP_AVAILABLE or self.shap_values is None:
            print("SHAP values not available.")
            return pd.DataFrame()
        
        # Calculate mean absolute SHAP values
        feature_importance = np.abs(self.shap_values).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': feature_importance
        }).sort_values('shap_importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(20)
        
        sns.barplot(data=top_features, y='feature', x='shap_importance', palette='viridis')
        plt.title('SHAP-based Feature Importance - Top 20 Features', fontsize=14, fontweight='bold')
        plt.xlabel('Mean |SHAP Value|')
        plt.ylabel('Features')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    def calculate_permutation_importance(self, X_test: pd.DataFrame, y_test: pd.Series, 
                                       n_repeats: int = 10) -> pd.DataFrame:
        """
        Calculate permutation importance for model features.
        
        Args:
            X_test: Test features
            y_test: Test targets
            n_repeats: Number of permutation repeats
            
        Returns:
            DataFrame with permutation importance results
        """
        print(f"Calculating permutation importance with {n_repeats} repeats...")
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            self.model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=42,
            scoring='roc_auc'
        )
        
        # Create results DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        self.permutation_importance_results = importance_df
        
        print("Permutation importance calculated successfully.")
        return importance_df
    
    def plot_permutation_importance(self, top_n: int = 20, save_path: Optional[str] = None) -> None:
        """Plot permutation importance results."""
        if self.permutation_importance_results is None:
            print("Permutation importance not calculated yet.")
            return
        
        top_features = self.permutation_importance_results.head(top_n)
        
        plt.figure(figsize=(12, 8))
        plt.errorbar(
            top_features['importance_mean'], 
            range(len(top_features)),
            xerr=top_features['importance_std'],
            fmt='o'
        )
        
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Permutation Importance (AUC decrease)')
        plt.ylabel('Features')
        plt.title(f'Permutation Importance - Top {top_n} Features', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_risk_factors(self) -> Dict[str, Any]:
        """
        Comprehensive risk factor analysis combining multiple interpretability methods.
        
        Returns:
            Dictionary with risk factor analysis results
        """
        print("Conducting comprehensive risk factor analysis...")
        
        analysis_results = {
            'top_risk_factors': {},
            'consistency_analysis': {},
            'biomechanical_insights': {}
        }
        
        # Collect feature importance from different methods
        importance_methods = {}
        
        # SHAP importance
        if SHAP_AVAILABLE and self.shap_values is not None:
            shap_importance = np.abs(self.shap_values).mean(axis=0)
            importance_methods['shap'] = dict(zip(self.feature_names, shap_importance))
        
        # Permutation importance
        if self.permutation_importance_results is not None:
            perm_dict = dict(zip(
                self.permutation_importance_results['feature'],
                self.permutation_importance_results['importance_mean']
            ))
            importance_methods['permutation'] = perm_dict
        
        # Model-specific importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            model_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            importance_methods['model_intrinsic'] = model_importance
        elif hasattr(self.model, 'named_steps') and hasattr(self.model.named_steps.get('classifier'), 'feature_importances_'):
            classifier = self.model.named_steps['classifier']
            model_importance = dict(zip(self.feature_names, classifier.feature_importances_))
            importance_methods['model_intrinsic'] = model_importance
        
        # Calculate consensus ranking
        if importance_methods:
            consensus_ranking = self._calculate_consensus_ranking(importance_methods)
            analysis_results['top_risk_factors'] = consensus_ranking
            
            # Analyze consistency
            consistency = self._analyze_method_consistency(importance_methods)
            analysis_results['consistency_analysis'] = consistency
        
        # Biomechanical insights
        biomech_insights = self._extract_biomechanical_insights(
            analysis_results.get('top_risk_factors', {})
        )
        analysis_results['biomechanical_insights'] = biomech_insights
        
        return analysis_results
    
    def _calculate_consensus_ranking(self, importance_methods: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate consensus ranking across different importance methods."""
        all_features = set()
        for method_scores in importance_methods.values():
            all_features.update(method_scores.keys())
        
        consensus_scores = {}
        
        for feature in all_features:
            scores = []
            for method_name, method_scores in importance_methods.items():
                if feature in method_scores:
                    # Normalize scores to 0-1 range within each method
                    max_score = max(method_scores.values())
                    min_score = min(method_scores.values())
                    if max_score > min_score:
                        normalized_score = (method_scores[feature] - min_score) / (max_score - min_score)
                    else:
                        normalized_score = 1.0
                    scores.append(normalized_score)
            
            if scores:
                consensus_scores[feature] = np.mean(scores)
        
        # Sort by consensus score
        return dict(sorted(consensus_scores.items(), key=lambda x: x[1], reverse=True))
    
    def _analyze_method_consistency(self, importance_methods: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze consistency between different importance methods."""
        if len(importance_methods) < 2:
            return {"message": "Need at least 2 methods for consistency analysis"}
        
        # Get top 10 features from each method
        top_features_by_method = {}
        for method_name, scores in importance_methods.items():
            sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_features_by_method[method_name] = [f[0] for f in sorted_features[:10]]
        
        # Calculate intersection of top features
        method_names = list(top_features_by_method.keys())
        intersection = set(top_features_by_method[method_names[0]])
        for method in method_names[1:]:
            intersection = intersection.intersection(set(top_features_by_method[method]))
        
        consistency_analysis = {
            'common_top_features': list(intersection),
            'consistency_score': len(intersection) / 10,  # Fraction of top 10 that are common
            'method_specific_features': {}
        }
        
        # Find method-specific top features
        for method_name, top_features in top_features_by_method.items():
            unique_features = set(top_features) - intersection
            consistency_analysis['method_specific_features'][method_name] = list(unique_features)
        
        return consistency_analysis
    
    def _extract_biomechanical_insights(self, top_risk_factors: Dict[str, float]) -> Dict[str, Any]:
        """Extract biomechanical insights from top risk factors."""
        insights = {
            'kinetic_chain_issues': [],
            'stress_concentrations': [],
            'movement_quality_indicators': [],
            'asymmetry_patterns': []
        }
        
        # Define biomechanical categories
        biomech_categories = {
            'elbow_stress': ['elbow_varus', 'elbow_extension', 'elbow_flexion'],
            'shoulder_stress': ['shoulder_internal_rotation', 'shoulder_external_rotation', 'shoulder_abduction'],
            'torso_mechanics': ['torso_rotation', 'torso_tilt', 'torso_anterior'],
            'hip_mechanics': ['hip_rotation', 'pelvis_tilt', 'pelvis_rotation'],
            'kinetic_chain': ['hip_shoulder_separation', 'energy_flow', 'sequential_timing'],
            'force_production': ['force_plate', 'ground_reaction', 'peak_force'],
            'asymmetry': ['asymmetry', 'left_right', 'dominant_non_dominant']
        }
        
        # Analyze top risk factors
        top_10_features = list(top_risk_factors.keys())[:10]
        
        for feature in top_10_features:
            feature_lower = feature.lower()
            
            # Categorize the feature
            for category, keywords in biomech_categories.items():
                if any(keyword in feature_lower for keyword in keywords):
                    if category == 'elbow_stress' or category == 'shoulder_stress':
                        insights['stress_concentrations'].append({
                            'feature': feature,
                            'category': category,
                            'importance': top_risk_factors[feature]
                        })
                    elif category == 'asymmetry':
                        insights['asymmetry_patterns'].append({
                            'feature': feature,
                            'importance': top_risk_factors[feature]
                        })
                    elif category == 'kinetic_chain':
                        insights['kinetic_chain_issues'].append({
                            'feature': feature,
                            'importance': top_risk_factors[feature]
                        })
                    else:
                        insights['movement_quality_indicators'].append({
                            'feature': feature,
                            'category': category,
                            'importance': top_risk_factors[feature]
                        })
                    break
        
        return insights
    
    def generate_interpretability_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comprehensive interpretability report."""
        report = "MODEL INTERPRETABILITY AND RISK FACTOR ANALYSIS\n"
        report += "=" * 60 + "\n\n"
        
        # Top risk factors
        if 'top_risk_factors' in analysis_results:
            report += "TOP INJURY RISK FACTORS (Consensus Ranking):\n"
            report += "-" * 45 + "\n"
            
            top_factors = analysis_results['top_risk_factors']
            for rank, (feature, score) in enumerate(list(top_factors.items())[:15], 1):
                report += f"{rank:2d}. {feature}: {score:.4f}\n"
            report += "\n"
        
        # Consistency analysis
        if 'consistency_analysis' in analysis_results:
            consistency = analysis_results['consistency_analysis']
            report += "METHOD CONSISTENCY ANALYSIS:\n"
            report += "-" * 30 + "\n"
            
            if 'consistency_score' in consistency:
                report += f"Consistency Score: {consistency['consistency_score']:.2f}\n"
                report += f"Common Top Features: {len(consistency.get('common_top_features', []))}/10\n\n"
                
                if consistency.get('common_top_features'):
                    report += "Features consistently ranked high:\n"
                    for feature in consistency['common_top_features']:
                        report += f"  - {feature}\n"
                    report += "\n"
        
        # Biomechanical insights
        if 'biomechanical_insights' in analysis_results:
            insights = analysis_results['biomechanical_insights']
            report += "BIOMECHANICAL INSIGHTS:\n"
            report += "-" * 25 + "\n"
            
            if insights.get('stress_concentrations'):
                report += "Key Stress Concentration Areas:\n"
                for item in insights['stress_concentrations']:
                    report += f"  - {item['feature']} ({item['category']}): {item['importance']:.4f}\n"
                report += "\n"
            
            if insights.get('kinetic_chain_issues'):
                report += "Kinetic Chain Issues:\n"
                for item in insights['kinetic_chain_issues']:
                    report += f"  - {item['feature']}: {item['importance']:.4f}\n"
                report += "\n"
            
            if insights.get('asymmetry_patterns'):
                report += "Asymmetry Patterns:\n"
                for item in insights['asymmetry_patterns']:
                    report += f"  - {item['feature']}: {item['importance']:.4f}\n"
                report += "\n"
            
            if insights.get('movement_quality_indicators'):
                report += "Movement Quality Indicators:\n"
                for item in insights['movement_quality_indicators']:
                    report += f"  - {item['feature']} ({item['category']}): {item['importance']:.4f}\n"
                report += "\n"
        
        # Recommendations
        report += "CLINICAL RECOMMENDATIONS:\n"
        report += "-" * 25 + "\n"
        report += self._generate_clinical_recommendations(analysis_results)
        
        return report
    
    def _generate_clinical_recommendations(self, analysis_results: Dict[str, Any]) -> str:
        """Generate clinical recommendations based on analysis results."""
        recommendations = ""
        
        if 'biomechanical_insights' not in analysis_results:
            return "Insufficient data for clinical recommendations.\n"
        
        insights = analysis_results['biomechanical_insights']
        
        if insights.get('stress_concentrations'):
            recommendations += "1. Stress Management:\n"
            elbow_stress = any('elbow' in item['category'] for item in insights['stress_concentrations'])
            shoulder_stress = any('shoulder' in item['category'] for item in insights['stress_concentrations'])
            
            if elbow_stress:
                recommendations += "   - Focus on elbow valgus stress reduction through improved mechanics\n"
                recommendations += "   - Implement UCL injury prevention protocols\n"
            
            if shoulder_stress:
                recommendations += "   - Address shoulder internal rotation stress patterns\n"
                recommendations += "   - Strengthen posterior shoulder stabilizers\n"
            
            recommendations += "\n"
        
        if insights.get('kinetic_chain_issues'):
            recommendations += "2. Kinetic Chain Optimization:\n"
            recommendations += "   - Improve hip-shoulder separation timing\n"
            recommendations += "   - Focus on sequential energy transfer patterns\n"
            recommendations += "   - Address movement efficiency deficits\n\n"
        
        if insights.get('asymmetry_patterns'):
            recommendations += "3. Asymmetry Correction:\n"
            recommendations += "   - Implement bilateral strength and mobility assessments\n"
            recommendations += "   - Address significant left-right imbalances\n"
            recommendations += "   - Focus on symmetrical movement patterns\n\n"
        
        recommendations += "4. Monitoring Recommendations:\n"
        recommendations += "   - Regular biomechanical assessments focusing on top risk factors\n"
        recommendations += "   - Workload management based on stress indicators\n"
        recommendations += "   - Early intervention when risk metrics exceed thresholds\n"
        
        return recommendations
    
    def save_interpretability_results(self, analysis_results: Dict[str, Any], 
                                    output_dir: str = "../results/phase3/interpretability") -> None:
        """Save all interpretability analysis results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save SHAP values if available
        if self.shap_values is not None:
            np.save(output_path / "shap_values.npy", self.shap_values)
        
        # Save permutation importance
        if self.permutation_importance_results is not None:
            self.permutation_importance_results.to_csv(
                output_path / "permutation_importance.csv", index=False
            )
        
        # Save analysis results
        import json
        with open(output_path / "risk_factor_analysis.json", 'w') as f:
            # Convert numpy types to regular Python types for JSON serialization
            json_compatible = self._make_json_compatible(analysis_results)
            json.dump(json_compatible, f, indent=2)
        
        # Save interpretability report
        report = self.generate_interpretability_report(analysis_results)
        with open(output_path / "interpretability_report.txt", 'w') as f:
            f.write(report)
        
        # Save feature rankings
        if 'top_risk_factors' in analysis_results:
            ranking_df = pd.DataFrame([
                {'feature': feature, 'consensus_score': score}
                for feature, score in analysis_results['top_risk_factors'].items()
            ])
            ranking_df.to_csv(output_path / "consensus_feature_ranking.csv", index=False)
        
        print(f"Interpretability results saved to {output_path}")
    
    def _make_json_compatible(self, obj):
        """Convert numpy types to JSON-compatible types."""
        if isinstance(obj, dict):
            return {key: self._make_json_compatible(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_compatible(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


def comprehensive_interpretability_analysis(model: BaseEstimator, 
                                          X_train: pd.DataFrame, 
                                          X_test: pd.DataFrame, 
                                          y_test: pd.Series,
                                          feature_names: List[str]) -> Dict[str, Any]:
    """
    Run comprehensive interpretability analysis for a trained model.
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        y_test: Test targets
        feature_names: List of feature names
        
    Returns:
        Dictionary with all analysis results
    """
    print("Starting Comprehensive Model Interpretability Analysis")
    print("=" * 55)
    
    # Initialize explainer
    explainer = InjuryRiskExplainer(model, X_train, X_test, feature_names)
    
    # SHAP analysis
    if SHAP_AVAILABLE:
        explainer.initialize_shap_explainer()
        explainer.calculate_shap_values()
        
        # Generate SHAP plots
        explainer.plot_shap_summary()
        explainer.plot_shap_waterfall(instance_idx=0)
        shap_importance = explainer.plot_shap_feature_importance()
    
    # Permutation importance
    perm_importance = explainer.calculate_permutation_importance(X_test, y_test)
    explainer.plot_permutation_importance()
    
    # Comprehensive risk factor analysis
    analysis_results = explainer.analyze_risk_factors()
    
    # Generate and print report
    report = explainer.generate_interpretability_report(analysis_results)
    print("\n" + report)
    
    # Save results
    explainer.save_interpretability_results(analysis_results)
    
    return analysis_results

