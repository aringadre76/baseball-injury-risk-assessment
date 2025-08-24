"""
Advanced machine learning models for injury risk assessment.
Phase 3: Model Optimization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, 
    GridSearchCV, RandomizedSearchCV
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier, VotingClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, Tuple, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import pickle
import json

warnings.filterwarnings('ignore', category=FutureWarning)


class AdvancedInjuryRiskModel:
    """Advanced machine learning model ensemble for injury risk assessment."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the advanced model ensemble.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.pipelines = {}
        self.performance_metrics = {}
        self.feature_importance = {}
        self.hyperparameter_results = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        
    def create_model_ensemble(self) -> Dict[str, Any]:
        """
        Create a comprehensive ensemble of models for injury risk prediction.
        
        Returns:
            Dictionary of model configurations
        """
        models = {
            'random_forest_optimized': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'gradient_boosting_optimized': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=4,
                subsample=0.8,
                random_state=self.random_state
            ),
            
            'extra_trees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'svm_rbf': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=self.random_state
            ),
            
            'logistic_regression_l2': LogisticRegression(
                penalty='l2',
                C=1.0,
                max_iter=1000,
                random_state=self.random_state
            ),
            
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=self.random_state
            )
        }
        
        return models
    
    def create_hyperparameter_grids(self) -> Dict[str, Dict]:
        """
        Create hyperparameter grids for optimization.
        
        Returns:
            Dictionary of hyperparameter grids for each model
        """
        param_grids = {
            'random_forest_optimized': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            
            'gradient_boosting_optimized': {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [6, 8, 10],
                'min_samples_split': [5, 10, 15],
                'subsample': [0.8, 0.9, 1.0]
            },
            
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            },
            
            'svm_rbf': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
        }
        
        return param_grids
    
    def train_ensemble_models(self, X: pd.DataFrame, y: pd.Series, 
                            optimize_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Train comprehensive ensemble of models with optional hyperparameter optimization.
        
        Args:
            X: Features
            y: Target
            optimize_hyperparameters: Whether to perform hyperparameter optimization
            
        Returns:
            Dictionary with trained models and performance metrics
        """
        print("Training advanced ensemble models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale features for algorithms that need it
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models_config = self.create_model_ensemble()
        param_grids = self.create_hyperparameter_grids()
        
        results = {}
        
        for model_name, base_model in models_config.items():
            print(f"\nTraining {model_name}...")
            
            # Determine if model needs scaling
            needs_scaling = model_name in ['svm_rbf', 'logistic_regression_l2', 'neural_network']
            
            if needs_scaling:
                X_train_model = X_train_scaled
                X_test_model = X_test_scaled
                # Create pipeline with scaling
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', base_model)
                ])
                # Adjust parameter grid for pipeline
                if optimize_hyperparameters and model_name in param_grids:
                    param_grid = {f'classifier__{k}': v for k, v in param_grids[model_name].items()}
                else:
                    param_grid = None
            else:
                X_train_model = X_train
                X_test_model = X_test
                model = base_model
                param_grid = param_grids.get(model_name) if optimize_hyperparameters else None
            
            # Hyperparameter optimization
            if optimize_hyperparameters and param_grid:
                print(f"  Optimizing hyperparameters for {model_name}...")
                
                # Use RandomizedSearchCV for efficiency
                search = RandomizedSearchCV(
                    model,
                    param_grid,
                    n_iter=20,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                    scoring='roc_auc',
                    random_state=self.random_state,
                    n_jobs=-1 if model_name != 'neural_network' else 1
                )
                
                search.fit(X_train_model if not needs_scaling else X_train, y_train)
                model = search.best_estimator_
                self.hyperparameter_results[model_name] = {
                    'best_params': search.best_params_,
                    'best_score': search.best_score_
                }
                print(f"  Best CV score: {search.best_score_:.3f}")
            else:
                # Train with default parameters
                model.fit(X_train_model if not needs_scaling else X_train, y_train)
            
            # Predictions
            if needs_scaling and not optimize_hyperparameters:
                y_pred = model.predict(X_test_model)
                y_pred_proba = model.predict_proba(X_test_model)[:, 1] if hasattr(model, 'predict_proba') else None
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = self._calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, 
                X_train if not (needs_scaling and not optimize_hyperparameters) else X_train_model, 
                y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring='roc_auc',
                n_jobs=-1 if model_name != 'neural_network' else 1
            )
            metrics['cv_auc_mean'] = cv_scores.mean()
            metrics['cv_auc_std'] = cv_scores.std()
            
            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[model_name] = feature_importance
            elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('classifier'), 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.named_steps['classifier'].feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[model_name] = feature_importance
            
            # Store results
            results[model_name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'test_indices': X_test.index
            }
            
            # Store in class attributes
            self.models[model_name] = model
            self.performance_metrics[model_name] = metrics
            
            self._print_model_metrics(model_name, metrics)
        
        # Create voting ensemble
        self._create_voting_ensemble(X_train, y_train, X_test, y_test)
        
        return results
    
    def _calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba=None) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        metrics = {
            'accuracy': (y_pred == y_true).mean(),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'specificity': self._calculate_specificity(y_true, y_pred),
        }
        
        if y_pred_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
        
        return metrics
    
    def _calculate_specificity(self, y_true, y_pred) -> float:
        """Calculate specificity (true negative rate)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def _print_model_metrics(self, model_name: str, metrics: Dict[str, float]) -> None:
        """Print model performance metrics."""
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  Specificity: {metrics['specificity']:.3f}")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
        if 'auc_roc' in metrics:
            print(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
        if 'avg_precision' in metrics:
            print(f"  Avg Precision: {metrics['avg_precision']:.3f}")
        print(f"  CV AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
    
    def _create_voting_ensemble(self, X_train, y_train, X_test, y_test) -> None:
        """Create a voting ensemble from best performing models."""
        print("\nCreating voting ensemble...")
        
        # Select top 3 models based on CV AUC
        sorted_models = sorted(
            self.performance_metrics.items(),
            key=lambda x: x[1]['cv_auc_mean'],
            reverse=True
        )[:3]
        
        ensemble_models = []
        for model_name, _ in sorted_models:
            ensemble_models.append((model_name, self.models[model_name]))
        
        # Create voting classifier
        voting_ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'
        )
        
        voting_ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        y_pred = voting_ensemble.predict(X_test)
        y_pred_proba = voting_ensemble.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
        
        # Cross-validation for ensemble
        cv_scores = cross_val_score(
            voting_ensemble, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='roc_auc'
        )
        metrics['cv_auc_mean'] = cv_scores.mean()
        metrics['cv_auc_std'] = cv_scores.std()
        
        self.ensemble_model = voting_ensemble
        self.models['voting_ensemble'] = voting_ensemble
        self.performance_metrics['voting_ensemble'] = metrics
        
        print("Ensemble model created from top 3 performers:")
        for model_name, _ in sorted_models:
            print(f"  - {model_name}")
        
        self._print_model_metrics('voting_ensemble', metrics)
    
    def plot_model_comparison(self, save_path: Optional[str] = None) -> None:
        """Plot comprehensive model comparison."""
        if not self.performance_metrics:
            print("No models trained yet. Train models first.")
            return
        
        # Prepare data for plotting
        metrics_df = pd.DataFrame(self.performance_metrics).T
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auc_roc']
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum', 'orange']
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i//3, i%3]
            
            if metric in metrics_df.columns and metrics_df[metric].notna().any():
                bars = metrics_df[metric].plot(kind='bar', ax=ax, color=colors[i])
                ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
                ax.set_ylabel('Score')
                ax.set_ylim(0, 1)
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for j, v in enumerate(metrics_df[metric]):
                    if not pd.isna(v):
                        ax.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
            else:
                ax.text(0.5, 0.5, f'No {metric} data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{metric.replace("_", " ").title()}')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance_comparison(self, top_n: int = 15, save_path: Optional[str] = None) -> None:
        """Plot feature importance comparison across models."""
        if not self.feature_importance:
            print("No feature importance data available.")
            return
        
        # Combine feature importance from all models
        combined_importance = pd.DataFrame()
        
        for model_name, importance_df in self.feature_importance.items():
            importance_df_top = importance_df.head(top_n).copy()
            importance_df_top['model'] = model_name
            combined_importance = pd.concat([combined_importance, importance_df_top], ignore_index=True)
        
        # Create pivot table for heatmap
        pivot_table = combined_importance.pivot(index='feature', columns='model', values='importance')
        pivot_table = pivot_table.fillna(0)
        
        # Plot heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Feature Importance'})
        plt.title(f'Feature Importance Comparison - Top {top_n} Features', fontsize=14, fontweight='bold')
        plt.xlabel('Models')
        plt.ylabel('Features')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, X_test, y_test, save_path: Optional[str] = None) -> None:
        """Plot ROC curves for all models."""
        plt.figure(figsize=(12, 8))
        
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive model performance report."""
        if not self.performance_metrics:
            return "No models trained yet."
        
        report = "ADVANCED MODEL ENSEMBLE PERFORMANCE REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Overall best model
        best_model = max(self.performance_metrics.keys(), 
                        key=lambda k: self.performance_metrics[k].get('cv_auc_mean', 0))
        
        report += f"Best performing model: {best_model.replace('_', ' ').title()}\n"
        report += f"Best CV AUC: {self.performance_metrics[best_model]['cv_auc_mean']:.3f}\n\n"
        
        # Model ranking by CV AUC
        report += "MODEL RANKING (by CV AUC):\n"
        report += "-" * 30 + "\n"
        
        sorted_models = sorted(
            self.performance_metrics.items(),
            key=lambda x: x[1]['cv_auc_mean'],
            reverse=True
        )
        
        for rank, (model_name, metrics) in enumerate(sorted_models, 1):
            report += f"{rank}. {model_name.replace('_', ' ').title()}: {metrics['cv_auc_mean']:.3f}\n"
        
        report += "\nDETAILED METRICS:\n"
        report += "=" * 30 + "\n\n"
        
        # Detailed metrics for each model
        for model_name, metrics in self.performance_metrics.items():
            report += f"{model_name.replace('_', ' ').title()}:\n"
            report += f"  Accuracy: {metrics['accuracy']:.3f}\n"
            report += f"  Precision: {metrics['precision']:.3f}\n"
            report += f"  Recall: {metrics['recall']:.3f}\n"
            report += f"  Specificity: {metrics['specificity']:.3f}\n"
            report += f"  F1-Score: {metrics['f1_score']:.3f}\n"
            if metrics.get('auc_roc'):
                report += f"  AUC-ROC: {metrics['auc_roc']:.3f}\n"
            if metrics.get('avg_precision'):
                report += f"  Avg Precision: {metrics['avg_precision']:.3f}\n"
            report += f"  CV AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}\n\n"
        
        # Hyperparameter optimization results
        if self.hyperparameter_results:
            report += "HYPERPARAMETER OPTIMIZATION RESULTS:\n"
            report += "=" * 40 + "\n\n"
            
            for model_name, hp_results in self.hyperparameter_results.items():
                report += f"{model_name.replace('_', ' ').title()}:\n"
                report += f"  Best CV Score: {hp_results['best_score']:.3f}\n"
                report += f"  Best Parameters:\n"
                for param, value in hp_results['best_params'].items():
                    report += f"    {param}: {value}\n"
                report += "\n"
        
        # Feature importance summary
        if self.feature_importance:
            report += "TOP RISK FACTORS SUMMARY:\n"
            report += "=" * 30 + "\n\n"
            
            # Average importance across models
            all_features = set()
            for importance_df in self.feature_importance.values():
                all_features.update(importance_df['feature'].tolist())
            
            feature_avg_importance = {}
            for feature in all_features:
                importances = []
                for importance_df in self.feature_importance.values():
                    feature_row = importance_df[importance_df['feature'] == feature]
                    if not feature_row.empty:
                        importances.append(feature_row['importance'].iloc[0])
                if importances:
                    feature_avg_importance[feature] = np.mean(importances)
            
            # Sort by average importance
            sorted_features = sorted(feature_avg_importance.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            report += "Top 15 features (averaged across models):\n"
            for rank, (feature, avg_importance) in enumerate(sorted_features[:15], 1):
                report += f"{rank:2d}. {feature}: {avg_importance:.4f}\n"
        
        return report
    
    def save_advanced_results(self, output_dir: str = "results/phase3") -> None:
        """Save comprehensive model results and analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = output_path / f"{model_name}_advanced.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save performance metrics
        metrics_df = pd.DataFrame(self.performance_metrics).T
        metrics_df.to_csv(output_path / "advanced_model_metrics.csv")
        
        # Save feature importance
        for model_name, importance_df in self.feature_importance.items():
            importance_df.to_csv(output_path / f"{model_name}_feature_importance.csv", index=False)
        
        # Save hyperparameter results
        if self.hyperparameter_results:
            with open(output_path / "hyperparameter_results.json", 'w') as f:
                json.dump(self.hyperparameter_results, f, indent=2)
        
        # Save comprehensive report
        report = self.generate_comprehensive_report()
        with open(output_path / "advanced_model_report.txt", 'w') as f:
            f.write(report)
        
        # Save scaler
        scaler_path = output_path / "feature_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Advanced results saved to {output_path}")
    
    def save_model(self, output_path: str) -> None:
        """Save the complete model instance for later use."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'wb') as f:
            pickle.dump(self, f)
        print(f"Complete model saved to {output_file}")
    
    def load_model(cls, model_path: str) -> 'AdvancedInjuryRiskModel':
        """Load a saved model instance."""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")
        return model


def run_advanced_model_analysis(data: pd.DataFrame, 
                               feature_columns: List[str],
                               target_column: str = 'injury_risk_label',
                               optimize_hyperparameters: bool = True) -> AdvancedInjuryRiskModel:
    """
    Complete advanced model analysis pipeline.
    
    Args:
        data: Input DataFrame
        feature_columns: List of feature columns
        target_column: Target column name
        optimize_hyperparameters: Whether to optimize hyperparameters
        
    Returns:
        Trained AdvancedInjuryRiskModel instance
    """
    print("Starting Advanced Model Analysis (Phase 3)")
    print("=" * 50)
    
    model = AdvancedInjuryRiskModel()
    
    # Prepare data (reuse from baseline model)
    from .baseline_injury_model import BaselineInjuryRiskModel
    baseline = BaselineInjuryRiskModel()
    X, y = baseline.prepare_features_and_target(data, feature_columns, target_column)
    
    # Train advanced models
    results = model.train_ensemble_models(X, y, optimize_hyperparameters)
    
    # Generate comprehensive visualizations
    print("\nGenerating visualizations...")
    model.plot_model_comparison()
    model.plot_feature_importance_comparison()
    
    # ROC curves (need test data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=model.random_state, stratify=y
    )
    model.plot_roc_curves(X_test, y_test)
    
    # Print comprehensive report
    print("\n" + model.generate_comprehensive_report())
    
    # Save results
    model.save_advanced_results()
    
    return model
