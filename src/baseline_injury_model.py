"""
Baseline machine learning models for injury risk assessment.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Dict, Tuple, List, Any
import matplotlib.pyplot as plt
import seaborn as sns


class BaselineInjuryRiskModel:
    """Baseline machine learning model for injury risk assessment."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the baseline model.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
    def prepare_features_and_target(self, data: pd.DataFrame, 
                                  feature_columns: List[str],
                                  target_column: str = 'injury_risk_label') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for modeling.
        
        Args:
            data: Input DataFrame
            feature_columns: List of feature column names
            target_column: Name of target column
            
        Returns:
            Tuple of (features, target)
        """
        # Filter to available columns
        available_features = [col for col in feature_columns if col in data.columns]
        
        if not available_features:
            raise ValueError("No valid feature columns found in data")
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Get complete cases only
        subset_data = data[available_features + [target_column]].dropna()
        
        X = subset_data[available_features]
        y = subset_data[target_column]
        
        # Ensure y is a Series, not DataFrame
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                y = y.iloc[:, 0]
            else:
                print(f"Warning: Target has {y.shape[1]} columns, using first column")
                y = y.iloc[:, 0]
        
        print(f"Prepared dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Handle target distribution safely
        try:
            target_dist = y.value_counts(dropna=False).to_dict()
            print(f"Target distribution: {target_dist}")
        except Exception as e:
            print(f"Target type: {type(y)}, shape: {getattr(y, 'shape', 'N/A')}")
            print(f"Target distribution calculation failed: {e}")
            # Simple fallback
            unique_vals, counts = np.unique(y, return_counts=True)
            target_dist = dict(zip(unique_vals, counts))
            print(f"Target distribution (fallback): {target_dist}")
        
        return X, y
    
    def train_baseline_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train multiple baseline models.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Dictionary with trained models and performance metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Define models
        models_to_train = {
            'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, max_depth=10
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=self.random_state, max_depth=6
            )
        }
        
        results = {}
        
        for model_name, model in models_to_train.items():
            print(f"\nTraining {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'accuracy': model.score(X_test, y_test),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'auc_roc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            }
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, 
                                      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                                      scoring='roc_auc')
            metrics['cv_auc_mean'] = cv_scores.mean()
            metrics['cv_auc_std'] = cv_scores.std()
            
            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
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
            
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1-Score: {metrics['f1_score']:.3f}")
            if metrics['auc_roc']:
                print(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
            print(f"  CV AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
        
        return results
    
    def plot_model_comparison(self) -> None:
        """Plot comparison of model performance metrics."""
        if not self.performance_metrics:
            print("No models trained yet. Train models first.")
            return
        
        # Prepare data for plotting
        metrics_df = pd.DataFrame(self.performance_metrics).T
        
        # Plot metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i//2, i%2]
            
            if metric in metrics_df.columns:
                metrics_df[metric].plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_ylabel('Score')
                ax.set_ylim(0, 1)
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for j, v in enumerate(metrics_df[metric]):
                    ax.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # AUC comparison if available
        if 'auc_roc' in metrics_df.columns and metrics_df['auc_roc'].notna().any():
            plt.figure(figsize=(10, 6))
            metrics_df['auc_roc'].plot(kind='bar', color='lightcoral')
            plt.title('AUC-ROC Comparison')
            plt.ylabel('AUC-ROC Score')
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            
            # Add value labels
            for i, v in enumerate(metrics_df['auc_roc']):
                if not pd.isna(v):
                    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
    
    def plot_feature_importance(self, model_name: str = 'random_forest', top_n: int = 15) -> None:
        """
        Plot feature importance for a specific model.
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to display
        """
        if model_name not in self.feature_importance:
            print(f"Feature importance not available for {model_name}")
            return
        
        importance_df = self.feature_importance[model_name].head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance - {model_name.replace("_", " ").title()}')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()
    
    def generate_model_report(self) -> str:
        """Generate a comprehensive model performance report."""
        if not self.performance_metrics:
            return "No models trained yet."
        
        report = "BASELINE MODEL PERFORMANCE REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Overall best model
        best_model = max(self.performance_metrics.keys(), 
                        key=lambda k: self.performance_metrics[k].get('cv_auc_mean', 0))
        
        report += f"Best performing model: {best_model.replace('_', ' ').title()}\n"
        report += f"Best CV AUC: {self.performance_metrics[best_model]['cv_auc_mean']:.3f}\n\n"
        
        # Detailed metrics for each model
        for model_name, metrics in self.performance_metrics.items():
            report += f"{model_name.replace('_', ' ').title()}:\n"
            report += f"  Accuracy: {metrics['accuracy']:.3f}\n"
            report += f"  Precision: {metrics['precision']:.3f}\n"
            report += f"  Recall: {metrics['recall']:.3f}\n"
            report += f"  F1-Score: {metrics['f1_score']:.3f}\n"
            if metrics.get('auc_roc'):
                report += f"  AUC-ROC: {metrics['auc_roc']:.3f}\n"
            report += f"  CV AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}\n\n"
        
        # Feature importance summary
        if self.feature_importance:
            report += "TOP RISK FACTORS:\n"
            report += "-" * 20 + "\n"
            
            # Get top features from best model
            if best_model in self.feature_importance:
                top_features = self.feature_importance[best_model].head(10)
                for idx, row in top_features.iterrows():
                    report += f"{row['feature']}: {row['importance']:.4f}\n"
        
        return report
    
    def save_results(self, output_dir: str = "../results") -> None:
        """Save model results and reports."""
        from pathlib import Path
        import pickle
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = output_path / f"{model_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save performance metrics
        metrics_df = pd.DataFrame(self.performance_metrics).T
        metrics_df.to_csv(output_path / "model_performance_metrics.csv")
        
        # Save feature importance
        for model_name, importance_df in self.feature_importance.items():
            importance_df.to_csv(output_path / f"{model_name}_feature_importance.csv", index=False)
        
        # Save text report
        report = self.generate_model_report()
        with open(output_path / "model_report.txt", 'w') as f:
            f.write(report)
        
        print(f"Results saved to {output_path}")


def quick_baseline_analysis(data: pd.DataFrame, 
                          feature_columns: List[str],
                          target_column: str = 'injury_risk_label') -> BaselineInjuryRiskModel:
    """
    Quick function to run complete baseline analysis.
    
    Args:
        data: Input DataFrame
        feature_columns: List of feature columns
        target_column: Target column name
        
    Returns:
        Trained BaselineInjuryRiskModel instance
    """
    model = BaselineInjuryRiskModel()
    
    # Prepare data
    X, y = model.prepare_features_and_target(data, feature_columns, target_column)
    
    # Train models
    results = model.train_baseline_models(X, y)
    
    # Generate visualizations
    model.plot_model_comparison()
    model.plot_feature_importance()
    
    # Print report
    print("\n" + model.generate_model_report())
    
    return model
