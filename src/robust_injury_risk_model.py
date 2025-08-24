"""
Robust Injury Risk Assessment Model
Can be properly saved, loaded, and used for predictions
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class RobustInjuryRiskModel:
    """Robust injury risk assessment model that can be saved and loaded properly."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.feature_engineer = None
        self.feature_columns = None
        self.best_model = None
        self.best_model_name = None
        self.performance_metrics = {}
        self.feature_importance = {}
        self.is_trained = False
        
    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare data for training with consistent feature engineering.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (features, target)
        """
        print("Preparing data with robust feature engineering...")
        
        # Create engineered features first
        data_with_features = self.feature_engineer.create_engineered_features(data)
        
        # Create injury risk labels
        key_vars = ['elbow_varus_moment', 'shoulder_internal_rotation_moment']
        available_vars = [var for var in key_vars if var in data_with_features.columns]
        
        if not available_vars:
            raise ValueError("No injury risk variables available for labeling")
        
        # Create composite risk score
        risk_score = np.zeros(len(data_with_features))
        for var in available_vars:
            values = data_with_features[var].fillna(0)
            if values.max() > values.min():
                normalized = (values - values.min()) / (values.max() - values.min())
                risk_score += normalized
        
        # Create binary labels (top 30% as high risk)
        threshold = np.percentile(risk_score, 70)
        injury_risk_label = (risk_score > threshold).astype(int)
        
        print(f"✓ Risk labels created: {np.bincount(injury_risk_label)}")
        
        # Select biomechanical features
        biomech_features = [col for col in data_with_features.columns if any(keyword in col.lower() 
                           for keyword in ['elbow', 'shoulder', 'torso', 'hip', 'force', 'moment', 'velo', 'pitch_speed', 'age', 'height', 'weight'])]
        
        # Remove target variables
        biomech_features = [col for col in biomech_features if col not in ['injury_risk_label', 'injury_risk_score']]
        
        print(f"✓ Features selected: {len(biomech_features)}")
        
        # Store feature columns for consistency
        self.feature_columns = biomech_features
        
        # Prepare features and target
        X = data_with_features[biomech_features].copy()
        y = injury_risk_label.copy()
        
        return X, y
    
    def train(self, data: pd.DataFrame, test_size: float = 0.3) -> Dict[str, Any]:
        """
        Train the injury risk assessment model.
        
        Args:
            data: Input DataFrame
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with training results
        """
        print("Training Robust Injury Risk Model...")
        
        # Prepare data
        X, y = self.prepare_data(data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        
        # Fit feature engineer on training data
        print("Fitting feature engineer...")
        self.feature_engineer.fit(X_train)
        
        # Transform training and test data
        X_train_processed = self.feature_engineer.transform(X_train)
        X_test_processed = self.feature_engineer.transform(X_test)
        
        # Train models
        self._train_models(X_train_processed, y_train, X_test_processed, y_test)
        
        # Evaluate on test set
        test_results = self._evaluate_test_set(X_test_processed, y_test)
        
        self.is_trained = True
        print("✓ Model training completed successfully")
        
        return {
            'training_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'features': len(self.feature_columns),
            'test_accuracy': test_results['accuracy'],
            'test_auc': test_results['auc'],
            'best_model': self.best_model_name
        }
    
    def _train_models(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                      X_test: pd.DataFrame, y_test: np.ndarray) -> None:
        """Train multiple models and select the best one."""
        
        # Initialize models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=self.random_state
            )
        }
        
        # Train each model
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring='roc_auc'
            )
            
            # Test set evaluation
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'auc': roc_auc_score(y_test, y_pred_proba),
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std()
            }
            
            # Store model and metrics
            self.models[name] = model
            self.performance_metrics[name] = metrics
            
            # Feature importance - use the processed feature columns
            if hasattr(model, 'feature_importances_'):
                processed_feature_columns = X_train.columns.tolist()
                self.feature_importance[name] = pd.DataFrame({
                    'feature': processed_feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            
            print(f"  {name}: CV AUC = {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
        
        # Select best model based on CV AUC
        self.best_model_name = max(
            self.performance_metrics.keys(),
            key=lambda x: self.performance_metrics[x]['cv_auc_mean']
        )
        self.best_model = self.models[self.best_model_name]
        
        print(f"✓ Best model: {self.best_model_name}")
    
    def _evaluate_test_set(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the best model on the test set."""
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        results['confusion_matrix'] = cm
        
        print(f"Test Set Performance:")
        print(f"  Accuracy: {results['accuracy']:.3f}")
        print(f"  Precision: {results['precision']:.3f}")
        print(f"  Recall: {results['recall']:.3f}")
        print(f"  F1-Score: {results['f1']:.3f}")
        print(f"  AUC-ROC: {results['auc']:.3f}")
        
        return results
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Array of predictions (0 = low risk, 1 = high risk)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Transform data using fitted feature engineer
        data_processed = self.feature_engineer.transform(data)
        
        # Make predictions
        predictions = self.best_model.predict(data_processed)
        
        return predictions
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Array of prediction probabilities [P(low_risk), P(high_risk)]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Transform data using fitted feature engineer
        data_processed = self.feature_engineer.transform(data)
        
        # Get probabilities
        probabilities = self.best_model.predict_proba(data_processed)
        
        return probabilities
    
    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Get feature importance from the best model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if self.best_model_name in self.feature_importance:
            return self.feature_importance[self.best_model_name].head(top_n)
        else:
            return pd.DataFrame()
    
    def save(self, filepath: str) -> None:
        """Save the complete model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        output_file = Path(filepath)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"✓ Complete model saved to {output_file}")
    
    @classmethod
    def load(cls, filepath: str) -> 'RobustInjuryRiskModel':
        """Load a saved model."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✓ Model loaded from {filepath}")
        return model
    
    def generate_report(self) -> str:
        """Generate a comprehensive model report."""
        if not self.is_trained:
            return "Model not trained yet."
        
        report = "ROBUST INJURY RISK MODEL REPORT\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Best Model: {self.best_model_name}\n"
        report += f"Feature Count: {len(self.feature_columns)}\n"
        report += f"Training Status: {'Trained' if self.is_trained else 'Not Trained'}\n\n"
        
        # Performance metrics
        if self.best_model_name in self.performance_metrics:
            metrics = self.performance_metrics[self.best_model_name]
            report += "Performance Metrics:\n"
            report += f"  CV AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}\n"
            report += f"  Test AUC: {metrics['auc']:.3f}\n"
            report += f"  Test Accuracy: {metrics['accuracy']:.3f}\n"
            report += f"  Test Precision: {metrics['precision']:.3f}\n"
            report += f"  Test Recall: {metrics['recall']:.3f}\n"
            report += f"  Test F1: {metrics['f1']:.3f}\n\n"
        
        # Feature importance
        if self.best_model_name in self.feature_importance:
            report += "Top 10 Feature Importances:\n"
            top_features = self.feature_importance[self.best_model_name].head(10)
            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                report += f"  {i:2d}. {row['feature']}: {row['importance']:.4f}\n"
        
        return report
