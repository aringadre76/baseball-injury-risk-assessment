"""
Feature selection and dimensionality reduction for enhanced biomechanical features.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, RFE,
    SelectFromModel, VarianceThreshold
)
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple, Optional, Union
import warnings
from .temporal_feature_extractor import batch_extract_temporal_features
from .injury_risk_scorer import batch_analyze_pitcher_risks
from .openbiomechanics_loader import OpenBiomechanicsLoader
from .biomechanical_feature_engineer import FeatureEngineer


class EnhancedFeatureSelector:
    """Enhanced feature selection for biomechanical injury risk assessment."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the enhanced feature selector.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.selectors = {}
        self.feature_importance = {}
        self.selected_features = {}
        
    def remove_low_variance_features(self, X: pd.DataFrame, 
                                   threshold: float = 0.01) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove features with low variance.
        
        Args:
            X: Feature matrix
            threshold: Variance threshold
            
        Returns:
            Tuple of (filtered_features, removed_features)
        """
        variance_selector = VarianceThreshold(threshold=threshold)
        
        # Fit on scaled data to ensure fair comparison
        X_scaled = self.scaler.fit_transform(X)
        variance_mask = variance_selector.fit(X_scaled).get_support()
        
        selected_features = X.columns[variance_mask].tolist()
        removed_features = X.columns[~variance_mask].tolist()
        
        print(f"Variance threshold selection: kept {len(selected_features)}/{len(X.columns)} features")
        print(f"Removed {len(removed_features)} low-variance features")
        
        return X[selected_features], removed_features
    
    def correlation_based_selection(self, X: pd.DataFrame, 
                                  correlation_threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove highly correlated features.
        
        Args:
            X: Feature matrix
            correlation_threshold: Correlation threshold for removal
            
        Returns:
            Tuple of (filtered_features, removed_features)
        """
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find pairs of highly correlated features
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Identify features to remove
        to_remove = [column for column in upper_triangle.columns 
                    if any(upper_triangle[column] > correlation_threshold)]
        
        selected_features = [col for col in X.columns if col not in to_remove]
        
        print(f"Correlation-based selection: kept {len(selected_features)}/{len(X.columns)} features")
        print(f"Removed {len(to_remove)} highly correlated features")
        
        return X[selected_features], to_remove
    
    def univariate_feature_selection(self, X: pd.DataFrame, y: pd.Series,
                                   k: int = 50, score_func=f_classif) -> Tuple[pd.DataFrame, Dict]:
        """
        Univariate feature selection using statistical tests.
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            score_func: Scoring function
            
        Returns:
            Tuple of (selected_features, selection_info)
        """
        selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        
        # Scale features for fair comparison
        X_scaled = self.scaler.fit_transform(X)
        selector.fit(X_scaled, y)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Get feature scores
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_,
            'selected': selected_mask
        }).sort_values('score', ascending=False)
        
        selection_info = {
            'selector': selector,
            'feature_scores': feature_scores,
            'selected_features': selected_features
        }
        
        print(f"Univariate selection: selected {len(selected_features)} features")
        
        return X[selected_features], selection_info
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series,
                                    n_features: int = 30,
                                    estimator=None) -> Tuple[pd.DataFrame, Dict]:
        """
        Recursive feature elimination with cross-validation.
        
        Args:
            X: Feature matrix  
            y: Target variable
            n_features: Number of features to select
            estimator: Base estimator (RandomForest if None)
            
        Returns:
            Tuple of (selected_features, selection_info)
        """
        if estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, max_depth=10
            )
        
        rfe = RFE(estimator=estimator, n_features_to_select=min(n_features, X.shape[1]))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        rfe.fit(X_scaled, y)
        
        # Get selected features
        selected_mask = rfe.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Get feature rankings
        feature_rankings = pd.DataFrame({
            'feature': X.columns,
            'ranking': rfe.ranking_,
            'selected': selected_mask
        }).sort_values('ranking')
        
        selection_info = {
            'selector': rfe,
            'feature_rankings': feature_rankings,
            'selected_features': selected_features
        }
        
        print(f"RFE selection: selected {len(selected_features)} features")
        
        return X[selected_features], selection_info
    
    def lasso_feature_selection(self, X: pd.DataFrame, y: pd.Series,
                               alpha: Optional[float] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Feature selection using Lasso regularization.
        
        Args:
            X: Feature matrix
            y: Target variable  
            alpha: Regularization parameter (auto-selected if None)
            
        Returns:
            Tuple of (selected_features, selection_info)
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if alpha is None:
            # Use cross-validation to select alpha
            lasso = LassoCV(cv=5, random_state=self.random_state, max_iter=2000)
        else:
            from sklearn.linear_model import Lasso
            lasso = Lasso(alpha=alpha, random_state=self.random_state, max_iter=2000)
        
        lasso.fit(X_scaled, y)
        
        # Get non-zero coefficients
        non_zero_mask = lasso.coef_ != 0
        selected_features = X.columns[non_zero_mask].tolist()
        
        # Feature importance based on absolute coefficients
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'coefficient': lasso.coef_,
            'abs_coefficient': np.abs(lasso.coef_),
            'selected': non_zero_mask
        }).sort_values('abs_coefficient', ascending=False)
        
        selection_info = {
            'selector': lasso,
            'feature_importance': feature_importance,
            'selected_features': selected_features,
            'alpha_used': lasso.alpha_ if hasattr(lasso, 'alpha_') else alpha
        }
        
        print(f"Lasso selection: selected {len(selected_features)} features")
        if hasattr(lasso, 'alpha_'):
            print(f"Optimal alpha: {lasso.alpha_:.6f}")
        
        return X[selected_features], selection_info
    
    def random_forest_feature_selection(self, X: pd.DataFrame, y: pd.Series,
                                      n_features: int = 40) -> Tuple[pd.DataFrame, Dict]:
        """
        Feature selection using Random Forest feature importance.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_features: Number of features to select
            
        Returns:
            Tuple of (selected_features, selection_info)
        """
        rf = RandomForestClassifier(
            n_estimators=200, random_state=self.random_state, 
            max_depth=15, min_samples_split=10
        )
        
        # Scale features  
        X_scaled = self.scaler.fit_transform(X)
        rf.fit(X_scaled, y)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top features
        top_features = feature_importance.head(min(n_features, len(feature_importance)))
        selected_features = top_features['feature'].tolist()
        
        selection_info = {
            'selector': rf,
            'feature_importance': feature_importance,
            'selected_features': selected_features
        }
        
        print(f"Random Forest selection: selected {len(selected_features)} features")
        
        return X[selected_features], selection_info
    
    def mutual_information_selection(self, X: pd.DataFrame, y: pd.Series,
                                   k: int = 35) -> Tuple[pd.DataFrame, Dict]:
        """
        Feature selection using mutual information.
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            
        Returns:
            Tuple of (selected_features, selection_info)
        """
        # Handle any infinite or extremely large values
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        selector.fit(X_scaled, y)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Get mutual information scores
        mi_scores = pd.DataFrame({
            'feature': X.columns,
            'mi_score': selector.scores_,
            'selected': selected_mask
        }).sort_values('mi_score', ascending=False)
        
        selection_info = {
            'selector': selector,
            'mi_scores': mi_scores,
            'selected_features': selected_features
        }
        
        print(f"Mutual information selection: selected {len(selected_features)} features")
        
        return X[selected_features], selection_info
    
    def ensemble_feature_selection(self, X: pd.DataFrame, y: pd.Series,
                                 n_final_features: int = 50) -> Tuple[pd.DataFrame, Dict]:
        """
        Ensemble feature selection combining multiple methods.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_final_features: Final number of features to select
            
        Returns:
            Tuple of (selected_features, selection_info)
        """
        print("Performing ensemble feature selection...")
        
        # 1. Remove low variance features
        X_var, removed_var = self.remove_low_variance_features(X, threshold=0.001)
        
        # 2. Remove highly correlated features
        X_corr, removed_corr = self.correlation_based_selection(X_var, correlation_threshold=0.95)
        
        # 3. Apply multiple selection methods
        methods_results = {}
        
        try:
            # Univariate selection
            _, univariate_info = self.univariate_feature_selection(
                X_corr, y, k=min(60, X_corr.shape[1])
            )
            methods_results['univariate'] = set(univariate_info['selected_features'])
        except Exception as e:
            print(f"Univariate selection failed: {e}")
        
        try:
            # Random Forest selection
            _, rf_info = self.random_forest_feature_selection(
                X_corr, y, n_features=min(50, X_corr.shape[1])
            )
            methods_results['random_forest'] = set(rf_info['selected_features'])
        except Exception as e:
            print(f"Random Forest selection failed: {e}")
        
        try:
            # Lasso selection
            _, lasso_info = self.lasso_feature_selection(X_corr, y)
            methods_results['lasso'] = set(lasso_info['selected_features'])
        except Exception as e:
            print(f"Lasso selection failed: {e}")
        
        try:
            # Mutual information selection
            _, mi_info = self.mutual_information_selection(
                X_corr, y, k=min(45, X_corr.shape[1])
            )
            methods_results['mutual_info'] = set(mi_info['selected_features'])
        except Exception as e:
            print(f"Mutual information selection failed: {e}")
        
        # 4. Combine results using voting
        if methods_results:
            all_features = set()
            for feature_set in methods_results.values():
                all_features.update(feature_set)
            
            # Count votes for each feature
            feature_votes = {}
            for feature in all_features:
                votes = sum(1 for feature_set in methods_results.values() if feature in feature_set)
                feature_votes[feature] = votes
            
            # Sort by votes and select top features
            sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
            final_features = [feature for feature, votes in sorted_features[:n_final_features]]
        else:
            # Fallback: use correlation-filtered features
            final_features = X_corr.columns.tolist()[:n_final_features]
        
        selection_info = {
            'removed_low_variance': removed_var,
            'removed_correlated': removed_corr,
            'method_results': methods_results,
            'feature_votes': feature_votes if 'feature_votes' in locals() else {},
            'final_features': final_features
        }
        
        print(f"Ensemble selection: selected {len(final_features)} final features")
        
        return X[final_features], selection_info
    
    def dimensionality_reduction_pca(self, X: pd.DataFrame, 
                                   n_components: Optional[int] = None,
                                   variance_explained: float = 0.95) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            X: Feature matrix
            n_components: Number of components (auto-selected if None)
            variance_explained: Target cumulative variance to explain
            
        Returns:
            Tuple of (transformed_features, pca_info)
        """
        # Scale features first
        X_scaled = self.scaler.fit_transform(X)
        
        if n_components is None:
            # Find number of components for target variance
            pca_temp = PCA()
            pca_temp.fit(X_scaled)
            
            cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum_var >= variance_explained) + 1
            n_components = min(n_components, X.shape[1])
        
        # Apply PCA with selected number of components
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create DataFrame with component names
        component_names = [f'PC{i+1}' for i in range(n_components)]
        X_pca_df = pd.DataFrame(X_pca, columns=component_names, index=X.index)
        
        # Feature loadings (contribution of original features to components)
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=component_names,
            index=X.columns
        )
        
        pca_info = {
            'pca': pca,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'loadings': loadings,
            'n_components': n_components
        }
        
        total_variance = pca.explained_variance_ratio_.sum()
        print(f"PCA: reduced to {n_components} components explaining {total_variance:.3f} of variance")
        
        return X_pca_df, pca_info
    
    def evaluate_feature_set(self, X: pd.DataFrame, y: pd.Series,
                           estimator=None) -> Dict[str, float]:
        """
        Evaluate feature set using cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            estimator: Estimator to use for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        if estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, max_depth=10
            )
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        cv_scores = cross_val_score(estimator, X_scaled, y, cv=cv, scoring='roc_auc')
        
        evaluation = {
            'mean_auc': cv_scores.mean(),
            'std_auc': cv_scores.std(),
            'n_features': X.shape[1],
            'feature_names': X.columns.tolist()
        }
        
        return evaluation


def create_enhanced_feature_dataset(max_pitches: int = 100) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create enhanced feature dataset combining POI, temporal, and risk features.
    
    Args:
        max_pitches: Maximum number of pitches to process
        
    Returns:
        Tuple of (features_df, target_series)
    """
    print("Creating enhanced feature dataset...")
    
    # 1. Load base POI data and create risk labels
    poi_loader = OpenBiomechanicsLoader()
    base_data = poi_loader.load_and_merge_data()
    
    # Create risk labels using existing method
    feature_engineer = FeatureEngineer()
    base_data = feature_engineer.create_risk_labels(base_data, method='composite_score')
    
    # Get available pitches
    available_pitches = base_data['session_pitch'].unique()[:max_pitches]
    
    print(f"Processing {len(available_pitches)} pitches...")
    
    # 2. Extract temporal features
    temporal_df = batch_extract_temporal_features(available_pitches, max_pitches=max_pitches)
    
    if temporal_df.empty:
        raise ValueError("Failed to extract temporal features")
    
    # 3. Create advanced risk scores
    risk_df = batch_analyze_pitcher_risks(available_pitches, max_pitches=max_pitches)
    
    # 4. Merge all features
    # Start with base POI data
    merged_data = base_data[base_data['session_pitch'].isin(available_pitches)].copy()
    
    # Add temporal features
    merged_data = merged_data.merge(temporal_df, on='session_pitch', how='inner')
    
    # Add risk scores
    if not risk_df.empty:
        risk_cols = ['session_pitch', 'elbow_stress_score', 'shoulder_stress_score', 
                    'kinetic_chain_efficiency', 'movement_quality']
        risk_subset = risk_df[risk_cols]
        merged_data = merged_data.merge(risk_subset, on='session_pitch', how='left')
    
    # 5. Prepare features and target
    # Exclude non-feature columns
    exclude_cols = ['session_pitch', 'session', 'user', 'injury_risk_label']
    feature_cols = [col for col in merged_data.columns 
                   if col not in exclude_cols and pd.api.types.is_numeric_dtype(merged_data[col])]
    
    features_df = merged_data[feature_cols].copy()
    target_series = merged_data['injury_risk_label'].copy()
    
    # Handle missing values
    features_df = features_df.fillna(features_df.median())
    
    print(f"Enhanced dataset created: {features_df.shape[0]} samples, {features_df.shape[1]} features")
    
    return features_df, target_series


def comprehensive_feature_selection_pipeline(max_pitches: int = 100,
                                            n_final_features: int = 50) -> Dict:
    """
    Run comprehensive feature selection pipeline.
    
    Args:
        max_pitches: Maximum number of pitches to process
        n_final_features: Final number of features to select
        
    Returns:
        Dictionary with all selection results
    """
    print("="*60)
    print("COMPREHENSIVE FEATURE SELECTION PIPELINE")
    print("="*60)
    
    # 1. Create enhanced dataset
    X, y = create_enhanced_feature_dataset(max_pitches=max_pitches)
    
    print(f"Initial dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # 2. Initialize feature selector
    selector = EnhancedFeatureSelector()
    
    # 3. Baseline evaluation
    print(f"\nEvaluating baseline (all features)...")
    baseline_eval = selector.evaluate_feature_set(X, y)
    print(f"Baseline AUC: {baseline_eval['mean_auc']:.3f} ± {baseline_eval['std_auc']:.3f}")
    
    # 4. Ensemble feature selection
    X_selected, selection_info = selector.ensemble_feature_selection(
        X, y, n_final_features=n_final_features
    )
    
    # 5. Evaluate selected features
    print(f"\nEvaluating selected features...")
    selected_eval = selector.evaluate_feature_set(X_selected, y)
    print(f"Selected features AUC: {selected_eval['mean_auc']:.3f} ± {selected_eval['std_auc']:.3f}")
    
    # 6. PCA analysis
    print(f"\nApplying PCA...")
    X_pca, pca_info = selector.dimensionality_reduction_pca(X_selected, variance_explained=0.90)
    
    pca_eval = selector.evaluate_feature_set(X_pca, y)
    print(f"PCA features AUC: {pca_eval['mean_auc']:.3f} ± {pca_eval['std_auc']:.3f}")
    
    # 7. Compile results
    results = {
        'original_features': X,
        'selected_features': X_selected,
        'pca_features': X_pca,
        'target': y,
        'baseline_evaluation': baseline_eval,
        'selected_evaluation': selected_eval,
        'pca_evaluation': pca_eval,
        'selection_info': selection_info,
        'pca_info': pca_info,
        'selector': selector
    }
    
    print(f"\n" + "="*60)
    print("FEATURE SELECTION PIPELINE COMPLETE")
    print("="*60)
    
    return results
