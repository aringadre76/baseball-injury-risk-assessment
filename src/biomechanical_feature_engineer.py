"""
Feature engineering and preprocessing for injury risk assessment.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import List, Tuple, Dict, Optional


class FeatureEngineer:
    """Feature engineering for biomechanical data."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
    def create_injury_risk_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite injury risk features.
        
        Args:
            data: Input DataFrame with biomechanical metrics
            
        Returns:
            DataFrame with additional risk features
        """
        df = data.copy()
        
        # Elbow stress composite score
        if all(col in df.columns for col in ['elbow_varus_moment', 'max_elbow_extension_velo']):
            df['elbow_stress_composite'] = (
                df['elbow_varus_moment'] * df['max_elbow_extension_velo'] / 1000
            )
        
        # Shoulder stress composite score
        if all(col in df.columns for col in ['shoulder_internal_rotation_moment', 
                                           'max_shoulder_internal_rotational_velo']):
            df['shoulder_stress_composite'] = (
                df['shoulder_internal_rotation_moment'] * 
                df['max_shoulder_internal_rotational_velo'] / 1000
            )
        
        # Kinetic chain efficiency
        if all(col in df.columns for col in ['max_pelvis_rotational_velo', 
                                           'max_torso_rotational_velo']):
            df['kinetic_chain_ratio'] = (
                df['max_torso_rotational_velo'] / 
                (df['max_pelvis_rotational_velo'] + 1e-6)  # Avoid division by zero
            )
        
        # Hip-shoulder separation efficiency
        if 'max_rotation_hip_shoulder_separation' in df.columns:
            df['hip_shoulder_separation_risk'] = np.where(
                df['max_rotation_hip_shoulder_separation'] < 20, 1,  # High risk
                np.where(df['max_rotation_hip_shoulder_separation'] > 50, 1, 0)  # Also high risk
            )
        
        # Postural stability score
        postural_cols = ['torso_anterior_tilt_fp', 'torso_lateral_tilt_fp', 
                        'pelvis_anterior_tilt_fp', 'pelvis_lateral_tilt_fp']
        if all(col in df.columns for col in postural_cols):
            df['postural_instability'] = np.sqrt(
                df['torso_anterior_tilt_fp']**2 + df['torso_lateral_tilt_fp']**2 +
                df['pelvis_anterior_tilt_fp']**2 + df['pelvis_lateral_tilt_fp']**2
            )
        
        # Velocity-based risk indicators
        velocity_cols = ['max_shoulder_internal_rotational_velo', 'max_elbow_extension_velo']
        if all(col in df.columns for col in velocity_cols):
            # High velocity combinations may indicate injury risk
            df['high_velocity_risk'] = (
                (df['max_shoulder_internal_rotational_velo'] > df['max_shoulder_internal_rotational_velo'].quantile(0.75)) &
                (df['max_elbow_extension_velo'] > df['max_elbow_extension_velo'].quantile(0.75))
            ).astype(int)
        
        return df
    
    def create_performance_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create performance-related features.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with additional performance features
        """
        df = data.copy()
        
        # Efficiency metrics
        if all(col in df.columns for col in ['pitch_speed_mph', 'max_shoulder_internal_rotational_velo']):
            df['velocity_efficiency'] = (
                df['pitch_speed_mph'] / (df['max_shoulder_internal_rotational_velo'] / 100)
            )
        
        # Anthropometric ratios
        if all(col in df.columns for col in ['session_height_m', 'session_mass_kg']):
            df['bmi'] = df['session_mass_kg'] / (df['session_height_m'] ** 2)
        
        if 'stride_length' in df.columns and 'session_height_m' in df.columns:
            df['stride_length_normalized'] = df['stride_length'] / df['session_height_m']
        
        # Speed percentiles by level
        if all(col in df.columns for col in ['pitch_speed_mph', 'playing_level']):
            df['speed_percentile_by_level'] = df.groupby('playing_level')['pitch_speed_mph'].rank(pct=True)
        
        return df
    
    def handle_missing_values(self, data: pd.DataFrame, 
                            strategy: str = 'median',
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data: Input DataFrame
            strategy: Imputation strategy ('mean', 'median', 'most_frequent')
            columns: Specific columns to impute (if None, impute all numeric)
            
        Returns:
            DataFrame with imputed values
        """
        df = data.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in df.columns and df[col].isnull().any():
                if col not in self.imputers:
                    self.imputers[col] = SimpleImputer(strategy=strategy)
                    df[[col]] = self.imputers[col].fit_transform(df[[col]])
                else:
                    df[[col]] = self.imputers[col].transform(df[[col]])
        
        return df
    
    def encode_categorical_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df = data.copy()
        
        categorical_cols = ['playing_level', 'p_throws', 'pitch_type']
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str))
        
        return df
    
    def scale_features(self, data: pd.DataFrame, 
                      columns: Optional[List[str]] = None,
                      method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            data: Input DataFrame
            columns: Columns to scale (if None, scale all numeric)
            method: Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            DataFrame with scaled features
        """
        df = data.copy()
        
        if columns is None:
            # Scale only the biomechanical features, not IDs or basic info
            exclude_cols = ['user', 'session', 'session_pitch', 'age_yrs', 
                          'session_height_m', 'session_mass_kg']
            columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                      if col not in exclude_cols]
        
        if method == 'standard':
            scaler_class = StandardScaler
        else:
            raise ValueError(f"Scaling method '{method}' not implemented")
        
        scaler_key = f"{method}_scaler"
        if scaler_key not in self.scalers:
            self.scalers[scaler_key] = scaler_class()
            scaled_data = self.scalers[scaler_key].fit_transform(df[columns])
        else:
            scaled_data = self.scalers[scaler_key].transform(df[columns])
        
        # Replace original columns with scaled versions
        for i, col in enumerate(columns):
            df[f'{col}_scaled'] = scaled_data[:, i]
        
        return df
    
    def create_risk_labels(self, data: pd.DataFrame, 
                          method: str = 'percentile_based') -> pd.DataFrame:
        """
        Create injury risk labels for supervised learning.
        
        Args:
            data: Input DataFrame
            method: Method for creating labels ('percentile_based', 'composite_score')
            
        Returns:
            DataFrame with risk labels
        """
        df = data.copy()
        
        if method == 'percentile_based':
            # Create risk labels based on high values of key injury risk variables
            risk_vars = ['elbow_varus_moment', 'shoulder_internal_rotation_moment']
            
            risk_scores = []
            for var in risk_vars:
                if var in df.columns:
                    # High risk if in top 25th percentile for that variable
                    high_risk_threshold = df[var].quantile(0.75)
                    risk_scores.append((df[var] >= high_risk_threshold).astype(int))
            
            if risk_scores:
                # High risk if high in any key variable
                df['injury_risk_label'] = np.maximum.reduce(risk_scores)
            else:
                # Fallback: random labels for development
                np.random.seed(42)
                df['injury_risk_label'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
        
        elif method == 'composite_score':
            # Create a composite risk score
            risk_vars = ['elbow_varus_moment', 'shoulder_internal_rotation_moment',
                        'max_shoulder_internal_rotational_velo', 'max_elbow_extension_velo']
            
            available_vars = [var for var in risk_vars if var in df.columns]
            
            if available_vars:
                # Normalize each variable to 0-1 scale and sum
                composite_score = 0
                for var in available_vars:
                    var_min, var_max = df[var].min(), df[var].max()
                    if var_max > var_min:
                        normalized_var = (df[var] - var_min) / (var_max - var_min)
                        composite_score += normalized_var
                
                # High risk if composite score is in top 30%
                risk_threshold = composite_score.quantile(0.7)
                df['injury_risk_label'] = (composite_score >= risk_threshold).astype(int)
            else:
                # Fallback
                np.random.seed(42)
                df['injury_risk_label'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
        
        return df
    
    def get_feature_importance_data(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Organize features by importance category.
        
        Returns:
            Dictionary with feature categories
        """
        all_columns = data.columns.tolist()
        
        feature_categories = {
            'primary_risk_factors': [
                'elbow_varus_moment', 'shoulder_internal_rotation_moment',
                'max_shoulder_internal_rotational_velo', 'max_elbow_extension_velo'
            ],
            'secondary_risk_factors': [
                'max_torso_rotational_velo', 'max_rotation_hip_shoulder_separation',
                'lead_knee_extension_angular_velo_fp', 'torso_anterior_tilt_fp'
            ],
            'performance_metrics': [
                'pitch_speed_mph', 'stride_length', 'arm_slot', 'max_cog_velo_x'
            ],
            'demographic_factors': [
                'age_yrs', 'session_height_m', 'session_mass_kg', 'playing_level_encoded'
            ],
            'engineered_features': [
                col for col in all_columns 
                if any(suffix in col for suffix in ['_composite', '_ratio', '_risk', '_efficiency'])
            ]
        }
        
        # Filter to only include columns that actually exist in the data
        for category in feature_categories:
            feature_categories[category] = [
                col for col in feature_categories[category] if col in all_columns
            ]
        
        return feature_categories
