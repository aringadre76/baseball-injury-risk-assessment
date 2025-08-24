"""
Robust Feature Engineering for Injury Risk Assessment
Ensures consistent features between training and inference
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import List, Tuple, Dict, Optional
import pickle
from pathlib import Path


class RobustFeatureEngineer:
    """Robust feature engineering that ensures consistency between training and inference."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_columns = None
        self.numeric_columns = None
        self.categorical_columns = None
        self.is_fitted = False
        
    def fit(self, data: pd.DataFrame) -> 'RobustFeatureEngineer':
        """
        Fit the feature engineer on training data to ensure consistency.
        
        Args:
            data: Training DataFrame
            
        Returns:
            Self for chaining
        """
        print("Fitting RobustFeatureEngineer...")
        
        # Store the exact feature columns used during training
        self.feature_columns = self._get_consistent_features(data)
        print(f"Identified {len(self.feature_columns)} consistent features")
        
        # Separate numeric and categorical columns
        available_data = data[self.feature_columns]
        self.numeric_columns = available_data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = available_data.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numeric columns: {len(self.numeric_columns)}")
        print(f"Categorical columns: {len(self.categorical_columns)}")
        
        # Fit imputers for numeric features only
        for col in self.numeric_columns:
            if col in data.columns:
                imputer = SimpleImputer(strategy='median')
                imputer.fit(data[[col]])
                self.imputers[col] = imputer
        
        # Fit encoders for categorical variables
        for col in self.categorical_columns:
            if col in data.columns:
                encoder = LabelEncoder()
                # Handle missing values before fitting encoder
                clean_data = data[col].fillna('Unknown')
                encoder.fit(clean_data)
                self.encoders[col] = encoder
        
        # Fit scaler for numeric features only
        if len(self.numeric_columns) > 0:
            self.scaler = StandardScaler()
            # Use imputed data for fitting scaler
            imputed_data = self._impute_features(data[self.numeric_columns])
            self.scaler.fit(imputed_data)
        
        self.is_fitted = True
        print("✓ RobustFeatureEngineer fitted successfully")
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted feature engineering.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed DataFrame with consistent features
        """
        if not self.is_fitted:
            raise ValueError("RobustFeatureEngineer must be fitted before transform")
        
        print("Transforming data with consistent features...")
        
        # Ensure all required features exist
        missing_features = [col for col in self.feature_columns if col not in data.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Add missing columns with default values
            for col in missing_features:
                if col in self.numeric_columns:
                    data[col] = 0.0
                else:
                    data[col] = 'Unknown'
        
        # Select only the features used during training
        data_subset = data[self.feature_columns].copy()
        
        # Create engineered features if they don't exist
        data_with_engineered = self.create_engineered_features(data_subset)
        
        # Impute missing values (numeric only)
        data_imputed = self._impute_features(data_with_engineered)
        
        # Encode categorical variables
        data_encoded = self._encode_features(data_imputed)
        
        # Scale numeric features
        if hasattr(self, 'scaler') and len(self.numeric_columns) > 0:
            data_scaled = self._scale_features(data_encoded)
        else:
            data_scaled = data_encoded
        
        print(f"✓ Data transformed: {data_scaled.shape}")
        return data_scaled
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)
    
    def _get_consistent_features(self, data: pd.DataFrame) -> List[str]:
        """Get a consistent set of features that will always be available."""
        # Core biomechanical features (always present)
        core_features = [
            'elbow_varus_moment',
            'shoulder_internal_rotation_moment',
            'max_shoulder_internal_rotational_velo',
            'max_elbow_extension_velo',
            'max_torso_rotational_velo',
            'max_rotation_hip_shoulder_separation',
            'lead_knee_extension_angular_velo_fp',
            'torso_anterior_tilt_fp',
            'torso_lateral_tilt_fp',
            'pelvis_anterior_tilt_fp',
            'pelvis_lateral_tilt_fp'
        ]
        
        # Performance features
        performance_features = [
            'pitch_speed_mph',
            'stride_length',
            'arm_slot',
            'max_cog_velo_x'
        ]
        
        # Demographic features
        demographic_features = [
            'age_yrs',
            'session_height_m',
            'session_mass_kg',
            'playing_level'
        ]
        
        # Additional biomechanical features
        additional_features = [
            'elbow_flexion_fp',
            'elbow_pronation_fp',
            'shoulder_abduction_fp',
            'shoulder_external_rotation_fp',
            'pelvis_rotation_fp',
            'max_pelvis_rotational_velo',
            'glove_shoulder_abduction_fp',
            'glove_shoulder_external_rotation_fp'
        ]
        
        # Combine all features
        all_features = core_features + performance_features + demographic_features + additional_features
        
        # Filter to only features that exist in the data
        available_features = [col for col in all_features if col in data.columns]
        
        # Return only available features (engineered features will be added later)
        return available_features
    
    def _impute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using fitted imputers (numeric only)."""
        data_imputed = data.copy()
        
        # Only impute numeric columns
        for col in self.numeric_columns:
            if col in data.columns and col in self.imputers:
                try:
                    # Ensure the column is numeric before imputation
                    numeric_data = pd.to_numeric(data[col], errors='coerce')
                    if numeric_data.notna().any():  # Only impute if we have some numeric data
                        data_imputed[col] = self.imputers[col].transform(numeric_data.values.reshape(-1, 1)).flatten()
                    else:
                        # If no numeric data, use default value
                        data_imputed[col] = 0.0
                except Exception:
                    # If imputation fails, use default value
                    data_imputed[col] = 0.0
            elif col in data.columns:
                # Use median for numeric columns without specific imputer
                try:
                    numeric_data = pd.to_numeric(data[col], errors='coerce')
                    if numeric_data.notna().any():
                        data_imputed[col] = numeric_data.fillna(numeric_data.median())
                    else:
                        data_imputed[col] = 0.0
                except Exception:
                    data_imputed[col] = 0.0
        
        # For categorical columns, fill with 'Unknown'
        for col in self.categorical_columns:
            if col in data.columns:
                data_imputed[col] = data[col].fillna('Unknown')
        
        return data_imputed
    
    def _encode_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using fitted encoders."""
        data_encoded = data.copy()
        
        for col in self.categorical_columns:
            if col in data.columns and col in self.encoders:
                # Handle missing values before encoding
                data_encoded[col] = data_encoded[col].fillna('Unknown')
                # Handle unseen categories by adding them to encoder
                try:
                    data_encoded[col] = self.encoders[col].transform(data_encoded[col])
                except ValueError:
                    # If we encounter new categories, encode them as -1
                    data_encoded[col] = data_encoded[col].map(
                        lambda x: self.encoders[col].transform([x])[0] 
                        if x in self.encoders[col].classes_ else -1
                    )
        
        return data_encoded
    
    def _scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric features using fitted scaler."""
        if not hasattr(self, 'scaler') or len(self.numeric_columns) == 0:
            return data
        
        data_scaled = data.copy()
        
        # Only scale numeric columns that exist in the data
        available_numeric = [col for col in self.numeric_columns if col in data.columns]
        if len(available_numeric) > 0:
            data_scaled[available_numeric] = self.scaler.transform(data[available_numeric])
        
        return data_scaled
    
    def create_engineered_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features consistently."""
        df = data.copy()
        
        # Elbow stress composite score
        if all(col in df.columns for col in ['elbow_varus_moment', 'max_elbow_extension_velo']):
            try:
                # Ensure numeric data
                elbow_moment = pd.to_numeric(df['elbow_varus_moment'], errors='coerce').fillna(0)
                elbow_velo = pd.to_numeric(df['max_elbow_extension_velo'], errors='coerce').fillna(0)
                df['elbow_stress_composite'] = (elbow_moment * elbow_velo / 1000)
            except Exception:
                df['elbow_stress_composite'] = 0.0
        else:
            df['elbow_stress_composite'] = 0.0
        
        # Shoulder stress composite score
        if all(col in df.columns for col in ['shoulder_internal_rotation_moment', 
                                           'max_shoulder_internal_rotational_velo']):
            try:
                # Ensure numeric data
                shoulder_moment = pd.to_numeric(df['shoulder_internal_rotation_moment'], errors='coerce').fillna(0)
                shoulder_velo = pd.to_numeric(df['max_shoulder_internal_rotational_velo'], errors='coerce').fillna(0)
                df['shoulder_stress_composite'] = (shoulder_moment * shoulder_velo / 1000)
            except Exception:
                df['shoulder_stress_composite'] = 0.0
        else:
            df['shoulder_stress_composite'] = 0.0
        
        # Kinetic chain efficiency
        if all(col in df.columns for col in ['max_pelvis_rotational_velo', 
                                           'max_torso_rotational_velo']):
            try:
                # Ensure numeric data
                pelvis_velo = pd.to_numeric(df['max_pelvis_rotational_velo'], errors='coerce').fillna(0)
                torso_velo = pd.to_numeric(df['max_torso_rotational_velo'], errors='coerce').fillna(0)
                df['kinetic_chain_ratio'] = (torso_velo / (pelvis_velo + 1e-6))
            except Exception:
                df['kinetic_chain_ratio'] = 1.0
        else:
            df['kinetic_chain_ratio'] = 1.0
        
        # Hip-shoulder separation efficiency
        if 'max_rotation_hip_shoulder_separation' in df.columns:
            try:
                # Ensure numeric data
                separation = pd.to_numeric(df['max_rotation_hip_shoulder_separation'], errors='coerce').fillna(0)
                df['hip_shoulder_separation_risk'] = np.where(
                    separation < 20, 1,
                    np.where(separation > 50, 1, 0)
                )
            except Exception:
                df['hip_shoulder_separation_risk'] = 0
        else:
            df['hip_shoulder_separation_risk'] = 0
        
        # Postural stability score
        postural_cols = ['torso_anterior_tilt_fp', 'torso_lateral_tilt_fp', 
                        'pelvis_anterior_tilt_fp', 'pelvis_lateral_tilt_fp']
        if all(col in df.columns for col in postural_cols):
            try:
                # Ensure numeric data
                postural_data = df[postural_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
                df['postural_instability'] = np.sqrt(
                    postural_data['torso_anterior_tilt_fp']**2 + postural_data['torso_lateral_tilt_fp']**2 +
                    postural_data['pelvis_anterior_tilt_fp']**2 + postural_data['pelvis_lateral_tilt_fp']**2
                )
            except Exception:
                df['postural_instability'] = 0.0
        else:
            df['postural_instability'] = 0.0
        
        # Velocity-based risk indicators
        velocity_cols = ['max_shoulder_internal_rotational_velo', 'max_elbow_extension_velo']
        if all(col in df.columns for col in velocity_cols):
            try:
                # Ensure numeric data
                shoulder_velo = pd.to_numeric(df['max_shoulder_internal_rotational_velo'], errors='coerce').fillna(0)
                elbow_velo = pd.to_numeric(df['max_elbow_extension_velo'], errors='coerce').fillna(0)
                df['high_velocity_risk'] = (
                    (shoulder_velo > shoulder_velo.quantile(0.75)) &
                    (elbow_velo > elbow_velo.quantile(0.75))
                ).astype(int)
            except Exception:
                df['high_velocity_risk'] = 0
        else:
            df['high_velocity_risk'] = 0
        
        # Efficiency metrics
        if all(col in df.columns for col in ['pitch_speed_mph', 'max_shoulder_internal_rotational_velo']):
            try:
                # Ensure numeric data
                pitch_speed = pd.to_numeric(df['pitch_speed_mph'], errors='coerce').fillna(0)
                shoulder_velo = pd.to_numeric(df['max_shoulder_internal_rotational_velo'], errors='coerce').fillna(0)
                df['velocity_efficiency'] = (pitch_speed / (shoulder_velo / 100))
            except Exception:
                df['velocity_efficiency'] = 1.0
        else:
            df['velocity_efficiency'] = 1.0
        
        # Anthropometric ratios
        if all(col in df.columns for col in ['session_height_m', 'session_mass_kg']):
            try:
                # Ensure numeric data
                height = pd.to_numeric(df['session_height_m'], errors='coerce').fillna(1.75)
                mass = pd.to_numeric(df['session_mass_kg'], errors='coerce').fillna(70)
                df['bmi'] = mass / (height ** 2)
            except Exception:
                df['bmi'] = 25.0  # Default BMI
        else:
            df['bmi'] = 25.0  # Default BMI
        
        if 'stride_length' in df.columns and 'session_height_m' in df.columns:
            try:
                # Ensure numeric data
                stride = pd.to_numeric(df['stride_length'], errors='coerce').fillna(0.5)
                height = pd.to_numeric(df['session_height_m'], errors='coerce').fillna(1.75)
                df['stride_length_normalized'] = stride / height
            except Exception:
                df['stride_length_normalized'] = 0.5  # Default ratio
        else:
            df['stride_length_normalized'] = 0.5  # Default ratio
        
        return df
    
    def save(self, filepath: str) -> None:
        """Save the fitted feature engineer."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"RobustFeatureEngineer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'RobustFeatureEngineer':
        """Load a fitted feature engineer."""
        with open(filepath, 'rb') as f:
            engineer = pickle.load(f)
        print(f"RobustFeatureEngineer loaded from {filepath}")
        return engineer
