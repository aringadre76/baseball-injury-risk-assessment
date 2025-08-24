"""
Data loading and initial processing for OpenBiomechanics baseball pitching data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional


class OpenBiomechanicsLoader:
    """Loader for OpenBiomechanics baseball pitching dataset."""
    
    def __init__(self, data_root: str = "./openbiomechanics/baseball_pitching/data"):
        """
        Initialize the data loader.
        
        Args:
            data_root: Path to the root data directory
        """
        self.data_root = Path(data_root)
        self.metadata_path = self.data_root / "metadata.csv"
        self.poi_path = self.data_root / "poi" / "poi_metrics.csv"
        
    def load_metadata(self) -> pd.DataFrame:
        """Load player and session metadata."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        metadata = pd.read_csv(self.metadata_path)
        print(f"Loaded metadata: {metadata.shape[0]} records, {metadata.shape[1]} columns")
        return metadata
    
    def load_poi_metrics(self) -> pd.DataFrame:
        """Load point-of-interest biomechanical metrics."""
        if not self.poi_path.exists():
            raise FileNotFoundError(f"POI file not found: {self.poi_path}")
        
        poi_data = pd.read_csv(self.poi_path)
        print(f"Loaded POI metrics: {poi_data.shape[0]} records, {poi_data.shape[1]} columns")
        return poi_data
    
    def load_and_merge_data(self) -> pd.DataFrame:
        """Load and merge metadata with POI metrics."""
        metadata = self.load_metadata()
        poi_data = self.load_poi_metrics()
        
        # Merge on session_pitch
        merged_data = poi_data.merge(metadata, on='session_pitch', how='left')
        
        print(f"Merged data: {merged_data.shape[0]} records, {merged_data.shape[1]} columns")
        
        # Check for missing merges
        missing_metadata = merged_data['user'].isnull().sum()
        if missing_metadata > 0:
            print(f"Warning: {missing_metadata} POI records missing metadata")
        
        return merged_data
    
    def get_injury_risk_variables(self) -> List[str]:
        """Return list of key injury risk variables from literature."""
        return [
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
    
    def get_performance_variables(self) -> List[str]:
        """Return list of performance-related variables."""
        return [
            'pitch_speed_mph',
            'stride_length',
            'arm_slot',
            'max_cog_velo_x',
            'timing_peak_torso_to_peak_pelvis_rot_velo'
        ]
    
    def get_demographic_variables(self) -> List[str]:
        """Return list of demographic/anthropometric variables."""
        return [
            'age_yrs',
            'session_height_m',
            'session_mass_kg',
            'playing_level',
            'p_throws'
        ]


def validate_data_quality(data: pd.DataFrame) -> dict:
    """
    Perform basic data quality validation.
    
    Args:
        data: DataFrame to validate
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'total_records': len(data),
        'unique_pitchers': data['user'].nunique() if 'user' in data.columns else 0,
        'unique_sessions': data['session'].nunique() if 'session' in data.columns else 0,
        'missing_data': data.isnull().sum().to_dict(),
        'data_completeness': (1 - data.isnull().mean()).to_dict(),
        'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': data.select_dtypes(include=['object']).columns.tolist()
    }
    
    # Check for obvious outliers in key variables
    if 'pitch_speed_mph' in data.columns:
        speed_stats = data['pitch_speed_mph'].describe()
        results['pitch_speed_range'] = (speed_stats['min'], speed_stats['max'])
        results['potential_speed_outliers'] = ((data['pitch_speed_mph'] < 40) | 
                                             (data['pitch_speed_mph'] > 110)).sum()
    
    if 'age_yrs' in data.columns:
        age_stats = data['age_yrs'].describe()
        results['age_range'] = (age_stats['min'], age_stats['max'])
        results['potential_age_outliers'] = ((data['age_yrs'] < 16) | 
                                           (data['age_yrs'] > 35)).sum()
    
    return results


def print_data_summary(data: pd.DataFrame, validation_results: dict) -> None:
    """Print a formatted summary of the dataset."""
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    print(f"Total Records: {validation_results['total_records']}")
    print(f"Unique Pitchers: {validation_results['unique_pitchers']}")
    print(f"Unique Sessions: {validation_results['unique_sessions']}")
    
    if 'pitch_speed_range' in validation_results:
        min_speed, max_speed = validation_results['pitch_speed_range']
        print(f"Pitch Speed Range: {min_speed:.1f} - {max_speed:.1f} mph")
    
    if 'age_range' in validation_results:
        min_age, max_age = validation_results['age_range']
        print(f"Age Range: {min_age:.1f} - {max_age:.1f} years")
    
    if 'playing_level' in data.columns:
        print(f"\nPlaying Level Distribution:")
        for level, count in data['playing_level'].value_counts().items():
            print(f"  {level}: {count}")
    
    # Show columns with most missing data
    missing_data = pd.Series(validation_results['missing_data'])
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) > 0:
        print(f"\nColumns with Missing Data (top 10):")
        for col, missing_count in missing_data.head(10).items():
            pct_missing = (missing_count / validation_results['total_records']) * 100
            print(f"  {col}: {missing_count} ({pct_missing:.1f}%)")
    else:
        print(f"\nNo missing data detected!")
    
    print("="*60)
