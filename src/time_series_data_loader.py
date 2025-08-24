"""
Time-series data loading and processing for OpenBiomechanics full signal data.
"""

import pandas as pd
import numpy as np
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import io
import warnings


class TimeSeriesLoader:
    """Loader for time-series biomechanical data from OpenBiomechanics."""
    
    def __init__(self, data_root: str = "./openbiomechanics/baseball_pitching/data/full_sig"):
        """
        Initialize the time-series data loader.
        
        Args:
            data_root: Path to the full signal data directory
        """
        self.data_root = Path(data_root)
        self.data_files = {
            'landmarks': self.data_root / 'landmarks.zip',
            'joint_angles': self.data_root / 'joint_angles.zip', 
            'joint_velos': self.data_root / 'joint_velos.zip',
            'forces_moments': self.data_root / 'forces_moments.zip',
            'force_plate': self.data_root / 'force_plate.zip',
            'energy_flow': self.data_root / 'energy_flow.zip'
        }
        
        self._cached_data = {}
        self._validate_files()
    
    def _validate_files(self) -> None:
        """Validate that all required data files exist."""
        missing_files = []
        for name, path in self.data_files.items():
            if not path.exists():
                missing_files.append(str(path))
        
        if missing_files:
            raise FileNotFoundError(f"Missing data files: {missing_files}")
    
    def _load_zip_csv(self, zip_path: Path, csv_name: str = None) -> pd.DataFrame:
        """
        Load CSV data from a ZIP file.
        
        Args:
            zip_path: Path to ZIP file
            csv_name: Name of CSV file within ZIP (auto-detected if None)
            
        Returns:
            DataFrame with the CSV data
        """
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            if csv_name is None:
                csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
                if len(csv_files) != 1:
                    raise ValueError(f"Expected 1 CSV file in {zip_path}, found {len(csv_files)}")
                csv_name = csv_files[0]
            
            with zip_file.open(csv_name) as csv_file:
                return pd.read_csv(io.TextIOWrapper(csv_file))
    
    def load_data_type(self, data_type: str, cache: bool = True) -> pd.DataFrame:
        """
        Load specific type of time-series data.
        
        Args:
            data_type: Type of data ('landmarks', 'joint_angles', 'joint_velos', 
                      'forces_moments', 'force_plate', 'energy_flow')
            cache: Whether to cache the loaded data
            
        Returns:
            DataFrame with time-series data
        """
        if data_type not in self.data_files:
            raise ValueError(f"Unknown data type: {data_type}. "
                           f"Available types: {list(self.data_files.keys())}")
        
        if cache and data_type in self._cached_data:
            return self._cached_data[data_type].copy()
        
        print(f"Loading {data_type} data...")
        zip_path = self.data_files[data_type]
        data = self._load_zip_csv(zip_path)
        
        print(f"Loaded {data_type}: {data.shape[0]} records, {data.shape[1]} columns")
        
        if cache:
            self._cached_data[data_type] = data.copy()
        
        return data
    
    def get_pitch_data(self, session_pitch: str, data_types: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get all time-series data for a specific pitch.
        
        Args:
            session_pitch: Session-pitch identifier (e.g., "1031_2")
            data_types: List of data types to load (all if None)
            
        Returns:
            Dictionary with data type as key and filtered DataFrame as value
        """
        if data_types is None:
            data_types = list(self.data_files.keys())
        
        pitch_data = {}
        
        for data_type in data_types:
            df = self.load_data_type(data_type)
            pitch_df = df[df['session_pitch'] == session_pitch].copy()
            
            if len(pitch_df) == 0:
                warnings.warn(f"No data found for pitch {session_pitch} in {data_type}")
            
            pitch_data[data_type] = pitch_df
        
        return pitch_data
    
    def get_available_pitches(self, data_type: str = 'landmarks') -> List[str]:
        """
        Get list of all available pitch identifiers.
        
        Args:
            data_type: Data type to check for available pitches
            
        Returns:
            List of session_pitch identifiers
        """
        df = self.load_data_type(data_type)
        return sorted(df['session_pitch'].unique().tolist())
    
    def get_pitch_summary(self, session_pitch: str = None) -> pd.DataFrame:
        """
        Get summary statistics for pitch(es).
        
        Args:
            session_pitch: Specific pitch to analyze (all if None)
            
        Returns:
            DataFrame with summary statistics
        """
        landmarks_df = self.load_data_type('landmarks')
        
        if session_pitch:
            pitches_to_analyze = [session_pitch]
        else:
            pitches_to_analyze = self.get_available_pitches()
        
        summary_data = []
        
        for pitch in pitches_to_analyze:
            pitch_data = landmarks_df[landmarks_df['session_pitch'] == pitch]
            
            if len(pitch_data) == 0:
                continue
            
            summary = {
                'session_pitch': pitch,
                'duration_seconds': pitch_data['time'].max() - pitch_data['time'].min(),
                'sample_count': len(pitch_data),
                'sampling_rate_hz': len(pitch_data) / (pitch_data['time'].max() - pitch_data['time'].min()) if pitch_data['time'].max() > pitch_data['time'].min() else 0,
                'has_timing_markers': all(col in pitch_data.columns for col in ['pkh_time', 'fp_10_time', 'MER_time'])
            }
            
            # Add timing information if available
            timing_cols = ['pkh_time', 'fp_10_time', 'fp_100_time', 'MER_time', 'BR_time', 'MIR_time']
            for col in timing_cols:
                if col in pitch_data.columns:
                    summary[col] = pitch_data[col].iloc[0] if not pitch_data[col].isna().all() else None
            
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def get_timing_events(self, session_pitch: str) -> Dict[str, float]:
        """
        Extract timing events for a specific pitch.
        
        Args:
            session_pitch: Session-pitch identifier
            
        Returns:
            Dictionary with event names and timing values
        """
        landmarks_df = self.load_data_type('landmarks')
        pitch_data = landmarks_df[landmarks_df['session_pitch'] == session_pitch]
        
        if len(pitch_data) == 0:
            raise ValueError(f"No data found for pitch {session_pitch}")
        
        timing_events = {}
        timing_cols = ['pkh_time', 'fp_10_time', 'fp_100_time', 'MER_time', 'BR_time', 'MIR_time']
        
        for col in timing_cols:
            if col in pitch_data.columns and not pitch_data[col].isna().all():
                timing_events[col] = pitch_data[col].iloc[0]
        
        return timing_events
    
    def sync_data_by_time(self, pitch_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Synchronize multiple data types by common time points.
        
        Args:
            pitch_data: Dictionary of data types and their DataFrames
            
        Returns:
            Dictionary with synchronized DataFrames
        """
        if len(pitch_data) < 2:
            return pitch_data
        
        # Find common time range
        time_ranges = []
        for data_type, df in pitch_data.items():
            if 'time' in df.columns and len(df) > 0:
                time_ranges.append((df['time'].min(), df['time'].max()))
        
        if not time_ranges:
            return pitch_data
        
        common_start = max(t[0] for t in time_ranges)
        common_end = min(t[1] for t in time_ranges)
        
        # Filter to common time range
        synced_data = {}
        for data_type, df in pitch_data.items():
            if 'time' in df.columns:
                mask = (df['time'] >= common_start) & (df['time'] <= common_end)
                synced_data[data_type] = df[mask].copy()
            else:
                synced_data[data_type] = df.copy()
        
        return synced_data
    
    def get_data_columns_by_category(self, data_type: str) -> Dict[str, List[str]]:
        """
        Categorize columns in a data type by body part or measurement type.
        
        Args:
            data_type: Type of data to categorize
            
        Returns:
            Dictionary with column categories
        """
        df = self.load_data_type(data_type)
        columns = df.columns.tolist()
        
        # Remove standard columns
        excluded_cols = ['session_pitch', 'time', 'pkh_time', 'fp_10_time', 'fp_100_time', 
                        'MER_time', 'BR_time', 'MIR_time']
        measurement_cols = [col for col in columns if col not in excluded_cols]
        
        # Categorize by body part
        categories = {
            'throwing_arm': [],
            'glove_arm': [],
            'torso_pelvis': [],
            'rear_leg': [],
            'lead_leg': [],
            'other': []
        }
        
        for col in measurement_cols:
            col_lower = col.lower()
            
            # Throwing arm (right side for right-handed pitchers)
            if any(keyword in col_lower for keyword in ['elbow', 'shoulder', 'wrist', 'hand']):
                if 'glove' not in col_lower:
                    categories['throwing_arm'].append(col)
                else:
                    categories['glove_arm'].append(col)
            
            # Glove arm
            elif 'glove' in col_lower:
                categories['glove_arm'].append(col)
            
            # Torso and pelvis
            elif any(keyword in col_lower for keyword in ['torso', 'pelvis', 'thorax', 'centerofmass']):
                categories['torso_pelvis'].append(col)
            
            # Rear leg
            elif any(keyword in col_lower for keyword in ['rear_ankle', 'rear_hip', 'rear_knee']):
                categories['rear_leg'].append(col)
            
            # Lead leg
            elif any(keyword in col_lower for keyword in ['lead_ankle', 'lead_hip', 'lead_knee']):
                categories['lead_leg'].append(col)
            
            # Force plate data
            elif any(keyword in col_lower for keyword in ['force', 'moment']):
                if 'rear' in col_lower:
                    categories['rear_leg'].append(col)
                elif 'lead' in col_lower:
                    categories['lead_leg'].append(col)
                else:
                    categories['other'].append(col)
            
            else:
                categories['other'].append(col)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}


def load_sample_pitch_data(session_pitch: str = None) -> Tuple[str, Dict[str, pd.DataFrame]]:
    """
    Load sample pitch data for testing and development.
    
    Args:
        session_pitch: Specific pitch to load (random if None)
        
    Returns:
        Tuple of (session_pitch, pitch_data_dict)
    """
    loader = TimeSeriesLoader()
    
    if session_pitch is None:
        available_pitches = loader.get_available_pitches()
        if not available_pitches:
            raise ValueError("No pitches available in dataset")
        session_pitch = available_pitches[0]  # Take first available pitch
    
    pitch_data = loader.get_pitch_data(session_pitch)
    
    return session_pitch, pitch_data


def get_dataset_overview() -> Dict[str, any]:
    """
    Get overview of the entire time-series dataset.
    
    Returns:
        Dictionary with dataset statistics
    """
    loader = TimeSeriesLoader()
    
    overview = {
        'total_pitches': 0,
        'data_types': list(loader.data_files.keys()),
        'sample_pitch_summary': None
    }
    
    # Get pitch count and summary
    try:
        available_pitches = loader.get_available_pitches()
        overview['total_pitches'] = len(available_pitches)
        
        if available_pitches:
            # Get summary for first few pitches
            sample_pitches = available_pitches[:5]
            summary_df = loader.get_pitch_summary()
            overview['sample_pitch_summary'] = summary_df.head().to_dict('records')
    
    except Exception as e:
        overview['error'] = str(e)
    
    return overview
