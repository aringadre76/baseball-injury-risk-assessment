"""
Temporal feature extraction for biomechanical injury risk assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
from .biomechanical_signal_analyzer import SignalAnalyzer
from .time_series_data_loader import TimeSeriesLoader


class TemporalFeatureExtractor:
    """Extract temporal features specifically for injury risk assessment."""
    
    def __init__(self, sampling_rate: float = 360.0):
        """
        Initialize the temporal feature extractor.
        
        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.analyzer = SignalAnalyzer(sampling_rate)
        
        # Define injury-relevant signal groups
        self.injury_signal_groups = {
            'elbow_stress': {
                'landmarks': ['elbow_jc_x', 'elbow_jc_y', 'elbow_jc_z'],
                'joint_angles': ['elbow_angle_x', 'elbow_angle_y', 'elbow_angle_z'],
                'joint_velos': ['elbow_velo_x', 'elbow_velo_y', 'elbow_velo_z'],
                'forces_moments': ['elbow_force_x', 'elbow_force_y', 'elbow_force_z',
                                  'elbow_moment_x', 'elbow_moment_y', 'elbow_moment_z']
            },
            'shoulder_stress': {
                'landmarks': ['shoulder_jc_x', 'shoulder_jc_y', 'shoulder_jc_z'],
                'joint_angles': ['shoulder_angle_x', 'shoulder_angle_y', 'shoulder_angle_z'],
                'joint_velos': ['shoulder_velo_x', 'shoulder_velo_y', 'shoulder_velo_z'],
                'forces_moments': ['shoulder_force_x', 'shoulder_force_y', 'shoulder_force_z',
                                  'shoulder_moment_x', 'shoulder_moment_y', 'shoulder_moment_z']
            },
            'kinetic_chain': {
                'landmarks': ['pelvis_angle_x', 'pelvis_angle_y', 'pelvis_angle_z',
                             'torso_angle_x', 'torso_angle_y', 'torso_angle_z'],
                'joint_angles': ['torso_pelvis_angle_x', 'torso_pelvis_angle_y', 'torso_pelvis_angle_z'],
                'joint_velos': ['torso_velo_x', 'torso_velo_y', 'torso_velo_z',
                               'pelvis_velo_x', 'pelvis_velo_y', 'pelvis_velo_z']
            },
            'lower_body': {
                'landmarks': ['rear_hip_x', 'rear_hip_y', 'rear_hip_z',
                             'lead_hip_x', 'lead_hip_y', 'lead_hip_z'],
                'joint_angles': ['rear_hip_angle_x', 'rear_hip_angle_y', 'rear_hip_angle_z',
                                'lead_hip_angle_x', 'lead_hip_angle_y', 'lead_hip_angle_z'],
                'force_plate': ['rear_force_x', 'rear_force_y', 'rear_force_z',
                               'lead_force_x', 'lead_force_y', 'lead_force_z']
            }
        }
    
    def extract_phase_timing_features(self, pitch_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Extract features related to pitch phase timing and sequencing.
        
        Args:
            pitch_data: Dictionary with data types and DataFrames
            
        Returns:
            Dictionary of timing features
        """
        features = {}
        
        # Get timing events from any available dataset
        timing_events = {}
        for data_type, df in pitch_data.items():
            if len(df) > 0:
                timing_cols = ['pkh_time', 'fp_10_time', 'fp_100_time', 'MER_time', 'BR_time', 'MIR_time']
                for col in timing_cols:
                    if col in df.columns and not df[col].isna().all():
                        timing_events[col] = df[col].iloc[0]
                break
        
        if not timing_events:
            warnings.warn("No timing events found in data")
            return features
        
        # Calculate phase durations
        phase_transitions = [
            ('wind_up_to_peak', 'pkh_time', 0.0),
            ('peak_to_foot_contact', 'fp_10_time', 'pkh_time'),
            ('foot_contact_to_mer', 'MER_time', 'fp_10_time'),
            ('mer_to_ball_release', 'BR_time', 'MER_time'),
            ('ball_release_to_followthrough', 'MIR_time', 'BR_time')
        ]
        
        for phase_name, end_event, start_event in phase_transitions:
            if end_event in timing_events:
                if isinstance(start_event, str) and start_event in timing_events:
                    duration = timing_events[end_event] - timing_events[start_event]
                elif start_event == 0.0:
                    duration = timing_events[end_event] - start_event
                else:
                    continue
                
                features[f'{phase_name}_duration'] = duration
        
        # Calculate timing ratios
        if 'pkh_time' in timing_events and 'BR_time' in timing_events:
            total_pitch_time = timing_events['BR_time'] - timing_events['pkh_time']
            features['total_pitch_duration'] = total_pitch_time
            
            for phase_name, _, _ in phase_transitions:
                phase_duration_key = f'{phase_name}_duration'
                if phase_duration_key in features:
                    features[f'{phase_name}_ratio'] = features[phase_duration_key] / total_pitch_time
        
        # Timing symmetry and efficiency metrics
        if 'fp_10_time' in timing_events and 'MER_time' in timing_events and 'BR_time' in timing_events:
            acceleration_phase = timing_events['MER_time'] - timing_events['fp_10_time']
            deceleration_phase = timing_events['BR_time'] - timing_events['MER_time']
            
            if deceleration_phase > 0:
                features['accel_decel_ratio'] = acceleration_phase / deceleration_phase
            
            features['acceleration_phase_duration'] = acceleration_phase
            features['deceleration_phase_duration'] = deceleration_phase
        
        return features
    
    def extract_velocity_sequencing_features(self, pitch_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Extract features related to kinetic chain velocity sequencing.
        
        Args:
            pitch_data: Dictionary with data types and DataFrames
            
        Returns:
            Dictionary of velocity sequencing features
        """
        features = {}
        
        if 'joint_velos' not in pitch_data or len(pitch_data['joint_velos']) == 0:
            warnings.warn("No joint velocity data available")
            return features
        
        velo_df = pitch_data['joint_velos']
        time_array = velo_df['time'].values
        
        # Define kinetic chain sequence (proximal to distal)
        kinetic_chain_velos = [
            'pelvis_velo_z',      # Pelvis rotation
            'torso_velo_z',       # Torso rotation  
            'shoulder_velo_z',    # Shoulder internal rotation
            'elbow_velo_y',       # Elbow extension
            'wrist_velo_z'        # Wrist snap
        ]
        
        # Find peak velocity times for each segment
        peak_times = {}
        peak_velocities = {}
        
        for velo_col in kinetic_chain_velos:
            if velo_col in velo_df.columns:
                signal = velo_df[velo_col].values
                
                # Find peak (could be positive or negative)
                peak_idx = np.nanargmax(np.abs(signal))
                peak_times[velo_col] = time_array[peak_idx]
                peak_velocities[velo_col] = signal[peak_idx]
        
        # Calculate sequencing timing
        if len(peak_times) >= 2:
            # Time delays between adjacent segments
            for i in range(len(kinetic_chain_velos) - 1):
                proximal = kinetic_chain_velos[i]
                distal = kinetic_chain_velos[i + 1]
                
                if proximal in peak_times and distal in peak_times:
                    delay = peak_times[distal] - peak_times[proximal]
                    features[f'{proximal}_to_{distal}_delay'] = delay
                    
                    # Velocity transfer efficiency
                    if proximal in peak_velocities and distal in peak_velocities:
                        if peak_velocities[proximal] != 0:
                            transfer_ratio = abs(peak_velocities[distal]) / abs(peak_velocities[proximal])
                            features[f'{proximal}_to_{distal}_transfer_ratio'] = transfer_ratio
        
        # Overall sequencing quality metrics
        valid_peak_times = list(peak_times.values())
        if len(valid_peak_times) >= 3:
            # Check if sequence is monotonically increasing (proper proximal-to-distal timing)
            is_sequential = all(valid_peak_times[i] <= valid_peak_times[i+1] 
                              for i in range(len(valid_peak_times)-1))
            features['proper_kinetic_sequence'] = float(is_sequential)
            
            # Total sequence time
            features['kinetic_chain_duration'] = max(valid_peak_times) - min(valid_peak_times)
            
            # Sequence variability (lower is more consistent)
            if len(valid_peak_times) > 1:
                time_intervals = np.diff(sorted(valid_peak_times))
                features['sequence_interval_std'] = np.std(time_intervals)
                features['sequence_interval_mean'] = np.mean(time_intervals)
        
        return features
    
    def extract_force_development_features(self, pitch_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Extract features related to force development and ground reaction forces.
        
        Args:
            pitch_data: Dictionary with data types and DataFrames
            
        Returns:
            Dictionary of force development features
        """
        features = {}
        
        if 'force_plate' not in pitch_data or len(pitch_data['force_plate']) == 0:
            warnings.warn("No force plate data available")
            return features
        
        force_df = pitch_data['force_plate']
        time_array = force_df['time'].values
        
        # Analyze force components
        force_components = {
            'rear_vertical': 'rear_force_z',
            'rear_horizontal': 'rear_force_x', 
            'rear_lateral': 'rear_force_y',
            'lead_vertical': 'lead_force_z',
            'lead_horizontal': 'lead_force_x',
            'lead_lateral': 'lead_force_y'
        }
        
        for force_name, force_col in force_components.items():
            if force_col in force_df.columns:
                force_signal = force_df[force_col].values
                
                # Basic force characteristics
                features[f'{force_name}_peak'] = np.nanmax(np.abs(force_signal))
                features[f'{force_name}_mean'] = np.nanmean(force_signal)
                features[f'{force_name}_impulse'] = np.trapz(force_signal, time_array)
                
                # Rate of force development
                force_gradient = np.gradient(force_signal, time_array)
                features[f'{force_name}_max_rfd'] = np.nanmax(force_gradient)
                features[f'{force_name}_min_rfd'] = np.nanmin(force_gradient)
                
                # Time to peak force
                peak_idx = np.nanargmax(np.abs(force_signal))
                features[f'{force_name}_time_to_peak'] = time_array[peak_idx] - time_array[0]
        
        # Ground reaction force patterns
        if 'rear_force_z' in force_df.columns and 'lead_force_z' in force_df.columns:
            rear_vertical = force_df['rear_force_z'].values
            lead_vertical = force_df['lead_force_z'].values
            
            # Weight transfer metrics
            total_vertical = rear_vertical + lead_vertical
            rear_proportion = rear_vertical / (total_vertical + 1e-6)  # Avoid division by zero
            
            features['max_rear_weight_proportion'] = np.nanmax(rear_proportion)
            features['min_rear_weight_proportion'] = np.nanmin(rear_proportion)
            features['weight_transfer_range'] = features['max_rear_weight_proportion'] - features['min_rear_weight_proportion']
            
            # Find weight shift timing
            rear_peak_idx = np.nanargmax(rear_vertical)
            lead_peak_idx = np.nanargmax(lead_vertical)
            
            features['rear_to_lead_peak_delay'] = time_array[lead_peak_idx] - time_array[rear_peak_idx]
        
        return features
    
    def extract_movement_efficiency_features(self, pitch_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Extract features related to movement efficiency and coordination.
        
        Args:
            pitch_data: Dictionary with data types and DataFrames
            
        Returns:
            Dictionary of movement efficiency features
        """
        features = {}
        
        # Analyze center of mass movement
        if 'landmarks' in pitch_data and 'centerofmass_x' in pitch_data['landmarks'].columns:
            landmarks_df = pitch_data['landmarks']
            time_array = landmarks_df['time'].values
            
            com_x = landmarks_df['centerofmass_x'].values
            com_y = landmarks_df['centerofmass_y'].values
            com_z = landmarks_df['centerofmass_z'].values
            
            # Center of mass displacement
            com_displacement = np.sqrt(
                (com_x - com_x[0])**2 + 
                (com_y - com_y[0])**2 + 
                (com_z - com_z[0])**2
            )
            
            features['com_total_displacement'] = np.nanmax(com_displacement)
            features['com_final_displacement'] = com_displacement[-1]
            
            # Center of mass velocity
            com_velocity = np.sqrt(
                np.gradient(com_x, time_array)**2 +
                np.gradient(com_y, time_array)**2 +
                np.gradient(com_z, time_array)**2
            )
            
            features['com_peak_velocity'] = np.nanmax(com_velocity)
            features['com_mean_velocity'] = np.nanmean(com_velocity)
        
        # Analyze stride mechanics
        if 'landmarks' in pitch_data:
            landmarks_df = pitch_data['landmarks']
            
            # Stride length estimation (rear foot to lead foot distance)
            if all(col in landmarks_df.columns for col in ['rear_ankle_jc_x', 'lead_ankle_jc_x']):
                rear_ankle_x = landmarks_df['rear_ankle_jc_x'].values
                lead_ankle_x = landmarks_df['lead_ankle_jc_x'].values
                
                stride_distance = lead_ankle_x - rear_ankle_x
                features['max_stride_distance'] = np.nanmax(stride_distance)
                features['final_stride_distance'] = stride_distance[-1]
                
                # Stride rate (how quickly stride develops)
                stride_velocity = np.gradient(stride_distance, time_array)
                features['peak_stride_velocity'] = np.nanmax(stride_velocity)
        
        # Postural stability during delivery
        if 'joint_angles' in pitch_data:
            angles_df = pitch_data['joint_angles']
            
            # Torso stability
            stability_angles = ['torso_angle_x', 'torso_angle_y', 'pelvis_angle_x', 'pelvis_angle_y']
            
            for angle_col in stability_angles:
                if angle_col in angles_df.columns:
                    angle_signal = angles_df[angle_col].values
                    
                    # Variability indicates less stability
                    features[f'{angle_col}_stability'] = 1.0 / (np.nanstd(angle_signal) + 1e-6)
                    features[f'{angle_col}_range'] = np.nanmax(angle_signal) - np.nanmin(angle_signal)
        
        return features
    
    def extract_asymmetry_features(self, pitch_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Extract features related to bilateral asymmetries.
        
        Args:
            pitch_data: Dictionary with data types and DataFrames
            
        Returns:
            Dictionary of asymmetry features
        """
        features = {}
        
        # Define bilateral pairs
        bilateral_pairs = {
            'hip_angles': [('rear_hip_angle_x', 'lead_hip_angle_x'),
                          ('rear_hip_angle_y', 'lead_hip_angle_y'),
                          ('rear_hip_angle_z', 'lead_hip_angle_z')],
            'ankle_positions': [('rear_ankle_jc_x', 'lead_ankle_jc_x'),
                               ('rear_ankle_jc_y', 'lead_ankle_jc_y'),
                               ('rear_ankle_jc_z', 'lead_ankle_jc_z')],
            'forces': [('rear_force_x', 'lead_force_x'),
                      ('rear_force_y', 'lead_force_y'),
                      ('rear_force_z', 'lead_force_z')]
        }
        
        for category, pairs in bilateral_pairs.items():
            for pair in pairs:
                left_signal, right_signal = pair
                
                # Find appropriate dataset
                df = None
                if category == 'hip_angles' and 'joint_angles' in pitch_data:
                    df = pitch_data['joint_angles']
                elif category == 'ankle_positions' and 'landmarks' in pitch_data:
                    df = pitch_data['landmarks']
                elif category == 'forces' and 'force_plate' in pitch_data:
                    df = pitch_data['force_plate']
                
                if df is not None and left_signal in df.columns and right_signal in df.columns:
                    left_values = df[left_signal].values
                    right_values = df[right_signal].values
                    
                    # Remove NaN values for comparison
                    valid_mask = ~(np.isnan(left_values) | np.isnan(right_values))
                    if valid_mask.sum() > 0:
                        left_clean = left_values[valid_mask]
                        right_clean = right_values[valid_mask]
                        
                        # Asymmetry indices
                        mean_diff = np.mean(left_clean) - np.mean(right_clean)
                        mean_sum = np.mean(left_clean) + np.mean(right_clean)
                        
                        if abs(mean_sum) > 1e-6:
                            features[f'{left_signal}_{right_signal}_asymmetry_index'] = abs(mean_diff) / abs(mean_sum) * 100
                        
                        # Peak asymmetry
                        peak_left = np.max(np.abs(left_clean))
                        peak_right = np.max(np.abs(right_clean))
                        
                        if peak_left + peak_right > 1e-6:
                            features[f'{left_signal}_{right_signal}_peak_asymmetry'] = abs(peak_left - peak_right) / (peak_left + peak_right) * 100
                        
                        # Temporal asymmetry (timing differences)
                        peak_left_idx = np.argmax(np.abs(left_clean))
                        peak_right_idx = np.argmax(np.abs(right_clean))
                        
                        features[f'{left_signal}_{right_signal}_timing_asymmetry'] = abs(peak_left_idx - peak_right_idx) / len(left_clean) * 100
        
        return features
    
    def extract_comprehensive_temporal_features(self, pitch_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Extract comprehensive temporal features for injury risk assessment.
        
        Args:
            pitch_data: Dictionary with data types and DataFrames
            
        Returns:
            Dictionary with all temporal features
        """
        all_features = {}
        
        # Extract different categories of temporal features
        feature_extractors = [
            self.extract_phase_timing_features,
            self.extract_velocity_sequencing_features,
            self.extract_force_development_features,
            self.extract_movement_efficiency_features,
            self.extract_asymmetry_features
        ]
        
        for extractor in feature_extractors:
            try:
                features = extractor(pitch_data)
                all_features.update(features)
            except Exception as e:
                warnings.warn(f"Error in {extractor.__name__}: {e}")
        
        return all_features


def extract_pitcher_temporal_features(session_pitch: str, 
                                    loader: Optional[TimeSeriesLoader] = None) -> Dict[str, float]:
    """
    Extract temporal features for a specific pitcher.
    
    Args:
        session_pitch: Session-pitch identifier
        loader: TimeSeriesLoader instance (creates new if None)
        
    Returns:
        Dictionary of temporal features
    """
    if loader is None:
        loader = TimeSeriesLoader()
    
    # Load pitch data
    pitch_data = loader.get_pitch_data(session_pitch)
    
    # Extract temporal features
    extractor = TemporalFeatureExtractor()
    temporal_features = extractor.extract_comprehensive_temporal_features(pitch_data)
    
    return temporal_features


def batch_extract_temporal_features(pitch_list: Optional[List[str]] = None,
                                   max_pitches: int = 50) -> pd.DataFrame:
    """
    Extract temporal features for multiple pitches.
    
    Args:
        pitch_list: List of pitch identifiers (all available if None)
        max_pitches: Maximum number of pitches to process
        
    Returns:
        DataFrame with pitch identifiers and temporal features
    """
    loader = TimeSeriesLoader()
    
    if pitch_list is None:
        pitch_list = loader.get_available_pitches()
    
    # Limit number of pitches to avoid overwhelming processing
    pitch_list = pitch_list[:max_pitches]
    
    all_features = []
    
    for i, session_pitch in enumerate(pitch_list):
        print(f"Processing pitch {i+1}/{len(pitch_list)}: {session_pitch}")
        
        try:
            temporal_features = extract_pitcher_temporal_features(session_pitch, loader)
            temporal_features['session_pitch'] = session_pitch
            all_features.append(temporal_features)
            
        except Exception as e:
            print(f"Error processing pitch {session_pitch}: {e}")
            continue
    
    if all_features:
        return pd.DataFrame(all_features)
    else:
        return pd.DataFrame()
