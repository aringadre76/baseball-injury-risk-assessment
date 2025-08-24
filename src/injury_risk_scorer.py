"""
Advanced composite injury risk scoring using time-series and POI features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional, Union
import warnings
from .temporal_feature_extractor import batch_extract_temporal_features, extract_pitcher_temporal_features
from .openbiomechanics_loader import OpenBiomechanicsLoader
from .biomechanical_signal_analyzer import analyze_pitch_signals


class AdvancedRiskScorer:
    """Advanced composite injury risk scoring system."""
    
    def __init__(self):
        """Initialize the advanced risk scorer."""
        self.poi_loader = OpenBiomechanicsLoader()
        self.scalers = {}
        self.risk_models = {}
        
        # Define injury-specific feature weights based on biomechanical literature
        self.injury_risk_weights = {
            'ucl_injury': {
                # Elbow-specific features (highest weight)
                'elbow_stress': 0.40,
                'elbow_varus_moment': 0.30,
                'max_elbow_extension_velo': 0.20,
                'elbow_timing': 0.10
            },
            'shoulder_injury': {
                # Shoulder-specific features
                'shoulder_stress': 0.35,
                'shoulder_internal_rotation_moment': 0.25,
                'max_shoulder_internal_rotational_velo': 0.20,
                'kinetic_chain_efficiency': 0.20
            },
            'overall_injury': {
                # Balanced approach across all systems
                'kinetic_chain': 0.25,
                'force_development': 0.20,
                'movement_efficiency': 0.20,
                'timing_coordination': 0.15,
                'asymmetry': 0.10,
                'workload': 0.10
            }
        }
    
    def create_elbow_stress_composite(self, features: Dict[str, float]) -> float:
        """
        Create composite elbow stress score.
        
        Args:
            features: Dictionary of all features
            
        Returns:
            Composite elbow stress score (0-100)
        """
        elbow_components = {}
        
        # Primary elbow stress indicators
        if 'elbow_varus_moment' in features:
            elbow_components['varus_moment'] = features['elbow_varus_moment']
        
        if 'max_elbow_extension_velo' in features:
            elbow_components['extension_velocity'] = features['max_elbow_extension_velo']
        
        # Time-series derived elbow features
        elbow_timing_features = [f for f in features.keys() if 'elbow' in f.lower() and ('time' in f or 'duration' in f)]
        if elbow_timing_features:
            timing_values = [features[f] for f in elbow_timing_features if not np.isnan(features[f])]
            if timing_values:
                elbow_components['timing_stress'] = np.mean(timing_values)
        
        elbow_force_features = [f for f in features.keys() if 'elbow' in f.lower() and ('force' in f or 'moment' in f)]
        if elbow_force_features:
            force_values = [features[f] for f in elbow_force_features if not np.isnan(features[f])]
            if force_values:
                elbow_components['dynamic_stress'] = np.mean(np.abs(force_values))
        
        # Asymmetry contributions
        elbow_asymmetry_features = [f for f in features.keys() if 'elbow' in f.lower() and 'asymmetry' in f]
        if elbow_asymmetry_features:
            asymmetry_values = [features[f] for f in elbow_asymmetry_features if not np.isnan(features[f])]
            if asymmetry_values:
                elbow_components['asymmetry_stress'] = np.mean(asymmetry_values)
        
        if not elbow_components:
            return 0.0
        
        # Normalize each component to 0-1 scale and combine
        normalized_components = {}
        for component, value in elbow_components.items():
            if component == 'varus_moment':
                # Higher varus moment = higher risk
                normalized_components[component] = min(1.0, abs(value) / 100.0)  # Normalize to typical max values
            elif component == 'extension_velocity':
                # Higher extension velocity = higher risk
                normalized_components[component] = min(1.0, abs(value) / 2000.0)  # Normalize to typical max values
            elif component == 'timing_stress':
                # Abnormal timing (too fast or too slow) = higher risk
                optimal_timing = 0.15  # Typical elbow timing
                normalized_components[component] = min(1.0, abs(value - optimal_timing) / optimal_timing)
            elif component == 'dynamic_stress':
                # Higher dynamic stress = higher risk
                normalized_components[component] = min(1.0, abs(value) / 1000.0)
            elif component == 'asymmetry_stress':
                # Higher asymmetry = higher risk
                normalized_components[component] = min(1.0, value / 20.0)  # 20% asymmetry as max
        
        # Weighted combination
        weights = {
            'varus_moment': 0.35,
            'extension_velocity': 0.25,
            'dynamic_stress': 0.20,
            'timing_stress': 0.15,
            'asymmetry_stress': 0.05
        }
        
        composite_score = 0.0
        total_weight = 0.0
        
        for component, norm_value in normalized_components.items():
            weight = weights.get(component, 0.1)
            composite_score += norm_value * weight
            total_weight += weight
        
        if total_weight > 0:
            composite_score = (composite_score / total_weight) * 100  # Scale to 0-100
        
        return min(100.0, composite_score)
    
    def create_shoulder_stress_composite(self, features: Dict[str, float]) -> float:
        """
        Create composite shoulder stress score.
        
        Args:
            features: Dictionary of all features
            
        Returns:
            Composite shoulder stress score (0-100)
        """
        shoulder_components = {}
        
        # Primary shoulder stress indicators
        if 'shoulder_internal_rotation_moment' in features:
            shoulder_components['internal_rotation_moment'] = features['shoulder_internal_rotation_moment']
        
        if 'max_shoulder_internal_rotational_velo' in features:
            shoulder_components['internal_rotation_velocity'] = features['max_shoulder_internal_rotational_velo']
        
        # Shoulder range of motion and positioning
        shoulder_angle_features = [f for f in features.keys() if 'shoulder' in f.lower() and 'angle' in f]
        if shoulder_angle_features:
            angle_values = [features[f] for f in shoulder_angle_features if not np.isnan(features[f])]
            if angle_values:
                shoulder_components['range_of_motion'] = np.std(angle_values)  # Variability as risk factor
        
        # Kinetic chain contribution to shoulder stress
        kinetic_chain_features = [f for f in features.keys() if any(keyword in f.lower() for keyword in ['sequence', 'transfer', 'chain'])]
        if kinetic_chain_features:
            chain_values = [features[f] for f in kinetic_chain_features if not np.isnan(features[f])]
            if chain_values:
                # Poor kinetic chain efficiency increases shoulder stress
                shoulder_components['kinetic_chain_deficiency'] = 1.0 / (np.mean(chain_values) + 0.1)
        
        # Timing and coordination factors
        shoulder_timing_features = [f for f in features.keys() if 'shoulder' in f.lower() and ('time' in f or 'delay' in f)]
        if shoulder_timing_features:
            timing_values = [features[f] for f in shoulder_timing_features if not np.isnan(features[f])]
            if timing_values:
                shoulder_components['timing_coordination'] = np.std(timing_values)  # Variability as risk
        
        if not shoulder_components:
            return 0.0
        
        # Normalize components
        normalized_components = {}
        for component, value in shoulder_components.items():
            if component == 'internal_rotation_moment':
                normalized_components[component] = min(1.0, abs(value) / 80.0)
            elif component == 'internal_rotation_velocity':
                normalized_components[component] = min(1.0, abs(value) / 7000.0)
            elif component == 'range_of_motion':
                normalized_components[component] = min(1.0, value / 50.0)  # High ROM variability
            elif component == 'kinetic_chain_deficiency':
                normalized_components[component] = min(1.0, value / 2.0)
            elif component == 'timing_coordination':
                normalized_components[component] = min(1.0, value / 0.2)
        
        # Weighted combination
        weights = {
            'internal_rotation_moment': 0.30,
            'internal_rotation_velocity': 0.25,
            'kinetic_chain_deficiency': 0.20,
            'range_of_motion': 0.15,
            'timing_coordination': 0.10
        }
        
        composite_score = 0.0
        total_weight = 0.0
        
        for component, norm_value in normalized_components.items():
            weight = weights.get(component, 0.1)
            composite_score += norm_value * weight
            total_weight += weight
        
        if total_weight > 0:
            composite_score = (composite_score / total_weight) * 100
        
        return min(100.0, composite_score)
    
    def create_kinetic_chain_efficiency_score(self, features: Dict[str, float]) -> float:
        """
        Create kinetic chain efficiency score.
        
        Args:
            features: Dictionary of all features
            
        Returns:
            Kinetic chain efficiency score (0-100, higher = more efficient)
        """
        efficiency_components = {}
        
        # Velocity sequencing efficiency
        sequence_features = [f for f in features.keys() if 'sequence' in f.lower()]
        if sequence_features:
            # Look for proper sequencing indicator
            if 'proper_kinetic_sequence' in features:
                efficiency_components['proper_sequencing'] = features['proper_kinetic_sequence']
        
        # Transfer ratios between segments
        transfer_features = [f for f in features.keys() if 'transfer_ratio' in f]
        if transfer_features:
            transfer_values = [features[f] for f in transfer_features if not np.isnan(features[f])]
            if transfer_values:
                # Ideal transfer ratios are around 1.0-2.0
                transfer_efficiency = [1.0 - abs(v - 1.5) / 1.5 for v in transfer_values if 0.5 <= v <= 3.0]
                if transfer_efficiency:
                    efficiency_components['transfer_efficiency'] = np.mean(transfer_efficiency)
        
        # Timing coordination
        if 'sequence_interval_std' in features and not np.isnan(features['sequence_interval_std']):
            # Lower timing variability = higher efficiency
            timing_consistency = 1.0 / (features['sequence_interval_std'] + 0.01)
            efficiency_components['timing_consistency'] = min(1.0, timing_consistency / 10.0)
        
        # Hip-shoulder separation
        if 'max_rotation_hip_shoulder_separation' in features:
            separation = features['max_rotation_hip_shoulder_separation']
            # Optimal separation is 30-45 degrees
            if not np.isnan(separation):
                optimal_separation = 37.5
                separation_efficiency = 1.0 - abs(separation - optimal_separation) / optimal_separation
                efficiency_components['hip_shoulder_separation'] = max(0.0, separation_efficiency)
        
        # Force development efficiency
        force_efficiency_features = [f for f in features.keys() if 'rfd' in f.lower() and 'max' in f]
        if force_efficiency_features:
            rfd_values = [features[f] for f in force_efficiency_features if not np.isnan(features[f])]
            if rfd_values:
                # Higher rate of force development = higher efficiency
                efficiency_components['force_development'] = min(1.0, np.mean(np.abs(rfd_values)) / 1000.0)
        
        if not efficiency_components:
            return 50.0  # Neutral score if no data
        
        # Combine efficiency components
        weights = {
            'proper_sequencing': 0.30,
            'transfer_efficiency': 0.25,
            'timing_consistency': 0.20,
            'hip_shoulder_separation': 0.15,
            'force_development': 0.10
        }
        
        efficiency_score = 0.0
        total_weight = 0.0
        
        for component, value in efficiency_components.items():
            weight = weights.get(component, 0.1)
            efficiency_score += value * weight
            total_weight += weight
        
        if total_weight > 0:
            efficiency_score = (efficiency_score / total_weight) * 100
        
        return min(100.0, max(0.0, efficiency_score))
    
    def create_movement_quality_score(self, features: Dict[str, float]) -> float:
        """
        Create overall movement quality score.
        
        Args:
            features: Dictionary of all features
            
        Returns:
            Movement quality score (0-100, higher = better quality)
        """
        quality_components = {}
        
        # Postural stability
        stability_features = [f for f in features.keys() if 'stability' in f]
        if stability_features:
            stability_values = [features[f] for f in stability_features if not np.isnan(features[f])]
            if stability_values:
                quality_components['postural_stability'] = np.mean(stability_values)
        
        # Movement smoothness (low variability in joint angles)
        range_features = [f for f in features.keys() if 'range' in f and 'angle' in f]
        if range_features:
            range_values = [features[f] for f in range_features if not np.isnan(features[f])]
            if range_values:
                # Lower range variability = smoother movement
                quality_components['movement_smoothness'] = 1.0 / (np.mean(range_values) + 1.0)
        
        # Asymmetry penalty (lower asymmetry = higher quality)
        asymmetry_features = [f for f in features.keys() if 'asymmetry' in f]
        if asymmetry_features:
            asymmetry_values = [features[f] for f in asymmetry_features if not np.isnan(features[f])]
            if asymmetry_values:
                # Convert asymmetry to quality score (inverse relationship)
                avg_asymmetry = np.mean(asymmetry_values)
                quality_components['bilateral_symmetry'] = max(0.0, 1.0 - avg_asymmetry / 20.0)
        
        # Center of mass control
        com_features = [f for f in features.keys() if 'com' in f.lower()]
        if com_features:
            com_values = [features[f] for f in com_features if not np.isnan(features[f])]
            if com_values:
                # Controlled COM movement indicates good balance
                quality_components['balance_control'] = min(1.0, 1.0 / (np.std(com_values) + 0.1))
        
        # Stride mechanics quality
        if 'final_stride_distance' in features and not np.isnan(features['final_stride_distance']):
            stride_distance = features['final_stride_distance']
            # Optimal stride is typically 80-110% of height
            # Assuming average height around 1.8m, optimal stride around 1.6m
            optimal_stride = 1.6
            stride_quality = 1.0 - abs(stride_distance - optimal_stride) / optimal_stride
            quality_components['stride_mechanics'] = max(0.0, stride_quality)
        
        if not quality_components:
            return 50.0  # Neutral score if no data
        
        # Combine quality components
        weights = {
            'postural_stability': 0.25,
            'movement_smoothness': 0.20,
            'bilateral_symmetry': 0.20,
            'balance_control': 0.20,
            'stride_mechanics': 0.15
        }
        
        quality_score = 0.0
        total_weight = 0.0
        
        for component, value in quality_components.items():
            # Ensure all components are normalized to 0-1
            normalized_value = max(0.0, min(1.0, value))
            weight = weights.get(component, 0.1)
            quality_score += normalized_value * weight
            total_weight += weight
        
        if total_weight > 0:
            quality_score = (quality_score / total_weight) * 100
        
        return min(100.0, max(0.0, quality_score))
    
    def create_comprehensive_risk_profile(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Create comprehensive injury risk profile.
        
        Args:
            features: Dictionary of all features (POI + temporal)
            
        Returns:
            Dictionary with risk scores and components
        """
        risk_profile = {}
        
        # Individual component scores
        risk_profile['elbow_stress_score'] = self.create_elbow_stress_composite(features)
        risk_profile['shoulder_stress_score'] = self.create_shoulder_stress_composite(features)
        risk_profile['kinetic_chain_efficiency'] = self.create_kinetic_chain_efficiency_score(features)
        risk_profile['movement_quality'] = self.create_movement_quality_score(features)
        
        # Specific injury risk scores
        risk_profile['ucl_injury_risk'] = (
            risk_profile['elbow_stress_score'] * 0.7 +
            (100 - risk_profile['kinetic_chain_efficiency']) * 0.2 +
            (100 - risk_profile['movement_quality']) * 0.1
        )
        
        risk_profile['shoulder_injury_risk'] = (
            risk_profile['shoulder_stress_score'] * 0.6 +
            (100 - risk_profile['kinetic_chain_efficiency']) * 0.3 +
            (100 - risk_profile['movement_quality']) * 0.1
        )
        
        # Overall injury risk (weighted average of specific risks)
        risk_profile['overall_injury_risk'] = (
            risk_profile['ucl_injury_risk'] * 0.4 +
            risk_profile['shoulder_injury_risk'] * 0.4 +
            (100 - risk_profile['kinetic_chain_efficiency']) * 0.1 +
            (100 - risk_profile['movement_quality']) * 0.1
        )
        
        # Risk categories
        risk_profile['risk_category'] = self._categorize_risk(risk_profile['overall_injury_risk'])
        
        return risk_profile
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into levels."""
        if risk_score < 25:
            return 'Low'
        elif risk_score < 50:
            return 'Moderate'
        elif risk_score < 75:
            return 'High'
        else:
            return 'Very High'
    
    def analyze_pitcher_risk(self, session_pitch: str) -> Dict[str, Union[float, str]]:
        """
        Perform comprehensive risk analysis for a single pitcher.
        
        Args:
            session_pitch: Session-pitch identifier
            
        Returns:
            Dictionary with comprehensive risk analysis
        """
        # Get temporal features
        temporal_features = extract_pitcher_temporal_features(session_pitch)
        
        # Get POI features
        poi_data = self.poi_loader.load_and_merge_data()
        pitch_poi = poi_data[poi_data['session_pitch'] == session_pitch]
        
        if len(pitch_poi) == 0:
            raise ValueError(f"No POI data found for pitch {session_pitch}")
        
        # Combine all features
        all_features = temporal_features.copy()
        for col in pitch_poi.columns:
            if col != 'session_pitch' and pd.api.types.is_numeric_dtype(pitch_poi[col]):
                all_features[col] = pitch_poi[col].iloc[0]
        
        # Create risk profile
        risk_profile = self.create_comprehensive_risk_profile(all_features)
        risk_profile['session_pitch'] = session_pitch
        
        return risk_profile


def batch_analyze_pitcher_risks(pitch_list: Optional[List[str]] = None,
                               max_pitches: int = 20) -> pd.DataFrame:
    """
    Perform batch risk analysis for multiple pitchers.
    
    Args:
        pitch_list: List of pitch identifiers (sample if None)
        max_pitches: Maximum number of pitches to analyze
        
    Returns:
        DataFrame with risk profiles for all pitches
    """
    from .time_series_data_loader import TimeSeriesLoader
    
    if pitch_list is None:
        loader = TimeSeriesLoader()
        available_pitches = loader.get_available_pitches()
        pitch_list = available_pitches[:max_pitches]
    
    scorer = AdvancedRiskScorer()
    risk_profiles = []
    
    for i, session_pitch in enumerate(pitch_list):
        print(f"Analyzing risk for pitch {i+1}/{len(pitch_list)}: {session_pitch}")
        
        try:
            risk_profile = scorer.analyze_pitcher_risk(session_pitch)
            risk_profiles.append(risk_profile)
            
        except Exception as e:
            print(f"Error analyzing pitch {session_pitch}: {e}")
            continue
    
    if risk_profiles:
        return pd.DataFrame(risk_profiles)
    else:
        return pd.DataFrame()
