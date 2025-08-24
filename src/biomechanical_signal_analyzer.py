"""
Signal analysis and feature extraction for biomechanical time-series data.
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
from scipy.stats import skew, kurtosis
from typing import Dict, List, Tuple, Optional, Union
import warnings


class SignalAnalyzer:
    """Analyzer for biomechanical signal processing and feature extraction."""
    
    def __init__(self, sampling_rate: float = 360.0):
        """
        Initialize the signal analyzer.
        
        Args:
            sampling_rate: Sampling rate in Hz (default 360 Hz for most OpenBiomechanics data)
        """
        self.sampling_rate = sampling_rate
        self.nyquist_freq = sampling_rate / 2.0
    
    def apply_filter(self, signal: np.ndarray, 
                    filter_type: str = 'lowpass',
                    cutoff_freq: Union[float, Tuple[float, float]] = 50.0,
                    order: int = 4) -> np.ndarray:
        """
        Apply digital filter to signal.
        
        Args:
            signal: Input signal array
            filter_type: Type of filter ('lowpass', 'highpass', 'bandpass')
            cutoff_freq: Cutoff frequency (or tuple for bandpass)
            order: Filter order
            
        Returns:
            Filtered signal
        """
        if len(signal) < 2 * order:
            warnings.warn("Signal too short for filtering, returning original")
            return signal
        
        # Handle NaN values
        if np.isnan(signal).any():
            valid_mask = ~np.isnan(signal)
            if valid_mask.sum() < len(signal) * 0.5:
                warnings.warn("Too many NaN values in signal")
                return signal
            
            # Interpolate over NaN values for filtering
            signal_interp = signal.copy()
            signal_interp[~valid_mask] = np.interp(
                np.where(~valid_mask)[0], 
                np.where(valid_mask)[0], 
                signal[valid_mask]
            )
        else:
            signal_interp = signal
        
        try:
            if filter_type == 'lowpass':
                b, a = butter(order, cutoff_freq / self.nyquist_freq, btype='low')
            elif filter_type == 'highpass':
                b, a = butter(order, cutoff_freq / self.nyquist_freq, btype='high')
            elif filter_type == 'bandpass':
                if isinstance(cutoff_freq, (tuple, list)) and len(cutoff_freq) == 2:
                    low, high = cutoff_freq
                    b, a = butter(order, [low / self.nyquist_freq, high / self.nyquist_freq], btype='band')
                else:
                    raise ValueError("Bandpass filter requires tuple of (low, high) frequencies")
            else:
                raise ValueError(f"Unknown filter type: {filter_type}")
            
            filtered_signal = filtfilt(b, a, signal_interp)
            
            # Restore NaN values in original positions
            if np.isnan(signal).any():
                filtered_signal[~valid_mask] = np.nan
            
            return filtered_signal
            
        except Exception as e:
            warnings.warn(f"Filtering failed: {e}, returning original signal")
            return signal
    
    def smooth_signal(self, signal: np.ndarray, 
                     method: str = 'savgol',
                     window_length: int = 11,
                     polyorder: int = 3) -> np.ndarray:
        """
        Smooth signal using various methods.
        
        Args:
            signal: Input signal
            method: Smoothing method ('savgol', 'moving_average')
            window_length: Length of smoothing window
            polyorder: Polynomial order (for Savitzky-Golay)
            
        Returns:
            Smoothed signal
        """
        if len(signal) < window_length:
            warnings.warn("Signal shorter than window length, returning original")
            return signal
        
        # Handle NaN values
        valid_mask = ~np.isnan(signal)
        if valid_mask.sum() < len(signal) * 0.5:
            warnings.warn("Too many NaN values for smoothing")
            return signal
        
        try:
            if method == 'savgol':
                # Ensure window_length is odd
                if window_length % 2 == 0:
                    window_length += 1
                
                if window_length > len(signal):
                    window_length = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
                
                if polyorder >= window_length:
                    polyorder = window_length - 1
                
                smoothed = savgol_filter(signal, window_length, polyorder)
                
            elif method == 'moving_average':
                smoothed = np.convolve(signal, np.ones(window_length)/window_length, mode='same')
                
            else:
                raise ValueError(f"Unknown smoothing method: {method}")
            
            return smoothed
            
        except Exception as e:
            warnings.warn(f"Smoothing failed: {e}, returning original signal")
            return signal
    
    def find_peaks_enhanced(self, signal: np.ndarray,
                           peak_type: str = 'both',
                           height_threshold: Optional[float] = None,
                           distance: Optional[int] = None,
                           prominence: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Find peaks and valleys in signal with enhanced detection.
        
        Args:
            signal: Input signal
            peak_type: Type of peaks to find ('peaks', 'valleys', 'both')
            height_threshold: Minimum height for peaks
            distance: Minimum distance between peaks (in samples)
            prominence: Minimum prominence of peaks
            
        Returns:
            Dictionary with peak/valley indices and properties
        """
        results = {}
        
        # Handle NaN values
        valid_mask = ~np.isnan(signal)
        if valid_mask.sum() < len(signal) * 0.5:
            warnings.warn("Too many NaN values for peak detection")
            return results
        
        # Set default parameters based on signal characteristics
        if distance is None:
            distance = max(1, int(self.sampling_rate * 0.05))  # 50ms minimum distance
        
        if prominence is None:
            signal_std = np.nanstd(signal)
            prominence = signal_std * 0.1  # 10% of signal standard deviation
        
        try:
            # Find peaks (local maxima)
            if peak_type in ['peaks', 'both']:
                peak_kwargs = {'distance': distance, 'prominence': prominence}
                if height_threshold is not None:
                    peak_kwargs['height'] = height_threshold
                
                peaks, peak_props = find_peaks(signal, **peak_kwargs)
                results['peaks'] = peaks
                results['peak_heights'] = signal[peaks]
                results['peak_prominences'] = peak_props.get('prominences', np.array([]))
            
            # Find valleys (local minima)
            if peak_type in ['valleys', 'both']:
                # Invert signal to find valleys as peaks
                inverted_signal = -signal
                valley_kwargs = {'distance': distance, 'prominence': prominence}
                if height_threshold is not None:
                    valley_kwargs['height'] = -height_threshold
                
                valleys, valley_props = find_peaks(inverted_signal, **valley_kwargs)
                results['valleys'] = valleys
                results['valley_depths'] = signal[valleys]
                results['valley_prominences'] = valley_props.get('prominences', np.array([]))
        
        except Exception as e:
            warnings.warn(f"Peak detection failed: {e}")
        
        return results
    
    def extract_timing_features(self, signal: np.ndarray, 
                               time_array: np.ndarray,
                               timing_events: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Extract timing-based features from signal.
        
        Args:
            signal: Input signal array
            time_array: Corresponding time array
            timing_events: Dictionary of known timing events
            
        Returns:
            Dictionary of timing features
        """
        features = {}
        
        if len(signal) != len(time_array):
            warnings.warn("Signal and time arrays must have same length")
            return features
        
        # Basic timing features
        features['duration'] = time_array[-1] - time_array[0]
        features['time_to_peak'] = time_array[np.nanargmax(signal)] - time_array[0]
        features['time_to_min'] = time_array[np.nanargmin(signal)] - time_array[0]
        
        # Peak and valley timing
        peak_results = self.find_peaks_enhanced(signal)
        
        if 'peaks' in peak_results and len(peak_results['peaks']) > 0:
            peak_times = time_array[peak_results['peaks']]
            features['first_peak_time'] = peak_times[0] - time_array[0]
            features['last_peak_time'] = peak_times[-1] - time_array[0]
            features['num_peaks'] = len(peak_times)
            
            if len(peak_times) > 1:
                features['peak_interval_mean'] = np.mean(np.diff(peak_times))
                features['peak_interval_std'] = np.std(np.diff(peak_times))
        
        # Event-based timing features
        if timing_events:
            for event_name, event_time in timing_events.items():
                if not np.isnan(event_time):
                    # Find signal value at event time
                    closest_idx = np.argmin(np.abs(time_array - event_time))
                    features[f'{event_name}_signal_value'] = signal[closest_idx]
                    
                    # Time relative to signal peak
                    max_idx = np.nanargmax(signal)
                    features[f'{event_name}_relative_to_peak'] = event_time - time_array[max_idx]
        
        return features
    
    def extract_statistical_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical features from signal.
        
        Args:
            signal: Input signal array
            
        Returns:
            Dictionary of statistical features
        """
        features = {}
        
        # Remove NaN values for statistics
        clean_signal = signal[~np.isnan(signal)]
        
        if len(clean_signal) == 0:
            warnings.warn("No valid data points for statistical analysis")
            return features
        
        # Basic statistics
        features['mean'] = np.mean(clean_signal)
        features['std'] = np.std(clean_signal)
        features['min'] = np.min(clean_signal)
        features['max'] = np.max(clean_signal)
        features['range'] = features['max'] - features['min']
        features['median'] = np.median(clean_signal)
        
        # Percentiles
        features['p25'] = np.percentile(clean_signal, 25)
        features['p75'] = np.percentile(clean_signal, 75)
        features['iqr'] = features['p75'] - features['p25']
        
        # Shape characteristics
        features['skewness'] = skew(clean_signal)
        features['kurtosis'] = kurtosis(clean_signal)
        
        # Variability measures
        features['coefficient_of_variation'] = features['std'] / abs(features['mean']) if features['mean'] != 0 else np.inf
        features['relative_range'] = features['range'] / abs(features['mean']) if features['mean'] != 0 else np.inf
        
        # Signal energy and power
        features['rms'] = np.sqrt(np.mean(clean_signal**2))
        features['signal_energy'] = np.sum(clean_signal**2)
        features['signal_power'] = features['signal_energy'] / len(clean_signal)
        
        return features
    
    def extract_frequency_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency domain features from signal.
        
        Args:
            signal: Input signal array
            
        Returns:
            Dictionary of frequency features
        """
        features = {}
        
        # Remove NaN values
        clean_signal = signal[~np.isnan(signal)]
        
        if len(clean_signal) < 8:  # Need minimum length for FFT
            warnings.warn("Signal too short for frequency analysis")
            return features
        
        try:
            # Compute FFT
            fft_signal = np.fft.fft(clean_signal)
            freqs = np.fft.fftfreq(len(clean_signal), 1/self.sampling_rate)
            
            # Power spectral density
            psd = np.abs(fft_signal)**2
            
            # Consider only positive frequencies
            pos_mask = freqs > 0
            freqs_pos = freqs[pos_mask]
            psd_pos = psd[pos_mask]
            
            if len(psd_pos) > 0:
                # Dominant frequency
                dominant_freq_idx = np.argmax(psd_pos)
                features['dominant_frequency'] = freqs_pos[dominant_freq_idx]
                features['dominant_power'] = psd_pos[dominant_freq_idx]
                
                # Spectral centroid (weighted mean frequency)
                features['spectral_centroid'] = np.sum(freqs_pos * psd_pos) / np.sum(psd_pos)
                
                # Spectral spread (frequency standard deviation)
                features['spectral_spread'] = np.sqrt(np.sum(((freqs_pos - features['spectral_centroid'])**2) * psd_pos) / np.sum(psd_pos))
                
                # Spectral rolloff (frequency below which 95% of energy is contained)
                cumulative_power = np.cumsum(psd_pos)
                total_power = cumulative_power[-1]
                rolloff_idx = np.where(cumulative_power >= 0.95 * total_power)[0]
                if len(rolloff_idx) > 0:
                    features['spectral_rolloff'] = freqs_pos[rolloff_idx[0]]
                
                # Band power ratios
                low_band_mask = (freqs_pos >= 0) & (freqs_pos <= 5)
                mid_band_mask = (freqs_pos > 5) & (freqs_pos <= 20) 
                high_band_mask = (freqs_pos > 20) & (freqs_pos <= 50)
                
                features['low_band_power'] = np.sum(psd_pos[low_band_mask])
                features['mid_band_power'] = np.sum(psd_pos[mid_band_mask])
                features['high_band_power'] = np.sum(psd_pos[high_band_mask])
                
                total_band_power = features['low_band_power'] + features['mid_band_power'] + features['high_band_power']
                if total_band_power > 0:
                    features['low_band_ratio'] = features['low_band_power'] / total_band_power
                    features['mid_band_ratio'] = features['mid_band_power'] / total_band_power
                    features['high_band_ratio'] = features['high_band_power'] / total_band_power
        
        except Exception as e:
            warnings.warn(f"Frequency analysis failed: {e}")
        
        return features
    
    def analyze_signal_comprehensive(self, signal: np.ndarray,
                                   time_array: np.ndarray,
                                   timing_events: Optional[Dict[str, float]] = None,
                                   signal_name: str = "signal") -> Dict[str, float]:
        """
        Perform comprehensive signal analysis.
        
        Args:
            signal: Input signal array
            time_array: Corresponding time array
            timing_events: Dictionary of timing events
            signal_name: Name prefix for features
            
        Returns:
            Dictionary of all extracted features
        """
        all_features = {}
        
        # Extract different types of features
        timing_features = self.extract_timing_features(signal, time_array, timing_events)
        statistical_features = self.extract_statistical_features(signal)
        frequency_features = self.extract_frequency_features(signal)
        
        # Add signal name prefix to all features
        for feature_dict in [timing_features, statistical_features, frequency_features]:
            for key, value in feature_dict.items():
                all_features[f"{signal_name}_{key}"] = value
        
        # Add peak analysis results
        peak_results = self.find_peaks_enhanced(signal)
        for key, value in peak_results.items():
            if isinstance(value, np.ndarray):
                if key.endswith('_heights') or key.endswith('_depths'):
                    if len(value) > 0:
                        all_features[f"{signal_name}_{key}_mean"] = np.mean(value)
                        all_features[f"{signal_name}_{key}_std"] = np.std(value) if len(value) > 1 else 0
                        all_features[f"{signal_name}_{key}_max"] = np.max(value)
                        all_features[f"{signal_name}_{key}_min"] = np.min(value)
                elif key in ['peaks', 'valleys']:
                    all_features[f"{signal_name}_num_{key}"] = len(value)
        
        return all_features


def analyze_pitch_signals(pitch_data: Dict[str, pd.DataFrame],
                         signal_columns: Optional[List[str]] = None,
                         sampling_rate: float = 360.0) -> Dict[str, Dict[str, float]]:
    """
    Analyze multiple signals from a pitch dataset.
    
    Args:
        pitch_data: Dictionary with data types and DataFrames
        signal_columns: Specific signal columns to analyze (auto-detect if None)
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary with signal names and their feature dictionaries
    """
    analyzer = SignalAnalyzer(sampling_rate)
    all_features = {}
    
    for data_type, df in pitch_data.items():
        if len(df) == 0 or 'time' not in df.columns:
            continue
        
        # Get timing events from the data
        timing_events = {}
        timing_cols = ['pkh_time', 'fp_10_time', 'fp_100_time', 'MER_time', 'BR_time', 'MIR_time']
        for col in timing_cols:
            if col in df.columns and not df[col].isna().all():
                timing_events[col] = df[col].iloc[0]
        
        # Determine which columns to analyze
        if signal_columns:
            columns_to_analyze = [col for col in signal_columns if col in df.columns]
        else:
            # Auto-detect numerical columns (excluding metadata and time)
            exclude_cols = ['session_pitch', 'time'] + timing_cols
            columns_to_analyze = [col for col in df.select_dtypes(include=[np.number]).columns 
                                if col not in exclude_cols]
        
        # Analyze each signal column
        for col in columns_to_analyze[:20]:  # Limit to first 20 columns to avoid overwhelming output
            signal = df[col].values
            time_array = df['time'].values
            
            # Skip if signal is all NaN
            if np.isnan(signal).all():
                continue
            
            feature_name = f"{data_type}_{col}"
            signal_features = analyzer.analyze_signal_comprehensive(
                signal, time_array, timing_events, feature_name
            )
            
            all_features.update(signal_features)
    
    return all_features
