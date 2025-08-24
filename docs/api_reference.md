# API Reference

## Overview

The OpenBiomechanics Baseball Injury Risk Assessment system provides a comprehensive API for biomechanical analysis and injury risk assessment. This document outlines the key classes, methods, and usage patterns.

## Core Modules

### 1. Data Loading (`src.openbiomechanics_loader`)

#### `OpenBiomechanicsLoader`
Primary class for loading POI (Point of Interest) metrics and metadata.

```python
from src.openbiomechanics_loader import OpenBiomechanicsLoader

loader = OpenBiomechanicsLoader()
data = loader.load_and_merge_data()
```

**Methods:**
- `load_metadata()` → `pd.DataFrame`: Load player and session metadata
- `load_poi_metrics()` → `pd.DataFrame`: Load biomechanical point-of-interest metrics
- `load_and_merge_data()` → `pd.DataFrame`: Combined metadata and POI data
- `get_injury_risk_variables()` → `List[str]`: List of key injury risk variables
- `get_performance_variables()` → `List[str]`: Performance-related variables
- `get_demographic_variables()` → `List[str]`: Demographic/anthropometric variables

### 2. Time-Series Loading (`src.time_series_data_loader`)

#### `TimeSeriesLoader`
Advanced loader for full signal time-series data across 6 data types.

```python
from src.time_series_data_loader import TimeSeriesLoader

loader = TimeSeriesLoader()
pitch_data = loader.get_pitch_data("1031_2")
```

**Methods:**
- `load_data_type(data_type: str)` → `pd.DataFrame`: Load specific data type
- `get_pitch_data(session_pitch: str, data_types: List[str])` → `Dict[str, pd.DataFrame]`: Get all data for a pitch
- `get_available_pitches()` → `List[str]`: List of available pitch identifiers
- `get_timing_events(session_pitch: str)` → `Dict[str, float]`: Extract timing events
- `sync_data_by_time(pitch_data: Dict)` → `Dict[str, pd.DataFrame]`: Synchronize data by time

**Data Types:**
- `landmarks`: 3D marker positions over time
- `joint_angles`: Joint angle time series
- `joint_velos`: Joint velocity time series  
- `forces_moments`: Force and moment time series
- `force_plate`: Ground reaction force time series
- `energy_flow`: Energy transfer patterns

### 3. Signal Analysis (`src.biomechanical_signal_analyzer`)

#### `SignalAnalyzer`
Advanced signal processing for biomechanical time-series data.

```python
from src.biomechanical_signal_analyzer import SignalAnalyzer

analyzer = SignalAnalyzer(sampling_rate=360.0)
features = analyzer.analyze_signal_comprehensive(signal, time_array)
```

**Methods:**
- `apply_filter(signal, filter_type, cutoff_freq)` → `np.ndarray`: Digital filtering
- `smooth_signal(signal, method, window_length)` → `np.ndarray`: Signal smoothing
- `find_peaks_enhanced(signal, peak_type)` → `Dict`: Enhanced peak detection
- `extract_timing_features(signal, time_array)` → `Dict[str, float]`: Timing-based features
- `extract_statistical_features(signal)` → `Dict[str, float]`: Statistical features
- `extract_frequency_features(signal)` → `Dict[str, float]`: Frequency domain features
- `analyze_signal_comprehensive(signal, time_array)` → `Dict[str, float]`: Complete analysis

### 4. Temporal Features (`src.temporal_feature_extractor`)

#### `TemporalFeatureExtractor`
Extract 108 temporal features for injury risk assessment.

```python
from src.temporal_feature_extractor import TemporalFeatureExtractor

extractor = TemporalFeatureExtractor()
features = extractor.extract_comprehensive_temporal_features(pitch_data)
```

**Methods:**
- `extract_phase_timing_features(pitch_data)` → `Dict[str, float]`: Phase timing analysis
- `extract_velocity_sequencing_features(pitch_data)` → `Dict[str, float]`: Kinetic chain analysis  
- `extract_force_development_features(pitch_data)` → `Dict[str, float]`: Force analysis
- `extract_movement_efficiency_features(pitch_data)` → `Dict[str, float]`: Movement quality
- `extract_asymmetry_features(pitch_data)` → `Dict[str, float]`: Bilateral asymmetries
- `extract_comprehensive_temporal_features(pitch_data)` → `Dict[str, float]`: All features

**Convenience Functions:**
```python
from src.temporal_feature_extractor import extract_pitcher_temporal_features, batch_extract_temporal_features

# Single pitcher
features = extract_pitcher_temporal_features("1031_2")

# Multiple pitchers
features_df = batch_extract_temporal_features(["1031_2", "1031_3"])
```

### 5. Advanced Risk Scoring (`src.injury_risk_scorer`)

#### `AdvancedRiskScorer` 
Multi-component injury risk assessment system.

```python
from src.injury_risk_scorer import AdvancedRiskScorer

scorer = AdvancedRiskScorer()
risk_profile = scorer.analyze_pitcher_risk("1031_2")
```

**Methods:**
- `create_elbow_stress_composite(features)` → `float`: Elbow stress score (0-100)
- `create_shoulder_stress_composite(features)` → `float`: Shoulder stress score (0-100)
- `create_kinetic_chain_efficiency_score(features)` → `float`: Efficiency score (0-100)
- `create_movement_quality_score(features)` → `float`: Movement quality (0-100)
- `create_comprehensive_risk_profile(features)` → `Dict`: Complete risk assessment
- `analyze_pitcher_risk(session_pitch)` → `Dict`: Full pitcher analysis

**Risk Profile Output:**
```python
{
    'elbow_stress_score': 78.5,           # 0-100, higher = more stress
    'shoulder_stress_score': 66.2,        # 0-100, higher = more stress  
    'kinetic_chain_efficiency': 50.5,     # 0-100, higher = more efficient
    'movement_quality': 34.0,             # 0-100, higher = better quality
    'ucl_injury_risk': 71.5,              # 0-100, overall UCL risk
    'shoulder_injury_risk': 61.2,         # 0-100, overall shoulder risk
    'overall_injury_risk': 64.6,          # 0-100, combined risk score
    'risk_category': 'High',              # Low/Moderate/High/Very High
    'session_pitch': '1031_2'
}
```

**Batch Processing:**
```python
from src.injury_risk_scorer import batch_analyze_pitcher_risks

risk_df = batch_analyze_pitcher_risks(["1031_2", "1031_3", "1097_1"])
```

### 6. Feature Selection (`src.feature_selection_engineer`)

#### `EnhancedFeatureSelector`
Intelligent feature selection and dimensionality reduction.

```python
from src.feature_selection_engineer import EnhancedFeatureSelector

selector = EnhancedFeatureSelector()
X_selected, selection_info = selector.ensemble_feature_selection(X, y)
```

**Methods:**
- `remove_low_variance_features(X, threshold)` → `Tuple`: Remove low-variance features
- `correlation_based_selection(X, correlation_threshold)` → `Tuple`: Remove correlated features
- `univariate_feature_selection(X, y, k)` → `Tuple`: Statistical feature selection
- `recursive_feature_elimination(X, y, n_features)` → `Tuple`: RFE selection
- `lasso_feature_selection(X, y, alpha)` → `Tuple`: L1 regularization selection
- `random_forest_feature_selection(X, y, n_features)` → `Tuple`: RF importance selection
- `ensemble_feature_selection(X, y, n_final_features)` → `Tuple`: Combined selection
- `dimensionality_reduction_pca(X, n_components)` → `Tuple`: PCA transformation
- `evaluate_feature_set(X, y, estimator)` → `Dict`: Cross-validation evaluation

### 7. Machine Learning Models (`src.baseline_injury_model`)

#### `BaselineInjuryRiskModel`
Machine learning models for injury risk classification.

```python
from src.baseline_injury_model import BaselineInjuryRiskModel

model = BaselineInjuryRiskModel()
X, y = model.prepare_features_and_target(data, feature_columns, 'injury_risk_label')
results = model.train_baseline_models(X, y)
```

**Methods:**
- `prepare_features_and_target(data, feature_columns, target_column)` → `Tuple`: Data preparation
- `train_baseline_models(X, y)` → `Dict`: Train multiple models (LogReg, RF, GBM)
- `plot_model_comparison()`: Visualize model performance
- `plot_feature_importance(model_name, top_n)`: Plot feature importance
- `generate_model_report()` → `str`: Performance report
- `save_results(output_dir)`: Save models and results

### 8. Advanced Models (`src.advanced_models`)

#### `AdvancedInjuryRiskModel`
Advanced ensemble models and neural networks for injury risk assessment.

```python
from src.advanced_models import AdvancedInjuryRiskModel

model = AdvancedInjuryRiskModel()
results = model.train_advanced_models(X, y)
```

**Methods:**
- `train_advanced_models(X, y)` → `Dict`: Train ensemble and neural network models
- `hyperparameter_tuning(X, y)` → `Dict`: Optimize model parameters
- `evaluate_ensemble_performance(X, y)` → `Dict`: Comprehensive model evaluation
- `save_advanced_results(output_dir)`: Save all models and results

### 9. Model Interpretability (`src.injury_risk_explainer`)

#### `InjuryRiskExplainer`
SHAP analysis and feature importance for model interpretability.

```python
from src.injury_risk_explainer import InjuryRiskExplainer

explainer = InjuryRiskExplainer()
shap_values = explainer.explain_model(model, X)
```

**Methods:**
- `explain_model(model, X)` → `np.ndarray`: Generate SHAP values
- `analyze_feature_importance(model, X, y)` → `Dict`: Multi-method feature importance
- `identify_risk_factors(model, X, y)` → `Dict`: Clinical risk factor analysis

## Usage Examples

### Complete Pitcher Assessment

```python
from src.time_series_data_loader import TimeSeriesLoader
from src.temporal_feature_extractor import extract_pitcher_temporal_features
from src.injury_risk_scorer import AdvancedRiskScorer

# Load data
loader = TimeSeriesLoader()
available_pitches = loader.get_available_pitches()

# Analyze first pitcher
session_pitch = available_pitches[0]

# Extract temporal features (108 features)
temporal_features = extract_pitcher_temporal_features(session_pitch)

# Assess injury risk
scorer = AdvancedRiskScorer()
risk_profile = scorer.analyze_pitcher_risk(session_pitch)

# Display results
print(f"Pitcher: {session_pitch}")
print(f"Overall Risk: {risk_profile['overall_injury_risk']:.1f}/100")
print(f"Risk Category: {risk_profile['risk_category']}")
print(f"UCL Risk: {risk_profile['ucl_injury_risk']:.1f}/100")
print(f"Shoulder Risk: {risk_profile['shoulder_injury_risk']:.1f}/100")
```

### Batch Processing Multiple Pitchers

```python
from src.temporal_feature_extractor import batch_extract_temporal_features
from src.injury_risk_scorer import batch_analyze_pitcher_risks

# Get pitcher list
pitcher_ids = ["1031_2", "1031_3", "1097_1", "1097_2", "1097_3"]

# Extract features for all pitchers
print("Extracting temporal features...")
temporal_df = batch_extract_temporal_features(pitcher_ids, max_pitches=5)

# Assess risks for all pitchers  
print("Assessing injury risks...")
risk_df = batch_analyze_pitcher_risks(pitcher_ids, max_pitches=5)

# Analyze results
print(f"\nProcessed {len(risk_df)} pitchers")
print(f"Average overall risk: {risk_df['overall_injury_risk'].mean():.1f}/100")

# Risk distribution
risk_counts = risk_df['risk_category'].value_counts()
for category, count in risk_counts.items():
    print(f"{category} Risk: {count} pitcher(s)")
```

### Custom Feature Engineering Pipeline

```python
from src.feature_selection_engineer import create_enhanced_feature_dataset, comprehensive_feature_selection_pipeline

# Create enhanced dataset with all features
print("Creating enhanced feature dataset...")
results = comprehensive_feature_selection_pipeline(max_pitches=20, n_final_features=40)

# Extract results
X_original = results['original_features']
X_selected = results['selected_features']
y = results['target']
baseline_eval = results['baseline_evaluation']
selected_eval = results['selected_evaluation']

print(f"Original features: {X_original.shape[1]}")
print(f"Selected features: {X_selected.shape[1]}")
print(f"Baseline AUC: {baseline_eval['mean_auc']:.3f}")
print(f"Enhanced AUC: {selected_eval['mean_auc']:.3f}")
```

### Signal Analysis Example

```python
from src.time_series_data_loader import load_sample_pitch_data
from src.biomechanical_signal_analyzer import SignalAnalyzer

# Load sample data
session_pitch, pitch_data = load_sample_pitch_data()

# Initialize analyzer
analyzer = SignalAnalyzer(sampling_rate=360.0)

# Analyze elbow position signal
landmarks_df = pitch_data['landmarks']
elbow_signal = landmarks_df['elbow_jc_x'].values
time_array = landmarks_df['time'].values

# Get timing events
timing_events = {
    'MER_time': landmarks_df['MER_time'].iloc[0],
    'BR_time': landmarks_df['BR_time'].iloc[0]
}

# Comprehensive signal analysis
features = analyzer.analyze_signal_comprehensive(
    elbow_signal, time_array, timing_events, "elbow_x"
)

print(f"Extracted {len(features)} signal features")
print(f"Signal duration: {features['elbow_x_duration']:.3f}s")
print(f"Peak value: {features['elbow_x_max']:.3f}")
print(f"Number of peaks: {features['elbow_x_num_peaks']}")
```

## Error Handling

All API functions include comprehensive error handling:

```python
try:
    risk_profile = scorer.analyze_pitcher_risk("invalid_id")
except ValueError as e:
    print(f"Invalid pitcher ID: {e}")
except Exception as e:
    print(f"Analysis failed: {e}")
```

## Performance Considerations

- **Memory**: Large datasets may require 8-16GB RAM
- **Processing**: ~20 seconds per pitcher for complete analysis
- **Batch Size**: Process 10-50 pitchers at once for optimal performance
- **Caching**: Data is cached automatically for faster repeated access

## Return Value Types

- **Features**: `Dict[str, float]` - Feature name to value mapping
- **Risk Profiles**: `Dict[str, Union[float, str]]` - Mixed risk metrics
- **DataFrames**: Standard pandas DataFrames for batch operations
- **Arrays**: NumPy arrays for signal data

## Version Compatibility

- **Python**: 3.8+
- **NumPy**: 1.19+
- **Pandas**: 1.3+
- **Scikit-learn**: 1.0+
- **SciPy**: 1.7+
