# Development Guide

## Development Setup

### Prerequisites
- Python 3.8+
- Git
- Virtual environment tool (conda/venv)

### Quick Setup
```bash
git clone <repo-url>
cd baseball-research
python -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

## Project Architecture

### Module Organization
```
src/
├── openbiomechanics_loader.py      # POI data loading
├── time_series_data_loader.py      # Time-series data handling  
├── biomechanical_signal_analyzer.py # Signal processing
├── temporal_feature_extractor.py   # Advanced feature extraction
├── biomechanical_feature_engineer.py # Traditional features
├── injury_risk_scorer.py           # Risk assessment
├── feature_selection_engineer.py   # ML optimization
├── baseline_injury_model.py        # Machine learning models
├── advanced_models.py              # Advanced ensemble models
└── injury_risk_explainer.py       # Model interpretability
```

### Design Principles
1. **Modular Design**: Each module has a single responsibility
2. **Error Handling**: Comprehensive exception handling throughout
3. **Documentation**: Docstrings for all public methods
4. **Testing**: Unit tests for all components
5. **Performance**: Optimized for production use

## Development Workflow

### 1. Feature Development
```bash
# Create feature branch
git checkout -b feature/new-feature

# Develop and test
python tests/unit_tests/test_new_feature.py

# Run full test suite
./tests/run_comprehensive_tests.sh
```

### 2. Code Quality Standards

#### Python Style
- **PEP 8** compliance
- **Type hints** for function signatures
- **Docstrings** for all public methods
- **Maximum line length**: 100 characters

#### Example:
```python
def extract_temporal_features(
    pitch_data: Dict[str, pd.DataFrame], 
    feature_type: str = "comprehensive"
) -> Dict[str, float]:
    """
    Extract temporal features from pitch data.
    
    Args:
        pitch_data: Dictionary with data types and DataFrames
        feature_type: Type of features to extract
        
    Returns:
        Dictionary of extracted features
        
    Raises:
        ValueError: If pitch_data is empty or invalid
    """
```

#### Code Formatting
```bash
# Install formatters
pip install black flake8 isort

# Format code
black src/
isort src/
flake8 src/
```

### 3. Testing Standards

#### Unit Tests
- **Coverage**: Aim for >90% code coverage
- **Isolation**: Each test should be independent
- **Assertions**: Clear, descriptive assertions
- **Data**: Use fixtures for test data

#### Example Test:
```python
import pytest
from src.temporal_feature_extractor import TemporalFeatureExtractor

class TestTemporalFeatureExtractor:
    def setup_method(self):
        self.extractor = TemporalFeatureExtractor()
        self.sample_data = self.create_sample_data()
    
    def test_phase_timing_extraction(self):
        features = self.extractor.extract_phase_timing_features(self.sample_data)
        
        assert isinstance(features, dict)
        assert len(features) > 0
        assert 'total_pitch_duration' in features
        assert features['total_pitch_duration'] > 0
```

#### Integration Tests
```bash
# Run comprehensive test suite
./tests/run_comprehensive_tests.sh

# Expected output:
# All components tested
# Performance benchmarks passed  
# Integration tests successful
```

## Adding New Features

### 1. New Temporal Feature
```python
# In src/temporal_feature_extractor.py
def extract_new_feature_type(self, pitch_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Extract new type of temporal features."""
    features = {}
    
    # Implementation here
    
    return features

# Add to comprehensive extraction
def extract_comprehensive_temporal_features(self, pitch_data):
    # ... existing code ...
    new_features = self.extract_new_feature_type(pitch_data)
    all_features.update(new_features)
```

### 2. New Risk Component
```python
# In src/injury_risk_scorer.py
def create_new_risk_component(self, features: Dict[str, float]) -> float:
    """Create new risk assessment component."""
    # Implementation
    return risk_score

# Add to comprehensive profile
def create_comprehensive_risk_profile(self, features):
    # ... existing code ...
    risk_profile['new_risk_component'] = self.create_new_risk_component(features)
```

### 3. New Model Type
```python
# In src/baseline_injury_model.py or new module
from sklearn.ensemble import NewModel

def train_new_model(self, X, y):
    """Train new type of model."""
    model = NewModel(random_state=self.random_state)
    model.fit(X, y)
    return model
```

## Performance Optimization

### 1. Memory Management
```python
import gc

def process_large_dataset(data):
    # Process in chunks
    chunk_size = 1000
    for chunk in data.groupby(data.index // chunk_size):
        process_chunk(chunk)
        gc.collect()  # Force garbage collection
```

### 2. Vectorization
```python
# Avoid loops, use vectorized operations
# Bad:
results = []
for row in df.iterrows():
    results.append(compute_feature(row))

# Good:
results = df.apply(compute_feature, axis=1)

# Better:
results = vectorized_compute_feature(df.values)
```

### 3. Caching
```python
from functools import lru_cache

class DataLoader:
    @lru_cache(maxsize=128)
    def load_expensive_data(self, session_pitch: str):
        # Expensive operation
        return data
```

## Debugging Guidelines

### 1. Logging
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_data(data):
    logger.info(f"Processing {len(data)} records")
    # ... processing ...
    logger.info("Processing complete")
```

### 2. Debugging Tools
```python
# Interactive debugging
import pdb; pdb.set_trace()

# Rich debugging output
from rich import print as rprint
rprint(complex_data_structure)

# Memory profiling
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Function implementation
    pass
```

### 3. Performance Profiling
```python
import cProfile
import pstats

# Profile function
cProfile.run('expensive_function()', 'profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(10)
```

## Data Handling Best Practices

### 1. Data Validation
```python
def validate_pitch_data(pitch_data: Dict[str, pd.DataFrame]) -> bool:
    """Validate pitch data structure and content."""
    required_types = ['landmarks', 'joint_angles', 'joint_velos']
    
    for data_type in required_types:
        if data_type not in pitch_data:
            raise ValueError(f"Missing required data type: {data_type}")
        
        if pitch_data[data_type].empty:
            raise ValueError(f"Empty data for type: {data_type}")
    
    return True
```

### 2. Error Recovery
```python
def robust_feature_extraction(pitch_data):
    """Extract features with graceful error handling."""
    features = {}
    
    try:
        features.update(extract_timing_features(pitch_data))
    except Exception as e:
        logger.warning(f"Timing feature extraction failed: {e}")
    
    try:
        features.update(extract_force_features(pitch_data))
    except Exception as e:
        logger.warning(f"Force feature extraction failed: {e}")
    
    return features
```

### 3. Data Type Safety
```python
from typing import Union, Optional

def safe_division(
    numerator: Union[int, float], 
    denominator: Union[int, float],
    default: float = 0.0
) -> float:
    """Safely divide with fallback value."""
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator
```

## Contributing Guidelines

### 1. Pull Request Process
1. **Fork** the repository
2. **Create** feature branch from main
3. **Implement** changes with tests
4. **Run** full test suite
5. **Update** documentation if needed
6. **Submit** pull request with description

### 2. Commit Message Format
```
type(scope): short description

Longer explanation if needed.

- Bullet points for details
- Reference issues: #123

Types: feat, fix, docs, style, refactor, test, chore
```

### 3. Code Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated for new features
- [ ] Documentation updated
- [ ] No performance regressions
- [ ] Error handling appropriate
- [ ] Type hints provided

## Release Process

### 1. Version Numbering
- **Major**: Breaking API changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes

### 2. Release Checklist
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Performance benchmarks run
- [ ] CHANGELOG.md updated
- [ ] Version numbers updated

### 3. Deployment
```bash
# Tag release
git tag -a v1.2.0 -m "Release version 1.2.0"

# Update README badges
# Deploy to production environment
# Notify users of new release
```

## Tools and Dependencies

### Development Tools
```bash
# Code quality
pip install black flake8 isort mypy

# Testing
pip install pytest pytest-cov

# Documentation
pip install sphinx sphinx-rtd-theme

# Profiling
pip install memory-profiler line-profiler

# Rich output
pip install rich tqdm
```

### IDE Configuration

#### VS Code Settings
```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/unit_tests"],
    "files.associations": {
        "*.py": "python"
    }
}
```

#### PyCharm Settings
- Enable type checking
- Configure code style to PEP 8
- Set up pytest as test runner
- Enable Git integration

## Common Issues and Solutions

### 1. Import Errors
```bash
# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or use relative imports
from .openbiomechanics_loader import OpenBiomechanicsLoader
```

### 2. Memory Issues
```python
# Process data in chunks
def process_large_dataset(data, chunk_size=1000):
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        yield process_chunk(chunk)
```

### 3. Performance Issues
```python
# Use vectorized operations
import numpy as np

# Slow
result = [x**2 for x in data]

# Fast  
result = np.array(data) ** 2
```

## Resources

### Documentation
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [SciPy Documentation](https://docs.scipy.org/)

### Best Practices
- [Python Enhancement Proposals](https://www.python.org/dev/peps/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Real Python Best Practices](https://realpython.com/)

### Testing
- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

### Performance
- [Python Performance Tips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
- [Profiling Python Code](https://docs.python.org/3/library/profile.html)
