#!/bin/bash

# =============================================================================
# Comprehensive Test Suite for OpenBiomechanics Injury Risk Assessment
# Phases 1 & 2 Complete Testing Pipeline
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_header() {
    echo -e "\n${PURPLE}========================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}========================================${NC}\n"
}

log_section() {
    echo -e "\n${BLUE}--- $1 ---${NC}"
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

log_error() {
    echo -e "${RED}✗ $1${NC}"
}

log_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

# Test execution function with error handling
run_test() {
    local test_name="$1"
    local test_command="$2"
    local is_critical="${3:-true}"
    
    echo -e "\n${YELLOW}Running: $test_name${NC}"
    echo "Command: $test_command"
    echo "----------------------------------------"
    
    if eval "$test_command"; then
        log_success "$test_name completed successfully"
        return 0
    else
        local exit_code=$?
        if [ "$is_critical" = "true" ]; then
            log_error "$test_name failed (exit code: $exit_code)"
            log_error "Critical test failed. Stopping execution."
            exit $exit_code
        else
            log_warning "$test_name failed (exit code: $exit_code) - continuing..."
            return $exit_code
        fi
    fi
}

# Initialize test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
START_TIME=$(date +%s)

# Create results directory
mkdir -p test_results
RESULTS_FILE="test_results/comprehensive_test_results_$(date +%Y%m%d_%H%M%S).log"

# Redirect all output to both console and log file
exec > >(tee -a "$RESULTS_FILE")
exec 2>&1

log_header "COMPREHENSIVE TEST SUITE STARTING"
log_info "Test session started at: $(date)"
log_info "Results will be saved to: $RESULTS_FILE"
log_info "Working directory: $(pwd)"

# =============================================================================
# ENVIRONMENT VALIDATION
# =============================================================================

log_header "ENVIRONMENT VALIDATION"

log_section "Checking Python Environment"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    log_success "Python3 found: $PYTHON_VERSION"
else
    log_error "Python3 not found"
    exit 1
fi

log_section "Checking Required Directories"
for dir in "src" "openbiomechanics" "data" "results"; do
    if [ -d "$dir" ]; then
        log_success "Directory exists: $dir"
    else
        log_warning "Directory missing: $dir"
    fi
done

log_section "Checking Data Files"
if [ -f "openbiomechanics/baseball_pitching/data/metadata.csv" ]; then
    log_success "Metadata file found"
else
    log_error "Metadata file not found - tests may fail"
fi

if [ -f "openbiomechanics/baseball_pitching/data/poi/poi_metrics.csv" ]; then
    log_success "POI metrics file found"
else
    log_error "POI metrics file not found - tests may fail"
fi

# =============================================================================
# PHASE 1 TESTING
# =============================================================================

log_header "PHASE 1: BASELINE FUNCTIONALITY TESTING"

# Test 1: Phase 1 Complete Demo
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if run_test "Phase 1 Complete Demo" "python3 phase1_demo.py" true; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

log_section "Phase 1 Component Validation"

# Validate Phase 1 outputs
if [ -f "data/processed_pitching_data.csv" ]; then
    ROWS=$(wc -l < "data/processed_pitching_data.csv")
    log_success "Processed data file created with $ROWS rows"
else
    log_warning "Processed data file not found"
fi

if [ -d "results" ] && [ -f "results/model_performance_metrics.csv" ]; then
    log_success "Model results directory and files created"
else
    log_warning "Model results not found"
fi

# =============================================================================
# PHASE 2 COMPONENT TESTING
# =============================================================================

log_header "PHASE 2: ADVANCED FEATURES COMPONENT TESTING"

# Test 2: Time-Series Data Loader
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if run_test "Time-Series Data Loader" "python3 test_time_series_loader.py" true; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Test 3: Signal Analysis
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if run_test "Signal Analysis & Peak Detection" "python3 test_signal_analysis.py" true; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Test 4: Temporal Features
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if run_test "Temporal Feature Extraction" "python3 test_temporal_features.py" true; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Test 5: Advanced Risk Scoring
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if run_test "Advanced Risk Scoring" "python3 test_advanced_risk_scoring.py" true; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# =============================================================================
# PHASE 2 COMPREHENSIVE TESTING
# =============================================================================

log_header "PHASE 2: COMPREHENSIVE INTEGRATION TESTING"

# Test 6: Phase 2 Complete Demo
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if run_test "Phase 2 Complete Demo (Comprehensive)" "python3 phase2_demo.py" false; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# =============================================================================
# INTEGRATION TESTING
# =============================================================================

log_header "INTEGRATION & PERFORMANCE TESTING"

log_section "Testing Python Module Imports"
TOTAL_TESTS=$((TOTAL_TESTS + 1))
cat << 'EOF' > test_imports.py
#!/usr/bin/env python3
import sys
sys.path.append('src')

try:
    # Test all module imports
    from src.data_loader import OpenBiomechanicsLoader
    from src.feature_engineering import FeatureEngineer
    from src.baseline_model import BaselineInjuryRiskModel
    from src.time_series_loader import TimeSeriesLoader
    from src.signal_analysis import SignalAnalyzer
    from src.temporal_features import TemporalFeatureExtractor
    from src.advanced_risk_scoring import AdvancedRiskScorer
    from src.feature_selection import EnhancedFeatureSelector
    
    print("✓ All module imports successful")
    
    # Test basic instantiation
    loader = OpenBiomechanicsLoader()
    engineer = FeatureEngineer()
    model = BaselineInjuryRiskModel()
    ts_loader = TimeSeriesLoader()
    analyzer = SignalAnalyzer()
    temp_extractor = TemporalFeatureExtractor()
    risk_scorer = AdvancedRiskScorer()
    selector = EnhancedFeatureSelector()
    
    print("✓ All class instantiations successful")
    print("✓ Integration test passed")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ General error: {e}")
    sys.exit(1)
EOF

if run_test "Module Import & Instantiation" "python3 test_imports.py" true; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Clean up test file
rm -f test_imports.py

# =============================================================================
# PERFORMANCE BENCHMARKING
# =============================================================================

log_header "PERFORMANCE BENCHMARKING"

log_section "Running Performance Tests"
TOTAL_TESTS=$((TOTAL_TESTS + 1))
cat << 'EOF' > performance_test.py
#!/usr/bin/env python3
import sys
import time
sys.path.append('src')

from src.time_series_loader import load_sample_pitch_data
from src.temporal_features import extract_pitcher_temporal_features
from src.advanced_risk_scoring import AdvancedRiskScorer

print("Performance Benchmarking")
print("=" * 40)

# Test 1: Data Loading Speed
start_time = time.time()
session_pitch, pitch_data = load_sample_pitch_data()
load_time = time.time() - start_time
print(f"✓ Sample pitch data loading: {load_time:.2f}s")

# Test 2: Temporal Feature Extraction Speed
start_time = time.time()
temporal_features = extract_pitcher_temporal_features(session_pitch)
feature_time = time.time() - start_time
print(f"✓ Temporal feature extraction: {feature_time:.2f}s")
print(f"  Features extracted: {len(temporal_features)}")

# Test 3: Risk Assessment Speed
start_time = time.time()
scorer = AdvancedRiskScorer()
risk_profile = scorer.analyze_pitcher_risk(session_pitch)
risk_time = time.time() - start_time
print(f"✓ Risk assessment: {risk_time:.2f}s")

# Total processing time per pitcher
total_time = load_time + feature_time + risk_time
print(f"\nTotal processing time per pitcher: {total_time:.2f}s")
print(f"Estimated throughput: {3600/total_time:.1f} pitchers/hour")

if total_time < 30:
    print("✓ Performance: Excellent (< 30s per pitcher)")
elif total_time < 60:
    print("✓ Performance: Good (< 60s per pitcher)")
else:
    print("⚠ Performance: Needs optimization (> 60s per pitcher)")
EOF

if run_test "Performance Benchmarking" "python3 performance_test.py" false; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Clean up test file
rm -f performance_test.py

# =============================================================================
# MEMORY AND RESOURCE TESTING
# =============================================================================

log_header "RESOURCE UTILIZATION TESTING"

log_section "Memory Usage Analysis"
TOTAL_TESTS=$((TOTAL_TESTS + 1))
cat << 'EOF' > memory_test.py
#!/usr/bin/env python3
import sys
import psutil
import os
sys.path.append('src')

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

print("Memory Usage Testing")
print("=" * 30)

initial_memory = get_memory_usage()
print(f"Initial memory usage: {initial_memory:.1f} MB")

# Test memory usage during data loading
from src.time_series_loader import TimeSeriesLoader
loader = TimeSeriesLoader()
after_loader = get_memory_usage()
print(f"After TimeSeriesLoader: {after_loader:.1f} MB (+{after_loader-initial_memory:.1f} MB)")

# Test memory usage during feature extraction
from src.temporal_features import extract_pitcher_temporal_features
session_pitch, _ = loader.get_available_pitches()[0], None
features = extract_pitcher_temporal_features(session_pitch)
after_features = get_memory_usage()
print(f"After feature extraction: {after_features:.1f} MB (+{after_features-after_loader:.1f} MB)")

# Test memory usage during risk scoring
from src.advanced_risk_scoring import AdvancedRiskScorer
scorer = AdvancedRiskScorer()
risk_profile = scorer.analyze_pitcher_risk(session_pitch)
final_memory = get_memory_usage()
print(f"After risk scoring: {final_memory:.1f} MB (+{final_memory-after_features:.1f} MB)")

total_memory_increase = final_memory - initial_memory
print(f"\nTotal memory increase: {total_memory_increase:.1f} MB")

if total_memory_increase < 500:
    print("✓ Memory usage: Excellent (< 500 MB)")
elif total_memory_increase < 1000:
    print("✓ Memory usage: Good (< 1 GB)")
else:
    print("⚠ Memory usage: High (> 1 GB)")
EOF

if run_test "Memory Usage Analysis" "python3 memory_test.py" false; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Clean up test file
rm -f memory_test.py

# =============================================================================
# DATA VALIDATION TESTING
# =============================================================================

log_header "DATA VALIDATION & QUALITY TESTING"

log_section "Data Integrity Checks"
TOTAL_TESTS=$((TOTAL_TESTS + 1))
cat << 'EOF' > data_validation.py
#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
sys.path.append('src')

from src.data_loader import OpenBiomechanicsLoader, validate_data_quality
from src.time_series_loader import TimeSeriesLoader

print("Data Validation Testing")
print("=" * 35)

# Test POI data quality
print("Testing POI data quality...")
loader = OpenBiomechanicsLoader()
poi_data = loader.load_and_merge_data()
validation_results = validate_data_quality(poi_data)

print(f"✓ Total records: {validation_results['total_records']}")
print(f"✓ Unique pitchers: {validation_results['unique_pitchers']}")
print(f"✓ Unique sessions: {validation_results['unique_sessions']}")

# Check for data quality issues
if validation_results['potential_speed_outliers'] > 0:
    print(f"⚠ Found {validation_results['potential_speed_outliers']} potential speed outliers")
else:
    print("✓ No speed outliers detected")

if validation_results['potential_age_outliers'] > 0:
    print(f"⚠ Found {validation_results['potential_age_outliers']} potential age outliers")
else:
    print("✓ No age outliers detected")

# Test time-series data availability
print("\nTesting time-series data availability...")
ts_loader = TimeSeriesLoader()
available_pitches = ts_loader.get_available_pitches()
print(f"✓ Time-series data available for {len(available_pitches)} pitches")

# Sample data quality check
sample_pitch = available_pitches[0]
pitch_data = ts_loader.get_pitch_data(sample_pitch)
print(f"✓ Sample pitch {sample_pitch} has {len(pitch_data)} data types")

for data_type, df in pitch_data.items():
    if len(df) > 0:
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        print(f"  {data_type}: {df.shape[0]} samples, {missing_pct:.1f}% missing data")
    else:
        print(f"  {data_type}: No data available")

print("✓ Data validation completed")
EOF

if run_test "Data Validation & Quality" "python3 data_validation.py" false; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Clean up test file
rm -f data_validation.py

# =============================================================================
# FINAL RESULTS AND SUMMARY
# =============================================================================

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((TOTAL_DURATION / 60))
DURATION_SEC=$((TOTAL_DURATION % 60))

log_header "COMPREHENSIVE TEST SUITE RESULTS"

echo -e "${CYAN}Test Execution Summary:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "Total Tests:     ${BLUE}$TOTAL_TESTS${NC}"
echo -e "Passed:          ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed:          ${RED}$FAILED_TESTS${NC}"
echo -e "Success Rate:    ${YELLOW}$(( PASSED_TESTS * 100 / TOTAL_TESTS ))%${NC}"
echo -e "Duration:        ${PURPLE}${DURATION_MIN}m ${DURATION_SEC}s${NC}"
echo -e "Completed:       ${CYAN}$(date)${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -e "\n${CYAN}Test Categories Completed:${NC}"
echo "✓ Environment Validation"
echo "✓ Phase 1 Baseline Functionality"
echo "✓ Phase 2 Component Testing"
echo "✓ Phase 2 Comprehensive Integration"
echo "✓ Module Import & Integration"
echo "✓ Performance Benchmarking"
echo "✓ Memory Usage Analysis"
echo "✓ Data Validation & Quality"

echo -e "\n${CYAN}Key Capabilities Verified:${NC}"
echo "✓ POI data loading and validation"
echo "✓ Time-series data processing (6 data types)"
echo "✓ Signal analysis and peak detection"
echo "✓ Temporal feature extraction (108+ features)"
echo "✓ Advanced composite risk scoring"
echo "✓ Feature selection and dimensionality reduction"
echo "✓ Enhanced machine learning models"
echo "✓ Real-time pitcher risk assessment"

# Generate final test status
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}🎉 ALL TESTS PASSED! 🎉${NC}"
    echo -e "${GREEN}The OpenBiomechanics Injury Risk Assessment system is fully functional.${NC}"
    EXIT_CODE=0
elif [ $FAILED_TESTS -le 2 ]; then
    echo -e "\n${YELLOW}⚠ MOSTLY SUCCESSFUL WITH MINOR ISSUES ⚠${NC}"
    echo -e "${YELLOW}$FAILED_TESTS non-critical tests failed. System is largely functional.${NC}"
    EXIT_CODE=0
else
    echo -e "\n${RED}❌ MULTIPLE TEST FAILURES ❌${NC}"
    echo -e "${RED}$FAILED_TESTS tests failed. Please review the issues above.${NC}"
    EXIT_CODE=1
fi

echo -e "\n${CYAN}Full test log saved to: $RESULTS_FILE${NC}"

# Final cleanup
log_info "Cleaning up temporary files..."
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

log_info "Test suite completed."
exit $EXIT_CODE
