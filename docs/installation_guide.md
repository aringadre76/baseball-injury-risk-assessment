# Installation Guide

## System Requirements

### Minimum Requirements
- **Operating System**: Linux, macOS, or Windows 10+
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space for data and models
- **Processor**: Multi-core CPU (4+ cores recommended)

### Recommended Requirements
- **RAM**: 32GB for large-scale batch processing
- **GPU**: NVIDIA GPU with CUDA support (optional, for future deep learning)
- **Storage**: SSD storage for faster data loading
- **Processor**: Intel i7/i9 or AMD Ryzen 7/9

## Quick Installation

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd baseball-research
```

### 2. Set Up Python Environment
```bash
# Using conda (recommended)
conda create -n baseball-research python=3.9
conda activate baseball-research

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Initialize Data
```bash
git submodule update --init --recursive
```

### 5. Verify Installation
```bash
python demos/baseline_model_demo.py
```

## Detailed Installation

### Step 1: System Prerequisites

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip git
```

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Git
brew install python git
```

#### Windows
1. Download Python from [python.org](https://python.org)
2. Install Git from [git-scm.com](https://git-scm.com)
3. Enable WSL2 for best performance (optional)

### Step 2: GPU Support (Optional)

For future deep learning capabilities:

#### NVIDIA GPU Setup
```bash
# Install CUDA toolkit (Linux)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Development Environment

#### Jupyter Notebook Setup
```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name=baseball-research
jupyter notebook
```

#### IDE Configuration
For VS Code:
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.testing.pytestEnabled": true,
    "python.linting.enabled": true,
    "python.formatting.provider": "black"
}
```

### Step 4: Data Verification

Verify data integrity:
```bash
python -c "
from src.openbiomechanics_loader import OpenBiomechanicsLoader
loader = OpenBiomechanicsLoader()
data = loader.load_and_merge_data()
print(f'✓ Loaded {len(data)} pitches from {data[\"user\"].nunique()} pitchers')
"
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or add to ~/.bashrc
echo 'export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"' >> ~/.bashrc
```

#### 2. Memory Issues
```bash
# For large datasets, adjust memory limits
export PYTHONHASHSEED=0
ulimit -v 16777216  # Limit virtual memory to 16GB
```

#### 3. Data Loading Errors
```bash
# Verify submodule initialization
git submodule status
git submodule update --recursive --remote
```

#### 4. Permission Issues (Linux/macOS)
```bash
chmod +x tests/run_comprehensive_tests.sh
```

### Performance Optimization

#### 1. NumPy/SciPy Optimization
```bash
# Install optimized BLAS libraries
sudo apt install libopenblas-dev  # Ubuntu
brew install openblas             # macOS

# Verify optimization
python -c "import numpy; numpy.show_config()"
```

#### 2. Pandas Optimization
```bash
# Enable faster CSV reading
pip install pyarrow fastparquet
```

#### 3. Memory Management
```python
# In Python scripts, use memory-efficient loading
import gc
gc.collect()  # Force garbage collection
```

## Docker Installation (Alternative)

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN git submodule update --init --recursive

CMD ["python", "demos/baseline_model_demo.py"]
```

### Build and Run
```bash
docker build -t baseball-research .
docker run -v $(pwd)/results:/app/results baseball-research
```

## Testing Installation

### Quick Test
```bash
python -c "
import sys
sys.path.append('src')
from src import *
print('✓ All modules imported successfully')
"
```

### Full Test Suite
```bash
./tests/run_comprehensive_tests.sh
```

### Performance Benchmark
```bash
python -c "
import time
from src.time_series_data_loader import load_sample_pitch_data

start = time.time()
pitch, data = load_sample_pitch_data()
duration = time.time() - start

print(f'Data loading: {duration:.2f}s')
if duration < 10:
    print('✓ Performance: Excellent')
elif duration < 30:
    print('✓ Performance: Good')
else:
    print('⚠ Performance: Consider SSD storage')
"
```

## Next Steps

After successful installation:

1. **Run demos**: Start with `demos/baseline_model_demo.py`
2. **Explore features**: Try `demos/feature_engineering_demo.py`
3. **Advanced models**: Test `demos/advanced_models_demo.py`
4. **Read documentation**: Review `docs/project_implementation_plan.md`
5. **Run tests**: Execute comprehensive test suite
6. **Try API**: Experiment with the risk assessment API

## Support

If you encounter issues:

1. **Check logs**: Review error messages carefully
2. **Update dependencies**: `pip install -r requirements.txt --upgrade`
3. **Clear cache**: Remove `__pycache__` directories
4. **Restart environment**: Deactivate and reactivate virtual environment
5. **Open issue**: Create GitHub issue with error details

## Hardware Recommendations

### For Development
- **CPU**: Intel i5/AMD Ryzen 5 or better
- **RAM**: 16GB
- **Storage**: 256GB SSD
- **OS**: Linux Ubuntu 20.04+ (preferred)

### For Production
- **CPU**: Intel i7/AMD Ryzen 7 or better  
- **RAM**: 32GB+
- **Storage**: 1TB NVMe SSD
- **GPU**: NVIDIA RTX 3070+ (future deep learning)
- **OS**: Linux server distribution
