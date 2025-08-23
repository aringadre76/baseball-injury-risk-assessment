# Docker Environment Setup

This directory contains the Driveline Baseball Science Docker setup for reproducible biomechanics research environments.

## Overview

The Driveline Baseball Science Docker environment provides a standardized, GPU-accelerated setup specifically designed for biomechanics research and analysis with OpenBiomechanics data.

## Key Features

- **NVIDIA CUDA 12.3.1**: GPU acceleration for deep learning and data processing
- **Conda Package Management**: Reproducible Python and R environments
- **Pre-installed ML Libraries**: TensorFlow, PyTorch, LightGBM, XGBoost
- **Biomechanics Tools**: Optimized for OpenBiomechanics data analysis
- **Development Tools**: Git, Vim, build-essential for development

## Quick Start

### Prerequisites
- Docker installed
- NVIDIA Docker runtime (for GPU support)
- At least 8GB RAM, 20GB disk space

### Build and Run
```bash
# Build the image
docker build -t baseball-science .

# Run with GPU support
docker run --gpus all -it -p 8888:8888 -v $(pwd):/workspace baseball-science

# Run without GPU (CPU only)
docker run -it -p 8888:8888 -v $(pwd):/workspace baseball-science
```

## Usage

### Jupyter Notebook Access
- Port 8888 is exposed for Jupyter notebooks
- Access at: http://localhost:8888
- Default password: `baseball`

### Data Mounting
- Mount your local project directory to `/workspace`
- OpenBiomechanics data will be available in the container

### GPU Verification
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

## Environment Details

### Python Packages
- Core ML: TensorFlow, PyTorch, scikit-learn
- Data Science: Pandas, NumPy, Matplotlib, Seaborn
- Biomechanics: ezc3d, pyomeca
- Computer Vision: OpenCV, Dlib

### R Packages
- Statistical analysis and visualization
- Biomechanical data processing
- Research paper generation

## Customization

### Adding Packages
```dockerfile
# In Dockerfile
RUN conda install -c conda-forge your-package-name
```

### Environment Variables
```bash
# Set in docker run command
-e CUDA_VISIBLE_DEVICES=0
-e JUPYTER_TOKEN=your-token
```

## Troubleshooting

### GPU Issues
- Ensure NVIDIA Docker runtime is installed
- Check `nvidia-smi` output
- Verify CUDA version compatibility

### Memory Issues
- Increase Docker memory limit
- Use CPU-only version for development
- Optimize data loading patterns

## References

- [Driveline Baseball Science Docker](https://github.com/deeplearningmidjourney/baseball-science-docker)
- [OpenBiomechanics Repository](https://github.com/drivelineresearch/openbiomechanics)
- [NVIDIA Docker Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
