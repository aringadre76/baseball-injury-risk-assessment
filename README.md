# Baseball Injury Risk Assessment Model

A machine learning model for assessing injury risk in baseball pitchers using biomechanical data from the OpenBiomechanics repository.

## Project Overview

This project analyzes high-fidelity motion capture data from elite baseball pitchers to identify biomechanical risk factors and predict injury probability. The system combines traditional sports science with cutting-edge machine learning techniques.

## Dataset

- **Source**: [OpenBiomechanics](https://github.com/drivelineresearch/openbiomechanics)
- **Data**: 100+ pitchers, 413 pitches, high-fidelity motion capture
- **Format**: C3D files, biomechanical metrics, synchronized video
- **Access**: Open source, no licensing restrictions

## Features

- **Injury Risk Assessment**: ML model predicting injury probability
- **Biomechanical Analysis**: Comprehensive movement pattern analysis
- **Player Comparison**: Identify similar athletes and movement patterns
- **MCP Server**: Natural language query interface for AI agents
- **Computer Vision**: Future markerless motion capture capabilities

## Quick Start

### Option 1: Docker Environment (Recommended)

The easiest way to get started is using our pre-configured Docker environment based on Driveline Baseball Science Docker.

#### Prerequisites
- Docker and Docker Compose installed
- NVIDIA Docker runtime (optional, for GPU acceleration)
- At least 8GB RAM, 20GB disk space

#### Setup
1. **Clone this repository:**
   ```bash
   git clone <your-repo-url>
   cd baseball-research
   ```

2. **Download the Driveline Baseball Science Dockerfile:**
   ```bash
   # Create docker directory
   mkdir -p docker
   
   # Download Dockerfile from Driveline's repository
   # Place it in ./docker/Dockerfile
   ```

3. **Run the setup script:**
   ```bash
   ./setup-docker.sh
   ```

4. **Access Jupyter notebooks:**
   - GPU version: http://localhost:8888
   - CPU version: http://localhost:8889
   - Password: `baseball`

### Option 2: Local Environment

#### Prerequisites
- Python 3.8+
- Git
- Access to OpenBiomechanics repository

#### Installation

1. **Clone this repository:**
   ```bash
   git clone <your-repo-url>
   cd baseball-research
   ```

2. **Add OpenBiomechanics as a submodule:**
   ```bash
   git submodule add https://github.com/drivelineresearch/openbiomechanics.git
   git submodule update --init --recursive
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up development environment:**
   ```bash
   pip install jupyter ipykernel
   python -m ipykernel install --user --name=baseball-research
   ```

## Project Structure

```
baseball-research/
├── openbiomechanics/          # Data submodule (git submodule)
├── docker/                    # Docker environment setup
├── injury_risk_assessment_plan.md  # Project implementation plan
├── requirements.txt           # Python dependencies
├── docker-compose.yml        # Docker Compose configuration
├── setup-docker.sh           # Docker setup script
├── src/                      # Source code (to be implemented)
├── notebooks/                # Jupyter notebooks (to be implemented)
├── tests/                    # Test files (to be implemented)
└── README.md                 # This file
```

## Implementation Plan

The project follows an 8-week phased approach:

- **Phase 1-2**: Foundation & baseline model
- **Phase 3-4**: Advanced features & optimization
- **Phase 5**: MCP server development
- **Phase 6-8**: Computer vision & extensions

See [injury_risk_assessment_plan.md](injury_risk_assessment_plan.md) for detailed timeline and deliverables.

## Key Biomechanical Variables

- **Elbow Varus Moment**: UCL injury risk indicator
- **Shoulder Internal Rotation Moment**: Shoulder stress
- **Torso Rotational Velocity**: Core mechanics
- **Hip-Shoulder Separation**: Movement efficiency
- **Ground Reaction Forces**: Lower body stress

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. See the OpenBiomechanics repository for data licensing terms.

## Contact

For questions about this project, please open an issue on GitHub.

## Acknowledgments

- Driveline Baseball for providing the OpenBiomechanics dataset
- The sports medicine and biomechanics research community
- Contributors to the open source tools used in this project
