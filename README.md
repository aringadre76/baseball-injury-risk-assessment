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

### Prerequisites

- Python 3.8+
- Git
- Access to OpenBiomechanics repository

### Installation

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
├── injury_risk_assessment_plan.md  # Project implementation plan
├── requirements.txt           # Python dependencies
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
