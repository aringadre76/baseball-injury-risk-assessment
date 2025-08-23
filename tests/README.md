# Tests

This directory contains test files for the project components.

## Test Structure

- `test_data_loader.py` - Test data loading functionality
- `test_feature_engineering.py` - Test feature creation
- `test_model.py` - Test injury risk model
- `test_mcp_server.py` - Test MCP server functionality

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_data_loader.py

# Run with coverage
pytest --cov=src
```
