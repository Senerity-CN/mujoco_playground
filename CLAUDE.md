# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

MuJoCo Playground is a comprehensive suite of GPU-accelerated environments for robot learning research and sim-to-real transfer, built with MuJoCo MJX. It provides:

- Classic control environments from `dm_control` reimplemented in MJX
- Quadruped and bipedal locomotion environments
- Non-prehensile and dexterous manipulation environments
- Vision-based support via Madrona-MJX

## Project Structure

- `mujoco_playground/_src/` - Core environment implementations
  - `dm_control_suite/` - Classic control environments
  - `locomotion/` - Bipedal and quadrupedal locomotion environments
  - `manipulation/` - Robotic manipulation environments
  - `mjx_env.py` - Core MJX environment classes
  - `wrapper.py` - Environment wrappers
- `mujoco_playground/config/` - Environment configuration and hyperparameters
- `learning/` - Training scripts and notebooks
- `mujoco_playground/experimental/` - Experimental features

## Development Commands

### Installation
```bash
# Install uv first (faster alternative to pip)
# From source installation:
uv venv --python 3.11
source .venv/bin/activate
uv pip install -U "jax[cuda12]"
uv pip install -e ".[all]"
```

### Building
```bash
# Package is built using hatch (configured in pyproject.toml)
# Standard Python package installation as shown above
```

### Running Tests
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=mujoco_playground
```

### Linting and Formatting
```bash
# Run pre-commit hooks (recommended)
pre-commit run --all-files

# Or run tools manually:
pyink .                    # Code formatting
ruff check .               # Linting
isort .                    # Import sorting
pylint . --rcfile=pylintrc # Detailed linting
pytype .                   # Type checking
```

### Training Agents
```bash
# Train a PPO agent on Cartpole
python learning/train_jax_ppo.py --env_name CartpoleBalance

# Train with visualization
python learning/train_jax_ppo.py --env_name PandaPickCube --rscope_envs 16 --run_evals=False --deterministic_rscope=True
```

## Key Architecture Components

1. **MjxEnv** (`mujoco_playground/_src/mjx_env.py`) - Base environment class that wraps MJX environments
2. **Registry System** (`mujoco_playground/_src/registry.py`) - Maps environment names to their configurations
3. **Environment Wrappers** (`mujoco_playground/_src/wrapper.py`) - Provide additional functionality like observation normalization
4. **Configuration System** (`mujoco_playground/config/`) - Environment-specific hyperparameters and settings

## GPU Precision Note

Users with NVIDIA Ampere architecture GPUs may experience reproducibility issues due to JAX's default use of TF32. To ensure consistent behavior, run:
```bash
export JAX_DEFAULT_MATMUL_PRECISION=highest
```