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
  - `dm_control_suite/` - Classic control environments (Pendulum, Cartpole, Cheetah, etc.)
  - `locomotion/` - Bipedal and quadrupedal locomotion environments (Go1, Spot, H1, etc.)
  - `manipulation/` - Robotic manipulation environments (Franka, ALOHA, Leap Hand, etc.)
  - `mjx_env.py` - Core MJX environment classes and utilities
  - `registry.py` - Environment registry system that maps names to implementations
  - `wrapper.py` - Environment wrappers for training and domain randomization
- `mujoco_playground/config/` - Environment configuration and hyperparameters for RL training
- `learning/` - Training scripts and notebooks for PPO, SAC, and other algorithms
- `mujoco_playground/experimental/` - Experimental features, benchmarking, and sim2sim tools

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

Verify installation:
```bash
python -c "import mujoco_playground"
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

# Run tests in parallel
pytest -n auto
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
# Train a PPO agent on Pendulum
python learning/train_jax_ppo.py --env_name PendulumSwingup

# Train with custom parameters
python learning/train_jax_ppo.py --env_name PandaPickCube --num_envs 2048 --learning_rate 3e-4

# Train with visualization using rscope
python learning/train_jax_ppo.py --env_name Go1Walk --rscope_envs 16 --run_evals=False --deterministic_rscope=True

# Resume training from checkpoint
python learning/train_jax_ppo.py --env_name LeapCubeReorient --load_checkpoint_path /path/to/checkpoint
```

### Interactive Demos
```bash
# Run interactive pendulum demo with trained model
python interactive_pendulum_demo.py
```

## Key Architecture Components

1. **MjxEnv** (`mujoco_playground/_src/mjx_env.py`) - Base environment class that wraps MJX environments with methods for reset, step, and observation handling.

2. **Registry System** (`mujoco_playground/_src/registry.py`) - Central registry that maps environment names to their implementations and configurations. Used by training scripts to load environments.

3. **Environment Wrappers** (`mujoco_playground/_src/wrapper.py`) - Provide additional functionality like episode management, auto-reset, domain randomization, and Brax compatibility.

4. **Configuration System** (`mujoco_playground/config/`) - Contains tuned hyperparameters for RL algorithms (PPO, SAC) for different environments, separating environment config from training config.

5. **Training Pipeline** (`learning/train_jax_ppo.py`) - Main training script that integrates with Brax PPO implementation, handles checkpointing, logging, and evaluation.

## Environment Development Patterns

### Creating New Environments
New environments should inherit from `mjx_env.MjxEnv` and implement:
- `reset()` - Initialize environment state
- `step()` - Execute one timestep
- Properties: `xml_path`, `action_size`, `mj_model`, `mjx_model`

### Configuration System
Environments use ML Collections ConfigDict for configuration:
- Environment-specific defaults in `*_params.py` files
- Separation of environment config (timestep, episode length) from RL config (learning rate, batch size)

### Observation Design
- Observations should be normalized/scaled appropriately
- Include relevant state information for the task
- Handle sensor noise for sim-to-real transfer

## GPU Precision Note

Users with NVIDIA Ampere architecture GPUs may experience reproducibility issues due to JAX's default use of TF32. To ensure consistent behavior, run:
```bash
export JAX_DEFAULT_MATMUL_PRECISION=highest
```