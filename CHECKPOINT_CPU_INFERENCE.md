# PPO Checkpoint Loading for CPU Inference

This guide explains how to load PPO checkpoints that were trained on GPU and run inference on CPU, handling device compatibility issues.

## Key Concepts

When working with JAX-based models like Brax PPO, device compatibility is important when moving models between different hardware:

1. **Device Placement**: Parameters and computations need to be explicitly placed on the target device
2. **JIT Compilation**: Functions should be compiled for the target backend
3. **Checkpoint Loading**: Checkpoint data may be associated with specific devices

## Implementation Details

We provide two versions of the interactive pendulum demo:

1. **interactive_pendulum_demo_gpu.py**: Original version that runs on GPU (if available)
2. **interactive_pendulum_demo_cpu.py**: CPU-only version that loads GPU-trained checkpoints for inference

### CPU Version Features

The CPU version (`interactive_pendulum_demo_cpu.py`) includes:

1. **Environment Setup**: Forces JAX to use CPU platform
2. **Checkpoint Loading**: Uses `ppo_checkpoint.load()` to load parameters
3. **Device Placement**: Explicitly moves parameters to CPU devices
4. **Inference Execution**: Ensures all computations happen on CPU

### Key Code Patterns

#### 1. Environment Setup for CPU
```python
# 确保在CPU上运行，避免CUDA错误
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
# 禁用所有CUDA相关功能
os.environ['JAX_ENABLE_X64'] = 'false'
os.environ['JAX_ENABLE_CUDA'] = 'false'
os.environ['JAX_ENABLE_TPU'] = 'false'
```

#### 2. Parameter Loading and Device Placement
```python
# 使用ppo_checkpoint加载检查点
self.params = ppo_checkpoint.load(self.checkpoint_path)

# 确保参数在CPU上
cpu_device = jax.devices("cpu")[0]
if hasattr(self, 'normalizer_params'):
    self.normalizer_params = jax.device_put(self.normalizer_params, cpu_device)
    self.policy_params = jax.device_put(self.policy_params, cpu_device)
else:
    self.policy_params = jax.device_put(self.policy_params, cpu_device)
```

#### 3. Inference Execution with CPU Compatibility
```python
# 确保参数在CPU上
cpu_device = jax.devices("cpu")[0]
if hasattr(self, 'normalizer_params'):
    # 确保参数在CPU上
    normalizer_params_cpu = jax.device_put(self.normalizer_params, cpu_device)
    policy_params_cpu = jax.device_put(self.policy_params, cpu_device)
    policy_params = (normalizer_params_cpu, policy_params_cpu)
else:
    policy_params = jax.device_put(self.policy_params, cpu_device)

# 确保观测数据在CPU上
obs_cpu = jax.device_put(obs_jax, cpu_device)
```

## Usage Examples

### Run GPU version (if CUDA is available):
```bash
python interactive_pendulum_demo_gpu.py
```

### Run CPU version (works on any machine):
```bash
python interactive_pendulum_demo_cpu.py
```

## Key Techniques for Device Compatibility

1. **Environment Variables**: Set JAX_PLATFORM_NAME to 'cpu' to force CPU usage
2. **Explicit Device Placement**: Use `jax.device_put()` to move data to target devices
3. **Attribute Access**: Use `.mean` and `.std` attributes instead of dictionary access for RunningStatisticsState
4. **Graceful Degradation**: Provide fallback mechanisms for inference failures

## Troubleshooting

If you encounter device placement errors:

1. Ensure JAX can see your CPU: `print(jax.devices())`
2. Check that the checkpoint path is correct
3. Verify that the environment name matches what was used during training
4. Confirm that all parameter accesses use attribute syntax for RunningStatisticsState objects

## File Structure

- `interactive_pendulum_demo_gpu.py`: Original GPU-capable version
- `interactive_pendulum_demo_cpu.py`: CPU-only version with explicit device management
- `load_checkpoint_cpu.py`: Example code for loading checkpoints on CPU
- `sim_to_sim_demo.py`: Another example of CPU inference from GPU-trained models