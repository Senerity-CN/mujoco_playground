#!/usr/bin/env python3
"""
验证修复后的ONNX模型与JAX模型的一致性
"""

import sys
sys.path.append('/home/balance/fork_mujoco_playground/mujoco_playground')

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from orbax import checkpoint as ocp
import onnxruntime as ort

def compare_models():
    """比较修复后的模型输出一致性"""
    checkpoint_path = "/home/balance/fork_mujoco_playground/mujoco_playground/learning/logs/PendulumSwingup-20250811-155911/checkpoints/000507248640"
    onnx_model_path = "pendulum_policy.onnx"
    
    print("🔬 验证修复后的ONNX模型输出一致性...")
    
    # 加载JAX模型
    print("🔄 加载JAX模型...")
    network_factory_kwargs = {
        "policy_hidden_layer_sizes": [32, 32, 32, 32],
        "value_hidden_layer_sizes": [256, 256, 256, 256, 256],
        "policy_obs_key": "state",
        "value_obs_key": "state"
    }
    
    network = ppo_networks.make_ppo_networks(
        observation_size=4,
        action_size=1,
        **network_factory_kwargs
    )
    
    checkpointer = ocp.PyTreeCheckpointer()
    params = checkpointer.restore(checkpoint_path)
    
    # 提取参数
    if isinstance(params, dict):
        if 'policy' in params and 'normalizer_params' in params:
            policy_params = params['policy']
            normalizer_params = params['normalizer_params']
        else:
            policy_params = params.get('policy', params.get('params', params))
            normalizer_params = params.get('normalizer_params', None)
    else:
        if isinstance(params, (list, tuple)) and len(params) >= 2:
            normalizer_params = params[0]
            policy_params = params[1]
        else:
            policy_params = params
            normalizer_params = None
    
    # 提取归一化参数
    if normalizer_params is not None:
        if isinstance(normalizer_params, dict):
            if 'mean' in normalizer_params:
                if isinstance(normalizer_params['mean'], dict) and 'state' in normalizer_params['mean']:
                    mean = np.array(normalizer_params['mean']['state'])
                else:
                    mean = np.array(normalizer_params['mean'])
            else:
                mean = np.zeros(4)
                
            if 'std' in normalizer_params:
                if isinstance(normalizer_params['std'], dict) and 'state' in normalizer_params['std']:
                    std = np.array(normalizer_params['std']['state'])
                else:
                    std = np.array(normalizer_params['std'])
            else:
                std = np.ones(4)
        else:
            mean = np.array(getattr(normalizer_params, 'mean', np.zeros(4)))
            std = np.array(getattr(normalizer_params, 'std', np.ones(4)))
    else:
        mean = np.zeros(4)
        std = np.ones(4)
    
    print(f"📊 JAX模型归一化参数:")
    print(f"   Mean: {mean}")
    print(f"   Std: {std}")
    
    # 创建JAX推理函数
    make_policy = ppo_networks.make_inference_fn(network)
    inference_fn = make_policy((normalizer_params, policy_params), deterministic=True)
    
    # 加载ONNX模型
    print("🔄 加载ONNX模型...")
    ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    
    # 使用相同的归一化参数
    onnx_mean = np.array([-0.02921442, -0.58093756, -0.9564421, -0.14311926], dtype=np.float32)
    onnx_std = np.array([0.47902852, 0.6574112, 6.7379885, 0.6200137], dtype=np.float32)
    
    print(f"📊 ONNX模型归一化参数:")
    print(f"   Mean: {onnx_mean}")
    print(f"   Std: {onnx_std}")
    
    # 测试关键观测值
    test_cases = [
        # [sin(θ), cos(θ), θ_dot, torque_history]
        ([0.0, -1.0, 0.0, 0.0], "顶部稳定位置"),  # 倒立
        ([0.0, 1.0, 0.0, 0.0], "底部位置"),       # 底部
        ([1.0, 0.0, 0.0, 0.0], "右侧位置"),       # 右侧
        ([-1.0, 0.0, 0.0, 0.0], "左侧位置"),      # 左侧
        ([0.0, -1.0, 1.0, 0.0], "顶部+顺时针速度"),
        ([0.0, -1.0, -1.0, 0.0], "顶部+逆时针速度"),
    ]
    
    print("\n" + "="*90)
    print("🎯 模型输出一致性验证")
    print("="*90)
    print(f"{'测试场景':<15} {'观测值':<25} {'JAX输出':<12} {'ONNX输出':<12} {'差异':<12} {'一致性'}")
    print("-"*90)
    
    total_diff = 0
    max_diff = 0
    
    for obs, description in test_cases:
        obs_array = np.array(obs, dtype=np.float32)
        
        # JAX模型推理
        obs_jax = jp.array(obs_array).reshape(1, -1)
        rng = jax.random.PRNGKey(0)
        jax_action, _ = inference_fn(obs_jax, rng)
        jax_result = float(np.array(jax_action).flatten()[0])
        
        # ONNX模型推理（使用正确的归一化）
        normalized_obs = (obs_array - onnx_mean) / (onnx_std + 1e-8)
        onnx_action = ort_session.run(None, {input_name: normalized_obs.reshape(1, -1).astype(np.float32)})[0]
        onnx_result = float(onnx_action.flatten()[0])
        
        # 计算差异
        diff = abs(jax_result - onnx_result)
        total_diff += diff
        max_diff = max(max_diff, diff)
        
        # 一致性判断（差异小于0.01认为一致）
        consistent = "✅" if diff < 0.01 else "❌"
        
        # 格式化观测值显示
        obs_str = f"[{', '.join([f'{x:4.1f}' for x in obs])}]"
        
        print(f"{description:<15} {obs_str:<25} {jax_result:<12.6f} {onnx_result:<12.6f} {diff:<12.6f} {consistent}")
    
    print("-"*90)
    avg_diff = total_diff / len(test_cases)
    print(f"📈 平均差异: {avg_diff:.6f}")
    print(f"📈 最大差异: {max_diff:.6f}")
    
    if avg_diff < 0.005:
        print("🎉 模型一致性良好！ONNX模型可以替代JAX模型使用。")
    elif avg_diff < 0.01:
        print("👌 模型一致性可接受，可用于实际应用。")
    else:
        print("⚠️  模型差异较大，请检查模型转换过程。")
    
    print("="*90)

if __name__ == "__main__":
    compare_models()