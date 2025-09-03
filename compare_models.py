#!/usr/bin/env python3
"""
比较JAX模型和ONNX模型的输出差异
"""

import sys
sys.path.append('/home/balance/fork_mujoco_playground/mujoco_playground')

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from orbax import checkpoint as ocp
import onnxruntime as ort

def load_jax_model(checkpoint_path):
    """加载JAX模型"""
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
    
    # 提取normalizer和policy参数
    if isinstance(params, dict):
        if 'policy' in params and 'normalizer_params' in params:
            policy_params = params['policy']
            normalizer_params = params['normalizer_params']
        else:
            # 尝试其他可能的结构
            policy_params = params.get('policy', params.get('params', params))
            normalizer_params = params.get('normalizer_params', None)
    else:
        # 处理tuple/list结构
        if isinstance(params, (list, tuple)) and len(params) >= 2:
            normalizer_params = params[0]
            policy_params = params[1]
        else:
            policy_params = params
            normalizer_params = None
    
    return network, (normalizer_params, policy_params)

def test_model_outputs():
    """测试两种模型的输出"""
    checkpoint_path = "/home/balance/fork_mujoco_playground/mujoco_playground/learning/logs/PendulumSwingup-20250811-155911/checkpoints/000507248640"
    onnx_model_path = "pendulum_policy.onnx"
    
    print("🚀 开始比较JAX模型和ONNX模型输出...")
    
    # 加载JAX模型
    print("🔄 加载JAX模型...")
    network, params = load_jax_model(checkpoint_path)
    normalizer_params, policy_params = params
    
    # 提取归一化参数
    if normalizer_params is not None:
        print(f"📊 Normalizer参数类型: {type(normalizer_params)}")
        if hasattr(normalizer_params, 'keys'):
            print(f"   Keys: {list(normalizer_params.keys())}")
        
        # 提取mean和std
        if isinstance(normalizer_params, dict):
            print(f"   Normalizer dict keys: {list(normalizer_params.keys())}")
            mean_dict = normalizer_params.get('mean', {})
            std_dict = normalizer_params.get('std', {})
            print(f"   Mean字典keys: {list(mean_dict.keys()) if hasattr(mean_dict, 'keys') else 'N/A'}")
            print(f"   Std字典keys: {list(std_dict.keys()) if hasattr(std_dict, 'keys') else 'N/A'}")
            
            # 直接从normalizer_params中提取mean和std
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
        
        print(f"   Mean: {mean}")
        print(f"   Std: {std}")
    else:
        mean = np.zeros(4)
        std = np.ones(4)
        print("⚠️ 未找到normalizer参数，使用默认值")
    
    # 创建JAX推理函数
    make_policy = ppo_networks.make_inference_fn(network)
    inference_fn = make_policy(params, deterministic=True)
    
    # 加载ONNX模型
    print("🔄 加载ONNX模型...")
    ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    
    # 测试不同的观测值
    test_observations = [
        np.array([0.0, 1.0, 0.0, 0.0]),    # 底部位置，无速度
        np.array([1.0, 0.0, 0.0, 0.0]),    # 90度位置，无速度
        np.array([0.0, -1.0, 0.0, 0.0]),   # 顶部位置，无速度
        np.array([0.0, 1.0, 1.0, 0.0]),    # 底部位置，有速度
        np.array([0.0, -1.0, 0.0, 1.0]),   # 顶部位置，有扭矩历史
    ]
    
    print("\n" + "="*80)
    print("🧪 模型输出比较测试")
    print("="*80)
    print(f"{'观测值':<25} {'JAX输出':<15} {'ONNX输出':<15} {'差异':<15}")
    print("-"*80)
    
    for i, obs in enumerate(test_observations):
        # JAX模型推理
        obs_jax = jp.array(obs).reshape(1, -1)
        rng = jax.random.PRNGKey(i)
        jax_action, _ = inference_fn(obs_jax, rng)
        jax_result = np.array(jax_action).flatten()[0]
        
        # ONNX模型推理（先归一化）
        normalized_obs = (obs - mean) / (std + 1e-8)
        # 确保使用float32类型
        normalized_obs = normalized_obs.astype(np.float32)
        onnx_action = ort_session.run(None, {input_name: normalized_obs.reshape(1, -1)})[0]
        onnx_result = onnx_action.flatten()[0]
        
        # 计算差异
        diff = abs(jax_result - onnx_result)
        
        # 格式化观测值显示
        obs_str = f"[{', '.join([f'{x:.1f}' for x in obs])}]"
        
        print(f"{obs_str:<25} {jax_result:<15.6f} {onnx_result:<15.6f} {diff:<15.6f}")
    
    print("-"*80)
    print("✅ 测试完成!")
    
    # 特别测试归一化的影响
    print("\n🔬 归一化效果测试:")
    test_obs = np.array([0.5, 0.8, 0.3, 0.1])
    print(f"原始观测值: {test_obs}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    normalized = (test_obs - mean) / (std + 1e-8)
    print(f"归一化后: {normalized}")

if __name__ == "__main__":
    test_model_outputs()