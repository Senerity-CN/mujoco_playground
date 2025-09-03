#!/usr/bin/env python3
"""
调试ONNX模型推理输出
"""

import sys
sys.path.append('/home/balance/fork_mujoco_playground/mujoco_playground')

import numpy as np
import onnxruntime as ort

def debug_onnx_model():
    """调试ONNX模型推理"""
    onnx_model_path = "pendulum_policy.onnx"
    
    print("🔍 调试ONNX模型推理输出...")
    
    # 加载ONNX模型
    print("🔄 加载ONNX模型...")
    ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    
    print(f"📊 ONNX模型信息:")
    print(f"   输入名称: {input_name}")
    print(f"   输入形状: {ort_session.get_inputs()[0].shape}")
    print(f"   输出名称: {[out.name for out in ort_session.get_outputs()]}")
    print(f"   输出形状: {[out.shape for out in ort_session.get_outputs()]}")
    
    # 设置归一化参数
    normalizer_mean = np.array([-0.02921442, -0.58093756, -0.9564421, -0.14311926], dtype=np.float32)
    normalizer_std = np.array([0.47902852, 0.6574112, 6.7379885, 0.6200137], dtype=np.float32)
    
    print(f"\n📈 归一化参数:")
    print(f"   Mean: {normalizer_mean}")
    print(f"   Std: {normalizer_std}")
    
    # 测试各种观测值
    test_observations = [
        ([0.0, 1.0, 0.0, 0.0], "底部位置, 无速度"),      # 底部，sin=0, cos=1
        ([0.0, -1.0, 0.0, 0.0], "顶部位置, 无速度"),     # 顶部，sin=0, cos=-1
        ([1.0, 0.0, 0.0, 0.0], "右侧位置, 无速度"),      # 右侧，sin=1, cos=0
        ([-1.0, 0.0, 0.0, 0.0], "左侧位置, 无速度"),     # 左侧，sin=-1, cos=0
        ([0.0, 1.0, 1.0, 0.0], "底部位置, 顺时针速度"),
        ([0.0, -1.0, 1.0, 0.0], "顶部位置, 顺时针速度"),
        ([0.0, -1.0, -1.0, 0.0], "顶部位置, 逆时针速度"),
    ]
    
    print(f"\n{'='*80}")
    print(f"{'观测值(原始)':<20} {'观测值(归一化)':<25} {'ONNX输出':<12} {'动作范围'}")
    print(f"{'-'*80}")
    
    for obs, description in test_observations:
        obs_array = np.array(obs, dtype=np.float32)
        
        # 归一化
        normalized_obs = (obs_array - normalizer_mean) / (normalizer_std + 1e-8)
        
        # ONNX推理
        action_result = ort_session.run(None, {input_name: normalized_obs.reshape(1, -1).astype(np.float32)})[0]
        action_value = float(action_result.flatten()[0])
        
        # 格式化显示
        obs_str = f"[{', '.join([f'{x:4.1f}' for x in obs])}]"
        norm_str = f"[{', '.join([f'{x:5.2f}' for x in normalized_obs])}]"
        
        # 动作分析
        if action_value > 0.5:
            action_desc = "右侧力矩"
        elif action_value < -0.5:
            action_desc = "左侧力矩"
        else:
            action_desc = "小力矩/无"
            
        print(f"{obs_str:<20} {norm_str:<25} {action_value:<12.6f} {action_desc}")
    
    # 对比原始方法（不归一化）
    print(f"\n{'='*80}")
    print("❌ 不使用归一化的输出（错误方法）:")
    print(f"{'观测值(原始)':<20} {'ONNX输出':<12} {'动作范围'}")
    print(f"{'-'*80}")
    
    for obs, description in test_observations[:4]:
        obs_array = np.array(obs, dtype=np.float32)
        
        # 不归一化直接推理
        action_result = ort_session.run(None, {input_name: obs_array.reshape(1, -1).astype(np.float32)})[0]
        action_value = float(action_result.flatten()[0])
        
        # 格式化显示
        obs_str = f"[{', '.join([f'{x:4.1f}' for x in obs])}]"
        
        # 动作分析
        if action_value > 0.5:
            action_desc = "右侧力矩"
        elif action_value < -0.5:
            action_desc = "左侧力矩"
        else:
            action_desc = "小力矩/无"
            
        print(f"{obs_str:<20} {action_value:<12.6f} {action_desc}")

if __name__ == "__main__":
    debug_onnx_model()