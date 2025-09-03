#!/usr/bin/env python3
"""
🎯 交互式摆杆Sim-to-Sim验证 (ONNX版本) - 改进版
- 启动MuJoCo可视化器
- 摆杆初始在最低点
- 按Enter开始AI控制
- 观察swing up过程
- 使用ONNX模型进行推理
"""

import os
import sys
sys.path.append('/home/balance/fork_mujoco_playground/mujoco_playground')

import time
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort

class InteractivePendulumDemoONNX:
    def __init__(self, onnx_model_path: str):
        """初始化交互式演示"""
        self.onnx_model_path = onnx_model_path
        self.model = None
        self.data = None
        self.viewer = None
        self.ort_session = None
        self.is_running = False
        self.reset_requested = False
        self._last_applied_action = 0.0
        
        # 归一化参数（从模型转换时提取）
        self.normalizer_mean = np.array([-0.02921442, -0.58093756, -0.9564421, -0.14311926], dtype=np.float32)
        self.normalizer_std = np.array([0.47902852, 0.6574112, 6.7379885, 0.6200137], dtype=np.float32)
        
        print(f"🎯 初始化演示，ONNX模型路径: {onnx_model_path}")
        
        # 初始化环境和模型
        self._setup_environment()
        self._load_onnx_model()
        
    def _setup_environment(self):
        """设置MuJoCo环境"""
        print("🎮 设置MuJoCo环境...")
        
        xml_file = "mujoco_playground/_src/dm_control_suite/xmls/pendulum.xml"
        
        if not os.path.exists(xml_file):
            print(f"❌ 未找到XML文件: {xml_file}")
            raise Exception("无法找到pendulum.xml文件")
        
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_file)
            self.data = mujoco.MjData(self.model)
            
            print(f"📊 模型信息:")
            print(f"   执行器数量: {self.model.nu}")
            print(f"   关节数量: {self.model.nq}")
            print(f"   传感器数量: {self.model.nsensor}")
            
            if self.model.nu == 0:
                print("⚠️ 警告：未检测到执行器，可能无法应用力矩")
            
            self._reset_to_bottom()
            print("✅ MuJoCo环境设置完成")
        except Exception as e:
            print(f"❌ MuJoCo模型加载失败: {e}")
            raise
        
    def _load_onnx_model(self):
        """加载ONNX模型"""
        print("🔄 加载ONNX模型...")
        
        try:
            # 创建ONNX Runtime会话
            self.ort_session = ort.InferenceSession(
                self.onnx_model_path,
                providers=['CPUExecutionProvider']
            )
            
            # 获取输入输出信息
            input_info = self.ort_session.get_inputs()
            output_info = self.ort_session.get_outputs()
            
            print(f"📊 ONNX模型信息:")
            print(f"   输入名称: {[inp.name for inp in input_info]}")
            print(f"   输入形状: {[inp.shape for inp in input_info]}")
            print(f"   输出名称: {[out.name for out in output_info]}")
            print(f"   输出形状: {[out.shape for out in output_info]}")
            
            print("✅ ONNX模型加载完成!")
            
        except Exception as e:
            print(f"❌ ONNX模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def _reset_to_bottom(self):
        """重置摆杆到最低点"""
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'hinge')
        if joint_id >= 0:
            self.data.qpos[joint_id] = 0.0
            self.data.qvel[joint_id] = 0.0
        
        mujoco.mj_forward(self.model, self.data)
        print("🎯 摆杆已重置到最低点")
    
    def _get_observation(self):
        """获取当前观测状态 - 匹配训练时的格式"""
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'hinge')
        
        if joint_id >= 0:
            angle = self.data.qpos[joint_id]
            angular_vel = self.data.qvel[joint_id]
        else:
            angle = 0.0
            angular_vel = 0.0
        
        current_torque = 0.0
        if hasattr(self, '_last_applied_action'):
            current_torque = self._last_applied_action
        
        sin_angle = np.sin(angle)
        cos_angle = np.cos(angle)
        
        obs = np.array([sin_angle, cos_angle, angular_vel, current_torque], dtype=np.float32)
        return obs
    
    def _apply_action(self, action):
        """应用动作到环境 - 匹配训练时的力矩范围"""
        action = np.clip(action, -1.2, 1.2)
        self._last_applied_action = float(action)
        
        if self.model.nu > 0:
            self.data.ctrl[0] = action
        else:
            print("❌ 未找到执行器!")
    
    def _get_status_info(self):
        """获取状态信息用于显示"""
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'hinge')
        
        if joint_id >= 0:
            angle = self.data.qpos[joint_id]
            angular_vel = self.data.qvel[joint_id]
        else:
            angle = 0.0
            angular_vel = 0.0
        
        angle_deg = np.degrees(abs(angle))
        
        # 确保角度在0-180度范围内
        if angle_deg > 180:
            angle_deg = 360 - angle_deg
        
        # 判断摆杆状态
        if angle_deg > 170 and abs(angular_vel) < 0.5:
            status = "🎯 倒立稳定"
        elif angle_deg > 150:
            status = "⚡ 接近稳定"
        elif angle_deg < 30:
            status = "🌊 底部摆动"
        elif abs(angular_vel) > 5:
            status = "🌪️ 快速摆动"
        else:
            status = "⚡ 上升中"
        
        return angle_deg, angular_vel, status
    
    def _keyboard_callback(self, keycode):
        """键盘回调函数"""
        if keycode == 256:  # ESC
            print("👋 退出演示")
            return False
        elif keycode == 257:  # Enter
            self.is_running = not self.is_running
            if self.is_running:
                print("🎯 开始AI控制...")
            else:
                print("⏸️ 暂停AI控制")
        elif keycode == 82 or keycode == 114:  # R 或 r
            self.reset_requested = True
            print("🔄 重置请求")
        
        return True
    
    def _normalize_observation(self, obs):
        """对观测值进行归一化处理"""
        # 确保使用正确的数据类型
        obs = np.array(obs, dtype=np.float32)
        normalized = (obs - self.normalizer_mean) / (self.normalizer_std + 1e-8)
        return normalized.astype(np.float32)
    
    def run(self):
        """运行交互式演示"""
        print("\n" + "="*60)
        print("🎮 交互式摆杆Sim-to-Sim验证 (ONNX版本 - 改进版)")
        print("="*60)
        print("⌨️  控制说明:")
        print("   Enter  - 开始/停止AI控制")
        print("   R      - 重置摆杆到最低点")
        print("   ESC    - 退出演示")
        print("="*60)
        print("✅ ONNX模型已就绪")
        print("🎯 按 Enter 开始AI控制...")
        
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self._keyboard_callback) as viewer:
            self.viewer = viewer
            
            while viewer.is_running():
                step_start = time.time()
                
                if self.reset_requested:
                    self._reset_to_bottom()
                    self.reset_requested = False
                
                if self.is_running:
                    try:
                        obs = self._get_observation()
                        
                        # 使用归一化处理
                        normalized_obs = self._normalize_observation(obs)
                        
                        # 使用ONNX模型进行推理
                        input_name = self.ort_session.get_inputs()[0].name
                        obs_input = normalized_obs.reshape(1, -1)
                        action_result = self.ort_session.run(None, {input_name: obs_input})[0]
                        action_np = action_result.flatten()
                        
                        # 应用动作
                        if len(action_np) > 0:
                            try:
                                action_value = float(action_np[0])
                                self._apply_action(action_value)
                            except (ValueError, TypeError):
                                print(f"❌ 动作值无法转换为float: {action_np[0]}")
                                self._apply_action(0.0)
                        else:
                            self._apply_action(0.0)
                        
                    except Exception as e:
                        print(f"\n❌ ONNX推理出错: {e}")
                        import traceback
                        traceback.print_exc()
                        self.is_running = False
                
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                
                # 状态显示
                angle_deg, vel, status = self._get_status_info()
                state_indicator = "🎯 运行中" if self.is_running else "⏸️ 暂停"
                is_stable = status == "🎯 倒立稳定"
                stable_indicator = " | ✅ 稳定" if is_stable else ""
                torque_display = f" | 力矩: {self._last_applied_action:.3f}" if hasattr(self, '_last_applied_action') else ""
                
                print(f"\r{state_indicator} | 倒立角度: {angle_deg:.1f}° | 速度: {vel:.2f} | {status}{stable_indicator}{torque_display}", end="")
                
                # 控制帧率
                time_until_next_step = 0.016 - (time.time() - step_start)  # 约60Hz
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

def main():
    """主函数"""
    onnx_model_path = "pendulum_policy.onnx"
    
    try:
        demo = InteractivePendulumDemoONNX(onnx_model_path)
        demo.run()
    except KeyboardInterrupt:
        print("\n👋 用户中断，退出程序")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()