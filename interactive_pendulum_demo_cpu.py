#!/usr/bin/env python3
"""
🎯 交互式摆杆Sim-to-Sim验证 (CPU版本)
- 启动MuJoCo可视化器
- 摆杆初始在最低点
- 按Enter开始AI控制
- 观察swing up过程
"""

import os
# 确保在CPU上运行，避免CUDA错误
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
# 禁用所有CUDA相关功能
os.environ['JAX_ENABLE_X64'] = 'false'
os.environ['JAX_ENABLE_CUDA'] = 'false'
os.environ['JAX_ENABLE_TPU'] = 'false'

import sys
import time
import warnings

# 抑制所有CUDA相关警告
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append('/home/balance/mujoco_playground')

# 抑制警告
warnings.filterwarnings('ignore', message='.*CUDA.*')
warnings.filterwarnings('ignore', message='.*cuInit.*')
warnings.filterwarnings('ignore', message='.*plugin.*')

import jax
import jax.numpy as jp
import numpy as np
import mujoco
import mujoco.viewer
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import checkpoint as ppo_checkpoint

class InteractivePendulumDemoCPU:
    def __init__(self, checkpoint_path: str):
        """初始化交互式演示"""
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.data = None
        self.viewer = None
        self.inference_fn = None
        self.params = None
        self.is_running = False
        self.reset_requested = False
        self._last_applied_action = 0.0
        
        print(f"🎯 初始化演示，checkpoint路径: {checkpoint_path}")
        
        # 初始化环境和模型
        self._setup_environment()
        self._load_trained_model()
        
    def _setup_environment(self):
        """设置MuJoCo环境"""
        print("🎮 设置MuJoCo环境...")
        
        xml_file = "mujoco_playground/_src/dm_control_suite/xmls/pendulum.xml"
        
        if not os.path.exists(xml_file):
            # 尝试其他可能的路径
            xml_file = "/home/balance/mujoco_playground/mujoco_playground/_src/dm_control_suite/xmls/pendulum.xml"
            if not os.path.exists(xml_file):
                raise Exception(f"未找到pendulum.xml文件: {xml_file}")
        
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
        
    def _load_trained_model(self):
        """加载训练好的模型"""
        print("🔄 加载训练模型...")
        
        try:
            network_factory_kwargs = {
                "policy_hidden_layer_sizes": [32, 32, 32, 32],
                "value_hidden_layer_sizes": [256, 256, 256, 256, 256],
                "policy_obs_key": "state",
                "value_obs_key": "state"
            }
            
            self.network = ppo_networks.make_ppo_networks(
                observation_size=4,
                action_size=1,
                **network_factory_kwargs
            )
            
            # 使用ppo_checkpoint加载检查点
            self.params = ppo_checkpoint.load(self.checkpoint_path)
            
            print("🔍 参数结构分析:")
            print(f"   参数类型: {type(self.params)}")
            
            if isinstance(self.params, dict):
                print(f"   参数键: {list(self.params.keys())}")
                if 'policy' in self.params:
                    self.policy_params = self.params['policy']
                    print("🎯 使用字典格式的policy参数")
                elif 'params' in self.params:
                    self.policy_params = self.params['params']
                    print("🎯 使用字典格式的params参数")
                else:
                    self.policy_params = self.params
                    print("⚠️  使用整个params字典作为policy参数")
            elif isinstance(self.params, (list, tuple)) and len(self.params) >= 2:
                self.normalizer_params = self.params[0]
                self.policy_params = self.params[1]
                print("🎯 成功解析参数结构: [normalizer, policy]")
                print(f"   Normalizer类型: {type(self.normalizer_params)}")
                print(f"   Policy参数类型: {type(self.policy_params)}")
            else:
                self.policy_params = self.params
                print("⚠️  使用params作为policy_params")
            
            # 确保参数在CPU上
            cpu_device = jax.devices("cpu")[0]
            if hasattr(self, 'normalizer_params'):
                self.normalizer_params = jax.device_put(self.normalizer_params, cpu_device)
                self.policy_params = jax.device_put(self.policy_params, cpu_device)
            else:
                self.policy_params = jax.device_put(self.policy_params, cpu_device)
            
            print("✅ 模型加载完成!")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def _reset_to_bottom(self):
        """重置摆杆到最低点"""
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'hinge')
        if joint_id >= 0:
            self.data.qpos[joint_id] = 0.0  # 最低点
            self.data.qvel[joint_id] = 0.0
        else:
            # 如果找不到关节，直接设置数组
            if len(self.data.qpos) > 0:
                self.data.qpos[0] = 0.0
            if len(self.data.qvel) > 0:
                self.data.qvel[0] = 0.0
        
        # 重置最后应用的动作
        self._last_applied_action = 0.0
        
        mujoco.mj_forward(self.model, self.data)
        print("摆杆已重置到最低点")
    
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
    
    def run(self):
        """运行交互式演示"""
        print("\n" + "="*60)
        print("🎮 交互式摆杆Sim-to-Sim验证 (CPU版本)")
        print("="*60)
        print("⌨️  控制说明:")
        print("   Enter  - 开始/停止AI控制")
        print("   R      - 重置摆杆到最低点")
        print("   ESC    - 退出演示")
        print("="*60)
        print("✅ AI模型已就绪")
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
                        
                        if hasattr(self, 'normalizer_params'):
                            # 使用属性访问而不是字典访问
                            mean = self.normalizer_params.mean
                            std = self.normalizer_params.std
                            normalized_obs = (obs - np.array(mean)) / (np.array(std) + 1e-8)
                            obs_jax = jp.array(normalized_obs).reshape(1, -1)
                        else:
                            obs_jax = jp.array(obs).reshape(1, -1)
                        
                        # 初始化计数器
                        if not hasattr(self, '_debug_counter'):
                            self._debug_counter = 0
                        self._debug_counter += 1
                        
                        try:
                            make_policy = ppo_networks.make_inference_fn(self.network)
                            
                            # 确保参数在CPU上
                            cpu_device = jax.devices("cpu")[0]
                            if hasattr(self, 'normalizer_params'):
                                # 确保参数在CPU上
                                normalizer_params_cpu = jax.device_put(self.normalizer_params, cpu_device)
                                policy_params_cpu = jax.device_put(self.policy_params, cpu_device)
                                policy_params = (normalizer_params_cpu, policy_params_cpu)
                            else:
                                policy_params = jax.device_put(self.policy_params, cpu_device)
                            
                            policy_fn = make_policy(policy_params, deterministic=True)
                            
                            # 确保观测数据在CPU上
                            obs_cpu = jax.device_put(obs_jax, cpu_device)
                            
                            rng = jax.random.PRNGKey(self._debug_counter)
                            action_result, _ = policy_fn(obs_cpu, rng)
                            action_np = np.array(action_result).flatten()
                                
                        except Exception as e_main:
                            print(f"❌ 推理方法失败: {e_main}")
                            # 回退方案：使用简单PD控制器
                            sin_theta, cos_theta, ang_vel, torque = obs
                            angle = np.arctan2(sin_theta, cos_theta)
                            target_angle = np.pi
                            angle_error = target_angle - angle
                            action_value = np.clip(2.0 * angle_error - 0.1 * ang_vel, -1.0, 1.0)
                            action_np = np.array([action_value])
                            print(f"🎯 使用PD控制器回退，动作: {action_np[0]:.3f}")
                        
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
                        print(f"\n❌ AI控制出错: {e}")
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
                time_until_next_step = 0.01 - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

def main():
    """主函数"""
    checkpoint_path = "/home/balance/mujoco_playground/learning/logs/PendulumSwingup-20250811-155911/checkpoints/000507248640"
    
    try:
        demo = InteractivePendulumDemoCPU(checkpoint_path)
        demo.run()
    except KeyboardInterrupt:
        print("\n👋 用户中断，退出程序")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()