# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pendulum environment."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "pendulum.xml"
_ANGLE_BOUND = 8
_COSINE_BOUND = np.cos(np.deg2rad(180 - _ANGLE_BOUND)) 


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.01,    # 🎯 100Hz控制频率，匹配CAN通信频率
      sim_dt=0.005,    # 🎯 更高仿真精度
      episode_length=2000,  # 🎯 给更多时间学习小扭矩控制
      action_repeat=1,
      vision=False,
      impl="jax",
      nconmax=0,
      njmax=0,
  )


class SwingUp(mjx_env.MjxEnv):
  """Swingup environment."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    if self._config.vision:
      raise NotImplementedError(
          f"Vision not implemented for {self.__class__.__name__}."
      )

    self._xml_path = _XML_PATH.as_posix()
    self._model_assets = common.get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        _XML_PATH.read_text(), self._model_assets
    )
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()
    # 🎯 简化传感器ID缓存 - 只保留3个核心传感器
    self._sensor_ids = {
        'motor_position': 0,    # jointpos传感器
        'motor_velocity': 1,    # jointvel传感器  
        'motor_torque': 2,      # actuatorfrc传感器
    }

  def _post_init(self) -> None:
    self._pole_body_id = self.mj_model.body("pole").id
    hinge_joint_id = self.mj_model.joint("hinge").id
    self._hinge_qposadr = self.mj_model.jnt_qposadr[hinge_joint_id]
    self._hinge_qveladr = self.mj_model.jnt_dofadr[hinge_joint_id]
    

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng1, rng_noise = jax.random.split(rng, 3)

    qpos = jp.zeros(self.mjx_model.nq)
    qpos = qpos.at[self._hinge_qposadr].set(jax.random.uniform(rng1) * jp.pi)

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    metrics = {}
    info = {
        "rng": rng,
        "noise_rng": rng_noise,
        "step_count": jp.array(0),
        "last_torque": jp.zeros(1),
        "torque_command": jp.zeros(1),
        "last_actual_torque": jp.zeros(1),
        "actual_torque": jp.zeros(1),
        # 🎯 传感器状态缓存
        "sensor_cos_theta": jp.array(1.0),
        "sensor_sin_theta": jp.array(0.0),
        "sensor_angular_vel": jp.array(0.0),
    }

    reward, done = jp.zeros(2)
    obs = self._get_obs(data, info)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  # 🎯 简化传感器访问函数 - 只返回核心CAN通信数据
  def get_sensor_data(self, data: mjx.Data) -> dict[str, float]:
      """获得核心传感器数据，模拟CAN通信返回的信息包"""
      return {
          'motor_position': data.sensordata[0],      # rad
          'motor_velocity': data.sensordata[1],      # rad/s  
          'motor_torque': data.sensordata[2],        # N·m
      }

  # 🎯 更新传感器噪声生成函数
  def _get_sensor_noise(self, info: dict[str, Any]) -> tuple[jax.Array, jax.Array, jax.Array]:
      """生成核心传感器噪声"""
      step_count = info.get('step_count', jp.array(0))
      noise_rng = info.get('noise_rng', jax.random.PRNGKey(0))
      
      pos_key = jax.random.fold_in(noise_rng, step_count * 3)
      vel_key = jax.random.fold_in(noise_rng, step_count * 3 + 1) 
      torque_key = jax.random.fold_in(noise_rng, step_count * 3 + 2)
      
      position_noise = 0.001 * jax.random.normal(pos_key)
      velocity_noise = 0.01 * jax.random.normal(vel_key)
      torque_noise = 0.005 * jax.random.normal(torque_key)
      
      return position_noise, velocity_noise, torque_noise

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    # 🎯 使用简化的传感器数据获取状态
    sensor_data = self.get_sensor_data(state.data)
    motor_position = sensor_data['motor_position']
    motor_velocity = sensor_data['motor_velocity']
    
    # 🎯 使用改进的传感器噪声生成
    position_noise, velocity_noise, torque_noise = self._get_sensor_noise(state.info)
    
    noisy_position = motor_position + position_noise
    noisy_velocity = motor_velocity + velocity_noise
    
    # 从传感器数据计算摆杆状态
    cos_theta = jp.cos(noisy_position)
    sin_theta = jp.sin(noisy_position)
    angular_vel = noisy_velocity
    
    # ✅ 只保留物理硬件限制
    max_torque = 1.2  # 固定的硬件限制，从XML文件读取
    torque_command = jp.clip(action, -max_torque, max_torque)
    
    # ✅ 轻微的硬件延迟/滤波（可选，模拟电机驱动器特性）
    last_actual_torque = state.info.get('actual_torque', jp.zeros(1))
    hardware_filter_factor = 0.95  # 固定的硬件特性
    actual_torque = hardware_filter_factor * torque_command + (1 - hardware_filter_factor) * last_actual_torque
      
    # 🎯 更新info
    new_info = state.info.copy()
    new_info['step_count'] = state.info.get('step_count', jp.array(0)) + 1
    new_info['last_torque'] = state.info.get('torque_command', jp.zeros(1))
    new_info['torque_command'] = torque_command.reshape(-1)
    new_info['last_actual_torque'] = last_actual_torque.reshape(-1)
    new_info['actual_torque'] = actual_torque.reshape(-1)
    
    # 传感器状态
    new_info['sensor_cos_theta'] = cos_theta
    new_info['sensor_sin_theta'] = sin_theta  
    new_info['sensor_angular_vel'] = angular_vel
    
    data = mjx_env.step(self.mjx_model, state.data, actual_torque, self.n_substeps)
    reward = self._get_reward(data, torque_command, new_info, state.metrics)
    obs = self._get_obs(data, new_info)
    done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)
    
    return mjx_env.State(data, obs, reward, done, state.metrics, new_info)

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    """🎯 使用简化的传感器系统获取观测值"""
    
    # 🎯 优先使用step函数中计算的传感器状态（避免重复计算）
    if 'sensor_cos_theta' in info and 'sensor_sin_theta' in info and 'sensor_angular_vel' in info:
        cos_theta = info['sensor_cos_theta']
        sin_theta = info['sensor_sin_theta']
        angular_vel = info['sensor_angular_vel']
        # 从传感器获取带噪声的力矩数据
        sensor_data = self.get_sensor_data(data)
        motor_torque = sensor_data['motor_torque']
        _, _, torque_noise = self._get_sensor_noise(info)
        noisy_torque = motor_torque + torque_noise
    else:
        # 🎯 回退模式：直接从传感器数据计算（用于reset等情况）
        sensor_data = self.get_sensor_data(data)
        motor_position = sensor_data['motor_position']
        motor_velocity = sensor_data['motor_velocity']  
        motor_torque = sensor_data['motor_torque']
        
        # 使用统一的传感器噪声生成
        position_noise, velocity_noise, torque_noise = self._get_sensor_noise(info)
        
        # 带噪声的传感器读数
        noisy_position = motor_position + position_noise
        noisy_velocity = motor_velocity + velocity_noise
        noisy_torque = motor_torque + torque_noise
        
        # 从关节角度计算摆杆方向
        cos_theta = jp.cos(noisy_position)
        sin_theta = jp.sin(noisy_position)
        angular_vel = noisy_velocity
    
    # 🎯 返回简化的观测值：只包含CAN通信的核心数据
    return jp.concatenate([
        jp.array([sin_theta, cos_theta]),    # 摆杆方向 (从传感器角度计算)
        angular_vel.reshape(1),              # 角速度 (带噪声)
        noisy_torque.reshape(1),             # 力矩反馈 (带噪声)
    ])

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
        del metrics  # Unused.
        
        # 🎯 使用传感器数据计算的状态 - 与step函数保持一致
        if 'sensor_cos_theta' in info:
            # 使用step函数中计算的传感器状态
            cos_theta = info['sensor_cos_theta']
            sin_theta = info['sensor_sin_theta']
            angular_vel = info['sensor_angular_vel']
        else:
            # 回退到传感器数据计算（用于reset等情况）
            motor_position = data.sensordata[self._sensor_ids['motor_position']]
            motor_velocity = data.sensordata[self._sensor_ids['motor_velocity']]
            
            # 使用统一的传感器噪声生成
            position_noise, velocity_noise, _ = self._get_sensor_noise(info)
            
            noisy_position = motor_position + position_noise
            noisy_velocity = motor_velocity + velocity_noise
            
            cos_theta = jp.cos(noisy_position)
            sin_theta = jp.sin(noisy_position)
            angular_vel = noisy_velocity
        
        # 🎯 重新设计阶段划分 - 避免中间区域陷阱
        is_very_bottom = cos_theta > 0.8        # 真正底部区域（约37度范围内）
        is_bottom_half = cos_theta > -0.5       # 下半区域
        is_middle = (cos_theta <= -0.5) & (cos_theta > -0.85)  # 中间过渡区
        is_near_top = (cos_theta <= -0.85) & (cos_theta > -0.95)  # 接近顶部
        is_very_top = cos_theta <= -0.95        # 顶部区域  
      
        # 🎯 底部能量积累奖励 - 大幅增强，鼓励高能量摆动
        energy_reward = jp.where(
            is_very_bottom,
            # 只有在真正底部且高速度时才给奖励
            jp.where(
                jp.abs(angular_vel) > 1.0,
                jp.minimum(angular_vel**2 / 3.0, 12.0),  # 高速度大奖励
                -5.0 * (1.0 - jp.abs(angular_vel))       # 低速度负奖励，强迫摆动
            ),
            jp.where(
                is_bottom_half & ~is_very_bottom,
                # 其他底部区域：奖励向上运动的能量
                jp.where(
                    jp.abs(angular_vel) > 0.5,
                    angular_vel**2 / 5.0 * (1.0 - cos_theta),  # 速度×高度奖励
                    0.0
                ),
                0.0
            )
        )
        
        # 🎯 向上突破奖励 - 大幅奖励从中间区域向顶部的移动
        upward_momentum_reward = jp.where(
            is_middle,
            # 在中间区域奖励继续向上的动量
            jp.where(
                jp.abs(angular_vel) > 1.0,  # 有足够速度
                (-cos_theta - 0.5) * 20.0 + jp.minimum(angular_vel**2 / 2.0, 15.0),  # 高度+速度奖励
                -3.0  # 速度不足时惩罚
            ),
            0.0
        )
        
        # 🎯 接近顶部奖励 - 渐进式增强
        approaching_top_reward = jp.where(
            is_near_top,
            (-cos_theta - 0.85) * 100.0 + 15.0,  # 大幅增强接近顶部的奖励
            0.0
        )
        
        # 🎯 顶部稳定奖励
        upright_reward = jp.where(
            is_very_top,
            reward.tolerance(cos_theta, (-1.0, -0.98)) * 50.0,  # 修正为负值范围
            0.0
        )
        
        # 🎯 速度稳定奖励 - 仅在顶部
        stability_reward = jp.where(
            is_very_top,
            jp.exp(-angular_vel**2 / 0.3) * 20.0,
            0.0
        )
        
        # 🎯 静止奖励 - 超高奖励
        stillness_reward = jp.where(
            is_very_top & (jp.abs(angular_vel) < 0.05),
            100.0,  # 超高静止奖励
            0.0
        )
        
        # 🎯 中间区域陷阱惩罚 - 防止智能体满足于中间摆动
        middle_trap_penalty = jp.where(
            is_middle & (jp.abs(angular_vel) < 1.5),  # 在中间区域但速度不足
            -6.0,  # 惩罚低速度的中间摆动
            0.0
        )
        
        # 🎯 能量损失惩罚 - 如果摆杆从高位置下降
        energy_loss_penalty = jp.where(
            (cos_theta > 0.5) & (angular_vel * jp.sign(cos_theta - 0.5) < 0),  # 从高位置向底部运动
            -5.0 * jp.abs(angular_vel),  # 增强惩罚
            0.0
        )
        
        # 🎯 扭矩效率 - 简化
        torque_magnitude = jp.abs(action[0])
        efficiency_reward = jp.where(
            is_very_top & (torque_magnitude < 0.2),
            2.0,
            0.0
        )
        
        # 🎯 传感器噪声适应奖励 - 奖励在噪声环境下的稳定性
        sensor_stability_reward = jp.where(
            is_very_top,
            # 基于角度和角速度的联合稳定性
            jp.exp(-(sin_theta**2 + angular_vel**2) / 0.1) * 5.0,
            0.0
        )

        # 🎯 大幅加强扭矩惩罚 - 在顶部严厉惩罚大扭矩
        torque_magnitude = jp.abs(action[0])
        
        # 顶部扭矩惩罚 - 非线性惩罚
        top_torque_penalty = jp.where(
            is_very_top,
            # 在顶部使用大扭矩会被严厉惩罚
            -jp.where(
                torque_magnitude > 0.2,
                (torque_magnitude - 0.2) * 200.0,  # 超过0.2的部分大幅惩罚
                0.0
            ),
            jp.where(
                is_near_top,
                # 接近顶部时也要惩罚大扭矩
                -jp.where(
                    torque_magnitude > 0.5,
                    (torque_magnitude - 0.5) * 50.0,
                    0.0
                ),
                0.0
            )
        )
        
        # 🎯 扭矩平滑奖励 - 鼓励连续的小调整
        if 'last_actual_torque' in info:
            last_torque = info['last_actual_torque']
            torque_change = jp.abs(action[0] - last_torque)
            
            smoothness_reward = jp.where(
                is_very_top | is_near_top,
                # 在顶部区域奖励平滑的扭矩变化
                jp.exp(-torque_change * 5.0) * 3.0,  # 越平滑奖励越高
                0.0
            )
        else:
            smoothness_reward = 0.0
        
        # 🎯 精确控制奖励 - 大幅提升
        precision_control_reward = jp.where(
            is_very_top & (torque_magnitude < 0.1),
            20.0,  # 使用极小扭矩时给予高奖励
            0.0
        )
        
         # 🎯 归一化各奖励项到[-1, 1]
        normalized_rewards = {
            'energy': jp.tanh(energy_reward / 10.0),                    # /10归一化
            'upward': jp.tanh(upward_momentum_reward / 20.0),          # /20归一化
            'approaching': jp.tanh(approaching_top_reward / 50.0),      # /50归一化，压制主导性
            'upright': jp.tanh(upright_reward / 30.0),                 # /30归一化
            'stability': jp.tanh(stability_reward / 15.0),             # /15归一化
            'stillness': jp.tanh(stillness_reward / 80.0),             # /80归一化，压制主导性
            'middle_trap': jp.tanh(middle_trap_penalty / 5.0),         # 惩罚也归一化
            'energy_loss': jp.tanh(energy_loss_penalty / 10.0),
            'efficiency': jp.tanh(efficiency_reward / 3.0),
            'sensor_stability': jp.tanh(sensor_stability_reward / 5.0),
        }
        
        # 🎯 现在权重调整变得直观（都是[-1,1]范围）
        total_reward = (
            normalized_rewards['energy'] * 1.0 +           # 底部能量
            normalized_rewards['upward'] * 1.0 +           # 向上突破  
            normalized_rewards['approaching'] * 1.0 +      # 接近顶部
            normalized_rewards['upright'] * 1.0 +          # 顶部稳定
            normalized_rewards['stability'] * 3.0 +        # 速度稳定（你之前的调整）
            normalized_rewards['stillness'] * 5.0 +        # 静止奖励（你之前的调整）
            normalized_rewards['middle_trap'] * 1.0 +      # 中间陷阱
            normalized_rewards['energy_loss'] * 1.0 +      # 能量损失
            normalized_rewards['efficiency'] * 1.0 +       # 扭矩效率
            normalized_rewards['sensor_stability'] * 1.0   # 传感器稳定
        )
        
        # 你的扭矩惩罚也归一化
        torque_magnitude = jp.abs(action[0])
        simple_torque_penalty = jp.where(
            is_very_top & (torque_magnitude > 0.3),
            -0.3,  # 归一化后的惩罚
            0.0
        )
        
        return total_reward + simple_torque_penalty
  
  def _pole_vertical(self, data: mjx.Data) -> jax.Array:
    """Returns vertical (z) component of pole frame."""
    return data.xmat[self._pole_body_id, 2, 2]

  def _angular_velocity(self, data: mjx.Data) -> jax.Array:
    """Returns the angular velocity of the pole."""
    return data.qvel[self._hinge_qveladr]

  def _pole_orientation(self, data: mjx.Data) -> jax.Array:
    """Returns both horizontal and vertical components of pole frame."""
    xz = data.xmat[self._pole_body_id, 0, 2]
    zz = data.xmat[self._pole_body_id, 2, 2]
    return jp.array([xz, zz])

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
