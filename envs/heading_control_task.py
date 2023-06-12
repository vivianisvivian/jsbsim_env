from turtle import st
from my_jsbsim_env.task import Task
from my_jsbsim_env.catalogs.catalog import Catalog as c
import math
import random
import numpy as np


altitude = 5000
roll_max = 80
heading_max = 60
T = 60 # 一共60s的周期

class HeadingControlTask(Task):
    """
    S 机动
    """
    jsbsim_freq = 100  # 交互频率 
    agent_interaction_steps = 5 # agent等待时长,这个要跟FG看一下对应哪个参数
    aircraft_name = 'f16'
    delta_t = agent_interaction_steps/jsbsim_freq # 多少秒决策一次（0.05s）

    state_var = [  # state dim = 14
        c.delta_altitude,   # 相对高度

        c.attitude_psi_deg,  # heading
        c.target_heading_deg,  # 目标heading

        c.aero_alpha_deg,  # 攻角侧滑角
        c.aero_beta_deg,

        c.attitude_theta_deg, # pitch 俯仰角 
        c.attitude_phi_deg,  # roll 滚转角

        c.velocities_u_fps,  # 三个方向的速度与角速度
        c.velocities_v_fps,
        c.velocities_w_fps,
        c.velocities_p_rad_sec,
        c.velocities_q_rad_sec,
        c.velocities_r_rad_sec,
        c.t,  # 时间轴
    ]

    action_var = [
        c.fcs_aileron_cmd_norm,
        c.fcs_elevator_cmd_norm,
        c.fcs_rudder_cmd_norm,
        # c.fcs_throttle_cmd_norm,
    ]


    init_conditions = {
        # 位置
        c.ic_h_sl_ft: 5000,
        c.target_altitude_ft: 5000,
        c.ic_terrain_elevation_ft: 0,
        c.ic_long_gc_deg: -120,
        c.ic_lat_gc_deg: 30,

        # # 姿态
        c.ic_psi_true_deg: 180,
        c.ic_theta_deg: 0,
        c.ic_phi_deg: 0,

        c.target_heading_deg: 180,

        # 机体坐标系速度
        c.ic_u_fps: 1000,
        c.ic_v_fps: 0,
        c.ic_w_fps: 0,
        # 角加速度
        c.ic_p_rad_sec: 0,
        c.ic_q_rad_sec: 0,
        c.ic_r_rad_sec: 0,
        c.ic_roc_fpm: 0,

        c.fcs_throttle_cmd_norm: 0.9,
        c.gear_gear_pos_norm: 0,
        c.gear_gear_cmd_norm: 0,
        c.t:0,
    }

    def get_reward(self, state, sim):
        """
        这里只返回terminate reward, 其他奖励见训练端
        """
        t = sim.get_property_value(c.t) 
        sim.set_property_value(c.t, t + self.agent_interaction_steps/self.jsbsim_freq)

        A = heading_max 
        k = 4*heading_max/T

        y1 = A * np.sin(2 * math.pi * t / T)
        y2 = np.where((t + T / 4) % T < T / 2, -A + k * ((t + T / 4) % T), A - k * ((t - T / 4) % T))
        new_heading = 180 + (0.7*y1+0.3*y2)

        sim.set_property_value(c.target_heading_deg, new_heading)

        # target_roll = 80 if ((t + T / 4) % T) < (T/2) else -80
        # sim.set_property_value(c.target_roll_deg, target_roll)

        ############################
        done, info = self.is_terminal(state,sim)

        if done and info not in ["time out", 'roll']:
            return -500

        # rerror = abs(state[1])
        # r_heading = 1.0 / (1.0 + math.exp((error - 24)/5))5

        return 0.


    def is_terminal(self, state, sim):

        if abs(sim.get_property_value(c.attitude_phi_deg)) >= 100:
            return True, 'roll过大,翻滚'

        # if sim.get_property_value(c.detect_extreme_state):
        #     return True, "极限状态"

        if sim.get_property_value(c.position_h_agl_ft) <= 2000:
            return True, "过低"

        if sim.get_property_value(c.position_h_agl_ft) >= 8000:
            return True, "过高"

        if sim.get_property_value(c.t) >= 2*T:
            return True, "time out"

        return False, ""