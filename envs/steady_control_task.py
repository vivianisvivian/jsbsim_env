from my_jsbsim_env.task import Task
from my_jsbsim_env.catalogs.catalog import Catalog as c
import numpy as np


class SteadyControlTask(Task):
    jsbsim_freq = 10  # 交互频率
    agent_interaction_steps = 1  # agent等待时长,这个要跟FG看一下对应哪个参数
    # aircraft_name = "f15"   # 飞机名
    aircraft_name = 'f16'

    state_var = [
        c.delta_vc_fps,
        c.delta_altitude,
        c.delta_heading,

        c.aero_alpha_deg,
        c.aero_beta_deg,

        c.attitude_theta_deg,
        c.attitude_phi_deg,
        c.velocities_u_fps,
        c.velocities_v_fps,
        c.velocities_w_fps,
        c.velocities_p_rad_sec,
        c.velocities_q_rad_sec,
        c.velocities_r_rad_sec,
    ]

    action_var = [
        c.fcs_aileron_cmd_norm,
        c.fcs_elevator_cmd_norm,
        c.fcs_rudder_cmd_norm,
        # c.fcs_throttle_cmd_norm
    ]

    init_conditions = {
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

        c.fcs_throttle_cmd_norm: 0,
        c.gear_gear_pos_norm: 0,
        c.gear_gear_cmd_norm: 0,
        c.t:0,
    }

    def get_state_norm(self):
        return np.array([100, 10, 1,
                        1, 0.1,
                        1, 1,
                        100, 1, 1, 
                        0.1, 0.1, 0.1])


    def get_reward(self, state, sim):
    
        done, info = self.is_terminal(state, sim)

        if done and ('time out' not in info):
            return -100
        else:
            return 0.

    def is_terminal(self,state, sim):
        done = False
        info = ''
        
        if abs(sim.get_property_value(c.delta_heading)) > 180:
            done = True
            info += "偏航差距过大"

        if abs(sim.get_property_value(c.delta_altitude)) > 4000:
            done = True
            info += "高度差距过大"

        if sim.get_property_value(c.simulation_sim_time_sec) >= 60:
            done = True
            info += "time out"

        return done, info
