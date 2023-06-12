
import gym
from my_jsbsim_env.catalogs.catalog import Catalog as c

class k_steady():
    ## 速度控制 PI
    v = 0.01
    v_i = 0.1
    ## 高度控制 PD  # 升降舵控制俯仰
    theta= -0.1 
    q= 0
    # theta_cmd PID
    h=0.1
    h_i=0.01

    vz=0.01
    r= -5 
    beta= 1
    # aileron 
    phi=-0.01 # roll
    phi_i=-0.0001 
    p=-0.1

    psi=10


def get_action(env):  # steady control

    psi_cmd = env.sim.get_property_value(c.target_heading_deg)

    h_cmd = env.sim.get_property_value(c.target_altitude_ft)
   

    h = env.sim.get_property_value(c.position_h_sl_ft)
    delta_h = h - h_cmd

    psi = env.sim.get_property_value(c.attitude_psi_deg)   # yaw
    delta_heading = psi- psi_cmd

    p = env.sim.get_property_value(c.velocities_p_rad_sec) 
    q = env.sim.get_property_value(c.velocities_q_rad_sec)
    r = env.sim.get_property_value(c.velocities_r_rad_sec)

    theta = env.sim.get_property_value(c.attitude_theta_deg)
    vz = env.sim.get_property_value(c.velocities_w_fps)
    vx = env.sim.get_property_value(c.velocities_u_fps)

    phi = env.sim.get_property_value(c.attitude_phi_deg)  # roll
    beta = env.sim.get_property_value(c.aero_beta_deg)


    k = k_steady()
    theta_cmd = - k.h * delta_h - k.vz * vz 
    phi_cmd = -k.psi * delta_heading 

    aileron = k.p * p  + k.phi*(phi - phi_cmd)
    elevator = - k.q * q - k.theta * (theta-theta_cmd)
    rudder = k.r * r - k.beta * beta

    return [aileron, elevator, rudder]


if __name__ == '__main__':

    env = gym.make("SteadyControlTask-v0")
    s, done = env.reset(), False

    info = ""
    while not done:
        action = get_action(env)
        env.sim.set_property_value(c.fcs_throttle_cmd_norm, 0.45)
        s_, r, done, info = env.step(action)
        s = s_

        t = env.sim.get_property_value(c.simulation_sim_time_sec)
        altitude = env.sim.get_property_value(c.position_h_sl_ft) * 0.3048

        pqr = env.sim.get_property_values(
                [
                    c.velocities_p_rad_sec,
                    c.velocities_q_rad_sec,
                    c.velocities_r_rad_sec,
                ]
                )
        
        print(f"t(s):{int(t)}, altitude(m):{altitude}, pqr:{pqr}")


    print(f"episode done with info:{info}")
