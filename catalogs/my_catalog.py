from enum import Enum
from gym.spaces import Box, Discrete
from my_jsbsim_env.catalogs.property import Property
from my_jsbsim_env.catalogs.jsbsim_catalog import JsbsimCatalog
from my_jsbsim_env.catalogs import utils
from numpy.linalg import norm


class MyCatalog(Property, Enum):
    """

    A class to define and store new properties not implemented in JSBSim

    """

    def update_delta_altitude(sim):
        value = sim.get_property_value(JsbsimCatalog.position_h_sl_ft) -  sim.get_property_value(MyCatalog.target_altitude_ft)
        sim.set_property_value(MyCatalog.delta_altitude, -value)

    def update_delta_heading(sim):
        value = sim.get_property_value(JsbsimCatalog.attitude_psi_deg)-sim.get_property_value(MyCatalog.target_heading_deg)
        sim.set_property_value(MyCatalog.delta_heading, value)

    def update_delta_vc(sim):
        value = sim.get_property_value(JsbsimCatalog.velocities_vc_fps) - sim.get_property_value(MyCatalog.target_vc_fps) 
        sim.set_property_value(MyCatalog.delta_vc_fps, value)

    def update_delta_roll(sim):
        value = sim.get_property_value(MyCatalog.target_roll_deg) - sim.get_property_value(JsbsimCatalog.attitude_phi_deg)
        sim.set_property_value(MyCatalog.delta_roll, value)

    @staticmethod
    def update_property_incr(sim, discrete_prop, prop, incr_prop):
        value = sim.get_property_value(discrete_prop)
        if value == 0:
            pass
        else:
            if value == 1:
                sim.set_property_value(prop, sim.get_property_value(prop) - sim.get_property_value(incr_prop))
            elif value == 2:
                sim.set_property_value(prop, sim.get_property_value(prop) + sim.get_property_value(incr_prop))
            sim.set_property_value(discrete_prop, 0)

    def update_throttle_cmd_dir(sim):
        MyCatalog.update_property_incr(
            sim, MyCatalog.throttle_cmd_dir, JsbsimCatalog.fcs_throttle_cmd_norm, MyCatalog.incr_throttle
        )

    def update_aileron_cmd_dir(sim):
        MyCatalog.update_property_incr(
            sim, MyCatalog.aileron_cmd_dir, JsbsimCatalog.fcs_aileron_cmd_norm, MyCatalog.incr_aileron
        )

    def update_elevator_cmd_dir(sim):
        MyCatalog.update_property_incr(
            sim, MyCatalog.elevator_cmd_dir, JsbsimCatalog.fcs_elevator_cmd_norm, MyCatalog.incr_elevator
        )

    def update_rudder_cmd_dir(sim):
        MyCatalog.update_property_incr(
            sim, MyCatalog.rudder_cmd_dir, JsbsimCatalog.fcs_rudder_cmd_norm, MyCatalog.incr_rudder
        )

    def update_detect_extreme_state(sim):
        """
        Check whether the simulation is going through excessive values before it returns NaN values.
        Store the result in detect_extreme_state property.
        """
        extreme_velocity = sim.get_property_value(JsbsimCatalog.velocities_eci_velocity_mag_fps) >= 1e10
        extreme_rotation = (
            norm(
                sim.get_property_values(
                    [
                        JsbsimCatalog.velocities_p_rad_sec,
                        JsbsimCatalog.velocities_q_rad_sec,
                        JsbsimCatalog.velocities_r_rad_sec,
                    ]
                )
            )
            >= 1000
        )
        extreme_acceleration = (
            max(
                [
                    abs(sim.get_property_value(JsbsimCatalog.accelerations_n_pilot_x_norm)),
                    abs(sim.get_property_value(JsbsimCatalog.accelerations_n_pilot_y_norm)),
                    abs(sim.get_property_value(JsbsimCatalog.accelerations_n_pilot_z_norm)),
                ]
            )
            > 1e1
        )  # acceleration larger than 10G
        sim.set_property_value(
            MyCatalog.detect_extreme_state,
            extreme_rotation or extreme_velocity or extreme_acceleration,
        )

    # position and attitude
    delta_altitude = Property(
        "position/delta-altitude-to-target-ft",
        "delta altitude to target [ft]",
        -40000,
        40000,
        access="R",
        update=update_delta_altitude,
    )

    delta_heading = Property(
        "position/delta-heading-to-target-deg",
        "delta heading to target [deg]",
        -180,
        180,
        access="R",
        update=update_delta_heading,
    )

    delta_vc_fps = Property(
        "tc/delta-vc-fps",
         "delta vc to target [fps]",
         - 1000,
         1000,
         access='R',
         update= update_delta_vc,
         )
  
    delta_roll = Property(
        "position/delta-roll-to-target-deg",
        "delta roll to target [deg]",
        -180,
        180,
        access="R",
        update=update_delta_roll,
    )

    # controls command

    throttle_cmd_dir = Property(
        "fcs/throttle-cmd-dir",
        "direction to move the throttle",
        0,
        2,
        spaces=Discrete,
        access="W",
        update=update_throttle_cmd_dir,
    )
    incr_throttle = Property("fcs/incr-throttle", "incrementation throttle", 0, 1)
    aileron_cmd_dir = Property(
        "fcs/aileron-cmd-dir",
        "direction to move the aileron",
        0,
        2,
        spaces=Discrete,
        access="W",
        update=update_aileron_cmd_dir,
    )
    incr_aileron = Property("fcs/incr-aileron", "incrementation aileron", 0, 1)
    elevator_cmd_dir = Property(
        "fcs/elevator-cmd-dir",
        "direction to move the elevator",
        0,
        2,
        spaces=Discrete,
        access="W",
        update=update_elevator_cmd_dir,
    )
    incr_elevator = Property("fcs/incr-elevator", "incrementation elevator", 0, 1)
    rudder_cmd_dir = Property(
        "fcs/rudder-cmd-dir",
        "direction to move the rudder",
        0,
        2,
        spaces=Discrete,
        access="W",
        update=update_rudder_cmd_dir,
    )
    incr_rudder = Property("fcs/incr-rudder", "incrementation rudder", 0, 1)

    # detect functions

    detect_extreme_state = Property(
        "detect/extreme-state",
        "detect extreme rotation, velocity and altitude",
        0,
        1,
        spaces=Discrete,
        access="R",
        update=update_detect_extreme_state,
    )

    # target conditions

    target_altitude_ft = Property(
        "tc/h-sl-ft",
        "target altitude MSL [ft]",
        JsbsimCatalog.position_h_sl_ft.min,
        JsbsimCatalog.position_h_sl_ft.max,
    )
    target_heading_deg = Property(
        "tc/target-heading-deg",
        "target heading [deg]",
        JsbsimCatalog.attitude_psi_deg.min,
        JsbsimCatalog.attitude_psi_deg.max,
    )
    target_roll_deg = Property(
        "tc/target-roll-deg",
        "target roll [deg]",
        180,
        -180,
    )

    target_vg = Property("tc/target-vg", "target ground velocity [ft/s]")
    target_vc_fps = Property("tc/target-airspeed", "target airspeed [fps]",0, 4400)
    target_time = Property("tc/target-time-sec", "target time [sec]", 0)
    t = Property("time-sec", "time [sec]", 0)
    target_latitude_geod_deg = Property("tc/target-latitude-geod-deg", "target geocentric latitude [deg]", -90, 90)
    target_longitude_geod_deg = Property(
        "tc/target-longitude-geod-deg", "target geocentric longitude [deg]", -180, 180
    )

    steady_flight = Property("steady_flight", "steady flight mode", 0, 1000000)