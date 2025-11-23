# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for the Agibot G1 humanoid robots.

The following configurations are available:

* :obj:`AGIBOT_G1_CFG`: Agibot G1 robot


"""

import os

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

TIENKUNG_2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/data/weyan/tiangong2_0_EVT_with_6d_force_sensor_hand_v3/tiangong2_0_EVT_with_6d_force_sensor_hand_v3.usd",
        #usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/X-Humanoid/tiangong2/tiangong2_0_EVT_with_6d_force_sensor_hand_v3.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,  ## set to True for easy debugging of gripper behavior
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  ## If set to True, the robot will move some other joints to avoid collision with itself.
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Lift joint initial position
            "body_yaw_joint":0.0,
            ## Head joints initial position
           # "head_yaw_joint": 0.0,
           # "head_pitch_joint": 0.0,
           # "head_roll_joint": 0.0,
            # Left arm joints initial position
            "L_joint_[0-6]":0.0,
            "R_joint_[0-6]": 0.0,
            # Left and right gripper initial position
            "L_Joint_[0-4][0-1]": 0.0,
            "L_Joint_4[2-3]": 0.0,
            "R_Joint_[0-4][0-1]": 0.0,
            "R_Joint_4[2-3]": 0.0,
            "hip_.*": 0.0,
            "knee_.*": 0.0,
            "ankle_.*": 0.0,
        },
        pos=(-0.6, 0.0, -1.05),  ## init pos of the articulation for teleop
    ),
    actuators={
        # Body lift and torso actuators
        "body": ImplicitActuatorCfg(
            joint_names_expr=["body_yaw_joint"],
            effort_limit_sim=10000.0,
            velocity_limit_sim=2.61,
            stiffness=10000000.0,
            damping=200.0,
        ),
        ## Head actuators
 #       "head": ImplicitActuatorCfg(
 #           joint_names_expr=["head_yaw_joint", "head_pitch_joint", "head_roll_joint"],
 #           effort_limit_sim=50.0,
 #           velocity_limit_sim=1.0,
 #           stiffness=80.0,
 #           damping=4.0,
 #       ),
        # Left arm actuator
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["L_joint_[0-6]"],
            effort_limit_sim=300,
            velocity_limit_sim=100,
            stiffness=None,#{
             #   "L_joint_0": 1000000,
             #   "L_joint_1": 1000000,
             #   "L_joint_2": 1000000,
             #   "L_joint_3": 1000000,
             #   "L_joint_4": 1000000,
             #   "L_joint_5": 1000000,
             #   "L_joint_6": 1000000,
             #   },
            damping=None,#{
             #   "L_joint_0": 1000,
             #   "L_joint_1": 1000,
             #   "L_joint_2": 1000,
             #   "L_joint_3": 1000,
             #   "L_joint_4": 1000,
             #   "L_joint_5": 1000,
             #   "L_joint_6": 1000,
             #   },
        ),
        # Right arm actuator
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["R_joint_[0-6]"],
            effort_limit_sim=300,
            velocity_limit_sim=100,
            stiffness=None,
            damping=None,
        ),
        "left_hand": ImplicitActuatorCfg(
            joint_names_expr=["L_Joint_[0-4][0-1]", "L_Joint_4[2-3]"],
            effort_limit_sim=None,
            velocity_limit_sim=None,
            stiffness=None,
            damping=None,
        ),
        "right-hand": ImplicitActuatorCfg(
            joint_names_expr=["R_Joint_[0-4][0-1]", "R_Joint_4[2-3]"],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_.*",
                "knee_.*",
                "ankle_.*",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
