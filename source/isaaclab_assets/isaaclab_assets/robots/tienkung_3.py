# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for the TienKung-3 humanoid robots.

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

TIENKUNG_3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/yanwei/tiangong3_dex_brainco2/tiangong3_dex_brainco2.usd",
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
            # Left arm joints initial position
            "shoulder_pitch_r_joint": 0.0,
            "shoulder_roll_r_joint": 0.0,
            "shoulder_yaw_r_joint": 0.0,
            "elbow_pitch_r_joint": 0.0,
            "elbow_yaw_r_joint": 0.0,
            "wrist_pitch_r_joint": 0.0,
            "wrist_roll_r_joint": 0.0,
            # left-arm
            "shoulder_pitch_l_joint": 0.0,
            "shoulder_roll_l_joint": 0.0,
            "shoulder_yaw_l_joint": 0.0,
            "elbow_pitch_l_joint": 0.0,
            "elbow_yaw_l_joint": 0.0,
            "wrist_pitch_l_joint": 0.0,
            "wrist_roll_l_joint": 0.0,
            # Left and right gripper initial position
            "left_tcp_joint": 0.0,
            "right_tcp_joint": 0.0,
            "left_index_proximal_joint": 0.0,
            "left_middle_proximal_joint": 0.0,
            "left_pinky_proximal_joint": 0.0,
            "left_ring_proximal_joint": 0.0,
            "left_thumb_metacarpal_joint": 0.0,
            "left_index_distal_joint": 0.0,
            "left_index_tip_joint": 0.0,
            #"left_index_touch_joint": 0.0,
            "left_middle_distal_joint": 0.0,
            "left_middle_tip_joint": 0.0,
            #"left_middle_touch_joint": 0.0,
            "left_pinky_distal_joint": 0.0,
            "left_pinky_tip_joint": 0.0,
            #"left_pinky_touch_joint": 0.0,
            "left_ring_distal_joint": 0.0,
            "left_ring_tip_joint": 0.0,
            #"left_ring_touch_joint": 0.0,
            "left_thumb_proximal_joint": 0.0,
            "left_thumb_distal_joint": 0.0,
            "left_thumb_tip_joint": 0.0,
            #"left_thumb_touch_joint": 0.0,
            "right_index_proximal_joint": 0.0,
            "right_middle_proximal_joint": 0.0,
            "right_pinky_proximal_joint": 0.0,
            "right_ring_proximal_joint": 0.0,
            "right_thumb_metacarpal_joint": 0.0,
            "right_index_distal_joint": 0.0,
            "right_index_tip_joint": 0.0,
            #"right_index_touch_joint": 0.0,
            "right_middle_distal_joint": 0.0,
            "right_middle_tip_joint": 0.0,
            #"right_middle_touch_joint": 0.0,
            "right_pinky_distal_joint": 0.0,
            "right_pinky_tip_joint": 0.0,
            #"right_pinky_touch_joint": 0.0,
            "right_ring_distal_joint": 0.0,
            "right_ring_tip_joint": 0.0,
            #"right_ring_touch_joint": 0.0,
            "right_thumb_proximal_joint": 0.0,
            "right_thumb_distal_joint": 0.0,
            "right_thumb_tip_joint": 0.0,
            #"right_thumb_touch_joint": 0.0,
            "hip_.*": 0.0,
            "knee_.*": 0.0,
            "ankle_.*": 0.0,
            "waist_*": 0.0,
            "imu_joint": 0.0,
            "RGB_head_*": 0.0,
            "camera_*": 0.0,
            "radar_head": 0.0,
            "head_*": 0.0,
        },
        pos=(-0.6, 0.0, -1.05),  ## init pos of the articulation for teleop
    ),
    actuators={
        # Body lift and torso actuators
        "body": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
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
            joint_names_expr=["shoulder_pitch_l_joint", "shoulder_yaw_l_joint", "shoulder_roll_l_joint", "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint"],
            effort_limit_sim=300,
            velocity_limit_sim=100,
            stiffness=None,
            damping=None,
        ),
        # Right arm actuator
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pitch_r_joint", "shoulder_yaw_r_joint", "shoulder_roll_r_joint", "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint"],
            effort_limit_sim=300,
            velocity_limit_sim=100,
            stiffness=None,
            damping=None,
        ),
        "left_hand": ImplicitActuatorCfg(
            joint_names_expr=[
               "left_index_proximal_joint",
               "left_middle_proximal_joint",
               "left_pinky_proximal_joint",
               "left_ring_proximal_joint",
               "left_thumb_metacarpal_joint",
               "left_index_distal_joint",
               #"left_index_tip_joint",
               #"left_index_touch_joint",
               "left_middle_distal_joint",
               #"left_middle_tip_joint",
               #"left_middle_touch_joint",
               "left_pinky_distal_joint",
               #"left_pinky_tip_joint",
               #"left_pinky_touch_joint",
               "left_ring_distal_joint",
               #"left_ring_tip_joint",
               #"left_ring_touch_joint",
               "left_thumb_proximal_joint",
               "left_thumb_distal_joint",
               #"left_thumb_tip_joint",
               #"left_thumb_touch_joint",
            ],
            effort_limit_sim=None,
            velocity_limit_sim=None,
            stiffness=None,
            damping=None,
        ),
        "right-hand": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_index_proximal_joint",
                "right_middle_proximal_joint",
                "right_pinky_proximal_joint",
                "right_ring_proximal_joint",
                "right_thumb_metacarpal_joint",
                "right_index_distal_joint",
                #"right_index_tip_joint",
                #"right_index_touch_joint",
                "right_middle_distal_joint",
                #"right_middle_tip_joint",
                #"right_middle_touch_joint",
                "right_pinky_distal_joint",
                #"right_pinky_tip_joint",
                #"right_pinky_touch_joint",
                "right_ring_distal_joint",
                #"right_ring_tip_joint",
                #"right_ring_touch_joint",
                "right_thumb_proximal_joint",
                "right_thumb_distal_joint",
                #"right_thumb_tip_joint",
                #"right_thumb_touch_joint",
                ],
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
