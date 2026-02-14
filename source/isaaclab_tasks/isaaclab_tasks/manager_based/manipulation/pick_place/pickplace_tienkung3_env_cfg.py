# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import torch

import carb
from pink.tasks import FrameTask

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import ManusViveCfg, OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.x_humanoid.tienkung3_retargeter import TienKung3RetargeterCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from . import mdp_tienkung3 as mdp

from isaaclab_assets.robots.tienkung_3 import TIENKUNG_3_CFG  # isort: skip


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):

    # Table
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # Object
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.35, 0.40, 1.0413], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CylinderCfg(
            radius=0.018,
            height=0.35,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15), metallic=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
        ),
    )

    # Humanoid robot w/ arms higher
    robot: ArticulationCfg = TIENKUNG_3_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.0),
            rot=(0.7071, 0, 0, 0.7071),
            #rot=(0.7071, 0, 0, 0.7071),
            joint_pos={
                # Lift joint initial position
               # "body_yaw_joint":0.0,
                ## Head joints initial position
                # Left arm joints initial position
                "shoulder_pitch_r_joint": 0.0,
                "shoulder_roll_r_joint": 0.0,
                "shoulder_yaw_r_joint": 0.0,
                "elbow_pitch_r_joint": -1.57,
                "elbow_yaw_r_joint": 0.0,
                "wrist_pitch_r_joint": 0.0,
                "wrist_roll_r_joint": 0.0,
                # left-arm
                "shoulder_pitch_l_joint": 0.0,
                "shoulder_roll_l_joint": 0.0,
                "shoulder_yaw_l_joint": 0.0,
                "elbow_pitch_l_joint": -1.57,
                "elbow_yaw_l_joint": 0.0,
                "wrist_pitch_l_joint": 0.0,
                "wrist_roll_l_joint": 0.0,
               # "left_tcp_joint": 0.0,
               # "right_tcp_joint": 0.0,
                "left_index_proximal_joint": 0.0,
                "left_middle_proximal_joint": 0.0,
                "left_pinky_proximal_joint": 0.0,
                "left_ring_proximal_joint": 0.0,
                "left_thumb_metacarpal_joint": 0.0,
                "left_index_distal_joint": 0.0,
                #"left_index_tip_joint": 0.0,
                #"left_index_touch_joint": 0.0,
                "left_middle_distal_joint": 0.0,
                #"left_middle_tip_joint": 0.0,
                #"left_middle_touch_joint": 0.0,
                "left_pinky_distal_joint": 0.0,
                #"left_pinky_tip_joint": 0.0,
                #"left_pinky_touch_joint": 0.0,
                "left_ring_distal_joint": 0.0,
                #"left_ring_tip_joint": 0.0,
                #"left_ring_touch_joint": 0.0,
                "left_thumb_proximal_joint": 0.0,
                "left_thumb_distal_joint": 0.0,
                #"left_thumb_tip_joint": 0.0,
                #"left_thumb_touch_joint": 0.0,
                "right_index_proximal_joint": 0.0,
                "right_middle_proximal_joint": 0.0,
                "right_pinky_proximal_joint": 0.0,
                "right_ring_proximal_joint": 0.0,
                "right_thumb_metacarpal_joint": 0.0,
                "right_index_distal_joint": 0.0,
               # "right_index_tip_joint": 0.0,
               #"right_index_touch_joint": 0.0,
                "right_middle_distal_joint": 0.0,
                #"right_middle_tip_joint": 0.0,
                #"right_middle_touch_joint": 0.0,
                "right_pinky_distal_joint": 0.0,
                #"right_pinky_tip_joint": 0.0,
                #"right_pinky_touch_joint": 0.0,
                "right_ring_distal_joint": 0.0,
                #"right_ring_tip_joint": 0.0,
                #"right_ring_touch_joint": 0.0,
                "right_thumb_proximal_joint": 0.0,
                "right_thumb_distal_joint": 0.0,
                #"right_thumb_tip_joint": 0.0,
                #"right_thumb_touch_joint": 0.0,
                "hip_.*": 0.0,
                "knee_.*": 0.0,
                "ankle_.*": 0.0,
                #"waist_*": 0.0,
                #"imu_joint": 0.0,
                #"RGB_head_*": 0.0,
                #"camera_*": 0.0,
                #"radar_head": 0.0,
                #"head_*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    upper_body_ik = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[
            # right-arm,
            "shoulder_pitch_r_joint",
            "shoulder_roll_r_joint",
            "shoulder_yaw_r_joint",
            "elbow_pitch_r_joint",
            "elbow_yaw_r_joint",
            "wrist_pitch_r_joint",
            "wrist_roll_r_joint",
            # left-arm
            "shoulder_pitch_l_joint",
            "shoulder_roll_l_joint",
            "shoulder_yaw_l_joint",
            "elbow_pitch_l_joint",
            "elbow_yaw_l_joint",
            "wrist_pitch_l_joint",
            "wrist_roll_l_joint",
        ],

        hand_joint_names=[
            "left_index_proximal_joint",
            "left_middle_proximal_joint",
            "left_pinky_proximal_joint",
            "left_ring_proximal_joint",
            "left_thumb_metacarpal_joint",
            "right_index_proximal_joint",
            "right_middle_proximal_joint",
            "right_pinky_proximal_joint",
            "right_ring_proximal_joint",
            "right_thumb_metacarpal_joint",
            "left_index_distal_joint",
            "left_middle_distal_joint",
            "left_pinky_distal_joint",
            "left_ring_distal_joint",
            "left_thumb_proximal_joint",
            "right_index_distal_joint",
            "right_middle_distal_joint",
            "right_pinky_distal_joint",
            "right_ring_distal_joint",
            "right_thumb_proximal_joint",
            "left_thumb_distal_joint",
            "right_thumb_distal_joint",
            #"left_thumb_tip_joint",
            #"right_thumb_tip_joint",
        ],

        target_eef_link_names={
            "left_wrist" : "wrist_roll_l_link",
            "right_wrist" : "wrist_roll_r_link",
        },
        # the robot in the sim scene we are controlling
        asset_name="robot",
        # Configuration for the IK controller
        # The frames names are the ones present in the URDF file
        # The urdf has to be generated from the USD that is being used in the scene
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="pelvis",
            num_hand_joints=22,
            show_ik_warnings=True,
            variable_input_tasks=[
                FrameTask(
                    "wrist_roll_l_link",
                    position_cost=1.0,  # [cost] / [m]
                    orientation_cost=1.0,  # [cost] / [rad]
                    lm_damping=10,  # dampening for solver for step jumps
                    gain=0.1,
                ),
                FrameTask(
                    "wrist_roll_r_link",
                    position_cost=1.0,  # [cost] / [m]
                    orientation_cost=1.0,  # [cost] / [rad]
                    lm_damping=10,  # dampening for solver for step jumps
                    gain=0.1,
                ),
                NullSpacePostureTask(
                    cost=0.5,
                    lm_damping=1,
                    controlled_frames=[
                        "wrist_roll_l_link",
                        "wrist_roll_r_link",
                    ],
                    controlled_joints=[
                        "shoulder_pitch_l_joint",
                        "shoulder_roll_l_joint",
                        "shoulder_yaw_l_joint",
                        "shoulder_pitch_r_joint",
                        "shoulder_roll_r_joint",
                        "shoulder_yaw_r_joint",
                        "waist_yaw_joint",
                        "waist_pitch_joint",
                        "waist_roll_joint",
                    ],
                    gain=0.3,
                ),
            ],
            fixed_input_tasks=[],
            xr_enabled=bool(carb.settings.get_settings().get("/app/xr/enabled")),
        ),
        enable_gravity_compensation=False,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        robot_links_state = ObsTerm(func=mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=mdp.get_left_eef_pos)
        left_eef_quat = ObsTerm(func=mdp.get_left_eef_quat)
        right_eef_pos = ObsTerm(func=mdp.get_right_eef_pos)
        right_eef_quat = ObsTerm(func=mdp.get_right_eef_quat)

        hand_joint_state = ObsTerm(func=mdp.get_hand_state)
        head_joint_state = ObsTerm(func=mdp.get_head_state)

        object = ObsTerm(func=mdp.object_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(func=mdp.task_done)


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.05, 0.0],
                "y": [0.0, 0.05],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PickPlaceTienKung3EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the GR1T2 environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Position of the XR anchor in the world frame
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()
    NUM_OPENXR_HAND_JOINTS = 26

    # Idle action to hold robot in default pose
    # Action format: [left arm pos (3), left arm quat (4), right arm pos (3), right arm quat (4),
    #                 left hand joint pos (11), right hand joint pos (11)]
    idle_action = torch.tensor([
        -0.22878,
        0.2536,
        1.0953,
        0.5,
        0.5,
        -0.5,
        0.5,
        0.22878,
        0.2536,
        1.0953,
        0.5,
        0.5,
        -0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ])

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 60  # 100Hz
        self.sim.render_interval = 2

        # Convert USD to URDF and change revolute joints to fixed
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )

#        temp_urdf_output_path = "/tmp/urdf/tiangong2_0_EVT_with_6d_force_sensor_hand_v3.urdf"
#        temp_urdf_meshes_output_path = "/tmp/meshes"

        # Set the URDF and mesh paths for the IK controller
        self.actions.upper_body_ik.controller.urdf_path = temp_urdf_output_path
        self.actions.upper_body_ik.controller.mesh_path = temp_urdf_meshes_output_path

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        TienKung3RetargeterCfg(
                            enable_visualization=True,
                            # number of joints in both hands
                            num_open_xr_hand_joints=2 * self.NUM_OPENXR_HAND_JOINTS,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.upper_body_ik.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "manusvive": ManusViveCfg(
                    retargeters=[
                        TienKung3RetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * 26,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.upper_body_ik.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )

