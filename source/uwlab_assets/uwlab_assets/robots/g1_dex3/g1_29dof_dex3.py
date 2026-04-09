# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ArticulationCfg for the Unitree G1-29DOF + Dex3 fixed-base robot.

USD source:
    $PROJECT_ROOT/assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd
    Fetched via unitree_sim_isaaclab/fetch_assets.sh from Hugging Face
    (unitreerobotics/unitree_sim_isaaclab_usds) into $PROJECT_ROOT/assets/.

The root joint is physically welded inside the USD ("base_fix" variant), so
PhysX treats the robot as fixed-base without needing fix_root_link=True in
the ArticulationCfg. The init_state.pos=(0, 0, 0.75) sets the world-space
position of the root body (approximately hip height), which determines where
the arms reach.

All five actuator groups (legs, waist, feet, arms, hands) are kept from the
upstream config even though the legs/waist/feet joints won't be actively
controlled. PhysX requires actuators declared for every joint in the USD
articulation tree; dropping them causes assertion errors on env init.

This file inlines the config from unitree_sim_isaaclab/robots/unitree.py
(G129_CFG_WITH_DEX3_BASE_FIX) to avoid depending on that repo being installed
as a Python package (it ships no __init__.py at the root or robots/ level).
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_project_root = os.environ.get("PROJECT_ROOT", "")

G1_DEX3_FIXED_BASE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{_project_root}/assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            # legs
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.05,
            "left_knee_joint": 0.2,
            "left_ankle_pitch_joint": -0.15,
            "left_ankle_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.05,
            "right_knee_joint": 0.2,
            "right_ankle_pitch_joint": -0.15,
            "right_ankle_roll_joint": 0.0,
            # waist
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            # arms
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
            # Dex3 fingers (both hands open)
            "left_hand_index_0_joint": 0.0,
            "left_hand_middle_0_joint": 0.0,
            "left_hand_thumb_0_joint": 0.0,
            "left_hand_index_1_joint": 0.0,
            "left_hand_middle_1_joint": 0.0,
            "left_hand_thumb_1_joint": 0.0,
            "left_hand_thumb_2_joint": 0.0,
            "right_hand_index_0_joint": 0.0,
            "right_hand_middle_0_joint": 0.0,
            "right_hand_thumb_0_joint": 0.0,
            "right_hand_index_1_joint": 0.0,
            "right_hand_middle_1_joint": 0.0,
            "right_hand_thumb_1_joint": 0.0,
            "right_hand_thumb_2_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
            armature=None,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
            effort_limit=1000.0,
            velocity_limit=0.0,
            stiffness={
                "waist_yaw_joint": 10000.0,
                "waist_roll_joint": 10000.0,
                "waist_pitch_joint": 10000.0,
            },
            damping={
                "waist_yaw_joint": 10000.0,
                "waist_roll_joint": 10000.0,
                "waist_pitch_joint": 10000.0,
            },
            armature=None,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness={
                ".*_shoulder_.*_joint": 300.0,
                ".*_elbow_joint": 400.0,
                ".*_wrist_.*_joint": 400.0,
            },
            damping={
                ".*_shoulder_.*_joint": 3.0,
                ".*_elbow_joint": 2.5,
                ".*_wrist_.*_joint": 2.5,
            },
            armature=None,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hand_index_.*_joint",
                ".*_hand_middle_.*_joint",
                ".*_hand_thumb_.*_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={".*": 100.0},
            damping={".*": 10.0},
            armature={".*": 0.1},
        ),
    },
)
