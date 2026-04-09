# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone Dex3 right-hand ArticulationCfg for grasp sampling.

Purpose
-------
Used by the Phase 2 grasp sampling env (Dex3GraspSamplingSceneCfg).
The grasp_sampling_event positions the hand by calling
  gripper_asset.write_root_pose_to_sim(candidate_pose, env_ids)
which sets the ROOT BODY of the articulation to each candidate pose.

For this to work the root body must BE the palm — not the pelvis of a
full robot.  This config uses a standalone hand USD (right_hand_palm_link
as articulation root) extracted from the full G1 by the extraction script:
  scripts/tools/extract_dex3_right_hand_usd.py

Generate the standalone USD once before running grasp sampling:
  cd /path/to/unitree_sim_isaaclab
  python3 /path/to/UWLab-g1/scripts/tools/extract_dex3_right_hand_usd.py

Comparison to ROBOTIQ_2F85 in ur5e_robotiq_2f85_gripper.py
-----------------------------------------------------------
Robotiq 2F85                     Dex3 right hand
----                             ----
1 joint (finger_joint)           7 joints (index×2, middle×2, thumb×3)
root body: robotiq_base_link     root body: right_hand_palm_link
disable_gravity=True             disable_gravity=True (same reason)
metadata.yaml co-located         metadata.yaml co-located (same reader)

Body name used by the recorder
-------------------------------
DEX3_PALM_BODY_NAME = "right_hand_palm_link"

This is the body whose world pose is recorded RELATIVE TO the cylinder at
each successful grasp.  Phase 3 uses these relative poses to initialise
the ObjectRestingEEGrasped reset state (full G1 arm placed so its palm
link is at the recorded pose relative to the cylinder).

Confirmed from the G1 Dex3 USD hierarchy
-----------------------------------------
  right_wrist_yaw_link
    └── right_hand_palm_joint  (FixedJoint → connects wrist to palm)
          └── right_hand_palm_link    ← this is our standalone root
                ├── right_hand_index_0_joint  → right_hand_index_0_link
                │     └── right_hand_index_1_joint → right_hand_index_1_link
                ├── right_hand_middle_0_joint → right_hand_middle_0_link
                │     └── right_hand_middle_1_joint → right_hand_middle_1_link
                └── right_hand_thumb_0_joint  → right_hand_thumb_0_link
                      └── right_hand_thumb_1_joint → right_hand_thumb_1_link
                            └── right_hand_thumb_2_joint → right_hand_thumb_2_link

Note: right_hand_camera_base_link and right_hand_camera_base_joint are
intentionally excluded from the standalone USD.  The camera is a perception
sensor not needed in simulation, and its joint has no actuator group —
leaving it undriven (stiffness=0) caused the camera module to detach and
spin freely in PhysX.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_project_root = os.environ.get("PROJECT_ROOT", "")

# Path to the standalone hand USD produced by extract_dex3_right_hand_usd.py.
# The extraction script writes to $PROJECT_ROOT/assets/robots/dex3-right-hand-usd/.
_dex3_hand_usd_path = os.path.join(
    _project_root, "assets/robots/dex3-right-hand-usd/dex3_right_hand.usd"
)

# Palm body name confirmed from the G1 Dex3 USD hierarchy.
# Used in grasp_sampling_cfg.py for gripper_cfg and by GraspRelativePoseRecorder.
DEX3_PALM_BODY_NAME = "right_hand_palm_link"

DEX3_RIGHT_STANDALONE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_dex3_hand_usd_path,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Gravity must be off during teleportation (grasp_sampling_event
            # resets the hand to a candidate pose each episode).
            # global_physics_control_event re-enables gravity from t=1s onward
            # so the grasp is physically tested under real load.
            disable_gravity=True,
            retain_accelerations=False,
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
        # Start above OBJECT_SPAWN_HEIGHT=0.3m.  This is only the position
        # before the first reset; grasp_sampling_event teleports the palm on
        # every episode reset.
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            # Fully open hand: all finger joints at 0.
            # Matches the open_command_expr in G1_DEX3_RIGHT_HAND_BINARY.
            "right_hand_index_0_joint": 0.0,
            "right_hand_index_1_joint": 0.0,
            "right_hand_middle_0_joint": 0.0,
            "right_hand_middle_1_joint": 0.0,
            "right_hand_thumb_0_joint": 0.0,
            "right_hand_thumb_1_joint": 0.0,
            "right_hand_thumb_2_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Only hand actuators — no arm groups.
        # Gains match the "hands" group in G1_DEX3_FIXED_BASE_CFG so finger
        # behaviour in grasp sampling is consistent with the full RL env.
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_hand_index_.*_joint",
                "right_hand_middle_.*_joint",
                "right_hand_thumb_.*_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={".*": 100.0},
            damping={".*": 10.0},
            armature={".*": 0.1},
        ),
    },
)
