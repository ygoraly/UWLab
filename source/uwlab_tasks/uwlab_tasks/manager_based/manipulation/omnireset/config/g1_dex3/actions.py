# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action configs for the G1-29DOF + Dex3 pick-cylinder environment.

Action space (total 7 dims):
  arm     — DifferentialInverseKinematicsActionCfg  (6-dim: dx, dy, dz, droll, dpitch, dyaw)
  gripper — BinaryJointPositionActionCfg            (1-dim: binary open/close scalar)

Comparison to the UR5e version
-------------------------------
The UR5e config uses RelCartesianOSCActionCfg, which wraps a custom analytical
Jacobian computed from the UR5e's specific DH parameters and applies a task-space
PD controller whose gains match the real robot's OSC implementation — that detail
is specifically for sim2real alignment with the DelayedPDActuator.

DifferentialInverseKinematicsActionCfg is Isaac Lab's robot-agnostic DiffIK: it
uses the numerical Jacobian from PhysX, sends joint *position* targets to the
actuators each step, and works with ImplicitActuatorCfg. No sim2real gain-matching
is needed for training from scratch in sim.

The Robotiq 2F-85 gripper has one joint (finger_joint) so its binary action was
trivial. The Dex3 has 7 joints per hand, so the binary action specifies a target
position for each. Close targets below are approximate starting values — inspect
visually after the env loads and tune before Phase 2 if needed.
"""

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Right arm — Differential IK
# ---------------------------------------------------------------------------
# scale=(tx, ty, tz, rx, ry, rz): 5 cm / unit for translation, 0.1 rad / unit
# for rotation. More conservative than the UR5e's 2 cm translation scale because
# the G1 arm is longer and Cartesian sensitivity at the EE is higher near
# mid-range configurations.
# ik_method="dls" (damped least-squares) is preferred over "pinv" because the
# G1 arm can reach singular configurations (e.g. fully extended) during random
# resets; damping prevents torque spikes at those points.
G1_RIGHT_ARM_DIFF_IK = DifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=[
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    body_name="right_wrist_yaw_link",
    controller=DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,
        ik_method="dls",
        ik_params={"lambda_val": 0.01},
    ),
    scale=(0.05, 0.05, 0.05, 0.1, 0.1, 0.1),
)

# ---------------------------------------------------------------------------
# Right Dex3 hand — binary open / close
# ---------------------------------------------------------------------------
# The Dex3 has 7 joints per hand (vs 1 for the Robotiq finger_joint).
# Open = all zeros, matching the USD default init_state so the robot starts
# with an open hand and reset_scene_to_default produces a consistent state.
# Close values are rough estimates (index/middle curl to ~1.0 rad proximal,
# ~0.8 rad distal; thumb more conservative at 0.5 rad). Verify with the
# sim viewer after env loads and adjust before Phase 2 grasp sampling.
G1_DEX3_RIGHT_HAND_BINARY = BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=[
        "right_hand_index_0_joint",
        "right_hand_middle_0_joint",
        "right_hand_thumb_0_joint",
        "right_hand_index_1_joint",
        "right_hand_middle_1_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
    ],
    open_command_expr={
        "right_hand_index_0_joint": 0.0,
        "right_hand_middle_0_joint": 0.0,
        "right_hand_thumb_0_joint": 0.0,
        "right_hand_index_1_joint": 0.0,
        "right_hand_middle_1_joint": 0.0,
        "right_hand_thumb_1_joint": 0.0,
        "right_hand_thumb_2_joint": 0.0,
    },
    close_command_expr={
        "right_hand_index_0_joint": 1.0,
        "right_hand_middle_0_joint": 1.0,
        "right_hand_thumb_0_joint": 0.0,  # opposition — rotate thumb to face fingers
        "right_hand_index_1_joint": 0.8,
        "right_hand_middle_1_joint": 0.8,
        "right_hand_thumb_1_joint": -0.3,  # proximal flexion — main curl
        "right_hand_thumb_2_joint": -0.8,  # distal flexion — tip
    },
)


# ---------------------------------------------------------------------------
# Combined action group (passed to ManagerBasedRLEnvCfg.actions)
# ---------------------------------------------------------------------------
@configclass
class G1Dex3PickAction:
    """Right arm DiffIK (6-dim) + right Dex3 binary hand (1-dim) = 7-dim total."""

    arm = G1_RIGHT_ARM_DIFF_IK
    gripper = G1_DEX3_RIGHT_HAND_BINARY
