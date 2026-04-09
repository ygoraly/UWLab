# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-states recording environment configs for the G1-29DOF + Dex3 pick-cylinder task.

Purpose
-------
This file defines the environment configurations used to record two reset-state
datasets that are consumed by MultiResetManager during RL training (Phase 4):

  ObjectAnywhereEEAnywhere   -- cylinder anywhere on/above table, arm anywhere reachable.
                                No grasp dataset required.  Cheap to record.
  ObjectRestingEEGrasped     -- cylinder resting on table, arm actively holding it.
                                Requires grasps.pt from Phase 2.

Comparison to ur5e_robotiq_2f85/reset_states_cfg.py
-----------------------------------------------------
Robot    : IMPLICIT_UR5E_ROBOTIQ_2F85          → G1_DEX3_FIXED_BASE_CFG
Object   : insertive_object = peg.usd           → object = CylinderCfg (same as rl_state_cfg.py)
           receptive_object = peg_hole.usd      → removed (no assembly target)
Table    : PAT Vention USD (kinematic)          → simple CuboidCfg (same as rl_state_cfg.py)
Support  : ur5_metal_support USD                → removed (G1 is fixed-base, no pedestal)
Variants : dict of 6 USD swaps                  → removed (cylinder only)
Subclasses: 6 (incl. PartiallyAssembled*)      → 2 (ObjectAnywhereEEAnywhere,
                                                      ObjectRestingEEGrasped)
"""

from __future__ import annotations

import numpy as np
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from uwlab_assets.robots.g1_dex3 import G1_DEX3_FIXED_BASE_CFG

from .actions import G1Dex3PickAction
from ... import mdp as task_mdp


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@configclass
class G1Dex3ResetStatesSceneCfg(InteractiveSceneCfg):
    """Scene for G1 reset-state recording.

    The scene is intentionally kept identical to G1Dex3SceneCfg (rl_state_cfg.py)
    so that state snapshots recorded here replay correctly in the training env:
    same table geometry → same table-surface z-height → cylinder does not float
    or clip through when a recorded state is loaded in training.

    Compared to ResetStatesSceneCfg (UR5e):

    robot
        IMPLICIT_UR5E_ROBOTIQ_2F85 (arm + Robotiq gripper) →
        G1_DEX3_FIXED_BASE_CFG (full 29-DOF arm + Dex3 right hand, fix_root_link=True).
        The full robot is required here (not the standalone hand) because
        reset_end_effector_round_fixed_asset and reset_end_effector_from_grasp_dataset
        both instantiate a DiffIK solver over the 6-DOF arm chain; those arm joints
        are absent from the standalone hand USD.

    object  (was insertive_object)
        peg.usd RigidObjectCfg, mass=0.001 kg →
        CylinderCfg primitive, radius=0.025 m, height=0.10 m, mass=0.05 kg.
        Asset key renamed from "insertive_object" to "object" to match rl_state_cfg.py;
        MultiResetManager and _reset_to look up assets by this key when replaying states.

    receptive_object  →  removed
        The pick task has no assembly target, so the peg-hole USD is dropped
        entirely.  Keeping it would (a) require updating every SceneEntityCfg
        reference, (b) cause MultiResetManager to crash at Phase 4 when it tries
        to read env.scene["receptive_object"], and (c) waste VRAM.

    table
        PAT Vention USD (complex mesh, kinematic) →
        simple CuboidCfg(size=(0.6, 0.8, 0.85)), kinematic=True.
        Centre at pos=(0.5, 0.0, 0.425) → top surface at z=0.85 m.
        Cylinder init z = table_top + half_height = 0.85 + 0.05 = 0.90 m,
        consistent with rl_state_cfg.py's init_state.pos=(0.5, 0.0, 0.90).

    ur5_metal_support  →  removed
        The UR5e needed this pedestal so that reset_robot_pose could jitter the
        robot base and its mounting plate together via offset_asset_cfg.  The G1
        base is fixed (fix_root_link=True), so the pedestal and the joint-base
        jitter term are both unnecessary.
    """

    # -- Robot ----------------------------------------------------------------
    robot = G1_DEX3_FIXED_BASE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # -- Task object ----------------------------------------------------------
    # Identical to G1Dex3SceneCfg in rl_state_cfg.py.
    # mass=0.05 kg (realistic cylinder) vs. peg.usd mass=0.001 kg in the UR5e
    # version.  The higher mass makes the stability check in
    # check_reset_state_success more meaningful: a 50-gram cylinder will not
    # stay in-hand unless the IK solution actually achieved the grasp pose.
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CylinderCfg(
            radius=0.025,
            height=0.10,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.8), metallic=0.1),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=1.5,
                dynamic_friction=1.5,
                restitution=0.0,
            ),
        ),
        # Cylinder centre at table_top (0.85) + half_height (0.05) = 0.90 m.
        # This is the default/fallback position; reset events override it each
        # episode.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.90), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # -- Table ----------------------------------------------------------------
    # Identical to G1Dex3SceneCfg in rl_state_cfg.py.
    # Centre at z=0.425 → top surface at z=0.85 m.
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.8, 0.85),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.6, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.425), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # -- Environment dressing -------------------------------------------------
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=10000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ---------------------------------------------------------------------------
# Base events (shared by both recording variants)
# ---------------------------------------------------------------------------

@configclass
class G1Dex3ResetStatesBaseEventCfg:
    """Startup and per-episode reset events shared by all recording variants.

    Compared to ResetStatesBaseEventCfg (UR5e):

    reset_robot_material  (kept)
        Unchanged in purpose: sets a single, fixed, low-friction value on every
        robot body at env startup so the stability check in
        check_reset_state_success is deterministic across runs.  num_buckets=1
        means all bodies share the same material (no per-body sampling).

    insertive_object_material  →  object_material  (renamed + make_consistent changed)
        Same low fixed friction as the UR5e version.  make_consistent is set to
        False (vs True in UR5e) because CylinderCfg is a PhysX primitive and
        the multi-body material bucketing that make_consistent=True relies on
        does not apply to primitive shapes; using True here can trigger PhysX
        assertions (same pattern as rl_state_cfg.py BaseEventCfg).

    receptive_object_material  →  removed
        No receptive object in the scene.

    reset_everything  (kept)
        Resets all assets back to their init_state before every recording
        episode.  Must run before any per-variant placement events so those
        events start from a clean baseline.

    reset_robot_pose  →  removed
        The UR5e version jittered the robot base position by ±1 cm and the
        mounting plate together via asset_cfgs + offset_asset_cfg.  With the G1
        fix_root_link=True, the robot body is anchored to the world frame and
        cannot be moved; the mounting-plate pedestal is also absent.

    reset_receptive_object_pose  →  removed
        No receptive object.
    """

    # -- Startup: deterministic friction for stable recording -----------------

    reset_robot_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.3),
            "dynamic_friction_range": (0.2, 0.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "asset_cfg": SceneEntityCfg("robot"),
            "make_consistent": True,
        },
    )

    # CylinderCfg is a PhysX primitive: make_consistent=False avoids a PhysX
    # assertion that fires when the multi-body bucketing path is taken for
    # non-mesh prims.  See the same note in rl_state_cfg.py BaseEventCfg.
    object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.3),
            "dynamic_friction_range": (0.2, 0.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "asset_cfg": SceneEntityCfg("object"),
            "make_consistent": False,
        },
    )

    # -- Per-episode: full scene reset ----------------------------------------

    # Must be declared last in this base class so that subclass reset events
    # (object placement, EE placement) are appended *after* it in declaration
    # order.  IsaacLab executes mode="reset" EventTerms in field-declaration
    # order, so reset_everything must run first to restore init_state before
    # the placement events apply their offsets on top.
    reset_everything = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset", params={})


# ---------------------------------------------------------------------------
# ObjectAnywhereEEAnywhere events
# ---------------------------------------------------------------------------

@configclass
class ObjectAnywhereEEAnywhereEventCfg(G1Dex3ResetStatesBaseEventCfg):
    """Place cylinder anywhere on/above the table, arm anywhere in its reachable workspace.

    No grasp dataset is required.  Almost every IK attempt succeeds because
    the EE target is any reachable pose, not a specific grasp pose.  This
    makes it the cheapest reset type to record and the first one to collect.

    Compared to ObjectAnywhereEEAnywhereEventCfg (UR5e):

    reset_insertive_object_pose  →  reset_object_pose
        Asset key: "insertive_object" → "object".
        offset_asset_cfg + use_bottom_offset removed: the UR5e version placed
        the peg relative to the ur5_metal_support base, using a "bottom_offset"
        from the peg's USD metadata to align peg-bottom to table-surface.
        CylinderCfg has no USD metadata file, so use_bottom_offset would crash
        on usd_path access in reset_root_states_uniform.__init__.  Instead the
        pose_range values are offsets from the cylinder's init_state.pos =
        (0.5, 0.0, 0.90):
            x: (-0.15, +0.15)  →  world x ∈ [0.35, 0.65]  (within 0.6 m table)
            y: (-0.25, +0.25)  →  world y ∈ [-0.25, +0.25] (within 0.8 m table)
            z: (0.0, +0.30)    →  world z ∈ [0.90, 1.20]   (on table to 30 cm above)
        Full RPY randomisation retained — "anywhere" means any orientation.

    reset_end_effector_pose
        joint_names: ["shoulder.*","elbow.*","wrist.*"]  →  ["right_shoulder.*",
            "right_elbow.*","right_wrist.*"]  (7 right-arm joints; side prefix
            needed because the G1 USD has both left and right arms).
        body_names: "robotiq_base_link"  →  "right_hand_palm_link"  (Dex3 palm;
            the body the grasp dataset is expressed relative to, so all IK-based
            resets use the same reference frame).
        fixed_asset_offset: None  →  uses robot root_pos_w = (0, 0, 0.75) as
            the reference centre (same as UR5e which also used None).
        pose_range_b adjusted for G1 geometry: offsets are added to root_pos_w,
        so with root at z = 0.75 m:
            x: (0.20, 0.70)  →  world x ∈ [0.20, 0.70]   (forward reach)
            y: (-0.40, 0.40) →  world y ∈ [-0.40, 0.40]  (lateral reach; same as UR5e)
            z: (0.10, 0.45)  →  world z ∈ [0.85, 1.20]   (table top to above table)
        Pitch/yaw sampling kept identical to UR5e — they sample the approach
        cone that keeps the hand in a plausible downward/forward orientation.
    """

    reset_object_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {
                # Offsets from init_state.pos = (0.5, 0.0, 0.90).
                "x": (-0.15, -0.05),
                "y": (-0.25, 0.25),
                # z=0.0 keeps cylinder at table-surface resting height (0.90 m);
                # z=0.30 places it 30 cm above — midair states are valid for
                # ObjectAnywhereEEAnywhere and add diversity to the dataset.
                "z": (0.0, 0.30),
                "roll": (-np.pi, np.pi),
                "pitch": (-np.pi, np.pi),
                "yaw": (-np.pi, np.pi),
            },
            "velocity_range": {},
            "asset_cfgs": {"object": SceneEntityCfg("object")},
            # No offset_asset_cfg: the G1 has no mounting plate, and CylinderCfg
            # has no USD metadata so use_bottom_offset would crash.
        },
    )

    reset_end_effector_pose = EventTerm(
        func=task_mdp.reset_end_effector_round_fixed_asset,
        mode="reset",
        params={
            # fixed_asset_cfg="robot" + fixed_asset_offset=None: the IK target
            # sphere is centred on robot.data.root_pos_w = (0, 0, 0.75).
            "fixed_asset_cfg": SceneEntityCfg("robot"),
            "fixed_asset_offset": None,
            "pose_range_b": {
                # Offsets from root_pos_w = (0, 0, 0.75).
                # x: 0.20–0.70 m forward → world x ∈ [0.20, 0.70] (G1 arm reach).
                "x": (-0.25, -0.05),
                "y": (-0.1, 0),
                # z: 0.10–0.45 m above root → world z ∈ [0.85, 1.20].
                # Lower bound 0.85 = table surface.  Upper bound 1.20 ≈ shoulder height.
                "z": (0.10, 0.25),
                # Orientation samples identical to UR5e — cover downward/forward
                # approach orientations without sampling fully inverted poses.
                "roll": (0.0, 0.0),
                "pitch": (np.pi / 4, 3 * np.pi / 4),
                "yaw": (np.pi / 2, 3 * np.pi / 2),
            },
            # 7 right-arm joints (matches G1_RIGHT_ARM_DIFF_IK in actions.py).
            # body_names="right_hand_palm_link": the Dex3 palm — equivalent to
            # "robotiq_base_link" in the UR5e version and consistent with what
            # the grasp dataset records relative poses against.
            "robot_ik_cfg": SceneEntityCfg(
                "robot",
                joint_names=["right_shoulder.*", "right_elbow.*", "right_wrist.*"],
                body_names="right_hand_palm_link",
            ),
        },
    )


# ---------------------------------------------------------------------------
# ObjectRestingEEGrasped events
# ---------------------------------------------------------------------------

@configclass
class ObjectRestingEEGraspedEventCfg(G1Dex3ResetStatesBaseEventCfg):
    """Cylinder resting flat on the table; arm teleported to a grasp pose from grasps.pt.

    Requires grasps.pt from Phase 2 to exist at:
        <dataset_dir>/Grasps/Cylinder/grasps.pt

    This is the harder and more expensive recording type: the IK solver must
    converge to a specific palm pose relative to the cylinder.  If the cylinder
    is placed near the edge of the arm's workspace, IK may not converge and the
    episode terminates without recording a state.  The Phase 3 verification gate
    monitors the success rate and catches low-convergence issues before training.

    Compared to ObjectRestingEEGraspedEventCfg (UR5e):

    reset_insertive_object_pose_from_reset_states  →  reset_object_pose
        The UR5e version used MultiResetManager to place the peg at a state
        sampled from the already-recorded ObjectAnywhereEEAnywhere dataset.
        That approach produces a peg pose that was already filtered for sim
        stability, and naturally tends to place the peg near the table surface
        because stable states cluster there.
        For the G1, MultiResetManager cannot be used here because it derives the
        dataset path from both insertive and receptive USD paths (compute_pair_dir),
        and there is no receptive object.  A simple uniform reset is used instead:
            x: (-0.15, 0.15)  →  world x ∈ [0.35, 0.65]
            y: (-0.25, 0.25)  →  world y ∈ [-0.25, 0.25]
            z: (0.0, 0.0)     →  world z = 0.90 exactly (cylinder on table surface)
            yaw: (-π, π)      →  any azimuthal orientation (cylinder is symmetric)
        roll/pitch omitted (default 0.0): cylinder stays upright.  The grasp
        dataset was recorded with an upright cylinder, so a tipped cylinder
        would present a grasp frame the IK solver cannot reach.

    reset_end_effector_pose_from_grasp_dataset
        joint_names updated to right arm (same as ObjectAnywhereEEAnywhere).
        body_names: "robotiq_base_link" → "right_hand_palm_link".
        gripper_cfg updated: ["finger_joint",".*right.*",".*left.*"] →
            ["right_hand_.*"] — matches the 7 Dex3 right-hand joints by the
            same pattern used in BaseEventCfg's randomize_gripper_actuator_parameters
            and G1_DEX3_RIGHT_HAND_BINARY in actions.py.
        object_name="Cylinder": new parameter added to events.py.  CylinderCfg
            is a procedural PhysX primitive with no usd_path attribute, so the
            original _compute_grasp_dataset_path() would crash on usd_path access.
            The object_name param bypasses the object_name_from_usd() derivation
            and directly produces the path: <dataset_dir>/Grasps/Cylinder/grasps.pt.
        pose_range_b: identical ±2 cm / ±π/16 jitter as UR5e.  This adds small
            variations around each sampled grasp so the recorded dataset does not
            collapse to a single mode per grasp candidate.
    """

    reset_object_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {
                # Same x/y table-surface range as ObjectAnywhereEEAnywhere.
                "x": (-0.25, -0.05),
                "y": (-0.1, 0),
                # z=0.0: cylinder stays at its init_state z = 0.90 m (table surface).
                # No mid-air placement — the "resting" constraint requires it on the table.
                "z": (0.0, 0.0),
                # Only yaw: cylinder is rotationally symmetric around Z, so any
                # azimuthal angle is valid.  roll/pitch stay 0 so the cylinder
                # remains upright, matching the orientation used in Phase 2 grasp
                # sampling (surface_bias_mode="center" with horizontal normals).
                "yaw": (-np.pi, np.pi),
            },
            "velocity_range": {},
            "asset_cfgs": {"object": SceneEntityCfg("object")},
        },
    )

    reset_end_effector_pose_from_grasp_dataset = EventTerm(
        func=task_mdp.reset_end_effector_from_grasp_dataset,
        mode="reset",
        params={
            "dataset_dir": "./Datasets/OmniReset",
            "fixed_asset_cfg": SceneEntityCfg("object"),
            # object_name bypasses _compute_grasp_dataset_path()'s usd_path access.
            # Path produced: ./Datasets/OmniReset/Grasps/Cylinder/grasps.pt
            "object_name": "Cylinder",
            "robot_ik_cfg": SceneEntityCfg(
                "robot",
                joint_names=["right_shoulder.*", "right_elbow.*", "right_wrist.*"],
                body_names="right_hand_palm_link",
            ),
            # right_hand_.* matches all 7 Dex3 right-hand joints — equivalent to
            # ["finger_joint",".*right.*",".*left.*"] for the Robotiq 2F-85.
            # The gripper_joint_positions recorded in grasps.pt are keyed by the
            # full joint name (e.g. "right_hand_index_0_joint"), so this pattern
            # must match exactly the names present in the robot's joint_names list.
            "gripper_cfg": SceneEntityCfg("robot", joint_names=["right_hand_.*"]),
            # ±2 cm / ±π/16 rad jitter around each sampled grasp pose.
            # Identical to the UR5e ObjectRestingEEGrasped value.
            "pose_range_b": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (-0.02, 0.02),
                "roll": (-np.pi / 16, np.pi / 16),
                "pitch": (-np.pi / 16, np.pi / 16),
                "yaw": (-np.pi / 16, np.pi / 16),
            },
        },
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------

@configclass
class G1Dex3ResetStatesTerminationCfg:
    """Terminations for G1 reset-state recording.

    Compared to ResetStatesTerminationCfg (UR5e):

    time_out / abnormal_robot  (kept unchanged)
        Identical purpose: bound episode length and guard against NaN/Inf states.

    success — object_cfgs
        [insertive_object, receptive_object] → [object].
        Only one rigid object in the scene; removing receptive_object from the
        list also removes one of the three collision analyzers.

    success — ee_body_name
        "robotiq_base_link" → "right_hand_palm_link".
        Used to look up the EE body index for position tracking and to get
        the EE quaternion.  Must match the body used by the IK reset events.

    success — collision_analyzer_cfgs
        Three analyzers (robot×insertive, robot×receptive, insertive×receptive) →
        one analyzer (robot×object).  The receptive analyzers are dropped with
        the receptive object.  min_dist=-0.0005 (0.5 mm allowed penetration) is
        kept from the original robot×insertive analyzer.

    success — max_object_pos_deviation
        MISSING — set per subclass, same as UR5e pattern.

    success — orientation_z_threshold = None
        The UR5e used the implicit default of -0.5, which checks that the
        gripper approach direction (from robot USD metadata) points within 60°
        of straight down in world frame.  The Dex3 uses a SIDE approach
        (gripper_approach_direction = [1, 0, 0] in its metadata), so its
        approach vector is horizontal (world z ≈ 0) for any valid grasp —
        always failing the z < -0.5 test, producing ZERO recorded states.

        None is chosen over a permissive positive threshold because:
        1. "Not too far upward" (e.g. z < 0.5) is vague and uncalibrated for
           the G1 arm geometry; it could silently pass bad poses.
        2. The cylinder stability check (max_object_pos_deviation, velocity
           thresholds) already guarantees the hand is *holding* the cylinder,
           which is the only orientation constraint that matters for a pick task.
        3. If variable orientations (top-down, angled) are added later, the
           correct action is to add a new geometry-aware constraint here, not
           to activate a threshold that was designed for a different robot.

        The parameter is still wired through __init__ and __call__ of
        check_reset_state_success so it is visible in the config rather than
        silently absent from the code path.
    """

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)

    success = DoneTerm(
        func=task_mdp.check_reset_state_success,
        params={
            "object_cfgs": [SceneEntityCfg("object")],
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_body_name": "right_hand_palm_link",
            "collision_analyzer_cfgs": [
                # Robot vs cylinder — same min_dist=-0.0005 as the UR5e robot×peg
                # analyzer: allows up to 0.5 mm contact penetration so that grasps
                # where the fingers are just touching (not deeply interpenetrating)
                # are accepted.
                task_mdp.CollisionAnalyzerCfg(
                    num_points=1024,
                    max_dist=0.5,
                    min_dist=-0.0005,
                    asset_cfg=SceneEntityCfg("robot"),
                    obstacle_cfgs=[SceneEntityCfg("object")],
                ),
            ],
            "max_robot_pos_deviation": 0.1,
            # Set per subclass: np.inf for ObjectAnywhereEEAnywhere (no constraint),
            # 0.01 for ObjectRestingEEGrasped (cylinder must stay within 1 cm).
            "max_object_pos_deviation": MISSING,
            "pos_z_threshold": -0.02,
            "consecutive_stability_steps": 5,
            # None: Dex3 is a side-grasping hand; see class docstring above.
            "orientation_z_threshold": None,
            # position_delta: compare asset positions between consecutive steps instead
            # of checking PhysX velocities.  The G1's fix_root_link=True causes large
            # solver-correction velocities that never settle, preventing the velocity
            # mode from reaching stability.
            "stability_mode": "position_delta",
            "pos_delta_threshold": 0.005,
        },
        time_out=True,
    )


# ---------------------------------------------------------------------------
# Concrete per-variant termination subclasses
# ---------------------------------------------------------------------------

@configclass
class ObjectAnywhereEEAnywhereTerminationCfg(G1Dex3ResetStatesTerminationCfg):
    """Terminations for ObjectAnywhereEEAnywhere recording.

    max_object_pos_deviation = np.inf: the cylinder can be anywhere, so no
    constraint on how far it drifts from its initial teleported position.
    Same as the UR5e ObjectAnywhereEEAnywhereResetStatesCfg.
    """

    def __post_init__(self):
        self.success.params["max_object_pos_deviation"] = np.inf


@configclass
class ObjectRestingEEGraspedTerminationCfg(G1Dex3ResetStatesTerminationCfg):
    """Terminations for ObjectRestingEEGrasped recording.

    max_object_pos_deviation = 0.01: the cylinder must stay within 1 cm of
    where it was placed (table surface).  Any drift beyond this means the hand
    failed to hold it.  Same tight tolerance as the UR5e version.
    """

    def __post_init__(self):
        self.success.params["max_object_pos_deviation"] = 0.01


# ---------------------------------------------------------------------------
# Empty observations / rewards (recording envs need no policy signal)
# ---------------------------------------------------------------------------

@configclass
class ResetStatesObservationsCfg:
    """No policy observations needed for reset-state recording.

    Identical to the UR5e ResetStatesObservationsCfg — the recording pass
    runs a zero-action policy; no obs are consumed by a network.
    """
    pass


@configclass
class ResetStatesRewardsCfg:
    """No rewards needed for reset-state recording.

    Success is determined entirely by check_reset_state_success; there is no
    RL signal to compute.  Identical to the UR5e ResetStatesRewardsCfg.
    """
    pass


# ---------------------------------------------------------------------------
# Top-level environment config (base — events MISSING)
# ---------------------------------------------------------------------------

@configclass
class G1Dex3ResetStatesCfg(ManagerBasedRLEnvCfg):
    """Base config for G1 reset-state recording environments.

    events is MISSING — concrete subclasses (ObjectAnywhereEEAnywhere,
    ObjectRestingEEGrasped) fill it in.  This mirrors the UR5e pattern in
    UR5eRobotiq2f85ResetStatesCfg.

    Compared to UR5eRobotiq2f85ResetStatesCfg:

    scene
        ResetStatesSceneCfg (UR5e) → G1Dex3ResetStatesSceneCfg.

    events
        MISSING in both base configs; set by subclasses.

    terminations
        ResetStatesTerminationCfg (MISSING max_object_pos_deviation) →
        G1Dex3ResetStatesTerminationCfg (same pattern; also MISSING).
        Concrete subclasses override with the per-variant termination cfg
        that fills in the correct value in __post_init__.

    actions
        Ur5eRobotiq2f85RelativeOSCAction →  G1Dex3PickAction.
        The recording pass sends zero actions, but the action space must exist
        for the env to construct and for StableStateRecorder to capture a
        complete scene state (joint_pos_target is part of the state snapshot).

    variants dict → removed
        No object USD variants for the simplified pick task.

    viewer
        Adjusted for G1 geometry: eye=(2.0, 1.0, 1.5) frames both the G1 arm
        (root z=0.75, shoulder z≈1.15) and the table (top z=0.85) in a single
        shot.  Matches the viewer in rl_state_cfg.py.

    episode_length_s = 2.0
        Same as UR5e: short episodes allow the sim to settle and trigger
        check_reset_state_success quickly, maximising recording throughput.

    decimation = 12 / sim.dt = 1/120
        Unchanged — same as RL training env so recorded states are in the same
        dynamical regime.
    """

    scene: G1Dex3ResetStatesSceneCfg = G1Dex3ResetStatesSceneCfg(num_envs=1, env_spacing=1.5)
    events: G1Dex3ResetStatesBaseEventCfg = MISSING
    terminations: G1Dex3ResetStatesTerminationCfg = G1Dex3ResetStatesTerminationCfg()
    observations: ResetStatesObservationsCfg = ResetStatesObservationsCfg()
    actions: G1Dex3PickAction = G1Dex3PickAction()
    rewards: ResetStatesRewardsCfg = ResetStatesRewardsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 1.0, 1.5), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        self.decimation = 12
        self.episode_length_s = 2.0
        self.sim.dt = 1 / 120.0

        # PhysX solver — identical to G1Dex3PickCylinderCfg (rl_state_cfg.py).
        # High iteration counts are necessary for stable finger-cylinder contact
        # during the recording stability window.
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.02
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.0005

        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**31

        self.sim.render.enable_dlssg = True
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dl_denoiser = True


# ---------------------------------------------------------------------------
# Concrete per-variant env configs
# ---------------------------------------------------------------------------

@configclass
class G1Dex3ObjectAnywhereEEAnywhereResetStatesCfg(G1Dex3ResetStatesCfg):
    """Recording env for ObjectAnywhereEEAnywhere.

    Wires in the matching event cfg and termination cfg.
    max_object_pos_deviation is set to np.inf by ObjectAnywhereEEAnywhereTerminationCfg.

    Registered as: OmniReset-G1Dex3-ResetStates-ObjectAnywhereEEAnywhere-v0
    """

    events: ObjectAnywhereEEAnywhereEventCfg = ObjectAnywhereEEAnywhereEventCfg()
    terminations: ObjectAnywhereEEAnywhereTerminationCfg = ObjectAnywhereEEAnywhereTerminationCfg()


@configclass
class G1Dex3ObjectRestingEEGraspedResetStatesCfg(G1Dex3ResetStatesCfg):
    """Recording env for ObjectRestingEEGrasped.

    Requires grasps.pt from Phase 2.
    max_object_pos_deviation is set to 0.01 by ObjectRestingEEGraspedTerminationCfg.

    Registered as: OmniReset-G1Dex3-ResetStates-ObjectRestingEEGrasped-v0
    """

    events: ObjectRestingEEGraspedEventCfg = ObjectRestingEEGraspedEventCfg()
    terminations: ObjectRestingEEGraspedTerminationCfg = ObjectRestingEEGraspedTerminationCfg()
