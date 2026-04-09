# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grasp sampling environment config for the Dex3 right hand + cylinder.

Purpose
-------
Generates grasps.pt: a dataset of stable SIDE-GRASP relative poses between the
Dex3 palm (right_hand_palm_link) and the cylinder.  Phase 3 uses this dataset to
initialise ObjectRestingEEGrasped reset states where the full G1 arm is already
holding the cylinder.

Comparison to ur5e_robotiq_2f85/grasp_sampling_cfg.py
------------------------------------------------------
Robot    : ROBOTIQ_2F85 (standalone 2F85 gripper-only USD, 1 joint) →
           DEX3_RIGHT_STANDALONE_CFG (standalone right-hand-only USD, 7 joints)
Object   : peg.usd (UsdFileCfg, USD mesh)   →
           sim_utils.CylinderCfg (procedural primitive, radius=0.025 m, height=0.10 m)
Body     : "robotiq_base_link"              → "right_hand_palm_link" (Dex3 palm)
Variants : dict of 6 object USD swaps       → removed (cylinder only)
Approach : any (peg is short, upright pegs grasp from side by default)  →
           SIDE approach: gripper_approach_direction=[1,0,0] in metadata.yaml

Standalone USD pre-requisite (Step 2)
--------------------------------------
Run once to create the standalone hand USD from the full G1 asset:

    cd /path/to/unitree_sim_isaaclab
    python3 /path/to/UWLab-g1/scripts/tools/extract_dex3_right_hand_usd.py

This writes:
  $PROJECT_ROOT/assets/robots/dex3-right-hand-usd/dex3_right_hand.usd
  $PROJECT_ROOT/assets/robots/dex3-right-hand-usd/metadata.yaml

The extraction script also writes metadata.yaml with side-grasp parameters.
Verify them with visualize_grasps=True before the full recording run.

Cylinder primitive pre-requisite (Step 4, still pending)
----------------------------------------------------------
Patch _extract_mesh_from_asset in mdp/events.py (shared UWLab-g1 copy) so
grasp_sampling_event can extract trimesh from a UsdGeom.Cylinder prim.
CylinderCfg spawns a UsdGeom.Cylinder, not a UsdGeom.Mesh; the upstream
_find_mesh_in_prim traversal returns None and crashes.
CollisionAnalyzerCfg already handles primitives (check_grasp_success is fine);
only the candidate-generation path in grasp_sampling_event needs fixing.

3-finger geometry note
-----------------------
The Dex3 has an asymmetric opposition: thumb (1) vs index+middle (2 fingers).
Key design choices reflected here:
  - maximum_aperture in metadata.yaml measures thumb-tip to (index+middle midpoint),
    not a single fingertip span.
  - surface_bias_mode="center" (new parameter in grasp_sampling_event): prefers
    mid-height points with horizontal normals instead of the UR5e "top" bias.
    This keeps both index and middle fingers (spread ~15 mm apart along Z) away
    from the cylinder end-caps where one finger would overhang.
  - lateral_sigma=0.0: grasp centre placed at the geometric midpoint of each
    antipodal pair (cylinder centre).  The azimuthal diversity comes from
    orientation_sample_axis=[0,0,1], which rotates the hand around Z.
  - orientation_sample_axis=[0,0,1] in metadata.yaml rotates the hand around the
    cylinder Z axis, giving full 360° azimuthal side-grasp coverage.
"""

from __future__ import annotations

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from uwlab_assets.robots.g1_dex3 import DEX3_RIGHT_STANDALONE_CFG

from .actions import G1_DEX3_RIGHT_HAND_BINARY
from ... import mdp as task_mdp


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OBJECT_SPAWN_HEIGHT = 0.3
"""Cylinder centre height (metres) during grasp sampling.

Chosen to give the floating hand plenty of approach space on all sides.
No table is present — the cylinder floats at this height and the
grasp_sampling_event teleports the hand to each candidate pose.

Compared to the UR5e version (OBJECT_SPAWN_HEIGHT = 0.5), 0.3 m works well
because the standalone hand starts at 0.45 m (init_state.pos z) and the
side-approach standoff directions are all horizontal, so vertical clearance
above the cylinder matters less than for a top-down approach.
"""

# Palm body name confirmed from the G1 Dex3 USD hierarchy:
#   right_wrist_yaw_link → right_hand_palm_joint (FixedJoint) → right_hand_palm_link
# This is the ARTICULATION ROOT of the standalone hand USD and the body whose
# pose is recorded relative to the cylinder by GraspRelativePoseRecorder.
# The UR5e equivalent is "robotiq_base_link".
DEX3_PALM_BODY_NAME = "right_hand_palm_link"


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@configclass
class Dex3GraspSamplingSceneCfg(InteractiveSceneCfg):
    """Scene for Dex3 grasp sampling.

    Compared to GraspSamplingSceneCfg (UR5e):
    - robot : ROBOTIQ_2F85 (standalone gripper-only USD, disable_gravity=True) →
              DEX3_RIGHT_STANDALONE_CFG (standalone hand-only USD, disable_gravity=True)
    - object: peg.usd via UsdFileCfg →
              CylinderCfg primitive (radius=0.025, height=0.10, mass=0.001)
    - Approach direction: both envs produce side grasps on upright cylinders;
              the Robotiq approach is set via its metadata.yaml; the Dex3 approach
              is configured via its own metadata.yaml (gripper_approach_direction=[1,0,0]).
    - No table: neither env needs one; the object floats at OBJECT_SPAWN_HEIGHT.
    """

    robot = DEX3_RIGHT_STANDALONE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CylinderCfg(
            radius=0.025,
            height=0.10,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            # Very light — mirrors peg.usd mass=0.001 in the UR5e version so
            # the grasp-stability velocity threshold is equally reachable.
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, OBJECT_SPAWN_HEIGHT), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@configclass
class Dex3GraspSamplingEventCfg:
    """Events for Dex3 grasp sampling.

    Identical in structure to GraspSamplingEventCfg (UR5e); only the
    body_names argument in grasp_sampling changes (palm body of the Dex3
    instead of robotiq_base_link).

    Execution sequence per episode:
      reset    → reset_object_position  fix cylinder at OBJECT_SPAWN_HEIGHT
      reset    → grasp_sampling         teleport hand to candidate pose
      interval → global_physics_control gravity on/off + force perturbation
    """

    reset_object_position = EventTerm(
        func=task_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (OBJECT_SPAWN_HEIGHT, OBJECT_SPAWN_HEIGHT),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    grasp_sampling = EventTerm(
        func=task_mdp.grasp_sampling_event,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("object"),
            # right_hand_palm_link is the root body of the standalone hand USD and
            # the body whose pose is recorded relative to the cylinder.
            # In the UR5e version this is "robotiq_base_link".
            "gripper_cfg": SceneEntityCfg("robot", body_names=DEX3_PALM_BODY_NAME),
            "num_candidates": 1e6,
            "num_standoff_samples": 32,
            "num_orientations": 16,
            # lateral_sigma controls the HORIZONTAL offset of the grasp centre along
            # the antipodal axis (left/right of the cylinder's vertical axis).
            # 0.0 = always at the geometric midpoint (cylinder centre).  Leave at 0.0
            # for the Dex3; the side-approach metadata already spreads candidates
            # azimuthally around the cylinder via orientation_sample_axis.
            "lateral_sigma": 0.0,
            # "center" bias: prefer mid-height surface points with horizontal normals.
            # Opposite of "top" (UR5e default): keeps both index and middle fingers
            # (spread ~15 mm apart along Z) away from the cylinder end-caps where one
            # finger would overhang.  Horizontal normals select the curved cylinder
            # side — exactly the surface where a side-facing power grasp contacts.
            "surface_bias_mode": "center",
            # Set visualize_grasps=True for the initial sanity-check run to
            # confirm the candidate poses look geometrically correct (side approach,
            # palm at cylinder mid-height, fingers wrapping around) before committing
            # a full 500-grasp recording session.
            "visualize_grasps": False,
            "visualization_scale": 0.01,
        },
    )

    global_physics_control_event = EventTerm(
        func=task_mdp.global_physics_control_event,
        mode="interval",
        interval_range_s=(0.1, 0.1),
        params={
            # Gravity is disabled during hand repositioning (reset), then
            # re-enabled after 1 s so the grasp is tested under real gravity.
            "gravity_on_interval": (1.0, np.inf),
            # Apply a small random force/torque to the cylinder from t=1 to t=2
            # to shake out marginal grasps that only hold under zero external load.
            "force_torque_on_interval": (1.0, 2.0),
            "force_torque_asset_cfgs": [SceneEntityCfg("object")],
            "force_torque_magnitude": 0.01,
        },
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------

@configclass
class Dex3GraspSamplingTerminationCfg:
    """Terminations for Dex3 grasp sampling.

    Compared to GraspSamplingTerminationCfg (UR5e):
    - Logic is identical; thresholds scale automatically with OBJECT_SPAWN_HEIGHT.
    - CollisionAnalyzerCfg works with the primitive cylinder unchanged:
      RigidObjectHasher explicitly handles UsdGeom.Cylinder prims (line 62 of
      rigid_object_hasher.py) and prim_to_warp_mesh falls back to
      create_primitive_mesh for non-Mesh prim types.
    """

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)

    success = DoneTerm(
        func=task_mdp.check_grasp_success,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "gripper_cfg": SceneEntityCfg("robot"),
            "collision_analyzer_cfg": task_mdp.CollisionAnalyzerCfg(
                num_points=1024,
                max_dist=0.5,
                min_dist=-0.0005,
                asset_cfg=SceneEntityCfg("robot"),
                obstacle_cfgs=[SceneEntityCfg("object")],
            ),
            # Grasp is invalid if the cylinder drifts more than half its spawn
            # height from where it started, or if it falls below that height.
            "max_pos_deviation": OBJECT_SPAWN_HEIGHT / 2,
            "pos_z_threshold": OBJECT_SPAWN_HEIGHT / 2,
        },
        time_out=True,
    )


# ---------------------------------------------------------------------------
# Observations / Rewards (empty — same as UR5e version)
# ---------------------------------------------------------------------------

@configclass
class Dex3GraspSamplingObservationsCfg:
    """No policy observations needed; grasp sampling is event-driven."""

    pass


@configclass
class Dex3GraspSamplingRewardsCfg:
    """No reward shaping needed; success is binary via the termination."""

    pass


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@configclass
class Dex3BinaryGripperAction:
    """Grasp sampling action: binary open/close for the 7 right-hand joints.

    Compared to Robotiq2f85BinaryGripperAction (single finger_joint),
    the Dex3 specifies targets for all 7 right-hand joints.
    Reuses G1_DEX3_RIGHT_HAND_BINARY from actions.py directly.

    Note: if the standalone hand USD uses different joint names from the
    right_hand_* pattern expected by G1_DEX3_RIGHT_HAND_BINARY, update
    actions.py before running record_grasps.py (Step 5).
    """

    gripper = G1_DEX3_RIGHT_HAND_BINARY


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@configclass
class Dex3GraspSamplingCfg(ManagerBasedRLEnvCfg):
    """Grasp sampling environment for the Dex3 right hand + cylinder.

    Registered as OmniReset-Dex3-GraspSampling-Cylinder-v0.
    (Registration is added to config/g1_dex3/__init__.py in Step 6.)

    Compared to Robotiq2f85GraspSamplingCfg:
    - scene.robot  : DEX3_RIGHT_STANDALONE_CFG (hand-only) vs ROBOTIQ_2F85
    - scene.object : CylinderCfg primitive vs peg.usd
    - actions      : Dex3BinaryGripperAction (7 joints) vs 1 joint
    - variants     : removed (no object swaps needed for pick task)
    - viewer.eye   : lowered to (1.0, 0.0, 0.55) to frame hand at ~0.45 m
                     and cylinder at 0.30 m in the same shot
    """

    scene: Dex3GraspSamplingSceneCfg = Dex3GraspSamplingSceneCfg(num_envs=1, env_spacing=1.5)
    events: Dex3GraspSamplingEventCfg = Dex3GraspSamplingEventCfg()
    terminations: Dex3GraspSamplingTerminationCfg = Dex3GraspSamplingTerminationCfg()
    observations: Dex3GraspSamplingObservationsCfg = Dex3GraspSamplingObservationsCfg()
    actions: Dex3BinaryGripperAction = Dex3BinaryGripperAction()
    rewards: Dex3GraspSamplingRewardsCfg = Dex3GraspSamplingRewardsCfg()
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.0, 0.0, 0.55), origin_type="world", env_index=0, asset_name="robot"
    )

    def __post_init__(self):
        self.decimation = 12
        self.episode_length_s = 4.0
        self.sim.dt = 1 / 120.0

        # PhysX solver — identical to the UR5e grasp sampling config; high
        # iteration counts stabilise finger-cylinder contact during the
        # perturbation phase.
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
