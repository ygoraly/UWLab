# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL state environment config for G1-29DOF + Dex3 pick-cylinder task."""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
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
class G1Dex3SceneCfg(InteractiveSceneCfg):
    """Scene for G1 fixed-base pick-cylinder.

    Compared to the UR5e RlStateSceneCfg:
    - robot: IMPLICIT_UR5E_ROBOTIQ_2F85 → G1_DEX3_FIXED_BASE_CFG
    - insertive_object (peg USD) → object (primitive CylinderCfg, no external asset)
    - receptive_object (peg hole USD) → removed (no assembly target needed)
    - ur5_metal_support → removed (UR5e-specific mounting plate)
    - table: PAT Vention USD → simple kinematic CuboidCfg
    """
    robot = G1_DEX3_FIXED_BASE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.90), rot=(1.0, 0.0, 0.0, 0.0)),
    )

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

    # ------------------------------------------------------------------
    # Environment dressing
    # ------------------------------------------------------------------
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
# Observations
# ---------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy (actor) network."""

        prev_actions = ObsTerm(func=task_mdp.last_action)

        joint_pos = ObsTerm(func=task_mdp.joint_pos)

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "axis_angle",
            },
        )

        object_pose_in_ee_frame = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
                "rotation_repr": "axis_angle",
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for the critic (value) network.

        Includes all policy observations plus privileged terms (velocities,
        material properties, mass, joint dynamics). These are available in
        simulation but not on a real robot, hence the critic-only placement.
        """

        prev_actions = ObsTerm(func=task_mdp.last_action)

        joint_pos = ObsTerm(func=task_mdp.joint_pos)

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "axis_angle",
            },
        )

        object_pose_in_ee_frame = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
                "rotation_repr": "axis_angle",
            },
        )

        # Privileged observations (sim-only)
        time_left = ObsTerm(func=task_mdp.time_left)

        joint_vel = ObsTerm(func=task_mdp.joint_vel)

        end_effector_vel_lin_ang_b = ObsTerm(
            func=task_mdp.asset_link_velocity_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        robot_material_properties = ObsTerm(
            func=task_mdp.get_material_properties,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        object_material_properties = ObsTerm(
            func=task_mdp.get_material_properties,
            params={"asset_cfg": SceneEntityCfg("object")},
        )

        table_material_properties = ObsTerm(
            func=task_mdp.get_material_properties,
            params={"asset_cfg": SceneEntityCfg("table")},
        )

        robot_mass = ObsTerm(
            func=task_mdp.get_mass,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        object_mass = ObsTerm(
            func=task_mdp.get_mass,
            params={"asset_cfg": SceneEntityCfg("object")},
        )

        table_mass = ObsTerm(
            func=task_mdp.get_mass,
            params={"asset_cfg": SceneEntityCfg("table")},
        )

        robot_joint_friction = ObsTerm(
            func=task_mdp.get_joint_friction,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        robot_joint_armature = ObsTerm(
            func=task_mdp.get_joint_armature,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        robot_joint_stiffness = ObsTerm(
            func=task_mdp.get_joint_stiffness,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        robot_joint_damping = ObsTerm(
            func=task_mdp.get_joint_damping,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg:
    """Reward specifications for G1 pick-cylinder."""

    # -- Regularisation (unchanged from UR5e) --------------------------------
    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-4)

    action_rate = RewTerm(func=task_mdp.action_rate_l2_clamped, weight=-1e-3)

    joint_vel = RewTerm(
        func=task_mdp.joint_vel_l2_clamped,
        weight=-1e-2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["right_shoulder.*", "right_elbow.*", "right_wrist.*"],
            )
        },
    )

    abnormal_robot = RewTerm(func=task_mdp.abnormal_robot_state, weight=-100.0)

    # -- Task rewards ---------------------------------------------------------
    ee_object_distance = RewTerm(
        func=task_mdp.ee_object_distance,
        weight=1.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
            "object_cfg": SceneEntityCfg("object"),
            "std": 0.2,
        },
    )

    object_lift = RewTerm(
        func=task_mdp.object_lift,
        weight=2.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "table_height": 0.85,
            "min_lift": 0.05,
        },
    )

    # -- Success tracker (zero reward, exposes .success for MultiResetManager) -
    pick_success_context = RewTerm(
        func=task_mdp.PickSuccessContext,
        weight=0.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "table_height": 0.85,
            "lift_threshold": 0.05,
        },
    )


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
@configclass
class BaseEventCfg:
    """Shared startup + per-episode events for G1 pick-cylinder."""

    # -- Startup: friction randomisation (once at env creation) ---------------
    robot_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("robot"),
            "make_consistent": True,
        },
    )

    object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.8, 2.0),
            "dynamic_friction_range": (0.6, 1.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "asset_cfg": SceneEntityCfg("object"),
            # False: primitive shapes handle material buckets differently from
            # multi-mesh USD assets; using True can hit PhysX assertions.
            "make_consistent": False,
        },
    )

    table_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.6),
            "dynamic_friction_range": (0.2, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "asset_cfg": SceneEntityCfg("table"),
            "make_consistent": False,
        },
    )

    # -- Startup: mass randomisation ------------------------------------------
    randomize_robot_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_object_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.03, 0.15),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    # -- Per-episode: actuator gain randomisation ------------------------------
    randomize_gripper_actuator_parameters = EventTerm(
        func=task_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["right_hand_.*_joint"]),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # -- Per-episode: full scene reset (must be last in BaseEventCfg so that
    #    subclass reset events run AFTER the scene is back at defaults) --------
    reset_everything = EventTerm(
        func=task_mdp.reset_scene_to_default,
        mode="reset",
        params={},
    )


@configclass
class TrainEventCfgSimple(BaseEventCfg):
    """Phase 1 training events: uniform cylinder pose reset (no dataset).

    Kept as a fallback for debugging without Phase 2/3 datasets.
    Registered as OmniReset-G1Dex3-Pick-Cylinder-Train-Simple-v0.

    Execution order of mode="reset" events (field declaration order):
        1. randomize_gripper_actuator_parameters  (from BaseEventCfg)
        2. reset_everything                       (from BaseEventCfg)
        3. reset_object_pose                      (runs after base)
    """

    reset_object_pose = EventTerm(
        func=task_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.15, 0.15), "y": (-0.20, 0.20)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class TrainEventCfg(BaseEventCfg):
    """Phase 4 training events: MultiResetManager with 2 reset-state datasets.

    Replaces the Phase 1 uniform reset with MultiResetManager, which samples
    pre-recorded scene snapshots from Phase 3 datasets.  This is the G1
    equivalent of the UR5e TrainEventCfg, with two differences:

    1. Two reset types instead of four — ObjectAnywhereEEGrasped and
       ObjectPartiallyAssembledEEGrasped are assembly-specific and don't
       apply to the simplified pick task.
    2. pair_dir="Cylinder" replaces the insertive/receptive USD path
       derivation (the G1 scene has no insertive_object or receptive_object).
    3. success reads PickSuccessContext.success (object lifted above table)
       instead of the UR5e ProgressContext.success (assembly alignment).
       This gives per-reset-type pick success rates in TensorBoard under
       Metrics/task_0_success_rate and Metrics/task_1_success_rate.

    Dataset files consumed (produced by record_reset_states_g1.py):
        ./Datasets/OmniReset/Resets/Cylinder/resets_ObjectAnywhereEEAnywhere.pt
        ./Datasets/OmniReset/Resets/Cylinder/resets_ObjectRestingEEGrasped.pt

    Execution order of mode="reset" events (field declaration order):
        1. randomize_gripper_actuator_parameters  (from BaseEventCfg)
        2. reset_everything                       (from BaseEventCfg)
        3. reset_from_reset_states                (overwrites with dataset state)
    """

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": "./Datasets/OmniReset",
            "pair_dir": "Cylinder",
            "reset_types": [
                "ObjectAnywhereEEAnywhere",
                "ObjectRestingEEGrasped",
            ],
            "probs": [0.5, 0.5],
            "success": "env.reward_manager.get_term_cfg('pick_success_context').func.success",
        },
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    pass


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    """Termination terms for the MDP.

    Identical to the UR5e version:
        time_out        — episode ends after episode_length_s seconds.
        abnormal_robot  — episode ends immediately if any joint/velocity
                          values become non-finite (NaN/Inf), protecting
                          the replay buffer from corrupt rollouts.
    """

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------
@configclass
class NoCurriculumsCfg:
    """No curriculum (Phase 1: fixed randomisation ranges, no ADR).

    The UR5e FinetuneCurriculumsCfg ramps sysid DR and action scales once
    the policy converges.  Those features are deferred to Phase 3 for the G1.
    """

    pass


# ---------------------------------------------------------------------------
# Top-level environment configuration
# ---------------------------------------------------------------------------
@configclass
class G1Dex3PickCylinderCfg(ManagerBasedRLEnvCfg):
    """Base environment config for the G1+Dex3 pick-cylinder task.

    This is a *base* config with ``events: MISSING``.  Concrete training /
    evaluation configs inherit from this and fill in the event schedule.
    This mirrors the pattern in Ur5eRobotiq2f85RlStateCfg, where events is
    also MISSING and the concrete TrainCfg / EvalCfg subclasses assign it.

    Compared to Ur5eRobotiq2f85RlStateCfg:
        scene   : RlStateSceneCfg        → G1Dex3SceneCfg
        actions : Ur5eRobotiq2f85...OSC  → G1Dex3PickAction
        commands: TaskCommandCfg present → CommandsCfg (empty)
        viewer  : eye adjusted for the G1 at (0,0,0.75) facing a table at
                  (0.5, 0, 0.85); a 2 m setback at 1.5 m height gives a
                  clear view of both the arm and the cylinder on the table.
    """

    scene: G1Dex3SceneCfg = G1Dex3SceneCfg(num_envs=32, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: G1Dex3PickAction = G1Dex3PickAction()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()
    events: BaseEventCfg = MISSING
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 1.2, 1.5), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        # Control rate: 120 Hz physics / 12 = 10 Hz policy.
        # Matches the UR5e setting so hyper-parameters transfer.
        self.decimation = 12
        self.episode_length_s = 16.0

        # Simulation timestep
        self.sim.dt = 1 / 120.0

        # PhysX solver — identical to UR5e; high iteration counts are needed
        # for stable contact between the Dex3 fingers and the cylinder.
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.02
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.0005

        # GPU buffer sizes — copied from UR5e; safe upper bounds for 32 envs.
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**31

        # Render quality
        self.sim.render.enable_dlssg = True
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dl_denoiser = True


# ---------------------------------------------------------------------------
# Concrete training configurations
# ---------------------------------------------------------------------------
@configclass
class G1Dex3PickTrainCfg(G1Dex3PickCylinderCfg):
    """Training config: MultiResetManager with Phase 3 datasets.

    Requires ./Datasets/OmniReset/Resets/Cylinder/resets_*.pt from Phase 3.
    """

    events: TrainEventCfg = TrainEventCfg()


@configclass
class G1Dex3PickTrainSimpleCfg(G1Dex3PickCylinderCfg):
    """Fallback training config: uniform cylinder reset, no datasets needed.

    Useful for debugging without Phase 2/3 datasets.
    """

    events: TrainEventCfgSimple = TrainEventCfgSimple()
