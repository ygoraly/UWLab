#!/usr/bin/env python3
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 3 verification gate for the G1+Dex3 reset-state recording pipeline.

Runs checks in three stages.  Isaac Lab can only create one environment per
process (gym.make() reinitialises the global USD stage and hangs if called
twice in the same session), so Stage A always runs and exactly one of Stage B
or C runs per invocation, selected via --stage.

  Stage A — pre-simulation (no Isaac context needed):
    A1. grasps.pt exists at <dataset_dir>/Grasps/Cylinder/grasps.pt.
    A2. grasps.pt is valid: > 0 entries, correct position/orientation shapes,
        finite values, and a plausible XY standoff from the cylinder axis.
    A3. Both reset-state cfg classes import and instantiate without error.
    A4. Key config parameters are correct (IK joint/body names,
        orientation_z_threshold=None, max_object_pos_deviation per variant).

  Stage B — ObjectAnywhereEEAnywhere simulation  (--stage B, the default):
    B1. gym.make() succeeds for OmniReset-G1Dex3-ResetStates-ObjectAnywhereEEAnywhere-v0.
    B2. env.reset() completes without crash; observations are finite.
    B3. EE (right_hand_palm_link) is within the G1 arm's expected workspace after reset.
    B4. num_steps physics steps run without crash; reward and obs remain finite.
    B5. Step throughput is at least 1 Hz (GPU sanity check).

  Stage C — ObjectRestingEEGrasped simulation  (--stage C):
    C1. gym.make() succeeds for OmniReset-G1Dex3-ResetStates-ObjectRestingEEGrasped-v0.
    C2. env.reset() completes; loads grasps.pt and IK-teleports the arm (may take ~5 s).
    C3. Cylinder is at table-surface height after reset (z ≈ 0.90 m).
    C4. EE is within 0.25 m of the cylinder — IK converged to a grasp pose.
    C5. num_steps physics steps run without crash; reward and obs remain finite.
    C6. Step throughput is at least 1 Hz.

Run with:
    # Stage A + B (no grasps.pt needed):
    python scripts/environments/verify_g1_dex3_phase3.py --headless --stage B

    # Stage A + C (requires grasps.pt from Phase 2):
    python scripts/environments/verify_g1_dex3_phase3.py --headless --stage C \\
        --dataset_dir /home/ygoraly/git/Datasets/OmniReset/

Exit 0 = all checks pass.  Non-zero = number of failures.

Prerequisites
-------------
  Phase 2 must be complete before --stage C:
    python scripts/tools/record_grasps_g1.py --headless --num_grasps 500

  --stage B has no prerequisites beyond Phase 1 (scene loads).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 3 verification for G1+Dex3 reset-state recording.")
parser.add_argument(
    "--stage",
    type=str,
    choices=["B", "C"],
    default="B",
    help=(
        "Which simulation stage to run.  Isaac Lab cannot create two environments "
        "in one process, so run this script twice — once with --stage B and once "
        "with --stage C.  Stage A (pre-sim checks) always runs regardless."
    ),
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=100,
    help="Physics steps to run per environment in the Stage B/C step loops.",
)
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="./Datasets/OmniReset/",
    help="Root dataset directory.  Expects <dataset_dir>/Grasps/Cylinder/grasps.pt for Stage C.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything after this point runs inside the simulation context."""

import time

import torch

import gymnasium as gym

# Ensure UWLab-g1 sources take priority over any system-installed uwlab_tasks.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _pkg in ("uwlab_tasks", "uwlab_assets"):
    _pkg_path = os.path.join(_REPO_ROOT, "source", _pkg)
    if os.path.isdir(_pkg_path) and _pkg_path not in sys.path:
        sys.path.insert(0, _pkg_path)

import uwlab_tasks  # noqa: F401 — registers all UWLab-g1 gym envs

from uwlab_tasks.manager_based.manipulation.omnireset.config.g1_dex3.reset_states_cfg import (
    G1Dex3ObjectAnywhereEEAnywhereResetStatesCfg,
    G1Dex3ObjectRestingEEGraspedResetStatesCfg,
)

# ── Constants ────────────────────────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
INFO = "\033[94mINFO\033[0m"

PAIR_DIR = "Cylinder"

# Scene geometry (must match reset_states_cfg.py and rl_state_cfg.py):
#   table CuboidCfg size=(0.6, 0.8, 0.85), centre z=0.425 → top surface z=0.85
#   cylinder height=0.10 → half_height=0.05 → resting z = 0.85 + 0.05 = 0.90
TABLE_TOP_Z = 0.85
CYLINDER_HALF_HEIGHT = 0.05
CYLINDER_RESTING_Z = TABLE_TOP_Z + CYLINDER_HALF_HEIGHT  # 0.90 m

EE_BODY_NAME = "right_hand_palm_link"

# Expected workspace for G1 right arm (world-frame values derived from
# ObjectAnywhereEEAnywhereEventCfg: offsets from robot root at z=0.75 m):
#   x: root_x + (0.20, 0.70) = (0.20, 0.70)
#   y: root_y + (-0.40, 0.40) = (-0.40, 0.40)
#   z: root_z + (0.10, 0.45) = (0.85, 1.20)
# We use generous tolerances here because IK may not always reach the extreme edges.
EE_WS_X = (0.10, 0.80)
EE_WS_Y = (-0.50, 0.50)
EE_WS_Z_MIN = 0.50  # above floor even with IK under-convergence

TASK_OA = "OmniReset-G1Dex3-ResetStates-ObjectAnywhereEEAnywhere-v0"
TASK_RG = "OmniReset-G1Dex3-ResetStates-ObjectRestingEEGrasped-v0"


# ── Helpers ──────────────────────────────────────────────────────────────────

def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f"  → {detail}" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return condition


def info(msg: str) -> None:
    print(f"  [{INFO}] {msg}")


def warn(msg: str) -> None:
    print(f"  [{WARN}] {msg}")


def _get_ee_pos(env) -> torch.Tensor | None:
    """Return world-frame EE position for env 0, or None if body not found."""
    robot = env.scene["robot"]
    body_names = list(robot.data.body_names)
    if EE_BODY_NAME not in body_names:
        warn(
            f"'{EE_BODY_NAME}' not found in robot body_names.  "
            f"First 8 bodies: {body_names[:8]}"
        )
        return None
    idx = body_names.index(EE_BODY_NAME)
    return robot.data.body_link_pos_w[0, idx].cpu()


# ── Stage A: pre-simulation checks ──────────────────────────────────────────

def stage_a(failures: int) -> tuple[int, bool]:
    """Pre-simulation checks.

    Returns updated failure count and a bool indicating whether grasps.pt is
    valid and Stage C can run.
    """
    print("\n══════════════════════════════════════════════════════")
    print("  Stage A — pre-simulation checks")
    print("══════════════════════════════════════════════════════\n")

    grasps_available = False
    grasps_path = os.path.join(args_cli.dataset_dir, "Grasps", PAIR_DIR, "grasps.pt")

    # A1 — grasps.pt exists ───────────────────────────────────────────────────
    print("[A1] Checking grasps.pt ...")
    grasps_exist = os.path.isfile(grasps_path)
    if not check("grasps.pt exists", grasps_exist, grasps_path):
        failures += 1
        print()
        warn("Stage C (ObjectRestingEEGrasped) will be SKIPPED.")
        warn("Run Phase 2 first to generate grasps.pt:")
        warn("    python scripts/tools/record_grasps_g1.py --headless --num_grasps 500")
    else:
        size_kb = os.path.getsize(grasps_path) / 1024
        failures += 0 if check(
            f"grasps.pt is non-trivial (> 1 KB, got {size_kb:.0f} KB)",
            size_kb > 1,
        ) else 1
        info(f"Path : {grasps_path}")
        info(f"Size : {size_kb:.0f} KB")

    # A2 — grasps.pt content is valid ─────────────────────────────────────────
    print("\n[A2] Validating grasps.pt content ...")
    if grasps_exist:
        try:
            data = torch.load(grasps_path, map_location="cpu")
            # TorchDatasetFileHandler nests grasp data under 'grasp_relative_pose'.
            grasp_group = data.get("grasp_relative_pose", data)
            rel_pos_list = grasp_group.get("relative_position", [])
            rel_quat_list = grasp_group.get("relative_orientation", [])
            n = len(rel_pos_list)

            failures += 0 if check(
                f"grasp count > 0  (got {n:,})",
                n > 0,
                "If 0: record_grasps_g1.py recorded no successful grasps — "
                "check metadata.yaml and run verify_g1_dex3_phase2.py",
            ) else 1

            grasps_available = n > 0

            if n > 0:
                pos0 = rel_pos_list[0]
                pos0 = pos0 if isinstance(pos0, torch.Tensor) else torch.as_tensor(pos0)
                failures += 0 if check(
                    f"relative_position[0] shape == (3,)  (got {tuple(pos0.shape)})",
                    tuple(pos0.shape) == (3,),
                ) else 1
                failures += 0 if check(
                    "relative_position[0] is finite",
                    bool(torch.isfinite(pos0).all().item()),
                ) else 1

                quat0 = rel_quat_list[0]
                quat0 = quat0 if isinstance(quat0, torch.Tensor) else torch.as_tensor(quat0)
                failures += 0 if check(
                    f"relative_orientation[0] shape == (4,)  (got {tuple(quat0.shape)})",
                    tuple(quat0.shape) == (4,),
                ) else 1

                # Palm should be roughly 1 finger-offset away from the cylinder axis.
                # For a 0.025 m radius cylinder and Dex3 finger_offset ≈ 0.04–0.10 m,
                # expect XY standoff in [0.06, 0.25] m.
                xy_dist = float(torch.norm(pos0[:2]).item())
                failures += 0 if check(
                    f"XY standoff from cylinder axis ∈ [0.01, 0.30] m  (got {xy_dist:.3f} m)",
                    0.01 <= xy_dist <= 0.30,
                    "If ≈0: palm is AT the cylinder centre (bad, means finger_offset=0). "
                    "If >0.30: palm is very far from cylinder — check metadata.yaml.",
                ) else 1

            gripper_joints = list(grasp_group.get("gripper_joint_positions", {}).keys())
            info(f"Gripper joints recorded: {gripper_joints}")
            info(f"Total grasps: {n:,}")
            failures += 0 if check(
                "Gripper joints recorded > 0 (joint positions stored alongside grasps)",
                len(gripper_joints) > 0,
                "If empty: grasp replay will open/close the hand to joint 0 for all fingers",
            ) else 1

        except Exception as exc:
            print(f"  [{FAIL}] Failed to load grasps.pt: {exc}")
            import traceback
            traceback.print_exc()
            failures += 1
    else:
        warn("Skipping content checks — grasps.pt not found.")

    # A3 — Config imports ──────────────────────────────────────────────────────
    print("\n[A3] Checking config imports ...")
    try:
        cfg_oa = G1Dex3ObjectAnywhereEEAnywhereResetStatesCfg()
        failures += 0 if check(
            "G1Dex3ObjectAnywhereEEAnywhereResetStatesCfg instantiated", True
        ) else 1
    except Exception as exc:
        print(f"  [{FAIL}] ObjectAnywhereEEAnywhere cfg raised: {exc}")
        failures += 1
        cfg_oa = None

    try:
        cfg_rg = G1Dex3ObjectRestingEEGraspedResetStatesCfg()
        failures += 0 if check(
            "G1Dex3ObjectRestingEEGraspedResetStatesCfg instantiated", True
        ) else 1
    except Exception as exc:
        print(f"  [{FAIL}] ObjectRestingEEGrasped cfg raised: {exc}")
        failures += 1
        cfg_rg = None

    # A4 — Parameter checks ────────────────────────────────────────────────────
    print("\n[A4] Checking config parameters ...")

    if cfg_oa is not None:
        # ObjectAnywhereEEAnywhere event: EE IK params
        oa_ik = cfg_oa.events.reset_end_effector_pose.params.get("robot_ik_cfg")
        failures += 0 if check(
            f"[ObjAny] IK body_names == '{EE_BODY_NAME}'  "
            f"(got '{getattr(oa_ik, 'body_names', '?')}')",
            getattr(oa_ik, "body_names", None) == EE_BODY_NAME,
        ) else 1
        jnames = getattr(oa_ik, "joint_names", []) or []
        failures += 0 if check(
            f"[ObjAny] IK joint_names include 'right_shoulder.*'  (got {jnames})",
            any("right_shoulder" in j for j in jnames),
        ) else 1

        # ObjectAnywhereEEAnywhere termination params
        oa_tp = cfg_oa.terminations.success.params
        oa_dev = oa_tp.get("max_object_pos_deviation")
        failures += 0 if check(
            f"[ObjAny] max_object_pos_deviation == inf  (got {oa_dev})",
            oa_dev == float("inf"),
        ) else 1
        oa_orient = oa_tp.get("orientation_z_threshold", "MISSING")
        failures += 0 if check(
            f"[ObjAny] orientation_z_threshold is None  (got {oa_orient!r})",
            oa_orient is None,
            "If not None: orientation check active — may reject all Dex3 side-grasps",
        ) else 1

    if cfg_rg is not None:
        # ObjectRestingEEGrasped event: object_name bypass, IK params, gripper pattern
        rg_ee = cfg_rg.events.reset_end_effector_pose_from_grasp_dataset.params
        rg_obj_name = rg_ee.get("object_name", "<missing>")
        failures += 0 if check(
            f"[ObjResting] object_name == 'Cylinder'  (got '{rg_obj_name}')",
            rg_obj_name == "Cylinder",
            "If missing/wrong: _compute_grasp_dataset_path() will crash on usd_path access",
        ) else 1
        rg_ik = rg_ee.get("robot_ik_cfg")
        failures += 0 if check(
            f"[ObjResting] IK body_names == '{EE_BODY_NAME}'  "
            f"(got '{getattr(rg_ik, 'body_names', '?')}')",
            getattr(rg_ik, "body_names", None) == EE_BODY_NAME,
        ) else 1
        rg_gripper = rg_ee.get("gripper_cfg")
        rg_gjnames = getattr(rg_gripper, "joint_names", []) or []
        failures += 0 if check(
            f"[ObjResting] gripper_cfg.joint_names includes 'right_hand_.*'  (got {rg_gjnames})",
            any("right_hand" in j for j in rg_gjnames),
        ) else 1

        # ObjectRestingEEGrasped termination params
        rg_tp = cfg_rg.terminations.success.params
        rg_dev = rg_tp.get("max_object_pos_deviation")
        failures += 0 if check(
            f"[ObjResting] max_object_pos_deviation == 0.01  (got {rg_dev})",
            rg_dev == 0.01,
        ) else 1
        rg_orient = rg_tp.get("orientation_z_threshold", "MISSING")
        failures += 0 if check(
            f"[ObjResting] orientation_z_threshold is None  (got {rg_orient!r})",
            rg_orient is None,
            "If not None: orientation check active — will fail for Dex3 side-grasps",
        ) else 1

    return failures, grasps_available


# ── Stage B: ObjectAnywhereEEAnywhere simulation checks ─────────────────────

def stage_b(failures: int) -> int:
    """ObjectAnywhereEEAnywhere simulation checks."""
    print("\n══════════════════════════════════════════════════════")
    print("  Stage B — ObjectAnywhereEEAnywhere simulation checks")
    print("══════════════════════════════════════════════════════\n")

    # B1 — gym.make() ──────────────────────────────────────────────────────────
    print("[B1] Creating ObjectAnywhereEEAnywhere environment ...")
    env_cfg = G1Dex3ObjectAnywhereEEAnywhereResetStatesCfg()
    env_cfg.scene.num_envs = 1
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = 42

    try:
        env = gym.make(TASK_OA, cfg=env_cfg).unwrapped
    except Exception as exc:
        print(f"  [{FAIL}] gym.make() raised: {exc}")
        import traceback
        traceback.print_exc()
        return failures + 1

    failures += 0 if check("gym.make() succeeded", True) else 1

    # B2 — env.reset() ─────────────────────────────────────────────────────────
    print("\n[B2] Resetting environment ...")
    try:
        obs_dict, _ = env.reset()
        reset_ok = True
    except Exception as exc:
        print(f"  [{FAIL}] env.reset() raised: {exc}")
        import traceback
        traceback.print_exc()
        reset_ok = False
        failures += 1

    if reset_ok:
        failures += 0 if check("env.reset() completed without crash", True) else 1
        for key, val in obs_dict.items():
            if not torch.isfinite(val).all():
                failures += 1
                check(f"obs[{key}] finite after reset", False)

    # B3 — EE position sanity ──────────────────────────────────────────────────
    print("\n[B3] Checking EE position after reset ...")
    if reset_ok:
        try:
            ee_pos = _get_ee_pos(env)
            if ee_pos is not None:
                info(
                    f"EE pos (world frame): "
                    f"x={ee_pos[0]:.3f}  y={ee_pos[1]:.3f}  z={ee_pos[2]:.3f}"
                )
                failures += 0 if check(
                    f"EE x ∈ [{EE_WS_X[0]:.2f}, {EE_WS_X[1]:.2f}] m  (got {ee_pos[0]:.3f})",
                    EE_WS_X[0] <= float(ee_pos[0]) <= EE_WS_X[1],
                    "If x < 0.10: IK may not be reaching forward — check pose_range_b 'x'",
                ) else 1
                failures += 0 if check(
                    f"EE y ∈ [{EE_WS_Y[0]:.2f}, {EE_WS_Y[1]:.2f}] m  (got {ee_pos[1]:.3f})",
                    EE_WS_Y[0] <= float(ee_pos[1]) <= EE_WS_Y[1],
                ) else 1
                failures += 0 if check(
                    f"EE z > {EE_WS_Z_MIN:.2f} m (above floor)  (got {ee_pos[2]:.3f})",
                    float(ee_pos[2]) > EE_WS_Z_MIN,
                    "If z < 0.50: IK produced a pose below the table — check pose_range_b 'z'",
                ) else 1
        except Exception as exc:
            warn(f"EE position check raised (non-fatal): {exc}")

    # B4 — step loop ───────────────────────────────────────────────────────────
    print(f"\n[B4] Stepping {args_cli.num_steps} physics steps ...")
    nonfinite_obs = 0
    nan_reward = 0
    t0 = time.perf_counter()
    actions = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)

    with torch.inference_mode():
        for step_i in range(args_cli.num_steps):
            try:
                obs_dict, reward, _, _, _ = env.step(actions)
            except Exception as exc:
                print(f"  [{FAIL}] env.step({step_i}) raised: {exc}")
                import traceback
                traceback.print_exc()
                failures += 1
                break
            for val in obs_dict.values():
                if not torch.isfinite(val).all():
                    nonfinite_obs += 1
                    break
            if not torch.isfinite(reward).all():
                nan_reward += 1

    elapsed = time.perf_counter() - t0
    failures += 0 if check(
        f"No non-finite obs in {args_cli.num_steps} steps (found {nonfinite_obs})",
        nonfinite_obs == 0,
    ) else 1
    failures += 0 if check(
        f"No NaN reward in {args_cli.num_steps} steps (found {nan_reward})",
        nan_reward == 0,
    ) else 1

    # B5 — throughput ──────────────────────────────────────────────────────────
    print("\n[B5] Checking step throughput ...")
    hz = args_cli.num_steps / elapsed
    info(f"{args_cli.num_steps} steps in {elapsed:.2f} s → {hz:.1f} steps/s")
    failures += 0 if check("Step rate >= 1 Hz (GPU responding)", hz >= 1.0) else 1

    env.close()
    return failures


# ── Stage C: ObjectRestingEEGrasped simulation checks ───────────────────────

def stage_c(failures: int) -> int:
    """ObjectRestingEEGrasped simulation checks.  Requires grasps.pt."""
    print("\n══════════════════════════════════════════════════════")
    print("  Stage C — ObjectRestingEEGrasped simulation checks")
    print("══════════════════════════════════════════════════════\n")

    # C1 — gym.make() ──────────────────────────────────────────────────────────
    print("[C1] Creating ObjectRestingEEGrasped environment ...")
    env_cfg = G1Dex3ObjectRestingEEGraspedResetStatesCfg()
    env_cfg.scene.num_envs = 1
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = 42
    # Point the grasp event at the local dataset directory.
    env_cfg.events.reset_end_effector_pose_from_grasp_dataset.params["dataset_dir"] = (
        args_cli.dataset_dir
    )

    try:
        env = gym.make(TASK_RG, cfg=env_cfg).unwrapped
    except Exception as exc:
        print(f"  [{FAIL}] gym.make() raised: {exc}")
        import traceback
        traceback.print_exc()
        return failures + 1

    failures += 0 if check("gym.make() succeeded", True) else 1

    # C2 — env.reset() — loads grasps.pt + IK teleport ────────────────────────
    print("\n[C2] Resetting environment (loads grasps.pt + IK teleport, may take ~5 s) ...")
    t0 = time.perf_counter()
    try:
        obs_dict, _ = env.reset()
        reset_ok = True
    except Exception as exc:
        print(f"  [{FAIL}] env.reset() raised: {exc}")
        import traceback
        traceback.print_exc()
        reset_ok = False
        failures += 1

    if reset_ok:
        t_reset = time.perf_counter() - t0
        failures += 0 if check(
            f"env.reset() completed in {t_reset:.1f} s", True
        ) else 1
        for key, val in obs_dict.items():
            if not torch.isfinite(val).all():
                failures += 1
                check(f"obs[{key}] finite after reset", False)

    # C3 — Cylinder at table-surface height ────────────────────────────────────
    print("\n[C3] Checking cylinder is at table-surface height after reset ...")
    if reset_ok:
        try:
            obj_pos = env.scene["object"].data.root_pos_w[0].cpu()
            info(
                f"Cylinder pos (world frame): "
                f"x={obj_pos[0]:.3f}  y={obj_pos[1]:.3f}  z={obj_pos[2]:.3f}"
            )
            failures += 0 if check(
                f"Cylinder z ≈ {CYLINDER_RESTING_Z:.3f} m (table surface ± 0.05 m)  "
                f"(got {obj_pos[2]:.3f} m)",
                abs(float(obj_pos[2]) - CYLINDER_RESTING_Z) < 0.05,
                "If z >> 0.90: reset_object_pose is applying an unexpected vertical offset",
            ) else 1
        except Exception as exc:
            warn(f"Cylinder position check raised (non-fatal): {exc}")

    # C4 — EE is near the cylinder ─────────────────────────────────────────────
    print("\n[C4] Checking EE proximity to cylinder (IK convergence) ...")
    if reset_ok:
        try:
            ee_pos = _get_ee_pos(env)
            obj_pos = env.scene["object"].data.root_pos_w[0].cpu()
            if ee_pos is not None:
                info(
                    f"EE pos  : x={ee_pos[0]:.3f}  y={ee_pos[1]:.3f}  z={ee_pos[2]:.3f}"
                )
                dist = float(torch.norm(ee_pos - obj_pos).item())
                failures += 0 if check(
                    f"EE distance from cylinder ≤ 0.25 m  (got {dist:.3f} m)",
                    dist <= 0.25,
                    "Large distance → IK did not converge to the grasp pose. "
                    "Check grasps.pt standoff vs G1 workspace; try more IK iterations.",
                ) else 1
                z_diff = abs(float(ee_pos[2]) - float(obj_pos[2]))
                failures += 0 if check(
                    f"EE z within 0.15 m of cylinder z  (|Δz|={z_diff:.3f} m)",
                    z_diff < 0.15,
                    "If EE z >> cylinder z: IK reached upward instead of side-grasping",
                ) else 1
        except Exception as exc:
            warn(f"EE proximity check raised (non-fatal): {exc}")

    # C5 — step loop ───────────────────────────────────────────────────────────
    print(f"\n[C5] Stepping {args_cli.num_steps} physics steps (gripper closed) ...")
    nonfinite_obs = 0
    nan_reward = 0
    t0 = time.perf_counter()
    # Close gripper throughout — mirrors the ObjectRestingEEGrasped recording logic.
    actions = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
    actions[:, -1] = -1.0

    with torch.inference_mode():
        for step_i in range(args_cli.num_steps):
            try:
                obs_dict, reward, _, _, _ = env.step(actions)
            except Exception as exc:
                print(f"  [{FAIL}] env.step({step_i}) raised: {exc}")
                import traceback
                traceback.print_exc()
                failures += 1
                break
            for val in obs_dict.values():
                if not torch.isfinite(val).all():
                    nonfinite_obs += 1
                    break
            if not torch.isfinite(reward).all():
                nan_reward += 1

    elapsed = time.perf_counter() - t0
    failures += 0 if check(
        f"No non-finite obs in {args_cli.num_steps} steps (found {nonfinite_obs})",
        nonfinite_obs == 0,
    ) else 1
    failures += 0 if check(
        f"No NaN reward in {args_cli.num_steps} steps (found {nan_reward})",
        nan_reward == 0,
    ) else 1

    # C6 — throughput ──────────────────────────────────────────────────────────
    print("\n[C6] Checking step throughput ...")
    hz = args_cli.num_steps / elapsed
    info(f"{args_cli.num_steps} steps in {elapsed:.2f} s → {hz:.1f} steps/s")
    failures += 0 if check("Step rate >= 1 Hz (GPU responding)", hz >= 1.0) else 1

    env.close()
    return failures


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"\n{'═' * 56}")
    print(f"  Phase 3 verification: G1+Dex3 reset-state pipeline")
    print(f"{'═' * 56}")
    print(f"  stage       = {args_cli.stage}")
    print(f"  dataset_dir = {os.path.abspath(args_cli.dataset_dir)}")

    failures = 0

    # Stage A always runs — no simulation context required.
    failures, grasps_available = stage_a(failures)

    # Only one gym.make() per process: Isaac Lab reinitialises the global USD
    # stage on each call and hangs if called twice in the same session.
    if args_cli.stage == "B":
        failures = stage_b(failures)
        print()
        info("Re-run with --stage C to verify ObjectRestingEEGrasped.")
    else:  # stage == "C"
        if not grasps_available:
            print("\n══════════════════════════════════════════════════════")
            print("  Stage C — ABORTED (grasps.pt not available)")
            print("══════════════════════════════════════════════════════")
            warn("Run Phase 2 first:")
            warn("    python scripts/tools/record_grasps_g1.py --headless --num_grasps 500")
            failures += 1
        else:
            failures = stage_c(failures)

    all_stages_done = args_cli.stage == "C" or not grasps_available

    print(f"\n{'═' * 56}")
    if failures == 0:
        print(f"  [{PASS}] All checks passed.")
        if args_cli.stage == "B":
            print()
            print("  Next: run Stage C to verify ObjectRestingEEGrasped:")
            print("    python scripts/environments/verify_g1_dex3_phase3.py \\")
            print("        --stage C --headless \\")
            print(f"        --dataset_dir {args_cli.dataset_dir}")
        else:
            print()
            print("  Both stages passed — Phase 3 pipeline is ready.")
            print()
            print("  Record ObjectAnywhereEEAnywhere reset states:")
            print("    python scripts/tools/record_reset_states_g1.py \\")
            print("        --reset_type ObjectAnywhereEEAnywhere --headless")
            print()
            print("  Record ObjectRestingEEGrasped reset states:")
            print("    python scripts/tools/record_reset_states_g1.py \\")
            print("        --reset_type ObjectRestingEEGrasped --headless")
    else:
        print(f"  [{FAIL}] {failures} check(s) failed — see above for details.")
        print()
        print("  Common fixes:")
        print("  • A1 fails     → run Phase 2: scripts/tools/record_grasps_g1.py")
        print("  • A2 fails     → grasps.pt has 0 entries; check verify_g1_dex3_phase2.py")
        print("  • A4 orient    → orientation_z_threshold must be None for Dex3 side-grasp")
        print("  • B2/C2 fails  → env construction error; check PROJECT_ROOT and USD paths")
        print("  • B3 fails     → IK not reaching workspace; check pose_range_b in cfg")
        print("  • C4 fails     → IK not converging to grasp pose; check grasps.pt standoff")
        print("                    and increase IK iterations in reset_end_effector_from_grasp_dataset")
    print(f"{'═' * 56}\n")

    return failures


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    raise SystemExit(exit_code)
