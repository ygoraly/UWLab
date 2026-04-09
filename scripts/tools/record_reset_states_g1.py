#!/usr/bin/env python3
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record reset-state datasets for the G1+Dex3 pick-cylinder task (Phase 3).

Usage
-----
    # ObjectAnywhereEEAnywhere (no prerequisites beyond Phase 1):
    python scripts/tools/record_reset_states_g1.py \
        --reset_type ObjectAnywhereEEAnywhere --headless

    # ObjectRestingEEGrasped (requires grasps.pt from Phase 2):
    python scripts/tools/record_reset_states_g1.py \
        --reset_type ObjectRestingEEGrasped --headless

    # Custom output directory and state count:
    python scripts/tools/record_reset_states_g1.py \
        --reset_type ObjectAnywhereEEAnywhere \
        --dataset_dir ./Datasets/OmniReset/ \
        --num_reset_states 500 \
        --headless

Output
------
    <dataset_dir>/Resets/Cylinder/resets_ObjectAnywhereEEAnywhere.pt
    <dataset_dir>/Resets/Cylinder/resets_ObjectRestingEEGrasped.pt

    Each .pt file is a dict of tensors containing the full scene state
    (robot joint positions/velocities, cylinder pose, table pose) at the
    moment check_reset_state_success fired.  Phase 4 passes these files to
    MultiResetManager to initialise training episodes.

Differences from UWLab/scripts_v2/tools/record_reset_states.py (UR5e)
----------------------------------------------------------------------
1.  No @hydra_task_compose decorator — cfg is instantiated directly to avoid
    a dependency on Hydra and the system gym registry, which can resolve to
    the UWLab (non-G1) package if both are installed.

2.  No insertive_object / receptive_object in the scene — the pair directory
    cannot be derived from USD paths (CylinderCfg is a procedural primitive
    with no usd_path).  Hard-coded to "Cylinder" so the output path matches
    what MultiResetManager will expect in Phase 4.

3.  --reset_type selects between two concrete cfg classes directly instead of
    going through the gym task ID.  The two supported values are:
        ObjectAnywhereEEAnywhere  — no grasp dataset required
        ObjectRestingEEGrasped    — requires grasps.pt from Phase 2

4.  Prerequisite check for ObjectRestingEEGrasped: grasps.pt must exist before
    the env is constructed (reset_end_effector_from_grasp_dataset loads it in
    __init__; a missing file produces an opaque FileNotFoundError).

Pre-requisites
--------------
    export PROJECT_ROOT=/home/ygoraly/git/unitree_sim_isaaclab

    # Phase 2 must be complete before recording ObjectRestingEEGrasped:
    #   ./Datasets/OmniReset/Grasps/Cylinder/grasps.pt
"""

from __future__ import annotations

"""Launch Isaac Sim first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

VALID_RESET_TYPES = ["ObjectAnywhereEEAnywhere", "ObjectRestingEEGrasped"]

parser = argparse.ArgumentParser(description="Record G1+Dex3 reset-state datasets (Phase 3).")
parser.add_argument(
    "--reset_type",
    type=str,
    required=True,
    choices=VALID_RESET_TYPES,
    help="Which reset-state type to record.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="./Datasets/OmniReset/",
    help="Root output directory.  States written to <dataset_dir>/Resets/Cylinder/resets_<type>.pt",
)
parser.add_argument("--num_reset_states", type=int, default=200, help="Number of successful states to record.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything after this point runs inside the simulation context."""

import time

import torch
from tqdm import tqdm
from typing import cast

import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.recorder_manager import DatasetExportMode

# Ensure UWLab-g1 sources take priority over any system-installed uwlab_tasks.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _pkg in ("uwlab_tasks", "uwlab_assets"):
    _pkg_path = os.path.join(_REPO_ROOT, "source", _pkg)
    if os.path.isdir(_pkg_path) and _pkg_path not in sys.path:
        sys.path.insert(0, _pkg_path)

_PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "")
if not _PROJECT_ROOT:
    raise EnvironmentError(
        "PROJECT_ROOT is not set.\n"
        "Set it to the unitree_sim_isaaclab repo root:\n"
        "    export PROJECT_ROOT=/home/ygoraly/git/unitree_sim_isaaclab\n"
    )

import uwlab_tasks  # noqa: F401 — triggers gym.register for all UWLab-g1 envs

from uwlab.utils.datasets.torch_dataset_file_handler import TorchDatasetFileHandler

import uwlab_tasks.manager_based.manipulation.omnireset.mdp as task_mdp
from uwlab_tasks.manager_based.manipulation.omnireset.config.g1_dex3.reset_states_cfg import (
    G1Dex3ObjectAnywhereEEAnywhereResetStatesCfg,
    G1Dex3ObjectRestingEEGraspedResetStatesCfg,
)

# The cylinder pair directory — hard-coded because CylinderCfg has no usd_path,
# so task_mdp.utils.compute_pair_dir() cannot be called.
# MultiResetManager in Phase 4 will be pointed at this same path.
PAIR_DIR = "Cylinder"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _check_prerequisites(reset_type: str, dataset_dir: str) -> None:
    """Abort early with a clear message if Phase 2 output is missing."""
    if reset_type == "ObjectRestingEEGrasped":
        grasps_path = os.path.join(dataset_dir, "Grasps", PAIR_DIR, "grasps.pt")
        if not os.path.exists(grasps_path):
            raise FileNotFoundError(
                f"ObjectRestingEEGrasped requires grasps.pt from Phase 2, but it was not found at:\n"
                f"    {grasps_path}\n"
                "Run Phase 2 first:\n"
                "    python scripts/tools/record_grasps_g1.py --headless"
            )
        print(f"  [OK] grasps.pt found: {grasps_path}")


def main() -> None:
    reset_type = args_cli.reset_type
    dataset_dir = args_cli.dataset_dir

    output_dir = os.path.join(dataset_dir, "Resets", PAIR_DIR)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"resets_{reset_type}.pt"
    output_path = os.path.join(output_dir, output_filename)

    print(f"\n=== G1+Dex3 reset-state recorder ===")
    print(f"  Reset type  : {reset_type}")
    print(f"  Pair dir    : {PAIR_DIR}")
    print(f"  Target      : {args_cli.num_reset_states} successful states")
    print(f"  Output      : {output_path}")
    print()

    # ------------------------------------------------------------------
    # Prerequisite checks
    # ------------------------------------------------------------------
    _check_prerequisites(reset_type, dataset_dir)

    # ------------------------------------------------------------------
    # Build cfg directly — avoids hydra_task_compose and system-registry
    # lookup issues when another uwlab_tasks copy is installed.
    # ------------------------------------------------------------------
    if reset_type == "ObjectAnywhereEEAnywhere":
        env_cfg = G1Dex3ObjectAnywhereEEAnywhereResetStatesCfg()
    else:
        env_cfg = G1Dex3ObjectRestingEEGraspedResetStatesCfg()
        # Point the grasp event at the local dataset directory.
        # The event reads this at __init__ time via _compute_grasp_dataset_path().
        env_cfg.events.reset_end_effector_pose_from_grasp_dataset.params["dataset_dir"] = dataset_dir

    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = None  # non-deterministic for diverse state discovery

    # ------------------------------------------------------------------
    # Attach recorder — StableStateRecorder fires at record_pre_reset,
    # capturing the full scene state the step before each successful reset.
    # ------------------------------------------------------------------
    env_cfg.recorders = task_mdp.StableStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_filename
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.dataset_file_handler_class_type = TorchDatasetFileHandler

    # ------------------------------------------------------------------
    # Create environment
    # ------------------------------------------------------------------
    task_id = (
        "OmniReset-G1Dex3-ResetStates-ObjectAnywhereEEAnywhere-v0"
        if reset_type == "ObjectAnywhereEEAnywhere"
        else "OmniReset-G1Dex3-ResetStates-ObjectRestingEEGrasped-v0"
    )
    print(f"[1/3] Creating environment ({task_id}) ...")
    env = cast(ManagerBasedRLEnv, gym.make(task_id, cfg=env_cfg)).unwrapped
    env.reset()
 
    # ------------------------------------------------------------------
    # Action setup
    # For ObjectRestingEEGrasped: close the gripper (-1.0 on last dim) so
    #   the hand attempts to hold the cylinder that the IK reset placed it on.
    # For ObjectAnywhereEEAnywhere: randomise gripper open/close each episode
    #   to add state diversity (gripper position doesn't affect the stability
    #   check, but varying it produces more realistic training initialisation).
    # Arm dimensions (first 6) are always zero — the reset events handle arm
    # placement; we don't want the policy to fight against them.
    # ------------------------------------------------------------------
    actions = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
    if reset_type == "ObjectRestingEEGrasped":
        actions[:, -1] = -1.0
    else:
        # Random open/close assigned at startup; updated per-done-env below.
        actions[:, -1] = torch.randint(0, 2, (env.num_envs,), device=env.device, dtype=torch.float32) * 2 - 1

    # ------------------------------------------------------------------
    # Recording loop — mirrors record_reset_states.py (UR5e)
    # ------------------------------------------------------------------
    print(f"[2/3] Running recording loop (target: {args_cli.num_reset_states} states) ...")
    num_episodes_evaluated = 0
    current_successful = 0
    pbar = tqdm(total=args_cli.num_reset_states, desc="Successful states", unit="states")
    t_start = time.time()

    while current_successful < args_cli.num_reset_states:
        _, _, terminated, truncated, _ = env.step(actions)
        dones = terminated | truncated
        done_idx = torch.where(dones)[0]

        # After mid-step resets (inside env.step), the same stale-data bug
        # applies: DiffIK will read pre-reset body_pos_w on the NEXT step
        # and drive the arm to the wrong pose.  Force a data-buffer sync so
        # the next process_actions() reads the IK-teleported state.
        if done_idx.numel() > 0:
            env.scene.update(env.physics_dt)

        new_count = env.recorder_manager.exported_successful_episode_count
        if new_count > current_successful:
            pbar.update(new_count - current_successful)
            current_successful = new_count

        num_episodes_evaluated += int(dones.sum().item())

        if env.sim.is_stopped():
            print("  [WARN] Simulation stopped early.")
            break

    pbar.close()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    final_count = env.recorder_manager.exported_successful_episode_count
    success_rate = final_count / num_episodes_evaluated if num_episodes_evaluated > 0 else 0.0

    print()
    print(f"[3/3] Done.")
    print(f"  Episodes evaluated : {num_episodes_evaluated:,}")
    print(f"  Successful states  : {final_count:,}")
    print(f"  Success rate       : {success_rate:.1%}")
    print(f"  Time               : {elapsed / 60:.1f} min")
    print(f"  Output             : {output_path}")

    if final_count == 0:
        print()
        print("  [WARN] Zero states recorded.  Debugging tips:")
        if reset_type == "ObjectRestingEEGrasped":
            print("    1. Check that grasps.pt has >0 entries (verify_g1_dex3_phase2.py).")
            print("    2. Re-run with --num_envs 1 (no --headless) and inspect the viewer:")
            print("       does the arm reach the cylinder after each reset?")
            print("       If not, IK is not converging — see Phase 3 notes on increasing")
            print("       IK iterations in reset_end_effector_from_grasp_dataset.")
            print("    3. Check orientation_z_threshold=None is set in the termination cfg.")
        else:
            print("    1. Re-run without --headless and inspect the viewer.")
            print("    2. Verify the arm reaches plausible poses after each reset.")
            print("    3. Check that episode_length_s (2.0 s) is long enough for the")
            print("       stability counter to reach consecutive_stability_steps=5.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
