#!/usr/bin/env python3
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a Dex3 grasp dataset for the cylinder pick task (Phase 2).

Usage
-----
    python scripts/tools/record_grasps_g1.py --headless

    # Custom output directory and grasp count:
    python scripts/tools/record_grasps_g1.py --headless \
        --dataset_dir ./Datasets/OmniReset/ \
        --num_grasps 500

    # Sanity-check with visualisation (shows candidate poses in the viewer):
    python scripts/tools/record_grasps_g1.py --visualize_grasps

Output
------
    <dataset_dir>/Grasps/Cylinder/grasps.pt

    The .pt file contains a list of tensors; each entry is a 4×4 homogeneous
    transform T_palm_in_cylinder_frame (palm pose expressed in the cylinder's
    local coordinate frame).  Phase 3 uses these to initialise the
    ObjectRestingEEGrasped reset state.

Differences from the original record_grasps.py (UR5e Robotiq version)
----------------------------------------------------------------------
1.  No hydra_task_compose decorator — config is instantiated directly to avoid
    a dependency on Hydra and the system gym registry lookup.

2.  Object name is hard-coded to "Cylinder" — CylinderCfg is a procedural
    primitive with no usd_path attribute, so object_name_from_usd() cannot be
    called.

3.  gripper_body_name is "right_hand_palm_link" — the articulation root of the
    standalone Dex3 hand USD, equivalent to "robotiq_base_link" for the 2F85.

Pre-requisites
--------------
    export PROJECT_ROOT=/home/ygoraly/git/unitree_sim_isaaclab

    # Run once if the standalone hand USD does not yet exist:
    cd $PROJECT_ROOT
    python3 /home/ygoraly/git/UWLab-g1/scripts/tools/extract_dex3_right_hand_usd.py
"""

from __future__ import annotations

"""Launch Isaac Sim first."""

import argparse
import os
import time
from typing import cast

from tqdm import tqdm

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record Dex3 grasp dataset for the cylinder pick task.")
parser.add_argument("--num_envs", type=int, default=1, help="Parallel environments (keep low for VRAM).")
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="./Datasets/OmniReset/",
    help="Root output directory.  Grasps written to <dataset_dir>/Grasps/Cylinder/grasps.pt",
)
parser.add_argument("--num_grasps", type=int, default=500, help="Number of successful grasps to record.")
parser.add_argument(
    "--visualize_grasps",
    action="store_true",
    help="Show candidate grasp frames in the Isaac viewport for a visual sanity-check.  "
    "Slows down sampling; use before a full 500-grasp run to verify geometry.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything after this point runs inside the simulation context."""

import sys

import gymnasium as gym
import torch

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
        "Then verify the standalone hand USD exists:\n"
        "    ls $PROJECT_ROOT/assets/robots/dex3-right-hand-usd/dex3_right_hand.usd\n"
        "If not, run the extraction script first:\n"
        "    python3 scripts/tools/extract_dex3_right_hand_usd.py"
    )

import uwlab_tasks  # noqa: F401 — triggers gym.register for all UWLab-g1 envs

from uwlab.utils.datasets.torch_dataset_file_handler import TorchDatasetFileHandler

import uwlab_tasks.manager_based.manipulation.omnireset.mdp as task_mdp
from uwlab_tasks.manager_based.manipulation.omnireset.config.g1_dex3.grasp_sampling_cfg import (
    Dex3GraspSamplingCfg,
)

TASK_ID = "OmniReset-Dex3-GraspSampling-Cylinder-v0"

# The object for which grasps are being recorded.
# Hard-coded because CylinderCfg is a procedural primitive with no usd_path.
OBJ_NAME = "Cylinder"

# The body whose pose is recorded relative to the cylinder.
# Equivalent to "robotiq_base_link" in the UR5e version.
GRIPPER_BODY_NAME = "right_hand_palm_link"


def main() -> None:
    # ------------------------------------------------------------------
    # Output path
    # ------------------------------------------------------------------
    output_dir = os.path.join(args_cli.dataset_dir, "Grasps", OBJ_NAME)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "grasps.pt")

    print(f"\n=== Dex3 grasp recorder ===")
    print(f"  Task        : {TASK_ID}")
    print(f"  Object      : {OBJ_NAME}")
    print(f"  Palm body   : {GRIPPER_BODY_NAME}")
    print(f"  Target      : {args_cli.num_grasps} successful grasps")
    print(f"  Output      : {output_path}")
    print()

    # ------------------------------------------------------------------
    # Build config directly — avoids hydra_task_compose and system-registry
    # lookup issues that arise when another uwlab_tasks copy is installed.
    # ------------------------------------------------------------------
    env_cfg = Dex3GraspSamplingCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = None  # non-deterministic so grasp diversity is maximised

    # Enable visualisation if requested (adds frame markers to the viewport)
    if args_cli.visualize_grasps:
        env_cfg.events.grasp_sampling.params["visualize_grasps"] = True
        print("  [INFO] visualize_grasps=True — grasp candidate frames will be drawn in the viewport.")
        print("         Reduce num_grasps or run headlessly for a faster full dataset recording run.")

    # ------------------------------------------------------------------
    # Attach recorder
    # ------------------------------------------------------------------
    env_cfg.recorders = task_mdp.GraspRelativePoseRecorderManagerCfg(
        robot_name="robot",
        object_name="object",
        gripper_body_name=GRIPPER_BODY_NAME,
    )
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = "grasps.pt"
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.dataset_file_handler_class_type = TorchDatasetFileHandler

    # ------------------------------------------------------------------
    # Create environment
    # ------------------------------------------------------------------
    print("[1/3] Creating grasp sampling environment ...")
    env = cast(ManagerBasedRLEnv, gym.make(TASK_ID, cfg=env_cfg)).unwrapped

    # First reset triggers grasp candidate generation (expensive — ~30s on GPU)
    print("[2/3] Generating grasp candidates (first reset may take ~30 s) ...")
    t0 = time.time()
    env.reset()
    t_gen = time.time() - t0

    num_candidates = len(env.grasp_candidates) if hasattr(env, "grasp_candidates") else 0
    print(f"      Candidates generated: {num_candidates:,}  ({t_gen:.1f} s)")

    if num_candidates == 0:
        print("  [ERROR] No grasp candidates generated. Check:")
        print("    - metadata.yaml exists at $PROJECT_ROOT/assets/robots/dex3-right-hand-usd/")
        print("    - dex3_right_hand.usd exists and PROJECT_ROOT is set correctly")
        env.close()
        return

    # ------------------------------------------------------------------
    # Recording loop
    # ------------------------------------------------------------------
    print(f"[3/3] Running grasp evaluation (target: {args_cli.num_grasps} successes) ...")

    # Close action: all fingers fully closed
    actions = -torch.ones(env.action_space.shape, device=env.device, dtype=torch.float32)

    num_grasps_evaluated = 0
    current_successful_grasps = 0
    pbar = tqdm(total=args_cli.num_grasps, desc="Successful grasps", unit="grasps")

    t_start = time.time()
    while current_successful_grasps < args_cli.num_grasps:
        _, _, terminated, truncated, _ = env.step(actions)
        dones = terminated | truncated
        num_grasps_evaluated += int(dones.sum().item())

        new_count = env.recorder_manager.exported_successful_episode_count
        if new_count > current_successful_grasps:
            pbar.update(new_count - current_successful_grasps)
            current_successful_grasps = new_count

        if env.sim.is_stopped():
            print("  [WARN] Simulation stopped early.")
            break

    pbar.close()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    final_count = env.recorder_manager.exported_successful_episode_count
    success_rate = final_count / num_grasps_evaluated if num_grasps_evaluated > 0 else 0.0

    print()
    print(f"  Grasps evaluated : {num_grasps_evaluated:,}")
    print(f"  Successful       : {final_count:,}")
    print(f"  Success rate     : {success_rate:.1%}")
    print(f"  Time             : {elapsed / 60:.1f} min")
    print(f"  Output           : {output_path}")

    if final_count == 0:
        print()
        print("  [WARN] Zero successful grasps recorded.")
        print("  Debugging tips:")
        print("    1. Re-run with --visualize_grasps to inspect candidate poses.")
        print("    2. Check metadata.yaml values (maximum_aperture, finger_offset,")
        print("       gripper_approach_direction) against the actual hand coordinate frame.")
        print("    3. Run verify_g1_dex3_phase2.py to check each component individually.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
