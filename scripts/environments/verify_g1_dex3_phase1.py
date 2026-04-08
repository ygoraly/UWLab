# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 1 verification gate for the G1+Dex3 pick-cylinder environment.

Run with:
    python scripts/environments/verify_g1_dex3_phase1.py --headless

Pass criteria (all must hold):
    1. gym.make() and env.reset() complete without crash.
    2. All observations are finite (no NaN / Inf) at reset and after each step.
    3. No NaN in reward at any step.
    4. At least one of ee_object_distance or object_lift reward is non-zero
       by step 100 (sanity-check that they are wired up correctly).
    5. environment steps at >= 2 Hz wall-clock (GPU throughput sanity).

Exit code 0 = all pass.  Non-zero = at least one check failed.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 1 verification for G1+Dex3 pick-cylinder.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel envs (small to keep VRAM low).")
parser.add_argument("--num_steps", type=int, default=100, help="Number of steps to run after reset.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import sys
import time

import gymnasium as gym
import torch

# Ensure this script always loads uwlab_tasks and uwlab_assets from the
# UWLab-g1 repo, not from any other installed copy.  This is necessary
# because the conda env may have UWLab (without g1_dex3) installed first
# on sys.path, which would cause gym.register for the G1 env never to run.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _src in ("uwlab_tasks", "uwlab_assets"):
    _src_path = os.path.join(_REPO_ROOT, "source", _src)
    if os.path.isdir(_src_path) and _src_path not in sys.path:
        sys.path.insert(0, _src_path)

_PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "")
if not _PROJECT_ROOT:
    raise EnvironmentError(
        "PROJECT_ROOT is not set.\n"
        "The G1 USD lives at $PROJECT_ROOT/assets/robots/g1-29dof-dex3-base-fix-usd/...\n"
        "Run once to fetch assets:\n"
        "    cd /home/ygoraly/git/unitree_sim_isaaclab && bash fetch_assets.sh\n"
        "Then set PROJECT_ROOT to the unitree_sim_isaaclab repo root (where assets/ lives):\n"
        "    export PROJECT_ROOT=/home/ygoraly/git/unitree_sim_isaaclab\n"
        "Or persist it: conda env config vars set PROJECT_ROOT=/home/ygoraly/git/unitree_sim_isaaclab -n env_isaaclab"
    )

import uwlab_tasks  # noqa: F401 — triggers gym.register for all UWLab-g1 envs

from uwlab_tasks.manager_based.manipulation.omnireset.config.g1_dex3.rl_state_cfg import (
    G1Dex3PickTrainCfg,
)

TASK_ID = "OmniReset-G1Dex3-Pick-Cylinder-Train-v0"

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(label: str, condition: bool) -> bool:
    print(f"  [{PASS if condition else FAIL}] {label}")
    return condition


def main() -> int:
    failures = 0

    # ------------------------------------------------------------------
    # 1. Environment construction
    # ------------------------------------------------------------------
    print(f"\n=== Phase 1 verification: {TASK_ID} ===\n")
    print("[1/5] Building environment ...")
    # Instantiate the config directly rather than going through parse_env_cfg
    # (which also relies on the gym registry lookup and can fail if the env ID
    # was not yet registered from the system-installed uwlab_tasks).
    env_cfg = G1Dex3PickTrainCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    try:
        env = gym.make(TASK_ID, cfg=env_cfg)
    except Exception as exc:
        print(f"  [{FAIL}] gym.make() raised: {exc}")
        return 1

    print(f"  obs space  : {env.observation_space}")
    print(f"  action space: {env.action_space}")

    # ------------------------------------------------------------------
    # 2. Reset
    # ------------------------------------------------------------------
    print("\n[2/5] Resetting environment ...")
    try:
        obs_dict, _ = env.reset()
    except Exception as exc:
        print(f"  [{FAIL}] env.reset() raised: {exc}")
        env.close()
        return 1

    for key, val in obs_dict.items():
        finite = torch.isfinite(val).all().item()
        failures += 0 if check(f"obs[{key}] finite after reset", finite) else 1

    # ------------------------------------------------------------------
    # 3. Step loop
    # ------------------------------------------------------------------
    print(f"\n[3/5] Stepping {args_cli.num_steps} steps with random actions ...")
    action_shape = env.action_space.shape
    reward_sum = torch.zeros(args_cli.num_envs, device=env.unwrapped.device)
    nonfinite_obs_steps = 0
    nan_reward_steps = 0

    t0 = time.perf_counter()
    with torch.inference_mode():
        for step in range(args_cli.num_steps):
            actions = 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1
            obs_dict, reward, terminated, truncated, _ = env.step(actions)

            # obs finiteness
            for val in obs_dict.values():
                if not torch.isfinite(val).all():
                    nonfinite_obs_steps += 1
                    break

            # reward finiteness
            if not torch.isfinite(reward).all():
                nan_reward_steps += 1

            reward_sum += reward

    elapsed = time.perf_counter() - t0
    hz = args_cli.num_steps / elapsed

    failures += 0 if check(f"No non-finite obs in {args_cli.num_steps} steps (found {nonfinite_obs_steps})",
                            nonfinite_obs_steps == 0) else 1
    failures += 0 if check(f"No NaN reward in {args_cli.num_steps} steps (found {nan_reward_steps})",
                            nan_reward_steps == 0) else 1

    # ------------------------------------------------------------------
    # 4. Reward signal sanity
    # ------------------------------------------------------------------
    print("\n[4/5] Checking reward signal is non-trivially wired ...")
    mean_reward = reward_sum.mean().item() / args_cli.num_steps
    print(f"  mean reward/step across {args_cli.num_envs} envs: {mean_reward:.4f}")
    # ee_object_distance returns 1-tanh(d/std); even at rest it is > 0 because
    # the EE starts far from the cylinder, so tanh(d) < 1 → value > 0.
    # Total reward should be clearly positive.
    failures += 0 if check("Mean reward/step > 0 (ee_object_distance is wired)", mean_reward > 0.0) else 1

    # ------------------------------------------------------------------
    # 5. Throughput
    # ------------------------------------------------------------------
    print("\n[5/5] Checking step throughput ...")
    print(f"  {args_cli.num_steps} steps in {elapsed:.2f}s → {hz:.1f} steps/s "
          f"(wall-clock, {args_cli.num_envs} envs)")
    failures += 0 if check("Step rate >= 2 Hz (GPU is doing work)", hz >= 2.0) else 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    if failures == 0:
        print(f"  [{PASS}] All checks passed — Phase 1 complete.")
    else:
        print(f"  [{FAIL}] {failures} check(s) failed — see above.")
    print(f"{'='*50}\n")

    env.close()
    return failures


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    raise SystemExit(exit_code)
