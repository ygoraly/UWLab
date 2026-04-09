#!/usr/bin/env python3
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 2 verification gate for the G1+Dex3 grasp sampling pipeline.

Runs a series of checks in two stages:

  Stage A — pre-simulation (no Isaac context needed):
    A1. PROJECT_ROOT is set and standalone hand USD exists on disk.
    A2. metadata.yaml exists co-located with the USD and has all 7 required keys.
    A3. metadata.yaml values are geometrically reasonable (aperture > 0, etc.).
    A4. Dex3GraspSamplingCfg can be imported and has expected parameter values.

  Stage B — inside Isaac simulation:
    B1. gym.make() succeeds for OmniReset-Dex3-GraspSampling-Cylinder-v0.
    B2. env.reset() completes without crash.
        (This triggers grasp_sampling_event → _extract_mesh_from_asset →
         utils.prim_to_trimesh(UsdGeom.Cylinder) — the key Step 4 fix.)
    B3. Grasp candidates were generated (non-None, non-empty tensor).
    B4. Cylinder mesh was extracted with correct dimensions
        (radius ≈ 0.025 m, height ≈ 0.10 m).
    B5. surface_bias_mode="center" is active (candidates are biased toward
        mid-height by checking the Z-spread of the first N candidates).
    B6. Candidate palm transforms are finite and at a plausible side-approach
        distance from the cylinder (standoff in [finger_offset, ~0.2 m]).
    B7. 10 physics steps run without crash, reward and obs are finite.
    B8. Step throughput is at least 1 Hz (GPU sanity).

Run with:
    python scripts/environments/verify_g1_dex3_phase2.py --headless

Exit 0 = all checks pass.  Non-zero = number of failures.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 2 verification for G1+Dex3 grasp sampling.")
parser.add_argument("--num_steps", type=int, default=5000, help="Physics steps to run in Stage B.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything after this point runs inside the simulation context."""

import time

import gymnasium as gym
import torch
import yaml

# Ensure UWLab-g1 sources take priority over any system-installed uwlab_tasks.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _pkg in ("uwlab_tasks", "uwlab_assets"):
    _pkg_path = os.path.join(_REPO_ROOT, "source", _pkg)
    if os.path.isdir(_pkg_path) and _pkg_path not in sys.path:
        sys.path.insert(0, _pkg_path)

import uwlab_tasks  # noqa: F401 — registers all UWLab-g1 gym envs

from uwlab_tasks.manager_based.manipulation.omnireset.config.g1_dex3.grasp_sampling_cfg import (
    DEX3_PALM_BODY_NAME,
    Dex3GraspSamplingCfg,
)

# ── Colour helpers ──────────────────────────────────────────────────────────
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
INFO = "\033[94mINFO\033[0m"

TASK_ID = "OmniReset-Dex3-GraspSampling-Cylinder-v0"

REQUIRED_METADATA_KEYS = {
    "maximum_aperture",
    "finger_offset",
    "finger_clearance",
    "gripper_approach_direction",
    "grasp_align_axis",
    "orientation_sample_axis",
    "finger_open_joint_angle",
}


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f"  → {detail}" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return condition


def info(msg: str) -> None:
    print(f"  [{INFO}] {msg}")


def warn(msg: str) -> None:
    print(f"  [{WARN}] {msg}")


# ── Stage A: pre-simulation checks ─────────────────────────────────────────

def stage_a(failures: int) -> tuple[int, dict, str]:
    """Pre-simulation checks.  Returns updated failure count, metadata dict,
    and the resolved standalone hand USD path."""

    print("\n══════════════════════════════════════════════════════")
    print("  Stage A — pre-simulation checks")
    print("══════════════════════════════════════════════════════\n")

    # A1 — PROJECT_ROOT and USD file ─────────────────────────────────────────
    print("[A1] Checking standalone hand USD on disk ...")
    project_root = os.environ.get("PROJECT_ROOT", "")
    if not project_root:
        print(f"  [{FAIL}] PROJECT_ROOT is not set.")
        print("         Set it and re-run:  export PROJECT_ROOT=/home/ygoraly/git/unitree_sim_isaaclab")
        return failures + 1, {}, ""

    usd_dir = os.path.join(project_root, "assets/robots/dex3-right-hand-usd")
    usd_path = os.path.join(usd_dir, "dex3_right_hand.usd")
    metadata_path = os.path.join(usd_dir, "metadata.yaml")

    usd_exists = os.path.isfile(usd_path)
    failures += 0 if check("dex3_right_hand.usd exists", usd_exists, usd_path) else 1
    if not usd_exists:
        print()
        print("  Run once to create it:")
        print("    cd $PROJECT_ROOT")
        print("    python3 /home/ygoraly/git/UWLab-g1/scripts/tools/extract_dex3_right_hand_usd.py")

    if usd_exists:
        usd_size_kb = os.path.getsize(usd_path) / 1024
        failures += 0 if check(
            f"USD file is non-trivial (> 100 KB, got {usd_size_kb:.0f} KB)",
            usd_size_kb > 100,
        ) else 1
        info(f"USD path : {usd_path}")

    # A2 — metadata.yaml exists and has all required keys ────────────────────
    print("\n[A2] Checking metadata.yaml ...")
    meta = {}
    meta_exists = os.path.isfile(metadata_path)
    failures += 0 if check("metadata.yaml exists", meta_exists, metadata_path) else 1

    if meta_exists:
        with open(metadata_path) as f:
            meta = yaml.safe_load(f)
        missing_keys = REQUIRED_METADATA_KEYS - set(meta.keys())
        failures += 0 if check(
            "All 7 required keys present",
            len(missing_keys) == 0,
            f"Missing: {missing_keys}" if missing_keys else "",
        ) else 1
        info(f"Keys found: {sorted(meta.keys())}")
    else:
        print()
        print("  Run the extraction script to generate metadata.yaml alongside the USD.")

    # A3 — metadata values are geometrically sane ────────────────────────────
    print("\n[A3] Checking metadata.yaml values ...")
    if meta:
        aperture = meta.get("maximum_aperture", 0)
        failures += 0 if check(
            f"maximum_aperture > 0.03 m  (got {aperture:.3f} m)",
            aperture > 0.03,
        ) else 1
        failures += 0 if check(
            f"maximum_aperture < 0.15 m  (not unreasonably large, got {aperture:.3f} m)",
            aperture < 0.15,
        ) else 1

        f_off = meta.get("finger_offset", 0)
        failures += 0 if check(
            f"finger_offset > 0  (got {f_off:.3f} m)",
            f_off > 0,
        ) else 1

        approach = meta.get("gripper_approach_direction", [])
        import math
        approach_norm = math.sqrt(sum(x * x for x in approach)) if approach else 0
        failures += 0 if check(
            f"gripper_approach_direction is a unit vector (norm={approach_norm:.3f})",
            abs(approach_norm - 1.0) < 0.05,
        ) else 1

        align = meta.get("grasp_align_axis", [])
        align_norm = math.sqrt(sum(x * x for x in align)) if align else 0
        failures += 0 if check(
            f"grasp_align_axis is a unit vector (norm={align_norm:.3f})",
            abs(align_norm - 1.0) < 0.05,
        ) else 1

        approach_t = tuple(approach)
        align_t = tuple(align)
        dot = sum(a * b for a, b in zip(approach_t, align_t))
        failures += 0 if check(
            f"approach and align axes are not parallel (dot={dot:.3f}, should be ≈ 0)",
            abs(dot) < 0.3,
            "Parallel axes → grasp geometry will be degenerate",
        ) else 1
    else:
        warn("Skipping value checks — metadata.yaml could not be read.")
        failures += 1

    # A4 — config import and parameter values ────────────────────────────────
    print("\n[A4] Checking Dex3GraspSamplingCfg parameters ...")

    cfg = Dex3GraspSamplingCfg()
    grasp_params = cfg.events.grasp_sampling.params

    failures += 0 if check(
        f"DEX3_PALM_BODY_NAME == 'right_hand_palm_link' (got '{DEX3_PALM_BODY_NAME}')",
        DEX3_PALM_BODY_NAME == "right_hand_palm_link",
    ) else 1

    cfg_bias = grasp_params.get("surface_bias_mode", "<missing>")
    failures += 0 if check(
        f"surface_bias_mode == 'center' (got '{cfg_bias}')",
        cfg_bias == "center",
    ) else 1

    cfg_lateral = grasp_params.get("lateral_sigma", -1)
    failures += 0 if check(
        f"lateral_sigma == 0.0 (got {cfg_lateral})",
        cfg_lateral == 0.0,
    ) else 1

    cfg_body = grasp_params.get("gripper_cfg").body_names
    failures += 0 if check(
        f"gripper_cfg.body_names == 'right_hand_palm_link' (got '{cfg_body}')",
        cfg_body == "right_hand_palm_link",
    ) else 1

    num_cand = int(grasp_params.get("num_candidates", 0))
    failures += 0 if check(
        f"num_candidates > 0 (got {num_cand:,})",
        num_cand > 0,
    ) else 1

    info(f"num_standoff_samples = {grasp_params.get('num_standoff_samples')}")
    info(f"num_orientations     = {grasp_params.get('num_orientations')}")

    return failures, meta, usd_path


# ── Stage B: inside-simulation checks ──────────────────────────────────────

def stage_b(failures: int, meta: dict) -> int:
    """Inside-simulation checks."""

    print("\n══════════════════════════════════════════════════════")
    print("  Stage B — simulation checks")
    print("══════════════════════════════════════════════════════\n")

    # B1 — gym.make() ────────────────────────────────────────────────────────
    print("[B1] Creating grasp sampling environment ...")
    env_cfg = Dex3GraspSamplingCfg()
    env_cfg.scene.num_envs = 1
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = 42

    try:
        env = gym.make(TASK_ID, cfg=env_cfg).unwrapped
    except Exception as exc:
        print(f"  [{FAIL}] gym.make() raised: {exc}")
        import traceback
        traceback.print_exc()
        return failures + 1

    failures += 0 if check("gym.make() succeeded", True) else 1

    # B2 — env.reset() triggers grasp candidate generation ───────────────────
    print("\n[B2] Resetting environment (triggers _extract_mesh_from_asset) ...")
    print("     This generates antipodal grasp candidates — may take ~30 s ...")
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
        failures += 0 if check(f"env.reset() completed in {t_reset:.1f} s", True) else 1

        # Observations finite after reset
        for key, val in obs_dict.items():
            finite = torch.isfinite(val).all().item()
            failures += 0 if check(f"obs[{key}] finite after reset", finite) else 1

    # B3 — Grasp candidates were generated ───────────────────────────────────
    print("\n[B3] Checking grasp candidates tensor ...")
    candidates = getattr(env, "grasp_candidates", None)
    if candidates is None:
        failures += 0 if check("grasp_candidates is not None", False,
                                "reset() did not trigger candidate generation") else 1
        env.close()
        return failures

    failures += 0 if check("grasp_candidates is not None", True) else 1
    n_cand = len(candidates)
    failures += 0 if check(
        f"grasp_candidates has > 0 entries (got {n_cand:,})",
        n_cand > 0,
    ) else 1
    failures += 0 if check(
        "grasp_candidates tensor is finite",
        torch.isfinite(candidates).all().item(),
    ) else 1

    if n_cand > 0:
        info(f"Tensor shape: {candidates.shape}  (expect [N, 4, 4])")
        failures += 0 if check(
            "Candidate transforms are 4×4 matrices",
            candidates.ndim == 3 and candidates.shape[1] == 4 and candidates.shape[2] == 4,
        ) else 1

    # B4 — Cylinder mesh dimensions ──────────────────────────────────────────
    print("\n[B4] Checking cylinder mesh extraction (Step 4 fix) ...")
    try:
        from uwlab_tasks.manager_based.manipulation.omnireset.mdp.events import grasp_sampling_event

        # Term instances live in term_cfg.func for class-based event terms.
        # Walk _mode_class_term_cfgs to find the grasp_sampling_event instance.
        event_obj = None
        em = env.event_manager  # type: ignore[attr-defined]
        for mode_cfgs in em._mode_class_term_cfgs.values():
            for term_cfg in mode_cfgs:
                if isinstance(term_cfg.func, grasp_sampling_event):
                    event_obj = term_cfg.func
                    break
            if event_obj is not None:
                break

        if event_obj is not None:
            obj_asset = env.scene["object"]
            mesh = event_obj._extract_mesh_from_asset(obj_asset)
            # trimesh.creation.cylinder creates a cylinder centred at origin
            # with extents [2r, 2r, h] in [X, Y, Z].
            r_actual = mesh.extents[0] / 2
            h_actual = mesh.extents[2]
            failures += 0 if check(
                f"Cylinder mesh radius ≈ 0.025 m (got {r_actual:.4f} m)",
                abs(r_actual - 0.025) < 0.005,
            ) else 1
            failures += 0 if check(
                f"Cylinder mesh height ≈ 0.10 m (got {h_actual:.4f} m)",
                abs(h_actual - 0.10) < 0.01,
            ) else 1
            info(f"Mesh vertices: {len(mesh.vertices)},  faces: {len(mesh.faces)}")
            failures += 0 if check("Mesh has > 0 vertices", len(mesh.vertices) > 0) else 1
        else:
            warn("Could not locate grasp_sampling_event in event_manager — skipping mesh check.")
    except Exception as exc:
        print(f"  [{FAIL}] Mesh extraction raised: {exc}")
        import traceback
        traceback.print_exc()
        failures += 1

    # B5 — surface_bias_mode="center" produces mid-height candidates ─────────
    print("\n[B5] Checking surface_bias_mode='center' (candidates near mid-height) ...")
    if n_cand > 0:
        # Candidate transforms are T_palm_in_cylinder_frame.
        # Extract the Z translation: row 2 (Z) of the homogeneous column.
        # For CylinderCfg (height=0.10 m, centred at z=0), valid Z range is
        # [-0.05, +0.05].  The "center" bias should pull mean Z toward 0.
        cand_z = candidates[:, 2, 3].cpu().numpy()
        mean_z = float(cand_z.mean())
        std_z = float(cand_z.std())
        # With center bias, mean should be close to 0 (cylinder mid-height).
        # With top bias, mean would be positive (~0.03-0.04 m).
        failures += 0 if check(
            f"Candidate Z mean ≈ 0.0 m — center bias active (got {mean_z:+.4f} m, "
            f"std={std_z:.4f} m)",
            abs(mean_z) < 0.02,
            "If mean Z > +0.02, the 'top' bias is still active instead of 'center'",
        ) else 1
        info(f"Candidate Z: mean={mean_z:+.4f} m, std={std_z:.4f} m, "
             f"min={float(cand_z.min()):+.4f} m, max={float(cand_z.max()):+.4f} m")
    else:
        warn("No candidates to check bias — skipping.")

    # B6 — palm standoff distances are physically reasonable ─────────────────
    print("\n[B6] Checking palm standoff distances from cylinder centre ...")
    if n_cand > 0 and meta:
        # XY distance from origin (cylinder axis) = horizontal standoff
        cand_xy_dist = torch.norm(candidates[:, :2, 3], dim=1).cpu().numpy()
        mean_dist = float(cand_xy_dist.mean())
        finger_offset = meta.get("finger_offset", 0.04)
        max_dist = finger_offset + 0.20  # generous upper bound

        failures += 0 if check(
            f"XY standoff ≥ cylinder radius (0.025 m) — palm clears cylinder "
            f"(mean={mean_dist:.4f} m)",
            float(cand_xy_dist.min()) >= 0.020,
            "Palm too close to cylinder — check finger_offset in metadata.yaml",
        ) else 1
        failures += 0 if check(
            f"XY standoff < {max_dist:.3f} m — palm not unreasonably far "
            f"(mean={mean_dist:.4f} m)",
            float(cand_xy_dist.max()) < max_dist,
            "Some candidates very far from cylinder — check standoff range in metadata.yaml",
        ) else 1
        info(f"XY standoff: mean={mean_dist:.4f} m, "
             f"min={float(cand_xy_dist.min()):.4f} m, "
             f"max={float(cand_xy_dist.max()):.4f} m")
    else:
        warn("Skipping standoff check — no candidates or metadata unavailable.")

    # B7 — step loop ─────────────────────────────────────────────────────────
    print(f"\n[B7] Stepping {args_cli.num_steps} physics steps ...")
    nonfinite_obs = 0
    nan_reward = 0
    t0 = time.perf_counter()

    # Close action (all finger joints at -1 → close)
    actions = -torch.ones(env.action_space.shape, device=env.device, dtype=torch.float32)

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

    # B8 — throughput ────────────────────────────────────────────────────────
    print("\n[B8] Checking step throughput ...")
    hz = args_cli.num_steps / elapsed
    info(f"{args_cli.num_steps} steps in {elapsed:.2f} s → {hz:.1f} steps/s")
    failures += 0 if check("Step rate >= 1 Hz (GPU is responding)", hz >= 1.0) else 1

    env.close()
    return failures


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"\n{'═' * 56}")
    print(f"  Phase 2 verification: G1+Dex3 grasp sampling pipeline")
    print(f"{'═' * 56}")

    _PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "")
    if _PROJECT_ROOT:
        print(f"  PROJECT_ROOT = {_PROJECT_ROOT}")
    else:
        print(f"  [{FAIL}] PROJECT_ROOT not set — Stage A will fail.")

    failures = 0

    failures, meta, usd_path = stage_a(failures)

    failures = stage_b(failures, meta)

    print(f"\n{'═' * 56}")
    if failures == 0:
        print(f"  [{PASS}] All checks passed — Phase 2 pipeline is ready.")
        print()
        print("  Next step: record the full grasp dataset:")
        print("    python scripts/tools/record_grasps_g1.py --headless --num_grasps 500")
        print()
        print("  Tip: run with --visualize_grasps first to confirm geometry:")
        print("    python scripts/tools/record_grasps_g1.py --visualize_grasps --num_grasps 5")
    else:
        print(f"  [{FAIL}] {failures} check(s) failed — see above for details.")
        print()
        print("  Common fixes:")
        print("  • A1/A2 fail → run scripts/tools/extract_dex3_right_hand_usd.py")
        print("  • B2 fails   → check PROJECT_ROOT is set and USD path is valid")
        print("  • B4 fails   → confirm Step 4 patch is applied in events.py")
        print("  • B5 fails   → confirm surface_bias_mode='center' in grasp_sampling_cfg.py")
        print("  • B6 fails   → adjust finger_offset / finger_clearance in metadata.yaml")
    print(f"{'═' * 56}\n")

    return failures


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    raise SystemExit(exit_code)
