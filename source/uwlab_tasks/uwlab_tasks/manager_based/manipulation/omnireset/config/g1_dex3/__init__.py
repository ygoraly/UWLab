# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 + Dex3 pick-cylinder OmniReset environments."""

import gymnasium as gym

from . import agents

# Base config (events=MISSING): useful as a programmatic base class, but not
# directly instantiable via gym.make.  Registered so it can be discovered by
# config sweeps / registry introspection tools.
gym.register(
    id="OmniReset-G1Dex3-Pick-Cylinder-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:G1Dex3PickCylinderCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# Training config (MultiResetManager with Phase 3 datasets).
# Requires ./Datasets/OmniReset/Resets/Cylinder/resets_*.pt.
gym.register(
    id="OmniReset-G1Dex3-Pick-Cylinder-Train-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:G1Dex3PickTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# Simple training config (uniform cylinder reset, no datasets needed).
# Useful for debugging or running without Phase 2/3 datasets.
gym.register(
    id="OmniReset-G1Dex3-Pick-Cylinder-Train-Simple-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:G1Dex3PickTrainSimpleCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# Phase 2 — Grasp sampling env for the Dex3 right hand + cylinder.
# Used by scripts/tools/record_grasps_g1.py and
#          scripts/environments/verify_g1_dex3_phase2.py.
#
# Pre-requisite: run the USD extraction script once to create the standalone
# hand USD before calling gym.make() with this id:
#   cd $PROJECT_ROOT && python3 .../scripts/tools/extract_dex3_right_hand_usd.py
gym.register(
    id="OmniReset-Dex3-GraspSampling-Cylinder-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_sampling_cfg:Dex3GraspSamplingCfg",
    },
)

# Phase 3 — Reset-state recording envs.
# These are NOT used via gym.make() during recording (record_reset_states_g1.py
# instantiates the cfg directly to avoid hydra/registry lookup issues).
# Registered here so the IDs are discoverable by config sweeps and for
# interactive viewer debugging: launch with
#   python scripts/environments/zero_agent.py \
#       --task OmniReset-G1Dex3-ResetStates-ObjectAnywhereEEAnywhere-v0
#
# Pre-requisite for ObjectRestingEEGrasped: grasps.pt must exist at
#   ./Datasets/OmniReset/Grasps/Cylinder/grasps.pt
#   (produced by scripts/tools/record_grasps_g1.py in Phase 2).
gym.register(
    id="OmniReset-G1Dex3-ResetStates-ObjectAnywhereEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.reset_states_cfg:G1Dex3ObjectAnywhereEEAnywhereResetStatesCfg"
        ),
    },
)

gym.register(
    id="OmniReset-G1Dex3-ResetStates-ObjectRestingEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.reset_states_cfg:G1Dex3ObjectRestingEEGraspedResetStatesCfg"
        ),
    },
)
