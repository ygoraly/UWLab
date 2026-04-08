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

# Training config (events=TrainEventCfg filled): this is the env to use for
# gym.make() during Phase 1 training and the verification gate.
gym.register(
    id="OmniReset-G1Dex3-Pick-Cylinder-Train-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:G1Dex3PickTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)
