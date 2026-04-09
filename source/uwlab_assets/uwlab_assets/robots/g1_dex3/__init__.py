# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Unitree G1-29DOF robot with Dex3 hands (fixed base).

* :obj:`G1_DEX3_FIXED_BASE_CFG`: Full 29-DOF + Dex3 robot (fixed base) for RL training.
* :obj:`DEX3_RIGHT_STANDALONE_CFG`: Right-hand-only USD for grasp sampling (Phase 2 stub;
  USD path must be updated in dex3_right_standalone.py before use).
"""

from .dex3_right_standalone import DEX3_RIGHT_STANDALONE_CFG
from .g1_29dof_dex3 import G1_DEX3_FIXED_BASE_CFG
