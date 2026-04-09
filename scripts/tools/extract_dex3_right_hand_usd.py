#!/usr/bin/env python3
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One-time script to extract the Dex3 right-hand subtree from the full G1 USD.

Creates a standalone hand USD (right_hand_palm_link as articulation root) that
the grasp sampling env can freely teleport to arbitrary poses around a cylinder,
exactly like the ROBOTIQ_2F85 standalone gripper USD used by the UR5e version.

Usage
-----
Run from the unitree_sim_isaaclab directory (or anywhere with PROJECT_ROOT set):

    cd /home/ygoraly/git/unitree_sim_isaaclab
    python3 /home/ygoraly/git/UWLab-g1/scripts/tools/extract_dex3_right_hand_usd.py

Output
------
  $PROJECT_ROOT/assets/robots/dex3-right-hand-usd/dex3_right_hand.usd
  $PROJECT_ROOT/assets/robots/dex3-right-hand-usd/metadata.yaml

Why a standalone USD is needed
-------------------------------
grasp_sampling_event positions the "robot" articulation by calling
  gripper_asset.write_root_pose_to_sim(candidate_pose, env_ids)
This sets the ROOT BODY of the articulation to the candidate pose.

For the full G1 fixed-base USD, the articulation root is the pelvis (at z=0.75).
Teleporting the pelvis to a grasp candidate pose near a 30cm-high cylinder is
meaningless — the palm is 6–7 links away from the root.

For this standalone USD, right_hand_palm_link IS the articulation root.
Teleporting it = directly positioning the palm at the candidate pose. ✓

USD structure produced
----------------------
/dex3_right_hand                   (Xform + ArticulationRootAPI, disable_gravity)
  /right_hand_palm_link            (Xform + RigidBodyAPI — the FREE-FLOATING root)
  /right_hand_camera_base_link     (Xform + RigidBodyAPI)
  /right_hand_index_0_link         (Xform + RigidBodyAPI + collision mesh)
  /right_hand_index_1_link
  /right_hand_middle_0_link
  /right_hand_middle_1_link
  /right_hand_thumb_0_link
  /right_hand_thumb_1_link
  /right_hand_thumb_2_link
  /world_palm_link_joint           (FixedJoint, body0=world, body1=right_hand_palm_link)
  /joints/
    right_hand_camera_base_joint   (FixedJoint)
    right_hand_index_0_joint       (RevoluteJoint)
    right_hand_index_1_joint
    right_hand_middle_0_joint
    right_hand_middle_1_joint
    right_hand_thumb_0_joint
    right_hand_thumb_1_joint
    right_hand_thumb_2_joint

Note: right_hand_palm_joint (the FixedJoint that connects the palm to the wrist
in the full G1) is intentionally NOT included — the palm is now the free root.
"""

from __future__ import annotations

import os
import sys
import yaml
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "")
if not _PROJECT_ROOT:
    # Fallback: assume script is run from unitree_sim_isaaclab directory
    _PROJECT_ROOT = str(Path(__file__).parent.parent.parent.parent / "unitree_sim_isaaclab")

SRC_USD = os.path.join(
    _PROJECT_ROOT,
    "assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd",
)
DST_DIR = os.path.join(_PROJECT_ROOT, "assets/robots/dex3-right-hand-usd")
DST_USD = os.path.join(DST_DIR, "dex3_right_hand.usd")
DST_METADATA = os.path.join(DST_DIR, "metadata.yaml")

# Prim path root in the source USD
SRC_ROOT = "/g1_29dof_with_hand_rev_1_0"

# Bodies to include (relative names under SRC_ROOT)
BODY_NAMES = [
    "right_hand_palm_link",
    # right_hand_camera_base_link intentionally excluded: the camera is a
    # perception sensor not needed in simulation.  Its joint has no matching
    # actuator group and would be left undriven (stiffness=0, damping=0),
    # causing the camera module to detach and spin freely in PhysX.
    "right_hand_index_0_link",
    "right_hand_index_1_link",
    "right_hand_middle_0_link",
    "right_hand_middle_1_link",
    "right_hand_thumb_0_link",
    "right_hand_thumb_1_link",
    "right_hand_thumb_2_link",
]

# Joints to include (under SRC_ROOT/joints/).
# right_hand_palm_joint is intentionally excluded — the palm becomes the free root.
# right_hand_camera_base_joint is excluded for the same reason as the camera link above.
JOINT_NAMES = [
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]

DST_ROOT = "/dex3_right_hand"


def _remap_path(path_str: str) -> str:
    """Rewrite SRC_ROOT prefix to DST_ROOT prefix in an SdfPath string."""
    if path_str.startswith(SRC_ROOT + "/"):
        return DST_ROOT + path_str[len(SRC_ROOT):]
    if path_str == SRC_ROOT:
        return DST_ROOT
    return path_str


def extract(src_usd: str, dst_usd: str) -> None:
    """Extract the right Dex3 hand subtree and save to dst_usd."""
    try:
        from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
    except ImportError:
        print("ERROR: pxr (OpenUSD) not found. Run inside the env_isaaclab conda env.")
        sys.exit(1)

    if not os.path.exists(src_usd):
        print(f"ERROR: source USD not found: {src_usd}")
        print("       Run fetch_assets.sh from unitree_sim_isaaclab first.")
        sys.exit(1)

    os.makedirs(os.path.dirname(dst_usd), exist_ok=True)

    print(f"Opening source USD: {src_usd}")
    src_stage = Usd.Stage.Open(src_usd)

    # Flatten to resolve all sub-layers and variant selections
    src_layer = src_stage.Flatten()

    print(f"Creating destination USD: {dst_usd}")
    if os.path.exists(dst_usd):
        os.remove(dst_usd)
    dst_layer = Sdf.Layer.CreateNew(dst_usd)

    # -----------------------------------------------------------------------
    # Create the root Xform with ArticulationRootAPI
    # -----------------------------------------------------------------------
    root_spec = Sdf.PrimSpec(dst_layer, "dex3_right_hand", Sdf.SpecifierDef, "Xform")
    dst_layer.defaultPrim = "dex3_right_hand"

    # Apply ArticulationRootAPI so PhysX treats this as an articulation
    root_spec.SetInfo(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit(["PhysicsArticulationRootAPI"]),
    )

    # -----------------------------------------------------------------------
    # Copy body prims (geometry + physics APIs preserved by CopySpec)
    # -----------------------------------------------------------------------
    print("Copying body prims...")
    for body_name in BODY_NAMES:
        src_path = Sdf.Path(f"{SRC_ROOT}/{body_name}")
        dst_path = Sdf.Path(f"{DST_ROOT}/{body_name}")

        if not src_layer.GetPrimAtPath(src_path):
            print(f"  WARNING: {src_path} not found in source — skipping")
            continue

        ok = Sdf.CopySpec(src_layer, src_path, dst_layer, dst_path)
        if ok:
            print(f"  ✓ {body_name}")
        else:
            print(f"  ✗ {body_name} — CopySpec failed")

    # -----------------------------------------------------------------------
    # Create joints/ scope and copy joints with remapped body references
    # -----------------------------------------------------------------------
    print("Copying joints...")
    joints_scope_spec = Sdf.PrimSpec(root_spec, "joints", Sdf.SpecifierDef, "Scope")

    for joint_name in JOINT_NAMES:
        src_path = Sdf.Path(f"{SRC_ROOT}/joints/{joint_name}")
        dst_path = Sdf.Path(f"{DST_ROOT}/joints/{joint_name}")

        if not src_layer.GetPrimAtPath(src_path):
            print(f"  WARNING: {src_path} not found — skipping")
            continue

        ok = Sdf.CopySpec(src_layer, src_path, dst_layer, dst_path)
        if not ok:
            print(f"  ✗ {joint_name} — CopySpec failed")
            continue

        # Remap physics:body0 and physics:body1 targets
        dst_joint_spec = dst_layer.GetPrimAtPath(dst_path)
        if dst_joint_spec is None:
            continue

        for rel_name in ("physics:body0", "physics:body1"):
            rel = dst_joint_spec.relationships.get(rel_name)
            if rel is None:
                continue
            old_targets = list(rel.targetPathList.GetAddedOrExplicitItems())
            new_targets = [Sdf.Path(_remap_path(str(t))) for t in old_targets]
            rel.targetPathList.ClearEdits()
            for t in new_targets:
                rel.targetPathList.Append(t)

        print(f"  ✓ {joint_name}")

    # -----------------------------------------------------------------------
    # Copy Flattened_Prototype prims (shared mesh geometry blobs)
    #
    # When Usd.Stage.Flatten() is called, prototype/instanced geometry gets
    # moved to root-level prims named "Flattened_Prototype_N".  The collision
    # sub-prims of each body link hold *internal* references to these prims
    # (e.g.  @./dex3_right_hand.usd@</Flattened_Prototype_194>).  Without the
    # prototype prims present in the destination layer those references are
    # unresolved and RigidObjectHasher finds zero collision geometry → the
    # CollisionAnalyzer crashes with "torch.cat: empty list".
    # -----------------------------------------------------------------------
    print("Copying Flattened_Prototype mesh prims (collision geometry blobs)...")
    proto_count = 0
    for child_spec in src_layer.pseudoRoot.nameChildren:
        if child_spec.name.startswith("Flattened_Prototype"):
            ok = Sdf.CopySpec(src_layer, child_spec.path, dst_layer, Sdf.Path(f"/{child_spec.name}"))
            if ok:
                proto_count += 1
    print(f"  ✓ {proto_count} prototype prims copied")

    # -----------------------------------------------------------------------
    # Add a world-anchored FixedJoint to pin the palm during physics simulation.
    #
    # Isaac Lab's fix_root_link=True normally creates this joint at spawn time,
    # but that code path requires the ArticulationRootAPI prim to also carry
    # RigidBodyAPI — which is not the case for this USD (the root is an Xform,
    # the first rigid body is right_hand_palm_link one level below).
    #
    # By baking the joint into the USD itself, Isaac Lab's
    # find_global_fixed_joint_prim finds it (it searches for any joint with
    # only one body target), and simply calls SetJointEnabled(True) instead
    # of trying to create a new one.  This allows fix_root_link=True to work
    # without hitting the NotImplementedError.
    #
    # The joint has:
    #   physics:body0 — absent (empty = world frame anchor)
    #   physics:body1 — right_hand_palm_link
    # -----------------------------------------------------------------------
    print("Adding world-anchored FixedJoint for fix_root_link support...")
    joint_spec = Sdf.PrimSpec(root_spec, "world_palm_link_joint", Sdf.SpecifierDef, "PhysicsJoint")
    joint_spec.SetInfo(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit(["PhysicsFixedJointAPI"]),
    )
    body1_rel = Sdf.RelationshipSpec(joint_spec, "physics:body1", False)
    body1_rel.targetPathList.Append(Sdf.Path(f"{DST_ROOT}/right_hand_palm_link"))
    print(f"  ✓ world_palm_link_joint (body0=world, body1={DST_ROOT}/right_hand_palm_link)")

    dst_layer.Save()
    print(f"\nStandalone hand USD saved: {dst_usd}")


def write_metadata(dst_metadata: str) -> None:
    """Write metadata.yaml co-located with the standalone hand USD.

    grasp_sampling_event reads these 7 keys at env creation time.
    Values below are calibrated for a SIDE GRASP on a 50mm-diameter cylinder
    (radius=0.025m, height=0.10m, cylinder axis = world Z).

    Coordinate frame convention assumed for right_hand_palm_link
    -------------------------------------------------------------
    +X : from wrist toward fingertips ("forward" for the hand)
    +Y : from index/middle side toward thumb side (opposition axis)
    +Z : from back of hand toward palm surface (palm face normal)

    Verify this with the sim viewer using visualize_grasps=True in the
    Dex3GraspSamplingEventCfg before committing a full recording run.

    Antipodal geometry for 3-finger Dex3
    -------------------------------------
    The Dex3 has an ASYMMETRIC opposition: thumb (1 finger) vs index+middle
    (2 fingers side-by-side).

    maximum_aperture is the distance from the THUMB TIP to the MIDPOINT
    between index and middle tips, not from one single finger to another.
    Set to 0.09 m (90 mm) — generous enough to span the 50mm cylinder
    diameter plus some approach clearance.

    finger_clearance = 0.025 m accounts for the axial spread of index and
    middle fingers (~15mm apart along the cylinder Z axis when grasping).
    Together with the standoff range, this ensures candidates at the very
    top and bottom of the cylinder (where one finger would hang off the end)
    are under-sampled and naturally screened out by the success check.

    Side-approach vs top-approach
    ------------------------------
    gripper_approach_direction = [1, 0, 0]   ← palm approaches fingertip-first
    grasp_align_axis           = [0, 1, 0]   ← Y = thumb-to-index opposition
    orientation_sample_axis    = [0, 0, 1]   ← rotate around cylinder Z axis

    After R_align maps the hand's Y axis to the horizontal cylinder-grasp axis,
    the standoff displacement ([1,0,0] in hand frame) becomes horizontal in
    world frame — the palm approaches from the SIDE, not from above.

    The 16 orientation samples (linspace -π to π around Z) produce grasps
    where the hand approaches from 16 different azimuthal directions around
    the cylinder, covering full 360° side coverage.
    """
    metadata = {
        # Max span from thumb tip to index/middle midpoint (metres)
        "maximum_aperture": 0.09,

        # Distance from right_hand_palm_link origin to the proximal contact
        # plane of the fingers (metres).  This sets the CLOSE end of the
        # standoff range (palm just barely reaching the cylinder surface).
        "finger_offset": 0.04,

        # Extra clearance added to the FAR end of the standoff range.
        # Larger than Robotiq (0.015) to account for the axial spread of
        # index and middle fingers along the cylinder height.
        "finger_clearance": 0.025,

        # Direction the palm moves INTO the object (in palm local frame).
        # [1, 0, 0] = fingertips lead the approach (X-forward convention).
        # After R_align rotates the palm so Y aligns with the grasp axis,
        # this X direction becomes HORIZONTAL → side approach. ✓
        "gripper_approach_direction": [1.0, 0.0, 0.0],

        # Axis in the palm frame that gets aligned with the antipodal grasp
        # axis (the vector from thumb contact to index/middle midpoint).
        # [0, 1, 0] = Y = the thumb-to-index/middle opposition direction.
        "grasp_align_axis": [0.0, 1.0, 0.0],

        # Axis around which the 16 orientation candidates are sampled.
        # [0, 0, 1] = world Z = cylinder axis.
        # Rotating around Z gives grasps approaching from all horizontal
        # directions around the cylinder (full 360° azimuthal coverage).
        "orientation_sample_axis": [0.0, 0.0, 1.0],

        # Joint angle for the "open" hand state used by grasp_sampling_event
        # to reset fingers before each candidate evaluation.
        # All Dex3 joints at 0.0 = fully open hand.
        "finger_open_joint_angle": 0.0,
    }

    with open(dst_metadata, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    print(f"metadata.yaml saved: {dst_metadata}")
    print()
    print("IMPORTANT: verify these values with visualize_grasps=True before a full run.")
    print("  The coordinate-frame assumptions above are based on typical humanoid")
    print("  hand conventions.  If the palm_link frame differs, the grasps will")
    print("  look geometrically wrong in the viewer.")


if __name__ == "__main__":
    print("=" * 60)
    print("Dex3 right-hand USD extractor")
    print("=" * 60)
    print(f"Source : {SRC_USD}")
    print(f"Output : {DST_USD}")
    print()

    extract(SRC_USD, DST_USD)
    write_metadata(DST_METADATA)

    print()
    print("Done.  Next steps:")
    print("  1. Set PROJECT_ROOT in your environment.")
    print("  2. Run the grasp sampling env with visualize_grasps=True to verify")
    print("     that candidate poses look geometrically correct on the cylinder.")
    print("  3. If the approach direction is wrong, edit metadata.yaml and re-run.")
    print("  4. Once verified, run record_grasps_g1.py for the full 500-grasp run.")
