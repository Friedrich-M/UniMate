"""Rule-based facing-direction joint selection → ``face_joint_names.json``.

For each rig this script identifies one symmetric joint pair that defines
the lateral (left-right) axis used to compute the facing direction:

  1. Search clean joint names for any symmetrical "Right <X>" / "Left <X>"
     pair, preferring anatomically meaningful joints (Thigh, Shoulder, ...).
  2. For body-axis animals (snakes like Anaconda) that lack bilateral
     symmetry, use longitudinal body-axis endpoints (tail-tip / head-tip)
     and flag ``"body_axis": true``. The tail-to-head vector is directly
     the facing direction (no cross product with Y-up needed).
  3. Animals with opaque joint names where no meaningful pair can be
     extracted get ``"source": "empty"``.

:mod:`face_select_llm` is the LLM-based complement and uses this
module both as a hint and as its fallback.

Usage:
    python -m data_process.joint_annotation.face_select_rule \
        --input_dir <export_dir> [--output PATH]
"""

import argparse
import copy
import json
import os
import re
from typing import Dict, List, Optional, Tuple

# Clean-name keywords for body-axis endpoint detection (serpentine skeletons
# and any rig with no bilateral symmetry but a clear longitudinal axis).
TAIL_KEYWORDS = {"Tail", "Tail Twist"}
HEAD_KEYWORDS = {"Head", "Skull", "Skull Base", "Head End", "Jaw",
                 "Upper Jaw", "Lower Jaw", "Tongue", "Muzzle", "Nose", "Chin"}

# Priority order for symmetrical pair search (suffix after stripping the
# "Right "/"Left " prefix). Higher entries win.
#
# Ordering rationale (top → bottom):
#   1. Hip-level limbs closest to the rig root give the most stable facing
#      axis (Thigh, Shoulder, and the quadruped "Front Shoulder"/"Back Hip"
#      markers Objaverse quadrupeds use after names_clean_rule.py).
#   2. Whole-limb roots (Upper Arm, Front Leg, Hind Leg, Wing, ...) come
#      next — they're unique per side and rarely ambiguous.
#   3. Aquatic / arthropod homologues (fins, pincers, mandibles) appear
#      where they're anatomically equivalent to the limbs above.
#   4. Mid-limb joints (Forearm, Shin, Knee, Elbow, ...) are good fallbacks
#      but drift a few cm from the true hip axis.
#   5. Extremities (Hand, Foot, Paw, Hoof, Claw) are weakest — they bend a
#      lot during motion — so they sit near the bottom.
#   6. Head-adjacent bilateral parts (Eye, Ear, Whisker, Antenna) are last
#      resorts; they give a cross-head axis, not a hip axis.
#   7. Tail is the absolute bottom (often a chain; the 'body_axis' fallback
#      uses it as a longitudinal endpoint instead).
SYMMETRIC_PAIR_PRIORITY = [
    # --- Hip / shoulder level (best) -----------------------------------
    "Thigh",
    "Shoulder",
    "Front Shoulder",  # quadruped pec (Objaverse)
    "Back Hip",        # quadruped pelvis (Objaverse)
    "Hip",
    "Scapula",
    # --- Whole-limb roots ----------------------------------------------
    "Upper Arm",
    "Arm",
    "Front Leg",
    "Hind Leg",
    "Middle Leg",      # insect / arachnid
    "Back Leg",
    "Wing",
    "Leg",
    # --- Aquatic / arthropod homologues --------------------------------
    "Pectoral Fin",
    "Pelvic Fin",
    "Fin",             # generic fish fin (L_Fin1-style rigs)
    "Gill",
    "Pincer",
    "Mandible",
    "Large Mandible",
    "Lower Mandible",
    "Stinger",
    "Claw",
    "Hand Claw",
    "Antenna",
    # --- Mid-limb joints -----------------------------------------------
    "Forearm",
    "Shin",
    "Knee",
    "Elbow",
    "Ankle",
    "Wrist",
    # --- Extremities ---------------------------------------------------
    "Hand",
    "Palm",
    "Foot",
    "Heel",
    "Front Paw",
    "Back Paw",
    "Paw",
    "Front Hoof",
    "Rear Hoof",
    "Hoof",
    "Fetlock",
    "Cannon",
    "Metacarpus",
    "Pastern",
    "Toe",
    # --- Fingers (hand-to-hand axis; both hands move, so below feet) ---
    "Thumb Finger",
    "Index Finger",
    "Middle Finger",
    "Ring Finger",
    "Pinky Finger",
    "Finger",
    # --- Coat / soft ---------------------------------------------------
    "Neck",
    # --- Head-adjacent bilateral parts (weak) --------------------------
    "Eye",
    "Eyeball",
    "Eyelid",
    "Eyebrow",
    "Ear",
    "Horn",
    "Cheek",
    "Whisker",
    "Fang",
    "Barbel",
    "Tentacle",
    "Feather",
    # --- Last resort ---------------------------------------------------
    "Tail",
]


def strip_trailing_num(name: str) -> str:
    """'Thigh 1' -> 'Thigh'."""
    return re.sub(r"\s+\d+$", "", name)


def _digit_runs(raw_name: str) -> Tuple[str, ...]:
    """All digit groups in a raw name: 'Bip01_R_Thigh_4' -> ('01', '4')."""
    return tuple(re.findall(r"\d+", raw_name))


def _match_pair_by_suffix(
    r_indices: List[int],
    l_indices: List[int],
    raw_names: List[str],
) -> Tuple[int, int]:
    """Pair one Right index with one Left index by matching raw-name digits.

    When multiple joints share the same clean name (e.g. multiple "Right
    Thigh" in Scorpion), pairs are matched by the digit signature of the raw
    name so that 'Bip01_R_Thigh_4' pairs with 'Bip01_L_Thigh_4', not
    'Bip01_L_Thigh_1'. Two tiers:

      1. Full digit signature — rigs whose mirrored bones share every index.
      2. Signature minus the LAST group — the Objaverse export stamps a
         per-joint global index on every name ('Bip01_R_Thigh_1_053' vs
         'Bip01_L_Thigh_1_054'), so the full signatures never match; dropping
         the final group compares only the chain indices.

    Falls back to the first of each side when neither tier matches.
    """
    if len(r_indices) == 1 and len(l_indices) == 1:
        return r_indices[0], l_indices[0]

    for key_fn in (lambda runs: runs, lambda runs: runs[:-1]):
        l_by_sig: Dict[Tuple[str, ...], int] = {}
        for li in l_indices:
            l_by_sig.setdefault(key_fn(_digit_runs(raw_names[li])), li)
        for ri in r_indices:
            li = l_by_sig.get(key_fn(_digit_runs(raw_names[ri])))
            if li is not None:
                return ri, li

    return r_indices[0], l_indices[0]


def find_symmetric_pair(
    clean_names: List[str],
    raw_names: List[str],
) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Find the highest-priority "Right <X>" / "Left <X>" pair.

    Returns ``(r_idx, l_idx, suffix)`` or ``(None, None, None)``.
    """
    right_map: Dict[str, List[int]] = {}
    left_map: Dict[str, List[int]] = {}
    for i, name in enumerate(clean_names):
        if name.startswith("Right "):
            right_map.setdefault(strip_trailing_num(name[6:]), []).append(i)
        elif name.startswith("Left "):
            left_map.setdefault(strip_trailing_num(name[5:]), []).append(i)

    common = set(right_map) & set(left_map)
    if not common:
        return None, None, None

    ordered = [s for s in SYMMETRIC_PAIR_PRIORITY if s in common]
    ordered += sorted(common - set(SYMMETRIC_PAIR_PRIORITY))
    suffix = ordered[0]
    ri, li = _match_pair_by_suffix(right_map[suffix], left_map[suffix], raw_names)
    return ri, li, suffix


def find_body_axis_pair(
    clean_names: List[str],
) -> Tuple[Optional[int], Optional[int]]:
    """Find body-axis endpoints (tail-tip and head-end) from clean names.

    For serpentine animals without bilateral symmetry, facing direction is
    the longitudinal axis from tail-tip to head-end. Uses the last occurrence
    of each keyword set to land on the extremities.
    """
    tail_idx: Optional[int] = None
    head_idx: Optional[int] = None
    for i, name in enumerate(clean_names):
        base = strip_trailing_num(name)
        if base in TAIL_KEYWORDS:
            tail_idx = i
        if base in HEAD_KEYWORDS:
            head_idx = i
    if tail_idx is not None and head_idx is not None:
        return tail_idx, head_idx
    return None, None


# Sentinel for "no facing pair could be determined". Always hand out a
# ``copy.deepcopy`` of it — a shallow ``dict()`` copy shares the r_hip/l_hip
# sub-dicts with this module-level constant, so one in-place mutation of a
# returned entry would poison every other 'empty' entry.
EMPTY_ENTRY = {
    "r_hip": {"raw": "", "clean": ""},
    "l_hip": {"raw": "", "clean": ""},
    "source": "empty",
}


def empty_entry() -> Dict:
    """A fresh, independent copy of :data:`EMPTY_ENTRY`."""
    return copy.deepcopy(EMPTY_ENTRY)


def _make_entry(
    raw_names: List[str],
    clean_names: List[str],
    r_idx: int,
    l_idx: int,
    source: str,
) -> Dict:
    return {
        "r_hip": {"raw": raw_names[r_idx], "clean": clean_names[r_idx]},
        "l_hip": {"raw": raw_names[l_idx], "clean": clean_names[l_idx]},
        "source": source,
    }


def resolve_face_joints(
    clean_names: List[str],
    raw_names: List[str],
) -> Dict:
    """Resolve one joint pair defining the facing direction."""
    r_idx, l_idx, suffix = find_symmetric_pair(clean_names, raw_names)
    if r_idx is not None:
        return _make_entry(raw_names, clean_names, r_idx, l_idx, suffix.lower())

    # r_hip = head-end (facing toward), l_hip = tail-tip (facing away from).
    tail_idx, head_idx = find_body_axis_pair(clean_names)
    if tail_idx is not None:
        entry = _make_entry(raw_names, clean_names, head_idx, tail_idx, "body_axis")
        entry["body_axis"] = True
        return entry

    return empty_entry()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_dir", type=str, nargs="+", required=True,
        help="Directory(ies) containing joint_names.json and clean_joint_names.json",
    )
    parser.add_argument(
        "--raw_name", type=str, default="joint_names.json",
        help="Filename of raw joint names inside each --input_dir",
    )
    parser.add_argument(
        "--clean_name", type=str, default="clean_joint_names.json",
        help="Filename of cleaned joint names inside each --input_dir",
    )
    parser.add_argument(
        "--output_name", type=str, default="face_joint_names.json",
        help="Output filename written inside each --input_dir (ignored if --output is set)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Explicit output path (default: <input_dir>/<output_name> per directory)",
    )
    args = parser.parse_args()

    # A single --output cannot hold the results of several --input_dir: only
    # the last directory would survive. Make that an error instead.
    if args.output and len(args.input_dir) > 1:
        parser.error(
            "--output takes a single path but {} --input_dir were given; drop "
            "--output (use --output_name, written inside each directory) or "
            "pass one directory at a time.".format(len(args.input_dir)))

    for input_dir in args.input_dir:
        raw_path = os.path.join(input_dir, args.raw_name)
        clean_path = os.path.join(input_dir, args.clean_name)
        with open(raw_path) as f:
            raw_data = json.load(f)
        with open(clean_path) as f:
            clean_data = json.load(f)

        # Drive the loop off the RAW key set: it is the authority on which
        # object types exist, and stage 4 raises when one has no face entry.
        extra = sorted(set(clean_data) - set(raw_data))
        if extra:
            print("WARNING: {} keys in {} are absent from {} and are ignored: "
                  "{}".format(len(extra), clean_path, raw_path, extra[:5]))

        result = {}
        for animal in raw_data:
            raw_names = raw_data[animal]
            if animal not in clean_data:
                raise KeyError(
                    "{!r} is in {} but missing from {}; re-run the joint-name "
                    "cleaner so every object type has cleaned names."
                    .format(animal, raw_path, clean_path))
            clean_names = clean_data[animal]
            # clean_joint_names is indexed POSITIONALLY against raw_names, so a
            # stale/short list would silently emit an r_hip.raw taken from the
            # wrong position — a real bone name that passes stage 4's
            # membership check and yields a wrong facing axis.
            if len(clean_names) != len(raw_names):
                raise ValueError(
                    "{!r}: {} has {} cleaned names but {} has {} raw names; "
                    "{} is stale — re-run the joint-name cleaner."
                    .format(animal, clean_path, len(clean_names),
                            raw_path, len(raw_names), clean_path))
            result[animal] = resolve_face_joints(clean_names, raw_names)

        output_path = args.output or os.path.join(input_dir, args.output_name)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        sources: Dict[str, List[str]] = {}
        for animal, entry in result.items():
            sources.setdefault(entry["source"], []).append(animal)

        print("Written {} entries to {}\n".format(len(result), output_path))
        for src, animals in sorted(sources.items()):
            print("  [{}] {} entries".format(src, len(animals)))
            for a in animals:
                e = result[a]
                r_clean = e["r_hip"]["clean"] or "(none)"
                l_clean = e["l_hip"]["clean"] or "(none)"
                r_raw = e["r_hip"]["raw"] or "(none)"
                l_raw = e["l_hip"]["raw"] or "(none)"
                body = " [BODY_AXIS]" if e.get("body_axis") else ""
                print("    {:20s}  r={:25s} ({}){}".format(a, r_clean, r_raw, body))
                print("    {:20s}  l={:25s} ({})".format("", l_clean, l_raw))


if __name__ == "__main__":
    main()
