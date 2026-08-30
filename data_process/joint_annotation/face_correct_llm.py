"""Re-run LLM face-joint selection on rigs whose ``source == "empty"``.

:mod:`face_select_llm` writes the empty sentinel
``{r_hip: {raw:"", clean:""}, l_hip: {raw:"", clean:""}, source: "empty"}``
whenever neither the rule-based hint nor the LLM finds a bilateral pair or
a head/tail body axis. This script re-attempts those rigs only (by default
every ``source == "empty"`` entry in ``--face``; optionally a subset listed
in ``--failed_list``) and, when the LLM now finds a valid pair, overwrites
the entry in place. Rigs that still fail keep their existing empty entry.

A ``.bak`` sibling of ``--face`` is created on first run. Rigs that remain
empty are written to ``still_empty_face_joints.txt``.

Usage:
    python -m data_process.joint_annotation.face_correct_llm \\
        --raw     <export_dir>/joint_names.json \\
        --cleaned <export_dir>/clean_joint_names.json \\
        --face    <export_dir>/face_joint_names.json [--model gpt-5]
"""

import argparse
import json
import os
import shutil
import sys

from loguru import logger
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data_process.joint_annotation.face_select_llm import (  # noqa: E402
    SYSTEM_PROMPT,
    resolve_rig,
)
from data_process.joint_annotation.llm import (  # noqa: E402
    LLMClient,
    MAX_RETRIES,
    add_llm_args,
    append_id_list,
    read_id_list,
    save_json,
)


# ─────────────────────────────────────────────────────────────────────────────
# Entry classification
# ─────────────────────────────────────────────────────────────────────────────

def _hips(entry):
    if not isinstance(entry, dict):
        return None, None
    r, l = entry.get("r_hip", {}), entry.get("l_hip", {})
    if not isinstance(r, dict) or not isinstance(l, dict):
        return None, None
    return r, l


def is_non_empty_entry(entry):
    # type: (object) -> bool
    """True iff ``entry`` has a real (non-blank) r_hip/l_hip pair."""
    r, l = _hips(entry)
    return bool(r and l and r.get("raw") and l.get("raw"))


def is_empty_entry(entry):
    # type: (object) -> bool
    """True for the canonical 'source==empty, both hips blank' sentinel
    (and for malformed entries, which are treated as empty)."""
    if not isinstance(entry, dict):
        return True
    if entry.get("source") != "empty":
        return False
    r, l = _hips(entry)
    if r is None:
        return True
    return not r.get("raw") and not l.get("raw")


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def correct_all(raw_path, clean_path, face_path, client, system_prompt=SYSTEM_PROMPT,
                failed_list_path=None, max_tokens=512, max_retries=MAX_RETRIES,
                timeout_seconds=15, still_empty_path=None, only_if_currently_empty=True):
    # type: (...) -> Dict[str, Dict]
    """Re-run LLM face-joint selection and write results back IN-PLACE to
    ``face_path``.

    Targets every ``source == "empty"`` rig in ``face_path``, or only the
    ids in ``failed_list_path`` when given. Rigs where the LLM fails or
    still returns an empty entry are left untouched.

    A missing ``failed_list_path`` is a hard error: a typo must not silently
    escalate a targeted repair into a full sweep.
    """
    with open(raw_path, "r") as f:
        raw_data = json.load(f)
    with open(clean_path, "r") as f:
        clean_data = json.load(f)
    with open(face_path, "r") as f:
        face_data = json.load(f)

    if failed_list_path:
        if not os.path.exists(failed_list_path):
            raise FileNotFoundError(
                "--failed_list {!r} does not exist. Drop the flag to re-attempt "
                "every source=='empty' rig in {}.".format(failed_list_path, face_path))
        target_ids = read_id_list(failed_list_path)
        scope_desc = "{} rigs from {}".format(len(target_ids), failed_list_path)
    else:
        target_ids = sorted(k for k, v in face_data.items() if is_empty_entry(v))
        scope_desc = "{} empty-source rigs in {}".format(len(target_ids), face_path)

    backup = face_path + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(face_path, backup)
        logger.info("Backed up {} -> {}".format(face_path, backup))

    if still_empty_path is None:
        still_empty_path = os.path.join(
            os.path.dirname(face_path) or ".", "still_empty_face_joints.txt")

    logger.info("Re-attempting {} (in-place update of {})".format(scope_desc, face_path))

    updated = skipped_not_empty = missing = 0
    still_empty = []  # type: List[str]
    pbar = tqdm(target_ids, desc="Correcting", unit="rig", dynamic_ncols=True)
    for rig_id in pbar:
        pbar.set_postfix_str(rig_id, refresh=False)

        if rig_id not in raw_data or rig_id not in clean_data:
            missing += 1
            logger.warning("[{}] missing from raw/clean; skipping".format(rig_id))
            continue

        # The failed list may be stale: only touch entries still empty now.
        if only_if_currently_empty and not is_empty_entry(face_data.get(rig_id)):
            skipped_not_empty += 1
            continue

        try:
            entry, used_fallback = resolve_rig(
                rig_id, raw_data[rig_id], clean_data[rig_id], client,
                system_prompt=system_prompt, max_tokens=max_tokens,
                max_retries=max_retries, timeout_seconds=timeout_seconds,
                fallback_on_error=False,
            )
        except Exception as err:  # noqa: BLE001 — exhausted retries: keep the empty entry
            logger.warning("[{}] LLM resolve failed: {}".format(rig_id, err))
            entry, used_fallback = None, True

        # Accept only clean non-empty LLM wins. used_fallback=True means the
        # rule-based hint came back on timeout — the same hint that already
        # produced the empty entry.
        if entry is not None and not used_fallback and is_non_empty_entry(entry):
            face_data[rig_id] = entry
            updated += 1
        else:
            still_empty.append(rig_id)

        save_json(face_data, face_path)

    # Append (like every other id log in this stage) rather than truncate: a
    # --failed_list run only sees a subset, and a truncating write would
    # replace the global record with just that subset.
    known_empty = (set(read_id_list(still_empty_path))
                   if os.path.exists(still_empty_path) else set())
    if still_empty:
        new_ids = [rig_id for rig_id in still_empty if rig_id not in known_empty]
        if new_ids:
            append_id_list(still_empty_path, new_ids)
        logger.info("{} rigs still empty ({} newly recorded) -> {}".format(
            len(still_empty), len(new_ids), still_empty_path))
    elif known_empty:
        logger.warning(
            "No rig came back empty in this run, but {} still lists {} id(s) "
            "from earlier runs — it may be stale.".format(
                still_empty_path, len(known_empty)))

    logger.info(
        "Done. updated={} | still_empty={} | skipped_not_empty={} | "
        "missing_raw_or_clean={} | total={}".format(
            updated, len(still_empty), skipped_not_empty, missing, len(target_ids)))
    return face_data


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Re-run LLM face-joint selection on rigs whose source is 'empty'; "
                    "updates face_joint_names.json in place when a pair is found.")
    parser.add_argument("--raw", required=True, help="Raw joint_names.json.")
    parser.add_argument("--cleaned", required=True, help="clean_joint_names.json.")
    parser.add_argument("--face", required=True,
                        help="face_joint_names.json — updated IN PLACE; a '.bak' sibling "
                             "is made on first run.")
    parser.add_argument("--failed_list", default=None,
                        help="Optional txt file with one rig id per line. Default: every "
                             "rig whose entry in --face has source=='empty'.")
    parser.add_argument("--still_empty_log", default=None,
                        help="Path to write rig ids the LLM still couldn't fill (default: "
                             "sibling 'still_empty_face_joints.txt').")
    parser.add_argument("--force_reattempt", action="store_true",
                        help="Re-run even if the current entry isn't source=='empty' anymore.")
    parser.add_argument("--rig_timeout", type=int, default=15,
                        help="Per-rig wall-time budget in seconds (SIGALRM). 0 disables.")
    add_llm_args(parser, default_max_tokens=512)
    args = parser.parse_args()

    client = LLMClient.from_args(args)
    logger.info("{} | Writing in-place to: {}".format(client, args.face))

    correct_all(
        raw_path=args.raw,
        clean_path=args.cleaned,
        face_path=args.face,
        client=client,
        failed_list_path=args.failed_list,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        timeout_seconds=args.rig_timeout,
        still_empty_path=args.still_empty_log,
        only_if_currently_empty=not args.force_reattempt,
    )


if __name__ == "__main__":
    main()
