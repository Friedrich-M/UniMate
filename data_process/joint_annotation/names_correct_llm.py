"""LLM check-and-correct pass over ``clean_joint_names.json``.

Feeds the LLM BOTH the raw joint name and the current cleaned label for
every joint, and asks it to either keep the current label or replace it with
a corrected canonical one. Useful for:

  - correcting rigs that fell back to the rule-based cleaner (digits,
    missing side prefixes, ``'Bone'`` for tokens the rules don't recognise),
  - sweeping the whole dataset for residual vocabulary / rule violations.

Inputs:
    --raw           raw ``joint_names.json``.
    --cleaned       current ``clean_joint_names.json``.
    --failed_list   OPTIONAL text file with one rig id per line (e.g. the
                    ``failed_clean_names.txt`` written by ``names_clean_llm``). If
                    omitted, EVERY rig in ``--raw`` is checked and corrected.

Output: the script WRITES BACK IN-PLACE to ``--cleaned``. A ``.bak`` sibling
is created on first run so the original is always recoverable.

Usage:
    python -m data_process.joint_annotation.names_correct_llm \\
        --raw <export_dir>/joint_names.json \\
        --cleaned <export_dir>/clean_joint_names.json [--failed_list failed_clean_names.txt]
"""

import argparse
import json
import os
import re
import shutil
import sys
import time

from loguru import logger
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data_process.joint_annotation.names_clean_llm import (  # noqa: E402
    SYSTEM_PROMPT,
    parse_response,
    rule_based_clean,
)
from data_process.joint_annotation.vocab import is_canonical_label  # noqa: E402
from data_process.joint_annotation.llm import (  # noqa: E402
    FatalLLMError,
    LLMClient,
    MAX_RETRIES,
    RigTimeout,
    add_llm_args,
    append_id_list,
    install_rig_alarm,
    read_id_list,
    save_json,
    uninstall_rig_alarm,
)


# ─────────────────────────────────────────────────────────────────────────────
# Correction-mode system prompt addendum
# ─────────────────────────────────────────────────────────────────────────────

CORRECTION_PREAMBLE = (
    "=========================================================================\n"
    "CHECK & CORRECT MODE — INPUT SHAPE OVERRIDES THE ABOVE; OUTPUT SHAPE "
    "IS UNCHANGED.\n"
    "=========================================================================\n"
    "INPUT change: instead of a plain JSON array of raw names you receive a "
    "JSON array of objects, one per joint:\n"
    "    {\"raw\": \"<raw joint name>\", \"current\": \"<current cleaned label>\"}\n"
    "The ``current`` label is whatever is already stored in "
    "clean_joint_names.json — it may be hand-curated, LLM-cleaned, or a "
    "rule-based fallback. Treat it as a candidate, not ground truth.\n\n"
    "PROCESS EACH JOINT IN TWO STEPS.\n\n"
    "STEP 1 — VALIDATE the current label against ALL cleaning rules above. "
    "The label is VALID only if every check below passes:\n"
    "  (a) Its BASE PART appears in the CANONICAL VOCABULARY (exact term, "
    "Title Case, single spaces). The base part is the label minus any "
    "'Left '/'Right ' side prefix, minus any Front/Back/Middle/Rear/Inner/"
    "Outer/Upper/Lower qualifier, minus a trailing ' End' — composed "
    "labels like 'Left Thigh', 'Right Front Shoulder', 'Leg End', "
    "'Inner Toe' are valid. EXCEPTION: a raw name that is ONLY digits "
    "(optionally with a leading underscore) keeps its numeric passthrough "
    "label unchanged ('_00' -> '_00') — rule 8.\n"
    "  (b) It contains NO digits, underscores, dots, or Blender '.NNN' "
    "counters — rule 1 — except the rule-8 numeric passthrough in (a).\n"
    "  (c) It carries NO leftover namespace (mixamorig:, Mutant:, Sif:, ...) "
    "or rig/Maya prefix/suffix (Bip01_, BN_, NPC_, jt_, DEF-, _jnt, _C, "
    "...) — rules 2-4.\n"
    "  (d) Sides agree with the raw name — rule 5. If the raw encodes a "
    "side (L_/R_, Lt_/Rt_, Left<Word>, 'Bip001 L/R ', trailing L/R on "
    "Japanese roots, F_/B_ for quadrupeds), the label starts with "
    "'Left '/'Right ' (with the Front/Back marker between side and part "
    "when applicable); if the raw encodes NO side, the label must NOT "
    "start with one. A trailing .L/.R/_L/_R token OVERRIDES a stale "
    "leading side word ('mixamorig:LeftShoulder.R' -> 'Right Shoulder').\n"
    "  (e) If the raw name clearly encodes a recognisable body part, the "
    "label is NOT the 'Bone' placeholder — rules 6-9 ('Appendage' is "
    "correct only for humanoid accessory chains, rule 7).\n"
    "  (f) Convention-dependent names are resolved correctly: finger roots "
    "and mixamo chain bones (UpLeg/Leg/Foot -> Thigh/Shin/Foot) — rule 6; "
    "3ds Max Biped 0-based finger codes (Finger0*=Thumb, 1*=Index, ...) — "
    "rule 6b; 1-based mocap segmented fingers (Finger1Metacarpal=Thumb, "
    "..., segment words dropped) — rule 6c; animal 'Hair*/Mane*' -> 'Mane' "
    "but humanoid accessories 'Ponytail*/Cape*/Cloth*/Skirt*' -> "
    "'Appendage' — rule 7; Japanese roots — rule 9.\n\n"
    "STEP 2 — DECIDE:\n"
    "  - If VALID on every check in Step 1: OUTPUT THE CURRENT LABEL "
    "VERBATIM (skip — do not rewrite).\n"
    "  - If INVALID on any check: IGNORE the current label and apply the "
    "cleaning rules above to the raw name from scratch; output the "
    "CORRECTED canonical label.\n\n"
    "OUTPUT unchanged (identical to the clean-mode contract): a single FLAT "
    "JSON array of STRINGS — one label per joint, same order, same length "
    "as the input array. Critical constraints:\n"
    "  - Output is an array of strings, NOT objects. Do NOT echo 'raw' or "
    "'current' keys. Do NOT wrap each entry in an object.\n"
    "  - Same length as input (number of objects in). One input object -> "
    "one output string.\n"
    "  - No prose, no markdown fences, no comments, no trailing text.\n"
    "  - Never dedupe, merge, split, reorder or skip — identical neighbouring "
    "outputs (e.g. 'Spine', 'Spine', 'Spine') are CORRECT.\n"
    "  - Title Case with single spaces. Canonical vocabulary terms only.\n"
    "  - Side first ('Left Upper Arm'); quadruped marker between side and "
    "part ('Right Front Shoulder').\n\n"
    "CHECK/CORRECT EXAMPLES (current -> final, with verdict):\n"
    "  raw='Spine',                 current='Spine'            -> 'Spine'             (VALID, skip)\n"
    "  raw='mixamorig:LeftUpLeg',   current='Left Thigh'       -> 'Left Thigh'        (VALID, skip)\n"
    "  raw='ToeBase.L',             current='Left Toe'         -> 'Left Toe'          (VALID, skip)\n"
    "  raw='mixamorig:LeftLeg',     current='Left Leg'         -> 'Left Shin'         (INVALID: mixamo mid-bone)\n"
    "  raw='mixamorig:Spine_02',    current='Spine 02'         -> 'Spine'             (INVALID: digits)\n"
    "  raw='Bip01_L_Thigh_07',      current='Bip01 L Thigh'    -> 'Left Thigh'        (INVALID: prefix + digits)\n"
    "  raw='QuickRigCharacter_LeftForeArm_014',\n"
    "       current='Bone'                                      -> 'Left Forearm'      (INVALID: lost anatomy)\n"
    "  raw='Bone.001_01',           current='Bone 001 01'      -> 'Bone'              (INVALID: digits)\n"
    "  raw='munabireL',             current='MunabireL'        -> 'Left Pectoral Fin' (INVALID: non-canonical)\n"
    "  raw='Hair_03',               current='Hair'             -> 'Mane'              (INVALID: animal Hair->Mane)\n"
    "  raw='Reins_02',              current='Reins'            -> 'Reins'             (VALID: equipment label)\n"
    "  raw='LegTip_R',              current='Right Leg Tip'    -> 'Right Leg End'     (INVALID: Tip not canonical)\n"
    "  raw='_00',                   current='_00'              -> '_00'               (VALID: numeric passthrough)\n"
    "  raw='mixamorig:LeftShoulder.R',\n"
    "       current='Left Shoulder'                            -> 'Right Shoulder'    (INVALID: .R overrides stale prefix)\n"
    "  raw='Thigh_03',              current='Left Thigh'       -> 'Thigh'             (INVALID: raw encodes no side)\n"
)

CORRECTION_SYSTEM_PROMPT = SYSTEM_PROMPT + "\n\n" + CORRECTION_PREAMBLE


def build_correction_prompt(rig_id, raw_names, current_labels, correction=None,
                            chunk_start=0, rig_raw_names=None):
    # type: (str, List[str], List[str], Optional[str], int, Optional[List[str]]) -> str
    """Build the user prompt for one chunk of a rig.

    When ``rig_raw_names`` (the FULL rig's raw list) is longer than this
    chunk, the prompt says so and includes it as read-only context —
    chain-dependent decisions (mixamo mid-bone 'Leg', 6b/6c finger
    conventions) need the neighbours a chunk boundary would otherwise
    hide, and 'Joint count: 16' on an 80-joint rig would mislead.
    """
    pairs = [{"raw": raw, "current": current}
             for raw, current in zip(raw_names, current_labels)]
    header = ["Rig identifier: {}".format(rig_id)]
    if rig_raw_names is not None and len(rig_raw_names) > len(raw_names):
        header.append("Rig joint count: {}".format(len(rig_raw_names)))
        header.append(
            "This request covers joints {}..{} (0-based) — a consecutive "
            "slice of the rig, not the whole rig.".format(
                chunk_start, chunk_start + len(raw_names) - 1))
        header.append(
            "Full rig raw joint list (READ-ONLY context for chain/side "
            "decisions; do NOT emit labels for it):\n{}".format(
                json.dumps(rig_raw_names, ensure_ascii=False)))
    else:
        header.append("Joint count: {}".format(len(raw_names)))
    prompt = (
        "{header}\n\n"
        "Joints (JSON array of {{raw, current}} objects, order matters):\n"
        "{pairs}\n\n"
        "For each joint: validate the 'current' label against the cleaning "
        "rules. If VALID, output it verbatim; if INVALID, output a corrected "
        "canonical label. Return a FLAT JSON array of EXACTLY {n} STRINGS, "
        "same order as the input. Do NOT output objects; strings only."
    ).format(header="\n".join(header), n=len(raw_names),
             pairs=json.dumps(pairs, ensure_ascii=False, indent=2))
    if correction:
        prompt += "\n\nIMPORTANT: " + correction
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary check on LLM output (warn-only)
# ─────────────────────────────────────────────────────────────────────────────

_SIDE_RE = re.compile(r'^(Left|Right)\s+')
_QUALIFIER_RE = re.compile(
    r'^(Front|Back|Middle|Rear|Hind|Inner|Outer|Upper|Lower)\s+')


def _is_vocab_label(label):
    # type: (str) -> bool
    """``vocab.is_canonical_label`` extended over composed qualifier forms
    the cleaners emit ('Right Front Shoulder', 'Back Hip') whose bare part
    is not itself a vocabulary value."""
    if is_canonical_label(label):
        return True
    base = _SIDE_RE.sub('', label)
    base = re.sub(r'\s+End$', '', base)
    stripped = _QUALIFIER_RE.sub('', base)
    return stripped != base and is_canonical_label(stripped)


def _warn_non_canonical(rig_id, chunk_tag, chunk_raw, labels):
    # type: (str, str, List[str], List[str]) -> None
    """Warn-only guard: flag LLM outputs outside the canonical vocabulary.

    Deliberately not a hard retry — ``canonical_parts`` under-covers a few
    legitimate composed labels, and a false reject would burn retries on a
    correct answer. The warning makes vocabulary drift visible in the log
    before it lands in clean_joint_names.json."""
    bad = [(raw, lbl) for raw, lbl in zip(chunk_raw, labels)
           if not _is_vocab_label(lbl)]
    if bad:
        shown = ", ".join("{!r}->{!r}".format(r, l) for r, l in bad[:5])
        logger.warning("[{}{}] {} label(s) outside the canonical vocabulary "
                       "(kept as-is): {}{}".format(
                           rig_id, chunk_tag, len(bad), shown,
                           ", ..." if len(bad) > 5 else ""))


# ─────────────────────────────────────────────────────────────────────────────
# Per-chunk + per-rig corrector (with retry + keep-rule-based fallback)
# ─────────────────────────────────────────────────────────────────────────────
#
# Long rigs (50+ joints) reliably break the "emit EXACTLY N entries" contract
# on both Qwen3 and GPT-5 — the LLM drifts by ±1-2 when near-duplicate names
# sit next to each other (Spine/Spine1/Spine2 → "Spine" × 3). Shrinking the
# per-call enumeration fixes it: at chunk_size ≤ 16 the count stays exact.
# The per-rig SIGALRM budget wraps the whole chunk sequence; a timeout keeps
# already-corrected chunks and leaves the rest as rule-based.

def _run_chunk(rig_id, chunk_raw, chunk_current, client, system_prompt,
               max_tokens, max_retries, chunk_tag="", chunk_start=0,
               rig_raw_names=None):
    # type: (...) -> Optional[List[str]]
    """Retry loop for a single chunk. Returns the parsed list on success, or
    ``None`` after exhausting retries. Lets ``RigTimeout`` propagate so the
    caller can stop early and preserve already-finished chunks."""
    correction = None
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            user_prompt = build_correction_prompt(
                rig_id, chunk_raw, chunk_current, correction=correction,
                chunk_start=chunk_start, rig_raw_names=rig_raw_names)
            raw_text = client.generate(system_prompt, user_prompt, max_tokens)
            parsed = parse_response(raw_text, len(chunk_raw))
            _warn_non_canonical(rig_id, chunk_tag, chunk_raw, parsed)
            return parsed
        except (RigTimeout, FatalLLMError):
            raise
        except Exception as err:  # noqa: BLE001 — retry on any backend/parse error
            last_err = err
            if isinstance(err, ValueError):
                correction = (
                    "Your previous response failed validation ({}). Return a "
                    "JSON array of EXACTLY {} strings — one corrected label "
                    "per raw joint, same order. No prose, no code fences."
                ).format(err, len(chunk_raw))
            logger.warning("[{}{}] attempt {}/{} failed: {}".format(
                rig_id, chunk_tag, attempt, max_retries, err))
            if attempt < max_retries:
                time.sleep(1.5)
    logger.error("[{}{}] chunk failed after {} attempts ({})".format(
        rig_id, chunk_tag, max_retries, last_err))
    return None


def correct_rig(rig_id, raw_names, current_labels, client,
                system_prompt=CORRECTION_SYSTEM_PROMPT, max_tokens=4096,
                max_retries=MAX_RETRIES, timeout_seconds=15, chunk_size=16):
    # type: (...) -> Tuple[List[str], bool]
    """Correct a single rig via the LLM. Returns ``(labels, still_failed)``.

    Rigs longer than ``chunk_size`` are split into consecutive chunks; each
    chunk is an independent LLM call with its own retries. ``timeout_seconds``
    is the PER-CHUNK share of the rig's SIGALRM budget (total = it times the
    chunk count — a flat per-rig budget starves long rigs, whose later
    chunks would never run); successfully-corrected chunks stay corrected
    even if a later chunk times out.
    """
    if not raw_names:
        return [], False

    n = len(raw_names)
    result = list(current_labels)  # failed / timed-out chunks keep their labels
    chunk_size = max(1, int(chunk_size))
    n_chunks = (n + chunk_size - 1) // chunk_size
    failed_chunks = 0
    timed_out = False

    budget = timeout_seconds * n_chunks if timeout_seconds else 0
    prev_handler = install_rig_alarm(budget)
    try:
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, n)
            chunk_tag = (" [{}/{} {}..{}]".format(i + 1, n_chunks, start, end - 1)
                         if n_chunks > 1 else "")
            chunk_out = _run_chunk(
                rig_id, raw_names[start:end], current_labels[start:end],
                client, system_prompt, max_tokens, max_retries,
                chunk_tag=chunk_tag, chunk_start=start, rig_raw_names=raw_names)
            if chunk_out is not None:
                result[start:end] = chunk_out
            else:
                failed_chunks += 1
    except RigTimeout:
        timed_out = True
    finally:
        uninstall_rig_alarm(prev_handler)

    if timed_out:
        logger.warning("[{}] exceeded {}s budget; keeping current labels for "
                       "unfinished chunks".format(rig_id, budget))
    elif failed_chunks:
        logger.warning("[{}] {}/{} chunks still failed; keeping current labels "
                       "for those".format(rig_id, failed_chunks, n_chunks))

    return result, timed_out or failed_chunks > 0


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def correct_all(raw_path, cleaned_path, client, system_prompt=CORRECTION_SYSTEM_PROMPT,
                failed_list_path=None, max_tokens=4096, max_retries=MAX_RETRIES,
                timeout_seconds=15, chunk_size=16, still_failed_path=None):
    # type: (...) -> Dict[str, List[str]]
    """Check + correct joint labels in ``cleaned_path`` via the LLM.

    When ``failed_list_path`` is provided, only rigs listed there are
    processed; otherwise every rig in ``raw_path``. The result is written
    BACK IN-PLACE to ``cleaned_path`` (a ``.bak`` sibling is created on first
    run). Rigs the LLM still couldn't fully correct are appended to
    ``still_failed_path``.

    A missing ``failed_list_path`` is a hard error (a typo must not silently
    escalate a targeted repair into a full sweep), and so is a missing
    ``cleaned_path`` in that mode — the run would otherwise replace the whole
    cleaned file with one holding ONLY the listed rigs, and with no ``.bak``.
    """
    with open(raw_path, "r") as f:
        raw_data = json.load(f)

    if failed_list_path:
        if not os.path.exists(failed_list_path):
            raise FileNotFoundError(
                "--failed_list {!r} does not exist. Drop the flag to correct "
                "every rig in {}.".format(failed_list_path, raw_path))
        if not os.path.exists(cleaned_path):
            raise FileNotFoundError(
                "--failed_list restricts this run to a subset of rigs, but "
                "--cleaned {!r} does not exist: the run would write a cleaned "
                "file containing ONLY the listed rigs (and make no .bak). Run "
                "the joint-name cleaner first.".format(cleaned_path))
        target_ids = read_id_list(failed_list_path)
        scope_desc = "{} rigs from {}".format(len(target_ids), failed_list_path)
    else:
        target_ids = sorted(raw_data.keys())
        scope_desc = "all {} rigs in {}".format(len(target_ids), raw_path)

    if os.path.exists(cleaned_path):
        with open(cleaned_path, "r") as f:
            cleaned_data = json.load(f)
    else:
        cleaned_data = {}
        logger.info("No existing {}; starting from rule-based labels.".format(cleaned_path))

    backup = cleaned_path + ".bak"
    if os.path.exists(cleaned_path) and not os.path.exists(backup):
        shutil.copy2(cleaned_path, backup)
        logger.info("Backed up {} -> {}".format(cleaned_path, backup))

    if still_failed_path is None:
        still_failed_path = os.path.join(
            os.path.dirname(cleaned_path) or ".", "still_failed_clean_names.txt")

    logger.info("Correcting {} (in-place) -> {}".format(scope_desc, cleaned_path))

    still_failed = []
    corrected_count = 0
    pbar = tqdm(target_ids, desc="Correcting", unit="rig", dynamic_ncols=True)
    for rig_id in pbar:
        pbar.set_postfix_str(rig_id, refresh=False)
        if rig_id not in raw_data:
            logger.warning("[{}] missing from raw file; skipping".format(rig_id))
            continue
        raw_names = raw_data[rig_id]
        current = cleaned_data.get(rig_id)
        if not isinstance(current, list) or len(current) != len(raw_names):
            current = rule_based_clean(rig_id, raw_names)

        new_labels, still_bad = correct_rig(
            rig_id, raw_names, current, client,
            system_prompt=system_prompt, max_tokens=max_tokens,
            max_retries=max_retries, timeout_seconds=timeout_seconds,
            chunk_size=chunk_size,
        )
        cleaned_data[rig_id] = new_labels
        if still_bad:
            still_failed.append(rig_id)
        else:
            corrected_count += 1
        save_json(cleaned_data, cleaned_path)

    if still_failed:
        # Dedup against earlier runs (same convention as face_correct_llm).
        known = (set(read_id_list(still_failed_path))
                 if os.path.exists(still_failed_path) else set())
        new_ids = [rig_id for rig_id in still_failed if rig_id not in known]
        if new_ids:
            append_id_list(still_failed_path, new_ids)
        logger.info("{} rigs still failed ({} newly recorded) -> {}".format(
            len(still_failed), len(new_ids), still_failed_path))

    logger.info("Done. Corrected {}/{} rigs (still failed: {}).".format(
        corrected_count, len(target_ids), len(still_failed)))
    return cleaned_data


def main():
    parser = argparse.ArgumentParser(
        description="LLM check-and-correct pass over clean_joint_names.json (in place).")
    parser.add_argument("--raw", required=True, help="Raw joint_names.json.")
    parser.add_argument("--cleaned", required=True,
                        help="clean_joint_names.json — updated IN PLACE; a '.bak' "
                             "sibling is made on first run.")
    parser.add_argument("--failed_list", default=None,
                        help="Optional txt file with one rig id per line. When omitted, "
                             "ALL rigs in --raw are checked and corrected.")
    parser.add_argument("--still_failed_log", default=None,
                        help="Append rig ids the LLM still couldn't fix (default: "
                             "sibling 'still_failed_clean_names.txt').")
    parser.add_argument("--rig_timeout", type=int, default=15,
                        help="Per-CHUNK wall-time budget in seconds (SIGALRM; a rig's "
                             "total budget is this times its chunk count). 0 disables.")
    parser.add_argument("--chunk_size", type=int, default=16,
                        help="Joints per LLM call. Smaller chunks avoid length drift on "
                             "long arrays; larger chunks save API calls.")
    add_llm_args(parser, default_max_tokens=4096)
    args = parser.parse_args()

    client = LLMClient.from_args(args)
    logger.info("{} | Writing in-place to: {}".format(client, args.cleaned))

    correct_all(
        raw_path=args.raw,
        cleaned_path=args.cleaned,
        client=client,
        failed_list_path=args.failed_list,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        timeout_seconds=args.rig_timeout,
        chunk_size=args.chunk_size,
        still_failed_path=args.still_failed_log,
    )


if __name__ == "__main__":
    main()
