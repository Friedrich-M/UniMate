"""Clean joint names with a Language Model.

Complement to the rule-based :mod:`names_clean_rule`: sends raw joint names to an
LLM and asks it to map each one to a canonical anatomical label from
:mod:`vocab`. Pure text in / text out — no images needed.

Backends (see :mod:`llm`): local HF causal LM, OpenAI API (default:
gpt-5-mini), DeepSeek API.

Input  : ``joint_names.json`` — ``{rig_id: [raw_joint_name, ...]}`` where
         ``rig_id`` keys may be humans, animals, props, vehicles or any other
         skeletal asset (Objaverse uses opaque UUIDs; Truebones uses animal
         names).
Output : ``clean_joint_names.json`` — same shape, with cleaned names.

Each rig is processed as a single batched request: the full joint list is
sent together so the LLM can reason about the rig as a whole, and the
response must be a JSON array of the same length. Rigs that fail after
``--max_retries`` fall back to the rule-based cleaner and are recorded in
``failed_clean_names.txt`` next to the output.

Runs resume: a rig that already has a correct-length entry in the output is
skipped. Because the rule-based stage writes the SAME default file
(``<export_dir>/clean_joint_names.json``), that makes this stage a no-op after
it — and it also skips the very rigs listed in ``failed_clean_names.txt``, since a
fallen-back rig has a full-length entry too. Use ``--overwrite`` to redo every
rig or ``--redo_failed`` to redo just the recorded failures.

Usage:
    python -m data_process.joint_annotation.names_clean_llm \\
        --input <export_dir>/joint_names.json               # OpenAI gpt-5-mini
    python -m data_process.joint_annotation.names_clean_llm \\
        --input <export_dir>/joint_names.json --model deepseek-v4-flash
    python -m data_process.joint_annotation.names_clean_llm \\
        --input <export_dir>/joint_names.json --model Qwen/Qwen3-8B
"""

import argparse
import json
import os
import sys
import time

from loguru import logger
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data_process.joint_annotation.names_clean_rule import (  # noqa: E402
    clean_joint_name,
    post_process,
    report_coverage,
)
from data_process.joint_annotation.llm import (  # noqa: E402
    FatalLLMError,
    LLMClient,
    MAX_RETRIES,
    add_llm_args,
    append_id_list,
    load_json,
    read_id_list,
    save_json,
    strip_llm_noise,
)


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — encodes the vocab.py canonical vocabulary
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Standardize 3D rig joint names to canonical anatomical labels. Inputs "
    "come from Mixamo, Maya, Blender, Unreal, Truebones and custom rigs. "
    "Focus on semantic meaning, not surface syntax.\n\n"
    "OUTPUT: a single JSON array of strings, same length and order as the "
    "input. No prose, no fences, no extra text. One input entry -> one "
    "output entry; never dedupe, merge, skip or reorder, even when "
    "neighbours produce identical labels.\n\n"
    "CLEAN each name by removing rig noise and extracting meaning:\n"
    "1. Digits and Blender '.NNN' counters are meaningless — rig-internal "
    "bookkeeping (chain index, mirror id, duplicate counter). Ignore them "
    "when matching, and never include a digit, underscore, or dot in the "
    "final label. 'Spine', 'Spine1', 'Spine_02', 'Spine.003' all map to "
    "'Spine'. Finger-chain segments ('Thumb1/Thumb2/Thumb3') all map to "
    "'Thumb Finger'. The Objaverse export pipeline also stamps a trailing "
    "global index on EVERY name ('_NN' or '_0NN': '_01', '_010', '_063'); "
    "indices can STACK ('Hip_01_41', 'Head_1_016', 'Index.R.001_013_6', "
    "'Bone.001_01', 'Spine_1_013') — strip ALL of them, in any position. Editor decorations are noise too: strip '(mirrored)' anywhere and a trailing '.x' center marker ('spine_01.x' -> 'Spine').\n"
    "2. Drop any leading '<Word>:' namespace (case-insensitive, trailing "
    "digits in the namespace OK): mixamorig:, Mixamorig1:, Mutant:, Sif:. "
    "The same words are noise without the colon too — leading "
    "'mixamorig_'/'Mixamorig'/'Character'/'Rig'/'QuickRigCharacter_' "
    "segments are dropped, never echoed into the label. The same goes "
    "for embedded ASSET/CHARACTER names and their decorations — "
    "'rp_karl_animated_006_warmingUp_spine_01' -> 'Spine', "
    "'CMan0205-M4-CS_Hips L Finger0' -> 'Left Thumb Finger': keep only "
    "the anatomical tokens, drop every name-like or counter token.\n"
    "3. Drop rig prefixes (match before ignoring digits so 'Bip01_' still "
    "strips cleanly): any Bip<digits> container with any separator (Bip01_, Bip002 , Bip01-, and separator-free Bip001LFinger0), BN_Bip01_, BN_, Bn_, NPC_, jt_, Elk, "
    "Sabrecat_, QuickRigCharacter_, Bind_, Skeleton_, Root_, DEF-, def_.\n"
    "4. Drop Maya suffixes: _jnt, _jt, _Jt, _JNT, _joint, _bone, _bn, _C.\n"
    "5. Extract side as explicit 'Left '/'Right ' prefix. Recognise:\n"
    "   prefix  L_ / R_, Lt_ / Rt_, Left_ / Right_, Left<UpperWord> / "
    "Right<UpperWord> (LeftHand, RightArm), L<UpperLetter> / R<UpperLetter> "
    "(LArm, RHand) — but NOT when followed by a lowercase letter (Lower, "
    "Ribcage).\n"
    "   3ds Max Biped  space-separated single-letter side: 'Bip001 L "
    "UpperArm' -> Left Upper Arm, 'Bip001 R Thigh' -> Right Thigh. Token "
    "boundaries are spaces, not underscores.\n"
    "   suffix  _L / _R / _l / _r, .L / .R, Japanese trailing L/R.\n"
    "   PRECEDENCE: a trailing .L/.R/_L/_R token OVERRIDES a leading side word — Blender's symmetrize renames only the suffix, leaving the prefix text stale: 'mixamorig:LeftShoulder.R' -> 'Right Shoulder', 'r_toe.L' -> 'Left Toe'.\n"
    "   quadruped F_/B_ = Front/Back (e.g. 'F_R_Shoulder' -> "
    "'Right Front Shoulder', 'B_L_Foot' -> 'Left Back Foot').\n"
    "6. Common body-part roots — translate case-insensitively, accepting "
    "Unreal snake_case AND CamelCase short-forms: pelvis/spine/neck/head/jaw/"
    "eye -> Pelvis/Spine/Neck/Head/Jaw/Eye; clavicle/collar -> Shoulder; "
    "upperarm / UpperArm / UpArm -> Upper Arm; lowerarm / LowerArm / LowArm / "
    "ForeArm -> Forearm; hand -> Hand; thigh / upleg / UpLeg -> Thigh; "
    "calf / lowleg / LowLeg / lowerleg -> Shin; upperleg -> Thigh; "
    "toes -> Toe; in a mixamo chain "
    "(UpLeg -> Leg -> Foot) the mid-bone 'Leg' is the shin -> Shin; foot -> "
    "Foot; toebase / Toe0 / toe -> Toe; ball -> Toe; eyelid -> Eyelid. "
    "Finger roots index/middle/ring/pinky/thumb -> '<Root> Finger'. "
    "*_twist -> '<Root> Twist'.\n"
    "6b. 3ds Max Biped fingers use numeric codes: Finger0*=Thumb, Finger1*="
    "Index, Finger2*=Middle, Finger3*=Ring, Finger4*=Pinky. Any trailing "
    "digits after that code are chain position — ignore. 'Bip001 L Finger0' "
    "and 'Bip001 L Finger01' both -> 'Left Thumb Finger'; 'Bip001 R Finger21'"
    " -> 'Right Middle Finger'; 'Bip001 R Toe0' -> 'Right Toe'.\n"
    "6c. Mocap-segmented fingers are 1-BASED and carry a segment word: Finger1..Finger5 + Metacarpal/Proximal/Medial/Distal/Tip, with Finger1=Thumb ... Finger5=Pinky. 'LeftFinger1Metacarpal' -> 'Left Thumb Finger'; the Tip segment -> '<Root> Finger End'. Rule 6b's 0-based codes apply only to BARE Finger<digit> names with no segment word. Segment words after a NAMED finger ('IndexDistal', 'thumb_proximal_l', 'RingIntermediate') are likewise chain position — drop them: 'IndexDistal' -> 'Index Finger'.\n"
    "7. Other direction words: Top->Upper, Low->Lower ('Topjaw'->'Upper Jaw'). "
    "'HeadTop_End' / '*_End' / '*Nub' -> '<Root> End'. Animal 'Hair*/Mane*' -> 'Mane'; humanoid accessories 'Ponytail*/Cape*/Cloth*/Skirt*' -> 'Appendage'.\n"
    "8. Placeholders -> 'Bone': Bone, joint, Xtra*, MagicEffectsNode, "
    "and any token with no clear anatomy (meshok, Capuche, ...). Do NOT "
    "fabricate body parts. EXCEPTION: a name that is ONLY digits, "
    "optionally with a leading underscore ('_00', '12'), is copied "
    "through UNCHANGED — it marks a rig with unnamed bones.\n"
    "9. Japanese roots (Alligator/Pirrana/Tukan):\n"
    "   body   momo=Thigh, hiza=Knee, ashi=Foot, hiji=Elbow, te=Hand, "
    "kata=Shoulder, mune=Chest, hara=Abdomen, koshi/kosi=Hips, kubi=Neck, "
    "atama/kao=Head, ago=Jaw\n"
    "   tail   sippo/shippo=Tail, o=Tail\n"
    "   fish   munabire=Pectoral Fin, harabire=Pelvic Fin, sebire=Dorsal "
    "Fin, obire=Caudal Fin, shiribire=Anal Fin, era=Gill\n"
    "   Trailing L/R on any of these -> Left/Right prefix.\n"
    "10. Output Title Case, single spaces. Prefer the CANONICAL vocabulary; "
    "if nothing fits, use 'Bone'.\n\n"
    "ALIASES & TYPOS (map to the canonical term): Spline=Spine, Scull=Skull, Nek=Neck, Tai=Tail, Tone/Thouge/Tunge=Tongue, Eyeleds=Eyelid, HorseLink=Fetlock, LargeCannon=Cannon, PhalanxPrima=Pastern, PhalangesManus=Phalanges, Foreleg=Front Leg, Hindleg=Hind Leg, Digit=Finger, Hair=Mane, Little=Pinky (finger), locator/Trajectory/Cog=Root, Clavicle/Collarbone=Shoulder. Species terms: insect 'Clip'/'Shall'=Mandible, 'Pliers'/'Piers'=Pincer, cricket 'Feeler'=Antenna but fish 'Feelers'=Barbel, bird 'ponitail'=Crest.\n\n"
    "CANONICAL VOCABULARY (use these exact terms verbatim):\n"
    "  Core: Pelvis, Hips, Spine, Ribcage, Neck, Head, Skull, Skull Base, "
    "Head End, Body, Upper Body, Lower Body, Chest, Abdomen, Waist, Collar, "
    "Hip, Belly, Root, Center\n"
    "  Arm : Shoulder, Scapula, Arm, Upper Arm, Forearm, Elbow, Wrist, Hand, Palm\n"
    "  Leg : Thigh, Shin, Leg, Knee, Ankle, Foot, Heel, Toe, Paw, Hoof, "
    "Fetlock, Cannon, Metacarpus, Phalanges, Pastern\n"
    "  Finger: Finger, Thumb Finger, Index Finger, Middle Finger, Ring Finger, "
    "Pinky Finger\n"
    "  Head: Jaw, Upper Jaw, Lower Jaw, Tongue, Ear, Eye, Eyeball, Eyebrow, "
    "Eyelid, Mouth, Lip, Upper Lip, Lower Lip, Nose, Muzzle, Chin, Cheek\n"
    "  Appendage: Tail, Wing, Feather, Antenna, Barbel, Tentacle, Claw, "
    "Hand Claw, Fang, Mandible, Large Mandible, Lower Mandible, Pincer, "
    "Stinger, Appendage\n"
    "  Fins: Fin, Pectoral Fin, Pelvic Fin, Dorsal Fin, Caudal Fin, Anal Fin, Gill\n"
    "  Coat/Equipment: Mane, Fur, Whisker, Shell, Dorsal Plate, Crest, Horn, Reins, Halter\n"
    "  Quadruped: Front Leg, Middle Leg, Hind Leg, Front Paw, Back Paw, "
    "Front Hoof, Rear Hoof, Front Shoulder, Back Hip\n"
    "  Physics: Twist, Upper Arm Twist, Forearm Twist, Thigh Twist, Shin Twist, "
    "Thigh Muscle, Neck Muscle, Tail Twist, Jiggle, Handle, IK Chain, Bone\n"
    "Side goes first ('Left Upper Arm', 'Right Pinky Finger'); quadruped "
    "markers sit between side and part ('Right Front Shoulder'). "
    "COMPOSED labels are also valid: '<part> End' for chain tips/Nub "
    "bones, 'Inner/Middle/Outer <part>' (raptor toes, claws, fingers), "
    "'Upper/Lower Eyelid', 'Upper/Lower Left/Right/Front Lip', "
    "'Wrist Back', 'Elbow Back', 'Front/Middle/Hind Leg End'.\n\n"
    "EXAMPLES (one per rig family):\n"
    "  mixamorig:LeftUpLeg         -> Left Thigh\n"
    "  Mutant:RightHandThumb1      -> Right Thumb Finger\n"
    "  Sif:calf_twist_01_r         -> Right Shin Twist\n"
    "  index_01_l                  -> Left Index Finger\n"
    "  Bip01_L_Thigh               -> Left Thigh\n"
    "  BN_Bip01_R_Forearm_03       -> Right Forearm\n"
    "  NPC_L_Finger02              -> Left Finger\n"
    "  Elk_RearHoof_L              -> Left Rear Hoof\n"
    "  jt_FrontLeg1_R_C            -> Right Front Leg\n"
    "  Lt_Thumb1_jt                -> Left Thumb Finger\n"
    "  R_toeBase_jnt               -> Right Toe\n"
    "  Eye.R.001                   -> Right Eye\n"
    "  mixamorig:LeftShoulder.R    -> Right Shoulder\n"
    "  LeftHandIndex1              -> Left Index Finger\n"
    "  mixamorig:RightLeg          -> Right Shin\n"
    "  LeftFinger2Distal           -> Left Index Finger\n"
    "  BN_Spline_03                -> Spine\n"
    "  Bone.001                    -> Bone\n"
    "  F_R_Shoulder                -> Right Front Shoulder\n"
    "  B_L_Foot                    -> Left Back Foot\n"
    "  Topjaw                      -> Upper Jaw\n"
    "  momoR                       -> Right Thigh\n"
    "  munabireL                   -> Left Pectoral Fin\n"
    "  joint12 / Xtra01 / meshok   -> Bone\n"
    "OBJAVERSE (trailing global '_NN', often stacked with a chain index):\n"
    "  mixamorig:LeftUpLeg_056     -> Left Thigh\n"
    "  mixamorig:LeftHandThumb1_012 -> Left Thumb Finger\n"
    "  QuickRigCharacter_LeftForeArm_014 -> Left Forearm\n"
    "  Bip001 L UpperArm_07        -> Left Upper Arm\n"
    "  Bip001 L Finger01_011       -> Left Thumb Finger\n"
    "  Bip001 R Finger21_036       -> Right Middle Finger\n"
    "  Bip001 R Toe0_054           -> Right Toe\n"
    "  UpArm.R_010_15              -> Right Upper Arm\n"
    "  LowLeg.L_038_39             -> Left Shin\n"
    "  Hip_01_41                   -> Pelvis\n"
    "  Index.R.001_013_6           -> Right Index Finger\n"
    "  Rt_Eyelid_jt_08             -> Right Eyelid\n"
    "  Skeleton_Root_02            -> Root\n"
    "  joint1_2 / Bone.001_01      -> Bone\n\n"
    "BATCH EXAMPLE (notice identical outputs are preserved, not merged):\n"
    "  input  : [\"Spine\", \"Spine1\", \"Spine2\", \"Spine3\", \"Neck\", "
    "\"Tail_01\", \"Tail_02\", \"Tail_03\"]\n"
    "  output : [\"Spine\", \"Spine\", \"Spine\", \"Spine\", \"Neck\", "
    "\"Tail\", \"Tail\", \"Tail\"]\n"
    "  WRONG  : [\"Spine\", \"Neck\", \"Tail\"]  (8 inputs must yield 8 "
    "outputs — deduping is a failure)\n"
)


def build_user_prompt(rig_id, raw_names, correction=None):
    # type: (str, List[str], Optional[str]) -> str
    prompt = (
        "Rig identifier: {}\n"
        "Joint count: {}\n\n"
        "Raw joint names (JSON array, order matters):\n"
        "{}\n\n"
        "Return a JSON array of the SAME length with each raw name replaced by "
        "its cleaned canonical label."
    ).format(rig_id, len(raw_names), json.dumps(raw_names, ensure_ascii=False))
    if correction:
        prompt += "\n\nIMPORTANT: " + correction
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_response(text, expected_len):
    # type: (str, int) -> List[str]
    """Extract a JSON string array of the expected length from an LLM response.

    Tolerates markdown fences, ``<think>…</think>`` reasoning blocks (Qwen3
    and other thinking models) and trailing prose; raises ``ValueError`` on
    structural mismatch so the caller can retry or fall back.
    """
    if not text or not text.strip():
        raise ValueError("empty response")

    cleaned = strip_llm_noise(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no JSON array found in response")
        data = json.loads(cleaned[start:end + 1])

    if not isinstance(data, list):
        raise ValueError("response is not a JSON array")
    if len(data) != expected_len:
        raise ValueError("length mismatch: got {}, expected {}".format(
            len(data), expected_len))
    result = []
    for i, item in enumerate(data):
        if not isinstance(item, str) or not item.strip():
            raise ValueError("element {} is not a non-empty string".format(i))
        result.append(post_process(item.strip()))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-rig cleaner (with retry + rule-based fallback)
# ─────────────────────────────────────────────────────────────────────────────

def rule_based_clean(rig_id, raw_names):
    # type: (str, List[str]) -> List[str]
    """Rule-based fallback. ``rig_id`` drives animal-specific dispatch for
    Truebones (e.g. Spider, Pirrana); opaque Objaverse UUIDs fall through to
    the generic cleaner."""
    return [post_process(clean_joint_name(raw, rig_id)) for raw in raw_names]


def clean_rig(rig_id, raw_names, client, system_prompt=SYSTEM_PROMPT,
              max_tokens=2048, max_retries=MAX_RETRIES, fallback_on_error=True):
    # type: (...) -> Tuple[List[str], bool]
    """Clean one rig's joint list via the LLM. Returns ``(names, used_fallback)``.

    Structural errors (length mismatch, bad JSON, ...) are fed back to the
    LLM on the next attempt so it can self-correct.
    """
    if not raw_names:
        return [], False

    last_err = None
    correction = None
    for attempt in range(1, max_retries + 1):
        try:
            raw_text = client.generate(
                system_prompt, build_user_prompt(rig_id, raw_names, correction=correction),
                max_tokens)
            return parse_response(raw_text, len(raw_names)), False
        except FatalLLMError:
            raise
        except Exception as err:  # noqa: BLE001 — retry on any backend/parse error
            last_err = err
            if isinstance(err, ValueError):
                correction = (
                    "Your previous response failed validation ({}). Return a "
                    "JSON array of EXACTLY {} strings — one cleaned label per "
                    "input joint, in the same order. Do not add, merge, split, "
                    "or skip any entries. Output only the JSON array, no prose "
                    "or code fences."
                ).format(err, len(raw_names))
            logger.warning("[{}] attempt {}/{} failed: {}".format(
                rig_id, attempt, max_retries, err))
            if attempt < max_retries:
                time.sleep(1.5)

    if not fallback_on_error:
        raise RuntimeError("LLM cleaning failed for {}: {}".format(rig_id, last_err))

    logger.error("[{}] all {} attempts failed ({}); falling back to rule-based "
                 "cleaner".format(rig_id, max_retries, last_err))
    return rule_based_clean(rig_id, raw_names), True


# ─────────────────────────────────────────────────────────────────────────────
# Batch driver
# ─────────────────────────────────────────────────────────────────────────────

def clean_all(input_path, output_path, client, system_prompt=SYSTEM_PROMPT,
              max_tokens=2048, max_retries=MAX_RETRIES, fallback_on_error=True,
              failed_path=None, overwrite=False, redo_failed=False):
    # type: (...) -> Dict[str, List[str]]
    """Clean every rig in ``input_path`` and write to ``output_path``.

    Resumes from a partial output file: rigs whose cleaned list is already
    present with the correct length are skipped. Rig ids that fall back to
    the rule-based cleaner are appended to ``failed_path`` (one per line) so
    the batch can be re-run later with a stronger prompt/model.

    That resume rule makes this stage a silent no-op when ``output_path``
    was already filled in by the rule-based cleaner (both wrappers default to
    ``<export_dir>/clean_joint_names.json``), and it also skips exactly the
    rigs recorded in ``failed_path`` — a fallen-back rig already has a
    full-length list. ``overwrite`` re-processes every rig; ``redo_failed``
    re-processes only the ids in ``failed_path``.
    """
    with open(input_path, "r") as f:
        raw_data = json.load(f)

    cleaned = load_json(output_path)
    rig_ids = sorted(raw_data.keys())

    if failed_path is None:
        failed_path = os.path.join(os.path.dirname(output_path) or ".", "failed_clean_names.txt")
    already_failed = set(read_id_list(failed_path)) if os.path.exists(failed_path) else set()
    redo_ids = already_failed if redo_failed else set()
    if redo_failed:
        logger.info("--redo_failed: re-processing {} rig ids listed in {}".format(
            len(redo_ids), failed_path))

    fallback_count = 0
    processed = skipped = 0
    pbar = tqdm(rig_ids, desc="Cleaning", unit="rig", dynamic_ncols=True)
    for rig_id in pbar:
        raw_names = raw_data[rig_id]
        pbar.set_postfix_str(rig_id, refresh=False)

        existing = cleaned.get(rig_id)
        if (not overwrite and rig_id not in redo_ids
                and isinstance(existing, list) and len(existing) == len(raw_names)):
            skipped += 1
            continue

        processed += 1
        names_out, used_fallback = clean_rig(
            rig_id, raw_names, client,
            system_prompt=system_prompt, max_tokens=max_tokens,
            max_retries=max_retries, fallback_on_error=fallback_on_error,
        )
        fallback_count += int(used_fallback)
        cleaned[rig_id] = names_out
        save_json(cleaned, output_path)

        if used_fallback and rig_id not in already_failed:
            append_id_list(failed_path, [rig_id])
            already_failed.add(rig_id)

    logger.info("Done. {}/{} rigs cleaned by the LLM ({} via rule-based "
                "fallback), {} skipped (already complete); {} now holds {} "
                "rigs.".format(processed, len(rig_ids), fallback_count,
                               skipped, output_path, len(cleaned)))
    if processed == 0 and skipped:
        logger.warning(
            "Nothing was re-processed: every rig in {} already had a "
            "correct-length entry in {} (the rule-based stage writes the same "
            "file). Pass --overwrite to redo them all, or --redo_failed to "
            "redo just the ids in {}.".format(input_path, output_path, failed_path))
    if fallback_count:
        logger.info("Failed rig ids recorded at {}".format(failed_path))
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Clean joint names via an LLM (local / openai / deepseek).")
    parser.add_argument("--input", required=True,
                        help="Input joint_names.json path.")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: sibling 'clean_joint_names.json').")
    parser.add_argument("--report", action="store_true",
                        help="Print a residual-token coverage report after cleaning.")
    parser.add_argument("--no_fallback", action="store_true",
                        help="Raise on LLM failure instead of falling back to the "
                             "rule-based cleaner.")
    parser.add_argument("--failed_log", default=None,
                        help="Path to append rig ids that hit the rule-based fallback "
                             "(default: sibling 'failed_clean_names.txt').")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process every rig, even one that already has a "
                             "correct-length entry in the output. Without this the "
                             "stage is a no-op when the rule-based cleaner already "
                             "wrote the same file.")
    parser.add_argument("--redo_failed", action="store_true",
                        help="Also re-process the rig ids listed in --failed_log "
                             "(they always have a full-length entry from the "
                             "rule-based fallback, so they are otherwise skipped).")
    add_llm_args(parser, default_max_tokens=2048)
    args = parser.parse_args()

    output_path = args.output or os.path.join(
        os.path.dirname(args.input), "clean_joint_names.json")
    client = LLMClient.from_args(args)
    logger.info("{} | Input: {} | Output: {}".format(client, args.input, output_path))

    clean_data = clean_all(
        input_path=args.input,
        output_path=output_path,
        client=client,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        fallback_on_error=not args.no_fallback,
        failed_path=args.failed_log,
        overwrite=args.overwrite,
        redo_failed=args.redo_failed,
    )

    with open(args.input, "r") as f:
        raw_data = json.load(f)
    for rig_id in list(clean_data.keys())[:3]:
        print("\n--- {} ---".format(rig_id))
        for raw, out in zip(raw_data.get(rig_id, []), clean_data[rig_id]):
            print("  {:50s} -> {}".format(raw, out))

    if args.report:
        report_coverage(clean_data)


if __name__ == "__main__":
    main()
