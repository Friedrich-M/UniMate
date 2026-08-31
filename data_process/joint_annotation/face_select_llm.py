"""Pick the facing-direction joint pair for each rig with a Language Model.

Complement to the rule-based :mod:`face_select_rule`: asks an LLM to choose ONE
bilaterally symmetric joint pair (hip / shoulder / wing / fin / ...) that
defines the rig's lateral axis for heading-angle computation. For serpentine
rigs with no bilateral symmetry, the LLM picks longitudinal body-axis
endpoints (head-tip / tail-tip) and sets ``body_axis: true``.

Backends (see :mod:`llm`): local HF causal LM, OpenAI API (default:
gpt-5-mini), DeepSeek API.

Inputs :
    ``joint_names.json``       — ``{rig_id: [raw_joint_name, ...]}``
    ``clean_joint_names.json`` — ``{rig_id: [clean_joint_name, ...]}``
Output :
    ``face_joint_names.json``  — ``{rig_id: {r_hip, l_hip, source, [body_axis]}}``

Pipeline per rig:
  1. Pre-filter the joint list into four small buckets (Right / Left /
     Head / Tail candidates). Dozens-of-joint rigs collapse to a handful.
  2. Short-circuit 'empty' if no candidates survive (no LLM call).
  3. Otherwise send only the compact candidate list to the LLM, together
     with the rule-based pick as a hint, and ask it to apply a three-step
     decision (bilateral -> body-axis -> empty).
  4. Validate the response (raw names must appear verbatim in the input);
     retry with a structured correction on failure.
  5. Fall back to the rule-based resolver after max retries / timeout.

Usage:
    python -m data_process.joint_annotation.face_select_llm \\
        --input_dir <export_dir>                      # OpenAI gpt-5-mini
    python -m data_process.joint_annotation.face_select_llm \\
        --input_dir <export_dir> --model deepseek-v4-flash
    python -m data_process.joint_annotation.face_select_llm \\
        --input_dir <export_dir> --model Qwen/Qwen3-8B
"""

import argparse
import json
import os
import re
import sys
import time

from loguru import logger
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data_process.joint_annotation.llm import (  # noqa: E402
    FatalLLMError,
    LLMClient,
    MAX_RETRIES,
    RigTimeout,
    add_llm_args,
    append_id_list,
    install_rig_alarm,
    load_json,
    read_id_list,
    save_json,
    strip_llm_noise,
    uninstall_rig_alarm,
)
from data_process.joint_annotation.face_select_rule import (  # noqa: E402
    HEAD_KEYWORDS,
    TAIL_KEYWORDS,
    empty_entry,
    resolve_face_joints,
    strip_trailing_num,
)


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "# TASK\n"
    "Pick the joint pair that defines one 3D rig's lateral facing axis. "
    "Rigs come from Objaverse (Mixamo, 3ds Max Biped, Maya QuickRig, "
    "Unreal mannequin, CAT, custom) and Truebones animals. All clean "
    "names are already normalised to a canonical vocabulary — reason on "
    "CLEAN names; RAW names are only for verbatim copy-back.\n\n"
    "# INPUT (user message)\n"
    "Four pre-filtered buckets (any may be empty):\n"
    "  RIGHT : rows whose clean name starts with 'Right '\n"
    "  LEFT  : rows whose clean name starts with 'Left '\n"
    "  HEAD  : rows whose clean name (minus any trailing ' N') is one of\n"
    "          {Head, Skull, Skull Base, Head End, Jaw, Upper Jaw,\n"
    "           Lower Jaw, Tongue, Muzzle, Nose, Chin}\n"
    "  TAIL  : rows whose clean name (minus any trailing ' N') is 'Tail'\n"
    "          or 'Tail Twist'\n"
    "Plus a 'Rule-based hint' JSON object — the deterministic resolver's "
    "pick. Use it as a sanity check, not ground truth (see HINT below).\n\n"
    "# ALGORITHM\n"
    "Execute the steps in order. Do not skip ahead.\n"
    "STEP 1. Compute OVERLAP.\n"
    "  - For each row in RIGHT, suffix = clean_name minus 'Right ' "
    "minus any trailing ' <digits>'. Collect into set R_SUFFIXES.\n"
    "  - For each row in LEFT,  suffix = clean_name minus 'Left ' "
    "minus any trailing ' <digits>'. Collect into set L_SUFFIXES.\n"
    "  - OVERLAP = R_SUFFIXES ∩ L_SUFFIXES.\n"
    "STEP 2. If OVERLAP is non-empty — APPLY RULE A AND RETURN.\n"
    "  a. Pick suffix `s` = first element of OVERLAP that appears in "
    "PRIORITY (see below). If none appear, pick the alphabetically "
    "first element of OVERLAP.\n"
    "  b. R_ROWS = RIGHT rows whose suffix == s; L_ROWS = LEFT rows "
    "whose suffix == s.\n"
    "  c. If len(R_ROWS)==1 and len(L_ROWS)==1, pair them.\n"
    "     Otherwise (chain duplicates, e.g. scorpion legs) match by "
    "DIGIT SIGNATURE: the tuple of ALL digit groups in the RAW name "
    "('Bip01_R_Thigh_4' -> (01,4)). Tier 1 — pair rows whose full "
    "signatures are equal. Tier 2 — no tier-1 match: drop the LAST "
    "group (the Objaverse per-joint global index: 'Bip01_R_Thigh_1_053' "
    "and 'Bip01_L_Thigh_1_054' both reduce to (01,1)) and pair on the "
    "rest. If neither tier matches, take R_ROWS[0] and L_ROWS[0].\n"
    "  d. Output: source = s.lower(); body_axis = false.\n"
    "  e. DO NOT consider HEAD or TAIL. Even if HEAD+TAIL look perfect, "
    "rule A wins whenever OVERLAP is non-empty.\n"
    "STEP 3. OVERLAP is empty. If HEAD is non-empty AND TAIL is non-"
    "empty — APPLY RULE B AND RETURN.\n"
    "  - r_hip = LAST HEAD row; l_hip = LAST TAIL row (chain endpoints).\n"
    "  - source = 'body_axis'; body_axis = true.\n"
    "STEP 4. Otherwise — APPLY RULE C (empty).\n"
    "  - Both hips are {raw: '', clean: ''}. source = 'empty'; "
    "body_axis = false.\n"
    "  - Vehicles, props, and abstract rigs land here. NEVER pair an "
    "unrelated Right/Left entry (e.g. lone 'Right Eye' without left "
    "mate) just to avoid emitting empty.\n\n"
    "# PRIORITY (highest → lowest; first match in OVERLAP wins)\n"
    "Hip-level  : thigh, shoulder, front shoulder, back hip, hip, "
    "scapula\n"
    "Whole-limb : upper arm, arm, front leg, hind leg, middle leg, "
    "back leg, wing, leg\n"
    "Aquatic    : pectoral fin, pelvic fin, fin, gill\n"
    "Arthropod  : pincer, mandible, large mandible, lower mandible, "
    "stinger, claw, hand claw, antenna\n"
    "Mid-limb   : forearm, shin, knee, elbow, ankle, wrist\n"
    "Extremity  : hand, palm, foot, heel, front paw, back paw, paw, "
    "front hoof, rear hoof, hoof, fetlock, cannon, metacarpus, "
    "pastern, toe\n"
    "Fingers    : thumb finger, index finger, middle finger, ring "
    "finger, pinky finger, finger, neck\n"
    "Weak       : eye, eyeball, eyelid, eyebrow, ear, horn, cheek, "
    "whisker, fang, barbel, tentacle, feather\n"
    "Last       : tail\n"
    "Suffix comparison is case-insensitive but uses the clean-name form "
    "('Upper Arm', 'Front Shoulder', 'Pectoral Fin'). Read the list above "
    "LITERALLY, left to right, top to bottom — it is exactly the order the "
    "rule-based resolver uses, so do not re-rank quadruped markers "
    "('Front Shoulder', 'Back Hip', 'Front Leg') against their plain "
    "counterparts ('Shoulder', 'Hip', 'Leg'). Note in particular that "
    "'shoulder' comes BEFORE 'front shoulder', while 'front leg' comes "
    "BEFORE 'leg'.\n\n"
    "# HINT\n"
    "The hint is the rule-based resolver's output. It is usually right "
    "but can err on edge cases:\n"
    "  - Hint says 'body_axis' or 'empty' yet OVERLAP is non-empty → "
    "hint is wrong, apply STEP 2 (rule A wins).\n"
    "  - Hint swapped sides (Right/Left in wrong fields) → fix.\n"
    "  - Hint chose a lower-priority suffix than STEP 2a finds → "
    "override with the higher-priority one.\n"
    "  - Hint paired mismatched chain indices on a multi-leg rig → fix "
    "via STEP 2c raw-suffix match.\n"
    "  - Your algorithm yields the same answer as the hint → return "
    "the hint verbatim.\n\n"
    "# OUTPUT\n"
    "Exactly one JSON object, no prose, no markdown fences, no comments:\n"
    '  {"r_hip":    {"raw": "<exact>", "clean": "<exact>"},\n'
    '   "l_hip":    {"raw": "<exact>", "clean": "<exact>"},\n'
    '   "source":   "<lowercase suffix or \'body_axis\' or \'empty\'>",\n'
    '   "body_axis": <true|false>}\n\n'
    "# INVARIANTS (verify before emitting)\n"
    "  I1. r_hip.raw != l_hip.raw, UNLESS both are '' (empty case).\n"
    "  I2. r_hip.raw and l_hip.raw each appear verbatim in the rig's "
    "input rows (copied character-for-character).\n"
    "  I3. r_hip always holds the Right (or head) side; l_hip always "
    "holds the Left (or tail) side. Never swap.\n"
    "  I4. body_axis == true  IFF  source == 'body_axis'.\n"
    "  I5. source is lowercase. If rule A fired, source is the suffix "
    "lowercased with single spaces (e.g. 'thigh', 'front shoulder', "
    "'pectoral fin').\n"
    "  I6. Unless body_axis, r_hip and l_hip mirror the SAME part: "
    "their clean names minus the side prefix and any trailing digits "
    "are identical. Never pair different parts ('Right Thigh' with "
    "'Left Shoulder' is invalid even though the sides are correct).\n\n"
    "# WORKED EXAMPLES\n"
    "## Ex1 — Mixamo humanoid (rule A, hip-level pick)\n"
    "RIGHT has 'Right Thigh' + 'Right Shoulder'; LEFT has the mirrors. "
    "OVERLAP = {Thigh, Shoulder}. Thigh outranks Shoulder → pick Thigh.\n"
    '  {"r_hip": {"raw": "mixamorig:RightUpLeg_056", "clean": '
    '"Right Thigh"},\n'
    '   "l_hip": {"raw": "mixamorig:LeftUpLeg_056",  "clean": '
    '"Left Thigh"},\n'
    '   "source": "thigh", "body_axis": false}\n'
    "## Ex2 — Objaverse quadruped (Front Shoulder beats Back Hip)\n"
    "No Thigh pair; OVERLAP = {Front Shoulder, Back Hip}. Front "
    "Shoulder outranks Back Hip.\n"
    '  {"r_hip": {"raw": "F_R_Shoulder_012", "clean": '
    '"Right Front Shoulder"},\n'
    '   "l_hip": {"raw": "F_L_Shoulder_012", "clean": '
    '"Left Front Shoulder"},\n'
    '   "source": "front shoulder", "body_axis": false}\n'
    "## Ex3 — Scorpion multi-leg (STEP 2c raw-suffix match)\n"
    "RIGHT has three 'Right Thigh' rows (Bip01_R_Thigh_1/_2/_4); LEFT "
    "the mirrors. OVERLAP = {Thigh}. Digit signatures: (01,1) on both "
    "sides -> tier-1 match.\n"
    '  {"r_hip": {"raw": "Bip01_R_Thigh_1", "clean": "Right Thigh"},\n'
    '   "l_hip": {"raw": "Bip01_L_Thigh_1", "clean": "Left Thigh"},\n'
    '   "source": "thigh", "body_axis": false}\n'
    "## Ex4 — OVERRIDE a mistaken body_axis hint\n"
    "RIGHT has 'Right Thigh'; LEFT has 'Left Thigh'; HEAD has 'Tongue'; "
    "TAIL has 'Tail'. Hint = body_axis Tongue+Tail. OVERLAP = {Thigh} "
    "is non-empty → STEP 2 wins, hint is wrong.\n"
    '  {"r_hip": {"raw": "RightUpLeg_033", "clean": "Right Thigh"},\n'
    '   "l_hip": {"raw": "LeftUpLeg_028",  "clean": "Left Thigh"},\n'
    '   "source": "thigh", "body_axis": false}\n'
    "## Ex5 — Snake / serpent (rule B fires)\n"
    "RIGHT and LEFT are empty; HEAD has 'Head'; TAIL has 'Tail_30'. "
    "STEP 3 applies.\n"
    '  {"r_hip": {"raw": "Head",    "clean": "Head"},\n'
    '   "l_hip": {"raw": "Tail_30", "clean": "Tail"},\n'
    '   "source": "body_axis", "body_axis": true}\n'
    "## Ex6 — Vehicle / prop (rule C fires)\n"
    "All sections empty, or only a lone 'Right Eye' with no left mate.\n"
    '  {"r_hip": {"raw": "", "clean": ""},\n'
    '   "l_hip": {"raw": "", "clean": ""},\n'
    '   "source": "empty", "body_axis": false}\n'
)


def prefilter_candidates(raw_names, clean_names):
    # type: (List[str], List[str]) -> Dict[str, List[Dict]]
    """Partition joints into LLM-relevant sections.

    Returns a dict with four keys — ``right``, ``left``, ``head``, ``tail``
    — each a list of ``{"raw": ..., "clean": ...}`` dicts preserving input
    order. Most rigs have 60+ joints but only a handful are candidates for
    the facing direction, so this drastically shortens the LLM's context.
    """
    buckets = {"right": [], "left": [], "head": [], "tail": []}
    for raw, clean in zip(raw_names, clean_names):
        if clean.startswith("Right "):
            buckets["right"].append({"raw": raw, "clean": clean})
            continue
        if clean.startswith("Left "):
            buckets["left"].append({"raw": raw, "clean": clean})
            continue
        base = strip_trailing_num(clean)
        if base in HEAD_KEYWORDS:
            buckets["head"].append({"raw": raw, "clean": clean})
        if base in TAIL_KEYWORDS:
            buckets["tail"].append({"raw": raw, "clean": clean})
    return buckets


def _format_section(name, rows):
    # type: (str, List[Dict]) -> str
    if not rows:
        return "{}: (none)".format(name)
    lines = ["{}:".format(name)]
    for row in rows:
        lines.append("  raw={!r:<40s} clean={!r}".format(row["raw"], row["clean"]))
    return "\n".join(lines)


def build_user_prompt(rig_id, candidates, rule_hint=None, correction=None):
    # type: (str, Dict[str, List[Dict]], Optional[Dict], Optional[str]) -> str
    sections = [
        "Rig identifier: {}".format(rig_id),
        "Candidate counts: right={}  left={}  head={}  tail={}".format(
            len(candidates["right"]), len(candidates["left"]),
            len(candidates["head"]), len(candidates["tail"])),
        "",
        _format_section("RIGHT", candidates["right"]),
        "",
        _format_section("LEFT", candidates["left"]),
        "",
        _format_section("HEAD", candidates["head"]),
        "",
        _format_section("TAIL", candidates["tail"]),
        "",
    ]
    if rule_hint is not None:
        sections.extend([
            "Rule-based hint (deterministic resolver's pick — HINT, not ground truth):",
            json.dumps(rule_hint, ensure_ascii=False, indent=2),
            "",
        ])
        sections.append(
            "Apply rules A -> B -> C and return the JSON object only. If the "
            "hint already satisfies A/B/C, return it verbatim; otherwise override."
        )
    else:
        sections.append(
            "Apply rules A -> B -> C and return the JSON object only."
        )
    prompt = "\n".join(sections)
    if correction:
        prompt += "\n\nIMPORTANT: " + correction
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────────────

def _hip_fields(hip, side):
    # type: (object, str) -> Tuple[str, str]
    if not isinstance(hip, dict):
        raise ValueError("{} is not a JSON object".format(side))
    raw = hip.get("raw", "")
    clean = hip.get("clean", "")
    if not isinstance(raw, str) or not isinstance(clean, str):
        raise ValueError("{} raw/clean must be strings".format(side))
    return raw, clean


def _clean_side(clean_name):
    # type: (str) -> Optional[str]
    """``"Left"`` / ``"Right"`` if the clean label carries a side, else None."""
    if clean_name.startswith("Left "):
        return "Left"
    if clean_name.startswith("Right "):
        return "Right"
    return None


def _validate_sides(r_clean, l_clean, body_axis):
    # type: (str, str, bool) -> None
    """Enforce invariant I3: ``r_hip`` holds the Right (or head) side and
    ``l_hip`` the Left (or tail) side.

    The prompt states I3 but a model can still return a swapped pair, and a
    swap is invisible downstream: ``get_root_facing_quat`` just builds the
    lateral axis the other way round, silently rotating every clip of that
    object type by 180 degrees. Raise ``ValueError`` so the caller retries
    with a correction and, if that fails too, falls back to the rule-based
    resolver.
    """
    if body_axis:
        if (strip_trailing_num(r_clean) in TAIL_KEYWORDS
                or strip_trailing_num(l_clean) in HEAD_KEYWORDS):
            raise ValueError(
                "body-axis pair is swapped: r_hip must be the HEAD end and "
                "l_hip the TAIL end, got r_hip={!r} / l_hip={!r}".format(
                    r_clean, l_clean))
        return
    if _clean_side(r_clean) == "Left" or _clean_side(l_clean) == "Right":
        raise ValueError(
            "left/right pair is swapped: r_hip must be the Right-side joint "
            "and l_hip the Left-side one, got r_hip={!r} / l_hip={!r}".format(
                r_clean, l_clean))


def parse_response(text, raw_names, clean_names):
    # type: (str, List[str], List[str]) -> Dict
    """Extract and validate a ``face_joint_names.json``-shaped object.

    Tolerates markdown fences and ``<think>…</think>`` blocks. Raises
    ``ValueError`` on structural mismatch so the caller can retry or fall
    back. Validation rules:

    - ``r_hip.raw`` / ``l_hip.raw`` must exist verbatim in ``raw_names``
      (or both be ``""`` for the empty sentinel).
    - ``r_hip.raw != l_hip.raw``.
    - The clean name is re-sourced from ``clean_names`` so the LLM can't
      drift the canonical label.
    - The sides are not swapped (invariant I3, see :func:`_validate_sides`).
    """
    if not text or not text.strip():
        raise ValueError("empty response")

    cleaned = strip_llm_noise(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no JSON object found in response")
        data = json.loads(cleaned[start:end + 1])

    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")
    for key in ("r_hip", "l_hip", "source"):
        if key not in data:
            raise ValueError("missing key '{}'".format(key))

    r_raw, _ = _hip_fields(data["r_hip"], "r_hip")
    l_raw, _ = _hip_fields(data["l_hip"], "l_hip")
    source = str(data["source"]).strip().lower()
    body_axis = bool(data.get("body_axis", False))

    if not r_raw and not l_raw:
        return empty_entry()
    if not r_raw or not l_raw:
        raise ValueError("one of r_hip/l_hip is empty; both must be filled "
                         "or both empty for 'empty' source")
    if r_raw == l_raw:
        raise ValueError("r_hip.raw == l_hip.raw ({!r}): pair must be two "
                         "distinct joints".format(r_raw))
    if r_raw not in raw_names:
        raise ValueError("r_hip.raw {!r} not in input raw joint list".format(r_raw))
    if l_raw not in raw_names:
        raise ValueError("l_hip.raw {!r} not in input raw joint list".format(l_raw))

    r_idx = raw_names.index(r_raw)
    l_idx = raw_names.index(l_raw)
    is_axis = body_axis or source == "body_axis"
    _validate_sides(clean_names[r_idx], clean_names[l_idx], is_axis)
    if not is_axis:
        # Invariant I6: a lateral pair must mirror the SAME part — a
        # cross-part pair ('Right Thigh' + 'Left Shoulder') passes the side
        # check yet yields a skewed facing axis.
        r_base = strip_trailing_num(
            re.sub(r'^(Left|Right)\s+', '', clean_names[r_idx]))
        l_base = strip_trailing_num(
            re.sub(r'^(Left|Right)\s+', '', clean_names[l_idx]))
        if r_base.lower() != l_base.lower():
            raise ValueError(
                "r_hip and l_hip are different parts ({!r} vs {!r}); the "
                "pair must be the mirrored copies of one joint".format(
                    clean_names[r_idx], clean_names[l_idx]))
    entry = {
        "r_hip": {"raw": raw_names[r_idx], "clean": clean_names[r_idx]},
        "l_hip": {"raw": raw_names[l_idx], "clean": clean_names[l_idx]},
        "source": source or "pair",
    }
    if body_axis or source == "body_axis":
        entry["body_axis"] = True
    return entry


# ─────────────────────────────────────────────────────────────────────────────
# Per-rig resolver (retry + rule-based hint + fallback + timeout)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_rig(rig_id, raw_names, clean_names, client, system_prompt=SYSTEM_PROMPT,
                max_tokens=512, max_retries=MAX_RETRIES, timeout_seconds=15,
                fallback_on_error=True, include_hint=True):
    # type: (...) -> Tuple[Dict, bool]
    """Pick the face-direction joint pair for one rig.

    Returns ``(entry, used_fallback)``. Rigs with no Right/Left or
    Head/Tail candidates short-circuit to the empty entry without an LLM
    call. On exhausted retries or timeout the rule-based pick is returned
    (or ``RuntimeError`` is raised when ``fallback_on_error`` is False and
    no timeout occurred).

    ``include_hint=False`` omits the rule-based hint from the prompt (it is
    still computed for the fallback path). The correction pass uses this:
    re-showing the very hint that already produced a wrong/empty entry
    anchors the model on it.
    """
    if not raw_names:
        return empty_entry(), False
    if len(raw_names) != len(clean_names):
        # clean_joint_names is indexed positionally against raw_names; a
        # stale/short list silently pairs the wrong bones (same guard as the
        # rule-based CLI). Abort instead of quietly emitting 'empty'.
        raise ValueError(
            "{!r}: {} cleaned names vs {} raw names — clean_joint_names.json "
            "is stale; re-run the joint-name cleaner.".format(
                rig_id, len(clean_names), len(raw_names)))

    candidates = prefilter_candidates(raw_names, clean_names)
    has_bilateral = bool(candidates["right"]) and bool(candidates["left"])
    has_body_axis = bool(candidates["head"]) and bool(candidates["tail"])
    if not has_bilateral and not has_body_axis:
        return empty_entry(), False

    # Rule-based hint (cheap, deterministic); also the fallback result.
    rule_hint = resolve_face_joints(clean_names, raw_names)

    last_err = None
    correction = None
    timed_out = False

    prev_handler = install_rig_alarm(timeout_seconds)
    try:
        for attempt in range(1, max_retries + 1):
            try:
                raw_text = client.generate(
                    system_prompt,
                    build_user_prompt(rig_id, candidates,
                                      rule_hint=rule_hint if include_hint else None,
                                      correction=correction),
                    max_tokens)
                return parse_response(raw_text, raw_names, clean_names), False
            except (RigTimeout, FatalLLMError):
                raise
            except Exception as err:  # noqa: BLE001 — retry on any backend/parse error
                last_err = err
                if isinstance(err, ValueError):
                    correction = (
                        "Your previous response failed validation ({}). Return "
                        "exactly one JSON object with keys r_hip, l_hip, source, "
                        "body_axis. r_hip and l_hip must be objects with 'raw' "
                        "and 'clean' string fields. 'raw' must be an exact copy "
                        "of one of the rig's raw joint names (or \"\" for the "
                        "'empty' source). r_hip.raw != l_hip.raw. r_hip MUST "
                        "be the Right-side joint (or, for body_axis, the HEAD "
                        "end) and l_hip the Left-side one (the TAIL end) — "
                        "never swap them, and both must mirror the SAME part. "
                        "No prose, no code fences."
                    ).format(err)
                logger.warning("[{}] attempt {}/{} failed: {}".format(
                    rig_id, attempt, max_retries, err))
                if attempt < max_retries:
                    time.sleep(1.5)
    except RigTimeout:
        timed_out = True
    finally:
        uninstall_rig_alarm(prev_handler)

    if not fallback_on_error and not timed_out:
        raise RuntimeError("LLM face-joint resolution failed for {}: {}".format(
            rig_id, last_err))

    if timed_out:
        logger.warning("[{}] exceeded {}s budget; using rule-based hint".format(
            rig_id, timeout_seconds))
    else:
        logger.error("[{}] all {} attempts failed ({}); falling back to "
                     "rule-based resolver".format(rig_id, max_retries, last_err))
    return rule_hint, True


# ─────────────────────────────────────────────────────────────────────────────
# Batch driver
# ─────────────────────────────────────────────────────────────────────────────

def _is_complete(entry):
    # type: (Dict) -> bool
    """A cached entry is complete if it's well-formed (even 'empty' counts)."""
    return (isinstance(entry, dict)
            and "r_hip" in entry and "l_hip" in entry and "source" in entry)


def resolve_all(raw_path, clean_path, output_path, client, system_prompt=SYSTEM_PROMPT,
                max_tokens=512, max_retries=MAX_RETRIES, timeout_seconds=15,
                fallback_on_error=True, failed_path=None):
    # type: (...) -> Dict[str, Dict]
    """Resolve face joints for every rig in ``raw_path`` / ``clean_path``.

    Resumes from a partial output file. Rig ids that fall back to the
    rule-based resolver are appended to ``failed_path`` (one per line).
    """
    with open(raw_path, "r") as f:
        raw_data = json.load(f)
    with open(clean_path, "r") as f:
        clean_data = json.load(f)

    result = load_json(output_path)
    rig_ids = sorted(set(raw_data.keys()) & set(clean_data.keys()))
    missing = set(raw_data) ^ set(clean_data)
    if missing:
        logger.warning("{} rig ids are not in both raw+clean; skipping: {}"
                       .format(len(missing), sorted(missing)[:5]))

    if failed_path is None:
        failed_path = os.path.join(
            os.path.dirname(output_path) or ".", "failed_face_joints.txt")
    already_failed = set(read_id_list(failed_path)) if os.path.exists(failed_path) else set()

    fallback_count = 0
    pbar = tqdm(rig_ids, desc="Resolving", unit="rig", dynamic_ncols=True)
    for rig_id in pbar:
        pbar.set_postfix_str(rig_id, refresh=False)
        if _is_complete(result.get(rig_id)):
            continue

        entry, used_fallback = resolve_rig(
            rig_id, raw_data[rig_id], clean_data[rig_id], client,
            system_prompt=system_prompt, max_tokens=max_tokens,
            max_retries=max_retries, timeout_seconds=timeout_seconds,
            fallback_on_error=fallback_on_error,
        )
        fallback_count += int(used_fallback)
        result[rig_id] = entry
        save_json(result, output_path)

        if used_fallback and rig_id not in already_failed:
            append_id_list(failed_path, [rig_id])
            already_failed.add(rig_id)

    logger.info("Done. {} rigs resolved ({} via rule-based fallback) -> {}"
                .format(len(result), fallback_count, output_path))
    if fallback_count:
        logger.info("Failed rig ids recorded at {}".format(failed_path))
    return result


def print_source_summary(result):
    # type: (Dict[str, Dict]) -> None
    sources = {}
    for rig_id, entry in result.items():
        sources.setdefault(entry["source"], []).append(rig_id)
    print("\nResolved {} rigs. Breakdown:".format(len(result)))
    for src, rigs in sorted(sources.items()):
        print("  [{}] {} rigs".format(src, len(rigs)))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pick facing-direction joint pairs via an LLM (local / openai / deepseek).")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing joint_names.json and clean_joint_names.json.")
    parser.add_argument("--raw_name", type=str, default="joint_names.json",
                        help="Filename of raw joint names inside --input_dir.")
    parser.add_argument("--clean_name", type=str, default="clean_joint_names.json",
                        help="Filename of cleaned joint names inside --input_dir.")
    parser.add_argument("--output_name", type=str, default="face_joint_names.json",
                        help="Output filename written inside --input_dir (ignored if "
                             "--output is set).")
    parser.add_argument("--output", type=str, default=None,
                        help="Explicit output path (default: <input_dir>/<output_name>).")
    parser.add_argument("--no_fallback", action="store_true",
                        help="Raise on LLM failure instead of falling back to the "
                             "rule-based resolver.")
    parser.add_argument("--failed_log", default=None,
                        help="Path to append rig ids that hit the rule-based fallback "
                             "(default: sibling 'failed_face_joints.txt').")
    parser.add_argument("--rig_timeout", type=int, default=15,
                        help="Per-rig wall-time budget in seconds (SIGALRM). 0 disables.")
    add_llm_args(parser, default_max_tokens=512)
    args = parser.parse_args()

    raw_path = os.path.join(args.input_dir, args.raw_name)
    clean_path = os.path.join(args.input_dir, args.clean_name)
    output_path = args.output or os.path.join(args.input_dir, args.output_name)
    client = LLMClient.from_args(args)
    logger.info("{} | Dir: {} | Output: {}".format(client, args.input_dir, output_path))

    result = resolve_all(
        raw_path=raw_path,
        clean_path=clean_path,
        output_path=output_path,
        client=client,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        timeout_seconds=args.rig_timeout,
        fallback_on_error=not args.no_fallback,
        failed_path=args.failed_log,
    )
    print_source_summary(result)


if __name__ == "__main__":
    main()
