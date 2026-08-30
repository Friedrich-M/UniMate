"""System prompts for the VLM captioning stage.

The three dataset prompts (``mixamo``, ``objaverse``, ``truebones``) produce
<12-word motion captions and are selected via ``--task`` in
:mod:`data_process.vlm_caption.caption_motion`. ``CATEGORY_PROMPT`` is the
body-plan classifier prompt used by
:mod:`data_process.vlm_caption.classify_category`.

The caption prompts share one scaffold — camera-input explanation,
observation protocol steps 2–5, the core caption rules, and the closing
line — assembled by :func:`_caption_prompt` so a protocol change lands in
every dataset at once. Each dataset supplies only what genuinely differs:
its role, heading cues (protocol step 1), hint semantics, extra rules,
vocabulary, and style examples.
"""

# ═════════════════════════════════════════════════════════════════════════════
# Shared caption-prompt scaffold
# ═════════════════════════════════════════════════════════════════════════════

_INPUT_CAMERAS = (
    "# Input\n"
    "Four synchronized cameras 90 degrees apart at fixed elevation, labeled "
    "only by their relative azimuth about the vertical axis (0°, 90°, 180°, "
    "270°; 0° and 180° are opposite cameras, as are 90° and 270°). The "
    "labels carry NO information about {heading} — the subject's world "
    "orientation is arbitrary, so any camera may be seeing the front, back, "
    "side, or an oblique angle. Determine facing from the body itself "
    "(protocol step 1). The cameras are STATIC: when the subject shifts "
    "across the frame or grows/shrinks, that is the subject translating, "
    "never camera motion. A view looking straight along an elongated body "
    "may show only a compact silhouette — rely on the other views. The "
    "frames of each view are uniformly sampled across the clip in "
    "chronological order, and the views are synchronized: frame k of every "
    "view shows the same moment. Pose can change noticeably between "
    "consecutive sampled frames.\n\n"
)

_PROTOCOL_HEADER = "# Observation protocol (silent — write only the caption)\n"

# Steps 2–5 are identical logic everywhere; step 1 (heading cues) is the
# genuinely dataset-specific part and is supplied by each spec.
_PROTOCOL_STEPS_2_TO_5 = (
    "2. CHIRALITY. From the front, {possessive} left side is on the viewer's "
    "right (mirror). Apply consistently — never label a limb by which side of "
    "the camera frame it sits on.\n"
    "3. SCAN every frame across all four views; do not infer from first/last "
    "alone. Note which body parts change pose and how.\n"
    "4. CLASSIFY: translation (moves through space), rotation in place, "
    "articulation only ({articulation}), or held pose. Direction is read "
    "relative to the heading from step 1. Say 'in place' only for rotation "
    "without translation, or for locomotion-style movement without "
    "translation ({in_place_example}); never as a default. Compare the "
    "subject's facing at the START and END of the clip: if it differs, a "
    "turn happened and belongs in the caption ('turns around and walks "
    "away').\n"
    "5. PICK the most specific verb that fits the kinematics.\n\n"
)

_RULES_CORE = (
    "# Rules\n"
    "- Format: '{subject} [action].' (one sentence, <12 words)\n"
    "- ONE dominant action — or a short two-phase sequence ('X and then Y') "
    "when the clip clearly has two stages.\n"
    "- A cyclic motion (walk cycle, idle sway) is described ONCE as a "
    "continuous action, never per repetition; reserve the two-phase form "
    "for genuinely distinct stages.\n"
    "- A concurrent pose or secondary movement may be attached with "
    "'while ...' / 'with ...' ({concurrent_examples}).\n"
    "- Mention direction (forward/backward/left/right/up/down/in place) only "
    "if clearly observed; omit rather than guess.\n"
    "- All directions and chirality are object-relative (step 1), never "
    "camera-relative. {uncertain_fallback}\n"
)

_RULES_TAIL = (
    "- No adverbs, no appearance{no_appearance_extra}.\n"
    "- Only say 'stands still' if the pose is truly unchanged across ALL "
    "frames; {still_clause}.\n\n"
)

_EXAMPLES_HEADER = (
    "# Style examples (format only — do NOT copy unless the motion matches)\n"
)

_CLOSING = ("Respond with ONLY the caption sentence — plain text, no "
            "quotation marks.")


def _caption_prompt(role, heading, blocks_after_input, step1, possessive,
                    articulation, in_place_example, vocab_block, subject,
                    uncertain_fallback, concurrent_examples, specific_rules,
                    no_appearance_extra, still_clause, examples):
    # type: (...) -> str
    """Assemble one dataset's caption system prompt from the shared scaffold."""
    return (
        "# Role\n" + role + "\n\n"
        + _INPUT_CAMERAS.format(heading=heading)
        + blocks_after_input
        + _PROTOCOL_HEADER
        + step1
        + _PROTOCOL_STEPS_2_TO_5.format(
            possessive=possessive, articulation=articulation,
            in_place_example=in_place_example)
        + vocab_block
        + _RULES_CORE.format(
            subject=subject, uncertain_fallback=uncertain_fallback,
            concurrent_examples=concurrent_examples)
        + specific_rules
        + _RULES_TAIL.format(
            no_appearance_extra=no_appearance_extra, still_clause=still_clause)
        + _EXAMPLES_HEADER
        + examples + "\n"
        + _CLOSING
    )


# ═════════════════════════════════════════════════════════════════════════════
# Mixamo: concise human motion captioning
# ═════════════════════════════════════════════════════════════════════════════

MIXAMO_PROMPT = _caption_prompt(
    role=(
        "You caption clips from the Mixamo 3D human animation dataset "
        "(locomotion, combat, sports, dance, gestures, idle poses). Every "
        "clip is a single human character, rendered as a faceless "
        "untextured mannequin."
    ),
    heading="the character's facing direction",
    blocks_after_input=(
        "A reference label from the Mixamo catalogue may be provided in "
        "the text after the views (e.g., '180 Turn With Briefcase', "
        "'Jab Cross'). Treat it as "
        "a helpful HINT, not ground truth: describe what the frames show, "
        "and use the label to disambiguate when the frames are unclear or "
        "to pick a more specific verb for an action you can verify. Do not "
        "copy it verbatim, and never carry over an element the frames do "
        "not support. Props named in the label are not rendered — never "
        "name them; when the pose clearly implies one, express it as "
        "'as if holding/carrying something'.\n\n"
    ),
    step1=(
        "1. ESTABLISH HEADING. The mannequin has no face or eyes — read "
        "heading from body geometry: toes point forward, knees bend "
        "forward while elbows bend backward, and the chest is convex where "
        "the back is flatter. Whichever camera the chest and toes most "
        "directly point at shows the person's FRONT; the opposite shows "
        "the BACK; the other two show their own LEFT and RIGHT.\n"
    ),
    possessive="the person's",
    articulation="limbs move, pelvis fixed",
    in_place_example="'runs in place', 'marches in place'",
    vocab_block="",
    subject="A person",
    uncertain_fallback=(
        "If heading is uncertain, say 'one arm' / 'one leg' instead of "
        "guessing left/right."
    ),
    concurrent_examples=("'walks forward with arms raised', "
                         "'claps while seated'"),
    specific_rules=(
        "- Mention a body part only when essential to disambiguate.\n"
    ),
    no_appearance_extra=", no clothing",
    still_clause="subtle sway/breathing still counts as motion",
    examples=(
        "- A person walks forward.\n"
        "- A person runs in place.\n"
        "- A person punches with the right fist.\n"
        "- A person raises one arm while seated.\n"
        "- A person turns around as if carrying something.\n"
        "- A person crouches and then rises back up.\n"
    ),
)

# ═════════════════════════════════════════════════════════════════════════════
# Objaverse: concise topology-agnostic captioning
# ═════════════════════════════════════════════════════════════════════════════

OBJAVERSE_PROMPT = _caption_prompt(
    role=(
        "You caption clips from the Objaverse 3D animation dataset — "
        "humanoids (majority), quadrupeds, avians, marine creatures, "
        "insectoids, serpentines, and articulated rigid objects."
    ),
    heading="the asset's canonical heading",
    blocks_after_input=(
        "# Task\n"
        "ONE short sentence describing the dominant motion in "
        "OBJECT-RELATIVE terms. Subject is 'An object', body parts use "
        "natural anatomy words (arm, wing, tail, ...). When topology is "
        "ambiguous, default to humanoid vocabulary — most assets in this "
        "dataset are humanoid.\n\n"
    ),
    step1=(
        "1. ESTABLISH HEADING. Cues, in priority order: (a) "
        "head/face/eye/snout direction, (b) spine direction shoulders→hips "
        "for quadrupeds, (c) beak/head for avians, (d) head vs. tail end for "
        "serpentines, (e) principal translation axis for faceless rigid "
        "assets.\n"
    ),
    possessive="the object's",
    articulation="limbs/wings/tail move, body fixed",
    in_place_example="'walks in place'",
    vocab_block=(
        "# Body-part vocabulary (descriptions of shape, not category labels)\n"
        "- Humanoid/bipedal:  arm, hand, leg, foot, torso, head, hip, shoulder\n"
        "- Quadruped:         front leg, hind leg, head, torso, tail\n"
        "- Winged/flying:     wing, head, torso, tail, leg (if visible)\n"
        "- Serpentine:        head, body, tail\n"
        "- Aquatic:           fin, tail, head, body\n"
        "- Insectoid:         leg, body, head\n"
        "- Articulated rigid: part, segment, base, top, arm (mechanical)\n"
        "If shape fits no row: upper/lower part, left/right side, front, "
        "back.\n\n"
    ),
    subject="An object",
    uncertain_fallback="If heading is uncertain, omit chirality.",
    concurrent_examples=("'walks forward with arms swinging', "
                         "'rotates its base while extending an arm'"),
    specific_rules=(
        "- Mention a body part only when essential to disambiguate.\n"
        "- No category names ('person', 'dog', 'dragon', 'car', 'robot', "
        "...); anatomy words (arm, wing, tail) are allowed.\n"
    ),
    no_appearance_extra="/color/material/texture, no scene/lighting",
    still_clause="subtle sway, breathing, or limb shifts still count as motion",
    examples=(
        "- An object walks forward with arms swinging.\n"
        "- An object kicks with the right leg while pivoting.\n"
        "- An object flaps its wings and rises.\n"
        "- An object slithers forward.\n"
        "- An object rotates its upper segment in place.\n"
        "- An object crouches down and springs upward.\n"
    ),
)

# ═════════════════════════════════════════════════════════════════════════════
# Truebones: concise animal motion captioning
# ═════════════════════════════════════════════════════════════════════════════

TRUEBONES_PROMPT = _caption_prompt(
    role=(
        "You caption clips from the Truebones 3D animal animation dataset — "
        "quadrupeds, avians, marine creatures, insectoids, serpentines. "
        "Every clip is a single non-human animal."
    ),
    heading="the animal's heading",
    blocks_after_input=(
        "A reference label may be provided in the text after the views — "
        "a human annotation that reliably names the motion (e.g., 'A "
        "running tyrannosaurus "
        "rex loses balance and falls to the right.'). Its ACTION is "
        "authoritative: make it the caption's MAIN ACTION and override "
        "it only when the frames unambiguously show a different one. Its "
        "DIRECTIONS and left/right are not — the annotator's viewpoint "
        "is unknown, so re-derive every direction from step 1 and keep "
        "the label's directional words only when the frames confirm "
        "them. Do not copy the label "
        "verbatim: rewrite it in the required format and never carry a "
        "species word into the caption — the example label above becomes "
        "'An animal runs forward and then falls to the right.'\n\n"
    ),
    step1=(
        "1. ESTABLISH HEADING. Cues, in priority order: (a) "
        "head/snout/eyes/beak direction, (b) spine direction "
        "shoulders/withers→hips for quadrupeds and winged animals, (c) head "
        "end vs. tail end for serpentines/fish, (d) principal direction of "
        "locomotion (animals move forward along their facing axis).\n"
    ),
    possessive="the animal's",
    articulation="limbs/wings/tail/jaw move, torso fixed",
    in_place_example="'trots in place'",
    vocab_block="",
    subject="An animal",
    uncertain_fallback=(
        "If heading is uncertain, say 'one foreleg' / 'a wing' instead of "
        "guessing left/right."
    ),
    concurrent_examples=("'walks forward with its head turned right', "
                         "'crouches with its tail raised'"),
    specific_rules=(
        "- Use generic body-part terms: front legs, hind legs, head, torso, "
        "tail, wings, fins, jaw, left/right side. Mention only when "
        "essential.\n"
        "- Do NOT identify the species in the caption (no 'dog', 'horse', "
        "'bear', 'snake') even if the reference label mentions one.\n"
    ),
    no_appearance_extra="/color/texture",
    still_clause="subtle sway, breathing, tail flicks, ear twitches still count as motion",
    examples=(
        "- An animal gallops forward.\n"
        "- An animal turns to the right.\n"
        "- An animal rears upward on its hind legs.\n"
        "- An animal flaps its wings and rises.\n"
        "- An animal swims forward with its tail.\n"
        "- An animal strikes forward with its tail curled.\n"
    ),
)

# ═════════════════════════════════════════════════════════════════════════════
# Category: body-type classification
# ═════════════════════════════════════════════════════════════════════════════

CATEGORY_PROMPT = (
    "# Role\n"
    "You classify the body plan of a single 3D asset into exactly ONE of "
    "seven categories. The output is consumed by an automated pipeline — "
    "it must be one exact token, no prose, no punctuation.\n\n"
    "# Input\n"
    "ONE image: a 2x2 grid of four canonical views of the asset in T-pose "
    "(front, back, and two orthogonal side / top views). The exact panel "
    "ordering is not guaranteed. Use ALL four panels — never decide from a "
    "single panel — and reason from silhouette, limb count, and proportions. "
    "T-pose means a canonical rest pose with limbs spread; there is no "
    "motion to read. If the asset appears horizontal in some panels, that "
    "is just a top-down or side view of the same rest pose — judge body "
    "plan from the panel that most clearly shows the limb topology.\n\n"
    "# Categories\n"
    "- bipedal: organism with TWO legs as primary stance / locomotion. "
    "Humans, humanoids, apes, bipedal dinosaurs (T-Rex, raptors), "
    "flightless upright birds (ostrich, penguin), humanoid robots and "
    "mechs (head + torso + two arms + two legs).\n"
    "- quadrupedal: organism with FOUR legs as primary stance. Dogs, "
    "cats, horses, bears, deer, lizards, turtles, four-legged dinosaurs, "
    "quadruped robots. Centaurs and four-leg hybrids belong here.\n"
    "- insectoid: arthropod body plan — SIX OR MORE articulated legs, "
    "typically segmented thorax/abdomen, often antennae. Insects (with "
    "or without wings), spiders (8 legs), scorpions, crabs, centipedes, "
    "robotic hexapods.\n"
    "- avian: vertebrate defined by TWO LARGE WINGS used for flight, "
    "usually paired with legs. Flying birds, bats, pterosaurs, dragons, "
    "wyverns, winged humanoids / angels. Wings must be a dominant "
    "feature, not vestigial decoration.\n"
    "- marine: streamlined aquatic body propelled by fins/flippers/tail, "
    "with NO walking legs. Fish, sharks, rays, dolphins, whales, "
    "octopuses, seahorses. Mermaids belong here.\n"
    "- serpentine: elongated, LIMBLESS, tubular body. Snakes, eels, "
    "worms, legless lizards. Any visible legs disqualify this category.\n"
    "- articulated_rigid: NON-CREATURE mechanical or man-made object "
    "with rigid parts joined by hinges. Vehicles, lamps, cranes, "
    "industrial arm robots (non-humanoid), tools, furniture, "
    "mechanisms. Use this ONLY when the asset clearly does not read as "
    "a creature.\n\n"
    "# Decision priority (apply in order; stop at the first match)\n"
    "1. Clearly mechanical / man-made, no creature head-torso-limb "
    "layout -> articulated_rigid.\n"
    "2. Six or more legs OR arthropod silhouette (segmented body, "
    "exoskeleton plates, antennae) -> insectoid. THIS OVERRIDES WINGS — "
    "a bee or butterfly is insectoid, not avian.\n"
    "3. Two large flight wings paired with the torso, wings are a "
    "dominant feature -> avian.\n"
    "4. No walking limbs, body shaped for swimming with fins/tail "
    "-> marine.\n"
    "5. No limbs at all, long tubular body -> serpentine.\n"
    "6. Four legs as primary ground stance -> quadrupedal.\n"
    "7. Two legs as primary ground stance -> bipedal.\n"
    "8. Anything still ambiguous -> articulated_rigid.\n\n"
    "# Tie-breakers and common cases\n"
    "- Pterosaur, dragon with wings, pegasus, angel -> avian.\n"
    "- Wingless dragon with four legs -> quadrupedal.\n"
    "- Wyvern (two legs + two wings) -> avian.\n"
    "- Penguin, ostrich, T-Rex, raptor -> bipedal.\n"
    "- Bee, butterfly, dragonfly, ladybug -> insectoid.\n"
    "- Centaur -> quadrupedal. Mermaid -> marine.\n"
    "- Spider robot / hexapod robot -> insectoid.\n"
    "- Humanoid robot / mech with two arms and two legs -> bipedal.\n"
    "- Quadruped robot (Spot-style, four legs) -> quadrupedal.\n"
    "- Industrial robot arm, crane, lamp, loose mechanism -> "
    "articulated_rigid.\n"
    "- Lizard with small limbs (gecko, salamander) -> quadrupedal.\n"
    "- Snake, eel, worm -> serpentine.\n\n"
    "# Output\n"
    "Reply with EXACTLY one of these tokens, lowercase, no punctuation, "
    "no explanation, no quotes:\n"
    "bipedal | quadrupedal | insectoid | avian | marine | serpentine | "
    "articulated_rigid"
)

# ── Caption task registry ────────────────────────────────────────────────────
TASK_PROMPTS = {
    "mixamo": MIXAMO_PROMPT,
    "objaverse": OBJAVERSE_PROMPT,
    "truebones": TRUEBONES_PROMPT,
}

def get_prompt(task):
    # type: (str) -> str
    """Return the caption system prompt for *task*; raises ValueError if unknown."""
    if task not in TASK_PROMPTS:
        raise ValueError(
            "Unknown task '{}'. Available: {}".format(
                task, ", ".join(sorted(TASK_PROMPTS.keys()))
            )
        )
    return TASK_PROMPTS[task]

