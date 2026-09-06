#!/usr/bin/env python3
"""LLM rewrite pass for captions that name props or scenery.

The rendered clips contain only the rigged body — no weapons, balls, phones,
furniture, ladders, walls, ... — yet the VLM captioner sometimes copies a
prop from the catalogue hint or infers one from the pose ("aims a rifle",
"kicks a ball", "climbs a ladder"). This pass finds those captions with a
word list, asks a text LLM to rewrite each one so that it describes only the
body's motion, validates the answer (no prop word, subject preserved, short,
one sentence) and retries with sampling until it passes.

Nothing is modified in place: the result is a patch JSON
``{clip: {"old": ..., "new": ..., "ok": bool, "tries": n}}`` that
``data_process/tools/patch_annotations.py`` applies (after review) to
``motion_captions.json``.

Usage:
    python data_process/vlm_caption/caption_rewrite_llm.py \\
        --captions dataset/export/mixamo/motion_captions.json \\
        --output   dataset/UniML3D/patches/mixamo_captions_llm.json \\
        [--model Qwen/Qwen3.5-9B | gpt-5-mini | deepseek-v4-flash] \\
        [--clips list.txt] [--pattern REGEX] [--max_tries 4]
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from data_process.joint_annotation.llm import LLMClient, strip_llm_noise  # noqa: E402

# Object nouns that never exist in the renders. Scene words that describe the
# body's own situation ("ground", "floor", "air", "water") are deliberately
# absent: "falls to the ground" is body motion, not a prop.
PROP_WORDS = (
    'rifle|pistol|gun|firearm|sword|shield|ball|phone|arrow|axe|staff|knife|dagger|briefcase|'
    'lightsaber|spear|hammer|mace|flower|book|cup|bottle|dice|guitar|weapon|soccer ball|basketball|'
    'football|baseball|bag|cigarette|umbrella|wand|rope|ladder|wall|chair|table|bench|steering wheel|'
    'door|barrel|crate|gift|basket|oar|racket|broom|mop|shovel|pickaxe|lantern|candle|helmet|'
    'hat|cape|bucket|stone|pole|tree|branch|fence|obstacle|platform|stairs|bed|pillow|tool|item|prop|'
    'frisbee|puck|javelin|discus|shot put|kettlebell|dumbbell|barbell|controller|joystick|camera|'
    'microphone|mic|piano|violin|flute|trumpet|lasso|bike|bicycle|motorcycle|horse|car|vehicle|boat|'
    'skateboard|surfboard|snowboard|skis|sled|glider|parachute|balloon|kite|scope|grenade|bomb|'
    'scabbard|holster|quiver|weights'
)
# Nouns that are also verbs ("bows", "steps back", "rocks", "boxes", "drums"):
# only counted when a determiner / handling verb precedes them.
AMBIGUOUS_WORDS = 'bow|box|rock|drum|whip|net|bat|glass|stick|club|torch|cane|handle|wheel|bell|hose|step|card|key|coin|rail|drink|paddle'
PROP_RE = re.compile(
    r'\b(?:%s)s?\b|\b(?:a|an|the|their|his|her|its|one|two|with|holding|carrying|swinging|using|'
    r'grabbing|lifting|raising|throwing|catching|kicking|hitting|striking|drawing|aiming)\s+'
    r'(?:%s)s?\b' % (PROP_WORDS, AMBIGUOUS_WORDS), re.IGNORECASE)
SUBJECT_RE = re.compile(r'^(A person|An object|An animal)\b')

SYSTEM_PROMPT = """You edit captions for a motion-capture dataset. Each caption describes a short animation of a rigged body rendered WITHOUT any props, weapons, tools, scenery or other objects: only the body itself is visible.

Rewrite the caption so that it names NO object at all and describes only the body's motion (limbs, torso, head, hands, feet) while keeping everything that is not about an object unchanged.

Rules:
- Keep the subject phrase exactly as given ("A person", "An object" or "An animal").
- One sentence, present tense, at most 15 words, ending with a period.
- Do not add actions that are not implied by the original caption; do not drop actions that are.
- Where a pose clearly implies handling something, you may write "as if holding something", "as if aiming" or "as if carrying something" (at most once). Never say what the thing is.
- Forbidden words: any object noun (rifle, sword, ball, phone, ladder, wall, chair, ...), and also weapon, item, tool, prop, object (except in the subject "An object").
- Do not mention that anything is invisible, imaginary or missing.
- Output only the rewritten caption, nothing else.

Examples:
A person sits and aims a pistol. -> A person sits and extends one arm forward as if aiming.
A person crouches and walks backward while aiming a rifle, then stops. -> A person crouches and walks backward with both hands raised as if aiming, then stops.
A person kicks a soccer ball. -> A person kicks forward with one leg.
A person swings a sword downward. -> A person swings one arm downward as if holding something.
A person climbs a ladder. -> A person climbs upward, alternating hands and feet.
A person talks on the phone while walking. -> A person walks while holding one hand to the ear.
An object crouches while holding a shield and sword. -> An object crouches with both arms raised as if holding something.
A person reloads a rifle while standing. -> A person stands and moves both hands in front of the chest.
A person leans against a wall. -> A person leans sideways with the shoulder raised.
A person sits in a chair and crosses their legs. -> A person sits and crosses their legs."""


def load_json(path):
    with open(path) as f:
        return json.load(f)


def clean_answer(text):
    text = strip_llm_noise(text)
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), '')
    line = re.sub(r'^(?:Rewritten|Caption|Output|Answer)\s*:\s*', '', line, flags=re.IGNORECASE)
    line = line.strip().strip('"“”\'')
    if line and not line.endswith('.'):
        line += '.'
    return line


def validate(new, old, pattern):
    if not new:
        return 'empty'
    subj = SUBJECT_RE.match(old)
    if subj and not new.startswith(subj.group(1)):
        return 'subject changed'
    if pattern.search(new):
        return 'prop word: ' + pattern.search(new).group(0)
    if len(new.split()) > 18:
        return 'too long'
    if new.count('.') > 1:
        return 'multiple sentences'
    if re.search(r'\b(invisible|imaginary|unseen|not shown|no object)\b', new, re.IGNORECASE):
        return 'meta wording'
    return ''


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--captions', required=True, help='motion_captions.json to scan')
    ap.add_argument('--output', required=True, help='patch JSON {clip: {old, new, ok, tries}}')
    ap.add_argument('--clips', default=None,
                    help='Optional txt with clip names to rewrite (one per line); '
                         'default: every caption matching --pattern')
    ap.add_argument('--pattern', default=None, help='Regex override for the prop detector')
    ap.add_argument('--model', default='Qwen/Qwen3.5-9B')
    ap.add_argument('--backend', default=None)
    ap.add_argument('--max_tokens', type=int, default=64)
    ap.add_argument('--max_tries', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0, help='Debug: stop after N captions')
    args = ap.parse_args()

    pattern = re.compile(args.pattern, re.IGNORECASE) if args.pattern else PROP_RE
    captions = load_json(args.captions)
    if args.clips:
        with open(args.clips) as f:
            clips = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
    else:
        clips = [c for c, v in captions.items() if pattern.search(v)]
    if args.limit:
        clips = clips[:args.limit]
    print(f'{len(clips)} captions to rewrite from {args.captions}', flush=True)

    result = load_json(args.output) if os.path.isfile(args.output) else {}
    client = LLMClient(args.model, backend=args.backend)
    print(client, flush=True)

    t0 = time.time()
    n_ok = 0
    for k, clip in enumerate(clips, 1):
        old = captions[clip]
        if clip in result and result[clip].get('ok') and result[clip].get('old') == old:
            n_ok += 1
            continue
        best, best_err, tries = '', 'no answer', 0
        for tries in range(1, args.max_tries + 1):
            raw = client.generate(SYSTEM_PROMPT, f'Caption: {old}', args.max_tokens)
            new = clean_answer(raw)
            err = validate(new, old, pattern)
            if not err:
                best, best_err = new, ''
                break
            if not best or best_err.startswith('empty'):
                best, best_err = new, err
        ok = not best_err
        n_ok += int(ok)
        result[clip] = {'old': old, 'new': best, 'ok': ok, 'tries': tries}
        if not ok:
            result[clip]['error'] = best_err
        flag = '' if ok else f'   !! {best_err}'
        print(f'[{k}/{len(clips)}] {clip}\n    {old}\n -> {best}{flag}', flush=True)
        if k % 20 == 0 or k == len(clips):
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=1, ensure_ascii=False)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(f'done: {n_ok}/{len(clips)} passed validation in {time.time() - t0:.0f}s -> {args.output}')


if __name__ == '__main__':
    main()
