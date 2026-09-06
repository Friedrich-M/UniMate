#!/usr/bin/env python3
"""Post-stage-3 QA patch for the export sidecars (joint labels, face pairs,
captions).

The stage-2/3 outputs are produced by rules plus LLM passes and were audited
by hand. This script applies the resulting corrections so that
they are reproducible on a fresh run of the pipeline and recorded in one
place. Two kinds of fixes:

  rules (in this file)
    - unsided ``Pelvis`` / ``Hip`` -> ``Hips`` (one label for the root pelvis
      bone across every rig; ``Left Hip`` / ``Right Hip`` are untouched);
    - numeric pass-through labels (``_2``, ``_1045``) -> ``Bone``;
    - numbered leg / claw-arm chains of the arthropod rigs get anatomical
      positions (Thigh, Shin, Foot, Toe ... / Upper Arm, Forearm, Hand, Claw)
      instead of ``Leg`` repeated along the chain;
    - rigs whose second side is a Blender/Maya duplicate (``LeftUpLeg.001``,
      ``Leg_L1``, ``UPLEG.001`` ...) get their side prefixes from the rest-pose
      X coordinate (+X = the character's left). About 6% of objaverse rigs face
      the other way, so this runs only on the hand-verified rigs listed below
      and, automatically, on rigs whose own side-carrying raw names confirm the
      orientation and where one sided label sits on both sides of the body
      (a duplicated chain) — there only that chain is relabelled;
    - the ``clean`` fields of face_joint_names.json are re-synced;
    - objaverse rigs are flagged in ``rig_flags.json``: automatically when the
      facing pair cannot be trusted (``source: empty``, a Bone/Bone pair, a body
      axis through unnamed bones), plus the hand-reviewed categories. Only
      ``tpose_wrong`` rigs are also written to ``filtered_objects.txt``, which
      stage 4 skips; every other flag is informational and the rig is kept;
    - caption grammar (``breaksdance`` -> ``breakdances``) and the per-dataset
      subject phrase;
    - locomotion qualifiers grounded in the root trajectory: ``<verb> forward``
      with no root travel -> ``<verb> in place``; ``<verb> in place`` with clear
      travel -> ``<verb> forward|backward`` when the travel is clearly along the
      facing axis (|cos| > 0.8 averaged over the clip), otherwise the qualifier
      is dropped (thresholds in body heights, see ``--still`` / ``--travel``).

  manual overrides (JSON files in ``--patch_dir``)
    <ds>_joint_labels.json   {rig: {raw_joint_name: clean_label}}
    <ds>_face_pairs.json     {rig: {"r_hip": raw, "l_hip": raw, "source": s,
                                    ["body_axis": true]}}
    <ds>_captions.json       {clip: caption}   (applied before the rules)
    <ds>_categories.json     {rig: category}   (moves the rig in category_groups.json)
    <ds>_filtered_clips.txt  <clip>   # <reason>   (any dataset: individual clips ->
                             export/<ds>/filtered_clips.txt, skipped by stage 4)
    <ds>_rig_flags.txt       <rig>    # <category>: <reason>   (objaverse: hand-reviewed rig flags ->
                             export/objaverse/rig_flags.json; only tpose_wrong also goes to
                             export/objaverse/filtered_objects.txt, which stage 4 skips)

Usage (from the repo root; ``dataset/export/<ds>`` may be symlinks):
    python data_process/tools/patch_annotations.py [--dry_run] [--datasets truebones mixamo]
    python data_process/tools/patch_annotations.py --export_root dataset/export \\
        --patch_dir dataset/UniML3D/patches

Pure numpy; reads one NPZ per rig for the rest pose and every clip NPZ once
for the root trajectory (cached in ``<patch_dir>/.root_motion_cache.json``).
"""

import argparse
import glob
import json
import os
import re
from collections import Counter, OrderedDict

import numpy as np

DATASETS = ('truebones', 'mixamo', 'objaverse')
SUBJECT = {'truebones': 'An animal', 'mixamo': 'A person', 'objaverse': 'An object'}

# Rigs whose multi-segment legs / claw arms are labelled by chain position.
CHAIN_RIGS = {
    'truebones': {
        'Crab': {'Leg': ('Thigh', 'Shin', 'Foot', 'Toe'), 'Upper Arm': ('Upper Arm', 'Forearm', 'Hand', 'Claw')},
        'HermitCrab': {'Leg': ('Thigh', 'Shin', 'Foot', 'Toe')},
        'Isopetra': {'Leg': ('Thigh', 'Shin', 'Foot', 'Toe')},
        'Spider': {'Leg': ('Thigh', 'Shin', 'Foot', 'Toe')},
        'SpiderG': {'Leg': ('Thigh', 'Shin', 'Foot', 'Toe')},
    },
}
# Rigs whose second side was created by duplicating the first (same side word
# or no side word in the raw names): side prefixes come from rest-pose X.
# Every rig here was checked to face +Z (so +X is its left) — by a foot->toe
# cue, by its own side-carrying raw names, or by its T-pose render.
X_SIDE_RIGS = {
    'objaverse': (
        '630247504c4b4f4b9b5c0371d89338c0',   # mixamorig:Left*.001 duplicates
        'b9d748ccc31344e2966d3f8176aed656',   # *Links / *Links.001 (Dutch "left")
        '669a872557bbceb71e864bc9_fbx',       # Leg_L / Leg_L1
        '669a6d9b57bbceb71e65252f_gltf',      # UPLEG / UPLEG.001 (unsided)
        '669a6f7f57bbceb71e679870_fbx',       # UPLEG / UPLEG.001 (unsided)
        '669a66c457bbceb71e5c8274_fbx',       # HIP_1 / HIP_2 (unsided)
        '669a858957bbceb71e84351d_fbx',       # shoulder_R / shoulder_R1
        '7f37f239817f41a2ad21b5f77cd0ff65',   # thigh.L / thigh.L.001
        '669a72e657bbceb71e6bfe20_fbx',       # Lhand / Rhand (unsided labels; names agree with X)
    ),
}
# Automatic candidates for side_from_x: a sided label that occurs on BOTH
# sides of the body (a duplicated appendage / whisker / finger chain labelled
# with one side) on a rig whose own raw names confirm the +X = left
# orientation (agreement >= X_SIDE_MIN_AGREE over >= X_SIDE_MIN_JOINTS sided
# joints). Rigs facing the other way are left alone.
X_SIDE_MIN_AGREE = 0.9
X_SIDE_MIN_JOINTS = 6
_RAW_SIDE_L = ('l', 'left', 'links', 'gauche', 'lewa', 'izq')
_RAW_SIDE_R = ('r', 'right', 'rechts', 'droit', 'prawa', 'der')

CENTER_LABELS = {'Root', 'Hips', 'Pelvis', 'Spine', 'Chest', 'Body', 'Neck', 'Head', 'Head End',
                 'Jaw', 'Tail', 'Bone', 'Center', 'Root End', 'Bone End', 'Abdomen', 'Ribcage'}

NUMERIC_RE = re.compile(r'^_?\d+$')
LOCO_VERBS = ('walks|runs|trots|jogs|sprints|marches|gallops|crawls|slithers|swims|flies|hops|'
              'bounds|scurries|waddles|limps|sneaks|strides|paddles|glides|dashes|skips|struts|'
              'creeps|prowls|stalks|shuffles|scampers|canters|flutters|soars|hovers|strolls|'
              'wanders|races')
FACING_COS = 0.8   # |cos(travel, facing)| needed to call travel forward / backward
LOCO_RE = re.compile(r'\b(%s)((?: \w+ly)?) (forward|in place)\b' % LOCO_VERBS)


# ---------------------------------------------------------------------------
# Kinematics (numpy only; quaternions are w,x,y,z as written by the exporter)
# ---------------------------------------------------------------------------

def qmul(a, b):
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def qrot(q, v):
    qv = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], -1)
    return qmul(qmul(q, qv), q * np.array([1, -1, -1, -1]))[..., 1:]


def fk(local_pos, local_rot, parents):
    n = len(parents)
    g = np.zeros((n, 3))
    gr = np.zeros((n, 4))
    for i in range(n):
        p = parents[i]
        if p < 0:
            g[i], gr[i] = local_pos[i], local_rot[i]
        else:
            g[i] = g[p] + qrot(gr[p], local_pos[i])
            gr[i] = qmul(gr[p], local_rot[i])
    return g


def rest_global(d):
    return fk(d['rest_local_pos'], d['rest_local_rot'], d['parents'])


def frame_global(d, t=0):
    return fk(d['anim_local_pos'][t], d['anim_local_rot'][t], d['parents'])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path, default=None):
    if not os.path.isfile(path):
        return default
    with open(path) as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def load_overrides(path):
    """Override JSON; keys starting with '_' (``_comment``) are documentation."""
    data = load_json(path, OrderedDict())
    return OrderedDict((k, v) for k, v in data.items() if not k.startswith('_'))


def save_json(path, obj, dry_run):
    if dry_run:
        return
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def object_type_of(ds, clip):
    return 'mixamo' if ds == 'mixamo' else clip.split('-', 1)[0]


def rig_npz(root, ds, rig):
    if ds == 'mixamo':
        files = sorted(glob.glob(os.path.join(root, 'motions', '*.npz')))
    else:
        files = sorted(glob.glob(os.path.join(root, 'motions', rig + '-*.npz')))
    return files[0] if files else None


def split_side(label):
    m = re.match(r'^(Left|Right) (.+)$', label)
    return (m.group(1), m.group(2)) if m else ('', label)


# ---------------------------------------------------------------------------
# Joint labels
# ---------------------------------------------------------------------------

def unify_root_labels(clean, stats):
    for rig, labels in clean.items():
        # a lone unsided 'Hip' is the pelvis; several unsided 'Hip' labels are
        # lateral hip bones (HIP_1 / HIP_2) and keep their base for side_from_x
        lone_hip = labels.count('Hip') == 1
        for i, lab in enumerate(labels):
            if lab == 'Pelvis' or (lab == 'Hip' and lone_hip):
                labels[i] = 'Hips'
                stats['pelvis->hips'] += 1
            elif NUMERIC_RE.match(lab):
                labels[i] = 'Bone'
                stats['numeric->bone'] += 1


def relabel_chains(names, labels, parents, spec, stats):
    """Chains of identical '<Side> <Base>' labels get positional labels."""
    children = {i: [] for i in range(len(names))}
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)
    for i in range(len(names)):
        side, base = split_side(labels[i])
        if base not in spec:
            continue
        p = parents[i]
        if p >= 0 and labels[p] == labels[i]:
            continue  # not the chain root
        seq = spec[base]
        j, k = i, 0
        while True:
            labels[j] = (side + ' ' + seq[min(k, len(seq) - 1)]).strip()
            stats['chain-relabel'] += 1
            nxt = [c for c in children[j] if split_side(labels[c]) == (side, base)]
            if len(nxt) != 1:
                break
            j, k = nxt[0], k + 1


def raw_side(name):
    """'L' / 'R' when a raw joint name carries a side token, else None."""
    n = name.lower()
    for prefix in ('mixamorig:', 'bip01', 'bip001'):
        n = n.replace(prefix, '')
    toks = [t for t in re.split(r'[^a-z]+', n) if t]
    if any(t in _RAW_SIDE_L or t.startswith('left') for t in toks):
        return 'L'
    if any(t in _RAW_SIDE_R or t.startswith('right') for t in toks):
        return 'R'
    return None


def needs_side_from_x(names, labels, d):
    """Base labels that sit on both sides of the body (a duplicated chain
    labelled with one side), or None — only for rigs whose raw names
    confirm the +X = left orientation."""
    g = rest_global(d)
    height = float(np.ptp(g, axis=0).max()) or 1.0
    x = (g[:, 0] - g[0, 0]) / height
    agree = total = 0
    seen = {}
    for i, (raw, lab) in enumerate(zip(names, labels)):
        if abs(x[i]) < 0.03:
            continue
        side, base = split_side(lab)
        if side:
            key = (side, base)
            seen.setdefault(key, set()).add(x[i] > 0)
        rs = raw_side(raw)
        if rs:
            total += 1
            agree += int((rs == 'L') == (x[i] > 0))
    both = {base for (_, base), v in seen.items() if len(v) == 2}
    if not both or total < X_SIDE_MIN_JOINTS:
        return None
    return both if agree / total >= X_SIDE_MIN_AGREE else None


def side_from_x(labels, d, stats, only_bases=None, min_dx=0.03):
    """+X = left, -X = right (rest pose); joints near the midline keep their label.

    ``only_bases`` restricts the relabel to those base labels (the duplicated
    chains of an auto-detected rig); the explicit rigs relabel every
    off-centre joint.
    """
    g = rest_global(d)
    height = float(np.ptp(g, axis=0).max()) or 1.0
    x0 = g[0, 0]
    for i, lab in enumerate(labels):
        _, base = split_side(lab)
        if base in CENTER_LABELS or (only_bases is not None and base not in only_bases):
            continue
        dx = (g[i, 0] - x0) / height
        if abs(dx) < min_dx:
            continue
        new = ('Left ' if dx > 0 else 'Right ') + base
        if new != lab:
            labels[i] = new
            stats['side-from-x'] += 1


def patch_joint_labels(ds, root, clean, names, overrides, stats, log):
    unify_root_labels(clean, stats)
    for rig, spec in CHAIN_RIGS.get(ds, {}).items():
        if rig not in clean:
            continue
        f = rig_npz(root, ds, rig)
        if not f:
            log.append(f'[{ds}] {rig}: no NPZ, chain relabel skipped')
            continue
        relabel_chains(names[rig], clean[rig], np.load(f, allow_pickle=True)['parents'], spec, stats)
    explicit = set(X_SIDE_RIGS.get(ds, ()))
    for rig in sorted(explicit):
        if rig not in clean:
            continue
        f = rig_npz(root, ds, rig)
        if not f:
            log.append(f'[{ds}] {rig}: no NPZ, side-from-x skipped')
            continue
        side_from_x(clean[rig], np.load(f, allow_pickle=True), stats)
    if ds == 'objaverse':      # auto: duplicated-side chains on rigs that face +Z per their names
        for rig in clean:
            if rig in explicit:
                continue
            f = rig_npz(root, ds, rig)
            if not f:
                continue
            d = np.load(f, allow_pickle=True)
            bases = needs_side_from_x(names[rig], clean[rig], d)
            if bases:
                stats['side-from-x-auto-rig'] += 1
                side_from_x(clean[rig], d, stats, only_bases=bases, min_dx=0.05)
    for rig, mapping in (overrides or {}).items():
        if rig not in clean:
            log.append(f'[{ds}] override for unknown rig {rig!r} ignored')
            continue
        for raw, label in mapping.items():
            if raw not in names[rig]:
                log.append(f'[{ds}] {rig}: override for unknown joint {raw!r} ignored')
                continue
            idx = names[rig].index(raw)
            if clean[rig][idx] != label:
                clean[rig][idx] = label
                stats['manual-label'] += 1


# ---------------------------------------------------------------------------
# Face pairs
# ---------------------------------------------------------------------------

def unreliable_facing(entry):
    if not entry or entry.get('source') == 'empty' or not entry['r_hip']['raw']:
        return 'empty'
    r, l = entry['r_hip']['clean'], entry['l_hip']['clean']
    if r == l == 'Bone':
        return 'bone/bone pair'
    if entry.get('body_axis') and ('Bone' in (r, l)):
        return 'body axis through unnamed bones'
    return ''


def patch_face_pairs(ds, face, clean, names, overrides, stats, log):
    for rig, spec in (overrides or {}).items():
        if rig not in names:
            log.append(f'[{ds}] face override for unknown rig {rig!r} ignored')
            continue
        entry = OrderedDict()
        for key in ('r_hip', 'l_hip'):
            raw = spec[key]
            if raw not in names[rig]:
                raise SystemExit(f'[{ds}] {rig}: face override joint {raw!r} not in rig')
            entry[key] = {'raw': raw, 'clean': ''}
        entry['source'] = spec.get('source', 'manual')
        if spec.get('body_axis'):
            entry['body_axis'] = True
        face[rig] = entry
        stats['manual-face-pair'] += 1
    # re-sync clean labels
    for rig, entry in face.items():
        if rig not in names or not entry.get('r_hip', {}).get('raw'):
            continue
        for key in ('r_hip', 'l_hip'):
            raw = entry[key]['raw']
            if raw in names[rig]:
                new = clean[rig][names[rig].index(raw)]
                if entry[key].get('clean') != new:
                    entry[key]['clean'] = new
                    stats['face-clean-resync'] += 1


# Rig flag categories. Only FILTER_CATEGORIES end up in filtered_objects.txt (stage 4 skips
# them); every other category is just recorded in rig_flags.json.
AUTO_FLAG = {'empty': 'empty_pair', 'bone/bone pair': 'bone_pair',
             'body axis through unnamed bones': 'body_axis_unnamed'}
FILTER_CATEGORIES = ('tpose_wrong',)


def load_rig_flags(path, names, log):
    """``patches/<ds>_rig_flags.txt``: one ``<rig>    # <category>: <reason>`` per line — hand-reviewed
    rigs (``tpose_wrong`` = rest pose lying / rotated, ``facing_wrong`` = the facing pair does not give the
    real front, ``object_no_front`` = prop without a front, ``verified_ok`` = hand-checked, clears an automatic
    flag). Unknown rigs are reported and ignored."""
    out = OrderedDict()
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            body, _, comment = line.partition('#')
            rig = body.strip()
            if not rig:
                continue
            if rig not in names:
                log.append(f'[objaverse] rig flag for unknown rig {rig!r} ignored')
                continue
            cat, _, reason = comment.strip().partition(':')
            out[rig] = (cat.strip() or 'flagged', reason.strip())
    return out


def write_filtered_clips(ds, root, patch_dir, frames, dry_run, log):
    """export/<ds>/filtered_clips.txt — hand-reviewed individual clips that stage 4 skips.

    Every dataset may have one: stage 4 applies the clip list before it groups
    clips into object types, so it is not objaverse-only the way
    filtered_objects.txt is. patches/<ds>_filtered_clips.txt holds
    ``<clip stem>    # <reason>`` lines (motion discontinuity, skeletons that
    disagree with the object's reference rig, and anything else reviewed by hand).
    """
    src = os.path.join(patch_dir, f'{ds}_filtered_clips.txt')
    path = os.path.join(root, 'filtered_clips.txt')
    if not os.path.isfile(src):
        return {}
    clips = OrderedDict()
    with open(src) as f:
        for line in f:
            body, _, comment = line.partition('#')
            clip = body.strip()
            if not clip:
                continue
            if frames and clip not in frames:
                log.append(f'[{ds}] filtered clip {clip!r} is not in clip_frames.json; kept anyway')
            clips[clip] = comment.strip()
    lines = ['# Individual export clips skipped by stage 4 (extract_features.py reads this file',
             '# automatically; --filtered_clips auto). The clip-level twin of filtered_objects.txt.',
             f'# Generated by data_process/tools/patch_annotations.py from '
             f'patches/{ds}_filtered_clips.txt — {len(clips)} clips.',
             '# Delete this file to keep them (they then go through the stage-4 quality checks only).']
    for clip, why in clips.items():
        lines.append(f'{clip}    # {why}' if why else clip)
    if not dry_run:
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
    kinds = Counter(w.split(':')[0] for w in clips.values())
    log.append(f'[{ds}] filtered_clips.txt: {len(clips)} clips ({kinds.most_common()})')
    return clips


def write_rig_flags(root, face, frames, manual, dry_run, log):
    """rig_flags.json (every flagged rig with its category) + filtered_objects.txt (FILTER_CATEGORIES only)."""
    flags = OrderedDict()
    for rig, entry in face.items():
        why = unreliable_facing(entry)
        if why:
            flags[rig] = {'category': AUTO_FLAG.get(why, why), 'reason': why, 'origin': 'auto'}
    verified = OrderedDict()
    for rig, (cat, reason) in (manual or {}).items():
        if cat == 'verified_ok':       # hand-checked: drop any automatic flag
            verified[rig] = reason
            flags.pop(rig, None)
            continue
        flags[rig] = {'category': cat, 'reason': reason, 'origin': 'manual'}
    cats = OrderedDict()
    for rig, fl in sorted(flags.items()):
        cats.setdefault(fl['category'], []).append(rig)
    bad = OrderedDict((rig, fl) for rig, fl in flags.items() if fl['category'] in FILTER_CATEGORIES)
    clips = [c for c in frames if object_type_of('objaverse', c) in bad]
    out = OrderedDict([
        ('_comment', 'Per-rig flags. Automatic facing checks (empty_pair: no bilateral joint pair; '
                     'bone_pair: the pair consists of unnamed Bone joints; body_axis_unnamed) plus the '
                     'hand-reviewed categories (tpose_wrong: the exported rest pose is unusable; facing_wrong: '
                     'the pair does not give the real front). Only ' + ', '.join(FILTER_CATEGORIES) +
                     ' is written to filtered_objects.txt, which stage 4 skips; every other flag is '
                     'informational and those rigs are kept. verified_ok lists rigs whose automatic flag was '
                     'cleared after review.'),
        ('filtered', list(FILTER_CATEGORIES)),
        ('counts', OrderedDict((c, len(r)) for c, r in cats.items())),
        ('categories', cats),
        ('rigs', OrderedDict(sorted(flags.items()))),
        ('verified_ok', verified),   # automatic flags cleared by hand review
    ])
    lines = ['# Object types skipped by stage 4 (extract_features.py reads this file automatically):',
             '# rigs flagged ' + ' / '.join(FILTER_CATEGORIES) + ' in patches/objaverse_rig_flags.txt — the exported',
             '# rest pose lies flat / is rotated / upside-down, so the stage-4 T-pose is unusable.',
             '# Other flags (empty_pair, bone_pair, facing_wrong, object_no_front, ...) are recorded in',
             '# rig_flags.json only and are NOT filtered.',
             f'# Generated by data_process/tools/patch_annotations.py — {len(bad)} rigs, {len(clips)} clips.',
             '# Delete this file to keep them (they then get the rest-pose facing as exported).']
    for rig, fl in bad.items():
        lines.append(f"{rig}    # {fl['category']}: {fl['reason']}")
    if not dry_run:
        with open(os.path.join(root, 'rig_flags.json'), 'w') as f:
            json.dump(out, f, indent=1)
        with open(os.path.join(root, 'filtered_objects.txt'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
    log.append(f'[objaverse] rig_flags.json: {dict(out["counts"])} (+{len(verified)} verified_ok); filtered_objects.txt: '
               f'{len(bad)} rigs / {len(clips)} clips ({", ".join(FILTER_CATEGORIES)})')
    return bad


# ---------------------------------------------------------------------------
# Body-plan categories
# ---------------------------------------------------------------------------

def patch_categories(ds, groups, overrides, names, stats, log):
    """Move rigs between the {category: [rigs]} buckets of category_groups.json."""
    for rig, cat in (overrides or {}).items():
        if rig not in names:
            log.append(f'[{ds}] category override for unknown rig {rig!r} ignored')
            continue
        current = next((c for c, members in groups.items() if rig in members), None)
        if current == cat:
            continue
        if current is not None:
            groups[current].remove(rig)
        groups.setdefault(cat, []).append(rig)
        groups[cat].sort()
        stats['manual-category'] += 1
    for cat in [c for c, members in groups.items() if not members]:
        del groups[cat]


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------

def root_motion(root, ds, clips, cache_path, face, log):
    """Per clip: net XZ root displacement in body heights and the travel
    direction relative to the facing pair at frame 0."""
    cache = load_json(cache_path, OrderedDict()) if cache_path else OrderedDict()
    key = lambda c: f'{ds}/{c}'
    todo = [c for c in clips if key(c) not in cache]
    if todo:
        log.append(f'[{ds}] computing root motion for {len(todo)} clips')
    for n, clip in enumerate(todo, 1):
        f = os.path.join(root, 'motions', clip + '.npz')
        if not os.path.isfile(f):
            continue
        d = np.load(f, allow_pickle=True)
        g0 = rest_global(d)
        height = float(np.ptp(g0, axis=0).max()) or 1.0
        r = d['anim_local_pos'][:, 0, :]
        disp = r[-1] - r[0]
        net = float(np.hypot(disp[0], disp[2]) / height)
        direction = ''
        rig = object_type_of(ds, clip)
        entry = face.get(rig)
        if net > 0 and entry and not unreliable_facing(entry):
            nm = list(d['names'])
            try:
                ri, li = nm.index(entry['r_hip']['raw']), nm.index(entry['l_hip']['raw'])
                dxz = np.array([disp[0], 0.0, disp[2]]) / (np.hypot(disp[0], disp[2]) or 1.0)
                cos = []
                nf = len(r)
                for t in range(0, nf, max(1, nf // 8)):     # facing averaged over the clip
                    gf = frame_global(d, t)
                    if entry.get('body_axis'):
                        fwd = gf[ri] - gf[li]          # head - tail
                    else:
                        fwd = np.cross(gf[li] - gf[ri], [0.0, 1.0, 0.0])   # left x up
                    fwd[1] = 0.0
                    if np.linalg.norm(fwd) > 1e-6:
                        cos.append(float(np.dot(dxz, fwd / np.linalg.norm(fwd))))
                if cos:
                    c = float(np.mean(cos))
                    # Strict: only a clear along-axis travel gets a word. Sideways or
                    # twisted-pelvis gaits (strafes) are left unqualified.
                    if c > FACING_COS:
                        direction = 'forward'
                    elif c < -FACING_COS:
                        direction = 'backward'
            except ValueError:
                pass
        cache[key(clip)] = {'net': round(net, 3), 'dir': direction}
        if cache_path and (n % 500 == 0 or n == len(todo)):  # derived data: cached even on --dry_run
            save_json(cache_path, cache, False)
    return {c: cache[key(c)] for c in clips if key(c) in cache}


def normalize_locomotion(caption, motion, still, travel, stats):
    if not motion:
        return caption

    def repl(m):
        verb, adv, qual = m.group(1), m.group(2), m.group(3)
        net, direction = motion['net'], motion['dir']
        if qual == 'forward' and net < still:
            stats['forward->in place'] += 1
            return f'{verb}{adv} in place'
        if qual == 'in place' and net >= travel:
            if direction:
                stats[f'in place->{direction}'] += 1
                return f'{verb}{adv} {direction}'
            stats['in place->(dropped)'] += 1
            return f'{verb}{adv}'
        return m.group(0)

    return LOCO_RE.sub(repl, caption)


def patch_captions(ds, caps, overrides, motion, args, stats, log):
    subject = SUBJECT[ds]
    for clip, text in (overrides or {}).items():
        if clip not in caps:
            log.append(f'[{ds}] caption override for unknown clip {clip!r} ignored')
            continue
        if caps[clip] != text:
            caps[clip] = text
            stats['manual-caption'] += 1
    for clip, text in caps.items():
        new = text.strip()
        new = re.sub(r'\bbreaksdance\b', 'breakdances', new)
        if not new.startswith(subject):
            fixed = re.sub(r'^(?:An?|The) [A-Za-z-]+\b', subject, new, count=1)
            if fixed.startswith(subject):
                new = fixed
                stats['subject-fixed'] += 1
            else:
                log.append(f'[{ds}] {clip}: caption does not start with {subject!r}: {new!r}')
        new = re.sub(r'\s+', ' ', new)
        if new and not new.endswith('.'):
            new += '.'
        new = normalize_locomotion(new, motion.get(clip), args.still, args.travel, stats)
        if new != text:
            caps[clip] = new
            stats['caption-changed'] += 1


# ---------------------------------------------------------------------------

def write_report(ds, report_dir, before, clean, face, caps, names, motion):
    """TSV of every change: kind, key, joint/-, old, new, note."""
    os.makedirs(report_dir, exist_ok=True)
    rows = []
    for rig, labels in clean.items():
        old = before['clean'].get(rig, [])
        for i, lab in enumerate(labels):
            if i < len(old) and old[i] != lab:
                rows.append(('label', rig, names[rig][i], old[i], lab, ''))
    for rig, entry in face.items():
        o = before['face'].get(rig)
        if o != entry:
            fmt = lambda e: f"{e['r_hip']['raw']} ({e['r_hip']['clean']}) / {e['l_hip']['raw']} ({e['l_hip']['clean']}) src={e.get('source')}" if e and e.get('r_hip') else str(e)
            rows.append(('face', rig, '-', fmt(o), fmt(entry), ''))
    for clip, text in caps.items():
        o = before['caps'].get(clip)
        if o != text:
            m = motion.get(clip, {})
            note = f"net={m.get('net')} dir={m.get('dir')}" if m else ''
            rows.append(('caption', clip, '-', o, text, note))
    path = os.path.join(report_dir, f'{ds}_changes.tsv')
    with open(path, 'w') as f:
        f.write('kind\tkey\tjoint\told\tnew\tnote\n')
        for r in rows:
            f.write('\t'.join(str(x) for x in r) + '\n')
    return path


def run_dataset(ds, args, log):
    root = os.path.join(args.export_root, ds)
    names = load_json(os.path.join(root, 'joint_names.json'))
    clean = load_json(os.path.join(root, 'clean_joint_names.json'))
    face = load_json(os.path.join(root, 'face_joint_names.json'))
    caps = load_json(os.path.join(root, 'motion_captions.json'))
    groups = load_json(os.path.join(root, 'category_groups.json'))
    frames = load_json(os.path.join(root, 'clip_frames.json'), {})
    if names is None or clean is None:
        log.append(f'[{ds}] no joint_names/clean_joint_names under {root}; skipped')
        return
    pd = args.patch_dir
    stats = Counter()
    before = {'clean': json.loads(json.dumps(clean)), 'face': json.loads(json.dumps(face)) if face else {},
              'caps': dict(caps) if caps else {}}

    patch_joint_labels(ds, root, clean, names, load_overrides(os.path.join(pd, f'{ds}_joint_labels.json')), stats, log)
    if face is not None:
        patch_face_pairs(ds, face, clean, names, load_overrides(os.path.join(pd, f'{ds}_face_pairs.json')), stats, log)
        if ds == 'objaverse':
            manual = load_rig_flags(os.path.join(pd, f'{ds}_rig_flags.txt'), names, log)
            write_rig_flags(root, face, frames, manual, args.dry_run, log)
            stats['manual-rig-flag'] += len(manual)
    stats['manual-filtered-clip'] += len(
        write_filtered_clips(ds, root, pd, frames, args.dry_run, log))
    if groups is not None:
        patch_categories(ds, groups, load_overrides(os.path.join(pd, f'{ds}_categories.json')), names, stats, log)
    if caps is not None:
        motion = root_motion(root, ds, list(caps), os.path.join(pd, '.root_motion_cache.json'),
                             face or {}, log) if not args.skip_locomotion else {}
        patch_captions(ds, caps, load_overrides(os.path.join(pd, f'{ds}_captions.json')), motion, args, stats, log)

    save_json(os.path.join(root, 'clean_joint_names.json'), clean, args.dry_run)
    if face is not None:
        save_json(os.path.join(root, 'face_joint_names.json'), face, args.dry_run)
    if caps is not None:
        save_json(os.path.join(root, 'motion_captions.json'), caps, args.dry_run)
    if groups is not None:
        save_json(os.path.join(root, 'category_groups.json'), groups, args.dry_run)
    if args.report_dir:
        write_report(ds, args.report_dir, before, clean, face or {}, caps or {}, names,
                     motion if caps is not None and not args.skip_locomotion else {})
    log.append(f'[{ds}] ' + ', '.join(f'{k}={v}' for k, v in sorted(stats.items())))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, '..', '..'))
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--export_root', default=os.path.join(repo, 'dataset', 'export'))
    ap.add_argument('--patch_dir', default=os.path.join(repo, 'dataset', 'UniML3D', 'patches'))
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS))
    ap.add_argument('--still', type=float, default=0.15,
                    help='net root XZ travel (body heights) below which locomotion is "in place"')
    ap.add_argument('--travel', type=float, default=0.5,
                    help='net root XZ travel (body heights) above which "in place" is replaced')
    ap.add_argument('--skip_locomotion', action='store_true', help='do not touch forward/in place')
    ap.add_argument('--report_dir', default=None,
                    help='write <ds>_changes.tsv (every label / face / caption change) here')
    ap.add_argument('--dry_run', action='store_true', help='report only, write nothing')
    args = ap.parse_args()
    os.makedirs(args.patch_dir, exist_ok=True)
    log = []
    for ds in args.datasets:
        run_dataset(ds, args, log)
    print('\n'.join(log))
    if args.dry_run:
        print('(dry run — nothing written)')


if __name__ == '__main__':
    main()
