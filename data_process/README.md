# UniML3D Data Processing

The pipeline that turns raw rigged assets (Truebones FBX, Mixamo FBX, Objaverse GLB) into **UniML3D** — the text-paired, topology-annotated motion clips used to train UniMate.

All commands are run **from the repository root**.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Directory Layout](#directory-layout)
- [Getting the Raw Data](#getting-the-raw-data)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Data Formats](#data-formats)
- [Conventions](#conventions)
- [Troubleshooting](#troubleshooting)

## Overview

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "fontFamily": "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif",
    "fontSize": "14px",
    "lineColor": "#94A3B8",
    "edgeLabelBackground": "#F1F5F9"
  },
  "flowchart": { "curve": "basis", "nodeSpacing": 40, "rankSpacing": 60 }
}}%%
flowchart LR
    raw(["📦 raw FBX / GLB"])
    export("<b>1 · Export</b><br/>Blender → NPZ")
    render("<b>2a · Render</b><br/>EEVEE multi-view")
    caption("<b>2b · Caption</b><br/>VLM")
    joints("<b>3 · Joints</b><br/>names + facing")
    extract("<b>4 · Extract</b><br/>canonicalize")
    animate("<b>5 · Animate</b><br/>rigged GLB/FBX")
    clips(["✨ training clips + cond.npy"])

    raw --> export
    raw --> render
    export -- "joint_names.json" --> joints
    render -- "frames" --> caption
    export -- "motions/*.npz" --> extract
    caption -- "captions / categories" --> extract
    joints -- "clean / face joints" --> extract
    extract --> clips
    clips --> animate
    raw -- "rigged mesh" --> animate

    classDef data fill:#1E1B4B,stroke:#1E1B4B,color:#F8FAFC
    classDef blender fill:#6D28D9,stroke:#5B21B6,color:#FFFFFF
    classDef vision fill:#0EA5E9,stroke:#0284C7,color:#FFFFFF
    classDef llm fill:#A855F7,stroke:#9333EA,color:#FFFFFF
    classDef out fill:#FFD21E,stroke:#CA8A04,color:#1E1B4B

    class raw data
    class export,animate blender
    class render,caption vision
    class joints,extract llm
    class clips out

    linkStyle default stroke:#94A3B8,stroke-width:1.5px
```

| Stage | Wrapper(s) | Input | Output |
|-------|-----------|-------|--------|
| 0. Download | `run_download.sh` | Hugging Face Hub | `dataset/raw/<dataset>/` |
| 1. Export | `run_export.sh`, `run_export_general.sh` | raw FBX / GLB | `export/<dataset>/{motions/*.npz, videos/*.mp4, tpose/*.png, joint_names.json}` |
| 2a. Render | `run_render_motion.sh`, `run_render_tpose.sh` | raw FBX / GLB (mixamo: animated characters) | `render/<dataset>/<clip>/v00{0..3}/*.png`, `render/<dataset>_tpose/<name>.png` |
| 2b. Caption | `run_caption_motion.sh`, `run_caption_category.sh` | multi-view renders | `motion_captions.json`, `category_groups.json` |
| 3. Joints | `run_joints_*.sh` | `joint_names.json` | `clean_joint_names.json`, `face_joint_names.json` |
| 4. Extract | `run_extract_features.sh` | export NPZs + stage-2/3 JSONs | per-clip NPZs, `cond.npy`, `captions.json` |
| 5. Animate | `run_animate_{motion,npz,fbx,mixamo,lbs}.sh`, `run_preprocess_char.sh` | motion clip + rigged mesh | animated `.glb` / `.fbx` |

Stage 4 reads the caption and joint-annotation JSONs from the export directory, so stages 2 and 3 must run first.

> [!NOTE]
> **Mixamo runs stage 5 in the middle of the pipeline.** Raw Mixamo FBXs are animation-only (armature without mesh), so the render/caption input is produced by `run_animate_mixamo.sh`: export → animate a character → render → caption. The result also ships in the mixamo repo as `animation_motion_ybot/`, so this step can be skipped by downloading it.

## Requirements

The pipeline uses the shared `unimate` conda environment — see [Environment Setup](../README.md#%EF%B8%8F-environment-setup) in the top-level README. Beyond it:

| Tool | Used by |
|------|---------|
| [Blender](https://www.blender.org/) on `PATH` | stages 1 & 5 (headless `blender -b -P`) |
| pip [`bpy`](https://pypi.org/project/bpy/) module (in the conda env) | stage 2a (EEVEE needs the module's GPU context) |
| `ffmpeg` on `PATH` | video visualizations |
| [`hf` CLI](https://hf.co/cli) | stage 0 |
| CUDA GPU | stages 2a & 2b |
| `OPENAI_API_KEY` / `GOOGLE_API_KEY` / `DEEPSEEK_API_KEY` | stage 2b (OpenAI / Gemini captioning) and stage 3 (OpenAI / DeepSeek joint annotation) API backends |

## Directory Layout

Code — one directory per stage, bash wrappers in `scripts/`, shared code in `utils/`:

```
data_process/
├── scripts/              Bash entry points — one run_<stage>_<task>.sh per task
├── motion_export/        Stage 1  raw assets → NPZ                 (Blender headless)
├── motion_rendering/     Stage 2a rigged assets → multi-view renders + T-pose grids (EEVEE)
├── vlm_caption/          Stage 2b multi-view renders → captions / body-plan categories (VLM)
├── joint_annotation/     Stage 3  joint-name cleanup + facing-direction joint selection
├── feature_extraction/   Stage 4  NPZ + metadata → canonical training clips + cond.npy
├── mesh_animation/       Stage 5  motion clip + rigged mesh → animated GLB/FBX (Blender);
│                                  single-asset preprocessing + manual NumPy FK/LBS
├── tools/                Standalone utilities (summary merging, FBX→GLB, QA visualizers)
└── utils/                Shared library code (no entry points)
```

Data — all wrappers share one convention (override any path through the environment variables each wrapper documents in its header):

```
dataset/raw/<dataset>/                      raw FBX / GLB assets
dataset/export/<dataset>/                   stage-1 NPZs + previews + stage-2/3 metadata
dataset/render/<dataset>/<clip>/v00{0..3}/  multi-view renders (captioning input)
dataset/render/<dataset>_tpose/             T-pose 2x2 grids (category input)
dataset/features/<dataset>/                 stage-4 training clips + cond.npy
```

`<dataset>` is one of `truebones`, `mixamo`, `objaverse`.

Both `dataset/render/` paths are symlinks into the dataset's own Hub mirror, so renders sit
beside the assets they came from and upload with them — `truebones` →
`raw/truebones/{animation_render,species_tpose}`, `mixamo` →
`raw/mixamo/{animation_motion_render,character_tpose}`, `objaverse` →
`raw/objaverse_renders/{glb_render,tpose}`. Objaverse keeps its renders in a separate repo
because there are 10,355 clip folders.

## Getting the Raw Data

The source assets are hosted on the Hugging Face Hub and download straight into the layout above:

```bash
bash data_process/scripts/run_download.sh mixamo      # → dataset/raw/mixamo/{animation_motion,character_refined,...}
bash data_process/scripts/run_download.sh objaverse   # → dataset/raw/objaverse/glb
bash data_process/scripts/run_download.sh truebones   # → dataset/raw/truebones (annotations only — see below)
bash data_process/scripts/run_download.sh objaverse_renders   # → dataset/raw/objaverse_renders (optional, ~6 GB)
```

| Dataset | Source |
|---------|--------|
| `mixamo` | [Linzhan/Mixamo-Animations-Characters](https://huggingface.co/datasets/Linzhan/Mixamo-Animations-Characters) |
| `objaverse` | [Linzhan/Objaverse-XL-Rigged-Animated](https://huggingface.co/datasets/Linzhan/Objaverse-XL-Rigged-Animated) |
| `truebones` | [Linzhan/Truebones-ZOO-Annotations](https://huggingface.co/datasets/Linzhan/Truebones-ZOO-Annotations) (prompts, metadata, renders, build scripts — no motion files) |
| `objaverse_renders` | [Linzhan/Objaverse-XL-Rigged-Animated-Renders](https://huggingface.co/datasets/Linzhan/Objaverse-XL-Rigged-Animated-Renders) (four-view clip MP4s + T-pose grids; download-only companion, not needed to run the pipeline) |

The mixamo and truebones repos also carry the stage-2a multi-view renders as per-view MP4 previews plus camera JSONs (`animation_motion_render/`, `animation_render/` — see each repo's README). The per-frame PNGs the caption stage reads are not hosted; stage 2a regenerates them (or extract stills from the MP4s).

> [!IMPORTANT]
> **The Truebones motion files are not downloadable from us.** The Truebones ZOO animal pack is a commercial product whose license does not allow redistribution, so its repo above ships annotations only. To reproduce the truebones part of the pipeline, purchase the pack from [Truebones](https://truebones.com), unpack it to `dataset/raw/truebones/Truebone_Z-OO/{Animal}/`, and run the downloaded `scripts/pipeline/` (see the repo's README) to rebuild the per-clip layout — one binary FBX per clip, flat in a single directory:
>
> ```
> dataset/raw/truebones/animation/{Species}-{Action}.fbx    # e.g. Alligator-Big_Mouth.fbx
> ```
>
> (Species names must not contain `-` — the first dash separates species from action.)
>
> Stage 5 additionally needs the **original per-animal layout** for the character meshes, since the flat per-clip files above are the animation source only:
>
> ```
> dataset/raw/truebones/Truebone_Z-OO/{Animal}/*.fbx        # e.g. Truebone_Z-OO/Dog-2/
> ```
>
> Directory names are matched ignoring separators and case, so the clip-name object type `Dog2` resolves to the `Dog-2` directory.

## Quick Start

End-to-end run for Objaverse (Truebones is identical with `objaverse` → `truebones`; see the [note above](#overview) for Mixamo's extra animate step):

```bash
# 0. Raw data
bash data_process/scripts/run_download.sh objaverse

# 1. Export raw GLBs to NPZ motion data
bash data_process/scripts/run_export.sh objaverse --multi-worker 8

# 2a. Render multi-view frames + T-pose grids
bash data_process/scripts/run_render_motion.sh objaverse --multi-worker 8
bash data_process/scripts/run_render_tpose.sh objaverse

# 2b. Caption motions + classify body plans (local Qwen3-VL by default)
bash data_process/scripts/run_caption_motion.sh objaverse --multi-gpu
bash data_process/scripts/run_caption_category.sh objaverse

# 3. Joint-name cleanup + facing-direction pair
bash data_process/scripts/run_joints_names_clean_llm.sh objaverse
bash data_process/scripts/run_joints_face_select_llm.sh objaverse

# 4. Canonicalized training clips + topology condition
bash data_process/scripts/run_extract_features.sh objaverse
```

Every stage is resumable — rerunning a wrapper skips already-complete outputs.

## Pipeline Stages

### Stage 1 — Export: raw assets → NPZ

```bash
bash data_process/scripts/run_export.sh truebones                    # flat per-clip {Species}-{Action}.fbx
bash data_process/scripts/run_export.sh mixamo                       # animation-only FBX (no mesh)
bash data_process/scripts/run_export.sh objaverse                    # GLB/GLTF
bash data_process/scripts/run_export.sh objaverse --multi-worker 8   # parallel workers (mixamo too)
bash data_process/scripts/run_export.sh mixamo --no-vis              # skip per-clip MP4 previews (much faster)
DATA_DIR=my_assets bash data_process/scripts/run_export.sh           # auto mode: .glb/.gltf and .fbx side by side
```

Skeletons are pruned of control/helper bones that carry no skin weight and converted to Y-up. Mixamo needs its explicit preset: animation-only FBXs carry no mesh, so auto mode cannot detect them. For custom assets there is a general exporter (single file or a directory of mixed GLB/GLTF/FBX):

```bash
bash data_process/scripts/run_export_general.sh my_model.glb
bash data_process/scripts/run_export_general.sh my_assets/ --multi-worker 8
```

### Stage 2a — Render: multi-view frames & T-pose grids

Renders each motion from four fixed views (front / back / left / right) with EEVEE, plus a T-pose 2x2 grid per asset for body-plan classification:

```bash
bash data_process/scripts/run_render_motion.sh truebones
bash data_process/scripts/run_render_motion.sh objaverse --multi-worker 8
bash data_process/scripts/run_render_motion.sh objaverse --missing-only    # resume: skip complete clips
bash data_process/scripts/run_render_tpose.sh objaverse
DATA_DIR=outputs/mixamo_characters bash data_process/scripts/run_render_motion.sh mixamo
```

Render folders use the same clip naming as stage 1, so they line up with the exported NPZs. Rendering is capped at the first 200 frames per clip (~6.7 s at 30 fps) — enough context for captioning without paying for full-length renders.

`--fps` (default 30) must match the export stage's `--fps`: glTF stores keyframe times in seconds, so the importer resamples them at the scene frame rate, and a mismatch makes the renders cover a different time window than the NPZs.

### Stage 2b — Caption: renders → text

One wrapper; the backend is picked from `MODEL`:

```bash
bash data_process/scripts/run_caption_motion.sh truebones                # local Qwen3-VL (default)
bash data_process/scripts/run_caption_motion.sh objaverse --multi-gpu    # shard across all visible GPUs
MODEL=gemini-3-flash-preview bash data_process/scripts/run_caption_motion.sh objaverse   # Gemini (GOOGLE_API_KEY)
MODEL=gpt-5-mini bash data_process/scripts/run_caption_motion.sh mixamo                  # OpenAI (OPENAI_API_KEY)
bash data_process/scripts/run_caption_category.sh objaverse              # body-plan classification
```

`--multi-gpu` applies to the local Qwen backend only; API backends scale via `NUM_WORKERS`. The local backend feeds each clip's four camera views as **native video** at the render frame rate (all frames, `--vision_input video`, the default); API backends receive frame sequences instead, capped by `--max_frames_per_view` (default 64, uniformly sampled including the first and last frame). For mixamo and truebones the official catalogue prompts (`animation_motion_prompts.json` / `animation_prompts.json`) are attached as reference hints automatically (`HINTS_JSON` overrides, `HINTS_JSON=""` disables). Clips whose renders are incomplete are recorded as failures and retried later rather than captioned from a partial set.

### Stage 3 — Joint annotation: name cleanup & facing direction

Two `<task>_<step>` trios — joint-name cleaning (`names_*`) and facing-direction pair selection (`face_*`) — each with a rule-based and an LLM variant. The LLM variants default to `deepseek-v4-flash` (also `gpt-*` or a local HF model id via `MODEL=`) and fall back to the rules on failure.

**Required** — these two produce the artifacts stage 4 consumes (`clean_joint_names.json`, `face_joint_names.json`):

```bash
bash data_process/scripts/run_joints_names_clean_llm.sh objaverse      # LLM names (falls back to rules)
bash data_process/scripts/run_joints_face_select_llm.sh objaverse      # LLM facing pair
```

**Optional** — refinement passes over the files above. Skip them entirely on a clean LLM run; reach for them when the required passes recorded failures (`failed_*.txt`) or when spot checks reveal residual label / pairing errors:

```bash
bash data_process/scripts/run_joints_names_correct_llm.sh objaverse    # re-check every label, correct in place
bash data_process/scripts/run_joints_face_correct_llm.sh objaverse     # retry rigs whose facing pair came back "empty"
```

`names_correct_llm` sends each `{raw, current}` label pair back to the LLM to keep or fix (most useful for rigs that fell back to the rules — restrict with `FAILED_LIST=.../failed_clean_names.txt`; long rigs are chunked with the full rig attached as read-only context). `face_correct_llm` re-attempts only `source == "empty"` entries by default; pass `--force_reattempt` to also retry the rule-fallback rigs recorded in `failed_face_joints.txt` — the correction pass re-derives from scratch (no rule hint, higher reasoning budget) and only a clean non-empty LLM win replaces an existing entry. Both edit in place and create a `.bak` sibling on first run.

Pure rule-based variants exist as `run_joints_names_clean_rule.sh` / `run_joints_face_select_rule.sh` (no API needed — useful offline or as a fast first pass). The QA visualizers behind this stage live in `data_process/tools/`: `run_joints_vis_tpose.sh` renders annotated T-pose PNGs (`tools/vis_tpose.py`), and to sanity-check a picked pair on an actual clip, `python -m data_process.tools.vis_motion <motion.npz> --direction face --face-joints I J` renders an MP4 with the implied heading arrow.

Both cleaners default to the same `clean_joint_names.json`, so the LLM pass skips rigs the rule-based pass already filled in; pass `--overwrite` to redo everything or `--redo_failed` to redo only the rigs listed in `failed_clean_names.txt`.

### Stage 4 — Extract: NPZ + metadata → training clips

```bash
bash data_process/scripts/run_extract_features.sh objaverse

# crop long motions into overlapping fixed-length clips instead of truncating
APPLY_CLIP=1 bash data_process/scripts/run_extract_features.sh truebones

# parallel over object types, skip MP4 previews
NUM_WORKERS=8 NO_VIS=1 bash data_process/scripts/run_extract_features.sh objaverse
```

Mixamo is reduced to a built-in **22-joint humanoid core** (finger chains and End bones dropped) via `--mixamo_core_joints` (default on); pass `--no-mixamo_core_joints` to keep the full 65-joint rig. Truebones and objaverse always use the full skeleton.

Every object type is canonicalized against its T-pose (facing → XZ-centering → diameter scaling → grounding), static pre/post-roll is trimmed, low-activity clips are filtered, and the per-object topology condition is written to `cond.npy`. Finished object types are cached under `cond_parts/`, so interrupted runs resume where they stopped; the cache is keyed on the clip/threshold settings **and** on digests of the caption / joint-annotation inputs, so changing any of them (including a stage-2/3 rerun) re-processes the affected object types instead of silently mixing old and new results.

Per-clip preview MP4s use a ground-plane view — checkerboard stage, smoothly following camera and root trajectory (`--vis_ground`, default on; `--no-vis_ground` renders the plain cubic view instead).

Each motion is grounded on its own lowest joint by default; pass `--use_tpos_ground_height` to reuse the T-pose's ground height for every clip instead. Either way `cond['ground_height']` and `cond['ground_height_mode']` record what was actually applied. Object types that fail are logged to `extract_errors.log` and skipped rather than aborting the run, and are retried on the next run.

### Stage 5 — Animate: drive a mesh with a motion

Used at inference time to render generated motions onto their rigged meshes:

```bash
ANIM_PATH=clip.npz bash data_process/scripts/run_animate_motion.sh objaverse   # stage-4 / generated clip + cond.npy
ANIM_PATH=clip.npz bash data_process/scripts/run_animate_npz.sh                # stage-1 export NPZ → rig
bash data_process/scripts/run_animate_fbx.sh                                   # raw animation FBXs → any character
bash data_process/scripts/run_animate_mixamo.sh                                # batch over Mixamo export NPZs (Y Bot)
```

`run_animate_fbx.sh` needs no export stage at all: it bakes a raw animation clip — or a whole directory of them — onto any character that shares the clips' bone names (the Mixamo convention), e.g. the full library on a different character:

```bash
CHAR_PATH=dataset/raw/mixamo/character_refined/Amy.fbx \
    bash data_process/scripts/run_animate_fbx.sh          # → outputs/animated_Amy/*.fbx
```

The character is auto-resolved for truebones / objaverse. For mixamo it defaults to `character_refined/Y_Bot.fbx` — the character whose skeleton the animation FBXs are authored on (identical 65-bone rig, so no bone reconciliation is needed); override with `CHAR_PATH` to use a different character. Armature bones absent from the motion are handled by `--extra_bones_strategy` (`merge` weights into the nearest kept ancestor, `remove` bones and their vertices, or `keep` them un-keyed).

#### Standalone assets: preprocess once, animate cond-free

For a single user-provided rigged asset, `run_preprocess_char.sh` runs the export + extraction stages in-process and bakes the result into a self-contained canonical asset:

```bash
CHAR_PATH=asset.glb OUTPUT_DIR=outputs/asset \
    FACE_R=R_Thigh FACE_L=L_Thigh FORMATS=glb,fbx \
    bash data_process/scripts/run_preprocess_char.sh
```

This leaves exactly three deliverables: `<name>_canonical.{glb,fbx}` (rest-pose asset rebuilt to the canonical T-pose — no animation, canonical joint order stored as a custom property), `cond.npy` (the model-side topology conditioning), and `motions/<clip>.npz` (one motion-feature NPZ per action). Assets **without** any action fall back to a rest-only cond (full skeleton — the export stage's motion-driven pruning needs animations — and no `motions/`).

A feature-format motion NPZ — generated by the model or extracted above — then drives the canonical asset directly, with no `cond.npy` or export NPZ at animate time:

```bash
CHAR_PATH=outputs/asset/<name>_canonical.glb ANIM_PATH=<motion.npz-or-directory> \
    bash data_process/scripts/run_animate_lbs.sh    # one animated GLB per action
```

`animate_lbs` computes FK and Linear Blend Skinning manually in NumPy (Blender only parses the asset; the math is numerically identical to Blender's armature modifier) and exports the animated rigged GLB/FBX through the same path as `animate_npz`; `SAVE=npz,obj` additionally dumps raw deformed vertex sequences / per-frame OBJs. It also accepts stage-1 export NPZs and, with `DATASET_TYPE`/`COND_PATH`, cond-resolved feature NPZs on non-preprocessed assets.

## Data Formats

**Stage-1 export NPZ** (`dataset/export/<dataset>/motions/*.npz`) — one file per clip:

| Field | Shape | Description |
|-------|-------|-------------|
| `rest_local_pos` / `rest_local_rot` | `(J, 3)` / `(J, 4)` | rest-pose local translations / rotations (quaternions) |
| `anim_local_pos` / `anim_local_rot` | `(T, J, 3)` / `(T, J, 4)` | per-frame local translations / rotations |
| `offsets` | `(J, 3)` | bind-pose bone offsets |
| `parents` | `(J,)` | parent indices, `-1` for the root |
| `names` | `(J,)` | bone name strings |
| `skin_matrix` | `(V, J)` | skinning weights (empty when no mesh) |
| `fps`, `action_name` | scalar | frame rate and source action |

**Stage-4 feature NPZ** (`dataset/features/<dataset>/motions/*.npz`) — canonicalized per-clip arrays, named `{object}-{motion}-{clip_idx}.npz`:

| Field | Shape | Description |
|-------|-------|-------------|
| `global_positions` | `(F, J, 3)` | global joint positions at canonical scale |
| `local_rotations` | `(F, J, 4)` | local joint rotations (quaternions) |
| `root_facing_quat` | `(F, 4)` | per-frame root quaternion aligning the skeleton to face Z+ |
| `fps` | scalar | frame rate of the clip (after any downsampling) |

**`cond.npy`** — one dict per object type holding everything the training dataloader needs about the skeleton: topology (`parents`, `offsets`, joint names), T-pose positions/rotations, graph structure (relations, distances, depths, spectral features, kinematic chains), calibration (`scale_factor`, `ground_height`), face joints and per-clip captions.

## Conventions

- **Run from the repo root.** Wrappers `cd` there themselves.
- **Resumable by default.** Every batch stage skips outputs that already exist and records failures in sibling error logs, so failed items are retried on the next run without redoing finished work.
- **Env-var overrides.** Every wrapper documents its overridable paths/knobs in its header comment; run any wrapper with `-h` to print it.
- **Parallelism.** Blender stages shard with `--multi-worker N`; local-model captioning shards with `--multi-gpu`; API backends use concurrent workers via `NUM_WORKERS`.
- **Naming.** Clips are named `{object_type}-{action}` (stage 1) and `{object_type}-{motion}-{clip_idx}` (stage 4). Mixamo is a single shared skeleton, so its files carry no object-type prefix.

## Troubleshooting

- **EEVEE renders fail / produce black frames under `blender -b`** — stage 2a must run with plain `python` and the pip `bpy` module (the wrappers already do); headless `blender -b` has no GPU display surface for EEVEE.
- **Offline compute nodes** — with a populated HuggingFace cache, export `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` before the local Qwen caption/classify runs.
- **`joint_names.json` is missing object types that have clips** — an export killed mid-run (Slurm walltime, OOM) can leave the summary JSONs behind the completion markers. Rebuild them from the markers:
  ```bash
  python -m data_process.tools.merge_summaries --output_dir dataset/export/<dataset>
  ```
  Additive and idempotent.
