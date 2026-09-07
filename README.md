<h1 align="center">UniMate</h1>

<p align="center"><b>One Unified Model to Animate Diverse Skeletons</b></p>

<p align="center">
  <a href="https://linzhanmou.com/unimate/"><img alt="Project Page" src="https://img.shields.io/badge/Project_Page-6D28D9?style=for-the-badge&logo=githubpages&logoColor=white"></a>
  <a href="https://arxiv.org/abs/2609.05415"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2609.05415-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white"></a>
  <a href="https://huggingface.co/papers/2609.05415"><img alt="Hugging Face Paper" src="https://img.shields.io/badge/Paper-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000000"></a>
  <a href="https://linzhanmou.com/unimate/interactive.html"><img alt="Interactive Demo" src="https://img.shields.io/badge/Interactive_Demo-0EA5E9?style=for-the-badge&logo=threedotjs&logoColor=white"></a>
  <a href="https://huggingface.co/collections/Linzhan/unimate"><img alt="Hugging Face Dataset" src="https://img.shields.io/badge/Dataset-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000000"></a>
  <a href="https://linzhanmou.com/unimate/resources/unimate-poster.png"><img alt="SIGGRAPH Asia 2026 Poster" src="https://img.shields.io/badge/SIGGRAPH_Asia_2026-Poster-6D28D9?style=for-the-badge&labelColor=1E1B4B"></a>
</p>

<p align="center">
  <a href="https://linzhanm.github.io/">Linzhan Mou</a> ·
  <a href="https://jiahuilei.com/">Jiahui Lei</a> ·
  <a href="https://frank-zy-dou.github.io/">Zhiyang Dou</a> ·
  <a href="https://chenyue-cai.com/">Chenyue Cai</a> ·
  <a href="https://chaoyuesong.github.io/">Chaoyue Song</a> ·
  <a href="https://www.cs.princeton.edu/~af/">Adam Finkelstein</a> ·
  <a href="https://www.cs.princeton.edu/~smr/">Szymon Rusinkiewicz</a>
</p>

<p align="center">Princeton · UC Berkeley · MIT · NTU</p>

<div align="center">
    <img src="assets/teaser.png" alt="UniMate teaser" width="100%">
</div>

---

## 🔥 News

- **[2026-09-06]** The **training and inference code** is released. 🚀
- **[2026-09-04]** Our paper is on [arXiv](https://arxiv.org/abs/2609.05415) and [Hugging Face Papers](https://huggingface.co/papers/2609.05415). 📄
- **[2026-08-30]** The raw **UniML3D dataset** and its [data-processing pipeline](data_process/) are released. 🚀
- **[2026-08-01]** Our [Interactive Demo](https://linzhanmou.com/unimate/interactive.html) is live — browse our animation results in 3D. 🎮
- **[2026-07-18]** UniMate is accepted to SIGGRAPH Asia 2026! 🎉

> **[TODO]** Pretrained checkpoints, processed data and detailed documents are coming soon.

## 🛠️ Environment Setup

All components share a single conda environment, specified in [`requirements.txt`](requirements.txt):

```bash
conda create -n unimate python=3.10 -y
conda activate unimate
pip install "setuptools<81"
pip install -r requirements.txt --no-build-isolation
```

## 📊 Dataset & Data Processing

We introduce **UniML3D**, a large-scale dataset of 13,006 text-paired motion sequences covering diverse skeletal topologies — bipedal, quadrupedal, avian, marine, insectoid, serpentine, and articulated rigid objects — all brought into a unified canonicalization.

The raw source assets are available on the Hugging Face Hub (collected under [UniMate](https://huggingface.co/collections/Linzhan/unimate)): [Mixamo-Animations-Characters](https://huggingface.co/datasets/Linzhan/Mixamo-Animations-Characters), [Objaverse-XL-Rigged-Animated](https://huggingface.co/datasets/Linzhan/Objaverse-XL-Rigged-Animated) and [Truebones-ZOO-Annotations](https://huggingface.co/datasets/Linzhan/Truebones-ZOO-Annotations) (prompts, metadata and renders only). The Truebones ZOO animal motions themselves are a commercial asset pack whose license does not permit redistribution — please purchase the pack directly from [Truebones](https://truebones.com); our pipeline consumes the stock `Truebone_Z-OO` folder layout as-is.

<div align="center">
    <img src="assets/dataset_overview.png" alt="UniML3D dataset overview" width="100%">
</div>

See [`data_process/README.md`](data_process/README.md) for the full data processing pipeline that turns the raw assets into UniML3D (download → export → rendering → captioning → joint annotation → feature extraction → animation).

## 🏋️ Training

Training reads the canonicalized clips under `dataset/features/<dataset>/`, produced by stage 4 of the [data-processing pipeline](data_process/README.md#stage-4--extract-npz--metadata--training-clips).

Runs are configured by the JSON files in [`configs/`](configs/).

<details>
<summary><b>Config naming</b> — <code>{dataset}_{frames}frames_{attention}_{text_cond}.json</code></summary>

24 configs: 4 data combinations x 3 clip lengths x 2 model variants.

| Prefix | Training data |
|--------|---------------|
| `uniml3d_*` | Full UniML3D dataset (Truebones + Mixamo + Objaverse) |
| `truebones_*` / `mixamo_*` / `objaverse_*` | A single source |

| Length | `dataset.max_motion_length` |
|--------|-----------------------------|
| `60frames` / `90frames` / `120frames` | 60 / 90 / 120 frames per clip |

| Suffix | `model.attention` x `model.text_cond` |
|--------|---------------------------------------|
| `_graph_adaln` | `graph` x `adaln` — attention factored into spatial (per frame) and temporal (per joint) passes with graph-distance, edge-type and depth biases; the caption is folded into the adaLN modulation |
| `_full_cross_attn` | `full` x `cross_attn` — one attention over the flattened joint x time tokens; the caption enters every block as cross-attention keys/values |

The two axes are independent and all four combinations are implemented, so `full` x `adaln` and `graph` x `cross_attn` also run if you set them in a config; the two shipped pairings are the ones the paper compares.

Not shared across configs: `training.batch_size` and `training.num_steps` are tuned per data combination and clip length (GPU memory tracks batch x frames x joints). Every other setting is identical — including `dataset.{min,max}_joints = {5, 100}`, which bounds the skeleton sizes a run admits (object types outside the range are dropped) and, through what survives, the joint-axis padding width.

</details>

Launch with [🤗 Accelerate](https://github.com/huggingface/accelerate). Single GPU:

```bash
accelerate launch -m unimate.training.train --config configs/uniml3d_60frames_graph_adaln.json
```

Multi-GPU on one node (e.g. 8 GPUs):

```bash
accelerate launch --num_processes 8 -m unimate.training.train --config configs/uniml3d_60frames_graph_adaln.json
```

`scripts/run_train.sh <config> [-- extra args]` wraps the single-GPU command with the conda environment activated, the GPU with the most free memory selected, and anything after `--` forwarded to the training module.

`--output_dir`, `--batch_size`, `--num_workers` and `--resume <checkpoint.pt>` override the config from the command line. Resuming restores model, EMA, optimizer, LR-scheduler and step counter, so a run continues exactly where it stopped.

Each run writes to `outputs/<experiment name>/`:

| Path | Content |
|------|---------|
| `config.json` | Resolved config, including the auto-computed `max_joints` / `max_depth`; inference reads it back to rebuild the model |
| `dataset_stats.npy` | Normalization statistics, reused at inference |
| `checkpoints/checkpoint_step_*.pt` | Model, EMA, optimizer and LR-scheduler state, every `training.save_interval` steps |
| `debug/` | Sample visualizations, rendered once before training and at every checkpoint (EMA weights, `sampling.cfg_scale`) |
| `logs/` | TensorBoard scalars (`tensorboard --logdir outputs/<experiment name>/logs`) |

<details>
<summary><b>What a training step does</b></summary>

Clips are drawn by a power-law-balanced sampler when `training.balanced` is set — a type with `n` clips is sampled in proportion to `n^(1-sampler_alpha)`, so at the default `sampler_alpha = 0.5` a species with 100 clips is seen ten times as often as one with a single clip rather than a hundred times — then augmented on the fly — joint addition, leaf removal, chain pooling and per-bone length perturbation (`dataset.use_*_aug`) — so the model sees more topologies than the data literally contains. Every clip is padded to `max_joints` on the joint axis and `max_motion_length` on the time axis, with masks carried alongside; nothing padded ever contributes to attention or to the loss.

Training is **flow matching** (`training.diff_model = "flow"`): the network predicts the velocity of a linear interpolant between noise and data, under a masked L2 loss plus an optional geodesic rotation term (`training.lambda_geo`). Conditioning is dropped with probability `model.cond_mask_prob` so the same weights serve the conditional and unconditional branches that classifier-free guidance interpolates at sampling time. AdamW with a cosine schedule and warmup, gradient clipping at `training.max_grad_norm`, and an EMA copy of the weights (`training.use_ema`) — the copy inference loads by default.

</details>

<details>
<summary><b>Pre-computing text embeddings</b></summary>

The text encoder (`google/flan-t5-base` by default) is fetched from the Hugging Face Hub on first use. Every run loads it once to embed all captions and joint names; pre-computing those embeddings beside the features keeps it out of the run entirely:

```bash
python -m unimate.tools.precompute_text_emb --config configs/uniml3d_60frames_graph_adaln.json
```

This writes `caption_emb_cache.npz` and `joint_emb_cache.npz` into each `dataset/features/<dataset>/` the config uses. Captions are cached per token (the sequence `cross_attn` attends; `adaln` mean-pools it), joint names as one pooled vector each, keyed by the **cleaned** joint vocabulary that stage 3 produces — the shared naming is what lets the same anatomical joint embed identically across rigs. Re-run it after regenerating captions or joint names: anything the cache misses is still encoded at load time, so a stale cache costs speed rather than correctness.

</details>

## 🎬 Inference

Given a rigged 3D asset and a text prompt, UniMate generates articulated motion for arbitrary skeletons in real time — with no per-skeleton retraining and no test-time optimization.

<div align="center">
    <img src="assets/qualitative.png" alt="Qualitative results" width="100%">
</div>

Sampling starts from the output directory of a training run (`config.json`, `dataset_stats.npy`, `checkpoints/`) — pretrained checkpoints in the same layout are coming soon. The target skeleton — T-pose and topology conditioning — is taken from the dataset, so the `dataset/features/<dataset>/` directory the model was trained on must be present.

```bash
python -m unimate.inference.sample \
    --exp_dir outputs/uniml3d_60frames_graph_adaln \
    --test_cases_json test_cases.json \
    --num_repetitions 3
```

<details>
<summary><b>Test cases, flags and outputs</b></summary>

Test cases are a JSON map from `<object_type>-<case_id>` to a prompt. `object_type` must exist in the dataset; `case_id` is a free-form tag that names the output files:

```json
{
  "Dog-walk": "a dog walks forward at a steady pace",
  "Dragon-takeoff": "a dragon flaps its wings and takes off"
}
```

`--test_cases_json` is itself optional: without it every clip of the dataset's eval split is enumerated as a test case (falling back to unique `(object_type, caption)` pairs on train when there is no eval split). `--test_cases_txt` (one `object_type` per line) drives unconditional sampling, which requires `--cfg_scale 1.0`.

| Flag | Effect |
|------|--------|
| `--cfg_scale` | Classifier-free guidance scale (>= 1.0); defaults to the value saved in the run's config |
| `--model_path` | A specific checkpoint; defaults to the latest step |
| `--output_dir` | Defaults to `<exp_dir>/samples` |
| `--num_repetitions` | Samples generated per test case |
| `--batch_size` | Per-chunk inference batch; caps GPU memory regardless of how many cases there are |
| `--seed` | Fixes the sampling noise |
| `--only_save_motion` | Skip the MP4 renders, write only the `.npy` features |
| `--save_ric` | Add the RIC-recovered render beside the FK one |

Each run writes:

| Path | Content |
|------|---------|
| `motions/<case_id>-rep_<r>-<i>.npy` | Generated motion features `(T, J, 12)`, one file per repetition |
| `animations/<case_id>-rep_<r>-sample<i>_fk.mp4` | Skeleton render of each sample (`_ric.mp4` variants with `--save_ric`) |
| `animations/<object_type>_tpos.png` | The conditioning T-pose |
| `captions.json` | Prompt used for every saved `.npy` |

`scripts/run_sample_motion_text.sh <exp_dir> [test_cases_json] [cfg_scale]` runs the command above with the conda environment activated, the GPU with the most free memory selected, and an output directory named after the test-case file. Run it with `-h` for its options.

</details>

<details>
<summary><b>Driving a rigged mesh</b></summary>

To animate the original mesh with a generated motion, hand the `.npy` files to stage 5 of the data pipeline, which exports an animated GLB + FBX:

```bash
bash scripts/run_animate_motion.sh objaverse \
    outputs/uniml3d_60frames_graph_adaln/samples/motions/Dog-walk-rep_0-0.npy \
    outputs/animated
```

It accepts several files or a directory, and reads the rig from `dataset/features/<dataset>/cond.npy` by default.

</details>

## 🎨 Applications

The same trained model does three more tasks with no extra training. Each is replacement-style sampling: part of the motion is pinned to a known signal and the flow ODE denoises only the rest at every step, so the constraint holds exactly rather than being encouraged by a loss.

### Motion in-betweening

Hold chosen keyframes at their ground truth and generate the transitions between them.

<div align="center">
    <img src="assets/motion-in-betweening.png" alt="Motion in-betweening" width="100%">
</div>

<details>
<summary><b>How to run it</b></summary>

`--keep_frames` takes signed indices (negatives count back from the generation window), so `"0,-1"` fills in everything between a clip's first and last pose.

```json
{ "mixamo-Squat-000": "A human squats and then rises back up" }
```

```bash
KEEP_FRAMES="0,-1" bash scripts/run_sample_motion_inbetween.sh \
    outputs/uniml3d_60frames_graph_adaln cases.json
```

</details>

### Text-guided motion editing

Hold chosen joints at their ground-truth motion for every frame and regenerate the rest under a new prompt — keep what should stay, re-animate the rest.

<div align="center">
    <img src="assets/motion-editing.png" alt="Text-guided motion editing" width="100%">
</div>

<details>
<summary><b>How to run it</b></summary>

`--keep_joints` matches case-insensitively against either the rig's own bone names or the cleaned vocabulary.

```json
{ "<objaverse_uid>-turn-head-000": "The robot walks forward." }
```

```bash
KEEP_JOINTS="Hips,Spine,Neck,Head" bash scripts/run_sample_motion_edit.sh \
    outputs/uniml3d_60frames_graph_adaln cases.json
```

</details>

### Motion expansion

Chain several prompts into one long motion. The first segment is generated freely; every later one pins its first few frames to the previous segment's tail, and the segments are stitched at the seam.

<div align="center">
    <img src="assets/motion-expansion.png" alt="Motion expansion" width="100%">
</div>

<details>
<summary><b>How to run it</b></summary>

Test-case values become *lists* of prompts, one per segment; `--expand_overlap` sets how many frames consecutive segments share.

```json
{ "mixamo-sequence": ["A human stands up.", "A human walks forward.", "A human turns around in place."] }
```

```bash
EXPAND_OVERLAP=10 bash scripts/run_sample_motion_expand.sh \
    outputs/uniml3d_60frames_graph_adaln cases.json
```

</details>

<details>
<summary><b>Shared behaviour</b></summary>

Each mode writes into its own subdirectory of `--output_dir` (`inbetween/`, `motion_edit/`, `motion_expand/`) alongside a small JSON recording the constraint that produced it, and each has a wrapper in `scripts/` — run any of them with `-h` for the full option list.

In-betweening and editing clamp against a real clip, so their test-case keys must be `<object_type>-<clip_id>` naming a clip the dataset actually holds; that clip's motion is saved beside the result as `<case_id>-gt_rep_<r>-<i>.npy` for side-by-side comparison. `--gt_start_frame` pins which window of the clip is used instead of a random one. Editing trims both the sample and the GT to the clip's true length, while in-betweening generates the full window and trims only the GT — so align the two on frame 0 rather than assuming equal lengths. All three modes need `--cfg_scale > 1.0` and are mutually exclusive with each other.

</details>

## 📝 Citation

If you find UniMate useful in your research, please consider citing our work:

```bibtex
@inproceedings{mou2026unimate,
  title         = {UniMate: One Unified Model to Animate Diverse Skeletons},
  author        = {Mou, Linzhan and Lei, Jiahui and Dou, Zhiyang and Cai, Chenyue and Song, Chaoyue and Finkelstein, Adam and Rusinkiewicz, Szymon},
  booktitle     = {SIGGRAPH Asia 2026 Conference Papers},
  year          = {2026},
  eprint        = {2609.05415},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2609.05415}
}
```

## ⚖️ License

The code in this repository is released under the [MIT License](LICENSE).

The datasets remain governed by the licenses of their original sources: the [Mixamo](https://www.mixamo.com/) assets by Adobe's Mixamo terms of use, the [Objaverse-XL](https://objaverse.allenai.org/) assets by the license attached to each original object, and the Truebones ZOO motions by [Truebones](https://truebones.com)' commercial license. Please review and comply with the respective source licenses before using the data.
