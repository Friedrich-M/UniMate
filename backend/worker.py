"""Run UniMate on an exported Blender skeleton, without loading its dataset."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
# Upstream Friedrich-M/UniMate commit merged into unimate/; update after each upstream merge.
UPSTREAM_COMMIT = "2c5b384715aa63d8639b1ed7eb74bfe614570c7a"

def load_geometry():
    spec = importlib.util.spec_from_file_location("unimate_geometry", ROOT / "addon" / "unimate_motion" / "motion.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def write_status(path, state, message):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps({"state": state, "message": message}), encoding="utf-8")
    temp.replace(path)
    print(message, flush=True)

def build_condition(request, config, encoder, stats, device):
    import numpy as np
    import torch
    from unimate.dataset.mixture.collate import mixture_batch_collate
    from unimate.dataset.transforms import build_parent_features
    from unimate.utils.topology_utils import (
        compute_edge_indexs, compute_joint_depths, compute_edge_relations_and_distances,
        compute_laplacian_eigenvectors,
    )
    geometry = load_geometry()
    skeleton = request["skeleton"]
    if skeleton["signature"] != geometry.signature(skeleton):
        raise ValueError("Skeleton signature is invalid.")
    canonical = geometry.canonicalize(skeleton)
    parents = canonical["parents"]
    joints = len(parents)
    if not config.dataset.min_joints <= joints <= config.dataset.max_joints:
        raise ValueError(f"Checkpoint supports {config.dataset.min_joints}–{config.dataset.max_joints} joints; rig exports {joints}.")
    if config.dataset.topology_condition_type != "tpos" or config.dataset.feature_len != 12:
        raise ValueError("This adapter requires UniMate's 12-feature rest-pose conditioning.")
    family = request["stats_family"]
    if family not in stats:
        raise ValueError(f"Checkpoint has no {family!r} statistics; available: {list(stats)}")
    selected = stats[family]
    mean = np.tile(selected["mean_local"], (joints, 1)).astype(np.float32)
    std = np.tile(selected["std_local"], (joints, 1)).astype(np.float32)
    mean[0], std[0] = selected["mean_root"], selected["std_root"]
    if mean.shape != (joints, 12) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid checkpoint normalization statistics.")
    tpose = np.zeros((joints, 12), dtype=np.float32)
    tpose[:, :3] = canonical["positions"]
    tpose[:, 3:9] = [1, 0, 0, 0, 1, 0]
    tpose = (tpose - mean) / std
    with torch.no_grad():
        tokens = encoder.tokenize([request["prompt"]] + skeleton["labels"])
        hidden = encoder(tokens).float()
    mask = tokens["attention_mask"].bool()
    embeddings = [(h[m].cpu().numpy() if m.any() else np.zeros((1, h.shape[-1]), np.float32))
                  for h, m in zip(hidden, mask)]
    relations, distances = compute_edge_relations_and_distances(parents, max_path_len=5)
    spectral, _ = compute_laplacian_eigenvectors(parents, max_freqs=config.model.max_freqs)
    length = config.dataset.max_motion_length
    batch = dict(
        motion=np.zeros((length, joints, 12), np.float32),
        motion_length=length, max_motion_length=length, max_joints=config.dataset.max_joints,
        start_idx=0, parents=parents, edge_indexs=compute_edge_indexs(parents),
        tpos_first_frame=tpose, offsets=canonical["offsets"],
        joint_graph_dist=distances, joint_relations=relations, joint_depths=compute_joint_depths(parents),
        spectral_feats=spectral, joint_names_emb=np.stack([e.mean(0) for e in embeddings[1:]]),
        mean=mean, std=std, caption=request["prompt"],
        caption_emb=embeddings[0].mean(0), caption_tokens=embeddings[0], object_type="blender",
        **build_parent_features(tpose, parents),
    )
    motion, cond = mixture_batch_collate([batch])
    cond = {k: (v.to(device) if torch.is_tensor(v) else
                [i.to(device) if torch.is_tensor(i) else i for i in v] if isinstance(v, list) else v)
            for k, v in cond.items()}
    return motion.shape, cond, canonical, mean, std

_loaded = {}

def load_models(exp, config, checkpoint, device, status):
    """Load the text encoder, model and transport; reuse them while serving."""
    import torch
    from unimate.models.factory import create_model, create_transport
    from unimate.models.text_encoder.factory import create_text_encoder
    from unimate.training.ema import EMAModel
    key = (str(exp), str(checkpoint), str(device))
    if _loaded.get("key") == key:
        return _loaded["models"]
    _loaded.clear()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    write_status(status, "running", f"Loading text encoder on {device} (first run may download it)")
    encoder = create_text_encoder(
        encoder_type=config.model.text_encoder_type,
        encoder_version=config.model.text_encoder_version, device=str(device), pool=False)
    write_status(status, "running", "Loading UniMate model and EMA weights")
    model = create_model(config.dataset, config.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state), strict=True)
    if config.training.use_ema and "ema_state_dict" in state:
        ema = EMAModel(model.parameters(), decay=config.training.ema_decay, use_ema_warmup=True)
        ema.load_state_dict(state["ema_state_dict"])
        ema.copy_to(model.parameters())
        del ema
    del state
    model.to(device).eval()
    models = (encoder, model, create_transport(training_config=config.training))
    _loaded.update(key=key, models=models)
    return models

def generate(request, output, status):
    started = time.perf_counter()
    import numpy as np
    import torch
    from unimate.configs.schema import MainConfig
    from unimate.models.flow.transport import Sampler
    from unimate.inference.generate import generate_samples
    from timeline import constraints, plan_windows, retime
    if request.get("schema") != 1 or not request["prompt"].strip():
        raise ValueError("A valid request and a non-empty prompt are required.")
    exp = Path(request["experiment"]).resolve()
    config = MainConfig.from_json(exp / "config.json")
    if config.training.diff_model != "flow":
        raise ValueError("This prototype supports the released flow-matching checkpoints.")
    frames = int(request["frames"])
    if not 2 <= frames <= config.dataset.max_motion_length:
        raise ValueError(f"Choose 2–{config.dataset.max_motion_length} frames.")
    cfg_scale = float(request["guidance"])
    if not np.isfinite(cfg_scale) or not 1 < cfg_scale <= 20:
        raise ValueError("Text guidance must be greater than 1 and at most 20.")
    seed = int(request["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    # Only load statistics/weights from a trusted model release.
    stats = np.load(exp / "dataset_stats.npy", allow_pickle=True).item()
    checkpoint = Path(request.get("checkpoint", "")) if request.get("checkpoint") else max(
        (exp / "checkpoints").glob("checkpoint_step_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]),
        default=None)
    if checkpoint is None or not checkpoint.is_file():
        raise ValueError("No model checkpoint found in the experiment directory.")
    clips = request.get("clips")
    if clips:
        spec = importlib.util.spec_from_file_location("schedule", ROOT / "addon" / "unimate_motion" / "schedule.py")
        schedule = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(schedule)
        schedule.validate_clips(clips, request["skeleton"]["signature"])
    else:
        clips = [dict(prompt=request["prompt"], start=1, end=frames, references=[])]
    encoder, model, diffusion = load_models(exp, config, checkpoint, device, status)
    conditions = []
    for clip in clips:
        clip_request = dict(request, prompt=clip["prompt"])
        shape, cond, canonical, mean, std = build_condition(clip_request, config, encoder, stats, device)
        conditions.append(cond)
    overlap = int(request.get("overlap", 10))
    if not 1 <= overlap < shape[-1]-1:
        raise ValueError("Transition context must be smaller than the model window.")
    if request.get("clips"):
        plan = plan_windows(clips, shape[-1], overlap, request.get("extend_clips", True))
    else:
        plan = [(0, {})]
    previous = None
    parts, spans = [], [None] * len(clips)
    total = 0
    with torch.inference_mode():
        for index, (clip_index, slots) in enumerate(plan):
            write_status(status, "running", f"Generating window {index+1}/{len(plan)} "
                         f"(prompt {clip_index+1}/{len(clips)}) on {device}")
            known, mask = constraints(previous, slots, shape, overlap, mean, std, device)
            samples = generate_samples(
                model, conditions[clip_index], shape, "flow", diffusion, Sampler(diffusion),
                device=device, cfg_scale=cfg_scale, x1_known=known, keep_mask=mask)
            previous = samples
            part = samples if index == 0 else samples[..., overlap:]
            parts.append(part)
            begin = spans[clip_index][0] if spans[clip_index] else total
            total += part.shape[-1]
            spans[clip_index] = (begin, total)
    samples = torch.cat(parts, dim=-1)
    joints = len(canonical["parents"])
    features = samples[0, :joints].permute(2, 0, 1).cpu().numpy()
    features = features * std[None] + mean[None]
    if not request.get("clips"):
        features = features[:frames]
    positions, rotations = load_geometry().decode_features(features, canonical)
    if request.get("smoothing", 0) > 0:
        from timeline import smooth_motion
        positions, rotations = smooth_motion(positions, rotations, request["skeleton"], float(request["smoothing"]))
    if request.get("clips"):
        positions, rotations = retime(positions, rotations, spans, clips, request["skeleton"],
                                     request.get("transition_frames", 12),
                                     request.get("pose_approach_frames", 60))
    collision_report = {}
    if request.get("motion_cleanup", True):
        from collision import cleanup
        write_status(status, "running", "Checking self-collisions, ground contact and joint limits")
        positions, rotations, collision_report = cleanup(
            positions, rotations, request["skeleton"], request.get("ground"))
    temp = output.with_suffix(".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, schema=1, positions=positions, rotations=rotations,
            collision_report_json=json.dumps(collision_report),
            postprocess_json=json.dumps({"method": "local FK retiming, inertial joins, eased reference approaches", "transition_frames": request.get("transition_frames", 12), "pose_approach_frames": request.get("pose_approach_frames", 60)}) if request.get("clips") else "{}",
            features=features, schedule_json=json.dumps(request.get("clips", [])),
            windows_json=json.dumps([dict(clip=c, reference_slots=sorted(s)) for c, s in plan]),
            seconds=time.perf_counter()-started, signature=request["skeleton"]["signature"],
            joint_names=np.asarray(request["skeleton"]["joint_names"]), fps=request["fps"],
            prompt=request["prompt"], seed=seed, checkpoint=str(checkpoint),
            upstream_commit=UPSTREAM_COMMIT)
    temp.replace(output)
    message = "Motion ready — Apply Motion to create an Action"
    warnings=[]
    if collision_report.get("frames_with_remaining_contacts", 0):
        warnings.append(f"{collision_report['frames_with_remaining_contacts']} collision frames")
    if collision_report.get("ground_contact",{}).get("unresolved_contact_frames",0):
        warnings.append(f"{collision_report['ground_contact']['unresolved_contact_frames']} uncertain contact frames")
    if warnings:
        message += " (review: " + ", ".join(warnings) + ")"
    write_status(status, "complete", message)

def run_job(request_path, output, status):
    try:
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        generate(request, Path(output), Path(status))
    except Exception as exc:
        traceback.print_exc()
        write_status(Path(status), "failed", f"{type(exc).__name__}: {exc}")
        return 1
    return 0

def serve():
    """Keep models loaded and run one job per stdin line: JSON with request,
    output, status and log paths. Exits at stdin EOF, when Blender closes,
    unloads the model or reaches its idle timeout.

    stdin is read only between jobs: on Windows, a read pending in another
    thread blocks handle operations such as process creation during a job."""
    import contextlib
    print("UniMate worker serving", flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        job = json.loads(line)
        with open(job["log"], "w", encoding="utf-8") as log, \
                contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            run_job(job["request"], job["output"], job["status"])
        print("Finished", job["status"], flush=True)
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--serve", action="store_true", help="Run jobs from stdin and keep models loaded")
    args = parser.parse_args()
    if args.serve:
        return serve()
    if not (args.request and args.output and args.status):
        parser.error("--request, --output and --status are required without --serve")
    return run_job(args.request, args.output, args.status)

if __name__ == "__main__":
    raise SystemExit(main())
