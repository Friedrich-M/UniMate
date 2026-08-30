"""Per-asset export completion markers (resume bookkeeping).

Skeleton pruning is joint across all of an asset's clips, so an asset only
completes atomically. A marker under ``<output_dir>/.completed/`` records
that — including assets that legitimately produce no clips
(``status="skipped"``), so reruns don't reload them. Delete a marker (or the
directory) to force a re-export, e.g. after changing joint-count limits.

Each marker also carries the asset's pruned joint names, which makes the
canonical ``joint_names.json`` rebuildable from disk at any time. That
matters because markers are durable and written per asset, while the
in-memory joint-name dict is only flushed at the very end of a run: without
the marker copy, a worker killed by a Slurm walltime or an OOM leaves
markers that suppress re-export but no joint names, losing those entries
permanently.

Deliberately free of any ``bpy`` import so plain-Python tools (notably
:mod:`data_process.tools.merge_summaries`) can use it outside Blender.
"""

import json
import os


def _asset_marker_path(output_dir, save_name):
    return os.path.join(output_dir, ".completed", f"{save_name}.json")


def is_asset_complete(output_dir, save_name):
    """True when a completion marker exists for *save_name*."""
    return os.path.isfile(_asset_marker_path(output_dir, save_name))


def mark_asset_complete(output_dir, save_name, status="exported", n_clips=None,
                        reason="", joint_names=None):
    """Write the completion marker for one fully-processed asset.

    Written atomically — a half-written marker would otherwise be
    indistinguishable from a complete one on the next run.
    """
    path = _asset_marker_path(output_dir, save_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"status": status}
    if n_clips is not None:
        payload["n_clips"] = int(n_clips)
    if reason:
        payload["reason"] = reason
    if joint_names is not None:
        payload["joint_names"] = [str(n) for n in joint_names]
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def collect_joint_names_from_markers(output_dir):
    """Rebuild ``{save_name: joint_names}`` from the per-asset markers.

    Only markers that actually recorded joint names contribute; assets that
    produced no clips (status ``"skipped"``) never do.
    """
    marker_dir = os.path.join(output_dir, ".completed")
    if not os.path.isdir(marker_dir):
        return {}

    recovered = {}
    for fname in sorted(os.listdir(marker_dir)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(marker_dir, fname)) as f:
                payload = json.load(f)
        except (OSError, ValueError):
            continue
        names = payload.get("joint_names")
        if names:
            recovered[os.path.splitext(fname)[0]] = names
    return recovered


def merge_joint_names(output_dir, all_joint_names, log=None):
    """Union this run's joint names over the markers and the existing file.

    Precedence (lowest to highest): the canonical ``joint_names.json`` on
    disk, the per-asset completion markers, then *all_joint_names* from the
    current run.
    """
    merged = {}

    canonical = os.path.join(output_dir, "joint_names.json")
    if os.path.isfile(canonical):
        try:
            with open(canonical) as f:
                merged.update(json.load(f))
        except (OSError, ValueError) as exc:
            if log:
                log(f"Could not read {canonical} ({exc}); rebuilding from markers only.")

    from_markers = collect_joint_names_from_markers(output_dir)
    new_from_markers = len(set(from_markers) - set(merged))
    merged.update(from_markers)
    if new_from_markers and log:
        log(f"Recovered {new_from_markers} joint-name entries from completion markers")

    merged.update(all_joint_names)
    return merged
