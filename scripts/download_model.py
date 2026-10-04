from pathlib import Path
from huggingface_hub import hf_hub_download
ROOT = Path(__file__).resolve().parents[1]
# General model (humans, animals, objects; up to 71 joints) and the Mixamo-only
# model the add-on uses for Human rigs of up to 22 joints. Both are CC BY-NC 4.0.
MODELS = {"unimate_uniml3d_f60_v2": "checkpoint_step_100000.pt",
          "unimate_mixamo_f60": "checkpoint_step_120000.pt"}
for prefix, checkpoint in MODELS.items():
    for filename in ("config.json", "dataset_stats.npy", "checkpoints/" + checkpoint):
        path = hf_hub_download("Linzhan/UniMate", prefix + "/" + filename, local_dir=str(ROOT / "models"))
        print(path)

import urllib.request
pose = ROOT / "models" / "pose" / "pose_landmarker_full.task"
pose.parent.mkdir(parents=True, exist_ok=True)
if not pose.is_file():
    temporary = pose.with_suffix(".download")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task", temporary)
    temporary.replace(pose)
print(pose)
