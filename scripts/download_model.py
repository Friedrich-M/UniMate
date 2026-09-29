from pathlib import Path
from huggingface_hub import hf_hub_download
ROOT = Path(__file__).resolve().parents[1]
PREFIX = "unimate_uniml3d_f60_v2"
for filename in ("config.json", "dataset_stats.npy", "checkpoints/checkpoint_step_100000.pt"):
    path = hf_hub_download("Linzhan/UniMate", PREFIX + "/" + filename, local_dir=str(ROOT / "models"))
    print(path)

import urllib.request
pose = ROOT / "models" / "pose" / "pose_landmarker_full.task"
pose.parent.mkdir(parents=True, exist_ok=True)
if not pose.is_file():
    temporary = pose.with_suffix(".download")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task", temporary)
    temporary.replace(pose)
print(pose)
