"""Local human pose estimation. Images are never uploaded."""
import argparse
import json
from pathlib import Path
import traceback
import sys
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")
import os
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "cache" / "matplotlib"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import mediapipe as mp
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_poses=2)
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp.Image.create_from_file(str(args.image)))
    if not result.pose_world_landmarks:
        raise ValueError("No human body detected. Use a clear full-body image.")
    if len(result.pose_world_landmarks) != 1:
        raise ValueError("Multiple people detected. Crop the image to one person.")
    world = [[p.x, p.y, p.z, p.visibility, p.presence] for p in result.pose_world_landmarks[0]]
    normalized = [[p.x, p.y, p.z, p.visibility] for p in result.pose_landmarks[0]]
    output = dict(schema=1, source=str(args.image), world=world, normalized=normalized,
                  model="MediaPipe Pose Landmarker Full", review_required=True)
    temp = args.output.with_suffix(".tmp")
    temp.write_text(json.dumps(output, indent=2), encoding="utf-8")
    temp.replace(args.output)
    print("Human pose estimated. Preview, correct, and capture it before generation.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
