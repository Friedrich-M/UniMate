"""Reapply temporal editing to saved neural features without running inference."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"backend"))
from timeline import retime
spec = importlib.util.spec_from_file_location("geometry", ROOT/"addon/unimate_motion/motion.py")
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)
source = ROOT/"tests/fixtures/seated-reference"
dest = ROOT/"tests/artifacts/seated-transition"
dest.mkdir(exist_ok=True, parents=True)
request = json.loads((source/"request.json").read_text(encoding="utf-8"))
request.update(transition_frames=12, pose_approach_frames=60)
(dest/"request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
with np.load(source/"motion.npz") as old:
    data = {k: old[k] for k in old.files}
positions, rotations = geometry.decode_features(data["features"], geometry.canonicalize(request["skeleton"]))
spans, total = [], 0
for i, clip in enumerate(request["clips"]):
    count = 60 if i == 0 else 60-request["overlap"]
    spans.append((total, total+count))
    total += count
pos, rot = retime(positions, rotations, spans, request["clips"], request["skeleton"], 12, 60)
def metrics(p, r):
    angular = np.degrees(Rotation.from_matrix((r[:-1].swapaxes(-1,-2)@r[1:]).reshape(-1,3,3)).magnitude()).reshape(len(r)-1,-1)
    delta = np.linalg.norm(np.diff(p[:,0],axis=0),axis=-1)
    return dict(seam_degrees=float(angular[59].max()), endpoint_degrees=float(angular[-1].max()),
                approach_max_degrees=float(angular[59:].max()), approach_root_step=float(delta[59:].max()),
                minimum_height=float(p[60:,:,2].min()))
report={"before":metrics(data["positions"],data["rotations"]),"after":metrics(pos,rot)}
data.update(positions=pos, rotations=rot,
            postprocess_json=json.dumps({"method":"local FK retiming, inertial joins, eased reference approaches",
                                         "transition_frames":12,"pose_approach_frames":60,
                                         "neural_features_source":"seated-reference/motion.npz"}))
np.savez_compressed(dest/"motion.npz", **data)
(dest/"status.json").write_text(json.dumps({"state":"complete","message":"Motion ready - transition corrected"}))
(ROOT/"tests/artifacts/transition-tests.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
