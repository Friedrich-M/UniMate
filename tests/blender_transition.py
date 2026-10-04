"""Verify corrected output remains deformation-only when baked in Blender."""
import sys,json
from pathlib import Path
import bpy
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"addon"),str(ROOT/"tests")]
import unimate_motion
from unimate_motion.rig import apply_result
from build_scene import build
unimate_motion.register()
human,_=build()
folder=ROOT/"tests/artifacts/seated-transition"
request=json.loads((folder/"request.json").read_text(encoding="utf-8"))
s=bpy.context.scene.unimate_motion
assert s.transition_frames==12 and s.pose_approach_frames==60
a=apply_result(human,request["skeleton"],folder/"motion.npz",11,bpy.context.scene)
maximum=0.
for frame in np.arange(70,130.1,.5):
    whole=int(frame)
    bpy.context.scene.frame_set(whole, subframe=float(frame-whole))
    for b in human.pose.bones:
        if b.parent:
            maximum=max(maximum,b.location.length)
            assert b.location.length<1e-5, b.name
        assert np.allclose(b.scale,(1,1,1),atol=1e-5)
report=dict(passed=["transition controls registered","Action bake","subframe child translations remain zero",
                    "subframe scales remain one"],max_child_translation=maximum)
(ROOT/"tests/artifacts/transition-blender.json").write_text(json.dumps(report,indent=2))
print("TRANSITION_BLENDER_PASSED",json.dumps(report))
unimate_motion.unregister()
