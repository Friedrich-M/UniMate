import sys,json
from pathlib import Path
import numpy as np
import bpy
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"addon"),str(ROOT/"tests")]
import unimate_motion
from unimate_motion.rig import export_skeleton
from unimate_motion.motion import canonicalize,decode_features
from unimate_motion.poses import capture_pose,auto_mapping,preview_estimate
from unimate_motion.schedule import validate_clips
from build_scene import build
unimate_motion.register()
human,creature=build()
skeleton=export_skeleton(human)
mapping=auto_mapping(human)
assert all(mapping[role] for role in ("hips","left_upper_arm","left_forearm","right_thigh","right_shin"))
estimate=json.loads((ROOT/"tests/fixtures/estimated-pose.json").read_text())
skipped=preview_estimate(human,skeleton,estimate,mapping)
bpy.context.view_layer.update()
pose=capture_pose(human,skeleton)
features=np.asarray(pose["features"])[None]
positions,rotations=decode_features(features,canonicalize(skeleton))
for j,name in enumerate(skeleton["bone_names"]):
    if name:
        delta=np.array(human.pose.bones[name].matrix.to_3x3())@np.array(human.data.bones[name].matrix_local.to_3x3()).T
        assert np.allclose(delta,rotations[0,j],atol=1e-5), "Pose feature round trip "+name
        assert np.allclose(human.pose.bones[name].head,positions[0,j],atol=1e-5), "Pose position round trip "+name
clips=[dict(prompt="A human walks slowly.",start=11,end=70,references=[]),
       dict(prompt="A human sits on the ground with both knees bent and legs spread apart.",start=71,end=130,references=[dict(frame=130,pose=pose)])]
validate_clips(clips,skeleton["signature"])
for ranges in ([(1,60),(60,120)],[(1,60),(62,120)],[(1,1)]):
    try:
        validate_clips([dict(prompt="test",start=a,end=b,references=[]) for a,b in ranges],skeleton["signature"])
        raise AssertionError("Invalid ranges accepted")
    except ValueError:
        pass
s=bpy.context.scene.unimate_motion
s.rig=human
s.mode="TIMELINE"
for data in clips:
    item=s.clips.add()
    item.prompt,item.start,item.end=data["prompt"],data["start"],data["end"]
ref=s.clips[1].references.add()
ref.frame=130;ref.uid="test-pose";ref.pose_json=json.dumps(pose)
from unimate_motion.clips import collect_schedule
schedule=collect_schedule(s,skeleton,bpy.context.scene)
assert len(schedule["clips"])==2
folder=ROOT/"tests/artifacts/workflow-request"
folder.mkdir(exist_ok=True,parents=True)
request=dict(schema=1,skeleton=skeleton,prompt="Walk then match reference pose",experiment="models/unimate_uniml3d_f60_v2",stats_family="mixamo",frames=60,fps=30,seed=10,guidance=3.,clips=clips,overlap=10)
(folder/"request.json").write_text(json.dumps(request,indent=2),encoding="utf-8")
# Image changes invalidate old capture data.
ref.image_path="//reference.jpg"
ref.pose_json=json.dumps(pose)
ref.estimate_path="old-estimate.json"
ref.image_path="//changed.jpg"
assert not ref.pose_json and not ref.estimate_path
# Cancellation must reap the child process and close the log.
import subprocess,tempfile
log=(ROOT/"tests/artifacts/cancel-test.log").open("w")
process=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"],
    cwd=tempfile.gettempdir(),stdout=log,stderr=subprocess.STDOUT,
    creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
unimate_motion._job=dict(process=process,log=log,scene=bpy.context.scene,directory=ROOT/"tests/artifacts")
assert bpy.ops.unimate.cancel()=={"FINISHED"}
assert process.poll() is not None and log.closed and unimate_motion._job is None
report=dict(passed=["human anatomical mapping","real image retarget","pose feature round trip","invalid timeline ranges rejected","UI schedule serialization","image change invalidates pose","cancel reaps process"],low_confidence_roles=skipped)
(ROOT/"tests/artifacts/workflow-tests.json").write_text(json.dumps(report,indent=2))
print("UNIMATE_WORKFLOW_TESTS_PASSED",json.dumps(report))
unimate_motion.unregister()
