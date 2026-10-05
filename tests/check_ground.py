"""Ground contact regression tests using a small articulated fixture."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"backend"))
from ground import Surface,plant,profiles_for,foot_capsules,sole_points,stabilize_feet,preserve_bend
from timeline import forward_kinematics,to_local
skeleton=dict(parents=[-1,0,1,2,3],
 heads=[[0,0,1],[.1,0,1],[.1,0,.6],[.1,0,.15],[.1,-.2,.1]],
 bone_names=["hips","thigh.left","shin.left","foot.left",None],
 labels=["hips","thigh left","shin left","foot left","foot end"],
 rest_matrices=[np.eye(4).tolist() for _ in range(4)]+[None],
 collision_capsules=[dict(joint=3,a=[.1,-.03,.13],b=[.1,-.17,.1],radius=.08)],
 foot_profiles=[dict(joint=3,parent=2,upper=1,leg_length=.85,stance_tilt=28,swing_tilt=50)])
frames=40
local=np.tile(np.eye(3),(frames,5,1,1))
root=np.column_stack([np.linspace(0,.23,frames),np.zeros(frames),np.ones(frames)])
positions,rotations=forward_kinematics(root,local,skeleton)
ground=dict(normal=[0,0,1],height=0.,triangles=[])
corrected,orient,report=plant(positions,rotations,skeleton,ground)
assert report["active_frames"]>=25
assert report["median_planted_step"]<report["median_planted_step_before"]*.25
assert np.max(np.abs(corrected[:,0,:2]-positions[:,0,:2]))<1e-9
reconstructed,_=forward_kinematics(corrected[:,0],to_local(orient,skeleton["parents"]),skeleton)
assert np.allclose(reconstructed,corrected,atol=1e-8)
# A tilted static mesh returns its interpolated height and normal.
triangles=[
 [[0,0,0],[1,0,.1],[1,1,.1]],
 [[0,0,0],[1,1,.1],[0,1,0]]]
surface=Surface(dict(normal=[0,0,1],height=0.,triangles=triangles),skeleton)
height,normal,hit=surface.sample([.5,.5,.3])
assert hit and abs(height-.05)<1e-8 and normal[2]>.9
_,_,hit=surface.sample([3,3,0])
assert not hit and surface.misses==1
# The same foot may bend further in a swing than while planted.
tilt=np.radians(70)
turn=Rotation.from_rotvec([tilt,0,0]).as_matrix()
twist=np.tile(np.eye(3),(2,5,1,1))
twist[:,3]=turn
posed,rotated=forward_kinematics(np.array([[0,0,1],[0,0,1]]),twist,skeleton)
foot=foot_capsules(skeleton,profiles_for(skeleton))
a,b,limits=stabilize_feet(posed,rotated,skeleton,foot,Surface(ground,skeleton),[[(0,1)]])
assert limits[0]["stance_limit"]==28 and limits[0]["swing_limit"]==50
assert np.allclose(a[:,0],posed[:,0])
# A planted foot may not pull a turning knee through a straight-leg reversal.
bend_rig=dict(skeleton)
bend_rig["heads"]=[[0,0,1],[.1,0,1],[.1,-.15,.6],[.1,0,.2],[.1,-.2,.15]]
source_local=np.tile(np.eye(3),(1,5,1,1))
source_p,source_r=forward_kinematics(np.array([[0,0,1]]),source_local,bend_rig)
flipped_local=source_local.copy()
flipped_local[0,1]=Rotation.from_euler("z",180,degrees=True).as_matrix()
flipped_p,flipped_r=forward_kinematics(np.array([[0,0,1]]),flipped_local,bend_rig)
restored_p,restored_r,bend_counts=preserve_bend(flipped_p,flipped_r,source_p,bend_rig,[dict(profile=bend_rig["foot_profiles"][0])])
assert bend_counts["foot.left"]==1
assert restored_p[0,2,1]<0 and np.allclose(restored_p[0,3],flipped_p[0,3],atol=1e-7)
rebuilt,_=forward_kinematics(restored_p[:,0],to_local(restored_r,bend_rig["parents"]),bend_rig)
assert np.allclose(rebuilt,restored_p,atol=1e-7)
# A knee swivelling gradually away from the generated side is held there on
# every frame; a threshold-based fix popped it back on single frames.
steps=30
swing_local=np.tile(np.eye(3),(steps,5,1,1))
for t in range(steps):
    swing_local[t,1]=Rotation.from_euler("z",4*t,degrees=True).as_matrix()
source_many=np.repeat(source_p,steps,axis=0)
swung_p,swung_r=forward_kinematics(np.tile([0,0,1.],(steps,1)),swing_local,bend_rig)
held_p,_,_=preserve_bend(swung_p,swung_r,source_many,bend_rig,[dict(profile=bend_rig["foot_profiles"][0])])
knee_jerk=np.linalg.norm(np.diff(held_p[:,2],2,axis=0),axis=1).max()
assert knee_jerk<.01, knee_jerk
assert held_p[:,2,1].max()<0, "Knee left the generated side"
result=dict(passed=["swivelling knee held smoothly","grounded support reduces slide","root path preserved","bone lengths preserved",
                    "inclined mesh height and normal","outside-mesh fallback","phase-aware limits",
                    "planted knee keeps generated bend side"],
            planted_step_before=report["median_planted_step_before"],
            planted_step_after=report["median_planted_step"])
(ROOT/"tests/artifacts/ground-regressions.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result))
