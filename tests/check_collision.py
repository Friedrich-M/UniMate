
from pathlib import Path
import sys,json
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from collision import closest, solve, CapsuleRig
from timeline import forward_kinematics,to_local
# Closest points must work for intersecting, parallel and zero-length capsules.
a=np.array([[0,0,0],[0,0,0],[0,0,0]],float)
b=np.array([[2,0,0],[2,0,0],[0,0,0]],float)
c=np.array([[1,-1,0],[0,2,0],[1,0,0]],float)
d=np.array([[1,1,0],[2,2,0],[1,0,0]],float)
p,q=closest(a,b,c,d)
assert np.allclose(np.linalg.norm(p-q,axis=1),[0,2,1])
for label in ['human','creature']:
 folder=ROOT/'tests/artifacts'/('collision-'+label)
 req=json.loads((folder/'request.json').read_text(encoding='utf-8'))
 with np.load(folder/'motion.npz') as data:
  pos,rot=data['positions'],data['rotations']
  report=json.loads(str(data['collision_report_json']))
 skel=req['skeleton']
 rebuilt,_=forward_kinematics(pos[:,0],to_local(rot,skel['parents']),skel)
 assert np.allclose(rebuilt,pos,atol=1e-8)
 assert np.allclose(rot.swapaxes(-1,-2)@rot,np.eye(3),atol=1e-8)
 assert report['max_penetration_after']<=report['tolerance']
 assert report['frames_with_remaining_contacts']==0
 if label=='human':
  speed=np.degrees(Rotation.from_matrix((rot[:-1].swapaxes(-1,-2)@rot[1:]).reshape(-1,3,3)).magnitude()).reshape(len(rot)-1,-1)
  assert speed[59:].max()<10, speed[59:].max()
 else:
  j=skel['bone_names'].index('front_paw.right')
  direction=rot[-1,j]@np.asarray(skel['rest_matrices'][j])[:3,1]
  assert np.degrees(np.arcsin(direction[2]))<45.01
# Missing meshes must be a documented no-op, not an exception.
empty={'parents':[-1,0], 'heads':[[0,0,0],[0,0,1]]}
pos=np.array([empty['heads']],float);rot=np.tile(np.eye(3),(1,2,1,1))
p,q,report=solve(pos,rot,empty)
assert np.array_equal(p,pos) and report['method']=='none'
result={'passed':['segment geometry','sphere capsules','FK invariance','valid rotations',
                  'self-collision tolerance','temporal continuity','paw tilt and stance contact','no-mesh fallback']}
(ROOT/'tests/artifacts/collision-regressions.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
