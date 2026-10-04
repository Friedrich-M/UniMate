"""Run motion cleanup on fixtures; writes tests/artifacts/collision-* for later tests."""
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'tests/artifacts'
sys.path.insert(0,str(ROOT/'backend'))
from collision import cleanup
for label,folder in [('Human',ART/'seated-transition'),('Creature',ROOT/'tests/fixtures/creature')]:
    request=json.loads((folder/'request.json').read_text(encoding='utf-8'))
    skeleton=json.loads((ROOT/'tests/fixtures'/(label.lower()+'-collision-skeleton.json')).read_text(encoding='utf-8'))
    request['skeleton']=skeleton
    with np.load(folder/'motion.npz') as data:
        output={k:data[k] for k in data.files}
    t=time.time()
    pos,rot,report=cleanup(output['positions'],output['rotations'],skeleton)
    dest=ART/('collision-'+label.lower())
    dest.mkdir(exist_ok=True)
    request['motion_cleanup']=True
    (dest/'request.json').write_text(json.dumps(request,indent=2),encoding='utf-8')
    (dest/'collision-report.json').write_text(json.dumps(report,indent=2))
    output.update(positions=pos,rotations=rot,collision_report_json=json.dumps(report),
                  cleanup_source=(folder/'motion.npz').name)
    np.savez_compressed(dest/'motion.npz',**output)
    print(label,"cleanup seconds",round(time.time()-t,2),"max penetration after",report["max_penetration_after"],flush=True)
