"""Check cleanup against evaluated Blender mesh surfaces and deformation transforms."""
import sys,json,os
from pathlib import Path
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"addon"),str(ROOT/"tests")]
import unimate_motion
from unimate_motion.rig import apply_result
from build_scene import build
unimate_motion.register()
human,creature=build()
report={}
for label,rig,start,default_frames in [("human",human,11,range(70,131)),("creature",creature,1,range(1,61))]:
    folder=Path(os.environ.get("UNIMATE_"+label.upper()+"_JOB",str(ROOT/"tests/artifacts"/("collision-"+label))))
    frames=range(start,start+len(np.load(folder/"motion.npz",allow_pickle=False)["positions"])) if os.environ.get("UNIMATE_"+label.upper()+"_JOB") else default_frames
    request=json.loads((folder/"request.json").read_text(encoding="utf-8"))
    skeleton=request["skeleton"]
    apply_result(rig,skeleton,folder/"motion.npz",start,bpy.context.scene)
    parents=skeleton["parents"]
    ancestors=[]
    for j in range(len(parents)):
        chain=[]
        while j>=0:
            chain.append(j);j=parents[j]
        ancestors.append(chain)
    objects=[]
    for obj in bpy.context.scene.objects:
        if obj.type=="MESH" and obj.find_armature()==rig:
            names=[g.name for g in obj.vertex_groups if g.name in skeleton["bone_names"]]
            assert len(names)==1,"Fixture should have one weighted bone per mesh"
            objects.append((obj,skeleton["bone_names"].index(names[0])))
    eligible=[]
    for i,(obj,a) in enumerate(objects):
        for k in range(i+1,len(objects)):
            b=objects[k][1]
            lca=next(j for j in ancestors[a] if j in ancestors[b])
            if ancestors[a].index(lca)+ancestors[b].index(lca)>2:
                eligible.append((i,k))
    overlaps=[]
    max_translation=0.
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        depsgraph=bpy.context.evaluated_depsgraph_get()
        trees=[]
        for obj,j in objects:
            evaluated=obj.evaluated_get(depsgraph)
            mesh=evaluated.to_mesh()
            trees.append(BVHTree.FromPolygons([evaluated.matrix_world@v.co for v in mesh.vertices],
                                            [p.vertices[:] for p in mesh.polygons]))
            evaluated.to_mesh_clear()
        for i,k in eligible:
            if trees[i].overlap(trees[k]):
                overlaps.append(dict(frame=frame,bones=[skeleton["bone_names"][objects[x][1]] for x in (i,k)]))
        for b in rig.pose.bones:
            if b.parent:
                max_translation=max(max_translation,b.location.length)
            assert np.allclose(b.scale,(1,1,1),atol=1e-5)
    report[label]=dict(frames_checked=len(frames),surface_intersections=overlaps,
                       max_child_translation=max_translation)
    assert not overlaps, f"{label}: remaining mesh intersections: {overlaps[:3]}"
    assert max_translation < 1e-5, "Cleanup stretched a joint"
    print(label,json.dumps(report[label]),flush=True)
Path(os.environ.get("UNIMATE_CLEANUP_REPORT",str(ROOT/"tests/artifacts/cleanup-blender.json"))).write_text(json.dumps(report,indent=2))
unimate_motion.unregister()
