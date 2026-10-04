"""Check Blender export of inferred support bones and a selected static ground."""
import sys,json
from pathlib import Path
import bpy
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"addon"),str(ROOT/"tests")]
import unimate_motion
from unimate_motion.rig import export_skeleton,export_ground
from build_scene import build
unimate_motion.register()
human,creature=build()
bpy.ops.mesh.primitive_plane_add(size=10, location=(0,0,0))
floor=bpy.context.object
floor.name="Ground export fixture"
report={}
for rig,count in [(human,2),(creature,4)]:
    skeleton=export_skeleton(rig)
    assert len(skeleton["foot_profiles"])==count
    ground=export_ground(rig,skeleton,floor)
    assert len(ground["triangles"])==2
    assert ground["object"]==floor.name
    report[rig.name]=dict(feet=count,ground_triangles=len(ground["triangles"]),
                          rest_sole=export_ground(rig,skeleton)["height"])
creature.data.bones["front_paw.left"]["unimate_stance_tilt_deg"]=21.
assert next(p for p in export_skeleton(creature)["foot_profiles"]
            if creature.data.bones["front_paw.left"].name ==
               export_skeleton(creature)["bone_names"][p["joint"]])["stance_tilt"]==21.
(ROOT/"tests/artifacts/ground-blender.json").write_text(json.dumps(report,indent=2))
print("GROUND_BLENDER_PASSED",json.dumps(report))
unimate_motion.unregister()
