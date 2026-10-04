"""Validate the saved run/jump/sword scene against its obstacle and floor."""
import json
from pathlib import Path
import bpy
from mathutils.bvhtree import BVHTree

ROOT = Path(__file__).resolve().parents[1]
scene = bpy.context.scene
rig = bpy.data.objects["UniMate Human"]
cube = bpy.data.objects["UniMate Jump Obstacle | Cube"]
ground = bpy.data.objects["UniMate Ground | contact plane"]
sword = bpy.data.objects["UniMate Sword | right hand"]
parts = [o for o in scene.objects if o.type == "MESH" and o not in (cube, ground)
         and (o == sword or o.find_armature() == rig)]
feet = {"UniMate Human | foot.left", "UniMate Human | foot.right"}
hits = []
minimum_ground = (1e9, None, None)
minimum_cube_clearance = (1e9, None, None)
for frame in range(1, 181):
    scene.frame_set(frame)
    dg = bpy.context.evaluated_depsgraph_get()
    ec = cube.evaluated_get(dg)
    cm = ec.to_mesh()
    tree = BVHTree.FromPolygons([ec.matrix_world @ v.co for v in cm.vertices],
                                [p.vertices[:] for p in cm.polygons])
    ec.to_mesh_clear()
    for obj in parts:
        eo = obj.evaluated_get(dg)
        mesh = eo.to_mesh()
        vertices = [eo.matrix_world @ v.co for v in mesh.vertices]
        part_tree = BVHTree.FromPolygons(vertices, [p.vertices[:] for p in mesh.polygons])
        eo.to_mesh_clear()
        if tree.overlap(part_tree):
            hits.append([frame, obj.name])
        if obj.name in feet:
            low = min(v.z - ground.location.z for v in vertices)
            if low < minimum_ground[0]:
                minimum_ground = (low, frame, obj.name)
            near = [v.z - (cube.location.z + .35) for v in vertices
                    if abs(v.x - cube.location.x) < .35 and abs(v.y - cube.location.y) < .35]
            if near and min(near) < minimum_cube_clearance[0]:
                minimum_cube_clearance = (min(near), frame, obj.name)
report = dict(frames_checked=180, cube_intersections=hits,
              minimum_foot_height_over_cube=minimum_cube_clearance,
              minimum_foot_height_over_ground=minimum_ground)
assert not hits, hits[:3]
assert minimum_cube_clearance[0] > .1, minimum_cube_clearance
assert minimum_ground[0] > -.01, minimum_ground
(ROOT / "tests/artifacts/obstacle-blender.json").write_text(json.dumps(report, indent=2))
print("OBSTACLE_CHECK_PASSED", json.dumps(report), flush=True)
