"""Mixamo-style armatures (Y-up data, rotated object, 0.01 scale) export, apply
and capture exactly like an equivalent Z-up rig; finger bones can be left out."""
import sys, json, math
from pathlib import Path
import numpy as np
import bpy
from mathutils import Matrix, Quaternion
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "addon"), str(ROOT / "tests")]
import unimate_motion
from unimate_motion.rig import export_skeleton, export_ground, apply_result, export_frame
from unimate_motion.motion import canonicalize, decode_features
from unimate_motion.poses import capture_pose, preview_estimate, auto_mapping
from unimate_motion import facing_from_feet
from build_scene import build, human_bones, make_rig, material

unimate_motion.register()
human, _ = build()
out = ROOT / "tests" / "artifacts"

# Existing Z-up rigs keep their exact export, so saved jobs and captures stay valid.
fixture = json.loads((ROOT / "tests/fixtures/seated-reference/request.json").read_text(encoding="utf-8"))
assert export_skeleton(human)["signature"] == fixture["skeleton"]["signature"], "Identity-frame signature changed"

bones = human_bones() + [("index_01.left", (.91, 0, 1.46), (.97, 0, 1.46), "hand.left", .015),
                         ("index_02.left", (.97, 0, 1.46), (1.01, 0, 1.46), "index_01.left", .012)]
collection = bpy.data.collections["UniMate Test Rigs"]
location = (0, 3, 0)
reference = make_rig("Frame Reference", bones, location, material("Frame Grey", (.5, .5, .5)), collection)
mixamo = make_rig("Frame Mixamo", bones, location, material("Frame Grey", (.5, .5, .5)), collection)
# Re-express the copy like an FBX import: data Y-up in centimetres, object rotated and scaled.
convert = Matrix.Rotation(math.radians(90), 4, "X") @ Matrix.Scale(.01, 4)
bpy.ops.object.select_all(action="DESELECT")
bpy.context.view_layer.objects.active = mixamo
bpy.ops.object.mode_set(mode="EDIT")
for bone in mixamo.data.edit_bones:
    bone.transform(convert.inverted(), scale=True, roll=True)
bpy.ops.object.mode_set(mode="OBJECT")
mixamo.matrix_world = Matrix.Translation(location) @ convert
bpy.context.view_layer.update()
linear, rotation = export_frame(mixamo)
assert linear is not rotation and np.allclose(np.linalg.norm(linear, axis=0), .01)

a, b = export_skeleton(reference), export_skeleton(mixamo)
assert a["joint_names"] == b["joint_names"]
assert np.allclose(a["heads"], b["heads"], atol=1e-5), "Export heads differ"
for ra, rb in zip(a["rest_matrices"], b["rest_matrices"]):
    if ra is not None:
        assert np.allclose(np.asarray(ra)[:3, :3], np.asarray(rb)[:3, :3], atol=1e-5), "Rest rotations differ"
assert [c["joint"] for c in a["collision_capsules"]] == [c["joint"] for c in b["collision_capsules"]]
assert np.allclose([c["radius"] for c in a["collision_capsules"]], [c["radius"] for c in b["collision_capsules"]], atol=1e-5)
assert [p["joint"] for p in a["foot_profiles"]] == [p["joint"] for p in b["foot_profiles"]]
assert facing_from_feet(a) == facing_from_feet(b) == "-Y", facing_from_feet(b)
ga, gb = export_ground(reference, a), export_ground(mixamo, b)
assert np.allclose(ga["normal"], [0, 0, 1]) and np.allclose(gb["normal"], [0, 0, 1])
assert abs(ga["height"] - gb["height"]) < 1e-5

# Finger exclusion drops finger bones and their children, nothing else.
names = {name for name, *_ in human_bones()}
no_fingers = export_skeleton(mixamo, fingers=False)
assert {n for n in no_fingers["bone_names"] if n} == names
assert no_fingers["fingers"] is False and len(no_fingers["parents"]) == len(b["parents"]) - 2  # two fingers out; the tip moves to the hand

# Applying the same motion poses both rigs identically in world space.
canon = canonicalize(b)
joints = len(b["parents"])
features = np.zeros((3, joints, 12))
features[:, :, 3:9] = [1, 0, 0, 0, 1, 0]
features[:, 0, 1] = canon["positions"][0, 1]
features[:-1, 0, 9] = .1
angle = .5
bend = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
for j, p in enumerate(b["parents"]):
    if p in (0, 3):
        features[:, j, 3:9] = np.concatenate([bend[:, 0], bend[:, 1]])
world = {}
for rig, skeleton in ((reference, a), (mixamo, b)):
    positions, rotations = decode_features(features, canonicalize(skeleton))
    path = out / (rig.name.replace(" ", "_") + ".npz")
    np.savez(path, schema=1, positions=positions, rotations=rotations, signature=skeleton["signature"],
             joint_names=np.array(skeleton["joint_names"]), fps=30., prompt="frame test", seed=0)
    apply_result(rig, skeleton, path, 1, bpy.context.scene)
    bpy.context.scene.frame_set(3)
    bpy.context.view_layer.update()
    world[rig.name] = {n: np.array(rig.matrix_world @ rig.pose.bones[n].head) for n in skeleton["bone_names"] if n}
    for j, n in enumerate(skeleton["bone_names"]):
        if n:
            expected = positions[2, j] + np.array(location)
            assert np.allclose(world[rig.name][n], expected, atol=1e-4), (rig.name, n, world[rig.name][n], expected)
for n in world[reference.name]:
    assert np.allclose(world[reference.name][n], world[mixamo.name][n], atol=1e-4), n

# Pose capture round trip on the rotated, scaled rig.
for rig in (reference, mixamo):
    rig.animation_data.action = None
    for pose in rig.pose.bones:
        pose.matrix_basis = Matrix()
    rig.pose.bones["upper_arm.left"].rotation_mode = "QUATERNION"
    rig.pose.bones["upper_arm.left"].rotation_quaternion = Quaternion((1, 0, 0), .7)
    rig.pose.bones["thigh.right"].rotation_mode = "QUATERNION"
    rig.pose.bones["thigh.right"].rotation_quaternion = Quaternion((0, 0, 1), -.4)
bpy.context.view_layer.update()
captured = [capture_pose(reference, a)["features"], capture_pose(mixamo, b)["features"]]
assert np.allclose(captured[0], captured[1], atol=1e-5), "Captured features differ"
positions, _ = decode_features(np.asarray(captured[1])[None], canonicalize(b))
for j, n in enumerate(b["bone_names"]):
    if n:
        head = np.array(mixamo.matrix_world @ mixamo.pose.bones[n].head) - np.array(location)
        assert np.allclose(positions[0, j], head, atol=1e-4), ("capture", n)

# The image-estimate preview places both rigs identically.
estimate = json.loads((ROOT / "tests/fixtures/estimated-pose.json").read_text())
heads = []
for rig, skeleton in ((reference, a), (mixamo, b)):
    preview_estimate(rig, skeleton, estimate, auto_mapping(rig))
    bpy.context.view_layer.update()
    heads.append(np.array([rig.matrix_world @ rig.pose.bones[n].head for n in skeleton["bone_names"] if n]))
assert np.allclose(heads[0], heads[1], atol=1e-4), "Estimate preview differs"

# Non-uniform object scale is rejected instead of silently distorting.
mixamo.scale = (.01, .02, .01)
bpy.context.view_layer.update()
try:
    export_skeleton(mixamo)
    raise AssertionError("Non-uniform scale accepted")
except ValueError:
    pass
report = dict(passed=["identity frame unchanged", "Y-up scaled export matches", "ground export",
                      "finger exclusion", "apply in world space", "capture round trip",
                      "estimate preview", "facing from feet", "non-uniform scale rejected"])
(out / "frame-blender.json").write_text(json.dumps(report, indent=2))
print("FRAME_BLENDER_PASSED", json.dumps(report))
unimate_motion.unregister()
