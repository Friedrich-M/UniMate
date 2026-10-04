import sys
from pathlib import Path
import json
import numpy as np
import bpy
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))
sys.path.insert(0, str(ROOT / "tests"))
import unimate_motion
from unimate_motion.rig import export_skeleton, apply_result
from unimate_motion.motion import canonicalize, decode_features
from build_scene import build

unimate_motion.register()
human, creature = build()
reports = []
out = ROOT / "tests" / "artifacts"
out.mkdir(parents=True, exist_ok=True)
for rig in (human, creature):
    skeleton = export_skeleton(rig)
    canon = canonicalize(skeleton)
    joints = len(skeleton["parents"])
    features = np.zeros((3, joints, 12))
    features[:, :, 3:9] = [1,0,0,0,1,0]
    features[:, 0, 1] = canon["positions"][0, 1]
    positions, rotations = decode_features(features, canon)
    assert np.allclose(positions[0], skeleton["heads"], atol=1e-6), "Rest pose reconstruction"
    assert np.allclose(rotations, np.eye(3)), "Rest rotation reconstruction"
    # Independent known trajectory: canonical +X velocity returns to rig-local axis.
    features[:-1, 0, 9] = .1
    positions, rotations = decode_features(features, canon)
    expected_delta = np.array([.2,0,0]) @ canon["basis"] / canon["scale"]
    assert np.allclose(positions[2,0]-positions[0,0], expected_delta), "Root integration"
    # Bend the root 30 degrees about canonical Y through all root-child slots.
    angle = np.pi / 6
    rotate = np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    for j, p in enumerate(skeleton["parents"]):
        if p == 0:
            features[:, j, 3:9] = np.concatenate([rotate[:,0], rotate[:,1]])
    positions, rotations = decode_features(features, canon)
    path = out / (rig.name.replace(" ", "_") + ".npz")
    np.savez(path, schema=1, positions=positions, rotations=rotations,
             signature=skeleton["signature"], joint_names=np.array(skeleton["joint_names"]),
             fps=30., prompt="Synthetic integration fixture — not AI output", seed=0)
    rig.animation_data_create()
    old_action = bpy.data.actions.new("Existing animation " + rig.name)
    rig.animation_data.action = old_action
    action = apply_result(rig, skeleton, path, 5, bpy.context.scene)
    assert old_action.name in bpy.data.actions and old_action.use_fake_user
    bpy.context.scene.frame_set(7)
    bpy.context.view_layer.update()
    for j, name in enumerate(skeleton["bone_names"]):
        if not name:
            continue
        actual = np.array(rig.pose.bones[name].matrix)
        expected_rotation = rotations[2,j] @ np.array(rig.data.bones[name].matrix_local)[:3,:3]
        assert np.allclose(actual[:3,3], positions[2,j], atol=1e-5), f"Head {name}: {actual[:3,3]} != {positions[2,j]}"
        assert np.allclose(actual[:3,:3], expected_rotation, atol=1e-5), "Bone roll conversion " + name
    # Stale rest pose / facing rejects before touching the active action.
    changed = dict(skeleton, forward="Y")
    try:
        apply_result(rig, changed, path, 1, bpy.context.scene)
        raise AssertionError("Stale signature accepted")
    except ValueError:
        pass
    assert rig.animation_data.action == action
    constraint = rig.pose.bones[0].constraints.new("COPY_ROTATION")
    try:
        export_skeleton(rig)
        raise AssertionError("Constraint rig accepted")
    except ValueError:
        pass
    rig.pose.bones[0].constraints.remove(constraint)
    reports.append({"rig": rig.name, "joints": joints, "tests": ["rest pose", "root trajectory", "rolled bone FK", "existing action preserved", "stale result rejected", "constraints rejected"]})
unimate_motion.unregister()
(out / "blender-tests.json").write_text(json.dumps({"blender": bpy.app.version_string, "passed": reports}, indent=2))
print("UNIMATE_BLENDER_TESTS_PASSED", json.dumps(reports))
