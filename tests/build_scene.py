"""Create two small rigged mannequins for integration testing; preserve existing objects."""
import bpy
import math
from mathutils import Vector
from pathlib import Path

def material(name, color, metallic=.1):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = .38
    return mat

def human_bones():
    bones = [
        ("hips", (0,0,1.0), (0,0,1.15), None, .17),
        ("spine", (0,0,1.15), (0,0,1.35), "hips", .17),
        ("chest", (0,0,1.35), (0,0,1.5), "spine", .23),
        ("neck", (0,0,1.5), (0,0,1.62), "chest", .075),
        ("head", (0,0,1.62), (0,-.015,1.86), "neck", .15),
    ]
    for s, side in ((1, "left"), (-1, "right")):
        bones += [
            ("upper_arm."+side, (s*.22,0,1.46), (s*.52,0,1.46), "chest", .07),
            ("forearm."+side, (s*.52,0,1.46), (s*.78,0,1.46), "upper_arm."+side, .055),
            ("hand."+side, (s*.78,0,1.46), (s*.91,0,1.46), "forearm."+side, .045),
            ("thigh."+side, (s*.105,0,1.02), (s*.12,-.012,.57), "hips", .09),
            ("shin."+side, (s*.12,-.012,.57), (s*.12,0,.13), "thigh."+side, .065),
            ("foot."+side, (s*.12,0,.13), (s*.12,-.20,.08), "shin."+side, .07),
        ]
    return bones

def creature_bones():
    bones = [
        ("pelvis", (0,.5,.82), (0,.28,.87), None, .24),
        ("spine", (0,.28,.87), (0,-.1,.90), "pelvis", .27),
        ("chest", (0,-.1,.90), (0,-.38,.94), "spine", .26),
        ("neck", (0,-.38,.94), (0,-.58,1.15), "chest", .15),
        ("head", (0,-.58,1.15), (0,-.90,1.13), "neck", .20),
        ("tail_base", (0,.57,.85), (0,.92,.92), "pelvis", .07),
        ("tail_tip", (0,.92,.92), (0,1.25,.80), "tail_base", .045),
    ]
    for s, side in ((1,"left"), (-1,"right")):
        bones += [
            ("front_upper_leg."+side, (s*.22,-.32,.9), (s*.24,-.25,.48), "chest", .075),
            ("front_lower_leg."+side, (s*.24,-.25,.48), (s*.24,-.38,.12), "front_upper_leg."+side, .055),
            ("front_paw."+side, (s*.24,-.38,.12), (s*.24,-.55,.075), "front_lower_leg."+side, .065),
            ("hind_thigh."+side, (s*.22,.48,.83), (s*.25,.28,.51), "pelvis", .1),
            ("hind_shin."+side, (s*.25,.28,.51), (s*.25,.55,.21), "hind_thigh."+side, .055),
            ("hind_paw."+side, (s*.25,.55,.21), (s*.25,.31,.075), "hind_shin."+side, .065),
        ]
    return bones

def make_rig(name, bones, location, mat, collection):
    existing = bpy.data.objects.get(name)
    if existing:
        return existing
    armature = bpy.data.armatures.new(name + " Skeleton")
    rig = bpy.data.objects.new(name, armature)
    collection.objects.link(rig)
    rig.location = location
    rig.show_in_front = True
    rig.display_type = "WIRE"
    armature.display_type = "STICK"
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for bone_name, head, tail, parent, radius in bones:
        bone = armature.edit_bones.new(bone_name)
        bone.head, bone.tail = head, tail
        if parent:
            bone.parent = armature.edit_bones[parent]
        # Deliberately vary roll: import must work for arbitrary bone frames.
        bone.roll = .17 if "left" in bone_name else -.11
    bpy.ops.object.mode_set(mode="OBJECT")
    pieces = []
    for bone_name, head, tail, parent, radius in bones:
        head, tail = Vector(head), Vector(tail)
        direction = tail - head
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=10)
        obj = bpy.context.object
        obj.name = name + " | " + bone_name
        rot = direction.to_track_quat("Z", "Y")
        center = (head + tail) * .5
        for vertex in obj.data.vertices:
            point = vertex.co.copy()
            point.x *= radius
            point.y *= radius
            point.z *= direction.length * .58 + radius * .25
            vertex.co = rot @ point + center
        obj.location = location
        obj.data.materials.append(mat)
        for poly in obj.data.polygons:
            poly.use_smooth = True
        group = obj.vertex_groups.new(name=bone_name)
        group.add(list(range(len(obj.data.vertices))), 1., "REPLACE")
        modifier = obj.modifiers.new("Deform", "ARMATURE")
        modifier.object = rig
        for coll in list(obj.users_collection):
            coll.objects.unlink(obj)
        collection.objects.link(obj)
        pieces.append(obj)
    rig["unimate_test_rig"] = True
    rig["unimate_description"] = "Simple weighted test mannequin, facing local -Y"
    return rig

def build():
    collection = bpy.data.collections.get("UniMate Test Rigs")
    if not collection:
        collection = bpy.data.collections.new("UniMate Test Rigs")
        bpy.context.scene.collection.children.link(collection)
    human = make_rig("UniMate Human", human_bones(), (-1.35,0,0),
                     material("UniMate Teal", (.045,.47,.44)), collection)
    creature = make_rig("UniMate Creature", creature_bones(), (1.35,0,0),
                        material("UniMate Copper", (.62,.22,.07)), collection)
    bpy.ops.object.select_all(action="DESELECT")
    human.select_set(True)
    bpy.context.view_layer.objects.active = human
    scene = bpy.context.scene
    scene.render.fps = 30
    scene.frame_start, scene.frame_end = 1, 60
    if hasattr(scene, "unimate_motion"):
        scene.unimate_motion.rig = human
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                space = area.spaces.active
                space.shading.type = "MATERIAL"
                space.region_3d.view_distance = 6.3
                space.region_3d.view_location = (0,0,.85)
                from mathutils import Quaternion
                from math import radians
                space.region_3d.view_rotation = (Vector((0,-6,3)).to_track_quat("Z","Y"))
                space.show_region_ui = True
    return human, creature

if __name__ == "__main__":
    build()
