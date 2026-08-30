"""EEVEE multi-view rendering utilities (pip ``bpy`` module).

Shared by the :mod:`data_process.motion_rendering` stage: render-engine
init, scene reset, asset import, material fixes, studio lighting, scene
normalization, spherical camera placement, per-action multi-view frame
rendering and T-pose 2x2 grids.

These helpers target the pip ``bpy`` module driven by plain ``python``
(EEVEE needs the module's GPU context; ``blender -b`` has no display
surface), unlike :mod:`data_process.utils.blender_export`, which also runs
under Blender's bundled interpreter.
"""

import fcntl
import os
import json
import math
import re
import tempfile
from contextlib import contextmanager
from typing import List, Literal, Optional, Tuple

import bpy
import imageio
import numpy as np
from mathutils import Euler, Matrix, Vector
from PIL import Image
from loguru import logger


# Frame rate for import, preview video and the per-view metadata JSON. Must
# match the export stage's --fps (also 30) — see load_and_prep_asset.
RENDER_FPS = 30


# ─────────────────────────────────────────────────────────────────────────────
# GPU serialization
# ─────────────────────────────────────────────────────────────────────────────

# Several EEVEE contexts rendering on one device can wedge the NVIDIA driver
# (Xid 109 CTX SWITCH TIMEOUT, then Xid 31 MMU fault): every worker spins at
# 100% CPU in os_acquire_rwlock and not a single frame is ever written. An
# exclusive lock around the render call keeps GPU work single-file while
# asset import and video encoding still overlap across workers. The lock is
# node-local and per-GPU; set RENDER_GPU_LOCK=0 to disable it.

_gpu_lock_handle = None


def _gpu_lock_path() -> str:
    job = os.environ.get("SLURM_JOB_ID", "local")
    gpu = (os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0")
    name = f"unimate_render_{job}_gpu{gpu}.lock"
    # Node-local by preference: the contended resource is one node's GPU, and
    # TMPDIR often points at shared scratch, where the lock would be slower
    # and — for a multi-node job — serialize workers that share no device.
    for directory in ("/tmp", tempfile.gettempdir()):
        if os.path.isdir(directory) and os.access(directory, os.W_OK):
            return os.path.join(directory, name)
    return os.path.join(tempfile.gettempdir(), name)


@contextmanager
def gpu_render_lock():
    """Serialize GPU rendering across worker processes sharing one GPU."""
    global _gpu_lock_handle
    if os.environ.get("RENDER_GPU_LOCK", "1") == "0":
        yield
        return
    if _gpu_lock_handle is None:
        _gpu_lock_handle = open(_gpu_lock_path(), "w")
    fcntl.flock(_gpu_lock_handle, fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(_gpu_lock_handle, fcntl.LOCK_UN)


# ─────────────────────────────────────────────────────────────────────────────
# Render engine
# ─────────────────────────────────────────────────────────────────────────────

def init_render_engine(render_samples: int = 64):
    """Initialize EEVEE with the given sample count. Call once per process."""
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    eevee = bpy.context.scene.eevee

    # Anti-aliasing
    eevee.taa_render_samples = render_samples

    # Ambient occlusion
    eevee.use_gtao = True
    eevee.gtao_distance = 1.0
    eevee.gtao_quality = 0.5

    # Screen-space reflections (half-res for efficiency at 512px)
    eevee.use_ssr = True
    eevee.use_ssr_refraction = True
    eevee.use_ssr_halfres = True
    eevee.ssr_quality = 0.5
    eevee.ssr_max_roughness = 0.5
    eevee.ssr_thickness = 0.2

    # Shadows
    eevee.use_soft_shadows = True
    eevee.shadow_cascade_size = "2048"
    eevee.shadow_cube_size = "1024"

    eevee.use_bloom = True

    bpy.context.scene.render.use_high_quality_normals = True
    bpy.context.scene.render.use_persistent_data = True

    # Color management — Filmic handles HDR values from PBR materials
    # properly (Standard clips > 1.0, causing dark metallic surfaces)
    bpy.context.scene.view_settings.view_transform = "Filmic"
    bpy.context.scene.view_settings.look = "Medium Contrast"


# ─────────────────────────────────────────────────────────────────────────────
# Scene reset
# ─────────────────────────────────────────────────────────────────────────────

def clear_scene():
    """Remove all objects, compositor nodes, actions; purge orphan data.

    Removing objects leaves their mesh/material/image datablocks behind
    with zero users; without the purge they accumulate across assets and
    grow memory unboundedly over long in-process batches.
    """
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.context.scene.use_nodes = True
    for node in list(bpy.context.scene.node_tree.nodes):
        bpy.context.scene.node_tree.nodes.remove(node)

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 0
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)

    bpy.data.orphans_purge(do_recursive=True)


# ─────────────────────────────────────────────────────────────────────────────
# Asset import
# ─────────────────────────────────────────────────────────────────────────────

_IMPORT_OPS = {
    ".fbx": lambda p: bpy.ops.import_scene.fbx(filepath=p, use_anim=True),
    ".glb": lambda p: bpy.ops.import_scene.gltf(filepath=p),
    ".gltf": lambda p: bpy.ops.import_scene.gltf(filepath=p),
}

RENDERABLE_EXTS = frozenset(_IMPORT_OPS)


def load_file(path: str):
    """Import a rigged asset (.fbx, .glb, .gltf)."""
    ext = os.path.splitext(path)[1].lower()
    op = _IMPORT_OPS.get(ext)
    if op is None:
        raise RuntimeError(f"Unsupported file format: {path}")
    op(path)


def load_armature(path: str):
    """Import a file and return its main newly added armature (or None).

    Picks the armature with the most bones, matching the export stage's
    ``choose_main_armature`` — on multi-armature assets both stages must
    agree, or the render animates a different skeleton than the NPZ.
    """
    existing = set(bpy.data.objects)
    load_file(path)
    armatures = [
        obj for obj in bpy.data.objects
        if obj not in existing and obj.type == "ARMATURE"
    ]
    if not armatures:
        return None
    return max(armatures, key=lambda a: len(a.data.bones))


def find_meshes():
    return [o for o in bpy.data.objects if o.type == "MESH"]


def get_scene_armatures():
    """Return all ARMATURE objects in the current scene."""
    return [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]


def get_root_objects():
    """Yield all top-level (parentless) objects in the current scene."""
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


# ─────────────────────────────────────────────────────────────────────────────
# Material fixes (post-import)
# ─────────────────────────────────────────────────────────────────────────────

def fix_materials():
    """Fix common GLTF/FBX material issues after import.

    1. Clamps Metallic to <= 0.9 so surfaces don't render black when
       film_transparent=True (pure mirrors reflect the empty background).
    2. Connects vertex color attributes when Base Color is at a default
       value (white or black), indicating the importer left it unset.
    3. Connects disconnected Image Texture nodes found in the shader tree.
    """
    # Pass 1: clamp metallic on all Principled BSDF materials
    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type != 'BSDF_PRINCIPLED':
                continue
            metallic_input = node.inputs.get('Metallic')
            if metallic_input is None:
                continue
            # Only fix unlinked (constant) metallic values
            if not metallic_input.is_linked and metallic_input.default_value > 0.9:
                logger.info(f"Clamping Metallic {metallic_input.default_value:.2f} -> 0.9 "
                            f"on material '{material.name}'")
                metallic_input.default_value = 0.9

    # Pass 2: fix unlinked Base Color defaults per mesh
    for obj in find_meshes():
        mesh = obj.data

        # Check if mesh has color attributes (vertex colors)
        color_attr_name = None
        if hasattr(mesh, 'color_attributes') and len(mesh.color_attributes) > 0:
            color_attr_name = mesh.color_attributes[0].name
        elif hasattr(mesh, 'vertex_colors') and len(mesh.vertex_colors) > 0:
            color_attr_name = mesh.vertex_colors[0].name

        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or not mat.use_nodes:
                continue

            tree = mat.node_tree
            nodes = tree.nodes
            links = tree.links

            principled = None
            for node in nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    principled = node
                    break
            if principled is None:
                continue

            base_color_input = principled.inputs.get('Base Color')
            if base_color_input is None or base_color_input.is_linked:
                continue

            existing_color = list(base_color_input.default_value)
            is_default_white = all(c > 0.95 for c in existing_color[:3])
            is_default_black = all(c < 0.05 for c in existing_color[:3])
            if not (is_default_white or is_default_black):
                continue

            # Priority 1: connect vertex colors if available
            if color_attr_name is not None:
                vc_node = nodes.new(type='ShaderNodeVertexColor')
                vc_node.layer_name = color_attr_name
                links.new(vc_node.outputs['Color'], base_color_input)
                logger.info(f"Connected vertex colors '{color_attr_name}' to "
                            f"material '{mat.name}' on '{obj.name}'")
                continue

            # Priority 2: find disconnected Image Texture nodes in the tree
            img_tex = None
            for node in nodes:
                if node.type == 'TEX_IMAGE' and node.image is not None:
                    if not any(l.from_node == node and l.from_socket == node.outputs['Color']
                               for l in links):
                        img_tex = node
                        break
            if img_tex is not None:
                links.new(img_tex.outputs['Color'], base_color_input)
                logger.info(f"Connected orphan image texture '{img_tex.image.name}' to "
                            f"material '{mat.name}' on '{obj.name}'")


def clear_normal_maps():
    """Remove normal map connections from all Principled BSDF materials.

    Some GLTF normal maps cause rendering artifacts; removing them
    produces cleaner results for motion-capture rendering.
    """
    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
        tree = material.node_tree
        for node in tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                normal_input = node.inputs.get("Normal")
                if normal_input and normal_input.is_linked:
                    for link in list(normal_input.links):
                        tree.links.remove(link)


def set_materials_opaque():
    """Force materials to OPAQUE blend mode, keeping real alpha as HASHED.

    Prevents transparency artifacts that cause parts of meshes to
    disappear or render incorrectly in multi-view rendering. Materials
    whose Principled Alpha input is actually driven (linked, or set below
    1.0) genuinely need transparency — e.g. Mixamo facial-feature decal
    planes layered over the face, whose atlas would otherwise cover it as
    an opaque black quad. Those get HASHED, which EEVEE resolves without
    the depth-sorting artifacts of BLEND.
    """
    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
        has_alpha = False
        for node in material.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                alpha_input = node.inputs.get('Alpha')
                if alpha_input is not None and (
                        alpha_input.is_linked or alpha_input.default_value < 1.0):
                    has_alpha = True
                    break
        material.blend_method = "HASHED" if has_alpha else "OPAQUE"
        material.show_transparent_back = False


def smooth_meshes():
    """Enable auto-smooth on all meshes (30-degree threshold)."""
    for obj in find_meshes():
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = np.deg2rad(30)


def keep_main_mesh_only(main_mesh):
    """Remove every mesh except *main_mesh*.

    Dropping stray helper geometry (icospheres, prop meshes, low-vertex
    duplicates) keeps the render clean and the scene bbox tight.
    """
    for m in find_meshes():
        if m is not main_mesh:
            bpy.data.objects.remove(m, do_unlink=True)


def keep_armature_meshes(armature):
    """Remove meshes that do not belong to *armature*.

    A mesh belongs to the rig when it is skinned to it (Armature
    modifier) or parented under it (bone-parented props, meshes grouped
    beneath the armature node). Stray helper geometry (icospheres, prop
    meshes of another rig) matches neither — so this drops the junk
    without amputating multi-mesh characters the way keeping the single
    largest mesh would. Falls back to the largest mesh when nothing
    belongs to the rig.
    """
    def belongs(mesh):
        if any(mod.type == 'ARMATURE' and mod.object == armature
               for mod in mesh.modifiers):
            return True
        parent = mesh.parent
        while parent is not None:
            if parent == armature:
                return True
            parent = parent.parent
        return False

    meshes = find_meshes()
    keep = [m for m in meshes if belongs(m)]
    if keep:
        for m in meshes:
            if m not in keep:
                bpy.data.objects.remove(m, do_unlink=True)
    else:
        keep_main_mesh_only(max(meshes, key=lambda m: len(m.data.vertices)))


def load_and_prep_asset(path: str, keep_only_main_mesh: bool = True,
                        fps: int = RENDER_FPS):
    """Clear the scene, import an asset, and fix it up for rendering.

    Sets up lighting, optionally drops meshes that do not belong to the
    main armature (stray helper geometry that would pollute the render
    and the scene bbox — multi-mesh characters keep every skinned or
    bone-parented part), then applies material fixes and auto-smooth.

    The scene frame rate is set *before* the import, matching the export
    stage (``blender_export.load_scene``). glTF stores keyframe times in
    seconds, so the importer resamples them at the current scene fps —
    leaving it at Blender's default 24 makes every GLB render 0.8x as many
    frames as the 30 fps NPZ the same asset exports to, which shifts the
    frame window and makes the previews play 25% fast. (The FBX importer
    takes the rate from the file, so only glTF was affected.)

    Args:
        path: Asset file (.fbx / .glb / .gltf).
        keep_only_main_mesh: Drop meshes that are neither skinned to nor
            parented under the main armature (falling back to the
            highest-vertex mesh if nothing belongs to it). Disable to
            keep every mesh regardless.
        fps: Scene frame rate to import and render at. Must match the
            export stage's ``--fps`` or renders and NPZs desynchronize.

    Returns:
        The imported armature, or None if the file has no armature or
        no mesh.
    """
    clear_scene()
    bpy.context.scene.render.fps = int(fps)
    setup_lighting()

    armature = load_armature(path)
    if armature is None:
        return None

    # Drop lights bundled in the asset — they stack on the studio lighting
    # and make per-asset exposure inconsistent. Cameras go too: add_camera
    # only creates one when scene.camera is unset, so an asset camera would
    # otherwise take over the render and follow the asset's own camera path.
    studio_lights = {"Key_Light", "Fill_Light", "Rim_Light", "Bottom_Light"}
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT" and obj.name not in studio_lights:
            bpy.data.objects.remove(obj, do_unlink=True)
        elif obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)

    meshes = find_meshes()
    if not meshes:
        return None
    if keep_only_main_mesh:
        keep_armature_meshes(armature)

    fix_materials()
    clear_normal_maps()
    set_materials_opaque()
    smooth_meshes()
    return armature


# ─────────────────────────────────────────────────────────────────────────────
# Lighting
# ─────────────────────────────────────────────────────────────────────────────

def _create_light(
    name: str,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
    rotation: Tuple[float, float, float],
    energy: float,
    use_shadow: bool = False,
    specular_factor: float = 1.0,
    angle: float = 0.0,
):
    """Create and link a light object to the scene."""
    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.rotation_euler = rotation
    light_data.use_shadow = use_shadow
    light_data.specular_factor = specular_factor
    light_data.energy = energy
    if light_type == "SUN" and angle > 0:
        light_data.angle = angle  # angular diameter → soft shadow edges
    return light_object


def _setup_world_environment(strength: float = 0.7):
    """Soft top-to-bottom gradient world for even ambient illumination."""
    world = bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    bg_node = nodes.new(type="ShaderNodeBackground")
    bg_node.inputs["Strength"].default_value = strength

    output_node = nodes.new(type="ShaderNodeOutputWorld")

    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    gradient = nodes.new(type="ShaderNodeTexGradient")
    gradient.gradient_type = "LINEAR"

    color_ramp = nodes.new(type="ShaderNodeValToRGB")
    color_ramp.color_ramp.elements[0].position = 0.3
    color_ramp.color_ramp.elements[0].color = (0.9, 0.9, 0.9, 1.0)
    color_ramp.color_ramp.elements[1].position = 0.7
    color_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], gradient.inputs["Vector"])
    links.new(gradient.outputs["Color"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], bg_node.inputs["Color"])
    links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])


def setup_lighting():
    """Studio-style lighting for multi-view rendering.

    Three-point neutral sun lights plus a soft ambient world fill so no
    view is completely dark:

    - Key light:   upper-front-left, shadow enabled for depth
    - Fill light:  upper-front-right, softer, no shadow
    - Rim light:   upper-rear, edge separation
    - Bottom fill: subtle upward fill to prevent harsh under-shadows
    """
    # Clear existing lights
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    _setup_world_environment()

    _create_light("Key_Light", "SUN",
                  rotation=(0.785, 0, -0.785),   # 45° elevation, 45° left
                  energy=1.5, use_shadow=True, angle=0.05)
    _create_light("Fill_Light", "SUN",
                  rotation=(0.785, 0, 2.356),    # 45° elevation, 135° right
                  energy=0.7, specular_factor=0.5)
    _create_light("Rim_Light", "SUN",
                  rotation=(-0.524, 0, -3.927),  # 30° above rear
                  energy=1.0, specular_factor=0.8)
    _create_light("Bottom_Light", "SUN",
                  rotation=(2.618, 0, 0),        # pointing upward from below
                  energy=0.3, specular_factor=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Bounding box & scene normalization
# ─────────────────────────────────────────────────────────────────────────────

def get_scene_bbox(single_obj=None, ignore_matrix=False):
    """Axis-aligned bounding box of all meshes (or a single object)."""
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3

    meshes = find_meshes() if single_obj is None else [single_obj]
    if not meshes:
        raise RuntimeError("No objects in scene to compute bounding box for")

    for obj in meshes:
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(a, b) for a, b in zip(bbox_min, coord))
            bbox_max = tuple(max(a, b) for a, b in zip(bbox_max, coord))

    return Vector(bbox_min), Vector(bbox_max)


def get_scene_bbox_all_frames():
    """Bounding box that encloses the entire animation sequence."""
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3

    saved_frame = bpy.context.scene.frame_current
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        bpy.context.scene.frame_set(frame)
        fmin, fmax = get_scene_bbox()
        bbox_min = tuple(min(a, b) for a, b in zip(bbox_min, fmin))
        bbox_max = tuple(max(a, b) for a, b in zip(bbox_max, fmax))

    bpy.context.scene.frame_set(saved_frame)
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene(scene_scale: float = 1.0, process_frames: bool = False):
    """Scale and center the scene to fit within *scene_scale* at the origin.

    Parents all root objects under a single "NormalizationNode" empty and
    applies scale/offset there. Safe to call multiple times — resets the
    node to identity first, so repeated calls don't compound.

    Args:
        scene_scale: Target bounding box size.
        process_frames: If True, the bbox covers the whole animation range.
    """
    node = bpy.data.objects.get("NormalizationNode")
    if node is not None:
        node.location = (0, 0, 0)
        node.rotation_euler = (0, 0, 0)
        node.scale = (1, 1, 1)
        bpy.context.view_layer.update()

    if process_frames:
        bbox_min, bbox_max = get_scene_bbox_all_frames()
    else:
        bbox_min, bbox_max = get_scene_bbox()

    scale = scene_scale / max(bbox_max - bbox_min)
    offset = -(bbox_min + bbox_max) / 2

    if node is None:
        node = bpy.data.objects.new("NormalizationNode", None)
        bpy.context.scene.collection.objects.link(node)
        for obj in get_root_objects():
            if obj is not node and obj.type != "CAMERA":
                obj.parent = node
                obj.matrix_parent_inverse = node.matrix_world.inverted()
    node.scale = (scale, scale, scale)
    node.location = offset * scale
    bpy.context.view_layer.update()

    bpy.ops.object.select_all(action="DESELECT")
    return scale, offset


# ─────────────────────────────────────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────────────────────────────────────

def build_transformation_mat(translation, rotation) -> np.ndarray:
    """Build a 4x4 transform from a (3,) translation and a (3,3) rotation or (3,) Euler."""
    translation = np.array(translation)
    rotation = np.array(rotation)

    mat = np.eye(4)
    if translation.shape[0] != 3:
        raise RuntimeError(
            f"Translation has invalid shape: {translation.shape}. Expected (3,).")
    mat[:3, 3] = translation

    if rotation.shape == (3, 3):
        mat[:3, :3] = rotation
    elif rotation.shape[0] == 3:
        mat[:3, :3] = np.array(Euler(rotation).to_matrix())
    else:
        raise RuntimeError(
            f"Rotation has invalid shape: {rotation.shape}. Expected (3,3) or (3,).")

    return mat


def camera_pose_on_sphere(
    azimuth_deg: float,
    elevation_deg: float,
    radius: float,
    center: Tuple[float, float, float] = (0, 0, 0),
) -> np.ndarray:
    """Cam-to-world 4x4 matrix for a camera on a sphere looking at *center*."""
    elev = math.radians(elevation_deg)
    azim = math.radians(azimuth_deg)
    phi = 0.5 * math.pi - elev

    x = center[0] + radius * math.sin(phi) * math.cos(azim)
    y = center[1] + radius * math.sin(phi) * math.sin(azim)
    z = center[2] + radius * math.cos(phi)
    cam_pos = Vector((x, y, z))

    look_dir = Vector(center) - cam_pos
    rot_euler = look_dir.to_track_quat("-Z", "Y").to_euler()
    return build_transformation_mat(cam_pos, rot_euler)


def fit_camera_distance(camera_dist: float, scene_scale: float = 1.0,
                        camera_sensor_width: int = 32, camera_lens: int = 35,
                        margin: float = 1.05) -> float:
    """Raise *camera_dist* if it would crop the normalized scene bbox.

    ``normalize_scene`` fits the subject into a cube of side *scene_scale*,
    whose worst case in frame is a near-face corner at depth
    ``d - scene_scale/2`` and lateral offset ``scene_scale/2``. With the
    fixed 35mm lens on a 32mm sensor the half-FOV is only atan(16/35) =
    24.6 degrees, so the default distance of 1.5 is marginally too close and
    clips the occasional wide subject (measured: ~0.3% of rendered frames,
    e.g. a crab's claw).

    Returns the larger of *camera_dist* and the distance that fits, so
    subjects that already frame correctly keep their exact previous framing
    and only the ones that would crop get pushed back.
    """
    half_fov = math.atan((camera_sensor_width / 2.0) / camera_lens)
    half_extent = scene_scale / 2.0
    required = half_extent + (half_extent / math.tan(half_fov)) * margin
    if required > camera_dist:
        logger.info(f"Camera distance {camera_dist:.3f} would crop a "
                    f"{scene_scale:.3f} bbox; using {required:.3f}")
        return required
    return camera_dist


def add_camera(
    cam2world_matrix,
    camera_sensor_width: int = 32,
    camera_lens: int = 35,
):
    """Set (or create) the scene camera from a cam-to-world matrix.

    Keyframes the pose at the current frame_end so per-view placement
    survives animation rendering (constant extrapolation holds it fixed).
    """
    if not isinstance(cam2world_matrix, Matrix):
        cam2world_matrix = Matrix(cam2world_matrix)

    if bpy.context.scene.camera is None:
        # camera_add leaves scene.camera unset, so the new camera has to be
        # picked up explicitly. Selecting "the last CAMERA in bpy.data.objects"
        # instead would hand the render to a camera bundled with the asset
        # (objects iterate by name), which then inherits its parent's animated
        # transform and flies along the asset's own camera path.
        before = set(bpy.data.objects)
        bpy.ops.object.camera_add(location=(0, 0, 0))
        new_cams = [o for o in bpy.data.objects
                    if o not in before and o.type == "CAMERA"]
        if not new_cams:
            raise RuntimeError("camera_add did not create a camera object")
        bpy.context.scene.camera = new_cams[0]

    cam = bpy.context.scene.camera
    cam.data.type = "PERSP"
    cam.data.sensor_width = camera_sensor_width
    cam.data.lens = camera_lens
    cam.matrix_world = cam2world_matrix

    frame = bpy.context.scene.frame_end
    cam.keyframe_insert(data_path="location", frame=frame)
    cam.keyframe_insert(data_path="rotation_euler", frame=frame)
    cam.data.keyframe_insert(data_path="lens", frame=frame)
    cam.data.keyframe_insert(data_path="sensor_width", frame=frame)

    return cam


def remove_camera():
    """Delete the active scene camera (clears stale keyframes between actions)."""
    cam = bpy.context.scene.camera
    if cam is not None:
        bpy.data.objects.remove(cam, do_unlink=True)
        bpy.context.scene.camera = None


# ─────────────────────────────────────────────────────────────────────────────
# Render output
# ─────────────────────────────────────────────────────────────────────────────

def enable_color_output(
    resolution: int,
    output_dir: str,
    file_prefix: str = "",
    mode: Literal["IMAGE", "VIDEO"] = "IMAGE",
    film_transparent: bool = True,
    fps: int = RENDER_FPS,
):
    """Configure color render output (square resolution).

    Args:
        resolution: Render resolution in pixels.
        output_dir: Directory for output files.
        file_prefix: Filename prefix (IMAGE) or MP4 stem (VIDEO).
        mode: 'IMAGE' for per-frame PNGs, 'VIDEO' for an MP4.
        film_transparent: Transparent background (IMAGE mode).
        fps: Video frame rate (VIDEO mode).
    """
    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = film_transparent
    scene.render.image_settings.quality = 100

    if mode == "IMAGE":
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.filepath = os.path.join(output_dir, file_prefix)
    elif mode == "VIDEO":
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.fps = fps
        scene.render.filepath = os.path.join(output_dir, file_prefix + ".mp4")


def render_animation():
    """Render the scene's frame range with compositing enabled."""
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True

    tree = bpy.context.scene.node_tree
    if "Render Layers" not in tree.nodes:
        tree.nodes.new("CompositorNodeRLayers")

    with gpu_render_lock():
        bpy.ops.render.render(animation=True, write_still=True)


def render_still(filepath: str):
    """Render the current frame to *filepath* with compositing enabled."""
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True

    tree = bpy.context.scene.node_tree
    if "Render Layers" not in tree.nodes:
        tree.nodes.new("CompositorNodeRLayers")

    bpy.context.scene.render.filepath = filepath
    with gpu_render_lock():
        bpy.ops.render.render(write_still=True)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-view rendering
# ─────────────────────────────────────────────────────────────────────────────

VIEW_NAMES = ["front", "back", "left", "right"]
VIEW_AZIMUTHS = [-90, 90, 180, 0]  # matches VIEW_NAMES order
NUM_VIEWS = len(VIEW_NAMES)
MAX_RENDER_FRAMES = 200


VIDEO_BACKGROUND = (235, 235, 235)


def _reset_frames_dir(frames_dir: str):
    """Create *frames_dir*, removing any PNGs left by a previous render.

    Blender overwrites ``{frame:04d}.png`` but never deletes frames outside
    the new range. Re-rendering a clip with a shorter range would otherwise
    leave the old tail in place — and since ``is_render_complete`` only
    checks that the four views hold the same non-zero count (all four are
    stale by the same amount), the clip still reports complete while every
    ``v00x.mp4`` ends with frames from the previous animation.
    """
    if os.path.isdir(frames_dir):
        for fname in os.listdir(frames_dir):
            if fname.endswith(".png"):
                os.remove(os.path.join(frames_dir, fname))
    else:
        os.makedirs(frames_dir, exist_ok=True)


def compose_frames_to_video(frames_dir: str, out_path: str, fps: int = RENDER_FPS,
                            background=VIDEO_BACKGROUND):
    """Compose a view's rendered PNG frames into an H.264 MP4.

    Reuses the already-rendered frames (no extra Blender render pass).
    Transparent-film PNGs are flattened onto *background* — H.264 carries
    no alpha channel.
    """
    frames = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    if not frames:
        logger.warning(f"No frames in {frames_dir}; skipping video compose.")
        return
    writer = imageio.get_writer(out_path, fps=fps)
    try:
        for fname in frames:
            img = Image.open(os.path.join(frames_dir, fname))
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGBA")
                bg = Image.new("RGB", img.size, background)
                bg.paste(img, mask=img.split()[-1])
                img = bg
            else:
                img = img.convert("RGB")
            writer.append_data(np.asarray(img))
    finally:
        writer.close()
    logger.info(f"Composed video: {out_path} ({len(frames)} frames @ {fps} fps)")


def _skip_marker_path(output_dir: str, name: str) -> str:
    return os.path.join(output_dir, ".skipped", name)


def is_asset_render_skipped(output_dir: str, name: str) -> bool:
    """True when *name* was marked as producing no renders (see below)."""
    return os.path.isfile(_skip_marker_path(output_dir, name))


def mark_asset_render_skipped(output_dir: str, name: str, reason: str = ""):
    """Record that *name* legitimately produces no renders (no armature /
    mesh / pose actions), so reruns don't reload the asset just to rediscover
    that. Delete the marker (or ``.skipped/``) to retry the asset."""
    path = _skip_marker_path(output_dir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(reason + "\n")


def _rendered_marker_path(output_dir: str, name: str) -> str:
    return os.path.join(output_dir, ".rendered", f"{name}.json")


def mark_asset_render_done(output_dir: str, name: str, clip_dirs):
    """Record the full set of clip directories *name* is expected to produce.

    ``--missing-only`` cannot otherwise tell "this asset rendered all five of
    its actions" from "a worker was killed after three of them" — both leave
    only complete directories on disk, and without the expected set the
    partially-rendered asset is treated as done and its remaining actions are
    never rendered. Written only after every action of the asset has been
    handled.
    """
    path = _rendered_marker_path(output_dir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump({"clip_dirs": sorted(clip_dirs)}, f)
    os.replace(tmp, path)


def get_expected_clip_dirs(output_dir: str, name: str):
    """Expected clip directory names for *name*, or None when unrecorded."""
    path = _rendered_marker_path(output_dir, name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f).get("clip_dirs")
    except (OSError, ValueError):
        return None


def is_render_complete(save_root: str, num_views: int = NUM_VIEWS,
                       require_video: bool = False) -> bool:
    """Check if all expected multi-view outputs exist for one clip.

    Requires *num_views* view directories (v000..) with the matching camera
    metadata JSONs (v000.json..), each holding the same non-zero number of
    PNGs — plus the composed v000.mp4.. when *require_video* is set. Views
    render sequentially, so an interrupted render always leaves either a
    missing view directory or a view with fewer frames than the first —
    both fail the equal-count check.
    """
    if not os.path.isdir(save_root):
        return False
    frame_counts = []
    for view_idx in range(num_views):
        view_name = f"v{view_idx:03d}"
        if not os.path.isfile(os.path.join(save_root, f"{view_name}.json")):
            return False
        if require_video and not os.path.isfile(os.path.join(save_root, f"{view_name}.mp4")):
            return False
        view_dir = os.path.join(save_root, view_name)
        if not os.path.isdir(view_dir):
            return False
        frame_counts.append(sum(1 for f in os.listdir(view_dir) if f.endswith(".png")))
    return frame_counts[0] > 0 and len(set(frame_counts)) == 1


def render_action_multiview(
    armature,
    action_name: str,
    start_frame: int,
    end_frame: int,
    save_root: str,
    camera_dist: float = 1.5,
    scene_scale: float = 1.0,
    resolution: int = 512,
    views: Optional[List[str]] = None,
    render_video: bool = False,
    render_frames: bool = True,
    compose_video: bool = True,
    fps: int = RENDER_FPS,
    max_render_frames: int = MAX_RENDER_FRAMES,
    save_camera_metadata: bool = True,
):
    """Render one action from multiple views.

    Output structure::

        save_root/
            v000/  0001.png ...   (front)
            v001/  ...            (back)
            v002/  ...            (left)
            v003/  ...            (right)
            v000.mp4 ...          (if render_video=True)
            v000.json ...         (camera metadata per view)

    Args:
        armature: Blender armature object to animate.
        action_name: Name of the action in bpy.data.actions.
        start_frame / end_frame: Frame range of the action.
        save_root: Output directory for this clip's renders.
        camera_dist: Camera distance from origin.
        scene_scale: Target bounding box size for normalization.
        resolution: Render resolution (square).
        views: Subset of VIEW_NAMES to render (default: all four).
        render_video: Also render an MP4 per view via Blender (second full
            render pass; prefer *compose_video*).
        render_frames: Render per-frame PNGs per view.
        compose_video: Compose the rendered PNGs into v00x.mp4 per view
            (no extra render pass).
        fps: Video frame rate.
        max_render_frames: Clamp on the number of rendered frames.
        save_camera_metadata: Write a JSON per view with camera pose.
    """
    if views is None:
        views = list(VIEW_NAMES)
    view_azimuths = [VIEW_AZIMUTHS[VIEW_NAMES.index(v)] for v in views]

    # Clamp frame range
    if end_frame - start_frame + 1 > max_render_frames:
        end_frame = start_frame + max_render_frames - 1

    # Ensure armatures are in POSE mode (may be stuck in REST from a
    # preceding T-pose render)
    for arm in get_scene_armatures():
        arm.data.pose_position = "POSE"

    # Bind the action
    if armature.animation_data:
        armature.animation_data_clear()
    armature.animation_data_create()
    action = bpy.data.actions[action_name]
    armature.animation_data.action = action

    # A discovered action can belong to another armature in the file; the
    # render then silently shows a static pose. Warn so it's diagnosable.
    target_bones = {
        m.group(1)
        for fc in action.fcurves
        for m in [re.match(r'pose\.bones\["(.+?)"\]', fc.data_path)]
        if m
    }
    if target_bones and not (target_bones & set(armature.pose.bones.keys())):
        logger.warning(f"Action '{action_name}' animates none of armature "
                       f"'{armature.name}'s bones; render will be static.")

    # Set the frame range and flush the depsgraph so the mesh deforms
    # before normalization measures it
    scene = bpy.context.scene
    scene.frame_start = start_frame
    scene.frame_end = end_frame
    scene.frame_set(start_frame)
    bpy.context.view_layer.update()

    # Reset camera so the previous action's keyframes don't carry over
    remove_camera()

    # Normalize per action (accounts for the full motion extent). The
    # returned transform maps original asset coordinates into the rendered
    # (normalized) space: p_rendered = norm_scale * (p_original + norm_offset).
    norm_scale, norm_offset = normalize_scene(scene_scale, process_frames=True)
    camera_dist = fit_camera_distance(camera_dist, scene_scale)

    os.makedirs(save_root, exist_ok=True)

    for view_idx, azimuth in enumerate(view_azimuths):
        cam_mat = camera_pose_on_sphere(azimuth, 0.0, camera_dist)
        camera = add_camera(cam_mat)
        view_name = f"v{view_idx:03d}"

        if save_camera_metadata:
            meta = {
                # view identity
                "view": views[view_idx],
                "azimuth": math.radians(azimuth),
                "elevation": 0.0,
                # camera intrinsics
                "resolution": resolution,
                "camera_angle_x": camera.data.angle_x,
                # camera extrinsics (cam-to-world, in normalized scene space)
                "transform_matrix": cam_mat.tolist(),
                # scene normalization: p_rendered = scene_scale * (p_original + scene_offset)
                "scene_scale": norm_scale,
                "scene_offset": list(norm_offset),
                # rendered clip timing (after the max_render_frames clamp)
                "frame_start": start_frame,
                "frame_end": end_frame,
                "fps": fps,
            }
            with open(os.path.join(save_root, f"{view_name}.json"), "w") as f:
                json.dump(meta, f, indent=2)
        if render_video:
            enable_color_output(resolution, save_root, file_prefix=view_name,
                                mode="VIDEO", fps=fps)
            render_animation()

        if render_frames:
            frames_dir = os.path.join(save_root, view_name)
            _reset_frames_dir(frames_dir)
            enable_color_output(resolution, frames_dir, mode="IMAGE")
            render_animation()
            if compose_video:
                compose_frames_to_video(
                    frames_dir, os.path.join(save_root, f"{view_name}.mp4"), fps=fps)

        logger.info(f"Rendered view '{views[view_idx]}' -> {save_root}/{view_name}")


# ─────────────────────────────────────────────────────────────────────────────
# T-pose grid rendering
# ─────────────────────────────────────────────────────────────────────────────

def compose_grid_2x2(image_paths: List[str], output_path: str):
    """Arrange 4 images into a 2x2 grid and save the result.

    Layout::

        [front]  [back]
        [left]   [right]
    """
    assert len(image_paths) == 4, f"Expected 4 images, got {len(image_paths)}"
    images = [Image.open(p) for p in image_paths]
    w, h = images[0].size

    grid = Image.new("RGBA", (w * 2, h * 2))
    grid.paste(images[0], (0, 0))
    grid.paste(images[1], (w, 0))
    grid.paste(images[2], (0, h))
    grid.paste(images[3], (w, h))
    # Written atomically: the T-pose resume check is a bare os.path.isfile, so
    # a grid truncated by a kill mid-write would be treated as complete and
    # never regenerated.
    tmp_path = f"{output_path}.tmp.{os.getpid()}.png"
    grid.save(tmp_path)
    os.replace(tmp_path, output_path)
    for img in images:
        img.close()


def render_tpose_grid(
    save_path: str,
    camera_dist: float = 1.5,
    scene_scale: float = 1.0,
    resolution: int = 512,
):
    """Render the rest pose from 4 views and save a 2x2 grid PNG.

    Forces every armature into ``pose_position='REST'`` and clears bound
    actions so the evaluated mesh reflects the true rest pose; the bbox
    used for normalization is measured after that, so the T-pose is
    centered and scaled independently of any animation state. The caller
    is expected to clear the scene afterwards (state is not restored).
    """
    scene = bpy.context.scene

    # Force rest pose and drop bound actions
    for arm in get_scene_armatures():
        if arm.animation_data and arm.animation_data.action is not None:
            arm.animation_data.action = None
        arm.data.pose_position = "REST"
        arm.data.update_tag()

    # Flush the depsgraph on a single frame so bbox reads rest-pose coords
    frame = scene.frame_current
    scene.frame_start = frame
    scene.frame_end = frame
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    bpy.context.evaluated_depsgraph_get()

    remove_camera()
    normalize_scene(scene_scale, process_frames=False)

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True

    tmp_dir = tempfile.mkdtemp(prefix="tpose_")
    image_paths = []
    try:
        for view_name, azimuth in zip(VIEW_NAMES, VIEW_AZIMUTHS):
            add_camera(camera_pose_on_sphere(
                azimuth, 0.0, fit_camera_distance(camera_dist, scene_scale)))
            out_file = os.path.join(tmp_dir, f"{view_name}.png")
            scene.frame_set(frame)
            render_still(out_file)
            image_paths.append(out_file)

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        compose_grid_2x2(image_paths, save_path)
        logger.info(f"Saved T-pose grid: {save_path}")
    finally:
        for p in image_paths:
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass
