"""Prompt clips, image references, and reviewable pose constraints."""
import json
from pathlib import Path
import subprocess
import tempfile
import time
import uuid
import bpy
import bpy.utils.previews
from bpy.props import StringProperty, IntProperty, BoolProperty, CollectionProperty
from bpy_extras.io_utils import ImportHelper
from .rig import export_skeleton
from .poses import ROLES, auto_mapping, capture_pose, preview_estimate
from .schedule import validate_clips

_previews = None

def image_changed(self, context):
    self.pose_json = ""
    self.estimate_path = ""

class UniMateReference(bpy.types.PropertyGroup):
    uid: StringProperty()
    image_path: StringProperty(name="Image", subtype="FILE_PATH", update=image_changed)
    frame: IntProperty(name="Reference frame", default=1)
    pose_json: StringProperty()
    estimate_path: StringProperty()

class UniMateClip(bpy.types.PropertyGroup):
    prompt: StringProperty(name="Prompt", default="A human walks forward.")
    start: IntProperty(name="Start", default=1)
    end: IntProperty(name="End", default=60)
    references: CollectionProperty(type=UniMateReference)
    reference_index: IntProperty(default=0)

class UniMateBoneMapping(bpy.types.PropertyGroup):
    role: StringProperty()
    bone: StringProperty()

def current_clip(settings):
    if not settings.clips:
        raise ValueError("Add a prompt clip first.")
    return settings.clips[min(max(settings.clip_index, 0), len(settings.clips)-1)]

def current_reference(settings):
    clip = current_clip(settings)
    if not clip.references:
        raise ValueError("Add a pose reference first.")
    return clip.references[min(max(clip.reference_index, 0), len(clip.references)-1)]

def collect_schedule(settings, skeleton, scene):
    clips = []
    for item in settings.clips:
        refs = [dict(frame=r.frame, image_path=bpy.path.abspath(r.image_path) if r.image_path else "",
                     pose=json.loads(r.pose_json) if r.pose_json else None) for r in item.references]
        clips.append(dict(prompt=item.prompt, start=item.start, end=item.end, references=refs))
    validate_clips(clips, skeleton["signature"])
    return dict(clips=clips, overlap=settings.overlap, fps=scene.render.fps / scene.render.fps_base)

class UNIMATE_UL_clips(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text=f"{item.start}–{item.end}", icon="ACTION")
        row.prop(item, "prompt", text="", emboss=False)

class UNIMATE_OT_clip_add(bpy.types.Operator):
    bl_idname = "unimate.clip_add"
    bl_label = "Add Prompt Clip"
    bl_options = {"REGISTER", "UNDO"}
    def execute(self, context):
        s = context.scene.unimate_motion
        start = s.clips[-1].end + 1 if s.clips else s.start_frame
        clip = s.clips.add()
        clip.start, clip.end = start, start+59
        clip.prompt = s.prompt
        s.clip_index = len(s.clips)-1
        return {"FINISHED"}

class UNIMATE_OT_clip_remove(bpy.types.Operator):
    bl_idname = "unimate.clip_remove"
    bl_label = "Remove Prompt Clip"
    bl_options = {"REGISTER", "UNDO"}
    def execute(self, context):
        s = context.scene.unimate_motion
        if s.clips:
            s.clips.remove(min(s.clip_index, len(s.clips)-1))
            s.clip_index = max(0, min(s.clip_index, len(s.clips)-1))
        return {"FINISHED"}

class UNIMATE_OT_reference_add(bpy.types.Operator):
    bl_idname = "unimate.reference_add"
    bl_label = "Add Pose Reference"
    bl_options = {"REGISTER", "UNDO"}
    def execute(self, context):
        try:
            clip = current_clip(context.scene.unimate_motion)
            ref = clip.references.add()
            ref.uid = uuid.uuid4().hex
            ref.frame = clip.start
            clip.reference_index = len(clip.references)-1
            return {"FINISHED"}
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

class UNIMATE_OT_reference_remove(bpy.types.Operator):
    bl_idname = "unimate.reference_remove"
    bl_label = "Remove Pose Reference"
    bl_options = {"REGISTER", "UNDO"}
    def execute(self, context):
        clip = current_clip(context.scene.unimate_motion)
        if clip.references:
            clip.references.remove(min(clip.reference_index, len(clip.references)-1))
            clip.reference_index = max(0, min(clip.reference_index, len(clip.references)-1))
        return {"FINISHED"}

class UNIMATE_OT_reference_image(bpy.types.Operator, ImportHelper):
    bl_idname = "unimate.reference_image"
    bl_label = "Choose Reference Image"
    filename_ext = ".png"
    filter_glob: StringProperty(default="*.png;*.jpg;*.jpeg;*.webp;*.bmp", options={"HIDDEN"})
    def execute(self, context):
        ref = current_reference(context.scene.unimate_motion)
        ref.image_path = self.filepath
        ref.estimate_path, ref.pose_json = "", ""
        return {"FINISHED"}

class UNIMATE_OT_map_human(bpy.types.Operator):
    bl_idname = "unimate.map_human"
    bl_label = "Auto-map Human Bones"
    bl_options = {"REGISTER", "UNDO"}
    def execute(self, context):
        from . import selected_rig
        rig = selected_rig(context)
        if rig is None:
            self.report({"ERROR"}, "Select a human rig first.")
            return {"CANCELLED"}
        s = context.scene.unimate_motion
        s.bone_mapping.clear()
        for role, name in auto_mapping(rig).items():
            item = s.bone_mapping.add()
            item.role, item.bone = role, name
        s.show_mapping = True
        return {"FINISHED"}

class UNIMATE_OT_estimate_pose(bpy.types.Operator):
    bl_idname = "unimate.estimate_pose"
    bl_label = "Estimate Human Pose"
    def execute(self, context):
        from . import selected_rig
        import sys
        main = sys.modules[__package__]
        s = context.scene.unimate_motion
        try:
            if main._job:
                raise ValueError("Wait for the current job or cancel it.")
            if s.family != "mixamo":
                raise ValueError("Automatic image estimation is for humans. Match creature poses manually.")
            ref = current_reference(s)
            if not ref.image_path or not Path(bpy.path.abspath(ref.image_path)).is_file():
                raise ValueError("Choose a readable reference image.")
            root = Path(bpy.path.abspath(s.project))
            model = root / "models" / "pose" / "pose_landmarker_full.task"
            python = root / ".venv" / "Scripts" / "python.exe"
            if not python.is_file():
                python = root / ".venv" / "bin" / "python"
            if not model.is_file():
                raise ValueError("Run project setup to download the human pose model.")
            directory = root / "outputs" / ("pose-" + uuid.uuid4().hex[:12])
            directory.mkdir(parents=True)
            log = (directory / "worker.log").open("w", encoding="utf-8")
            try:
                process = subprocess.Popen([str(python), "-u", str(root/"backend"/"estimate_pose.py"),
                    "--image", bpy.path.abspath(ref.image_path), "--model", str(model),
                    "--output", str(directory/"pose.json")],
                    cwd=tempfile.gettempdir(), stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except Exception:
                log.close()
                raise
            main._job = dict(process=process, log=log, directory=directory, scene=context.scene,
                             kind="pose", reference_uid=ref.uid, source_image=bpy.path.abspath(ref.image_path))
            if not bpy.app.timers.is_registered(main.poll_job):
                bpy.app.timers.register(main.poll_job, first_interval=.5)
            s.status = "Estimating human pose locally"
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

class UNIMATE_OT_preview_pose(bpy.types.Operator):
    bl_idname = "unimate.preview_pose"
    bl_label = "Preview Estimated Pose"
    bl_options = {"REGISTER", "UNDO"}
    def execute(self, context):
        from . import selected_rig
        s = context.scene.unimate_motion
        try:
            ref = current_reference(s)
            if not ref.estimate_path:
                raise ValueError("Estimate the human pose first.")
            rig = selected_rig(context)
            skeleton = export_skeleton(rig, s.forward, s.tips)
            if not s.bone_mapping:
                bpy.ops.unimate.map_human()
            mapping = {entry.role: entry.bone for entry in s.bone_mapping}
            estimate = json.loads(Path(ref.estimate_path).read_text(encoding="utf-8"))
            skipped = preview_estimate(rig, skeleton, estimate, mapping)
            context.view_layer.update()
            s.status = "Review pose, adjust if needed, then Capture Current Pose"
            if skipped:
                s.status = "Uncertain landmarks left neutral: " + ", ".join(skipped)
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

class UNIMATE_OT_capture_pose(bpy.types.Operator):
    bl_idname = "unimate.capture_pose"
    bl_label = "Capture Current Pose"
    bl_options = {"REGISTER", "UNDO"}
    def execute(self, context):
        from . import selected_rig
        s = context.scene.unimate_motion
        try:
            ref = current_reference(s)
            rig = selected_rig(context)
            skeleton = export_skeleton(rig, s.forward, s.tips)
            context.view_layer.update()
            ref.pose_json = json.dumps(capture_pose(rig, skeleton))
            s.status = f"Pose captured for frame {ref.frame}"
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

def draw_timeline(layout, context):
    global _previews
    from . import selected_rig
    s = context.scene.unimate_motion
    row = layout.row()
    row.template_list("UNIMATE_UL_clips", "", s, "clips", s, "clip_index", rows=3)
    col = row.column(align=True)
    col.operator("unimate.clip_add", text="", icon="ADD")
    col.operator("unimate.clip_remove", text="", icon="REMOVE")
    if not s.clips:
        return
    clip = current_clip(s)
    layout.prop(clip, "prompt", text="")
    row = layout.row(align=True)
    row.prop(clip, "start")
    row.prop(clip, "end")
    box = layout.box()
    box.label(text="Pose references", icon="IMAGE_DATA")
    row = box.row(align=True)
    row.operator("unimate.reference_add", text="Add", icon="ADD")
    row.operator("unimate.reference_remove", text="Remove", icon="REMOVE")
    if clip.references:
        if len(clip.references) > 1:
            box.prop(clip, "reference_index", text="Reference index (0-based)")
        ref = current_reference(s)
        box.prop(ref, "frame")
        box.prop(ref, "image_path", text="")
        box.operator("unimate.reference_image", icon="FILE_IMAGE")
        if ref.image_path and Path(bpy.path.abspath(ref.image_path)).is_file():
            if _previews is None:
                _previews = bpy.utils.previews.new()
            path = bpy.path.abspath(ref.image_path)
            if path not in _previews:
                try:
                    _previews.load(path, path, "IMAGE")
                except Exception:
                    pass
            if path in _previews:
                box.template_icon(icon_value=_previews[path].icon_id, scale=6.)
        if s.family == "mixamo":
            box.operator("unimate.estimate_pose")
            row = box.row()
            row.enabled = bool(ref.estimate_path)
            row.operator("unimate.preview_pose")
            box.prop(s, "show_mapping", text="Human bone mapping")
            if s.show_mapping:
                box.operator("unimate.map_human")
                rig = selected_rig(context)
                if rig:
                    for item in s.bone_mapping:
                        box.prop_search(item, "bone", rig.data, "bones", text=item.role.replace("_", " ").title())
        else:
            box.label(text="Match the rig pose to the image")
        box.operator("unimate.capture_pose", icon="KEY_HLT")
        box.label(text="Pose captured" if ref.pose_json else "No pose captured yet",
                  icon="CHECKMARK" if ref.pose_json else "INFO")

def cleanup():
    global _previews
    if _previews is not None:
        bpy.utils.previews.remove(_previews)
        _previews = None

CLASSES = (UniMateReference, UniMateClip, UniMateBoneMapping, UNIMATE_UL_clips,
    UNIMATE_OT_clip_add, UNIMATE_OT_clip_remove, UNIMATE_OT_reference_add, UNIMATE_OT_reference_remove,
    UNIMATE_OT_reference_image, UNIMATE_OT_map_human, UNIMATE_OT_estimate_pose,
    UNIMATE_OT_preview_pose, UNIMATE_OT_capture_pose)
