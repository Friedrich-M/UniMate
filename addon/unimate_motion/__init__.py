bl_info = {
    "name": "UniMate Motion", "author": "Nopeburger",
    "version": (0, 6, 0), "blender": (4, 2, 0),
    "location": "3D View > Sidebar > UniMate", "category": "Animation",
    "description": "Local text-to-motion for simple human and creature deform rigs",
}
import json
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import time
import uuid
import bpy
from bpy.app.handlers import persistent
from bpy.props import StringProperty, IntProperty, FloatProperty, EnumProperty, BoolProperty, PointerProperty, CollectionProperty
from .rig import armature_for, export_skeleton, export_ground, apply_result
from . import clips

_job = None
# Persistent inference worker: dict(process, log, key, last_used, idle).
_server = None

def selected_rig(context):
    return context.scene.unimate_motion.rig or armature_for(context.object)

class UniMateSettings(bpy.types.PropertyGroup):
    mode: EnumProperty(name="Workflow", items=[("SINGLE", "Single prompt", ""), ("TIMELINE", "Prompt timeline", "")], default="TIMELINE")
    clips: CollectionProperty(type=clips.UniMateClip)
    clip_index: IntProperty(default=0, min=0)
    bone_mapping: CollectionProperty(type=clips.UniMateBoneMapping)
    show_mapping: BoolProperty(default=False)
    ground_object: PointerProperty(name="Ground mesh", type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "MESH",
        description="Optional static surface for foot and paw contact; otherwise use the rest sole level")
    smoothing: FloatProperty(name="Motion smoothing", default=0.0, min=0.0, max=4.0, precision=1,
        description="Smooth the generated motion over time, in frames. Calms jitter and fast head or limb jerks that some non-human rigs get (dragons, long necks) at the cost of some sharpness; 1.5 to 2.5 is a good start. 0 leaves the motion as generated")
    motion_cleanup: BoolProperty(name="Motion cleanup", default=True,
        description="Correct self-collisions, ground contact and extreme foot/paw rotations; intersecting reference poses may be adjusted")
    overlap: IntProperty(name="Transition context", default=10, min=1, max=30)
    transition_frames: IntProperty(name="Prompt blend frames", default=12, min=0, max=120, description="Smooth joins between generated prompt clips; zero disables")
    pose_approach_frames: IntProperty(name="Pose approach frames", default=60, min=0, max=600, description="Ease into captured reference poses over this many frames within their clip; replaces motion in that approach with a pose blend; zero disables")
    rig: PointerProperty(name="Rig", type=bpy.types.Object, poll=lambda self, obj: obj.type == "ARMATURE")
    project: StringProperty(name="Project folder", subtype="DIR_PATH", default="")
    experiment: StringProperty(name="Model folder", subtype="DIR_PATH", default="")
    human_model: BoolProperty(name="Mixamo model for humans", default=True,
        description="For Human characters, use the Mixamo-only checkpoint (unimate_mixamo_f60, installed beside the model folder) when the rig fits its 22-joint limit")
    prompt: StringProperty(name="Motion", default="A human walks forward at a steady pace.")
    forward: EnumProperty(name="Rig faces", items=[("-Y", "-Y", ""), ("Y", "+Y", ""), ("X", "+X", ""), ("-X", "-X", "")], default="-Y",
        description="World direction the character faces in its rest pose; the armature object's rotation is taken into account")
    family: EnumProperty(name="Character", items=[("mixamo", "Human", ""), ("truebones", "Animal / Creature", ""), ("objaverse", "Other articulated model", "")])
    tips: BoolProperty(name="Animate terminal bones", default=True, description="Add virtual endpoint joints; these count toward the model joint limit")
    fingers: BoolProperty(name="Animate finger bones", default=True,
        description="Include finger and thumb bones. Turn off for rigs with full hands (such as Mixamo); fingers then keep their rest pose")
    frames: IntProperty(name="Frames", default=60, min=2, max=60)
    fps: FloatProperty(name="Motion FPS", default=30, min=1, max=120, description="Playback interpretation; the source training clips use varying frame rates")
    seed: IntProperty(name="Seed", default=10, min=0)
    guidance: FloatProperty(name="Text guidance", default=3, min=1.01, max=20)
    extend_clips: BoolProperty(name="Generate long clips in full", default=True,
        description="Chain extra model windows so clips longer than 60 frames get new motion; off stretches one window over the clip")
    keep_loaded: BoolProperty(name="Keep model loaded", default=True,
        description="Keep the inference worker running between generations to skip model loading; it uses GPU memory while loaded")
    idle_minutes: IntProperty(name="Unload after (minutes)", default=15, min=1, max=240,
        description="Stop the loaded worker after this long without a generation")
    start_frame: IntProperty(name="Start frame", default=1)
    status: StringProperty(default="Select a deform rig and describe its motion")
    job_dir: StringProperty(subtype="DIR_PATH")
    advanced: BoolProperty(name="Setup and generation settings", default=False)

def refresh():
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()

def end_process(process, close_stdin=False):
    if close_stdin and process.stdin:
        try:
            process.stdin.close()
            process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            pass
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)

def stop_server():
    global _server
    if _server:
        end_process(_server["process"], close_stdin=True)
        _server["log"].close()
        _server = None

def stop_job():
    global _job
    if _job:
        if _job.get("server"):
            # Cancelling a served job stops the worker; the next run loads again.
            stop_server()
        else:
            end_process(_job["process"])
            _job["log"].close()
        _job = None

def server_idle():
    if not _server:
        return None
    if _job is None and time.monotonic() - _server["last_used"] > _server["idle"]:
        stop_server()
        refresh()
        return None
    return 30.

def submit_to_server(python, worker, folder, idle_minutes):
    """Send a job to the persistent worker, starting it if needed."""
    global _server
    key = (str(python), str(worker))
    if _server and (_server["key"] != key or _server["process"].poll() is not None):
        stop_server()
    if not _server:
        log = (worker.parents[1] / "outputs" / "worker-server.log").open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                [str(python), "-u", str(worker), "--serve"], stdin=subprocess.PIPE,
                cwd=tempfile.gettempdir(), stdout=log, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:
            log.close()
            raise
        _server = dict(process=process, log=log, key=key)
        if not bpy.app.timers.is_registered(server_idle):
            bpy.app.timers.register(server_idle, first_interval=30., persistent=True)
    _server.update(last_used=time.monotonic(), idle=idle_minutes * 60)
    job = dict(request=str(folder / "request.json"), output=str(folder / "motion.npz"),
               status=str(folder / "status.json"), log=str(folder / "worker.log"))
    try:
        _server["process"].stdin.write(json.dumps(job) + "\n")
        _server["process"].stdin.flush()
    except OSError:
        stop_server()
        raise ValueError("The loaded UniMate worker stopped; generate again to restart it.")
    return _server["process"]

@persistent
def before_load(_):
    stop_job()
    stop_server()
    if bpy.app.timers.is_registered(poll_job):
        bpy.app.timers.unregister(poll_job)

def poll_job():
    global _job
    if not _job:
        return None
    job = _job
    try:
        settings = job["scene"].unimate_motion
        status_path = job["directory"] / "status.json"
        status = {}
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            settings.status = status["message"][:220]
        code = job["process"].poll()
        if job.get("server"):
            # The served worker keeps running; the status file marks the end of a job.
            finished = status.get("state") in ("complete", "failed")
            if not finished and code is None:
                refresh()
                return .5
            if _server:
                _server["last_used"] = time.monotonic()
            code = 0 if status.get("state") == "complete" else 1
            if not finished:
                stop_server()
        elif code is None:
            refresh()
            return .5
        if job.get("kind") == "pose":
            result = job["directory"] / "pose.json"
            if code == 0 and result.is_file():
                matched = False
                for clip in settings.clips:
                    for ref in clip.references:
                        if ref.uid == job["reference_uid"] and bpy.path.abspath(ref.image_path) == job["source_image"]:
                            ref.estimate_path = str(result)
                            matched = True
                settings.status = "Pose estimated - preview, then capture" if matched else "Reference changed; unused estimate saved in job folder"
            else:
                settings.status = "Pose estimation failed; see " + str(job["directory"] / "worker.log")
        elif code == 0 and (job["directory"] / "motion.npz").is_file():
            settings.status = status.get("message", "Motion ready - Apply Motion")[:220]
        elif not status_path.is_file() or status.get("state") != "failed":
            settings.status = "Generation failed; see worker.log in the job folder"
        if not job.get("server"):
            job["log"].close()
        _job = None
    except (ReferenceError, AttributeError):
        stop_job()
        return None
    except (OSError, ValueError):
        return .5
    refresh()
    return None

HUMAN_MODEL = "unimate_mixamo_f60"  # Mixamo-only checkpoint, next to the general model

def model_limits(folder):
    path = folder / "config.json"
    if not path.is_file():
        return None
    dataset = json.loads(path.read_text())["dataset"]
    return dataset.get("min_joints", 5), dataset["max_joints"]

def choose_model(settings, joints):
    """Return (model folder, label, note). Humans use the Mixamo-only model when
    it is installed beside the general model and the rig fits its joint limit."""
    general = Path(bpy.path.abspath(settings.experiment)).resolve()
    if settings.family != "mixamo" or not settings.human_model:
        return general, "general model", ""
    human = general.parent / HUMAN_MODEL
    limits = model_limits(human)
    if limits is None:
        return general, "general model", ""
    if limits[0] <= joints <= limits[1]:
        return human, "Mixamo model", ""
    options = [name for name, on in (("Animate finger bones", settings.fingers),
                                     ("Animate terminal bones", settings.tips)) if on]
    advice = f"; turn off {' or '.join(options)} to use it" if options else ""
    return general, "general model", f"The Mixamo model allows {limits[1]} joints{advice}"

def joint_hint(settings):
    options = [name for name, on in (("Animate finger bones", settings.fingers),
                                     ("Animate terminal bones", settings.tips)) if on]
    if options:
        return "Turn off " + " or ".join(options) + " in setup settings, or simplify the rig."
    return "Simplify the rig."

def facing_from_feet(skeleton):
    """World axis the feet point along at rest, or None when they point mostly down."""
    directions = [np.asarray(skeleton["rest_matrices"][p["joint"]])[:3, 1] for p in skeleton["foot_profiles"]]
    if not directions:
        return None
    mean = np.mean(directions, axis=0)
    horizontal = mean[:2]
    if np.linalg.norm(horizontal) < .3 * np.linalg.norm(mean):
        return None
    axis = int(np.argmax(np.abs(horizontal)))
    return ("-" if horizontal[axis] < 0 else "") + "XY"[axis]

class UNIMATE_OT_validate(bpy.types.Operator):
    bl_idname = "unimate.validate"
    bl_label = "Check Rig"
    def execute(self, context):
        settings = context.scene.unimate_motion
        try:
            data = export_skeleton(selected_rig(context), settings.forward, settings.tips, settings.fingers)
            count = len(data["parents"])
            folder, label, note = choose_model(settings, count)
            limit = (model_limits(folder) or (5, 71))[1]
            if count > limit:
                raise ValueError(f"{count} joints exceed this model's {limit}-joint limit. " + joint_hint(settings))
            if count < 5:
                raise ValueError("This model requires at least 5 joints.")
            settings.status = f"Rig ready: {len([n for n in data['bone_names'] if n])} bones, {count}/{limit} joints, {label}"
            warnings = [note] if note else []
            facing = facing_from_feet(data)
            if facing and facing != settings.forward:
                warnings.append(f"Feet point {facing.replace('Y', '+Y').replace('X', '+X').replace('-+', '-')}; check Rig faces")
            if warnings:
                settings.status += ". " + ". ".join(warnings)
                self.report({"WARNING"}, settings.status)
                return {"FINISHED"}
            self.report({"INFO"}, settings.status)
            return {"FINISHED"}
        except Exception as exc:
            settings.status = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

class UNIMATE_OT_generate(bpy.types.Operator):
    bl_idname = "unimate.generate"
    bl_label = "Generate Motion"
    bl_description = "Run local UniMate inference in a separate process"
    @classmethod
    def poll(cls, context):
        return _job is None and selected_rig(context) is not None
    def execute(self, context):
        global _job
        settings = context.scene.unimate_motion
        try:
            rig = selected_rig(context)
            skeleton = export_skeleton(rig, settings.forward, settings.tips, settings.fingers)
            if not settings.project.strip() or not settings.experiment.strip():
                raise ValueError("Set Project folder and Model folder in Advanced settings.")
            root = Path(bpy.path.abspath(settings.project)).resolve()
            exp, _, _ = choose_model(settings, len(skeleton["parents"]))
            python = root / ".venv" / "Scripts" / "python.exe"
            if not python.is_file():
                python = root / ".venv" / "bin" / "python"
            worker = root / "backend" / "worker.py"
            if not python.is_file() or not worker.is_file():
                raise ValueError("Set the project folder to a configured UniMate installation.")
            for name in ("config.json", "dataset_stats.npy"):
                if not (exp / name).is_file():
                    raise ValueError(f"Missing model file: {name}")
            config = json.loads((exp / "config.json").read_text())
            minimum = config["dataset"].get("min_joints", 5)
            maximum = config["dataset"]["max_joints"]
            if not minimum <= len(skeleton["parents"]) <= maximum:
                hint = " " + joint_hint(settings) if len(skeleton["parents"]) > maximum else ""
                raise ValueError(f"Model requires {minimum}–{maximum} joints; rig has {len(skeleton['parents'])}.{hint}")
            if not list((exp / "checkpoints").glob("checkpoint_step_*.pt")):
                raise ValueError("Download a UniMate checkpoint into the model folder.")
            if settings.mode == "SINGLE" and not settings.prompt.strip():
                raise ValueError("Describe the motion first.")
            folder = root / "outputs" / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
            folder.mkdir(parents=True)
            request = dict(schema=1, skeleton=skeleton, prompt=settings.prompt.strip(),
                           experiment=str(exp), stats_family=settings.family, frames=settings.frames,
                           fps=settings.fps, seed=settings.seed, guidance=settings.guidance,
                           motion_cleanup=settings.motion_cleanup, smoothing=settings.smoothing,
                           ground=export_ground(rig, skeleton, settings.ground_object) if settings.motion_cleanup else None)
            if settings.mode == "TIMELINE":
                request.update(clips.collect_schedule(settings, skeleton, context.scene))
                request.update(transition_frames=settings.transition_frames,
                               pose_approach_frames=settings.pose_approach_frames,
                               extend_clips=settings.extend_clips)
                request["prompt"] = " / ".join(c["prompt"] for c in request["clips"])
            (folder / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
            if settings.keep_loaded:
                loaded = _server is not None and _server["process"].poll() is None
                process = submit_to_server(python, worker, folder, settings.idle_minutes)
                _job = dict(process=process, server=True, directory=folder, scene=context.scene)
                settings.status = "Sending to loaded UniMate" if loaded else "Starting local UniMate"
            else:
                stop_server()
                log = (folder / "worker.log").open("w", encoding="utf-8")
                try:
                    process = subprocess.Popen(
                        [str(python), "-u", str(worker), "--request", str(folder / "request.json"),
                         "--output", str(folder / "motion.npz"), "--status", str(folder / "status.json")],
                        cwd=tempfile.gettempdir(), stdout=log, stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                except Exception:
                    log.close()
                    raise
                _job = dict(process=process, log=log, directory=folder, scene=context.scene)
                settings.status = "Starting local UniMate"
            settings.rig = rig
            settings.job_dir = str(folder)
            if not bpy.app.timers.is_registered(poll_job):
                bpy.app.timers.register(poll_job, first_interval=.5)
            return {"FINISHED"}
        except Exception as exc:
            settings.status = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

class UNIMATE_OT_cancel(bpy.types.Operator):
    bl_idname = "unimate.cancel"
    bl_label = "Cancel Generation"
    def execute(self, context):
        stop_job()
        context.scene.unimate_motion.status = "Generation cancelled"
        refresh()
        return {"FINISHED"}

class UNIMATE_OT_unload(bpy.types.Operator):
    bl_idname = "unimate.unload"
    bl_label = "Unload Model"
    bl_description = "Stop the loaded inference worker and free its memory"
    @classmethod
    def poll(cls, context):
        return _job is None and _server is not None
    def execute(self, context):
        stop_server()
        context.scene.unimate_motion.status = "Model unloaded"
        refresh()
        return {"FINISHED"}

class UNIMATE_OT_apply(bpy.types.Operator):
    bl_idname = "unimate.apply"
    bl_label = "Apply Motion"
    bl_options = {"REGISTER", "UNDO"}
    @classmethod
    def poll(cls, context):
        folder = context.scene.unimate_motion.job_dir
        return _job is None and bool(folder) and (Path(folder) / "motion.npz").is_file()
    def execute(self, context):
        settings = context.scene.unimate_motion
        try:
            folder = Path(settings.job_dir)
            request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
            start = settings.start_frame
            if request.get("clips"):
                start = request["clips"][0]["start"]
                fps = context.scene.render.fps / context.scene.render.fps_base
                if abs(fps - request["fps"]) > .001:
                    raise ValueError("Scene FPS changed since timeline generation; restore it before applying.")
            action = apply_result(selected_rig(context), request["skeleton"], folder / "motion.npz",
                                  start, context.scene)
            settings.status = f"Created Action: {action.name}"
            self.report({"INFO"}, settings.status)
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

class UNIMATE_PT_main(bpy.types.Panel):
    bl_label = "UniMate Motion"
    bl_idname = "UNIMATE_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UniMate"
    def draw(self, context):
        layout, settings = self.layout, context.scene.unimate_motion
        layout.prop(settings, "rig")
        layout.prop(settings, "family")
        layout.prop(settings, "forward")
        layout.operator("unimate.validate", icon="CHECKMARK")
        layout.separator()
        layout.prop(settings, "mode", text="")
        if settings.mode == "TIMELINE":
            clips.draw_timeline(layout, context)
            layout.prop(settings, "seed")
            layout.prop(settings, "transition_frames")
            layout.prop(settings, "pose_approach_frames")
            layout.prop(settings, "extend_clips")
        else:
            layout.prop(settings, "prompt", text="")
            row = layout.row(align=True)
            row.prop(settings, "frames")
            row.prop(settings, "seed")
            layout.prop(settings, "start_frame")
        layout.prop(settings, "smoothing")
        layout.prop(settings, "motion_cleanup")
        if _job:
            layout.operator("unimate.cancel", icon="CANCEL")
        else:
            layout.operator("unimate.generate", icon="PLAY")
        layout.operator("unimate.apply", icon="ACTION")
        box = layout.box()
        for offset in range(0, len(settings.status), 44):
            box.label(text=settings.status[offset:offset+44])
        layout.prop(settings, "advanced", icon="TRIA_DOWN" if settings.advanced else "TRIA_RIGHT", emboss=False)
        if settings.advanced:
            layout.prop(settings, "ground_object")
            layout.prop(settings, "tips")
            layout.prop(settings, "fingers")
            layout.prop(settings, "guidance")
            layout.prop(settings, "fps")
            layout.prop(settings, "overlap")
            layout.prop(settings, "keep_loaded")
            row = layout.row(align=True)
            row.enabled = settings.keep_loaded
            row.prop(settings, "idle_minutes")
            row.operator("unimate.unload", text="", icon="X")
            layout.prop(settings, "project")
            layout.prop(settings, "experiment")
            layout.prop(settings, "human_model")
            layout.prop(settings, "job_dir", text="Last job")
        layout.label(text="Experimental • simple deform rigs", icon="INFO")

classes = clips.CLASSES + (UniMateSettings, UNIMATE_OT_validate, UNIMATE_OT_generate, UNIMATE_OT_cancel, UNIMATE_OT_unload, UNIMATE_OT_apply, UNIMATE_PT_main)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.unimate_motion = PointerProperty(type=UniMateSettings)
    bpy.app.handlers.load_pre.append(before_load)

def unregister():
    clips.cleanup()
    stop_job()
    stop_server()
    for timer in (poll_job, server_idle):
        if bpy.app.timers.is_registered(timer):
            bpy.app.timers.unregister(timer)
    if before_load in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(before_load)
    del bpy.types.Scene.unimate_motion
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
