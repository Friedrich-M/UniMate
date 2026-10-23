"""UniMate Studio GUI — Sleek Dark Mode Desktop App for animating 3D rigs with UniMate.

Provides:
1. Top-level 'Prompts & Batch Animation' tab with dynamic 1..10 sub-tabs (each with
   source GLB, prompt text, destination directory, distinct file name, per-tab
   progress bar, and status log), plus auto-advancing tab focus during batch generation.
2. Top-level 'Dependencies & Hardware' tab that runs automatic environment,
   dependency, checkpoint, and hardware checks on startup, with an NVIDIA CUDA
   vs. CPU toggle (auto-disabled if no NVIDIA GPU is detected), one-click
   'Fix / Install' buttons, and a link to the official GitHub repository.
"""

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QSize, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QStyleFactory,
    QTabBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Resolve paths relative to the UniMate folder (always 1 directory up from GUI/)
if getattr(sys, "frozen", False):
    GUI_DIR = Path(sys.executable).resolve().parent
else:
    GUI_DIR = Path(__file__).resolve().parent
REPO_ROOT = GUI_DIR.parent
EXP_DIR = REPO_ROOT / "outputs" / "unimate_uniml3d_f60_v3"
CONFIG_PATH = EXP_DIR / "config.json"
STATS_PATH = EXP_DIR / "dataset_stats.npy"
CKPT_PATH = EXP_DIR / "checkpoints" / "checkpoint_step_150000.pt"
GITHUB_URL = "https://github.com/Friedrich-M/UniMate"
HF_MODEL_REPO = "Linzhan/UniMate"
HF_MODEL_URL = f"https://huggingface.co/{HF_MODEL_REPO}"

REQUIRED_GIT_FILES = [
    "data_process/mesh_animation/animate_motion.py",
    "data_process/rig_preprocess/__main__.py",
    "unimate/inference/sample.py",
    "unimate/models/factory.py",
]

DEFAULT_PROMPTS = [
    ("An object walks forward.", "walk_anim"),
    ("An object runs forward.", "run_anim"),
    ("An object jumps in place.", "jump_anim"),
    ("An object talks and gestures.", "talk_anim"),
    ("An object dances in place.", "dance_anim"),
    ("An object turns around in place.", "turn_anim"),
    ("An object attacks forward.", "attack_anim"),
    ("An object waves with its right arm.", "wave_anim"),
    ("An object squats and then stands up.", "squat_anim"),
    ("An object looks around cautiously.", "look_anim"),
]


def _is_same_as_frozen_exe(candidate: Path) -> bool:
    """Prevent the frozen UniMate_Studio.exe from ever treating itself as python.exe."""
    if not getattr(sys, "frozen", False):
        return False
    try:
        return candidate.resolve() == Path(sys.executable).resolve()
    except Exception:
        return True


def build_python_env(python_exe: str) -> dict:
    """Build an environment dict equivalent to 'conda activate unimate' so Conda's SSL, DLLs, and git.exe are on PATH."""
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    if not python_exe:
        return env
    py_path = Path(python_exe).resolve()
    env_dir = py_path.parent
    conda_paths = [
        str(env_dir),
        str(env_dir / "Library" / "mingw-w64" / "bin"),
        str(env_dir / "Library" / "usr" / "bin"),
        str(env_dir / "Library" / "bin"),
        str(env_dir / "Scripts"),
        str(env_dir / "bin"),
    ]
    existing_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(conda_paths + ([existing_path] if existing_path else []))
    env["CONDA_PREFIX"] = str(env_dir)
    return env


def find_conda_exe() -> str:
    """Locate conda.exe on the current PC if installed (no hardcoded usernames)."""
    local_app = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    prog_data = Path(os.environ.get("PROGRAMDATA", "C:/ProgramData"))
    candidates = [
        REPO_ROOT / "miniconda3" / "Scripts" / "conda.exe",
        Path.home() / "miniconda3" / "Scripts" / "conda.exe",
        Path.home() / "anaconda3" / "Scripts" / "conda.exe",
        local_app / "miniconda3" / "Scripts" / "conda.exe",
        local_app / "anaconda3" / "Scripts" / "conda.exe",
        prog_data / "miniconda3" / "Scripts" / "conda.exe",
        prog_data / "anaconda3" / "Scripts" / "conda.exe",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    which_conda = shutil.which("conda")
    if which_conda and Path(which_conda).is_file():
        return str(Path(which_conda).resolve())
    return ""


def _is_working_python_310(candidate: Path) -> bool:
    """Verify the candidate python.exe exists, is not the frozen GUI exe, and is a working Python 3.10 interpreter."""
    if not candidate.is_file() or _is_same_as_frozen_exe(candidate):
        return False
    try:
        proc = subprocess.run(
            [str(candidate), "-c", "import sys; assert sys.version_info[:2] == (3, 10)"],
            capture_output=True,
            text=True,
            timeout=10,
            env=build_python_env(str(candidate)),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return proc.returncode == 0
    except Exception:
        return False


def find_unimate_python() -> str:
    """Locate the unimate conda Python 3.10 executable if available (never picks broken .venv shims).

    IMPORTANT: When running as a frozen PyInstaller .exe, NEVER return sys.executable,
    otherwise subprocess calls will launch duplicate GUI windows in a loop!
    """
    local_app = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    prog_data = Path(os.environ.get("PROGRAMDATA", "C:/ProgramData"))
    candidates = [
        Path.home() / "miniconda3" / "envs" / "unimate" / "python.exe",
        Path.home() / "anaconda3" / "envs" / "unimate" / "python.exe",
        local_app / "miniconda3" / "envs" / "unimate" / "python.exe",
        local_app / "anaconda3" / "envs" / "unimate" / "python.exe",
        prog_data / "miniconda3" / "envs" / "unimate" / "python.exe",
        prog_data / "anaconda3" / "envs" / "unimate" / "python.exe",
        REPO_ROOT / ".conda_env" / "python.exe",
        REPO_ROOT / "env" / "python.exe",
    ]
    for c in candidates:
        if _is_working_python_310(c):
            return str(c)

    if not getattr(sys, "frozen", False) and _is_working_python_310(Path(sys.executable)):
        return sys.executable

    return ""


def check_git_repo_files() -> Tuple[bool, str]:
    """Check 1 directory up (REPO_ROOT) for required UniMate GitHub repository files."""
    missing = [rel for rel in REQUIRED_GIT_FILES if not (REPO_ROOT / rel).is_file()]
    if missing:
        return False, f"Missing in {REPO_ROOT.name}/: {', '.join(missing)}"
    return True, f"Verified in {REPO_ROOT}"


def check_hf_checkpoint_files() -> Tuple[bool, str]:
    """Check 1 directory up (REPO_ROOT/outputs/unimate_uniml3d_f60_v3) for Hugging Face model files."""
    missing = []
    if not CONFIG_PATH.is_file():
        missing.append("config.json")
    if not STATS_PATH.is_file():
        missing.append("dataset_stats.npy")
    if not CKPT_PATH.is_file():
        missing.append("checkpoint_step_150000.pt")
    elif CKPT_PATH.stat().st_size < 50 * 1024 * 1024:
        missing.append("checkpoint_step_150000.pt (incomplete)")
    if missing:
        return False, f"Missing in outputs/: {', '.join(missing)}"
    size_mb = CKPT_PATH.stat().st_size / (1024 * 1024)
    return True, f"Verified ({size_mb:.0f} MB checkpoint ready)"


def inspect_rig_bone_count(file_path: str) -> Tuple[Optional[int], str]:
    """Quickly inspect a .glb / .gltf file for its skeletal joint/bone count."""
    path_str = file_path.strip()
    if not path_str:
        return None, "No rig selected"
    p = Path(path_str)
    if not p.is_file():
        return None, "File not found"

    ext = p.suffix.lower()
    try:
        if ext == ".glb":
            with open(p, "rb") as f:
                header = f.read(20)
                if len(header) < 20:
                    return None, "Invalid GLB"
                magic, _ver, _length, chunk_len, chunk_type = struct.unpack("<4sIII4s", header)
                if magic != b"glTF" or chunk_type != b"JSON":
                    return None, "Invalid GLB header"
                g = json.loads(f.read(chunk_len).decode("utf-8"))
        elif ext == ".gltf":
            with open(p, "r", encoding="utf-8") as f:
                g = json.load(f)
        else:
            return None, "FBX rig (counted on run)"

        skins = g.get("skins", [])
        if skins:
            joints = max(len(s.get("joints", [])) for s in skins)
            return joints, f"Bones: {joints}"
        nodes = g.get("nodes", [])
        return 0, f"0 skinned bones ({len(nodes)} nodes)"
    except Exception:
        return None, "Unable to read bones"


class HalfWidthTabBar(QTabBar):
    """Top-level tab bar where each tab stretches to fill 50% of the full UI width."""

    def tabSizeHint(self, index: int) -> QSize:
        hint = super().tabSizeHint(index)
        parent = self.parentWidget()
        count = max(1, self.count())
        if parent and parent.width() > 100:
            w = max(hint.width(), (parent.width() - 4) // count)
            return QSize(w, max(hint.height(), 28))
        return QSize(hint.width(), max(hint.height(), 28))



# ---------------------------------------------------------------------------
# Dependency Definitions & Checker Thread
# ---------------------------------------------------------------------------

@dataclass
class DependencySpec:
    key: str
    name: str
    description: str
    check_code: str
    install_args: List[str]


DEPENDENCIES: List[DependencySpec] = [
    DependencySpec(
        key="git_repo",
        name="UniMate GitHub Repo Files (../)",
        description="Core UniMate repository scripts 1 folder up (data_process/mesh_animation/, data_process/rig_preprocess/, unimate/)",
        check_code="",
        install_args=["__DOWNLOAD_GIT_REPO__"],
    ),
    DependencySpec(
        key="python_env",
        name="Python 3.10 (unimate env)",
        description="Required Python 3.10 conda environment for bpy 4.0.0 compatibility",
        check_code="import sys; assert sys.version_info[:2] == (3, 10), f'Found {sys.version.split()[0]}'; print(sys.version.split()[0])",
        install_args=["__BOOTSTRAP_CONDA_PYTHON_310__"],
    ),
    DependencySpec(
        key="setuptools",
        name="setuptools (<81)",
        description="Provides pkg_resources required by Motion library",
        check_code="import setuptools, pkg_resources; print(setuptools.__version__)",
        install_args=["-m", "pip", "install", "setuptools<81"],
    ),
    DependencySpec(
        key="torch",
        name="PyTorch (torch & torchvision)",
        description="Core deep learning framework (CPU & CUDA 12.4 supported)",
        check_code="import torch, torchvision; print(torch.__version__)",
        install_args=[
            "-m", "pip", "install", "torch==2.5.1+cu124", "torchvision==0.20.1+cu124",
            "--extra-index-url", "https://download.pytorch.org/whl/cu124",
        ],
    ),
    DependencySpec(
        key="torch_geometric",
        name="torch-geometric",
        description="Graph neural network utilities for skeletal topology",
        check_code="import torch_geometric; print(torch_geometric.__version__)",
        install_args=["-m", "pip", "install", "torch-geometric==2.7.0"],
    ),
    DependencySpec(
        key="torchdiffeq",
        name="torchdiffeq",
        description="ODE/SDE integrators for flow matching sampling",
        check_code="import torchdiffeq; print(torchdiffeq.__version__)",
        install_args=["-m", "pip", "install", "torchdiffeq==0.2.5"],
    ),
    DependencySpec(
        key="transformers",
        name="transformers & sentencepiece",
        description="T5 text encoder (google/flan-t5-base) for prompt conditioning",
        check_code="import transformers, sentencepiece; print(transformers.__version__)",
        install_args=["-m", "pip", "install", "transformers==5.16.1", "sentencepiece==0.2.1"],
    ),
    DependencySpec(
        key="spacy",
        name="spacy & en_core_web_sm",
        description="Prompt text normalization for T5 conditioner",
        check_code="import spacy, en_core_web_sm, num2words; print(spacy.__version__)",
        install_args=[
            "-m", "pip", "install", "spacy>=3.8,<3.9", "num2words==0.5.14",
            "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
        ],
    ),
    DependencySpec(
        key="motion",
        name="Motion (Animation & Quaternions)",
        description="Holden-style skeletal Animation and Quaternions library",
        check_code=(
            "import Animation, Quaternions; "
            "from Animation import Animation as A, offsets_from_positions, positions_global; "
            "from Quaternions import Quaternions as Q; "
            "print('OK')"
        ),
        install_args=[
            "-m", "pip", "install", "git+https://github.com/inbar-2344/Motion.git", "--no-build-isolation",
        ],
    ),
    DependencySpec(
        key="bpy",
        name="Blender Python API (bpy==4.0.0)",
        description="Headless Blender 4.0 module for rig preprocessing & GLB/FBX export",
        check_code="import bpy; print(bpy.app.version_string)",
        install_args=[
            "-m", "pip", "install", "bpy==4.0.0",
            "--extra-index-url", "https://download.blender.org/pypi/",
        ],
    ),
    DependencySpec(
        key="viz_cli",
        name="Core Utilities & GUI (PySide6, einops, tyro, accelerate, loguru, matplotlib, imageio)",
        description="PySide6 desktop UI, tensor ops, CLI parser, logger, and preview rendering",
        check_code="import PySide6, einops, tyro, accelerate, loguru, matplotlib, imageio, huggingface_hub; print('OK')",
        install_args=[
            "-m", "pip", "install",
            "pyside6", "pyinstaller",
            "einops==0.8.2", "tyro==1.0.12", "accelerate==1.14.0",
            "loguru==0.7.3", "matplotlib==3.10.8", "imageio==2.37.3", "imageio-ffmpeg==0.6.0",
            "huggingface_hub",
        ],
    ),
    DependencySpec(
        key="checkpoint",
        name="Hugging Face Model (Linzhan/UniMate)",
        description="1 folder up in outputs/unimate_uniml3d_f60_v3 (config.json, dataset_stats.npy, checkpoint_step_150000.pt)",
        check_code="",
        install_args=["__DOWNLOAD_HF_CHECKPOINT__"],
    ),
]


def _detect_nvidia_gpu_via_smi() -> dict:
    """Fallback GPU detection using nvidia-smi when Python/PyTorch is not yet installed."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=8,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            first_gpu = proc.stdout.strip().splitlines()[0].strip()
            if first_gpu:
                return {"available": True, "device_name": first_gpu}
    except Exception:
        pass
    return {"available": False, "device_name": "No NVIDIA GPU detected"}


class DependencyCheckWorker(QThread):
    """Runs all dependency and hardware checks in a background thread."""

    result_ready = Signal(dict)

    def __init__(self, python_exe: str):
        super().__init__()
        self.python_exe = python_exe

    def run(self):
        # Clean up any rogue stub files in repo root that shadow the Motion package
        for stub in ("Animation.py", "Quaternions.py"):
            stub_path = REPO_ROOT / stub
            if stub_path.is_file():
                try:
                    stub_path.unlink()
                except OSError:
                    pass

        # Always check Git repo files and Hugging Face checkpoint files 1 directory up (REPO_ROOT) directly
        git_ok, git_detail = check_git_repo_files()
        ckpt_ok, ckpt_detail = check_hf_checkpoint_files()

        # Re-detect python_exe in case it was just created
        if not self.python_exe or not Path(self.python_exe).is_file() or _is_same_as_frozen_exe(Path(self.python_exe)):
            self.python_exe = find_unimate_python()

        # If no valid unimate Python 3.10 exists on this PC yet, DO NOT spawn sys.executable!
        if not self.python_exe or not Path(self.python_exe).is_file() or _is_same_as_frozen_exe(Path(self.python_exe)):
            deps_map = {}
            for d in DEPENDENCIES:
                if d.key == "git_repo":
                    deps_map[d.key] = {"ok": git_ok, "detail": git_detail}
                elif d.key == "checkpoint":
                    deps_map[d.key] = {"ok": ckpt_ok, "detail": ckpt_detail}
                elif d.key == "python_env":
                    deps_map[d.key] = {
                        "ok": False,
                        "detail": "Python 3.10 'unimate' environment not installed on this PC",
                    }
                else:
                    deps_map[d.key] = {
                        "ok": False,
                        "detail": "Requires Python 3.10 'unimate' environment",
                    }
            self.result_ready.emit({"deps": deps_map, "cuda": _detect_nvidia_gpu_via_smi()})
            return

        checks_dict = {d.key: d.check_code for d in DEPENDENCIES if d.check_code}
        runner_code = f"""
import json, sys, traceback

checks = {json.dumps(checks_dict)}
results = {{}}
for key, code in checks.items():
    try:
        ns = {{}}
        exec(code, ns, ns)
        results[key] = {{"ok": True, "detail": "Verified"}}
    except Exception as e:
        results[key] = {{"ok": False, "detail": str(e) or e.__class__.__name__}}

cuda_info = {{"available": False, "device_name": "No NVIDIA GPU detected"}}
try:
    import torch
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        cuda_info["available"] = True
        cuda_info["device_name"] = torch.cuda.get_device_name(0)
except Exception as e:
    cuda_info["device_name"] = f"CUDA check error: {{e}}"

print("___UNIMATE_CHECK_START___")
print(json.dumps({{"deps": results, "cuda": cuda_info}}))
print("___UNIMATE_CHECK_END___")
"""
        try:
            proc = subprocess.run(
                [self.python_exe, "-c", runner_code],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=60,
                env=build_python_env(self.python_exe),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            out = proc.stdout
            if "___UNIMATE_CHECK_START___" in out and "___UNIMATE_CHECK_END___" in out:
                payload = out.split("___UNIMATE_CHECK_START___")[1].split("___UNIMATE_CHECK_END___")[0].strip()
                data = json.loads(payload)
                data.setdefault("deps", {})["git_repo"] = {"ok": git_ok, "detail": git_detail}
                data.setdefault("deps", {})["checkpoint"] = {"ok": ckpt_ok, "detail": ckpt_detail}
                if not data.get("cuda", {}).get("available"):
                    smi_cuda = _detect_nvidia_gpu_via_smi()
                    if smi_cuda.get("available"):
                        data["cuda"] = smi_cuda
                self.result_ready.emit(data)
                return
        except Exception:
            pass

        fallback_deps = {d.key: {"ok": False, "detail": "Check failed"} for d in DEPENDENCIES}
        fallback_deps["git_repo"] = {"ok": git_ok, "detail": git_detail}
        fallback_deps["checkpoint"] = {"ok": ckpt_ok, "detail": ckpt_detail}
        fallback = {
            "deps": fallback_deps,
            "cuda": _detect_nvidia_gpu_via_smi(),
        }
        self.result_ready.emit(fallback)


class PackageInstallerWorker(QThread):
    """Installs one or more dependencies sequentially (including Git repo files, HF checkpoint, and Miniconda + Python 3.10)."""

    log_line = Signal(str)
    finished_install = Signal(bool, str)

    def __init__(self, python_exe: str, specs: List[DependencySpec]):
        super().__init__()
        self.python_exe = python_exe
        self.specs = specs

    def _run_streamed(self, cmd: List[str], label: str) -> bool:
        self.log_line.emit(f"\n>>> {label}:\n    {' '.join(cmd)}\n")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=build_python_env(self.python_exe),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                self.log_line.emit(line)
            proc.wait()
            if proc.returncode != 0:
                self.log_line.emit(f"[ERROR] {label} exited with code {proc.returncode}\n")
                return False
            self.log_line.emit(f"[OK] {label} completed successfully.\n")
            return True
        except Exception as exc:
            self.log_line.emit(f"[ERROR] Failed to run {label}: {exc}\n")
            return False

    def _download_file_with_progress(self, url: str, dest: Path, label: str) -> bool:
        import urllib.request

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_dest = dest.with_suffix(dest.suffix + ".part")
        self.log_line.emit(f"\n>>> Downloading {label}:\n    {url}\n    -> {dest}\n")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "UniMate-Studio/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_dest, "wb") as out_f:
                total = int(resp.headers.get("Content-Length", 0) or 0)
                downloaded = 0
                last_pct = -10
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = int(downloaded * 100 / total)
                        if pct >= last_pct + 10:
                            last_pct = pct
                            mb_done = downloaded / (1024 * 1024)
                            mb_tot = total / (1024 * 1024)
                            self.log_line.emit(f"    {label}: {pct}% ({mb_done:.1f} / {mb_tot:.1f} MB)\n")
            if dest.exists():
                dest.unlink()
            tmp_dest.replace(dest)
            self.log_line.emit(f"[OK] Downloaded {label} to {dest}\n")
            return True
        except Exception as exc:
            self.log_line.emit(f"[ERROR] Failed to download {label}: {exc}\n")
            return False

    def _download_missing_git_repo_files(self) -> bool:
        """Download and extract missing UniMate GitHub repository files into REPO_ROOT (1 directory up)."""
        import zipfile

        zip_url = f"{GITHUB_URL}/archive/refs/heads/main.zip"
        zip_path = Path(os.environ.get("TEMP", ".")) / "UniMate_repo_main.zip"
        if not self._download_file_with_progress(zip_url, zip_path, "UniMate GitHub Repository Archive"):
            return False
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.infolist():
                    parts = member.filename.split("/", 1)
                    if len(parts) < 2 or not parts[1]:
                        continue
                    rel_path = parts[1]
                    target_path = REPO_ROOT / rel_path
                    if member.is_dir():
                        target_path.mkdir(parents=True, exist_ok=True)
                    elif not target_path.exists():
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(member) as src_f, open(target_path, "wb") as dst_f:
                            shutil.copyfileobj(src_f, dst_f)
            ok, detail = check_git_repo_files()
            self.log_line.emit(f"[OK] GitHub repository files restored in {REPO_ROOT} ({detail})\n")
            return ok
        except Exception as exc:
            self.log_line.emit(f"[ERROR] Failed to extract GitHub repository archive: {exc}\n")
            return False

    def _download_hf_checkpoint_files(self) -> bool:
        """Download Hugging Face model files (Linzhan/UniMate) directly into REPO_ROOT/outputs/unimate_uniml3d_f60_v3."""
        hf_files = [
            ("unimate_uniml3d_f60_v3/config.json", CONFIG_PATH),
            ("unimate_uniml3d_f60_v3/dataset_stats.npy", STATS_PATH),
            ("unimate_uniml3d_f60_v3/checkpoints/checkpoint_step_150000.pt", CKPT_PATH),
        ]
        all_ok = True
        for rel_url, target_path in hf_files:
            if target_path.is_file() and (target_path != CKPT_PATH or target_path.stat().st_size > 50 * 1024 * 1024):
                self.log_line.emit(f"[OK] Already exists: {target_path}\n")
                continue
            url = f"{HF_MODEL_URL}/resolve/main/{rel_url}?download=true"
            if not self._download_file_with_progress(url, target_path, target_path.name):
                all_ok = False
        return all_ok

    def _bootstrap_conda_and_unimate_env(self) -> bool:
        """Ensure Conda and the 'unimate' Python 3.10 environment exist on this PC."""
        existing_py = find_unimate_python()
        if existing_py and Path(existing_py).is_file():
            self.python_exe = existing_py
            self.log_line.emit(f"[OK] Found existing Python 3.10 environment at: {self.python_exe}\n")
            return True

        conda_exe = find_conda_exe()
        if not conda_exe:
            installer_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
            installer_path = Path(os.environ.get("TEMP", ".")) / "Miniconda3_UniMate_Setup.exe"
            if not self._download_file_with_progress(installer_url, installer_path, "Miniconda3 Windows Installer"):
                return False

            target_dir = Path.home() / "miniconda3"
            self.log_line.emit(f"[Bootstrap] Installing Miniconda3 silently to {target_dir} (please wait 1–2 minutes)...\n")
            ok = self._run_streamed(
                [
                    str(installer_path),
                    "/InstallationType=JustMe",
                    "/RegisterPython=0",
                    "/S",
                    f"/D={target_dir}",
                ],
                "Miniconda3 Silent Install",
            )
            if not ok:
                return False
            conda_exe = find_conda_exe()
            if not conda_exe:
                self.log_line.emit("[ERROR] Miniconda3 installed, but conda.exe was not found.\n")
                return False

        # Re-check in case unimate env already exists inside the detected conda installation
        existing_py = find_unimate_python()
        if existing_py and Path(existing_py).is_file():
            self.python_exe = existing_py
            self.log_line.emit(f"[OK] Found existing Python 3.10 environment at: {self.python_exe}\n")
            return True

        # Create the 'unimate' Python 3.10 environment using conda-forge to avoid ToS prompts
        self.log_line.emit("\n[Bootstrap] Creating 'unimate' Python 3.10 environment via Conda...\n")
        ok = self._run_streamed(
            [
                conda_exe,
                "create",
                "-n",
                "unimate",
                "-c",
                "conda-forge",
                "--override-channels",
                "python=3.10",
                "pip",
                "git",
                "-y",
            ],
            "Create 'unimate' Python 3.10 Conda Environment",
        )
        if not ok:
            return False

        self.python_exe = find_unimate_python()
        if not self.python_exe or not Path(self.python_exe).is_file():
            self.log_line.emit("[ERROR] Created environment, but python.exe was not found.\n")
            return False

        self.log_line.emit(f"[OK] Python 3.10 environment ready at: {self.python_exe}\n")
        return True

    def run(self):
        all_ok = True

        # 1. Handle direct file downloads (GitHub repo & Hugging Face model) 1 folder up (REPO_ROOT)
        for spec in self.specs:
            if spec.key == "git_repo":
                if not self._download_missing_git_repo_files():
                    all_ok = False
            elif spec.key == "checkpoint":
                if not self._download_hf_checkpoint_files():
                    all_ok = False

        # 2. Check if any Python environment / pip packages are requested
        py_specs = [s for s in self.specs if s.key not in ("git_repo", "checkpoint")]
        if py_specs:
            needs_env_bootstrap = (
                not self.python_exe
                or not Path(self.python_exe).is_file()
                or _is_same_as_frozen_exe(Path(self.python_exe))
                or any(s.key == "python_env" for s in py_specs)
            )
            if needs_env_bootstrap:
                if not self._bootstrap_conda_and_unimate_env():
                    self.finished_install.emit(False, "Failed to bootstrap Python 3.10 environment.")
                    return

            for spec in py_specs:
                if spec.key == "python_env" or not spec.install_args:
                    continue
                cmd = [self.python_exe] + spec.install_args
                if not self._run_streamed(cmd, f"Installing {spec.name}"):
                    all_ok = False

        self.finished_install.emit(all_ok, "Installation complete." if all_ok else "Some installations failed.")


# ---------------------------------------------------------------------------
# Batch Animation Worker Thread
# ---------------------------------------------------------------------------

@dataclass
class AnimationJob:
    tab_index: int
    source_glb: str
    prompts: List[str]
    dest_dir: str
    file_name: str
    cfg_scale: float = 5.0
    seed: Optional[int] = None
    num_repetitions: int = 1
    anim_mode: str = "fk"
    extra_bones_strategy: str = "merge"
    rest_rotation: Optional[str] = None
    motion_expand: bool = False
    expand_overlap: int = 10


class BatchAnimationWorker(QThread):
    """Runs rig_preprocess (cached per unique source GLB + rest_rotation), sample.py, and animate_motion.py."""

    job_started = Signal(int, str)          # (tab_index, status_text)
    job_progress = Signal(int, int, str)    # (tab_index, percent 0..100, status_text)
    job_log = Signal(int, str)              # (tab_index, log_line)
    job_finished = Signal(int, bool, str)   # (tab_index, success, output_glb_or_err)
    batch_finished = Signal(int, int)       # (succeeded_count, total_count)

    def __init__(self, python_exe: str, jobs: List[AnimationJob], device: str):
        super().__init__()
        self.python_exe = python_exe
        self.jobs = jobs
        self.device = device
        self._stop_requested = False
        self._current_proc: Optional[subprocess.Popen] = None

    def request_stop(self):
        self._stop_requested = True
        if self._current_proc and self._current_proc.poll() is None:
            try:
                self._current_proc.terminate()
            except Exception:
                pass

    def _run_cmd(self, tab_index: int, cmd: List[str], stage_label: str, start_pct: int, end_pct: int) -> Tuple[bool, str]:
        self.job_log.emit(tab_index, f"\n$ {' '.join(cmd)}\n")
        try:
            self._current_proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=build_python_env(self.python_exe),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            assert self._current_proc.stdout is not None
            collected = []
            line_count = 0
            for line in self._current_proc.stdout:
                if self._stop_requested:
                    self._current_proc.terminate()
                    return False, "Cancelled by user."
                collected.append(line)
                self.job_log.emit(tab_index, line)
                line_count += 1
                prog = min(end_pct - 2, start_pct + int((end_pct - start_pct) * (1.0 - (1.0 / (1.0 + line_count * 0.08)))))
                self.job_progress.emit(tab_index, prog, stage_label)

            self._current_proc.wait()
            rc = self._current_proc.returncode
            self._current_proc = None
            if rc != 0:
                tail = "".join(collected[-20:])
                return False, f"{stage_label} failed (exit code {rc}):\n{tail}"
            self.job_progress.emit(tab_index, end_pct, f"{stage_label} complete")
            return True, ""
        except Exception as exc:
            return False, f"Error running {stage_label}: {exc}"

    def _prepare_config(self, required_max_joints: int = 71):
        """Ensure outputs/unimate_uniml3d_f60_v3/config.json has the selected device and sufficient max_joints."""
        if not CONFIG_PATH.is_file():
            return
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        changed = False
        if cfg.get("sampling", {}).get("device") != self.device:
            cfg.setdefault("sampling", {})["device"] = self.device
            changed = True
        curr_max = cfg.get("dataset", {}).get("max_joints", 71)
        target_max = max(71, required_max_joints)
        if curr_max < target_max:
            cfg.setdefault("dataset", {})["max_joints"] = target_max
            changed = True
        if changed:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=4)

    def _get_or_build_processed_rig(self, job: AnimationJob, cache: Dict[str, Tuple[Path, str, int]]) -> Tuple[Optional[Path], str, int, str]:
        """Preprocess source GLB into a canonical rig folder (or reuse cached preprocessing from same file + rest_rotation)."""
        src_path = Path(job.source_glb).resolve()
        if not src_path.is_file():
            return None, "", 0, f"Source GLB file does not exist: {src_path}"

        stat = src_path.stat()
        rot_tag = job.rest_rotation or "none"
        cache_key = f"{src_path}|{stat.st_mtime_ns}|{stat.st_size}|{rot_tag}"
        if cache_key in cache:
            rig_dir, asset_name, n_joints = cache[cache_key]
            if (rig_dir / "cond.npy").is_file() and (rig_dir / f"{asset_name}.glb").is_file():
                self.job_log.emit(job.tab_index, f"Reusing preprocessed rig at {rig_dir} ({n_joints} joints, rest_rot={rot_tag}).\n")
                self.job_progress.emit(job.tab_index, 25, f"Rig ready ({n_joints} joints)")
                return rig_dir, asset_name, n_joints, ""

        digest = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:8]
        safe_stem = re.sub(r"[- \t\n\r\f\v\[\]*?]+", "_", src_path.stem)
        rig_dir = REPO_ROOT / "assets" / "processed_rigs_cache" / f"{safe_stem}_{digest}"
        cond_file = rig_dir / "cond.npy"
        canon_glb = rig_dir / f"{safe_stem}.glb"
        summary_file = rig_dir / "summary.json"

        if cond_file.is_file() and canon_glb.is_file() and summary_file.is_file():
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    sdata = json.load(f)
                n_joints = int(sdata.get("n_joints", 71))
                asset_name = sdata.get("name", safe_stem)
                cache[cache_key] = (rig_dir, asset_name, n_joints)
                self.job_log.emit(job.tab_index, f"Found existing preprocessed rig in cache: {rig_dir} ({n_joints} joints).\n")
                self.job_progress.emit(job.tab_index, 25, f"Rig ready ({n_joints} joints)")
                return rig_dir, asset_name, n_joints, ""
            except Exception:
                pass

        rig_dir.mkdir(parents=True, exist_ok=True)
        self.job_progress.emit(job.tab_index, 5, "Step 1/3: Preprocessing rig in Blender...")
        cmd = [
            self.python_exe, "-m", "data_process.rig_preprocess", "run",
            "--input", str(src_path),
            "--output_dir", str(rig_dir),
            "--no_review",
            "--overwrite",
        ]
        if job.rest_rotation:
            cmd.extend(["--rest_rotation", job.rest_rotation])

        ok, err = self._run_cmd(job.tab_index, cmd, "Step 1/3: Preprocessing rig", 5, 25)
        if not ok:
            return None, "", 0, err

        n_joints = 71
        asset_name = safe_stem
        if summary_file.is_file():
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    sdata = json.load(f)
                n_joints = int(sdata.get("n_joints", 71))
                asset_name = sdata.get("name", safe_stem)
            except Exception:
                pass

        cache[cache_key] = (rig_dir, asset_name, n_joints)
        return rig_dir, asset_name, n_joints, ""

    def run(self):
        succeeded = 0
        total = len(self.jobs)
        rig_cache: Dict[str, Tuple[Path, str, int]] = {}

        for job in self.jobs:
            if self._stop_requested:
                break

            idx = job.tab_index
            self.job_started.emit(idx, "Starting animation pipeline...")
            self.job_progress.emit(idx, 2, "Preparing folders & rig...")

            clean_name = re.sub(r'[<>:"/\\|?*]+', "_", job.file_name.strip())
            if clean_name.lower().endswith(".glb"):
                clean_name = clean_name[:-4]
            if not clean_name:
                clean_name = f"animation_{idx + 1}"

            base_dest = Path(job.dest_dir).resolve()
            job_out_dir = base_dest / clean_name
            job_out_dir.mkdir(parents=True, exist_ok=True)

            self.job_log.emit(idx, f"=== Prompt #{idx + 1}: {clean_name} ===\n")
            self.job_log.emit(idx, f"Source GLB     : {job.source_glb}\n")
            self.job_log.emit(idx, f"Prompts        : {' | '.join(job.prompts)}\n")
            self.job_log.emit(idx, f"Output Dir     : {job_out_dir}\n")
            self.job_log.emit(idx, f"Device         : {self.device.upper()}\n")
            self.job_log.emit(
                idx,
                f"Options        : cfg={job.cfg_scale}, seed={job.seed if job.seed is not None else 'random'}, "
                f"variations={job.num_repetitions}, mode={job.anim_mode.upper()}, "
                f"extra_bones={job.extra_bones_strategy}, rest_rot={job.rest_rotation or 'none'}, "
                f"expand={job.motion_expand} (overlap={job.expand_overlap})\n",
            )

            # 1. Preprocess rig (or reuse cache)
            rig_dir, asset_name, n_joints, err = self._get_or_build_processed_rig(job, rig_cache)
            if rig_dir is None:
                self.job_finished.emit(idx, False, err)
                continue

            padded_joints = max(71, n_joints + 1)
            self._prepare_config(required_max_joints=padded_joints)
            if n_joints > 71:
                self.job_log.emit(
                    idx,
                    f"Note: Rig has {n_joints} bones (>71 default). Automatically expanded max_joints to {padded_joints} in config.json.\n",
                )

            # 2. Run UniMate inference sample
            self.job_progress.emit(idx, 30, f"Step 2/3: Generating motion ({self.device.upper()})...")
            sample_cmd = [
                self.python_exe, "-m", "unimate.inference.sample",
                "--exp_dir", str(EXP_DIR),
                "--cfg_scale", str(job.cfg_scale),
                "--num_repetitions", str(job.num_repetitions),
                "--asset", str(rig_dir),
                "--prompt", *job.prompts,
                "--output_dir", str(job_out_dir),
            ]
            if job.seed is not None:
                sample_cmd.extend(["--seed", str(job.seed)])
            if job.motion_expand and len(job.prompts) > 1:
                sample_cmd.extend(["--motion_expand", "--expand_overlap", str(job.expand_overlap)])

            ok, err = self._run_cmd(idx, sample_cmd, f"Step 2/3: Sampling motion ({self.device.upper()})", 30, 80)
            if not ok:
                self.job_finished.emit(idx, False, err)
                continue

            # When --motion_expand is active, sample.py writes into job_out_dir/motion_expand/motions
            sample_sub_dir = (job_out_dir / "motion_expand") if (job.motion_expand and len(job.prompts) > 1) else job_out_dir
            motions_dir = sample_sub_dir / "motions"
            npy_files = sorted(motions_dir.glob("*.npy"), key=lambda p: p.stat().st_mtime) if motions_dir.is_dir() else []
            if not npy_files:
                self.job_finished.emit(idx, False, f"No .npy motion file was produced in {motions_dir}")
                continue

            # Keep the most recent N repetitions from this run
            target_npys = npy_files[-job.num_repetitions:] if len(npy_files) >= job.num_repetitions else npy_files

            # 3. Bake each generated variation onto canonical character GLB & FBX
            canon_glb = rig_dir / f"{asset_name}.glb"
            cond_npy = rig_dir / "cond.npy"
            last_target_report = ""

            for v_idx, npy_path in enumerate(target_npys):
                v_label = f" (Variation {v_idx + 1}/{len(target_npys)})" if len(target_npys) > 1 else ""
                start_p = 80 + int(18 * (v_idx / max(1, len(target_npys))))
                end_p = 80 + int(18 * ((v_idx + 1) / max(1, len(target_npys))))
                self.job_progress.emit(idx, start_p, f"Step 3/3: Baking animated GLB & FBX{v_label}...")

                bake_cmd = [
                    self.python_exe, str(REPO_ROOT / "data_process" / "mesh_animation" / "animate_motion.py"),
                    "--",
                    "--dataset_type", "general",
                    "--anim_path", str(npy_path),
                    "--char_path", str(canon_glb),
                    "--cond_path", str(cond_npy),
                    "--output_dir", str(job_out_dir),
                    "--anim_mode", job.anim_mode,
                    "--extra_bones_strategy", job.extra_bones_strategy,
                ]
                ok, err = self._run_cmd(idx, bake_cmd, f"Step 3/3: Baking GLB & FBX{v_label}", start_p, end_p)
                if not ok:
                    self.job_finished.emit(idx, False, err)
                    break

                produced_glb = job_out_dir / f"{npy_path.stem}.glb"
                produced_fbx = job_out_dir / f"{npy_path.stem}.fbx"
                suffix = f"_v{v_idx + 1}" if len(target_npys) > 1 else ""
                final_glb = job_out_dir / f"{clean_name}{suffix}.glb"
                final_fbx = job_out_dir / f"{clean_name}{suffix}.fbx"

                try:
                    if produced_glb.is_file() and produced_glb.resolve() != final_glb.resolve():
                        shutil.copy2(produced_glb, final_glb)
                    if produced_fbx.is_file() and produced_fbx.resolve() != final_fbx.resolve():
                        shutil.copy2(produced_fbx, final_fbx)
                except Exception as exc:
                    self.job_log.emit(idx, f"Warning copying final named file: {exc}\n")

                last_target_report = str(final_glb if final_glb.is_file() else produced_glb)
            else:
                self.job_progress.emit(idx, 100, f"Completed: {Path(last_target_report).name}")
                self.job_log.emit(idx, f"\nSUCCESS! Animated GLB saved to:\n  {last_target_report}\n")
                self.job_finished.emit(idx, True, last_target_report)
                succeeded += 1

        self.batch_finished.emit(succeeded, total)


# ---------------------------------------------------------------------------
# Sample Prompts Guide & Non-Modal Dockable Popup
# ---------------------------------------------------------------------------

GOLDEN_RULES_HTML = (
    "<div style='line-height: 1.45; color: #e2e8f0; font-size: 12px;'>"
    "<b style='color: #38bdf8; font-size: 13px;'>The 5 Golden Rules of UniMate Prompting (UniML3D Dataset):</b><br>"
    "<b>1. Always start with <span style='color: #fbbf24;'>\"An object ...\"</span></b> — "
    "100% of UniMate's training captions were rewritten to begin with <i>\"An object ...\"</i> (never \"A man\", \"A dragon\", or \"A character\").<br>"
    "<b>2. Keep sentences short &amp; present-tense (&lt; 12–15 words)</b> — "
    "Describe pure physical body mechanics (e.g., <i>\"An object walks forward, then turns right.\"</i>).<br>"
    "<b>3. Never name weapons, props, or scenery</b> — "
    "Instead of <i>\"swings a sword\"</i> or <i>\"drinks a potion\"</i>, write <span style='color: #4ade80;'>\"as if holding something\"</span>.<br>"
    "<b>4. Specify <span style='color: #fbbf24;'>\"forward\"</span> vs. <span style='color: #fbbf24;'>\"in place\"</span></b> — "
    "Use <i>\"in place\"</i> for stationary game loops and <i>\"forward / backward / in a circle\"</i> for root movement.<br>"
    "<b>5. Supported anatomy keywords</b> — "
    "<i>left/right arm, left/right leg, both hands, head, torso, upper body, hip, knees, wings, tail, tentacles</i>."
    "</div>"
)

SAMPLE_PROMPTS_CATEGORIES: List[Tuple[str, List[Tuple[str, str, bool]]]] = [
    (
        "1. Official Showcase Prompts (examples.json & README)",
        [
            ("An object walks forward.", "Official README & examples.json baseline walk", False),
            ("An object runs forward.", "Official examples.json baseline run", False),
            ("An object jumps in place.", "Official examples.json stationary jump", False),
            ("An object talks and gestures.", "Official examples.json dialogue/emote", False),
            ("An object dances in place.", "Official examples.json stationary dance", False),
            ("An object turns around in place.", "Official examples.json 180/360 turn", False),
            ("An object attacks forward.", "Official examples.json forward melee strike", False),
        ],
    ),
    (
        "2. Bipedal / Humanoid — Locomotion & Movement",
        [
            ("An object walks forward at a steady pace.", "Clean standard walk cycle", False),
            ("An object walks forward slowly and cautiously.", "Stealth / cautious walk", False),
            ("An object walks backward slowly.", "Retreating / backing up walk", False),
            ("An object walks in a circle.", "Curved path locomotion", False),
            ("An object walks forward, then turns right and continues walking.", "Walk with 90° turn transition", False),
            ("An object walks forward and then stops to stand still.", "Walk-to-idle stop transition", False),
            ("An object jogs forward.", "Medium-speed jog cycle", False),
            ("An object sprints forward quickly.", "Fast run / sprint", False),
            ("An object runs forward and then jumps.", "Run-to-jump combo", False),
            ("An object sidesteps to the left and then to the right.", "Lateral strafe movement", False),
            ("An object crouches down and sneaks forward.", "Crouched stealth locomotion", False),
            ("An object crawls forward on its hands and knees.", "Prone / four-point crawl", False),
            ("An object stumbles forward as if losing balance.", "Injured / dizzy stagger", False),
        ],
    ),
    (
        "3. Combat, Attacks, Defending & Hit Reactions",
        [
            ("An object lunges forward and attacks with its right arm.", "Single-hand melee lunge", False),
            ("An object swings its right arm forward as if holding something.", "Sword / club / axe slash (no prop name)", False),
            ("An object swings both arms downward forcefully as if holding something.", "Heavy two-handed overhead smash", False),
            ("An object punches forward with its left hand and then its right hand.", "Left-right boxing jab combo", False),
            ("An object kicks forward with its right leg.", "Front kick attack", False),
            ("An object spins around and kicks with its leg.", "Spinning roundhouse kick", False),
            ("An object raises both arms in front of its body to block.", "Defensive guard / shield block", False),
            ("An object dodges quickly to the side.", "Quick evasive sidestep", False),
            ("An object steps backward as if hit and then regains balance.", "Damage / hit stagger reaction", False),
            ("An object falls to the ground and lies still.", "Knockdown / death animation", False),
        ],
    ),
    (
        "4. Jumps, Acrobatics, Fitness & Dance",
        [
            ("An object jumps forward with both legs.", "Standing broad jump", False),
            ("An object jumps up high and raises both arms.", "Celebration / high vertical jump", False),
            ("An object hops on its left leg.", "Single-leg hop", False),
            ("An object does a front flip in place.", "Acrobatic front somersault", False),
            ("An object does a backflip in place.", "Acrobatic back somersault", False),
            ("An object does a cartwheel to the side.", "Sideways cartwheel", False),
            ("An object squats down and then stands back up.", "Squat exercise / crouch cycle", False),
            ("An object does jumping jacks in place.", "Full-body calisthenics loop", False),
            ("An object dances energetically, moving its arms and hips.", "Upbeat full-body dance", False),
            ("An object spins around rhythmically while dancing.", "Spinning dance move", False),
        ],
    ),
    (
        "5. Idle, Gestures, Emotes & Object-Free Interactions",
        [
            ("An object stands still and breathes gently.", "Subtle breathing idle loop", False),
            ("An object stands in place and looks around from left to right.", "Idle scanning environment", False),
            ("An object waves hello with its right arm.", "Friendly greeting emote", False),
            ("An object claps both hands together in front of its chest.", "Applause / clapping emote", False),
            ("An object raises both arms above its head and cheers.", "Victory / cheering emote", False),
            ("An object crosses its arms and nods its head.", "Agreement / confident idle", False),
            ("An object shakes its head and waves its hand.", "Refusal / 'no' gesture", False),
            ("An object bends down to pick something up and stands back up.", "Loot / pickup interaction", False),
            ("An object reaches forward with its right hand as if pushing something.", "Push button / door interaction", False),
            ("An object raises its right hand to its mouth as if drinking.", "Potion / drink emote (no prop)", False),
            ("An object stretches its arms overhead and yawns.", "Tired / idle stretch", False),
            ("An object sits down and rests.", "Sit transition", False),
        ],
    ),
    (
        "6. Quadrupeds, Beasts, Dinosaurs & Monsters",
        [
            ("An object walks forward on all fours.", "Quadruped walk cycle", False),
            ("An object trots forward at a steady pace.", "Canine / equine trot", False),
            ("An object gallops forward quickly.", "Fast quadruped gallop", False),
            ("An object pounces forward and lands on the ground.", "Predator leap / pounce attack", False),
            ("An object rears up and lunges forward to attack.", "Monster / beast heavy attack", False),
            ("An object bites forward with its head and neck.", "Jaw / head lunge bite", False),
            ("An object wags its tail while standing in place.", "Happy pet / tail idle", False),
            ("An object swings its tail from side to side.", "Reptile / dragon tail sweep", False),
            ("An object shakes its head and body.", "Wet dog shake / beast idle", False),
            ("An object lowers its front body and stretches.", "Play bow / quadruped stretch", False),
        ],
    ),
    (
        "7. Winged, Avian, Serpentine, Aquatic & Insectoid",
        [
            ("An object flaps its wings in place.", "Hovering flight loop (birds, dragons, fairies)", False),
            ("An object flaps its wings and flies forward.", "Forward flight cycle", False),
            ("An object glides forward with its wings spread wide.", "Soaring / gliding pose", False),
            ("An object folds its wings and lands on the ground.", "Landing transition", False),
            ("An object slithers forward in an S-shaped motion.", "Snake / serpentine movement", False),
            ("An object swims forward by undulating its body and tail.", "Fish / aquatic creature swim", False),
            ("An object waves its tentacles rhythmically.", "Octopus / squid / eldritch rig", False),
            ("An object scutters forward quickly on its legs.", "Spider / crab / insect rig", False),
        ],
    ),
    (
        "8. Multi-Line Chains for Motion Expansion (Auto-Enables Chaining)",
        [
            (
                "An object walks forward.\nAn object stops and waves with its right arm.\nAn object turns around and walks forward.",
                "3-Part Patrol & Greet Chain (Walk → Wave → Turn & Walk)",
                True,
            ),
            (
                "An object runs forward.\nAn object jumps forward.\nAn object attacks forward with its right arm.",
                "3-Part Action Combo (Run → Jump → Attack)",
                True,
            ),
            (
                "An object stands still and looks around.\nAn object crouches down.\nAn object jumps up high and cheers.",
                "3-Part Emote Sequence (Look Around → Crouch → Jump & Cheer)",
                True,
            ),
            (
                "An object walks forward cautiously.\nAn object dodges quickly to the side.\nAn object swings its right arm forward as if holding something.",
                "3-Part Combat Encounter (Stealth Walk → Dodge → Slash)",
                True,
            ),
        ],
    ),
]


class SamplePromptsDock(QDockWidget):
    """Non-modal, movable, and dockable Sample Prompts & Prompting Guide window."""

    use_prompt_requested = Signal(str, bool)   # (prompt_text, is_chained_multiline)
    append_prompt_requested = Signal(str)      # (prompt_text_to_append)

    def __init__(self, parent: QMainWindow):
        super().__init__("UniMate Sample Prompts & Prompting Guide", parent)
        self.setObjectName("SamplePromptsDock")
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.setMinimumWidth(420)
        self.setMinimumHeight(360)

        container = QWidget()
        container.setObjectName("DockContainer")
        root = QVBoxLayout(container)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # Top control bar: Search filter & status tip
        top_ctrl = QFrame()
        top_ctrl.setObjectName("ControlBar")
        top_vbox = QVBoxLayout(top_ctrl)
        top_vbox.setContentsMargins(8, 5, 8, 5)
        top_vbox.setSpacing(4)

        search_row = QHBoxLayout()
        search_row.setSpacing(6)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Filter sample prompts (e.g., walk, attack, wings, tail, dance, chain)...")
        self.search_edit.textChanged.connect(self._filter_prompts)
        btn_clear = QPushButton("Clear")
        btn_clear.setFixedWidth(56)
        btn_clear.setStyleSheet("padding: 3px 6px; font-size: 11px;")
        btn_clear.clicked.connect(self.search_edit.clear)
        search_row.addWidget(self.search_edit, 1)
        search_row.addWidget(btn_clear)
        top_vbox.addLayout(search_row)

        self.target_tab_lbl = QLabel(
            "Tip: Keep this panel open while typing! Click 'Use' to replace the active tab's prompt, or '+ Chain' to append a new line."
        )
        self.target_tab_lbl.setStyleSheet("color: #4ade80; font-size: 11px; font-weight: 600;")
        self.target_tab_lbl.setWordWrap(True)
        top_vbox.addWidget(self.target_tab_lbl)

        root.addWidget(top_ctrl)

        # Golden Rules Banner
        self.rules_frame = QFrame()
        self.rules_frame.setObjectName("AdvancedOptionsBar")
        rules_layout = QVBoxLayout(self.rules_frame)
        rules_layout.setContentsMargins(8, 6, 8, 6)
        rules_lbl = QLabel(GOLDEN_RULES_HTML)
        rules_lbl.setWordWrap(True)
        rules_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rules_layout.addWidget(rules_lbl)
        root.addWidget(self.rules_frame)

        # Scrollable Prompt Cards List
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(2, 2, 4, 2)
        self.scroll_layout.setSpacing(6)

        self._category_boxes: List[Tuple[QGroupBox, List[Tuple[QWidget, str]]]] = []

        for cat_title, items in SAMPLE_PROMPTS_CATEGORIES:
            grp = QGroupBox(cat_title)
            grp_vbox = QVBoxLayout(grp)
            grp_vbox.setContentsMargins(6, 12, 6, 6)
            grp_vbox.setSpacing(4)
            row_widgets: List[Tuple[QWidget, str]] = []

            for prompt_text, note_text, is_chain in items:
                row_frame = QFrame()
                row_frame.setStyleSheet(
                    "QFrame { background-color: #0f172a; border: 1px solid #334155; border-radius: 4px; }"
                )
                r_layout = QHBoxLayout(row_frame)
                r_layout.setContentsMargins(8, 4, 8, 4)
                r_layout.setSpacing(6)

                text_col = QVBoxLayout()
                text_col.setSpacing(1)
                p_lbl = QLabel(prompt_text.replace("\n", "  ↵\n"))
                p_lbl.setStyleSheet("color: #f8fafc; font-weight: 700; font-size: 11px; border: none;")
                p_lbl.setWordWrap(True)
                p_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

                n_lbl = QLabel(note_text)
                n_lbl.setStyleSheet("color: #94a3b8; font-size: 10px; border: none;")
                n_lbl.setWordWrap(True)
                text_col.addWidget(p_lbl)
                text_col.addWidget(n_lbl)
                r_layout.addLayout(text_col, 1)

                btn_col = QHBoxLayout()
                btn_col.setSpacing(4)

                btn_use = QPushButton("Use")
                btn_use.setObjectName("PrimaryButton")
                btn_use.setFixedWidth(46)
                btn_use.setStyleSheet("padding: 3px 6px; font-size: 11px;")
                btn_use.setToolTip("Set this prompt in the currently active Animation Prompt tab.")
                btn_use.clicked.connect(
                    lambda checked=False, p=prompt_text, c=is_chain: self.use_prompt_requested.emit(p, c)
                )

                btn_append = QPushButton("+ Chain")
                btn_append.setFixedWidth(56)
                btn_append.setStyleSheet("padding: 3px 5px; font-size: 11px;")
                btn_append.setToolTip(
                    "Append as a new line in the active Animation Prompt tab and enable Motion Expansion."
                )
                btn_append.clicked.connect(
                    lambda checked=False, p=prompt_text: self.append_prompt_requested.emit(p)
                )

                btn_copy = QPushButton("Copy")
                btn_copy.setFixedWidth(46)
                btn_copy.setStyleSheet("padding: 3px 5px; font-size: 11px;")
                btn_copy.setToolTip("Copy prompt text to clipboard.")
                btn_copy.clicked.connect(
                    lambda checked=False, p=prompt_text, b=btn_copy: self._copy_prompt(p, b)
                )

                btn_col.addWidget(btn_use)
                btn_col.addWidget(btn_append)
                btn_col.addWidget(btn_copy)
                r_layout.addLayout(btn_col)

                grp_vbox.addWidget(row_frame)
                searchable = f"{cat_title} {prompt_text} {note_text}".lower()
                row_widgets.append((row_frame, searchable))

            self.scroll_layout.addWidget(grp)
            self._category_boxes.append((grp, row_widgets))

        self.scroll_layout.addStretch(1)
        self.scroll_area.setWidget(self.scroll_content)
        root.addWidget(self.scroll_area, 1)

        self.setWidget(container)

    def _copy_prompt(self, text: str, btn: QPushButton):
        QApplication.clipboard().setText(text)
        self.target_tab_lbl.setText(f"✓ Copied to clipboard: {text.splitlines()[0]!r}")
        btn.setText("Copied!")

    def _filter_prompts(self, query: str):
        q = query.strip().lower()
        for grp, rows in self._category_boxes:
            any_vis = False
            for row_widget, searchable in rows:
                match = (not q) or (q in searchable)
                row_widget.setVisible(match)
                if match:
                    any_vis = True
            grp.setVisible(any_vis)


# ---------------------------------------------------------------------------
# Prompt Sub-Tab Widget
# ---------------------------------------------------------------------------

class PromptSubTab(QWidget):
    """Individual prompt configuration and progress panel for one animation."""

    source_changed = Signal(str)
    sample_prompts_requested = Signal()

    def __init__(self, index: int, default_source: str, default_dest: str):
        super().__init__()
        self.index = index
        default_prompt, default_name = DEFAULT_PROMPTS[index % len(DEFAULT_PROMPTS)]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Settings Card
        form_group = QGroupBox(f"Animation Prompt #{index + 1} Configuration")
        grid = QGridLayout(form_group)
        grid.setContentsMargins(8, 12, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(5)

        # Row 0: Source GLB + Bone Count Badge + Browse
        lbl_src = QLabel("Source Rig (.glb / .fbx):")
        lbl_src.setObjectName("FieldLabel")
        src_row = QHBoxLayout()
        src_row.setSpacing(6)
        self.src_edit = QLineEdit(default_source)
        self.src_edit.setPlaceholderText("Select your exported rigged character GLB...")
        self.src_edit.textChanged.connect(self.source_changed.emit)
        self.src_edit.textChanged.connect(self._update_bone_count)
        self.bone_badge_lbl = QLabel("Bones: --")
        self.bone_badge_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bone_badge_lbl.setMinimumWidth(95)
        src_row.addWidget(self.src_edit, 1)
        src_row.addWidget(self.bone_badge_lbl)

        btn_src = QPushButton("Browse...")
        btn_src.setFixedWidth(78)
        btn_src.clicked.connect(self._browse_source)
        grid.addWidget(lbl_src, 0, 0)
        grid.addLayout(src_row, 0, 1)
        grid.addWidget(btn_src, 0, 2)

        # Row 1: Prompt (compact 3-line box) + Sample Prompts Button
        prompt_lbl_col = QVBoxLayout()
        prompt_lbl_col.setContentsMargins(0, 0, 0, 0)
        prompt_lbl_col.setSpacing(4)
        lbl_prompt = QLabel("Animation Prompt:")
        lbl_prompt.setObjectName("FieldLabel")
        lbl_prompt.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        self.sample_prompts_btn = QPushButton("Sample Prompts")
        self.sample_prompts_btn.setObjectName("SamplePromptsBtn")
        self.sample_prompts_btn.setToolTip(
            "Open the non-modal, movable & dockable Sample Prompts & Guide window.\n"
            "You can float it or dock it on the side and continue typing in the prompt!"
        )
        self.sample_prompts_btn.clicked.connect(self.sample_prompts_requested.emit)

        prompt_lbl_col.addWidget(lbl_prompt)
        prompt_lbl_col.addWidget(self.sample_prompts_btn)
        prompt_lbl_col.addStretch(1)

        self.prompt_edit = QPlainTextEdit(default_prompt)
        self.prompt_edit.setPlaceholderText("e.g., An object walks forward.")
        self.prompt_edit.setFixedHeight(50)
        grid.addLayout(prompt_lbl_col, 1, 0)
        grid.addWidget(self.prompt_edit, 1, 1, 1, 2)

        # Row 2: Destination Directory + Distinct Output Name (combined on 1 compact row)
        lbl_dest = QLabel("Destination & Name:")
        lbl_dest.setObjectName("FieldLabel")
        dest_name_row = QHBoxLayout()
        dest_name_row.setContentsMargins(0, 0, 0, 0)
        dest_name_row.setSpacing(6)

        self.dest_edit = QLineEdit(default_dest)
        self.dest_edit.setPlaceholderText("Select base output folder...")
        btn_dest = QPushButton("Browse...")
        btn_dest.setFixedWidth(78)
        btn_dest.clicked.connect(self._browse_dest)

        lbl_name = QLabel("Output Name:")
        lbl_name.setObjectName("FieldLabel")
        self.name_edit = QLineEdit(default_name)
        self.name_edit.setPlaceholderText("e.g., walk_anim")
        self.name_edit.setMinimumWidth(130)

        dest_name_row.addWidget(self.dest_edit, 2)
        dest_name_row.addWidget(btn_dest)
        dest_name_row.addSpacing(6)
        dest_name_row.addWidget(lbl_name)
        dest_name_row.addWidget(self.name_edit, 1)

        self.preview_path_lbl = QLabel()
        self.preview_path_lbl.setObjectName("PathPreviewLabel")
        self.dest_edit.textChanged.connect(self._update_path_preview)
        self.name_edit.textChanged.connect(self._update_path_preview)
        grid.addWidget(lbl_dest, 2, 0)
        grid.addLayout(dest_name_row, 2, 1, 1, 2)
        grid.addWidget(self.preview_path_lbl, 3, 1, 1, 2)

        # Row 4: Per-Prompt Generation & Rig Controls (7 Options)
        adv_frame = QFrame()
        adv_frame.setObjectName("AdvancedOptionsBar")
        adv_grid = QGridLayout(adv_frame)
        adv_grid.setContentsMargins(8, 6, 8, 6)
        adv_grid.setHorizontalSpacing(8)
        adv_grid.setVerticalSpacing(5)

        # 1. Prompt Guidance Strength (--cfg_scale)
        lbl_cfg = QLabel("Guidance Strength (1.0–15.0):")
        lbl_cfg.setObjectName("FieldLabel")
        lbl_cfg.setToolTip("Classifier-Free Guidance scale (--cfg_scale, range: 1.0–15.0, default: 5.0). Higher values follow prompt strictly.")
        self.cfg_scale_spin = QDoubleSpinBox()
        self.cfg_scale_spin.setRange(1.0, 15.0)
        self.cfg_scale_spin.setSingleStep(0.5)
        self.cfg_scale_spin.setDecimals(1)
        self.cfg_scale_spin.setValue(5.0)
        self.cfg_scale_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cfg_scale_spin.setToolTip(lbl_cfg.toolTip())

        # 2. Random Seed (--seed)
        lbl_seed = QLabel("Random Seed (0–999999999):")
        lbl_seed.setObjectName("FieldLabel")
        lbl_seed.setToolTip("Optional integer seed (--seed, range: 0–999999999) for reproducible animations. Leave blank for random.")
        self.seed_edit = QLineEdit()
        self.seed_edit.setPlaceholderText("Random (or 0–999999999)")
        self.seed_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.seed_edit.setToolTip(lbl_seed.toolTip())

        # 3. Setting Variations (--num_repetitions)
        lbl_vars = QLabel("Setting Variations (1–10):")
        lbl_vars.setObjectName("FieldLabel")
        lbl_vars.setToolTip("Number of motion variations to generate & bake for this prompt (--num_repetitions, range: 1–10, default: 1).")
        self.variations_spin = QSpinBox()
        self.variations_spin.setRange(1, 10)
        self.variations_spin.setValue(1)
        self.variations_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.variations_spin.valueChanged.connect(self._update_path_preview)
        self.variations_spin.setToolTip(lbl_vars.toolTip())

        # 4. Animation Recovery Mode (--anim_mode)
        lbl_mode = QLabel("Recovery Mode (FK / IK):")
        lbl_mode.setObjectName("FieldLabel")
        lbl_mode.setToolTip("Blender animation baking solver (--anim_mode): FK (Forward Kinematics) or IK (Inverse Kinematics).")
        self.anim_mode_combo = QComboBox()
        self.anim_mode_combo.addItem("FK (Forward Kinematics)", "fk")
        self.anim_mode_combo.addItem("IK (Inverse Kinematics)", "ik")
        self.anim_mode_combo.setToolTip(lbl_mode.toolTip())

        # 5. Extra Bones Strategy (--extra_bones_strategy)
        lbl_extra = QLabel("Extra Bones Strategy:")
        lbl_extra.setObjectName("FieldLabel")
        lbl_extra.setToolTip("How non-animated helper/leaf bones are handled (--extra_bones_strategy): merge, keep, or remove.")
        self.extra_bones_combo = QComboBox()
        self.extra_bones_combo.addItem("merge (Transfer weights to parent)", "merge")
        self.extra_bones_combo.addItem("keep (Keep extra bones in rest pose)", "keep")
        self.extra_bones_combo.addItem("remove (Delete extra bones & verts)", "remove")
        self.extra_bones_combo.setToolTip(lbl_extra.toolTip())

        # 6. Rest Pose Rotation Fix (--rest_rotation)
        lbl_rot = QLabel("Rest Pose Rotation (-360°–360°):")
        lbl_rot.setObjectName("FieldLabel")
        lbl_rot.setToolTip("Pre-rotate character rig before inference (--rest_rotation, range: -360° to 360° per X/Y/Z axis, e.g., x90, y180, z-90).")
        self.rest_rotation_combo = QComboBox()
        self.rest_rotation_combo.setEditable(True)
        for label_txt, val in [
            ("None (As Exported)", ""),
            ("x90 (+90° X)", "x90"),
            ("x-90 (-90° X)", "x-90"),
            ("x180 (180° X)", "x180"),
            ("y90 (+90° Y)", "y90"),
            ("y-90 (-90° Y)", "y-90"),
            ("y180 (180° Y)", "y180"),
            ("z90 (+90° Z)", "z90"),
            ("z-90 (-90° Z)", "z-90"),
            ("z180 (180° Z)", "z180"),
        ]:
            self.rest_rotation_combo.addItem(label_txt, val)
        self.rest_rotation_combo.setToolTip(lbl_rot.toolTip())

        # 7. Motion Expansion / Chaining (--motion_expand & --expand_overlap)
        lbl_expand = QLabel("Motion Expansion (2+ lines):")
        lbl_expand.setObjectName("FieldLabel")
        lbl_expand.setToolTip(
            "Chain multiple prompt segments (2+ lines or separated by '|') into one continuous motion (--motion_expand)."
        )
        expand_row = QHBoxLayout()
        expand_row.setContentsMargins(0, 0, 0, 0)
        expand_row.setSpacing(10)
        self.motion_expand_chk = QCheckBox("Chain Multi-Line Prompts (2+ segments)")
        self.motion_expand_chk.setToolTip(lbl_expand.toolTip())
        self.motion_expand_chk.toggled.connect(self._on_motion_expand_toggled)

        lbl_overlap = QLabel("Overlap Frames (1–30):")
        lbl_overlap.setObjectName("FieldLabel")
        self.expand_overlap_spin = QSpinBox()
        self.expand_overlap_spin.setRange(1, 30)
        self.expand_overlap_spin.setValue(10)
        self.expand_overlap_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.expand_overlap_spin.setFixedWidth(75)
        self.expand_overlap_spin.setEnabled(False)
        self.expand_overlap_spin.setToolTip("Frame overlap between chained motion segments (--expand_overlap, range: 1–30, default: 10).")

        lbl_range_note = QLabel("Note: Defaults — Guidance: 5.0 | Variations: 1 | Overlap: 10 frames")
        lbl_range_note.setStyleSheet("color: #94a3b8; font-size: 11px; font-style: italic;")

        expand_row.addWidget(self.motion_expand_chk)
        expand_row.addSpacing(12)
        expand_row.addWidget(lbl_overlap)
        expand_row.addWidget(self.expand_overlap_spin)
        expand_row.addSpacing(12)
        expand_row.addWidget(lbl_range_note)
        expand_row.addStretch(1)

        # Layout the 7 options in a clean 3-column grid inside adv_frame
        adv_grid.addWidget(lbl_cfg, 0, 0)
        adv_grid.addWidget(self.cfg_scale_spin, 0, 1)
        adv_grid.addWidget(lbl_seed, 0, 2)
        adv_grid.addWidget(self.seed_edit, 0, 3)
        adv_grid.addWidget(lbl_vars, 0, 4)
        adv_grid.addWidget(self.variations_spin, 0, 5)

        adv_grid.addWidget(lbl_mode, 1, 0)
        adv_grid.addWidget(self.anim_mode_combo, 1, 1)
        adv_grid.addWidget(lbl_extra, 1, 2)
        adv_grid.addWidget(self.extra_bones_combo, 1, 3)
        adv_grid.addWidget(lbl_rot, 1, 4)
        adv_grid.addWidget(self.rest_rotation_combo, 1, 5)

        adv_grid.addWidget(lbl_expand, 2, 0)
        adv_grid.addLayout(expand_row, 2, 1, 1, 5)

        adv_grid.setColumnStretch(1, 1)
        adv_grid.setColumnStretch(3, 1)
        adv_grid.setColumnStretch(5, 1)

        grid.addWidget(adv_frame, 4, 0, 1, 3)

        layout.addWidget(form_group)

        # Progress Card (Compact 1-Row Status + Progress Bar + Open Output Folder)
        prog_group = QGroupBox(f"Prompt #{index + 1} Live Progress && Console")
        prog_layout = QVBoxLayout(prog_group)
        prog_layout.setContentsMargins(8, 12, 8, 8)
        prog_layout.setSpacing(5)

        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        self.status_lbl = QLabel("Idle — Ready")
        self.status_lbl.setStyleSheet("color: #94a3b8; font-weight: 600; font-size: 12px;")
        self.status_lbl.setMinimumWidth(140)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(18)

        self.open_folder_btn = QPushButton("Open Output Folder")
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self._open_output_folder)

        status_row.addWidget(self.status_lbl)
        status_row.addWidget(self.progress_bar, 1)
        status_row.addWidget(self.open_folder_btn)
        prog_layout.addLayout(status_row)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Detailed pipeline logs for this prompt will stream here...")
        font = QFont("Consolas", 9)
        self.log_box.setFont(font)
        prog_layout.addWidget(self.log_box, 1)

        layout.addWidget(prog_group, 1)
        self._update_path_preview()
        self._update_bone_count()

    def _update_bone_count(self):
        count, label = inspect_rig_bone_count(self.src_edit.text())
        self.bone_badge_lbl.setText(label)
        if count is None:
            self.bone_badge_lbl.setStyleSheet(
                "background-color: #1e293b; color: #94a3b8; border: 1px solid #475569; "
                "border-radius: 4px; padding: 3px 8px; font-weight: 600;"
            )
            self.bone_badge_lbl.setToolTip("Select a valid .glb / .gltf rig to inspect bone count.")
        elif count == 0:
            self.bone_badge_lbl.setStyleSheet(
                "background-color: #450a0a; color: #f87171; border: 1px solid #dc2626; "
                "border-radius: 4px; padding: 3px 8px; font-weight: 700;"
            )
            self.bone_badge_lbl.setToolTip("Warning: No skinned joints detected in this file.")
        elif count > 71:
            self.bone_badge_lbl.setStyleSheet(
                "background-color: #172554; color: #38bdf8; border: 1px solid #2563eb; "
                "border-radius: 4px; padding: 3px 8px; font-weight: 700;"
            )
            self.bone_badge_lbl.setToolTip(
                f"Detected {count} bones (>71 default). UniMate Studio will automatically expand max_joints to {count + 1}."
            )
        else:
            self.bone_badge_lbl.setStyleSheet(
                "background-color: #064e3b; color: #4ade80; border: 1px solid #059669; "
                "border-radius: 4px; padding: 3px 8px; font-weight: 700;"
            )
            self.bone_badge_lbl.setToolTip(f"Detected {count} bones (within standard 71-bone budget).")

    def _browse_source(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Rigged 3D Asset",
            self.src_edit.text() or str(REPO_ROOT / "assets"),
            "3D Rig Files (*.glb *.gltf *.fbx);;All Files (*.*)",
        )
        if path:
            self.src_edit.setText(str(Path(path)))

    def _browse_dest(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Destination Directory",
            self.dest_edit.text() or str(REPO_ROOT / "outputs"),
        )
        if folder:
            self.dest_edit.setText(str(Path(folder)))

    def _on_motion_expand_toggled(self, checked: bool):
        self.expand_overlap_spin.setEnabled(checked)
        if checked:
            self.prompt_edit.setPlaceholderText(
                "Motion Expansion active: Enter 1 prompt segment per line (or separate with '|'), e.g.:\n"
                "An object walks forward.\n"
                "An object jumps in place."
            )
            if self.cfg_scale_spin.value() <= 1.0:
                self.cfg_scale_spin.setValue(5.0)
        else:
            self.prompt_edit.setPlaceholderText("e.g., An object walks forward.")

    def get_parsed_prompts(self) -> List[str]:
        raw = self.prompt_edit.toPlainText().strip()
        if not raw:
            return []
        if self.motion_expand_chk.isChecked():
            # Split by newline or '|' for chained segments
            parts = [p.strip() for line in raw.splitlines() for p in line.split("|") if p.strip()]
            return parts
        return [" ".join(raw.split())]

    def get_seed_value(self) -> Optional[int]:
        txt = self.seed_edit.text().strip()
        if not txt or txt.lower() == "random":
            return None
        try:
            return int(txt)
        except ValueError:
            return None

    def get_rest_rotation_value(self) -> str:
        idx = self.rest_rotation_combo.currentIndex()
        curr_text = self.rest_rotation_combo.currentText().strip()
        if idx >= 0 and curr_text == self.rest_rotation_combo.itemText(idx):
            return str(self.rest_rotation_combo.itemData(idx) or "")
        if curr_text.lower().startswith("none"):
            return ""
        return curr_text

    def _update_path_preview(self):
        base = self.dest_edit.text().strip() or str(REPO_ROOT / "outputs")
        name = re.sub(r'[<>:"/\\|?*]+', "_", self.name_edit.text().strip()) or f"animation_{self.index + 1}"
        if name.lower().endswith(".glb"):
            name = name[:-4]
        out_folder = Path(base) / name
        num_vars = self.variations_spin.value() if hasattr(self, "variations_spin") else 1
        if num_vars > 1:
            self.preview_path_lbl.setText(
                f"Output path: {out_folder}  →  {name}_v1.glb .. {name}_v{num_vars}.glb ({num_vars} variations)"
            )
        else:
            out_glb = out_folder / f"{name}.glb"
            self.preview_path_lbl.setText(f"Output path: {out_folder}  →  {out_glb.name}")

    def _open_output_folder(self):
        base = self.dest_edit.text().strip() or str(REPO_ROOT / "outputs")
        name = re.sub(r'[<>:"/\\|?*]+', "_", self.name_edit.text().strip()) or f"animation_{self.index + 1}"
        if name.lower().endswith(".glb"):
            name = name[:-4]
        target = Path(base) / name
        if not target.exists():
            target = Path(base)
        if target.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def reset_progress(self):
        self.progress_bar.setValue(0)
        self.status_lbl.setText("Queued...")
        self.status_lbl.setStyleSheet("color: #60a5fa; font-weight: 600; font-size: 13px;")
        self.log_box.clear()


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class UniMateStudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UniMate Studio — 3D Rig Animation Generator")
        self.resize(980, 620)

        self.python_exe = find_unimate_python()
        self.cuda_available = False
        self.selected_device = "cpu"
        self.check_worker: Optional[DependencyCheckWorker] = None
        self.install_worker: Optional[PackageInstallerWorker] = None
        self.anim_worker: Optional[BatchAnimationWorker] = None

        self._build_ui()
        self._apply_dark_theme()

        # Run dependency and hardware checks automatically on startup
        self.run_dependency_checks()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 6, 8, 8)
        root_layout.setSpacing(6)

        # Compact single-row Header banner
        header_frame = QFrame()
        header_frame.setObjectName("HeaderBanner")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(12, 5, 12, 5)
        header_layout.setSpacing(10)

        title_lbl = QLabel("UniMate Animation Studio")
        title_lbl.setStyleSheet("color: #f8fafc; font-size: 15px; font-weight: 700; background: transparent;")
        subtitle_lbl = QLabel("— Batch Text-to-Animation Generator for Rigged Characters (.glb / .fbx)")
        subtitle_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; background: transparent;")

        self.hw_badge = QLabel("Checking hardware...")
        self.hw_badge.setStyleSheet(
            "background: #1e293b; color: #cbd5e1; border: 1px solid #334155; "
            "padding: 3px 10px; border-radius: 4px; font-weight: 600; font-size: 11px;"
        )

        header_layout.addWidget(title_lbl)
        header_layout.addWidget(subtitle_lbl, 1)
        header_layout.addWidget(self.hw_badge)
        root_layout.addWidget(header_frame)

        # Top-level tabs: [Prompt Batch Animations]  [Dependencies & Hardware]
        self.main_tabs = QTabWidget()
        self.main_tabs.setObjectName("MainTabs")
        self.main_tab_bar = HalfWidthTabBar(self.main_tabs)
        self.main_tab_bar.setExpanding(True)
        self.main_tabs.setTabBar(self.main_tab_bar)
        self.prompts_tab = QWidget()
        self.deps_tab = QWidget()

        self.main_tabs.addTab(self.prompts_tab, "Prompt Batch Animations")
        self.main_tabs.addTab(self.deps_tab, "Dependencies && Hardware")
        root_layout.addWidget(self.main_tabs, 1)

        self._build_prompts_tab()
        self._build_deps_tab()

        # Non-modal, movable, and dockable Sample Prompts & Prompting Guide window
        self.sample_prompts_dock = SamplePromptsDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.sample_prompts_dock)
        self.sample_prompts_dock.setFloating(True)
        self.sample_prompts_dock.resize(490, 620)
        self.sample_prompts_dock.hide()
        self.sample_prompts_dock.use_prompt_requested.connect(self._on_use_sample_prompt)
        self.sample_prompts_dock.append_prompt_requested.connect(self._on_append_sample_prompt)
        self._sample_dock_positioned = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "main_tab_bar"):
            self.main_tab_bar.updateGeometry()

    def closeEvent(self, event):
        if self.anim_worker and self.anim_worker.isRunning():
            self.anim_worker.request_stop()
            self.anim_worker.wait(2000)
        if self.check_worker and self.check_worker.isRunning():
            self.check_worker.wait(3000)
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Tab 1: Prompts & Batch Animation
    # ------------------------------------------------------------------
    def _build_prompts_tab(self):
        layout = QVBoxLayout(self.prompts_tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Top control bar: Number of Animations (1-10) + Sync Source button
        top_bar = QFrame()
        top_bar.setObjectName("ControlBar")
        top_row = QHBoxLayout(top_bar)
        top_row.setContentsMargins(10, 4, 10, 4)

        lbl_count = QLabel("Number of Animations to Process (1–10):")
        lbl_count.setStyleSheet("color: #e2e8f0; font-weight: 600; font-size: 12px; background: transparent;")
        self.anim_count_spin = QSpinBox()
        self.anim_count_spin.setRange(1, 10)
        self.anim_count_spin.setValue(1)
        self.anim_count_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.anim_count_spin.setFixedWidth(70)
        self.anim_count_spin.setMinimumHeight(24)
        self.anim_count_spin.valueChanged.connect(self._on_anim_count_changed)

        btn_sync_src = QPushButton("Apply Prompt 1 Source && Destination to All Tabs")
        btn_sync_src.clicked.connect(self._sync_first_tab_paths_to_all)

        self.overall_status_lbl = QLabel("Ready")
        self.overall_status_lbl.setStyleSheet("color: #94a3b8; font-weight: 600; background: transparent;")

        top_row.addWidget(lbl_count)
        top_row.addWidget(self.anim_count_spin)
        top_row.addSpacing(10)
        top_row.addWidget(btn_sync_src)
        top_row.addStretch(1)
        top_row.addWidget(self.overall_status_lbl)
        layout.addWidget(top_bar)

        # Dynamic Sub-Tabs container for Prompt 1..N
        self.prompt_sub_tabs = QTabWidget()
        self.prompt_sub_tabs.setObjectName("SubTabs")
        layout.addWidget(self.prompt_sub_tabs, 1)

        default_src = str(REPO_ROOT / "assets" / "my_rig.glb")
        default_dest = str(REPO_ROOT / "outputs")
        self.sub_tab_widgets: List[PromptSubTab] = []
        for i in range(10):
            w = PromptSubTab(i, default_src, default_dest)
            w.sample_prompts_requested.connect(self.show_sample_prompts_dock)
            self.sub_tab_widgets.append(w)

        # Bottom action bar: Generate button + Stop button
        bottom_bar = QHBoxLayout()
        bottom_bar.setSpacing(8)
        self.generate_btn = QPushButton("Generate Animation")
        self.generate_btn.setObjectName("PrimaryButton")
        self.generate_btn.setMinimumHeight(34)
        self.generate_btn.clicked.connect(self.start_batch_generation)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("DangerButton")
        self.stop_btn.setMinimumHeight(34)
        self.stop_btn.setFixedWidth(95)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_batch_generation)

        bottom_bar.addWidget(self.generate_btn, 1)
        bottom_bar.addWidget(self.stop_btn)
        layout.addLayout(bottom_bar)

        # Populate initial 1 tab
        self._on_anim_count_changed(1)

    def _on_anim_count_changed(self, count: int):
        current_idx = self.prompt_sub_tabs.currentIndex()
        while self.prompt_sub_tabs.count() < count:
            idx = self.prompt_sub_tabs.count()
            if idx > 0:
                if not self.sub_tab_widgets[idx].src_edit.text().strip():
                    self.sub_tab_widgets[idx].src_edit.setText(self.sub_tab_widgets[0].src_edit.text())
                if not self.sub_tab_widgets[idx].dest_edit.text().strip():
                    self.sub_tab_widgets[idx].dest_edit.setText(self.sub_tab_widgets[0].dest_edit.text())
            self.prompt_sub_tabs.addTab(self.sub_tab_widgets[idx], f"  Prompt {idx + 1}  ")

        while self.prompt_sub_tabs.count() > count:
            self.prompt_sub_tabs.removeTab(self.prompt_sub_tabs.count() - 1)

        if 0 <= current_idx < count:
            self.prompt_sub_tabs.setCurrentIndex(current_idx)

        self.generate_btn.setText(
            "Generate Animation" if count == 1 else f"Generate All {count} Animations"
        )

    def _sync_first_tab_paths_to_all(self):
        src = self.sub_tab_widgets[0].src_edit.text()
        dest = self.sub_tab_widgets[0].dest_edit.text()
        for i in range(1, self.prompt_sub_tabs.count()):
            self.sub_tab_widgets[i].src_edit.setText(src)
            self.sub_tab_widgets[i].dest_edit.setText(dest)

    def show_sample_prompts_dock(self):
        """Show and raise the non-modal, movable & dockable Sample Prompts window."""
        if not self._sample_dock_positioned and self.sample_prompts_dock.isFloating():
            geo = self.geometry()
            self.sample_prompts_dock.move(max(40, geo.right() - 260), max(40, geo.top() + 40))
            self._sample_dock_positioned = True
        self.sample_prompts_dock.show()
        self.sample_prompts_dock.raise_()

    def _on_use_sample_prompt(self, prompt_text: str, is_chain: bool):
        idx = max(0, self.prompt_sub_tabs.currentIndex())
        w = self.sub_tab_widgets[idx]
        if is_chain or ("\n" in prompt_text):
            w.motion_expand_chk.setChecked(True)
        w.prompt_edit.setPlainText(prompt_text)
        w.prompt_edit.setFocus()
        self.sample_prompts_dock.target_tab_lbl.setText(
            f"✓ Applied to Prompt {idx + 1}: {prompt_text.splitlines()[0]!r}"
        )

    def _on_append_sample_prompt(self, prompt_text: str):
        idx = max(0, self.prompt_sub_tabs.currentIndex())
        w = self.sub_tab_widgets[idx]
        existing = w.prompt_edit.toPlainText().strip()
        combined = f"{existing}\n{prompt_text}" if existing else prompt_text
        if "\n" in combined:
            w.motion_expand_chk.setChecked(True)
        w.prompt_edit.setPlainText(combined)
        w.prompt_edit.setFocus()
        self.sample_prompts_dock.target_tab_lbl.setText(
            f"✓ Appended to Prompt {idx + 1} (Motion Expansion enabled)"
        )

    # ------------------------------------------------------------------
    # Tab 2: Dependencies & Hardware
    # ------------------------------------------------------------------
    def _build_deps_tab(self):
        layout = QVBoxLayout(self.deps_tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Compute Device Toggle & GitHub Link Group
        hw_group = QGroupBox("Compute Device && Repository")
        hw_layout = QHBoxLayout(hw_group)
        hw_layout.setContentsMargins(10, 12, 10, 8)

        dev_col = QVBoxLayout()
        dev_col.setSpacing(4)
        dev_title = QLabel("Inference Compute Device:")
        dev_title.setStyleSheet("color: #f1f5f9; font-weight: 600; font-size: 12px;")
        radio_row = QHBoxLayout()
        radio_row.setSpacing(14)
        self.device_group = QButtonGroup(self)
        self.radio_cuda = QRadioButton("NVIDIA GPU (CUDA)")
        self.radio_cpu = QRadioButton("CPU Mode (No NVIDIA GPU Required)")
        self.device_group.addButton(self.radio_cuda)
        self.device_group.addButton(self.radio_cpu)
        self.radio_cpu.setChecked(True)
        self.radio_cuda.toggled.connect(self._on_device_toggled)
        radio_row.addWidget(self.radio_cuda)
        radio_row.addWidget(self.radio_cpu)
        radio_row.addStretch(1)

        self.cuda_status_lbl = QLabel("Detecting NVIDIA GPU...")
        self.cuda_status_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        dev_col.addWidget(dev_title)
        dev_col.addLayout(radio_row)
        dev_col.addWidget(self.cuda_status_lbl)

        right_col = QVBoxLayout()
        right_col.setSpacing(3)
        links_row = QHBoxLayout()
        links_row.setSpacing(14)
        gh_link = QLabel(
            f'<a href="{GITHUB_URL}" style="color: #60a5fa; font-weight: 600; font-size: 12px; text-decoration: none;">'
            f'Official UniMate GitHub Repository ↗</a>'
        )
        gh_link.setOpenExternalLinks(True)
        hf_link = QLabel(
            f'<a href="{HF_MODEL_URL}" style="color: #fbbf24; font-weight: 600; font-size: 12px; text-decoration: none;">'
            f'Hugging Face Model ({HF_MODEL_REPO}) ↗</a>'
        )
        hf_link.setOpenExternalLinks(True)
        links_row.addStretch(1)
        links_row.addWidget(gh_link)
        links_row.addWidget(hf_link)

        root_lbl = QLabel(f"UniMate Root (..): {REPO_ROOT}")
        root_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        py_display = self.python_exe if self.python_exe else "Not Installed (Click 'Fix / Install' below)"
        self.py_lbl = QLabel(f"Python Env: {py_display}")
        self.py_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        right_col.addLayout(links_row)
        right_col.addWidget(root_lbl, alignment=Qt.AlignmentFlag.AlignRight)
        right_col.addWidget(self.py_lbl, alignment=Qt.AlignmentFlag.AlignRight)

        hw_layout.addLayout(dev_col, 1)
        hw_layout.addLayout(right_col)
        layout.addWidget(hw_group)

        # Dependencies Table
        deps_group = QGroupBox("Environment && Dependency Checks (Runs Automatically on Startup)")
        deps_vbox = QVBoxLayout(deps_group)
        deps_vbox.setContentsMargins(10, 12, 10, 8)
        deps_vbox.setSpacing(6)

        btn_row = QHBoxLayout()
        self.recheck_btn = QPushButton("Re-run Checks")
        self.recheck_btn.clicked.connect(self.run_dependency_checks)
        self.fix_all_btn = QPushButton("Fix / Install All Missing Dependencies")
        self.fix_all_btn.setObjectName("PrimaryButton")
        self.fix_all_btn.setEnabled(False)
        self.fix_all_btn.clicked.connect(self.install_all_missing)
        self.deps_summary_lbl = QLabel("Checking dependencies...")
        self.deps_summary_lbl.setStyleSheet("color: #e2e8f0; font-weight: 600;")

        btn_row.addWidget(self.deps_summary_lbl, 1)
        btn_row.addWidget(self.recheck_btn)
        btn_row.addWidget(self.fix_all_btn)
        deps_vbox.addLayout(btn_row)

        self.deps_table = QTableWidget(len(DEPENDENCIES), 4)
        self.deps_table.setHorizontalHeaderLabels(["Component", "Status", "Details / Version", "Action"])
        self.deps_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.deps_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.deps_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.deps_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.deps_table.verticalHeader().setVisible(False)
        self.deps_table.verticalHeader().setDefaultSectionSize(28)
        self.deps_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.deps_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.deps_table.setAlternatingRowColors(True)

        self.fix_buttons: Dict[str, QPushButton] = {}
        for row, spec in enumerate(DEPENDENCIES):
            name_item = QTableWidgetItem(spec.name)
            name_item.setToolTip(spec.description)
            status_item = QTableWidgetItem("Checking...")
            detail_item = QTableWidgetItem(spec.description)
            self.deps_table.setItem(row, 0, name_item)
            self.deps_table.setItem(row, 1, status_item)
            self.deps_table.setItem(row, 2, detail_item)

            btn = QPushButton("Fix / Install")
            btn.setEnabled(False)
            btn.clicked.connect(lambda checked=False, s=spec: self.install_single_dependency(s))
            self.fix_buttons[spec.key] = btn
            self.deps_table.setCellWidget(row, 3, btn)

        deps_vbox.addWidget(self.deps_table, 1)

        self.deps_log = QPlainTextEdit()
        self.deps_log.setReadOnly(True)
        self.deps_log.setMaximumHeight(95)
        self.deps_log.setPlaceholderText("Dependency installer logs will stream here...")
        self.deps_log.setFont(QFont("Consolas", 9))
        deps_vbox.addWidget(self.deps_log)

        layout.addWidget(deps_group, 1)

    def _on_device_toggled(self):
        if self.radio_cuda.isChecked() and self.cuda_available:
            self.selected_device = "cuda"
        else:
            self.selected_device = "cpu"
        self.hw_badge.setText(f"Mode: {self.selected_device.upper()}")

    # ------------------------------------------------------------------
    # Dependency Check & Install Logic
    # ------------------------------------------------------------------
    def run_dependency_checks(self):
        self.python_exe = find_unimate_python()
        if hasattr(self, "py_lbl"):
            py_display = self.python_exe if self.python_exe else "Not Installed (Click 'Fix / Install' below)"
            self.py_lbl.setText(f"Python Env: {py_display}")
        self.recheck_btn.setEnabled(False)
        self.deps_summary_lbl.setText("Running environment & hardware checks...")
        self.deps_summary_lbl.setStyleSheet("color: #60a5fa; font-weight: 600;")
        self.check_worker = DependencyCheckWorker(self.python_exe)
        self.check_worker.result_ready.connect(self._on_dependency_results)
        self.check_worker.start()

    def _on_dependency_results(self, data: dict):
        self.recheck_btn.setEnabled(True)
        deps_res = data.get("deps", {})
        cuda_res = data.get("cuda", {})

        self.cuda_available = bool(cuda_res.get("available", False))
        dev_name = cuda_res.get("device_name", "")
        if self.cuda_available:
            self.radio_cuda.setEnabled(True)
            self.radio_cuda.setChecked(True)
            self.selected_device = "cuda"
            self.cuda_status_lbl.setText(f"✓ NVIDIA GPU Detected: {dev_name} (Defaulted to NVIDIA CUDA; CPU also selectable)")
            self.cuda_status_lbl.setStyleSheet("color: #4ade80; font-weight: 600;")
            self.hw_badge.setText(f"GPU: {dev_name} (CUDA)")
            self.hw_badge.setStyleSheet(
                "background: #064e3b; color: #6ee7b7; border: 1px solid #059669; "
                "padding: 3px 10px; border-radius: 4px; font-weight: 600; font-size: 11px;"
            )
        else:
            self.radio_cpu.setChecked(True)
            self.radio_cuda.setEnabled(False)
            self.selected_device = "cpu"
            self.cuda_status_lbl.setText(
                "No NVIDIA GPU detected on this machine — NVIDIA option greyed out and defaulted to CPU mode."
            )
            self.cuda_status_lbl.setStyleSheet("color: #fbbf24; font-weight: 600;")
            self.hw_badge.setText("Compute Mode: CPU (No NVIDIA GPU)")
            self.hw_badge.setStyleSheet(
                "background: #451a03; color: #fcd34d; border: 1px solid #d97706; "
                "padding: 3px 10px; border-radius: 4px; font-weight: 600; font-size: 11px;"
            )

        missing_specs = []
        for row, spec in enumerate(DEPENDENCIES):
            info = deps_res.get(spec.key, {"ok": False, "detail": "Not checked"})
            ok = bool(info.get("ok", False))
            detail = str(info.get("detail", ""))

            status_item = self.deps_table.item(row, 1)
            detail_item = self.deps_table.item(row, 2)
            btn = self.fix_buttons[spec.key]

            if ok:
                status_item.setText("✓ Installed")
                status_item.setForeground(QColor("#4ade80"))
                detail_item.setText(f"{spec.description} [{detail}]")
                btn.setText("Installed")
                btn.setEnabled(False)
            else:
                status_item.setText("✗ Missing")
                status_item.setForeground(QColor("#f87171"))
                detail_item.setText(f"{detail} — {spec.description}")
                btn.setText("Fix / Install")
                btn.setEnabled(bool(spec.install_args))
                if spec.install_args:
                    missing_specs.append(spec)

        self._missing_specs = missing_specs
        if not missing_specs:
            self.deps_summary_lbl.setText(
                f"✓ All {len(DEPENDENCIES)} dependencies, Git repo files & Hugging Face model are installed and verified!"
            )
            self.deps_summary_lbl.setStyleSheet("color: #4ade80; font-weight: 600;")
            self.fix_all_btn.setEnabled(False)
        else:
            self.deps_summary_lbl.setText(f"⚠ {len(missing_specs)} dependency issue(s) found. Click 'Fix / Install' to resolve.")
            self.deps_summary_lbl.setStyleSheet("color: #f87171; font-weight: 600;")
            self.fix_all_btn.setEnabled(True)

    def install_single_dependency(self, spec: DependencySpec):
        self._start_installer([spec])

    def install_all_missing(self):
        if getattr(self, "_missing_specs", None):
            self._start_installer(self._missing_specs)

    def _start_installer(self, specs: List[DependencySpec]):
        self.recheck_btn.setEnabled(False)
        self.fix_all_btn.setEnabled(False)
        for b in self.fix_buttons.values():
            b.setEnabled(False)
        self.deps_log.clear()
        self.install_worker = PackageInstallerWorker(self.python_exe, specs)
        self.install_worker.log_line.connect(self._append_deps_log)
        self.install_worker.finished_install.connect(self._on_install_finished)
        self.install_worker.start()

    def _append_deps_log(self, text: str):
        self.deps_log.moveCursor(self.deps_log.textCursor().MoveOperation.End)
        self.deps_log.insertPlainText(text)
        self.deps_log.moveCursor(self.deps_log.textCursor().MoveOperation.End)

    def _on_install_finished(self, ok: bool, msg: str):
        self._append_deps_log(f"\n=== {msg} Re-checking environment... ===\n")
        self.run_dependency_checks()

    # ------------------------------------------------------------------
    # Batch Animation Execution & Auto-Tab Switching
    # ------------------------------------------------------------------
    def start_batch_generation(self):
        self.python_exe = find_unimate_python()
        if not self.python_exe or not Path(self.python_exe).is_file():
            self.main_tabs.setCurrentIndex(1)
            QMessageBox.warning(
                self,
                "Missing UniMate Python Environment",
                "The 'unimate' Python 3.10 environment is not installed on this PC yet.\n\n"
                "Please click 'Fix / Install All Missing Dependencies' in the Dependencies & Hardware tab first.",
            )
            return

        count = self.anim_count_spin.value()
        jobs: List[AnimationJob] = []

        for i in range(count):
            w = self.sub_tab_widgets[i]
            src = w.src_edit.text().strip()
            prompts = w.get_parsed_prompts()
            dest = w.dest_edit.text().strip()
            name = w.name_edit.text().strip()

            if not src or not Path(src).is_file():
                self.main_tabs.setCurrentIndex(0)
                self.prompt_sub_tabs.setCurrentIndex(i)
                QMessageBox.warning(
                    self,
                    f"Missing Source File (Prompt {i + 1})",
                    f"Please select a valid rigged .glb or .fbx file in Prompt {i + 1}.\nCurrent path: {src!r}",
                )
                return
            if not prompts:
                self.main_tabs.setCurrentIndex(0)
                self.prompt_sub_tabs.setCurrentIndex(i)
                QMessageBox.warning(
                    self,
                    f"Empty Prompt (Prompt {i + 1})",
                    f"Please enter an animation prompt in Prompt {i + 1}.",
                )
                return

            motion_expand = w.motion_expand_chk.isChecked()
            if motion_expand and len(prompts) < 2:
                self.main_tabs.setCurrentIndex(0)
                self.prompt_sub_tabs.setCurrentIndex(i)
                QMessageBox.warning(
                    self,
                    f"Motion Expansion Requires Multiple Prompts (Prompt {i + 1})",
                    f"Motion Expansion / Chaining is enabled in Prompt {i + 1}, which requires at least 2 prompt segments.\n\n"
                    f"Please enter 2 or more lines in the Animation Prompt box (or separate segments with '|'), "
                    f"or uncheck 'Chain Multi-Line Prompts'.",
                )
                return

            raw_seed_txt = w.seed_edit.text().strip()
            seed_val = w.get_seed_value()
            if raw_seed_txt and raw_seed_txt.lower() != "random" and seed_val is None:
                self.main_tabs.setCurrentIndex(0)
                self.prompt_sub_tabs.setCurrentIndex(i)
                QMessageBox.warning(
                    self,
                    f"Invalid Random Seed (Prompt {i + 1})",
                    f"Random Seed in Prompt {i + 1} must be a whole number (or left blank for Random).\nCurrent value: {raw_seed_txt!r}",
                )
                return

            if not dest:
                dest = str(REPO_ROOT / "outputs")
                w.dest_edit.setText(dest)
            if not name:
                name = f"animation_{i + 1}"
                w.name_edit.setText(name)

            cfg_scale = float(w.cfg_scale_spin.value())
            if motion_expand and cfg_scale <= 1.0:
                cfg_scale = 5.0
                w.cfg_scale_spin.setValue(5.0)

            w.reset_progress()
            jobs.append(
                AnimationJob(
                    tab_index=i,
                    source_glb=src,
                    prompts=prompts,
                    dest_dir=dest,
                    file_name=name,
                    cfg_scale=cfg_scale,
                    seed=seed_val,
                    num_repetitions=int(w.variations_spin.value()),
                    anim_mode=str(w.anim_mode_combo.currentData() or "fk"),
                    extra_bones_strategy=str(w.extra_bones_combo.currentData() or "merge"),
                    rest_rotation=w.get_rest_rotation_value(),
                    motion_expand=motion_expand,
                    expand_overlap=int(w.expand_overlap_spin.value()),
                )
            )

        # No matter what tab the user is on, jump to the Prompts tab and focus Prompt 1 immediately
        self.main_tabs.setCurrentIndex(0)
        self.prompt_sub_tabs.setCurrentIndex(0)

        self.generate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.anim_count_spin.setEnabled(False)
        self.overall_status_lbl.setText(f"Processing 1 / {count}...")
        self.overall_status_lbl.setStyleSheet("color: #60a5fa; font-weight: 700; background: transparent;")

        self.anim_worker = BatchAnimationWorker(self.python_exe, jobs, self.selected_device)
        self.anim_worker.job_started.connect(self._on_job_started)
        self.anim_worker.job_progress.connect(self._on_job_progress)
        self.anim_worker.job_log.connect(self._on_job_log)
        self.anim_worker.job_finished.connect(self._on_job_finished)
        self.anim_worker.batch_finished.connect(self._on_batch_finished)
        self.anim_worker.start()

    def stop_batch_generation(self):
        if self.anim_worker and self.anim_worker.isRunning():
            self.overall_status_lbl.setText("Stopping...")
            self.anim_worker.request_stop()

    def _on_job_started(self, tab_index: int, status_text: str):
        # Automatically snap to the sub-tab currently being processed
        self.main_tabs.setCurrentIndex(0)
        self.prompt_sub_tabs.setCurrentIndex(tab_index)
        total = self.anim_count_spin.value()
        self.overall_status_lbl.setText(f"Processing Prompt {tab_index + 1} of {total}...")

        w = self.sub_tab_widgets[tab_index]
        w.status_lbl.setText(status_text)
        w.status_lbl.setStyleSheet("color: #60a5fa; font-weight: 600; font-size: 13px;")

    def _on_job_progress(self, tab_index: int, pct: int, status_text: str):
        w = self.sub_tab_widgets[tab_index]
        w.progress_bar.setValue(pct)
        w.status_lbl.setText(status_text)

    def _on_job_log(self, tab_index: int, line: str):
        w = self.sub_tab_widgets[tab_index]
        w.log_box.moveCursor(w.log_box.textCursor().MoveOperation.End)
        w.log_box.insertPlainText(line)
        w.log_box.moveCursor(w.log_box.textCursor().MoveOperation.End)

    def _on_job_finished(self, tab_index: int, success: bool, detail: str):
        w = self.sub_tab_widgets[tab_index]
        if success:
            w.progress_bar.setValue(100)
            w.status_lbl.setText(f"✓ Completed — Saved {Path(detail).name}")
            w.status_lbl.setStyleSheet("color: #4ade80; font-weight: 700; font-size: 13px;")
            w.open_folder_btn.setEnabled(True)
        else:
            w.status_lbl.setText("✗ Failed (see log below)")
            w.status_lbl.setStyleSheet("color: #f87171; font-weight: 700; font-size: 13px;")
            w.log_box.appendPlainText(f"\n[ERROR] {detail}\n")

    def _on_batch_finished(self, succeeded: int, total: int):
        self.generate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.anim_count_spin.setEnabled(True)
        self.overall_status_lbl.setText(f"Batch Complete: {succeeded}/{total} succeeded")
        self.overall_status_lbl.setStyleSheet("color: #4ade80; font-weight: 700; background: transparent;")

    def _apply_dark_theme(self):
        self.setStyleSheet(
            """
            QWidget {
                background-color: #0f172a;
                color: #f1f5f9;
                font-family: "Segoe UI", sans-serif;
                font-size: 12px;
            }
            QMainWindow {
                background-color: #0b1120;
            }
            QFrame#HeaderBanner {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 6px;
            }
            QFrame#ControlBar {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 6px;
            }
            QFrame#AdvancedOptionsBar {
                background-color: #162032;
                border: 1px solid #334155;
                border-radius: 6px;
            }
            /* Tab Widgets & Bars */
            QTabWidget::pane {
                background-color: #111827;
                border: 1px solid #334155;
                border-radius: 6px;
                top: -1px;
            }
            QTabBar::tab {
                background-color: #1e293b;
                color: #94a3b8;
                border: 1px solid #334155;
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                padding: 5px 12px;
                margin-right: 3px;
                font-weight: 600;
            }
            QTabBar::tab:hover {
                background-color: #273549;
                color: #e2e8f0;
            }
            QTabBar::tab:selected {
                background-color: #2563eb;
                color: #ffffff;
                border-color: #3b82f6;
            }
            /* GroupBoxes */
            QGroupBox {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 6px;
                margin-top: 8px;
                font-weight: 700;
                color: #e2e8f0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                top: 1px;
                padding: 1px 6px;
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 3px;
                color: #60a5fa;
            }
            QLabel {
                background: transparent;
                color: #e2e8f0;
            }
            QLabel#FieldLabel {
                color: #cbd5e1;
                font-weight: 600;
            }
            QLabel#PathPreviewLabel {
                color: #38bdf8;
                font-size: 11px;
            }
            /* Inputs */
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #0f172a;
                color: #f8fafc;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 3px 7px;
                selection-background-color: #2563eb;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 1px solid #3b82f6;
                background-color: #090d16;
            }
            QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled, QLineEdit:disabled {
                background-color: #1e293b;
                color: #64748b;
                border-color: #334155;
            }
            QComboBox QAbstractItemView {
                background-color: #0f172a;
                color: #f8fafc;
                border: 1px solid #3b82f6;
                selection-background-color: #2563eb;
                selection-color: #ffffff;
                outline: 0;
            }
            QCheckBox {
                background: transparent;
                color: #f1f5f9;
                font-weight: 600;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
                border-radius: 3px;
                border: 1px solid #64748b;
                background-color: #0f172a;
            }
            QCheckBox::indicator:checked {
                background-color: #2563eb;
                border-color: #3b82f6;
            }
            QPlainTextEdit {
                background-color: #090d16;
                color: #cbd5e1;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 4px 6px;
            }
            /* Table */
            QTableWidget {
                background-color: #0f172a;
                alternate-background-color: #162032;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 4px;
                gridline-color: #1e293b;
            }
            QHeaderView::section {
                background-color: #1e293b;
                color: #94a3b8;
                font-weight: 700;
                border: none;
                border-bottom: 1px solid #334155;
                border-right: 1px solid #334155;
                padding: 4px 8px;
            }
            /* Radio buttons */
            QRadioButton {
                background: transparent;
                color: #f1f5f9;
                font-weight: 600;
                spacing: 6px;
            }
            QRadioButton:disabled {
                color: #475569;
            }
            /* Buttons */
            QPushButton {
                background-color: #334155;
                color: #f8fafc;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 4px 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #475569;
                border-color: #64748b;
            }
            QPushButton:disabled {
                background-color: #1e293b;
                color: #475569;
                border-color: #1e293b;
            }
            QPushButton#PrimaryButton {
                background-color: #2563eb;
                border: 1px solid #3b82f6;
                color: #ffffff;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton#PrimaryButton:hover {
                background-color: #1d4ed8;
            }
            QPushButton#PrimaryButton:disabled {
                background-color: #1e3a8a;
                border-color: #1e3a8a;
                color: #64748b;
            }
            QPushButton#DangerButton {
                background-color: #dc2626;
                border: 1px solid #ef4444;
                color: #ffffff;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton#DangerButton:hover {
                background-color: #b91c1c;
            }
            QPushButton#DangerButton:disabled {
                background-color: #450a0a;
                border-color: #450a0a;
                color: #7f1d1d;
            }
            QPushButton#SamplePromptsBtn {
                background-color: #172554;
                border: 1px solid #3b82f6;
                color: #60a5fa;
                font-size: 11px;
                font-weight: 700;
                padding: 3px 8px;
            }
            QPushButton#SamplePromptsBtn:hover {
                background-color: #1e3a8a;
                color: #93c5fd;
                border-color: #60a5fa;
            }
            QDockWidget {
                background-color: #0b1120;
                color: #f8fafc;
                font-weight: 700;
                border: 1px solid #334155;
            }
            QDockWidget::title {
                background-color: #1e293b;
                color: #38bdf8;
                padding: 5px 10px;
                border-bottom: 1px solid #334155;
            }
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            /* ProgressBar */
            QProgressBar {
                background-color: #0f172a;
                color: #f8fafc;
                border: 1px solid #334155;
                border-radius: 4px;
                text-align: center;
                font-weight: 700;
            }
            QProgressBar::chunk {
                background-color: #2563eb;
                border-radius: 3px;
            }
            """
        )


def apply_fusion_dark_palette(app: QApplication):
    """Force Qt Fusion style with a dark palette so Windows never renders white-on-white controls."""
    app.setStyle(QStyleFactory.create("Fusion"))
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#0b1120"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#f1f5f9"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#0f172a"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#162032"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#1e293b"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#f1f5f9"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#f8fafc"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#334155"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#f8fafc"))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#2563eb"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)


def main():
    # Hard guard: if a frozen .exe is ever invoked with Python CLI flags (-c or -m), exit immediately
    # so it can never recursively spawn GUI windows.
    if getattr(sys, "frozen", False) and any(arg in sys.argv[1:] for arg in ("-c", "-m")):
        sys.exit(1)

    # Single-instance guard on Windows using a named mutex
    _mutex_handle = None
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            _mutex_handle = kernel32.CreateMutexW(None, False, "Local\\UniMateStudioSingleInstanceMutex")
            if kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
                sys.exit(0)
        except Exception:
            pass

    app = QApplication(sys.argv)
    apply_fusion_dark_palette(app)
    win = UniMateStudioWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
