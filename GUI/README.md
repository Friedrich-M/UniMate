# UniMate Animation Studio — Desktop GUI

A sleek, dark-mode PySide6 desktop application for **UniMate** that streamlines single and batch 3D character animation (`.glb` / `.fbx`), provides a movable & dockable **Sample Prompts & 5 Golden Rules Guide**, and includes a **1-Click Zero-Setup Dependency & Hardware Installer** for Windows.

---

## Key Features

### 1. Batch Text-to-Animation Studio (1–10 Prompts)
- **Batch Queue (1–10 Animations):** Configure up to 10 distinct prompts in sub-tabs and generate them all in a single batch run with automatic tab focus switching and live console streaming.
- **Automatic Skeletal Bone Counter & `max_joints` Expansion:** Inspects `.glb` / `.gltf` rigs on selection, displays a color-coded bone badge (`Bones: N`), and automatically expands `max_joints` in `config.json` when a rig exceeds the default 71-bone budget.
- **Smart Rig Preprocessing Cache:** Reuses preprocessed Blender canonical rigs across batch jobs when the source `.glb` and rest rotation are unchanged.
- **7 Per-Prompt Generation & Rig Controls (with Range Notes):**
  - **Guidance Strength (`1.0–15.0`):** Classifier-Free Guidance scale (`--cfg_scale`, default `5.0`).
  - **Random Seed (`0–999999999`):** Optional integer seed for reproducible motions (or leave blank for random).
  - **Setting Variations (`1–10`):** Generate and bake multiple motion variations (`_v1.glb .. _vN.glb` & `.fbx`) per prompt (`--num_repetitions`).
  - **Recovery Mode (`FK / IK`):** Choose Forward Kinematics (`fk`) or Inverse Kinematics (`ik`) during Blender motion baking (`--anim_mode`).
  - **Extra Bones Strategy (`merge / keep / remove`):** Control how non-animated leaf/helper bones are handled (`--extra_bones_strategy`).
  - **Rest Pose Rotation (`-360°–360°`):** Pre-rotate rigs along X/Y/Z axes before inference (`--rest_rotation`, e.g., `x90`, `y180`, `z-90`).
  - **Motion Expansion / Chaining (`2+ lines`, Overlap `1–30` frames):** Chain multi-line prompts into a single continuous animation (`--motion_expand` & `--expand_overlap`).

### 2. Non-Modal, Movable & Dockable Sample Prompts Guide
- Click **`Sample Prompts`** under `Animation Prompt:` to open a non-modal popup containing:
  - **The 5 Golden Rules of UniMate Prompting** (derived from the `UniML3D` caption pipeline).
  - **8 Categorized Collections of Verified Prompts** (Official Showcase, Bipedal Locomotion, Combat & Defending, Jumps/Acrobatics/Dance, Idle/Gestures/Interactions, Quadrupeds & Monsters, Winged/Serpentine/Aquatic/Insectoid, and Multi-Line Motion Expansion Chains).
  - **One-Click `Use`, `+ Chain`, and `Copy` Buttons:** Insert prompts directly or append chained segments into the active prompt tab without closing the guide window.

### 3. Dependencies & Hardware Manager (1-Click Setup)
- **Automatic Startup Verification:** Checks 1 directory up (`REPO_ROOT`) for:
  - Core UniMate GitHub repository scripts (`data_process/`, `unimate/`)
  - Conda `unimate` Python 3.10 environment (auto-downloads & silently installs Miniconda3 if missing)
  - `setuptools<81`, `PyTorch 2.5.1+cu124`, `torch-geometric`, `torchdiffeq`, `transformers`, `spacy` (`en_core_web_sm`), `Motion`, `bpy==4.0.0`, and core utilities
  - Official Hugging Face checkpoint (`Linzhan/UniMate` in `outputs/unimate_uniml3d_f60_v3`)
- **NVIDIA GPU (CUDA) vs. CPU Mode Toggle:** Automatically detects NVIDIA GPUs (via PyTorch or `nvidia-smi`), defaults to CUDA when available, and gracefully falls back to CPU mode if no NVIDIA GPU is present.

---

## Quick Start

### Option A: Launch directly via Batch Script
Double-click **`Start_UniMate_GUI.bat`** inside the `GUI/` folder (or run with Python):
```powershell
python GUI/unimate_gui.py
```

### Option B: Build Standalone Windows Executable (`UniMate_Studio.exe`)
Double-click **`Build_EXE.bat`** inside the `GUI/` folder. It will automatically install `PySide6` and `PyInstaller` into the `unimate` environment and produce `GUI/UniMate_Studio.exe`.
