"""Mesh animation: drive rigged characters with motion data and export GLB/FBX.

Entry points (all run under Blender headless, ``blender -b -P <script> -- <args>``):

- ``animate_motion``  — feature-format clip (stage-4 / generated) + ``cond.npy``
- ``animate_npz``     — export-stage motion NPZ (bone names carried in the file)
- ``animate_fbx``     — raw animation FBX/GLB clips, action transfer by bone name
- ``animate_mixamo``  — batch wrapper pairing one character with a directory of NPZs
- ``animate_lbs``     — manual NumPy FK+LBS: deform a rigged asset's skinned mesh
  directly with a motion NPZ (either flavor) and save the vertex animation
- ``preprocess_char`` — run one rigged, animated asset through the real export +
  feature pipeline in-process: writes export NPZs, motion-feature NPZs, the
  model-side ``cond.npy``, and a canonical rest-pose GLB/FBX that feature NPZs
  drive through ``animate_lbs`` cond-free

Shared plumbing lives in :mod:`.common`; rig primitives in
:mod:`data_process.utils.blender_rig`.
"""
