# Contributing to UniMate-B3D

UniMate-B3D is the Blender integration (`addon/`, `backend/`) around the upstream [UniMate](https://github.com/Friedrich-M/UniMate) code in `unimate/`. Keep integration changes in `addon/` and `backend/`; change upstream files only when merging upstream.

## Development setup

1. Clone the repository and run `powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1` from its root (Python 3.10). This creates `.venv`, installs `backend/requirements.txt` and downloads the checkpoint and pose model into `models/`.
2. Install the add-on from `addon/unimate_motion` (or a ZIP built with `python scripts/package.py`) and set **Project folder** to the clone.

`.venv/`, `models/`, `cache/` and `outputs/` are ignored by Git. Never commit them, generated jobs, `.blend1` files or anything containing paths from your machine.

## Layout

| Path | Contents |
| --- | --- |
| `addon/unimate_motion/` | Blender add-on. `motion.py` and `schedule.py` have no Blender dependency and are also loaded by the worker. |
| `backend/worker.py` | Inference process. One job per run, or `--serve` to keep models loaded and read one JSON job per stdin line. |
| `backend/timeline.py` | Window planning (`plan_windows`), known-frame masks and retiming. |
| `backend/collision.py`, `backend/ground.py` | Motion cleanup. |
| `tests/` | Regression tests and small fixtures (see below). |
| `unimate/`, `configs/`, `data_process/` | Upstream UniMate. |

## Tests

```
.venv\Scripts\python.exe tests\run_all.py --blender "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
```

The suite does not load the model, so it needs no GPU. File prefixes say how each test runs:

- `test_*.py`: unittest modules (`python -m unittest discover -s tests`).
- `check_*.py`: plain Python scripts that exit with an error on failure.
- `blender_*.py`: scripts run with `blender --background --python`.
- `make_*.py`: derive intermediate inputs for later tests.

Inputs live in `tests/fixtures/`; everything the tests write goes to `tests/artifacts/` (ignored). The fixture motions are UniMate outputs for the test rigs in `tests/build_scene.py`, so they fall under the model weights' CC BY-NC 4.0 license. `blender_obstacle.py` needs the demo scene and is skipped by Blender versions older than the one that saved it.

Run the suite on the minimum supported Blender (4.2) and a current release before opening a pull request. Changes to generation should also be checked with a real model run, since the suite only replays saved outputs.

## Merging upstream

```
git remote add upstream https://github.com/Friedrich-M/UniMate.git
git fetch upstream
git merge upstream/main
```

The root `README.md` belongs to this fork; on a conflict, keep it and apply upstream's README changes to `docs/UPSTREAM_README.md`, prefixing relative links with `../`. Then set `UPSTREAM_COMMIT` in `backend/worker.py` to the merged upstream commit and run the tests.

## Releases

The maintainer bumps the version (`blender_manifest.toml`, `bl_info`, `scripts/package.py`, README install link) and rebuilds `dist/`. Pull requests should not modify `dist/`.

## Licensing

Add-on and backend contributions are GPL-3.0-or-later. Upstream code is MIT. The released model weights are CC BY-NC 4.0 and are not distributed here. Do not copy code under incompatible or more restrictive licenses (for example AGPL); integrations should go through documented data formats instead.
