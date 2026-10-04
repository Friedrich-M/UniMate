"""Run the regression suite without loading the UniMate model.

    python tests/run_all.py --blender "<path to blender executable>"

test_*.py are unittest modules, check_*.py plain Python scripts and
blender_*.py scripts that run inside Blender. Python tests use the
interpreter that runs this file (it needs numpy and scipy, e.g. the project
.venv). Blender tests run in background mode from a temporary working
directory. Results are written to tests/artifacts/.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

# Order matters: the make_* steps derive inputs for later tests.
PYTHON = ["check_timeline.py", "make_transition_fixture.py", "make_collision_fixtures.py",
          "check_collision.py", "check_ground.py"]
BLENDER = [("blender_apply.py", None), ("blender_workflow.py", None), ("blender_ground.py", None),
           ("blender_transition.py", None), ("blender_cleanup.py", None),
           ("blender_frame.py", None),
           ("blender_obstacle.py", ROOT / "demo" / "UniMate_Run_Jump_Sword.blend")]

def run(command, cwd):
    print(">", " ".join(Path(str(c)).name for c in command), flush=True)
    result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    if "not a blend file" in result.stdout:
        print("  SKIPPED: this Blender cannot open a file saved by a newer version", flush=True)
        return True
    lines = [line for line in result.stdout.strip().splitlines()
             if line.strip() and "Not freed memory" not in line and "Blender quit" not in line]
    print("\n".join("  " + line[:160] for line in lines[-2:]), flush=True)
    # Blender may return 0 when a --python script raises, so check for a traceback too.
    failed = result.returncode != 0 or "Traceback (most recent call last)" in result.stdout
    if failed:
        print(result.stdout[-4000:], flush=True)
    return not failed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", help="Blender executable; Blender tests are skipped without it")
    parser.add_argument("--only", nargs="*", help="Run only these script names")
    args = parser.parse_args()
    (TESTS / "artifacts").mkdir(exist_ok=True)
    work = tempfile.mkdtemp(prefix="unimate-tests-")
    selected = lambda name: not args.only or name in args.only
    failures = []
    if selected("unittest") and not run([sys.executable, "-m", "unittest", "discover", "-s", str(TESTS)], work):
        failures.append("unittest")
    for name in PYTHON:
        if selected(name) and not run([sys.executable, str(TESTS / name)], work):
            failures.append(name)
    if args.blender:
        for name, blend in BLENDER:
            if not selected(name):
                continue
            command = [args.blender, "--background", "--factory-startup"]
            if blend:
                command.append(str(blend))
            command += ["--python-exit-code", "1", "--python", str(TESTS / name)]
            if not run(command, work):
                failures.append(name)
    else:
        print("Blender tests skipped; pass --blender to run them.")
    print("FAILED: " + ", ".join(failures) if failures else "ALL PASSED")
    return 1 if failures else 0

if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    raise SystemExit(main())
