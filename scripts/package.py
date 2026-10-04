from pathlib import Path
import zipfile
ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "addon" / "unimate_motion"
destination = ROOT / "dist" / "unimate_motion-0.4.0.zip"
destination.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in source.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            archive.write(path, path.relative_to(source.parent))
print(destination)
