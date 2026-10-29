"""Package the verified local export and refresh artifact checksums."""
from pathlib import Path
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent


def main():
    out = HERE / "outputs/shared_trends"
    validation = json.loads((out / "artifact_validation.json").read_text(encoding="utf-8"))
    assert validation["notebook_error_outputs"] == 0
    assert validation["original_groups_retained"] == 4610
    snapshot = json.loads((HERE / "browser/src/data.json").read_text(encoding="utf-8"))
    assert snapshot["buildStatus"] == "complete"
    offline = HERE / "browser/.data-app-offline/exports/Camera_Ticket_Trends.html"
    assert offline.exists()
    target = HERE / "Camera_Ticket_Trends.html"
    shutil.copyfile(offline, target)
    manifest_path = HERE / "outputs/deliverable_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = {HERE / name for name in manifest}
    for pattern in ("*.ipynb", "*.html", "*.py", "*.md", "*.ps1"):
        files.update(HERE.glob(pattern))
    # Include every experiment extension and its evidence, not only shared trends.
    files.update(path for path in (HERE / "outputs").rglob("*")
                 if path.is_file() and path != manifest_path)
    for path in sorted(files):
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest[path.relative_to(HERE).as_posix()] = {"bytes": path.stat().st_size, "sha256": checksum}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Updated local standalone HTML ({target.stat().st_size:,} bytes) and {len(manifest)} checksums.")


if __name__ == "__main__":
    main()
