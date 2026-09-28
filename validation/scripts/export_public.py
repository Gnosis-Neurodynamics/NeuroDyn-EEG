"""Export only the new validation code, never legacy models, data, or results."""

import argparse
from pathlib import Path
import zipfile
from project_paths import PROJECT_ROOT, resolve_project_path


def export_public(output):
    source = PROJECT_ROOT / "validation"
    files = [source / name for name in ("__init__.py", "README.md", "requirements.txt")]
    for folder, extensions in (
        ("src", {".py"}),
        ("scripts", {".py"}),
        ("config", {".json", ".csv"}),
    ):
        files.extend(
            path
            for path in (source / folder).iterdir()
            if path.is_file() and path.suffix in extensions
        )
    plotting = source / "neurodyneeg-validation"
    files.extend(
        plotting / name for name in ("README.md", "requirements.txt", ".gitignore")
    )
    for folder in ("neurodyneeg_plots", "scripts"):
        files.extend((plotting / folder).glob("*.py"))
    # Explicit asset list: never collect figures, caches, model weights or test runtimes.
    files.extend(
        plotting / "assets" / name
        for name in ("README.md", "aal_regions.csv", "ROI_MNI_V4.nii")
    )
    output = resolve_project_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, mode="x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(PROJECT_ROOT).as_posix())
    print(f"Exported {len(files)} validation and plotting files: {output}")
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="dist/neurodyn_validation.zip")
    export_public(p.parse_args().output)
