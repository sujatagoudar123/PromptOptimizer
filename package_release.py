#!/usr/bin/env python3
"""Package the LLM Optimization Gateway into a distributable zip.

Usage:
    python package_release.py

Produces:
    dist/llm-optimization-gateway-<version>.zip

The zip is self-contained and can be extracted, then run with:
    cd llm-optimization-gateway && bash scripts/run_local.sh
"""
from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

# Paths to EXCLUDE from the zip (patterns match full relative paths or basenames)
EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".git",
    ".idea",
    ".vscode",
    ".mypy_cache",
    "node_modules",
    ".egg-info",
}
EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".log", ".swp")
EXCLUDE_NAMES = {".DS_Store", "Thumbs.db", ".env"}

ROOT = Path(__file__).resolve().parent
PROJECT_NAME = "llm-optimization-gateway"


def read_version() -> str:
    init_py = ROOT / "src" / "llm_gateway" / "__init__.py"
    for line in init_py.read_text().splitlines():
        if line.startswith("__version__"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "0.0.0"


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_DIRS:
        return True
    if any(p.endswith(".egg-info") for p in path.parts):
        return True
    if path.name in EXCLUDE_NAMES:
        return True
    if path.suffix in EXCLUDE_SUFFIXES:
        return True
    return False


def collect_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dir_path = Path(dirpath)
        # Prune excluded directories in-place
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDE_DIRS and not d.endswith(".egg-info")
        ]
        for fn in filenames:
            abs_path = dir_path / fn
            rel = abs_path.relative_to(root)
            if should_skip(rel):
                continue
            files.append(abs_path)
    return sorted(files)


def build_zip(out_path: Path, files: list[Path], *, archive_root: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            rel = f.relative_to(ROOT)
            # Put everything under a top-level folder so extraction is tidy
            arcname = f"{archive_root}/{rel.as_posix()}"
            zf.write(f, arcname=arcname)


def main() -> int:
    version = read_version()
    archive_root = f"{PROJECT_NAME}"
    out = ROOT / "dist" / f"{PROJECT_NAME}-{version}.zip"

    print(f"[package] Project:  {PROJECT_NAME}")
    print(f"[package] Version:  {version}")
    print(f"[package] Output:   {out.relative_to(ROOT)}")

    files = collect_files(ROOT)
    print(f"[package] Including {len(files)} files...")

    build_zip(out, files, archive_root=archive_root)

    size_kb = out.stat().st_size / 1024.0
    print(f"[package] Done. {out} ({size_kb:.1f} KB)")
    print()
    print("To use:")
    print(f"  unzip {out.relative_to(ROOT)}")
    print(f"  cd {archive_root}")
    print(f"  bash scripts/run_local.sh     # Linux/macOS/EC2")
    print(f"  scripts\\run_local.bat        # Windows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
