"""Validate bundled files inside a PyInstaller executable."""

from __future__ import annotations

import argparse
from pathlib import Path

from PyInstaller.archive.readers import CArchive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify required files are present in a PyInstaller bundle.",
    )
    parser.add_argument("exe", type=Path, help="Path to the bundled executable.")
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        help="Required file name to check inside the bundle (repeatable).",
    )
    parser.add_argument(
        "--write",
        type=Path,
        default=None,
        help="Optional path to write a manifest of bundled files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exe_path = args.exe
    if not exe_path.exists():
        raise FileNotFoundError(f"Executable not found: {exe_path}")

    archive = CArchive(str(exe_path))
    archive.open()
    names = [entry[0] for entry in archive.toc]

    missing = [name for name in args.require if name not in names]
    if missing:
        missing_list = ", ".join(missing)
        raise SystemExit(f"Missing bundled files: {missing_list}")

    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text("\n".join(sorted(names)), encoding="utf-8")

    print(f"Bundle OK. {len(names)} entries found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
