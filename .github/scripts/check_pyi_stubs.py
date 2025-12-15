#!/usr/bin/env python3
"""
Check that required .pyi stub files exist in the repository.

This script ensures that essential type stub files are not accidentally
deleted (e.g., via `git commit -a`).
"""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import sys
from pathlib import Path

# Required .pyi stub files (relative to repository root)
REQUIRED_PYI_FILES = [
    "python/src/qdk_chemistry/_core/__init__.pyi",
    "python/src/qdk_chemistry/_core/_algorithms.pyi",
    "python/src/qdk_chemistry/_core/constants.pyi",
    "python/src/qdk_chemistry/_core/data.pyi",
    "python/src/qdk_chemistry/_core/utils.pyi",
]


def find_repo_root() -> Path:
    """Find the repository root by looking for .git directory."""
    script = Path(__file__).resolve()
    return next(p for p in script.parents if (p / ".git").exists())


def main() -> int:
    """Check that all required .pyi stub files exist."""
    repo_root = find_repo_root()
    missing_files = []

    for rel_path in REQUIRED_PYI_FILES:
        full_path = repo_root / rel_path
        if not full_path.exists():
            missing_files.append(rel_path)

    if missing_files:
        print("ERROR: Required .pyi stub files are missing!")
        print("The following type stub files must exist in the repository:")
        for f in missing_files:
            print(f"  - {f}")
        print(
            "These files may have been accidentally deleted (e.g., via `git commit -a`)."
        )
        print("Please restore them before committing.")
        print("To restore deleted files, you can use:")
        print("  git checkout HEAD -- <file>")
        return 1

    print(f"All {len(REQUIRED_PYI_FILES)} required .pyi stub files are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
