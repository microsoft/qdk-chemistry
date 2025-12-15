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

# Expected content for placeholder stub files
STUB_PLACEHOLDER = "# This file is a placeholder and will be replaced with generated stubs on first import\n"


def find_repo_root() -> Path:
    """Find the repository root by looking for .git directory."""
    script = Path(__file__).resolve()
    return next(p for p in script.parents if (p / ".git").exists())


def main() -> int:
    """Check that all required .pyi stub files exist and are placeholders."""
    repo_root = find_repo_root()
    missing_files = []
    modified_files = []

    for rel_path in REQUIRED_PYI_FILES:
        full_path = repo_root / rel_path
        if not full_path.exists():
            missing_files.append(rel_path)
        elif full_path.read_text() != STUB_PLACEHOLDER:
            modified_files.append(rel_path)

    if missing_files or modified_files:
        print("ERROR: Required .pyi stub files are missing or modified!")
        if missing_files:
            print("\nMissing files:")
            for f in missing_files:
                print(f"  - {f}")
        if modified_files:
            print("\nModified files (should be placeholder stubs):")
            for f in modified_files:
                print(f"  - {f}")
        print(
            "\nThese files may have been accidentally deleted or changed "
            "(e.g., via `git commit -a`)."
        )
        print("Please restore them before committing.")
        print("To restore deleted files, you can use:")
        print("  git checkout HEAD -- <file>")
        return 1

    print(
        f"All {len(REQUIRED_PYI_FILES)} required .pyi stub files are present and valid."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
