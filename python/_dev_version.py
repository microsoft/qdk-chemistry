"""Scikit-build-core metadata provider: appends +local for from-source dev builds."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import re
import subprocess
from pathlib import Path


def dynamic_metadata(field, settings):
    """Return version from VERSION file, appending +local if HEAD is not release-tagged."""
    assert field == "version"
    version_file = Path(settings.get("input", "../VERSION"))
    version = version_file.read_text().strip()

    # If VERSION already carries a pre-release/dev suffix (e.g. CI set it), use as-is.
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        return version

    repo_root = version_file.resolve().parent

    # Only consider appending .dev0 inside a git checkout.
    if not (repo_root / ".git").exists():
        return version

    # Check whether HEAD sits on a tag that matches the VERSION content.
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
        if result.returncode == 0:
            tag = result.stdout.strip()
            if tag in (version, f"v{version}"):
                return version
    except FileNotFoundError:
        # git not installed — not a dev build
        return version

    return f"{version}+local"
