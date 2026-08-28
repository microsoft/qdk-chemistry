"""Tests for optional dependency extra composition in pyproject metadata."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import re
from pathlib import Path


def _get_optional_extra_block(pyproject_text: str, extra_name: str) -> str:
    """Extract a list-style optional dependency block by extra name."""
    match = re.search(rf"^{extra_name}\s*=\s*\[(.*?)^\]", pyproject_text, flags=re.MULTILINE | re.DOTALL)
    assert match is not None, f"Optional dependency block '{extra_name}' not found in pyproject.toml"
    return match.group(1)


def test_jupyter_extra_excludes_plugins_and_includes_widget_support():
    """The jupyter extra should not pull plugin dependencies transitively."""
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject_text = pyproject_path.read_text(encoding="utf-8")

    jupyter_block = _get_optional_extra_block(pyproject_text, "jupyter")
    assert '"ipykernel>=6.0"' in jupyter_block
    assert '"pandas>=2.0.0"' in jupyter_block
    assert '"qdk[jupyter]>=1.30.0"' in jupyter_block
    assert "qdk-chemistry[plugins]" not in jupyter_block
