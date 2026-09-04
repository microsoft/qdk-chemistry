"""Tests for optional dependency extra composition in pyproject metadata."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import importlib.util
import re
from pathlib import Path

import pytest


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


def test_mcp_is_optional_and_enabled_for_supported_test_installs():
    """MCP should not block the core wheel or Windows ARM64 test installs."""
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject_text = pyproject_path.read_text(encoding="utf-8")
    required_block = pyproject_text.split("dependencies = [", maxsplit=1)[1].split("]", maxsplit=1)[0]
    mcp_block = _get_optional_extra_block(pyproject_text, "mcp")
    test_block = _get_optional_extra_block(pyproject_text, "test")

    assert '"mcp>=2,<3"' not in required_block
    assert '"mcp>=2,<3"' in mcp_block
    assert "\"qdk-chemistry[mcp]; sys_platform != 'win32' or platform_machine != 'ARM64'\"" in test_block


def test_inactive_mcp_server_preserves_shared_tool_functions(monkeypatch):
    """The CLI-facing tool module can use MCP decorators without MCP installed."""
    module_path = Path(__file__).resolve().parents[1] / "src" / "qdk_chemistry" / "ui" / "_mcp.py"
    spec = importlib.util.spec_from_file_location("qdk_optional_mcp_test_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    optional_mcp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(optional_mcp)

    def shared_tool() -> str:
        return "available"

    assert optional_mcp.app.tool()(shared_tool) is shared_tool
    assert shared_tool() == "available"
    with pytest.raises(ModuleNotFoundError, match=r"qdk-chemistry\[mcp\]"):
        optional_mcp.require_mcp()
