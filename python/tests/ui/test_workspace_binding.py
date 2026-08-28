"""Tests for QDK Chemistry plugin workspace binding."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "qdk_chemistry" / "ui" / "workspace.py"
_SPEC = importlib.util.spec_from_file_location("qdk_workspace_test_module", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
workspace = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(workspace)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _reset_workspace_binding(monkeypatch: pytest.MonkeyPatch):
    try:
        original_cwd = Path.cwd()
    except FileNotFoundError:
        original_cwd = _REPOSITORY_ROOT
        os.chdir(original_cwd)
    monkeypatch.setattr(workspace, "_WORKSPACE_ROOT", None)
    yield
    os.chdir(original_cwd)


def test_configure_workspace_sets_stable_absolute_root(tmp_path) -> None:
    result = workspace.configure_workspace(tmp_path)

    assert result == {"bound": True, "workspace_root": str(tmp_path)}
    assert os.environ["QDK_WORKSPACE_ROOT"] == str(tmp_path)
    assert Path.cwd() == tmp_path
    assert workspace.configure_workspace(tmp_path)["bound"] is True
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(RuntimeError, match="already bound"):
        workspace.configure_workspace(other)


def test_plugin_middleware_rejects_unbound_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDK_REQUIRE_WORKSPACE_BINDING", "1")

    async def no_workspace(_context):
        return None, "no roots"

    async def call_next(_context):
        return "called"

    monkeypatch.setattr(workspace, "_workspace_from_client", no_workspace)
    context = SimpleNamespace(method="tools/call", params={"name": "create_structure"})

    with pytest.raises(RuntimeError, match="Call bind_workspace"):
        asyncio.run(workspace.workspace_binding_middleware(context, call_next))

    context.params = {"name": "bind_workspace"}
    assert asyncio.run(workspace.workspace_binding_middleware(context, call_next)) == "called"
