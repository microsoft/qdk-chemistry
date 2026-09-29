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

from qdk_chemistry.ui.config import config

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
    original_directories = (config.scratch_dir, config.projects_dir, config.cache_dir, config.jobs_dir)
    monkeypatch.delenv("QDK_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("QDK_CACHE_DIR", raising=False)
    monkeypatch.delenv("QDK_JOBS_DIR", raising=False)
    monkeypatch.setattr(workspace, "_WORKSPACE_ROOT", None)
    yield
    config.scratch_dir, config.projects_dir, config.cache_dir, config.jobs_dir = original_directories
    os.chdir(original_cwd)


def test_configure_workspace_sets_stable_absolute_root(tmp_path) -> None:
    original_cwd = Path.cwd()
    result = workspace.configure_workspace(tmp_path)

    assert result == {"bound": True, "workspace_root": str(tmp_path)}
    assert os.environ["QDK_WORKSPACE_ROOT"] == str(tmp_path)
    assert config.scratch_dir == tmp_path
    assert config.projects_dir == tmp_path / "projects"
    assert config.cache_dir == tmp_path / "cache"
    assert config.jobs_dir == tmp_path / "jobs"
    assert all(path.is_dir() for path in (config.projects_dir, config.cache_dir, config.jobs_dir))
    assert Path.cwd() == original_cwd
    assert workspace.configure_workspace(tmp_path)["bound"] is True
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(RuntimeError, match="already bound"):
        workspace.configure_workspace(other)


def test_configure_workspace_never_changes_cwd(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_chdir(path: str | os.PathLike[str]) -> None:
        pytest.fail(f"configure_workspace unexpectedly changed directory to {path}")

    with monkeypatch.context() as context:
        context.setattr(os, "chdir", fail_chdir)

        assert workspace.configure_workspace(tmp_path)["bound"] is True


def test_configure_workspace_does_not_publish_root_when_storage_setup_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_storage_setup(_root: Path) -> None:
        raise OSError("storage setup failed")

    monkeypatch.setattr(config, "set_workspace_root", fail_storage_setup)

    with pytest.raises(OSError, match="storage setup failed"):
        workspace.configure_workspace(tmp_path)

    assert workspace.current_workspace_root() is None
    assert "QDK_WORKSPACE_ROOT" not in os.environ


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
