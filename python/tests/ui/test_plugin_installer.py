"""Tests for venv-bound QDK Chemistry plugin installation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from qdk_chemistry.ui import plugin_installer
from qdk_chemistry.ui.cli import cmd_plugin, create_parser

if TYPE_CHECKING:
    from pathlib import Path


def _write_plugin(home: Path) -> Path:
    plugin_dir = home / "installed-plugins" / "qdk-chemistry" / "qdk-chemistry"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.json").write_text(
        json.dumps({"name": "qdk-chemistry", "agents": "agents/", "skills": "skills/", "mcpServers": ".mcp.json"}),
        encoding="utf-8",
    )
    (plugin_dir / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "qdk_chemistry": {
                        "type": "stdio",
                        "command": "qcmcp",
                        "timeout": 21 * 24 * 60 * 60 * 1000,
                        "env": {"QDK_REQUIRE_WORKSPACE_BINDING": "1"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    agents = plugin_dir / "agents"
    agents.mkdir()
    (agents / "chemist.agent.md").write_text("# Chemist\n", encoding="utf-8")
    skill = plugin_dir / "skills" / "qdk-chemistry-mcp"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# QDK Chemistry MCP\n", encoding="utf-8")
    return plugin_dir


def test_workspace_install_delegates_fetch_and_writes_cross_client_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copilot_home = tmp_path / ".qdk_chem" / "copilot"
    plugin_dir = _write_plugin(copilot_home)
    command = "/project/.venv/bin/qcmcp"
    calls: list[tuple[list[str], Path]] = []
    monkeypatch.setattr(
        plugin_installer,
        "_commands_for_current_environment",
        lambda _name: {"qdk_chemistry": command},
    )
    monkeypatch.setattr(
        plugin_installer,
        "_run_copilot",
        lambda arguments, *, home: calls.append((arguments, home)),
    )

    result = plugin_installer.install_plugin("qdk-chemistry@qdk-chemistry", target_dir=tmp_path)

    assert calls == [(["plugin", "install", "qdk-chemistry@qdk-chemistry"], copilot_home)]
    assert result["scope"] == "workspace"
    assert result["mcp_config"] == str(tmp_path / ".vscode" / "mcp.json")
    assert (tmp_path / ".github" / "agents" / "chemist.agent.md").is_file()
    assert (tmp_path / ".github" / "skills" / "qdk-chemistry-mcp" / "SKILL.md").is_file()
    vscode = json.loads((tmp_path / ".vscode" / "mcp.json").read_text(encoding="utf-8"))
    github = json.loads((tmp_path / ".github" / "mcp.json").read_text(encoding="utf-8"))
    assert vscode["servers"]["qdk_chemistry"]["command"] == command
    assert vscode["servers"]["qdk_chemistry"]["timeout"] == 21 * 24 * 60 * 60 * 1000
    assert github["mcpServers"]["qdk_chemistry"]["type"] == "local"
    assert github["mcpServers"]["qdk_chemistry"]["tools"] == ["*"]
    assert github["mcpServers"]["qdk_chemistry"]["timeout"] == 21 * 24 * 60 * 60 * 1000
    assert json.loads((plugin_dir / ".mcp.json").read_text())["mcpServers"]["qdk_chemistry"]["command"] == command


def test_component_copy_rejects_symlink_outside_plugin_tree(tmp_path: Path) -> None:
    source = tmp_path / "plugin" / "skills"
    source.mkdir(parents=True)
    outside = tmp_path / "secret.md"
    outside.write_text("secret\n", encoding="utf-8")
    (source / "SKILL.md").symlink_to(outside)
    destination = tmp_path / "workspace" / ".github" / "skills"

    with pytest.raises(plugin_installer.PluginInstallError, match="symlink traversal is not allowed"):
        plugin_installer._copy_component_directories([source], destination)

    assert not (destination / "SKILL.md").exists()


def test_local_install_registers_ancestor_marketplace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    copilot_home = workspace / ".qdk_chem" / "copilot"
    _write_plugin(copilot_home)
    checkout = tmp_path / "checkout"
    source = checkout / "copilot-plugins" / "qdk-chemistry"
    source.mkdir(parents=True)
    (source / "plugin.json").write_text(json.dumps({"name": "qdk-chemistry"}), encoding="utf-8")
    marketplace = checkout / ".github" / "plugin" / "marketplace.json"
    marketplace.parent.mkdir(parents=True)
    marketplace.write_text(
        json.dumps(
            {
                "name": "qdk-chemistry",
                "plugins": [{"name": "qdk-chemistry", "source": "copilot-plugins/qdk-chemistry"}],
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[list[str], Path]] = []
    monkeypatch.setattr(
        plugin_installer,
        "_commands_for_current_environment",
        lambda _name: {"qdk_chemistry": "/project/.venv/bin/qcmcp"},
    )

    def fake_copilot(arguments: list[str], *, home: Path) -> str:
        calls.append((arguments, home))
        return ""

    monkeypatch.setattr(plugin_installer, "_run_copilot", fake_copilot)

    plugin_installer.install_plugin(str(source), target_dir=workspace)

    assert calls == [
        (["plugin", "marketplace", "list"], copilot_home),
        (["plugin", "marketplace", "add", str(checkout)], copilot_home),
        (["plugin", "install", "qdk-chemistry@qdk-chemistry"], copilot_home),
    ]
    state = json.loads((workspace / ".qdk_chem" / "qdk-chemistry-plugin-bindings.json").read_text(encoding="utf-8"))
    assert state["plugins"]["qdk-chemistry"]["source"] == str(source)
    assert state["plugins"]["qdk-chemistry"]["update_spec"] == "qdk-chemistry@qdk-chemistry"


def test_local_install_reuses_registered_marketplace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "checkout"
    home = tmp_path / "copilot"
    expected_home = home
    calls: list[list[str]] = []

    def fake_copilot(arguments: list[str], *, home: Path) -> str:
        assert home == expected_home
        calls.append(arguments)
        return f"qdk-chemistry (Local: {root})\n"

    monkeypatch.setattr(plugin_installer, "_run_copilot", fake_copilot)

    plugin_installer._ensure_local_marketplace(root, "qdk-chemistry", home=home)

    assert calls == [["plugin", "marketplace", "list"]]


def test_workspace_install_preserves_existing_jsonc_mcp_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_plugin(tmp_path / ".qdk_chem" / "copilot")
    vscode_mcp = tmp_path / ".vscode" / "mcp.json"
    vscode_mcp.parent.mkdir()
    vscode_mcp.write_text(
        '{\n  // Keep this server.\n  "servers": {"existing": {"command": "existing"},},\n}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        plugin_installer,
        "_commands_for_current_environment",
        lambda _name: {"qdk_chemistry": "/project/.venv/bin/qcmcp"},
    )
    monkeypatch.setattr(plugin_installer, "_run_copilot", lambda _arguments, **_kwargs: None)

    plugin_installer.install_plugin("qdk-chemistry@qdk-chemistry", target_dir=tmp_path)

    config = json.loads(vscode_mcp.read_text(encoding="utf-8"))
    assert config["servers"]["existing"] == {"command": "existing"}
    assert "qdk_chemistry" in config["servers"]


def test_update_reapplies_recorded_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copilot_home = tmp_path / ".qdk_chem" / "copilot"
    plugin_dir = _write_plugin(copilot_home)
    command = tmp_path / ".venv" / "bin" / "qcmcp"
    command.parent.mkdir(parents=True)
    command.touch()
    plugin_installer._write_binding(
        tmp_path / ".qdk_chem",
        "qdk-chemistry",
        {
            "commands": {"qdk_chemistry": str(command)},
            "plugin_dir": str(plugin_dir),
            "source": "qdk-chemistry@qdk-chemistry",
            "update_spec": "qdk-chemistry@qdk-chemistry",
        },
    )

    def fake_update(arguments: list[str], *, home: Path) -> None:
        assert arguments == ["plugin", "update", "qdk-chemistry@qdk-chemistry"]
        assert home == copilot_home
        config = json.loads((plugin_dir / ".mcp.json").read_text(encoding="utf-8"))
        config["mcpServers"]["qdk_chemistry"]["command"] = "qcmcp"
        (plugin_dir / ".mcp.json").write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setattr(plugin_installer, "_run_copilot", fake_update)

    plugin_installer.update_plugin("qdk-chemistry", target_dir=tmp_path)

    config = json.loads((tmp_path / ".vscode" / "mcp.json").read_text(encoding="utf-8"))
    assert config["servers"]["qdk_chemistry"]["command"] == str(command)


def test_current_environment_must_be_a_venv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plugin_installer.sys, "prefix", "/usr")
    monkeypatch.setattr(plugin_installer.sys, "base_prefix", "/usr")

    with pytest.raises(plugin_installer.PluginInstallError, match="requires a virtual environment"):
        plugin_installer._commands_for_current_environment("qdk-chemistry")


def test_current_environment_requires_mcp_support(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_mcp() -> None:
        raise ModuleNotFoundError(plugin_installer.MCP_INSTALL_MESSAGE)

    monkeypatch.setattr(plugin_installer, "require_mcp", missing_mcp)

    with pytest.raises(plugin_installer.PluginInstallError, match=r"qdk-chemistry\[mcp\]"):
        plugin_installer._commands_for_current_environment("qdk-chemistry")


def test_plugin_commands_are_registered() -> None:
    args = create_parser().parse_args(
        ["plugin", "install", "qdk-chemistry@qdk-chemistry", "--target-dir", "/workspace"]
    )

    assert args.plugin_command == "install"
    assert args.source == "qdk-chemistry@qdk-chemistry"
    assert args.target_dir == "/workspace"


def test_plugin_command_forwards_workspace_target(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_install(source: str, *, name: str | None = None, target_dir: str | None = None) -> dict[str, object]:
        captured.update(source=source, name=name, target_dir=target_dir)
        return {"status": "installed", "plugin": "qdk-chemistry"}

    monkeypatch.setattr(plugin_installer, "install_plugin", fake_install)
    args = create_parser().parse_args(
        ["plugin", "install", "qdk-chemistry@qdk-chemistry", "--target-dir", "/workspace"]
    )

    cmd_plugin(args)

    assert captured == {
        "source": "qdk-chemistry@qdk-chemistry",
        "name": None,
        "target_dir": "/workspace",
    }
    assert json.loads(capsys.readouterr().out)["status"] == "installed"
