"""Tests for the repository-hosted QDK Chemistry agent plugin."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_PLUGIN = _ROOT / "copilot-plugins" / "qdk-chemistry"
_EXPECTED_AGENTS = {"chemist", "quantum-agent", "reporter", "researcher", "reviewer"}
_EXPECTED_SKILLS = {
    "qdk-chemistry-coding",
    "qdk-chemistry-mcp",
    "qdk-chemistry-overview",
    "remote-execution",
}


def test_marketplace_lists_qdk_chemistry_plugin() -> None:
    marketplace = json.loads((_ROOT / ".github" / "plugin" / "marketplace.json").read_text(encoding="utf-8"))
    assert marketplace["plugins"][0]["source"] == "copilot-plugins/qdk-chemistry"


def test_plugin_owns_complete_agent_and_skill_catalog() -> None:
    agents = {path.name.removesuffix(".agent.md") for path in (_PLUGIN / "agents").glob("*.agent.md")}
    skills = {path.parent.name for path in (_PLUGIN / "skills").glob("*/SKILL.md")}

    assert agents == _EXPECTED_AGENTS
    assert skills == _EXPECTED_SKILLS
    assert "{{QDK_CHEMISTRY_VERSION}}" not in "".join(
        path.read_text(encoding="utf-8") for path in _PLUGIN.rglob("*") if path.is_file()
    )


def test_plugin_mcp_requires_runtime_workspace_binding() -> None:
    path = _PLUGIN / ".mcp.json"
    text = path.read_text(encoding="utf-8")
    server = json.loads(text)["mcpServers"]["qdk_chemistry"]
    assert server["timeout"] == 21 * 24 * 60 * 60 * 1000
    assert server["env"]["QDK_REQUIRE_WORKSPACE_BINDING"] == "1"
    assert "${workspaceFolder}" not in text
