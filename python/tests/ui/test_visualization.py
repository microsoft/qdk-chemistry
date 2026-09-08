"""Security tests for MCP Apps visualization resources."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest

pytest.importorskip("mcp")

from mcp.server.apps import APP_MIME_TYPE, Apps

from qdk_chemistry.ui import visualization


def _registered_apps(monkeypatch: pytest.MonkeyPatch) -> Apps:
    monkeypatch.setattr(visualization, "_WIDGETS_AVAILABLE", True)
    apps = Apps()
    visualization.register_visualization_tools(apps)
    return apps


def test_visualization_tools_use_static_apps_resources(monkeypatch):
    apps = _registered_apps(monkeypatch)
    resources = {str(binding.resource.uri): binding.resource for binding in apps.resources()}
    tools = {binding.fn.__name__: binding for binding in apps.tools()}

    assert set(resources) == {
        "ui://qdk-chem-mcp/circuit-viewer",
        "ui://qdk-chem-mcp/orbital-entanglement",
        "ui://qdk-chem-mcp/molecule-viewer",
        "ui://qdk-chem-mcp/scatter-plot",
    }
    assert set(tools) == {
        "visualize_circuit",
        "visualize_orbital_entanglement",
        "visualize_molecule",
        "visualize_orbitals",
        "visualize_scatter_plot",
    }
    for resource in resources.values():
        assert resource.mime_type == APP_MIME_TYPE
        assert "ui/initialize" in resource.text
        assert "ui/notifications/tool-result" in resource.text
    for binding in tools.values():
        assert binding.meta is not None
        assert binding.meta["ui"]["resourceUri"] in resources


def test_visualization_result_has_structured_content_and_text_fallback():
    payload = {"label": "result"}

    result = visualization._tool_result(payload)

    assert result.structured_content == payload
    assert result.content[0].text == '{"label": "result"}'
    assert result.meta is None


def test_scatter_app_does_not_embed_tool_controlled_strings(monkeypatch):
    apps = _registered_apps(monkeypatch)

    hostile = '</title><script>window.injected=true</script><svg onload="window.injected=true">'
    resources = {str(binding.resource.uri): binding.resource for binding in apps.resources()}
    html = resources["ui://qdk-chem-mcp/scatter-plot"].text

    assert hostile not in html
    assert "textContent=text" in html
    assert "title.textContent=" in html
