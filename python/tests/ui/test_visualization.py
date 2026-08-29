"""Security tests for MCP Apps visualization resources."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.ui import visualization


def test_widget_bridge_uses_unique_single_use_resource_uris(monkeypatch):
    monkeypatch.setattr(
        visualization,
        "_build_html",
        lambda **kwargs: kwargs["embedded_data"]["label"],
    )
    bridge = visualization._WidgetBridge(
        resource_uri="ui://test/viewer",
        component_name="Test",
        app_name="test-viewer",
        title="Test Viewer",
    )

    first = bridge.send({"label": "first"})
    second = bridge.send({"label": "second"})
    first_uri = first.meta["ui"]["resourceUri"]
    second_uri = second.meta["ui"]["resourceUri"]

    assert first_uri != second_uri
    assert first.meta["ui/resourceUri"] == first_uri
    assert bridge.receive_html(first_uri.rsplit("/", maxsplit=1)[-1]) == "first"
    assert bridge.receive_html(second_uri.rsplit("/", maxsplit=1)[-1]) == "second"
    assert "already opened" in bridge.receive_html(first_uri.rsplit("/", maxsplit=1)[-1])


def test_widget_bridge_discards_expired_payloads(monkeypatch):
    now = 10.0
    monkeypatch.setattr(visualization.time, "monotonic", lambda: now)
    bridge = visualization._WidgetBridge(
        resource_uri="ui://test/viewer",
        component_name="Test",
        app_name="test-viewer",
        title="Test Viewer",
        payload_ttl=5,
    )

    result = bridge.send({"secret": "value"})
    token = result.meta["ui"]["resourceUri"].rsplit("/", maxsplit=1)[-1]
    now = 15.0

    html = bridge.receive_html(token)

    assert "expired" in html
    assert "secret" not in html


class _App:
    def __init__(self) -> None:
        self.resources = {}
        self.tools = {}

    def resource(self, uri: str, **_kwargs):
        def register(function):
            self.resources[uri] = function
            return function

        return register

    def tool(self, **_kwargs):
        def register(function):
            self.tools[function.__name__] = function
            return function

        return register


def test_scatter_plot_escapes_tool_controlled_strings(monkeypatch):
    monkeypatch.setattr(visualization, "_WIDGETS_AVAILABLE", True)
    app = _App()
    visualization.register_visualization_tools(app)

    hostile = '</title><script>window.injected=true</script><svg onload="window.injected=true">'
    payload = {
        "title": hostile,
        "x_label": hostile,
        "y_label": hostile,
        "log_x": False,
        "log_y": False,
        "series": [{"x": [1], "y": [2], "name": hostile, "text": [hostile]}],
    }
    monkeypatch.setattr(
        visualization._WidgetBridge,
        "receive",
        lambda _self, _token: payload,
    )
    html = app.resources["ui://qdk-chem-mcp/scatter-plot/{token}"]("test-token")

    assert hostile not in html
    assert "\\u003c/title\\u003e\\u003cscript\\u003e" in html
    assert "document.title=chart.title" in html
    assert "textContent=text" in html
    assert "tip.replaceChildren()" in html