"""MCP Apps visualisation tools for qdk-chemistry.

This module conditionally registers ``ui://`` resources backed by
the JavaScript components shipped with ``qsharp_widgets`` and
exposes interactive MCP tools:

* ``visualize_circuit``  - interactive quantum-circuit diagram
* ``visualize_orbital_entanglement`` - orbital-entanglement chord diagram
* ``visualize_molecule`` - interactive 3D molecule viewer
* ``visualize_orbitals`` - 3D molecule viewer with orbital isosurfaces

These tools are only registered when ``qsharp_widgets`` is installed.
The tools follow the same conventions as the rest of ``tools.py``:
they accept a ``project_name`` / filename pair, load a qdk/chemistry
data object, and return either an error string or a list of
``TextContent`` items with JSON data for the MCP Apps host.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# ruff: noqa: ARG001
# MCP tool functions accept ``project_name`` consumed by the
# ``@validate_project`` decorator.

from __future__ import annotations

import json
import pathlib
import re
import threading

from mcp.types import CallToolResult, TextContent

from qdk_chemistry import data

from .io import load_data_object
from .validation import validate_project

__all__ = ["register_visualization_tools"]

# ---------------------------------------------------------------------------
# Check for qsharp_widgets availability
# ---------------------------------------------------------------------------

try:
    import qsharp_widgets as _qsharp_widgets

    _WIDGETS_AVAILABLE = True
except ImportError:
    _WIDGETS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Locate the JavaScript / CSS bundle shipped with qsharp_widgets
# ---------------------------------------------------------------------------

_WIDGETS_CACHE: dict[str, pathlib.Path] = {}


def _widgets_static_dir() -> pathlib.Path:
    """Return the ``qsharp_widgets/static`` directory, raising if missing."""
    if "static" not in _WIDGETS_CACHE:
        _WIDGETS_CACHE["static"] = pathlib.Path(_qsharp_widgets.__file__).parent / "static"
    return _WIDGETS_CACHE["static"]


# ---------------------------------------------------------------------------
# HTML builder (uses MCP Apps SDK + patched widget bundle)
# ---------------------------------------------------------------------------


def _build_html(
    *,
    title: str,
    component_name: str,
    app_name: str,
    embedded_data_json: str | None = None,
    min_height: int = 500,
) -> str:
    """Build a self-contained HTML page for a qsharp-widgets component.

    The JavaScript and CSS from ``qsharp_widgets/static`` are inlined so
    the page can be served via ``ui://`` without any external resources.

    When *embedded_data_json* is provided (a JSON string), the widget
    renders immediately on page load with that data — no MCP Apps host
    handshake or message protocol required.
    """
    static = _widgets_static_dir()
    js_text = (static / "index.js").read_text(encoding="utf-8")
    css_text = (static / "index.css").read_text(encoding="utf-8")

    # Rewrite the ESM export to a window assignment so a second
    # <script type=module> can access the widget render functions.
    # The minified variable names change across builds, so match
    # the pattern rather than hardcoding specific identifiers.
    m = re.search(r"export\{(\w+) as default,(\w+) as mdRenderer\}", js_text)
    if m:
        default_var, md_var = m.group(1), m.group(2)
        js_patched = js_text.replace(
            m.group(0),
            f"window.__qdk_widget={{default:{default_var},mdRenderer:{md_var}}}",
        )
    else:
        # Fallback: leave as-is and hope ESM import works
        js_patched = js_text

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8" />\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        "<title>" + title + "</title>\n"
        "<style>\n"
        "  :root { color-scheme: light dark; }\n"
        "  html, body {\n"
        "    margin: 0; padding: 0;\n"
        "    font-family: system-ui, -apple-system, sans-serif;\n"
        "  }\n"
        "  #widget-root {\n"
        "    width: 100%;\n"
        "  }\n"
        "  #loading { text-align: center; padding: 2em; opacity: 0.6; }\n"
        "  .widget-error { padding: 16px; background: #3a1a1a; color: #f88;\n"
        "    border: 1px solid #f44; border-radius: 8px; font-family: monospace;\n"
        "    font-size: 13px; white-space: pre-wrap; word-break: break-word; }\n" + css_text + "\n</style>\n"
        "</head>\n"
        "<body>\n"
        '  <div id="widget-root"><div id="loading">Loading widget\u2026</div></div>\n'
        "\n"
        "  <!-- Widget bundle (rewritten export \u2192 window assignment) -->\n"
        '  <script type="module">\n' + js_patched + "\n  </script>\n"
        "\n"
        "  <!-- Render widget with embedded data -->\n"
        '  <script type="module">\n'
        "    try {\n"
        "    async function waitForWidget(ms = 5000) {\n"
        "      const t0 = Date.now();\n"
        "      while (!window.__qdk_widget && Date.now() - t0 < ms)\n"
        "        await new Promise(r => setTimeout(r, 50));\n"
        "      return window.__qdk_widget;\n"
        "    }\n"
        "\n"
        "    const widgetModule = await waitForWidget();\n"
        "    if (!widgetModule) {\n"
        '      document.getElementById("loading").textContent = "Failed to load widget JS";\n'
        '      throw new Error("widget bundle did not load");\n'
        "    }\n"
        "\n"
        "    function renderToolData(data) {\n"
        '      const widgetType = data.__widget_type || "' + component_name + '";\n'
        "      let stateKeys;\n"
        '      if (widgetType === "MoleculeViewer") {\n'
        "        stateKeys = {\n"
        '          comp: "MoleculeViewer",\n'
        "          molecule_data: data.molecule_data,\n"
        "          cube_data: data.cube_data ?? {},\n"
        "          isoval: data.isoval ?? 0.02,\n"
        "        };\n"
        '      } else if (widgetType === "Circuit") {\n'
        "        stateKeys = {\n"
        '          comp: "Circuit",\n'
        "          ...data,\n"
        "        };\n"
        "      } else {\n"
        "        stateKeys = {\n"
        "          comp: widgetType,\n"
        "          ...data,\n"
        "        };\n"
        "      }\n"
        "\n"
        "      const model = {\n"
        "        get(key) { return stateKeys[key]; },\n"
        "        set(key, val) { stateKeys[key] = val; },\n"
        "        save_changes() {},\n"
        "        on() {},\n"
        "        send() {},\n"
        "      };\n"
        "\n"
        '      const el = document.getElementById("widget-root");\n'
        '      el.innerHTML = "";\n'
        '      if (widgetType === "MoleculeViewer") {\n'
        '        el.style.minHeight = "400px";\n'
        "      }\n"
        "      widgetModule.default.render({ model, el });\n"
        "    }\n"
        "\n"
        + (
            # Embedded data: render immediately without host messages
            "    renderToolData(" + embedded_data_json + ");\n"
            if embedded_data_json is not None
            # Fallback: no embedded data
            else '    document.getElementById("loading")?.remove();\n'
        )
        + "    } catch(e) {\n"
        '      const el = document.getElementById("loading") || document.getElementById("widget-root");\n'
        "      if (el) { el.innerHTML = "
        "'<div class=\"widget-error\">Error: ' + e.message + '\\n\\n' + e.stack + '</div>'; }\n"
        "    }\n"
        "  </script>\n"
        "</body>\n"
        "</html>"
    )


# ---------------------------------------------------------------------------
# Widget data bridge: sync tool data → resource HTML
# ---------------------------------------------------------------------------


class _WidgetBridge:
    """Synchronise data between a tool call and its ``ui://`` resource.

    The tool stores its payload via :meth:`send`, which wakes the
    resource handler.  The resource handler calls :meth:`receive` to
    block until data is available and returns HTML with the data
    embedded.
    """

    def __init__(
        self, *, component_name: str, app_name: str, title: str, timeout: float = 15, min_height: int = 500
    ) -> None:
        self._component_name = component_name
        self._app_name = app_name
        self._title = title
        self._timeout = timeout
        self._min_height = min_height
        self._data: dict = {}
        self._ready = threading.Event()

    # Called by the tool
    def send(self, payload: dict) -> CallToolResult:
        """Store *payload*, wake the resource, and return a ``CallToolResult``."""
        self._data = dict(payload)
        self._ready.set()
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(payload))],
            structuredContent=payload,
        )

    # Called by the resource handler
    def receive_html(self) -> str:
        """Block until data arrives, then return self-contained HTML."""
        self._ready.wait(timeout=self._timeout)
        self._ready.clear()
        return _build_html(
            title=self._title,
            component_name=self._component_name,
            app_name=self._app_name,
            embedded_data_json=json.dumps(self._data) if self._data else None,
            min_height=self._min_height,
        )


# ---------------------------------------------------------------------------
# Public API: register resources and tools on the FastMCP app
# ---------------------------------------------------------------------------

_MINIMAL_TEST_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"/><title>MCP Test</title>
<style>
html, body { margin: 0; padding: 10px; font-family: monospace; font-size: 12px; }
.box { width: 200px; height: 200px; border-radius: 12px;
       display: flex; align-items: center; justify-content: center;
       color: white; font-size: 18px; font-weight: bold; }
#log { margin-top: 10px; background: #111; color: #0f0; padding: 8px;
       max-height: 400px; overflow: auto; white-space: pre-wrap; word-break: break-all; }
</style>
</head>
<body>
<div id="root"><div class="box" style="background:#0078d4">Waiting...</div></div>
<div id="log"></div>
<script>
const logEl = document.getElementById("log");
function log(msg) {
  logEl.textContent += msg + "\\n";
  logEl.scrollTop = logEl.scrollHeight;
}
log("iframe loaded, listening for messages...");

function _sendNotification(method, params) {
  window.parent.postMessage({ jsonrpc: "2.0", method, params }, "*");
}

function tryRender(data) {
  if (!data) return false;
  const color = data.color || '#00cc00';
  const label = data.label || 'Got data!';
  const el = document.getElementById("root");
  el.innerHTML = '<div class="box" style="background:' + color + '">' + label + '</div>';
  log("RENDERED: " + JSON.stringify(data));
  requestAnimationFrame(() => {
    const rect = el.getBoundingClientRect();
    _sendNotification("ui/notifications/size-changed", {
      width: Math.ceil(rect.width), height: Math.ceil(rect.height + logEl.offsetHeight + 30),
    });
  });
  return true;
}

window.addEventListener("message", (ev) => {
  const d = ev.data;
  if (!d) { log("msg: empty"); return; }
  if (d.jsonrpc !== "2.0") { log("msg (non-jsonrpc): " + JSON.stringify(d).slice(0, 200)); return; }

  log("JSONRPC method=" + d.method + " id=" + (d.id ?? "none"));

  if (d.method === "ui/initialize" && d.id != null) {
    log("  -> responding to ui/initialize");
    window.parent.postMessage({
      jsonrpc: "2.0", id: d.id,
      result: {
        protocolVersion: "2026-01-26",
        appCapabilities: {},
        appInfo: { name: "mcp-test-square", version: "1.0.0" },
      },
    }, "*");
    _sendNotification("ui/notifications/initialized", {});
    return;
  }

  // Log all params keys for any tool-related message
  if (d.params) {
    log("  params keys: " + Object.keys(d.params).join(", "));
    if (d.params.structuredContent) {
      log("  structuredContent: " + JSON.stringify(d.params.structuredContent).slice(0, 300));
    }
    if (d.params.content) {
      log("  content: " + JSON.stringify(d.params.content).slice(0, 300));
    }
  }

  // Try to extract data from multiple possible locations
  if (d.method === "ui/notifications/tool-result" || d.method === "ui/notifications/tool-input") {
    log("  -> tool result/input received!");
    // Try structuredContent first
    if (tryRender(d.params?.structuredContent)) return;
    // Try content text
    const textPart = d.params?.content?.find((c) => c.type === "text");
    if (textPart) {
      log("  text content: " + textPart.text.slice(0, 200));
      try {
        const parsed = JSON.parse(textPart.text);
        if (tryRender(parsed)) return;
      } catch(e) { log("  parse error: " + e); }
    }
    log("  -> could not extract render data");
  }
});

log("message listener registered");
</script>
</body>
</html>
"""


def register_visualization_tools(app) -> None:
    """Register interactive widget-based visualization tools on *app*.

    Tools are only registered when ``qsharp_widgets`` is installed.
    If not available, this function is a no-op.

    Parameters
    ----------
    app : FastMCP
        The FastMCP application instance from ``tools.py``.

    """
    if not _WIDGETS_AVAILABLE:
        return

    # ── Minimal test tool ─────────────────────────────────────────
    @app.resource(
        "ui://qdk-chem-mcp/test-square",
        name="test_square",
        description="Minimal test: renders a colored square",
        mime_type="text/html;profile=mcp-app",
    )
    def test_square_resource() -> str:
        return _MINIMAL_TEST_HTML

    @app.tool(
        meta={"ui": {"resourceUri": "ui://qdk-chem-mcp/test-square"}},
        structured_output=False,
    )
    def visualize_test_square(
        color: str = "#0078d4",
        label: str = "Hello MCP!",
    ) -> CallToolResult:
        """Render a simple colored square. Used to test MCP Apps UI rendering.

        Args:
            color: CSS color for the square (default: blue)
            label: Text to show inside the square

        """
        return CallToolResult(
            content=[TextContent(type="text", text=f"Test square: {label} ({color})")],
            structuredContent={"color": color, "label": label},
        )

    # ── Circuit viewer ────────────────────────────────────────────
    _circuit_bridge = _WidgetBridge(
        component_name="Circuit",
        app_name="qdk-circuit-viewer",
        title="Circuit Viewer",
        min_height=600,
    )

    @app.resource(
        "ui://qdk-chem-mcp/circuit-viewer",
        name="circuit_viewer",
        description="Interactive quantum-circuit diagram (qsharp-widgets Circuit component)",
        mime_type="text/html;profile=mcp-app",
    )
    def circuit_viewer_resource() -> str:
        return _circuit_bridge.receive_html()

    @app.tool(
        meta={
            "ui": {"resourceUri": "ui://qdk-chem-mcp/circuit-viewer"},
            "ui/resourceUri": "ui://qdk-chem-mcp/circuit-viewer",
        },
        structured_output=False,
    )
    @validate_project
    def visualize_circuit(
        project_name: str,
        circuit_filename: str,
    ) -> str | list:
        """Render an interactive quantum-circuit diagram for a saved qdk/chemistry Circuit object.

        Loads a serialised ``data.Circuit`` from the project directory and renders it
        using the ``qsharp_widgets.Circuit`` widget component via MCP Apps.

        Typical workflow context:

        1. Run ``run_scf`` to get an initial wavefunction
        2. Run ``run_state_preparation`` to generate a circuit from the wavefunction
        3. (THIS TOOL) Run ``visualize_circuit`` to inspect the quantum circuit

        Args:
            project_name (str): Name of the current qdk/chemistry project
            circuit_filename (str): Filename of the saved circuit (e.g. "h2.circuit.json")

        Returns:
            list[TextContent]: containing the circuit JSON for the interactive viewer
            str: containing an error message if the circuit could not be loaded

        """
        circuit_filename = circuit_filename.rsplit("/", maxsplit=1)[-1]

        try:
            circuit_obj = load_data_object(circuit_filename, data.Circuit)
        except (RuntimeError, ValueError) as e:
            return f"Failed to load circuit from {circuit_filename}: {e!s}"

        # Convert the qdk-chemistry Circuit to the widget-compatible format:
        #   1. Extract QASM via circuit_obj.get_qasm()
        #   2. Convert to a qsharp Circuit via qdk.openqasm.circuit()
        #   3. Serialise with .json() → widget-compatible JSON string
        from qdk.openqasm import circuit as openqasm_circuit  # noqa: PLC0415

        try:
            qasm_str = circuit_obj.get_qasm()
            qsharp_circuit = openqasm_circuit(qasm_str)
            circuit_json_str = qsharp_circuit.json()
        except Exception as e:  # noqa: BLE001
            return (
                f"Cannot build circuit visualisation from {circuit_filename}: {e!s}. "
                f"Make sure the file contains a valid qdk/chemistry Circuit object."
            )

        circuit_data = {
            "circuit_json": circuit_json_str,
        }

        return _circuit_bridge.send(circuit_data)

    # ── Orbital-entanglement chord diagram ────────────────────────
    _entanglement_bridge = _WidgetBridge(
        component_name="Entanglement",
        app_name="qdk-orbital-entanglement",
        title="Orbital Entanglement",
        min_height=700,
    )

    @app.resource(
        "ui://qdk-chem-mcp/orbital-entanglement",
        name="orbital_entanglement",
        description="Interactive orbital-entanglement chord diagram (qsharp-widgets Entanglement component)",
        mime_type="text/html;profile=mcp-app",
    )
    def orbital_entanglement_resource() -> str:
        return _entanglement_bridge.receive_html()

    @app.tool(
        meta={
            "ui": {"resourceUri": "ui://qdk-chem-mcp/orbital-entanglement"},
            "ui/resourceUri": "ui://qdk-chem-mcp/orbital-entanglement",
        },
        structured_output=False,
    )
    @validate_project
    def visualize_orbital_entanglement(
        project_name: str,
        wavefunction_filename: str,
        selected_indices: list[int] | None = None,
        group_selected: bool = False,
        mi_threshold: float | None = None,
    ) -> str | list:
        """Render an interactive orbital-entanglement chord diagram for a saved qdk/chemistry Wavefunction.

        The wavefunction must contain single-orbital entropies and mutual information, which requires
        a multi-configurational calculation with RDM and mutual-information calculation enabled
        (``calculate_one_rdm=True``, ``calculate_two_rdm=True``, ``calculate_mutual_information=True``).

        Typical workflow context:

        This is a visualisation tool that can be used after running a multi-configurational calculation:

        1. Run ``run_scf`` to get an initial wavefunction
        2. Run ``run_active_space_selector`` to define active orbitals
        3. Run ``run_multi_configuration_calculation`` with ``calculate_one_rdm=True``,
           ``calculate_two_rdm=True``, and ``calculate_mutual_information=True``
        4. (THIS TOOL) Run ``visualize_orbital_entanglement`` to inspect orbital correlations

        The chord diagram shows single-orbital entropies on the arcs and mutual information on the
        chords connecting orbital pairs. This is useful for identifying strongly correlated orbitals
        and refining active space selections.

        **Index convention:** ``selected_indices`` accepts **absolute orbital indices** — the same
        indices shown as labels on the diagram arcs.  For example, if the active space starts at
        orbital 6 and you want to highlight orbitals 8, 9, 10, pass ``selected_indices=[8, 9, 10]``.
        The conversion to diagram-relative positions is handled automatically.

        Args:
            project_name (str): Name of the current qdk/chemistry project
            wavefunction_filename (str): Filename of the saved wavefunction (e.g. "h2.wavefunction.json")
            selected_indices (List[int], optional): **Absolute** orbital indices to highlight in the
                diagram (matching the arc labels).  The tool converts these to diagram-relative
                positions automatically.
            group_selected (bool): When True, reorder arcs so highlighted orbitals sit adjacent. Default: False
            mi_threshold (float, optional): Minimum mutual-information value to draw a chord

        Returns:
            list[TextContent]: containing the entanglement data JSON for the interactive viewer
            str: containing an error message if there was a problem in the workflow

        """
        wavefunction_filename = wavefunction_filename.rsplit("/", maxsplit=1)[-1]

        try:
            wavefunction = load_data_object(wavefunction_filename, data.Wavefunction)
        except (RuntimeError, ValueError) as e:
            return f"Failed to load wavefunction from {wavefunction_filename}: {e!s}"

        # ── Convert absolute orbital indices to diagram-relative positions ──
        # The widget labels arcs with absolute orbital indices (e.g., 6..11)
        # but its selected_indices parameter expects 0-based positions into
        # the diagram (0 = first arc, 1 = second arc, ...).
        #
        # We build the label list first, then map the caller's absolute
        # indices to their positions in that list.
        diagram_selected = None
        if selected_indices is not None:
            try:
                import numpy as _np  # noqa: PLC0415

                n_entropies = len(_np.asarray(wavefunction.get_single_orbital_entropies()))
                # Build the same label list the widget would generate
                try:
                    orbitals = wavefunction.get_orbitals()
                    if orbitals.has_active_space():
                        active_indices = list(orbitals.get_active_space_indices()[0])
                    else:
                        active_indices = list(range(n_entropies))
                except (AttributeError, TypeError, IndexError):
                    active_indices = list(range(n_entropies))

                # Map absolute indices → diagram positions
                abs_to_pos = {abs_idx: pos for pos, abs_idx in enumerate(active_indices)}
                diagram_selected = []
                bad_indices = []
                for idx in selected_indices:
                    if idx in abs_to_pos:
                        diagram_selected.append(abs_to_pos[idx])
                    else:
                        bad_indices.append(idx)

                if bad_indices:
                    return (
                        f"selected_indices {bad_indices} are not valid absolute orbital indices "
                        f"for this wavefunction.  Valid absolute indices are: {active_indices}"
                    )
            except (RuntimeError, ValueError, AttributeError):
                # Fall back to treating them as-is (diagram-relative)
                diagram_selected = selected_indices

        # Use the Entanglement widget to extract all data from the
        # wavefunction.  The widget is a Python-side convenience that maps
        # wavefunction data onto ChordDiagram traitlets — it handles
        # entropy/MI extraction, orbital labels, and default options.
        from qsharp_widgets import Entanglement  # noqa: PLC0415

        opts: dict = {}
        if group_selected:
            opts["group_selected"] = group_selected
        if mi_threshold is not None:
            opts["mi_threshold"] = mi_threshold

        try:
            widget = Entanglement(
                wavefunction=wavefunction,
                selected_indices=diagram_selected,
                **opts,
            )
        except (RuntimeError, ValueError, AttributeError) as e:
            return (
                f"Cannot build orbital-entanglement diagram from {wavefunction_filename}: {e!s}. "
                f"Make sure the wavefunction was produced by a multi-configurational calculation "
                f"with calculate_one_rdm=True, calculate_two_rdm=True, and "
                f"calculate_mutual_information=True."
            )

        # Read the traitlet values the widget computed
        s1_entropies = list(widget.s1_entropies)
        mutual_info = [list(row) for row in widget.mutual_information]
        labels = list(widget.labels)
        options = dict(widget.options)

        entanglement_data = {
            "s1_entropies": s1_entropies,
            "mutual_information": mutual_info,
            "labels": labels,
            "selected_indices": list(widget.selected_indices) if widget.selected_indices else None,
            "options": options,
        }

        return _entanglement_bridge.send(entanglement_data)

    # ── Molecule viewer ───────────────────────────────────────────
    molecule_viewer_uri = "ui://qdk-chem-mcp/molecule-viewer"
    _molecule_bridge = _WidgetBridge(
        component_name="MoleculeViewer",
        app_name="qdk-molecule-viewer",
        title="Molecule Viewer",
        timeout=30,
        min_height=550,
    )

    @app.resource(
        molecule_viewer_uri,
        name="molecule_viewer",
        description="Interactive 3D molecule viewer (qsharp-widgets MoleculeViewer component)",
        mime_type="text/html;profile=mcp-app",
    )
    def molecule_viewer_resource() -> str:
        return _molecule_bridge.receive_html()

    @app.tool(
        meta={
            "ui": {"resourceUri": molecule_viewer_uri},
            "ui/resourceUri": molecule_viewer_uri,
        },
        structured_output=False,
    )
    @validate_project
    def visualize_molecule(
        project_name: str,
        structure_filename: str,
    ) -> str | list:
        """Show an interactive 3D view of a molecular structure.

        Loads a saved Structure from the project directory and renders it
        in a 3D viewer where you can rotate, zoom, and switch between
        Sphere, Stick, and Line visualisation styles.

        Args:
            project_name (str): Name of the current qdk/chemistry project
            structure_filename (str): Filename of the saved structure (e.g. "h2.structure.json")

        Returns:
            list[TextContent]: containing the molecule data JSON for the interactive viewer
            str: containing an error message if the structure could not be loaded

        """
        structure_filename = structure_filename.rsplit("/", maxsplit=1)[-1]

        try:
            structure = load_data_object(structure_filename, data.Structure)
        except (RuntimeError, ValueError) as e:
            return f"Failed to load structure from {structure_filename}: {e!s}"

        xyz_str = structure.to_xyz()

        payload = {
            "__widget_type": "MoleculeViewer",
            "molecule_data": xyz_str,
            "cube_data": {},
            "isoval": 0.02,
        }

        return _molecule_bridge.send(payload)

    # ── Orbital viewer (molecule + orbital isosurfaces) ───────────
    @app.tool(
        meta={
            "ui": {"resourceUri": molecule_viewer_uri},
            "ui/resourceUri": molecule_viewer_uri,
        },
        structured_output=False,
    )
    @validate_project
    def visualize_orbitals(
        project_name: str,
        wavefunction_filename: str,
        orbital_indices: list[int] | None = None,
        isoval: float = 0.02,
        grid_size: int = 40,
    ) -> str | list:
        """Show a 3D molecule viewer with orbital isosurfaces.

        Loads a Wavefunction, extracts the orbitals and molecular structure,
        generates volumetric cube data for each orbital, and renders
        them in an interactive 3D viewer with positive/negative isosurface
        lobes.  Use the isovalue slider and style controls to explore.

        Typical workflow context:

        1. Run ``run_scf`` to get an initial wavefunction
        2. (optional) Run ``run_active_space_selector`` / ``run_orbital_localization``
        3. (THIS TOOL) Run ``visualize_orbitals`` to inspect molecular orbitals

        Args:
            project_name (str): Name of the current qdk/chemistry project
            wavefunction_filename (str): Filename of the saved wavefunction (e.g. "h2.wavefunction.json")
            orbital_indices (List[int], optional): Orbital indices to render. If None, all orbitals are rendered.
            isoval (float): Isovalue for the orbital surfaces. Default: 0.02
            grid_size (int): Number of grid points per dimension for cube generation. Default: 40

        Returns:
            list[TextContent]: containing the molecule + cube data JSON for the interactive viewer
            str: containing an error message if there was a problem

        """
        wavefunction_filename = wavefunction_filename.rsplit("/", maxsplit=1)[-1]

        try:
            wavefunction = load_data_object(wavefunction_filename, data.Wavefunction)
        except (RuntimeError, ValueError) as e:
            return f"Failed to load wavefunction from {wavefunction_filename}: {e!s}"

        # Extract orbitals and structure from the wavefunction
        try:
            orbitals = wavefunction.get_orbitals()
            structure = orbitals.get_basis_set().get_structure()
        except (RuntimeError, AttributeError) as e:
            return f"Cannot extract structure/orbitals from {wavefunction_filename}: {e!s}"

        # Generate XYZ string for the molecule
        xyz_str = structure.to_xyz()

        # Generate cube data for the requested orbitals
        from qdk_chemistry.utils.cubegen import generate_cubefiles_from_orbitals  # noqa: PLC0415

        try:
            cube_data = generate_cubefiles_from_orbitals(
                orbitals,
                output_folder=None,  # return dict[label, cube_content]
                indices=orbital_indices,
                grid_size=(grid_size, grid_size, grid_size),
            )
        except (RuntimeError, ValueError) as e:
            return f"Failed to generate cube data: {e!s}"

        payload = {
            "__widget_type": "MoleculeViewer",
            "molecule_data": xyz_str,
            "cube_data": cube_data,
            "isoval": isoval,
        }

        return _molecule_bridge.send(payload)

    # ── Scatter plot (inline SVG) ─────────────────────────────────

    def _build_plotly_html(payload: dict) -> str:
        """Build a self-contained HTML page with an inline SVG scatter plot.

        Uses pure SVG + vanilla JS (no CDN) so it works in VS Code
        webview sandboxed iframes.
        """
        import math  # noqa: PLC0415

        title = payload.get("title", "Scatter Plot")
        x_label = payload.get("x_label", "X")
        y_label = payload.get("y_label", "Y")
        log_x = payload.get("log_x", False)
        log_y = payload.get("log_y", False)
        series_list = payload.get("series", [])

        # Collect all x/y values to compute axis ranges
        all_x: list[float] = []
        all_y: list[float] = []
        for s in series_list:
            all_x.extend(s.get("x", []))
            all_y.extend(s.get("y", []))

        if not all_x or not all_y:
            return "<html><body><p>No data to plot.</p></body></html>"

        if log_x:
            all_x = [math.log10(v) if v > 0 else 0 for v in all_x]
        if log_y:
            all_y = [math.log10(v) if v > 0 else 0 for v in all_y]

        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_pad = (x_max - x_min) * 0.08 or 1
        y_pad = (y_max - y_min) * 0.08 or 1
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad

        # SVG layout
        w, h = 700, 450
        ml, mr, mt, mb = 80, 30, 50, 60  # margins
        pw = w - ml - mr
        ph = h - mt - mb

        def tx(v: float) -> float:
            return ml + (v - x_min) / (x_max - x_min) * pw

        def ty(v: float) -> float:
            return mt + ph - (v - y_min) / (y_max - y_min) * ph

        colors = ["#89b4fa", "#f38ba8", "#a6e3a1", "#fab387", "#cba6f7", "#94e2d5", "#f9e2af", "#74c7ec"]

        def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
            rng = hi - lo
            if rng <= 0:
                return [lo]
            raw = rng / n
            mag = 10 ** math.floor(math.log10(raw))
            for step in (1, 2, 5, 10):
                s = step * mag
                if rng / s <= n + 1:
                    break
            start = math.ceil(lo / s) * s
            ticks = []
            v = start
            while v <= hi + s * 0.01:
                ticks.append(round(v, 10))
                v += s
            return ticks

        x_ticks = _nice_ticks(x_min, x_max)
        y_ticks = _nice_ticks(y_min, y_max)

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'preserveAspectRatio="xMidYMid meet" '
            f'style="font-family:system-ui,sans-serif">\n',
            f'<rect width="{w}" height="{h}" fill="#1e1e2e"/>\n',
            # plot area background
            f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="#181825" stroke="#45475a" stroke-width="1"/>\n',
        ]

        # Grid lines + tick labels
        for xv in x_ticks:
            px = tx(xv)
            if ml <= px <= ml + pw:
                label = f"{10**xv:.0f}" if log_x else f"{xv:g}"
                svg_parts.append(
                    f'<line x1="{px}" y1="{mt}" x2="{px}" y2="{mt + ph}" stroke="#313244" stroke-width="0.5"/>\n'
                )
                svg_parts.append(
                    f'<text x="{px}" y="{mt + ph + 16}" text-anchor="middle" '
                    f'fill="#a6adc8" font-size="11">{label}</text>\n'
                )

        for yv in y_ticks:
            py = ty(yv)
            if mt <= py <= mt + ph:
                label = f"{10**yv:.1e}" if log_y else f"{yv:g}"
                svg_parts.append(
                    f'<line x1="{ml}" y1="{py}" x2="{ml + pw}" y2="{py}" stroke="#313244" stroke-width="0.5"/>\n'
                )
                svg_parts.append(
                    f'<text x="{ml - 8}" y="{py + 4}" text-anchor="end" fill="#a6adc8" font-size="11">{label}</text>\n'
                )

        # Title
        svg_parts.append(
            f'<text x="{w / 2}" y="28" text-anchor="middle" fill="#cdd6f4" '
            f'font-size="15" font-weight="600">{title}</text>\n'
        )
        # Axis labels
        svg_parts.append(
            f'<text x="{ml + pw / 2}" y="{h - 8}" text-anchor="middle" fill="#a6adc8" font-size="12">{x_label}</text>\n'
        )
        svg_parts.append(
            f'<text x="16" y="{mt + ph / 2}" text-anchor="middle" '
            f'fill="#a6adc8" font-size="12" '
            f'transform="rotate(-90,16,{mt + ph / 2})">{y_label}</text>\n'
        )

        # Plot data series
        for si, s in enumerate(series_list):
            color = colors[si % len(colors)]
            xs = s.get("x", [])
            ys = s.get("y", [])
            texts = s.get("text", [])
            mode = s.get("mode", "markers")
            ms = s.get("marker_size", 8)

            pts = []
            for i, (xv, yv) in enumerate(zip(xs, ys, strict=False)):
                px_val = math.log10(xv) if log_x and xv > 0 else xv
                py_val = math.log10(yv) if log_y and yv > 0 else yv
                pts.append((tx(px_val), ty(py_val), xv, yv, texts[i] if i < len(texts) else ""))

            # Lines
            if "lines" in mode and len(pts) > 1:
                path_d = " ".join(f"{'M' if j == 0 else 'L'}{p[0]:.1f},{p[1]:.1f}" for j, p in enumerate(pts))
                svg_parts.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2" opacity="0.7"/>\n')

            # Markers with hover
            if "markers" in mode:
                for _pi, (px, py, raw_x, raw_y, text) in enumerate(pts):
                    hover = text or f"{raw_x:g}, {raw_y:g}"
                    svg_parts.append(
                        f'<circle class="pt" cx="{px:.1f}" cy="{py:.1f}" r="{ms / 2}" '
                        f'fill="{color}" opacity="0.9" '
                        f'data-x="{raw_x}" data-y="{raw_y}" data-label="{hover}" '
                        f'data-series="{si}"/>\n'
                    )

        # Legend
        if len(series_list) > 1 or (series_list and series_list[0].get("name")):
            ly = mt + 12
            for si, s in enumerate(series_list):
                color = colors[si % len(colors)]
                name = s.get("name", f"Series {si + 1}")
                svg_parts.append(f'<rect x="{ml + pw - 140}" y="{ly}" width="10" height="10" fill="{color}" rx="2"/>\n')
                svg_parts.append(
                    f'<text x="{ml + pw - 125}" y="{ly + 9}" fill="#cdd6f4" font-size="11">{name}</text>\n'
                )
                ly += 18

        # Crosshair lines (hidden by default)
        svg_parts.append(
            f'<line id="xhair-h" x1="{ml}" y1="0" x2="{ml + pw}" y2="0" '
            f'stroke="#585b70" stroke-width="0.5" stroke-dasharray="4,3" visibility="hidden"/>\n'
        )
        svg_parts.append(
            f'<line id="xhair-v" x1="0" y1="{mt}" x2="0" y2="{mt + ph}" '
            f'stroke="#585b70" stroke-width="0.5" stroke-dasharray="4,3" visibility="hidden"/>\n'
        )

        svg_parts.append("</svg>")
        svg_content = "".join(svg_parts)

        # Interactive JS for hover tooltip, crosshairs, point highlight
        js = (
            "<script>\n"
            "const svg=document.querySelector('svg');\n"
            "const tip=document.getElementById('tooltip');\n"
            "const xH=document.getElementById('xhair-h');\n"
            "const xV=document.getElementById('xhair-v');\n"
            "const pts=document.querySelectorAll('.pt');\n"
            "let activePt=null;\n"
            "pts.forEach(c=>{\n"
            "  c.style.cursor='pointer';\n"
            "  c.addEventListener('mouseenter',function(ev){\n"
            "    if(activePt) activePt.setAttribute('r',activePt.dataset.origR);\n"
            "    activePt=c;\n"
            "    c.dataset.origR=c.getAttribute('r');\n"
            "    c.setAttribute('r',parseFloat(c.getAttribute('r'))*1.8);\n"
            "    c.setAttribute('opacity','1');\n"
            "    const lbl=c.dataset.label;\n"
            "    const x=c.dataset.x, y=c.dataset.y;\n"
            "    tip.innerHTML='<b>'+lbl+'</b><br/>'+'" + x_label + ": '+x+'<br/>'+'" + y_label + ": '+y;\n"
            "    tip.style.display='block';\n"
            "    const cx=parseFloat(c.getAttribute('cx'));\n"
            "    const cy=parseFloat(c.getAttribute('cy'));\n"
            "    xH.setAttribute('y1',cy); xH.setAttribute('y2',cy); xH.setAttribute('visibility','visible');\n"
            "    xV.setAttribute('x1',cx); xV.setAttribute('x2',cx); xV.setAttribute('visibility','visible');\n"
            "  });\n"
            "  c.addEventListener('mousemove',function(ev){\n"
            "    tip.style.left=(ev.clientX+16)+'px';\n"
            "    tip.style.top=(ev.clientY-12)+'px';\n"
            "  });\n"
            "  c.addEventListener('mouseleave',function(){\n"
            "    c.setAttribute('r',c.dataset.origR);\n"
            "    c.setAttribute('opacity','0.9');\n"
            "    tip.style.display='none';\n"
            "    xH.setAttribute('visibility','hidden');\n"
            "    xV.setAttribute('visibility','hidden');\n"
            "    activePt=null;\n"
            "  });\n"
            "});\n"
            "</script>\n"
        )

        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            "<head>\n"
            '<meta charset="utf-8"/>\n'
            '<meta name="viewport" content="width=device-width,initial-scale=1"/>\n'
            f"<title>{title}</title>\n"
            "<style>\n"
            "  html, body { margin:0; padding:0; background:#1e1e2e;\n"
            "    width:100%; height:100%; overflow:hidden; }\n"
            "  svg { display:block; width:100%; height:100%; }\n"
            "  #tooltip { position:fixed; pointer-events:none; display:none;\n"
            "    background:#313244; color:#cdd6f4; padding:8px 12px;\n"
            "    border-radius:6px; font-size:12px; font-family:system-ui,sans-serif;\n"
            "    box-shadow:0 2px 8px rgba(0,0,0,0.4); z-index:100;\n"
            "    max-width:260px; line-height:1.5; }\n"
            "</style>\n"
            "</head>\n"
            "<body>\n"
            f"{svg_content}\n"
            '<div id="tooltip"></div>\n'
            f"{js}"
            "</body>\n"
            "</html>"
        )

    _scatter_bridge_data: dict = {}
    _scatter_bridge_event = threading.Event()

    @app.resource(
        "ui://qdk-chem-mcp/scatter-plot",
        name="scatter_plot",
        description="Interactive Plotly scatter plot with optional log axes and multiple series",
        mime_type="text/html;profile=mcp-app",
    )
    def scatter_plot_resource() -> str:
        _scatter_bridge_event.wait(timeout=15)
        _scatter_bridge_event.clear()
        return _build_plotly_html(_scatter_bridge_data)

    @app.tool(
        meta={
            "ui": {"resourceUri": "ui://qdk-chem-mcp/scatter-plot"},
            "ui/resourceUri": "ui://qdk-chem-mcp/scatter-plot",
        },
        structured_output=False,
    )
    def visualize_scatter_plot(
        series: list[dict],
        title: str = "Scatter Plot",
        x_label: str = "X",
        y_label: str = "Y",
        log_x: bool = False,
        log_y: bool = False,
    ) -> CallToolResult:
        """Render an interactive scatter plot with Plotly.

        Supports multiple data series, optional logarithmic axes, and
        configurable marker styles. Use this to visualize any tabular
        x/y data — Pareto frontiers, convergence curves, energy surfaces, etc.

        Each entry in ``series`` is a dict with:
          - ``x`` (list[float]): X-axis values
          - ``y`` (list[float]): Y-axis values
          - ``name`` (str, optional): Legend label for this series
          - ``mode`` (str, optional): Plotly trace mode — ``"markers"``,
            ``"lines"``, ``"lines+markers"`` (default: ``"markers"``)
          - ``marker_size`` (int, optional): Marker size in px
          - ``text`` (list[str], optional): Hover text per point

        Args:
            series: List of data series dicts (see above).
            title: Chart title.
            x_label: X-axis label.
            y_label: Y-axis label.
            log_x: If True, use logarithmic X axis.
            log_y: If True, use logarithmic Y axis.

        Returns:
            Interactive Plotly scatter plot rendered in the MCP Apps viewer.

        """
        nonlocal _scatter_bridge_data
        payload = {
            "title": title,
            "x_label": x_label,
            "y_label": y_label,
            "log_x": log_x,
            "log_y": log_y,
            "series": series,
        }
        _scatter_bridge_data = payload
        _scatter_bridge_event.set()
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(payload))],
            structuredContent=payload,
        )
