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

# ruff: noqa: ARG001, E501
# MCP tool functions accept ``project_name`` consumed by the
# ``@validate_project`` decorator.

from __future__ import annotations

import json
import pathlib
import re

from mcp.types import CallToolResult, TextContent

from qdk_chemistry import data

from .io import load_data_object
from .validation import strip_filename_path, validate_project

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


def _json_script(identifier: str, value: object) -> str:
    """Return *value* as an inert, HTML-safe JSON script element."""
    serialized = (
        json.dumps(value, ensure_ascii=True).replace("<", r"\u003c").replace(">", r"\u003e").replace("&", r"\u0026")
    )
    return f'<script id="{identifier}" type="application/json">{serialized}</script>'


# ---------------------------------------------------------------------------
# HTML builder (uses MCP Apps SDK + patched widget bundle)
# ---------------------------------------------------------------------------


def _build_html(
    *,
    title: str,
    component_name: str,
    app_name: str,
    min_height: int = 500,
) -> str:
    """Build a self-contained HTML page for a qsharp-widgets component."""
    static = _widgets_static_dir()
    js_text = (static / "index.js").read_text(encoding="utf-8")
    css_text = (static / "index.css").read_text(encoding="utf-8")

    m = re.search(r"export\{(\w+) as default,(\w+) as mdRenderer\}", js_text)
    if m:
        default_var, md_var = m.group(1), m.group(2)
        js_patched = js_text.replace(
            m.group(0),
            f"window.__qdk_widget={{default:{default_var},mdRenderer:{md_var}}}",
        )
    else:
        js_patched = js_text

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8" />\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        "<title></title>\n"
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
        '  <script type="module">\n' + js_patched + "\n  </script>\n"
        "\n"
        '  <script type="module">\n'
        "    try {\n"
        '    const config = JSON.parse(document.getElementById("widget-config").textContent);\n'
        "    document.title = config.title;\n"
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
        "      const widgetType = data.__widget_type || config.componentName;\n"
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
        "    let nextRequestId = 1;\n"
        "    const pending = new Map();\n"
        "    window.addEventListener('message', event => {\n"
        "      if (event.source !== window.parent) return;\n"
        "      const message = event.data;\n"
        "      if (message?.jsonrpc !== '2.0') return;\n"
        "      const responseId = message.id == null ? null : String(message.id);\n"
        "      if (responseId != null && pending.has(responseId)) {\n"
        "        const {resolve, reject} = pending.get(responseId);\n"
        "        pending.delete(responseId);\n"
        "        message.error ? reject(new Error(message.error.message)) : resolve(message.result);\n"
        "      } else if (message.method === 'ui/notifications/tool-result') {\n"
        "        const result = message.params || {};\n"
        "        let data = result.structuredContent;\n"
        "        if (!data) {\n"
        "          const text = result.content?.find(item => item.type === 'text')?.text;\n"
        "          if (text) try { data = JSON.parse(text); } catch {}\n"
        "        }\n"
        "        if (data) renderToolData(data);\n"
        "      }\n"
        "    });\n"
        "    function request(method, params) {\n"
        "      const id = nextRequestId++;\n"
        "      window.parent.postMessage({jsonrpc:'2.0', id, method, params}, '*');\n"
        "      return new Promise((resolve, reject) => pending.set(String(id), {resolve, reject}));\n"
        "    }\n"
        "    const initialized = await request('ui/initialize', {\n"
        "      appCapabilities: {},\n"
        "      clientInfo: {name: config.appName, version: '1.0.0'},\n"
        "      protocolVersion: '2026-01-26',\n"
        "    });\n"
        "    window.parent.postMessage({jsonrpc:'2.0', method:'ui/notifications/initialized'}, '*');\n"
        + "    } catch(e) {\n"
        '      const el = document.getElementById("loading") || document.getElementById("widget-root");\n'
        "      if (el) {\n"
        '        const error = document.createElement("div");\n'
        '        error.className = "widget-error";\n'
        '        error.textContent = "Error: " + e.message + "\\n\\n" + e.stack;\n'
        "        el.replaceChildren(error);\n"
        "      }\n"
        "    }\n"
        "  </script>\n"
        "</body>\n"
        "</html>".replace(
            "</body>",
            _json_script("widget-config", {"title": title, "componentName": component_name, "appName": app_name})
            + "\n</body>",
        )
    )


def _build_scatter_html() -> str:
    """Build a static MCP App that renders scatter data from tool results."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Scatter Plot</title>
<style>
:root { color-scheme: light dark; }
html,body { margin:0; width:100%; height:100%; overflow:hidden; background:var(--color-background-primary,#181818); }
svg { display:block; width:100%; height:100%; }
text { fill:var(--color-text-primary,#ddd); font-family:var(--font-sans,sans-serif); }
</style>
</head>
<body>
<svg id="chart" viewBox="0 0 700 450" preserveAspectRatio="xMidYMid meet"></svg>
<script>
const ns='http://www.w3.org/2000/svg',svg=document.getElementById('chart');
function add(name,attrs={},text){const el=document.createElementNS(ns,name);for(const [key,value] of Object.entries(attrs))el.setAttribute(key,String(value));if(text!==undefined)el.textContent=text;svg.append(el);return el;}
function render(data){
    svg.replaceChildren(); document.title=data.title||'Scatter Plot';
    const series=Array.isArray(data.series)?data.series:[], points=[];
    for(const [group,item] of series.entries())for(let index=0;index<Math.min(item.x?.length||0,item.y?.length||0);index++){
        const rawX=Number(item.x[index]),rawY=Number(item.y[index]);
        if(!Number.isFinite(rawX)||!Number.isFinite(rawY)||(data.log_x&&rawX<=0)||(data.log_y&&rawY<=0))continue;
        points.push({x:data.log_x?Math.log10(rawX):rawX,y:data.log_y?Math.log10(rawY):rawY,rawX,rawY,group,label:item.text?.[index]||item.name||''});
    }
    add('rect',{width:700,height:450,fill:'var(--color-background-primary,#181818)'});
    if(!points.length){add('text',{x:350,y:225,'text-anchor':'middle'},'No data to plot.');return;}
    let minX=Math.min(...points.map(point=>point.x)),maxX=Math.max(...points.map(point=>point.x));
    let minY=Math.min(...points.map(point=>point.y)),maxY=Math.max(...points.map(point=>point.y));
    const padX=(maxX-minX)*.08||1,padY=(maxY-minY)*.08||1;minX-=padX;maxX+=padX;minY-=padY;maxY+=padY;
    const left=75,top=45,width=595,height=340,tx=value=>left+(value-minX)/(maxX-minX)*width,ty=value=>top+height-(value-minY)/(maxY-minY)*height;
    add('rect',{x:left,y:top,width,height,fill:'var(--color-background-secondary,#222)',stroke:'var(--color-border-primary,#666)'});
    function ticks(low,high,count=5){const range=high-low,raw=range/count,magnitude=10**Math.floor(Math.log10(raw));let step=magnitude;for(const candidate of [1,2,5,10])if(range/(candidate*magnitude)<=count+1){step=candidate*magnitude;break;}const values=[];for(let value=Math.ceil(low/step)*step;value<=high+step*.01;value+=step)values.push(value);return values;}
    const format=(value,log)=>log?(10**value).toExponential(1):Number(value.toPrecision(4)).toString();
    for(const value of ticks(minX,maxX)){const x=tx(value);add('line',{x1:x,y1:top,x2:x,y2:top+height,stroke:'var(--color-border-secondary,#444)','stroke-width':.5});add('text',{x,y:top+height+17,'text-anchor':'middle','font-size':11},format(value,data.log_x));}
    for(const value of ticks(minY,maxY)){const y=ty(value);add('line',{x1:left,y1:y,x2:left+width,y2:y,stroke:'var(--color-border-secondary,#444)','stroke-width':.5});add('text',{x:left-8,y:y+4,'text-anchor':'end','font-size':11},format(value,data.log_y));}
    add('text',{x:350,y:28,'text-anchor':'middle','font-size':16},data.title||'Scatter Plot');
    add('text',{x:372,y:430,'text-anchor':'middle','font-size':12},data.x_label||'X');
    add('text',{x:18,y:215,'text-anchor':'middle','font-size':12,transform:'rotate(-90 18 215)'},data.y_label||'Y');
    const colors=['#4f8cff','#e05263','#38a169','#d69e2e','#805ad5','#319795'];
    for(const [group,item] of series.entries()){const groupPoints=points.filter(point=>point.group===group),color=colors[group%colors.length];if(item.mode?.includes('lines')&&groupPoints.length>1)add('path',{d:groupPoints.map((point,index)=>`${index?'L':'M'}${tx(point.x)},${ty(point.y)}`).join(' '),fill:'none',stroke:color,'stroke-width':2});if(item.mode?.includes('markers')??true)for(const point of groupPoints){const dot=add('circle',{cx:tx(point.x),cy:ty(point.y),r:(item.marker_size??8)/2,fill:color});const title=document.createElementNS(ns,'title');title.textContent=`${point.label}${point.label?' - ':''}${point.rawX}, ${point.rawY}`;dot.append(title);}}
    if(series.length>1||series[0]?.name){let legendY=top+12;for(const [index,item] of series.entries()){const color=colors[index%colors.length];add('rect',{x:left+width-140,y:legendY,width:10,height:10,fill:color,rx:2});add('text',{x:left+width-125,y:legendY+9,'font-size':11},item.name||`Series ${index+1}`);legendY+=18;}}
}
let requestId=1;
window.addEventListener('message',event=>{if(event.source!==window.parent)return;const message=event.data;if(message?.jsonrpc!=='2.0')return;if(message.id!=null&&String(message.id)===String(requestId)){if(message.error){svg.replaceChildren();add('text',{x:350,y:225,'text-anchor':'middle'},`Initialize failed: ${message.error.message||'unknown error'}`);return;}window.parent.postMessage({jsonrpc:'2.0',method:'ui/notifications/initialized'},'*');}else if(message.method==='ui/notifications/tool-result'){const result=message.params||{};let data=result.structuredContent;if(!data){const text=result.content?.find(item=>item.type==='text')?.text;if(text)try{data=JSON.parse(text);}catch{}}if(data)render(data);}});
window.parent.postMessage({jsonrpc:'2.0',id:requestId,method:'ui/initialize',params:{appCapabilities:{},clientInfo:{name:'qdk-scatter-plot',version:'1.0.0'},protocolVersion:'2026-01-26'}},'*');
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Widget tool results
# ---------------------------------------------------------------------------


def _tool_result(payload: dict) -> CallToolResult:
    """Return a structured visualization result with a text fallback."""
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))],
        structuredContent=payload,
    )


# ---------------------------------------------------------------------------
# Public API: register resources and tools on the MCPServer app
# ---------------------------------------------------------------------------


def register_visualization_tools(apps) -> None:
    """Register interactive widget-based visualization tools on an MCP server.

    Tools are registered only when ``qsharp_widgets`` is installed. Otherwise,
    this function is a no-op.

    Args:
        apps: MCP Apps extension that receives app resources and tools.

    """
    if not _WIDGETS_AVAILABLE:
        return

    # ── Circuit viewer ────────────────────────────────────────────
    circuit_uri = "ui://qdk-chem-mcp/circuit-viewer"
    apps.add_html_resource(
        circuit_uri,
        _build_html(
            component_name="Circuit",
            app_name="qdk-circuit-viewer",
            title="Circuit Viewer",
            min_height=600,
        ),
        name="circuit_viewer",
        description="Interactive quantum-circuit diagram (qsharp-widgets Circuit component)",
    )

    @apps.tool(
        resource_uri=circuit_uri,
        description="Render a saved Circuit in VS Code MCP Apps.",
        structured_output=False,
    )
    @validate_project
    def visualize_circuit(
        project_name: str,
        circuit_filename: str,
    ) -> str | list:
        """Render a saved Circuit in VS Code MCP Apps."""
        circuit_filename = strip_filename_path(circuit_filename)

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

        return _tool_result(circuit_data)

    # ── Orbital-entanglement chord diagram ────────────────────────
    entanglement_uri = "ui://qdk-chem-mcp/orbital-entanglement"
    apps.add_html_resource(
        entanglement_uri,
        _build_html(
            component_name="Entanglement",
            app_name="qdk-orbital-entanglement",
            title="Orbital Entanglement",
            min_height=700,
        ),
        name="orbital_entanglement",
        description="Interactive orbital-entanglement chord diagram (qsharp-widgets Entanglement component)",
    )

    @apps.tool(
        resource_uri=entanglement_uri,
        description="Render RDM/MI orbital entanglement with absolute indices in VS Code MCP Apps.",
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
        """Render RDM/MI orbital entanglement with absolute indices in VS Code MCP Apps."""
        wavefunction_filename = strip_filename_path(wavefunction_filename)

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

        return _tool_result(entanglement_data)

    # ── Molecule viewer ───────────────────────────────────────────
    molecule_viewer_uri = "ui://qdk-chem-mcp/molecule-viewer"
    apps.add_html_resource(
        molecule_viewer_uri,
        _build_html(
            component_name="MoleculeViewer",
            app_name="qdk-molecule-viewer",
            title="Molecule Viewer",
            min_height=550,
        ),
        name="molecule_viewer",
        description="Interactive 3D molecule viewer (qsharp-widgets MoleculeViewer component)",
    )

    @apps.tool(
        resource_uri=molecule_viewer_uri,
        description="Render a saved Structure in VS Code MCP Apps.",
        structured_output=False,
    )
    @validate_project
    def visualize_molecule(
        project_name: str,
        structure_filename: str,
    ) -> str | list:
        """Render a saved Structure in VS Code MCP Apps."""
        structure_filename = strip_filename_path(structure_filename)

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

        return _tool_result(payload)

    # ── Orbital viewer (molecule + orbital isosurfaces) ───────────
    @apps.tool(
        resource_uri=molecule_viewer_uri,
        description="Render saved Wavefunction orbitals in VS Code MCP Apps.",
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
        """Render saved Wavefunction orbitals in VS Code MCP Apps."""
        wavefunction_filename = strip_filename_path(wavefunction_filename)

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

        return _tool_result(payload)

    # ── Scatter plot (inline SVG) ─────────────────────────────────

    scatter_uri = "ui://qdk-chem-mcp/scatter-plot"
    apps.add_html_resource(
        scatter_uri,
        _build_scatter_html(),
        name="scatter_plot",
        description="Interactive SVG scatter plot with optional log axes and multiple series",
    )

    @apps.tool(
        resource_uri=scatter_uri,
        description="Render numeric series as an SVG scatter plot in VS Code MCP Apps.",
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
        """Render numeric series as an SVG scatter plot in VS Code MCP Apps."""
        payload = {
            "title": title,
            "x_label": x_label,
            "y_label": y_label,
            "log_x": log_x,
            "log_y": log_y,
            "series": series,
        }
        return _tool_result(payload)
