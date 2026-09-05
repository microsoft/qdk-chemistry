---
name: visualization
version: 'v2.2.0'
description: 'Describes QDK Chemistry visualization tools, their inputs, and their outputs.'
---

# QDK Chemistry Visualization

Visualization tools return interactive MCP App content in clients that support
MCP Apps. They do not modify the source artifact.

| Tool | Input | Output |
|---|---|---|
| `visualize_molecule` | A stored `Structure` | Interactive molecular geometry |
| `visualize_orbitals` | A stored `Wavefunction` and optional orbital indices | Interactive orbital surfaces |
| `visualize_orbital_entanglement` | A stored `Wavefunction` and optional absolute orbital indices | Orbital-entanglement chord diagram |
| `visualize_circuit` | A stored `Circuit` | Interactive circuit diagram |
| `visualize_scatter_plot` | Axis labels and one or more numeric series | Interactive SVG scatter plot |

Project artifact visualizers accept `project_name` and the filename of the
stored artifact. Optional index arguments restrict the displayed subset.
`visualize_orbital_entanglement` reads RDM and mutual-information data from the
wavefunction; the tool returns an error if those fields are absent.

The active tool schema defines the complete argument list and supported
optional display controls.

## Rendering Mechanics

Artifact visualizers load the named file from the project, validate its data
type, convert it to the widget payload, and return MCP App metadata and content.
The client renders that content in an interactive view. A client without MCP
Apps support receives the tool result but may not display the widget.

Molecular views use atomic symbols, coordinates, and available connectivity.
Orbital views read orbital coefficients and basis information from a
wavefunction. Circuit views render the operations and registers stored in a
circuit. Scatter plots render caller-supplied numeric series and can apply the
log-axis options defined by the tool schema.

Visualization calls read existing artifacts. They do not execute the algorithm
that created an artifact and do not alter its stored numerical data.
