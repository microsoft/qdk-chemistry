---
name: visualization
version: 'v2.2.1'
description: 'Describes QDK Chemistry visualization tools, simulated measurement histograms, and QRE Pareto frontier plots. Use when visualizing simulation results or resource-estimation tradeoffs.'
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

## Simulated Measurement Histograms

Plot the measured bitstring counts returned by circuit execution or phase
estimation. A `QpeResult` can contain `bitstring_counts`; check that it is
present before plotting. Building a circuit alone does not produce a
histogram. If counts are absent, run the selected simulator with the requested
shots rather than inventing a distribution from a single phase estimate.

Use the installed `qdk.widgets.Histogram` API in a Python notebook, inspecting
its accepted input format first. For a static plot or a client without widget
support, use a Python bar chart with bitstrings on the horizontal axis and
counts or probabilities on the vertical axis. The MCP tools listed above do
not currently include a dedicated histogram visualizer; do not describe the
scatter-plot tool as a histogram renderer.

Preserve the result's bit order, label the number of shots, and distinguish
counts from normalized probabilities. Identify any postselection and the
number of retained shots. Do not combine the separate bit measurements of an
iterative-QPE run into a multi-shot phase histogram unless complete phase
bitstrings were actually sampled repeatedly.

## QRE Pareto Frontiers

For a stored circuit, call `estimate_circuit_qre` and use the returned
`estimates` records. For a direct `qdk.qre` study, retain the returned
`EstimateTable` and use `qdk.qre.plot_estimates([table], runtime_unit="s")`.
See [Resource Estimation](../resource-estimation/SKILL.md) for the estimator
settings and direct-code workflow. Inspect the installed plotting API before
generating Python code.

To display MCP estimator results in an MCP App, inspect the returned column
names and pass the physical-qubit and runtime values to
`visualize_scatter_plot` as numeric series. Use physical qubits on the
horizontal axis and runtime on the vertical axis, with explicit time units;
logarithmic axes are useful when all plotted values are positive. Follow the
active tool schema and series payload format rather than guessing field
names. Serialized `estimates` records are not an `EstimateTable` and cannot
be passed directly to `plot_estimates`.

A Pareto frontier contains nondominated estimates: no other point uses no
more physical qubits and no more runtime while strictly improving at least
one. If combining tables, compute the frontier only within comparable
architecture, physical-error-rate, and total-error-budget assumptions. Label
different assumptions as separate series, and distinguish all sampled points
from the frontier. Report an empty feasible set instead of plotting invented
points. Display the circuit, architecture, error rate, error budget, and sweep
settings with the plot; a frontier describes the evaluated configurations,
not a guaranteed global optimum.

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
