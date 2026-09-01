---
name: resource-estimation
version: 'v2.1.0'
description: 'Explains QDK circuit resource-estimation parameters, result forms, and conceptual construction of custom QRE architecture and ISA studies. Use when estimating a stored circuit, choosing estimate_circuit parameters, interpreting estimates, or generating Python code for a custom qdk.qre analysis.'
---

# Circuit Resource Estimation

Choose between exactly two paths:

1. **Stored-circuit tool path** — call `estimate_circuit` with the parameter
   objects documented below. Use this for a concise estimate of an existing
   project circuit.
2. **Direct QDK code path** — generate Python that uses `qdk.qre` directly,
   after inspecting the installed QDK API. Use this when the user needs more
   control over architectures, trace transforms, ISA enumeration, result
   columns, comparisons, or plots.

Do not present these as successive steps. They are alternative interfaces for
different levels of control.

## Path 1: Estimate a stored circuit

Use `estimate_circuit` to apply the estimator exposed by a stored QDK Chemistry
`Circuit`. The tool returns its result inline and does not create another
project artifact.

The tool accepts `project_name`, `circuit_filename`, and optional `params`.
`params` is either one estimator-parameter object or a list of objects for a
batch comparison. Omit it to use QDK defaults. MCP calls use the camel-case JSON
field names below; Python code that constructs `EstimatorParams` uses the
corresponding snake-case attributes.

## Parameter objects

A parameter object may contain:

| Field | Purpose |
|---|---|
| `qubitParams` | Selects a built-in physical-qubit model or describes a custom one. |
| `qecScheme` | Selects a built-in error-correction scheme or describes a custom one. |
| `distillationUnitSpecifications` | Supplies custom magic-state distillation units. |
| `constraints` | Restricts runtime, physical qubits, factories, or logical depth. |
| `errorBudget` | Sets the total error budget or partitions it by source. |
| `estimateType` | Requests `singlePoint` or `frontier`. |

A list applies independent parameter objects to the same circuit. Use it to
compare assumptions rather than making separate tool calls.

### Physical-qubit parameters

`qubitParams.name` accepts these built-in models:

- `qubit_gate_us_e3`
- `qubit_gate_us_e4`
- `qubit_gate_ns_e3`
- `qubit_gate_ns_e4`
- `qubit_maj_ns_e4`
- `qubit_maj_ns_e6`

Instead of `name`, a custom model can use the fields
`instructionSet`, `oneQubitMeasurementTime`,
`twoQubitJointMeasurementTime`, `oneQubitGateTime`, `twoQubitGateTime`,
`tGateTime`, `oneQubitMeasurementErrorRate`,
`twoQubitJointMeasurementErrorRate`, `oneQubitGateErrorRate`,
`twoQubitGateErrorRate`, `tGateErrorRate`, and `idleErrorRate`.
Measurement-error fields accept either one probability or an object with
`process` and `readout` probabilities. Time values use the duration strings
accepted by QDK.

### Error correction

`qecScheme.name` accepts `surface_code` or `floquet_code`. A custom scheme can
instead use `errorCorrectionThreshold`, `crossingPrefactor`,
`distanceCoefficientPower`, `logicalCycleTime`,
`physicalQubitsPerLogicalQubit`, and `maxCodeDistance`. Formula fields use the
expression syntax accepted by the QDK estimator.

### Error budget and constraints

`errorBudget` may be a probability strictly between zero and one, or an object
with `logical`, `tstates`, and `rotations` probabilities. A partition describes
how the total failure budget is divided among logical errors, distilled
T states, and synthesized rotations.

`constraints` may contain `logicalDepthFactor`, `maxTFactories`, `maxDuration`,
and `maxPhysicalQubits`. `maxDuration` and `maxPhysicalQubits` are alternative
space/time constraints and cannot be supplied together.

### Distillation units

Each `distillationUnitSpecifications` item can contain `name`, `displayName`,
`numInputTs`, `numOutputTs`, `failureProbabilityFormula`,
`outputErrorRateFormula`, `physicalQubitSpecification`,
`logicalQubitSpecification`, and
`logicalQubitSpecificationFirstRoundOverride`. A protocol-specific
specification supplies `numUnitQubits` and `durationInQubitCycleTime`.

Custom qubit, QEC, formula, and distillation settings are coupled. Generate them
only from a coherent hardware model; do not combine individually plausible
values without checking the QDK estimator's validation rules.

## Results

A single parameter object returns one estimator result. A parameter list returns
a list in the same order. Depending on `estimateType`, a result can include a
single selected estimate or frontier entries. Common sections include logical
counts, physical counts, formatted counts, logical-qubit/QEC information,
T-factory data, report data, and assumptions. Inspect status and error fields
before reading resource counts.

## Path 2: Generate direct `qdk.qre` code

When the stored-circuit tool is too restrictive, generate a Python version of
the workflow using `qdk.qre` directly. First inspect the installed `qdk.qre`
modules, callable signatures, model classes, query constructors, instruction
identifiers, property keys, and result-table methods. Do not infer the current
API from this skill alone: QDK evolves independently of QDK Chemistry.

After that inspection, generated code should have roughly this shape, adapted
to the APIs actually present and with models, ranges, and transforms chosen for
the requested study:

```python
from qdk.qre import LatticeSurgery, PSSPC, estimate, plot_estimates
from qdk.qre.models import Majorana, RoundBasedFactory, ThreeAux

# `circuit` may be loaded from a QDK Chemistry project or produced by a
# chemistry circuit-building workflow.
application = circuit.get_qre_application()

# Describe physical operations independently of the logical circuit.
architecture = Majorana(error_rate=physical_error_rate)

# Transform application operations into logical operations supported by the
# selected code. Lists ask QRE to enumerate alternatives.
trace_query = (
   application.q()
   * PSSPC.q(num_ts_per_rotation=rotation_t_counts)
   * LatticeSurgery.q(slow_down_factor=slow_down_factors)
)

# Enumerate compatible compute-code and magic-state-factory implementations.
compute_code = ThreeAux.q(distance=compute_distances)
factory_code = ThreeAux.q(distance=factory_distances)
isa_query = compute_code * RoundBasedFactory.q(
   code_query=factory_code,
   use_cache=True,
)

results = estimate(
   application=application,
   architecture=architecture,
   isa_query=isa_query,
   trace_query=trace_query,
   max_error=max_error,
   use_graph=use_graph,
   name=study_name,
)

# The table can be enriched, inspected, compared, or plotted.
frame = results.as_frame()
figure = plot_estimates([results], runtime_unit="s")
```

This is a code-generation path, not another MCP estimation function. The code
is intentionally a skeleton: an agent should use what it learned from the
installed QDK to replace its Majorana, `ThreeAux`, factory, transform, and sweep
choices when the requested hardware model or scientific comparison differs.

Conceptually, generated code can:

1. Load or construct a QDK Chemistry `Circuit`, then call
   `get_qre_application()`.
2. Select or construct an architecture containing physical-operation times and
   error rates.
3. Build a trace query from the application and compatible transforms. This may
   sweep rotation-synthesis choices with `PSSPC` and logical-operation timing
   choices with `LatticeSurgery` when the installed API supports them.
4. Build an ISA query from compatible QEC and factory models. This may sweep
   compute-code distances independently from factory code distances using
   `ThreeAux` and `RoundBasedFactory` when those are suitable and available.
5. Call `qdk.qre.estimate` with the application, architecture, ISA query, trace
   query, and error limit.
6. Enrich the returned table with query provenance and derived properties,
   convert it to a data frame, compare several tables, or use `plot_estimates`
   to visualize physical-qubit/runtime Pareto fronts.

Direct QDK code permits more customization than `estimate_circuit`: an agent
can vary trace transforms, instruction implementations, code and factory
families, independent parameter sweeps, architecture models, error composition,
post-processing, graph pruning, custom result columns, and comparative plots.
The generated architecture and query graph should follow the scientific
question rather than copy the illustrative skeleton unchanged.

Before generating code, inspect the installed `qdk.qre` API and the relevant
model/query classes because this interface evolves independently of the stored
circuit format. Keep architecture, QEC, factory, and trace assumptions visible
in generated output. Use `use_graph=False` when completeness of the enumerated
Pareto frontier is more important than pruning cost.
