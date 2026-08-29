# MCP Diagnostics Reference

Use the returned `status`, error message, and tool schema to diagnose a failed
call. This reference documents interface constraints and their effects. It does
not prescribe a scientific method, accuracy target, resource budget, or retry
sequence.

## Coordinate Units

**Symptom:** Structure visualization is implausible, or an SCF calculation
fails early for a structure that should be valid.

**Cause:** `create_structure` expects coordinates in Bohr while many input
sources use Angstrom.

**Correction:** Convert coordinates with `convert_coordinates` before calling
`create_structure`. The conversion is $1\ \mathring{\mathrm{A}} =
1.8897259886\ \mathrm{Bohr}$.

## Restricted-Orbital Active-Space Inputs

**Symptom:** A valence selector, ASCI calculation, or AutoCAS stage rejects an
open-shell SCF wavefunction as incompatible.

**Cause:** This active-space path consumes one shared set of spatial orbitals;
an unrestricted result has separate alpha and beta orbital sets.

**Correction:** Generate the active-space input with `run_scf` using:

```json
{"method": "hf", "scf_type": "restricted"}
```

For an open-shell system this produces the compatible restricted-orbital
reference. Pass its returned wavefunction filename to the downstream tool.

## Missing RDM or Mutual-Information Data

**Symptom:** Entropy-based AutoCAS cannot use an SCF result, or an orbital
entanglement visualization has no mutual-information data.

**Cause:** A bare SCF wavefunction does not include the reduced density matrices
needed by entropy-based selection. Mutual information is a separate requested
output.

**Correction:** Create a multi-configuration result with
`calculate_one_rdm=True` and `calculate_two_rdm=True` before entropy-based
selection. Also pass `calculate_mutual_information=True` when the result will
be sent to `visualize_orbital_entanglement`.

## QPE Sentinel Defaults

**Symptom:** `run_phase_estimation` reports an error involving `num_bits` or
evolution time.

**Cause:** The nested defaults are intentionally invalid:
`settings.qpe_circuit_builder.num_bits=-1` and
`settings.qpe_circuit_builder.unitary_builder.time=0.0`.

**Correction:** Supply both values in the nested `settings` dictionary. Their
scientific values are caller-supplied; use the algorithm catalog and active
input schema to validate the surrounding sub-algorithm configuration.

```json
{
  "qpe_circuit_builder": {
    "num_bits": 12,
    "unitary_builder": {"algorithm_name": "trotter", "time": 1.0}
  }
}
```

## Existing Output Files

**Symptom:** A tool returns `{"status": "exists"}` instead of running.

**Cause:** Output-producing tools do not overwrite an existing typed artifact
when `overwrite` is omitted or `false`.

**Correction:** Choose a new output filename or pass `overwrite=True` when
replacing the existing artifact is intended.

## Remote Execution and Retrieval

**Symptom:** A remote call returns a job identifier without local output files,
or completed outputs are not yet available locally.

**Cause:** Remote execution and artifact retrieval are separate operations.

**Correction:** Use `check_remote_job(project_name, job_id)` to inspect the
returned job state and logs. Use `retrieve_remote_results(project_name, job_id)`
after completion. Preserve the `project_name` and job identifier returned by
submission.

## Endpoint and Evidence Mismatches

**Symptom:** A report labels a gate count as an energy result, or labels a
mapping qubit count as a physical-resource estimate.

**Cause:** Mapping, circuit construction, circuit execution, QPE, and resource
estimation produce distinct artifacts.

**Correction:** Use `get_circuit_stats` for logical metrics of a supplied
circuit and `run_resource_estimation` for physical-resource Pareto points of a
supplied circuit. A circuit-analysis workflow does not execute
`run_phase_estimation`; a QPE result is not a resource estimate.

## Algorithm Settings Errors

**Symptom:** A tool rejects an algorithm name, setting, or nested configuration.

**Cause:** Available implementations and accepted settings depend on the active
installation and selected parent algorithm.

**Correction:** Call `list_algorithms` to discover implementations,
`describe_algorithm` for accepted settings, and the default-setting tools to
inspect nested defaults. Apply an override at the nested algorithm that owns
the setting, preserving unspecified settings.

## Result Handling

**Symptom:** A later call fails because it received a missing filename or an
error object as an input.

**Cause:** MCP responses use a status envelope and downstream calls require
actual output filenames from successful results.

**Correction:** Check `status` before reading `result`, then pass the returned
filename with its data type to the next tool. Keep the diagnostic response with
the failed call rather than treating it as a produced artifact.