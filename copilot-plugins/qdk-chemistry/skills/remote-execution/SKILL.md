---
name: remote-execution
version: 'v2.1.0'
description: 'Describes the QDK Chemistry remote execution API and job lifecycle.'
---

# Remote Execution

`algorithm.run(..., remote=name_or_backend)` submits a call through a registered
backend and waits for its result. The module-level
`qdk_chemistry.remote.run(algorithm, ..., remote=name_or_backend)` function has
the same blocking behavior. For asynchronous execution,
`qdk_chemistry.remote.submit(algorithm, ..., remote=name_or_backend)` returns a
`Job` immediately.

A `Job` supports `check()`, `wait()`, `fetch()`, `cancel()`, `save()`, and
`load()`. Pass `job_dir` to `submit` to save the job record automatically;
without it, the SDK returns an in-memory job that can be saved explicitly.

MCP `run_*` tools accept `remote`, `remote_config`, and `remote_timeout` when
the selected algorithm supports remote execution. A submitted call returns a
job identifier. `check_remote_job`, `retrieve_remote_results`,
`list_remote_jobs`, and `cancel_remote_job` operate on persisted jobs.

The SDK's `available_backends()` reports registered names. MCP
`list_remote_backends` and `describe_backend` report the registered backends
and their MCP-safe configuration arguments.

## Execution Modes

Blocking execution submits a call, waits for terminal job state, retrieves the
outputs, and deserializes the algorithm result. Asynchronous SDK execution
returns before result retrieval and persists the job only when `job_dir` is
provided or `Job.save()` is called. MCP-managed asynchronous execution persists
the record automatically so another process or session can inspect the same
submitted job.

## Job Lifecycle

A submitted job progresses through backend-defined nonterminal states and then
to a success, failure, or cancellation state. Status inspection updates the
persisted record with backend status, logs, elapsed time, and error information
when available. Result retrieval downloads and deserializes outputs from a
successful job. Cancellation requests termination from the backend.

## Serialization

Remote execution serializes algorithm identity, settings, positional and
keyword inputs, and content hashes into a manifest. Data objects are written as
separate files referenced by that manifest. The backend transfers the manifest
and inputs, invokes the algorithm remotely, and transfers output files back.
Nested algorithm-reference settings remain nested in the manifest.

## Backends

The package ships one backend: `local`. It runs the serialization and job
protocol in an isolated local subprocess for testing and development.

Installed plugins can register additional backends through a
`QdkChemistryPlugin` in the `qdk_chemistry.plugins` entry-point group. A custom
backend can also be registered directly with `register_backend`. Registered
custom backends implement connection, transfer, submission, status, retrieval,
and cancellation operations exposed by `RemoteBackend`.

The selected remote environment needs a compatible QDK Chemistry installation
and access to dependencies required by the selected algorithm. Backend
configuration is backend-specific; MCP exposes only options declared safe by
that backend and reported by `describe_backend`.

## MCP Results

If a remote MCP call completes within `remote_timeout`, the tool returns its
normal result envelope. Otherwise it returns submitted-job metadata containing
a job identifier. Retrieved artifact files are placed in the project directory
and can be passed to other MCP tools by filename.
