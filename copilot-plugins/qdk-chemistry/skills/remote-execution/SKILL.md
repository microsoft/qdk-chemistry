---
name: remote-execution
version: 'v2.0.0'
description: 'Describes the QDK Chemistry remote execution API and job lifecycle.'
---

# Remote Execution

`on_remote` binds an algorithm to a configured backend. `run` submits the call
and waits for a result. `submit` returns a persisted job handle. A job handle
supports status inspection, result retrieval, and cancellation.

MCP `run_*` tools accept `remote`, `remote_config`, and `remote_timeout` when
the selected algorithm supports remote execution. A submitted call returns a
job identifier. `check_remote_job`, `retrieve_remote_results`,
`list_remote_jobs`, and `cancel_remote_job` operate on persisted jobs.

Backend discovery and description APIs define available backend names and
configuration arguments.

## Execution Modes

Blocking execution submits a call, waits for terminal job state, retrieves the
outputs, and deserializes the algorithm result. Asynchronous execution persists
a job record and returns before result retrieval. The job record allows another
process or session to inspect the same submitted job.

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

The SSH backend transfers files and invokes the configured Python interpreter
on an SSH-accessible system. The local backend runs the same serialization and
job protocol in a subprocess. Registered custom backends implement connection,
transfer, execution, status, retrieval, and cancellation operations exposed by
the remote backend interface.

The selected remote environment needs a compatible QDK Chemistry installation
and access to dependencies required by the selected algorithm. Backend
configuration can include connection details, interpreter location, timeouts,
and backend-specific options reported by `describe_backend`.

## MCP Results

If a remote MCP call completes within `remote_timeout`, the tool returns its
normal result envelope. Otherwise it returns submitted-job metadata containing
a job identifier. Retrieved artifact files are placed in the project directory
and can be passed to other MCP tools by filename.
