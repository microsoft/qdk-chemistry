---
name: remote-execution
version: '{{QDK_CHEMISTRY_VERSION}}'
description: 'Run QDK Chemistry computations on remote systems — SSH servers, HPC clusters, or local backends. Use when: offloading heavy calculations to remote compute, setting up SSH-based remote execution, running parameter sweeps on HPC, or understanding the remote execution architecture.'
---

# Remote Execution

## When to Use

- Computations are too large or slow for the local machine
- Offloading to any SSH-accessible compute server
- Setting up automated remote pipelines or parameter sweeps
- Understanding how remote serialization and job submission work

## Two Modes: Blocking and Async

Remote execution supports two modes:

| Mode | Method | Blocks? | Use When |
|------|--------|---------|----------|
| **Blocking** | `.run()` | Yes — waits for completion | Simple scripts, small jobs, interactive use |
| **Async** | `.submit()` → `.check()` → `.fetch()` | No — returns immediately | Long-running jobs, multi-job pipelines, agent workflows |

### Blocking Mode (Quick Start)

Any algorithm can be transparently offloaded with `.on_remote()`:

```python
from qdk_chemistry.algorithms import create

# Local execution (default)
scf = create("scf_solver")
energy, wfn = scf.run(structure, charge=0, spin_multiplicity=1, basis_or_guess="cc-pvdz")

# Remote execution — same API, different backend
scf_remote = scf.on_remote("ssh", host="compute-server.example.com")
energy, wfn = scf_remote.run(structure, charge=0, spin_multiplicity=1, basis_or_guess="cc-pvdz")
```

The `.run()` call is identical — serialization, file transfer, remote execution, and result deserialization happen automatically. Internally it calls `submit_and_wait()` which polls until the job completes.

### Async Mode (Submit + Check + Fetch)

For long-running jobs, submit without blocking and check later — even from a different script or session:

```python
from qdk_chemistry.algorithms import create

scf = create("scf_solver").on_remote("ssh", host="compute-server.example.com")

# Submit returns immediately with a RemoteJob handle
job = scf.submit(structure, charge=0, spin_multiplicity=1,
                 job_dir="./jobs")
# → writes ./jobs/job_<id>.json and returns a RemoteJob
print(f"Submitted: {job.job_id}")
```

Later (even from a different script or machine):

```python
from qdk_chemistry.remote.job import RemoteJob

# Re-load from disk — the job file is fully self-contained
job = RemoteJob.load("./jobs/job_abc123def456.json")

# Check status (updates the job file on disk)
status = job.check()
print(f"Status: {status.status}")  # "submitted", "running", "Succeeded", "Failed"
print(f"Logs: {status.logs}")

if status.status == "Succeeded":
    # Download and deserialize results
    energy, wfn = job.fetch()
    print(f"Energy: {energy:.6f} Ha")
elif status.status == "Failed":
    print(f"Error: {status.error}")

# Cancel a running job
job.cancel()
```

### Discovering Jobs

```python
from qdk_chemistry.remote.job import RemoteJob

# Find all jobs in a directory
jobs = RemoteJob.discover("./jobs")
for j in jobs:
    print(f"{j.job_id}: {j.status} ({j.backend})")
```

### RemoteJob Lifecycle

```
submit() → job file saved to disk
    │
    ├── check() → queries backend, updates status + file
    │   └── status: "submitted" → "running" → "Succeeded" / "Failed"
    │
    ├── cancel() → cancels on backend, sets status to "canceled"
    │
    └── fetch() → downloads results, deserializes, sets status to "retrieved"
        └── Also computes output_hashes for cache verification
```

The job file (`job_<id>.json`) stores everything needed to resume: backend name, config, opaque backend state, algorithm info, and content hashes.

## Available Backends

### 1. SSH Backend

Run on any server with SSH access and Python installed.

```python
scf = create("scf_solver").on_remote("ssh",
    host="compute-server.example.com",
    identity_file="~/.ssh/id_rsa",    # Optional SSH key
    python_path="/usr/bin/python3",    # Remote Python interpreter
    timeout=3600,                       # Max seconds
)
```

**Prerequisites:**
- SSH access to target server
- Python + qdk_chemistry installed on remote
- SCP available for file transfer

### 2. Local Backend

Subprocess execution for testing remote workflows locally.

```python
scf = create("scf_solver").on_remote("local",
    python_path="/usr/bin/python3",
    timeout=600,
    keep_workdir=True,  # Keep temp dir for debugging
)
```

### 3. Custom Backends

Register your own backend:

```python
from qdk_chemistry.remote import register_backend, RemoteBackend

@register_backend("my_cluster")
class MyClusterBackend(RemoteBackend):
    def connect(self): ...
    def upload(self, local_dir, remote_dir): ...
    def execute(self, script_path, remote_dir): ...
    def download(self, remote_dir, local_dir): ...
    def disconnect(self): ...
```

## Pre-Configured Backend

For reuse across multiple algorithms:

```python
from qdk_chemistry.remote import create_remote

# Configure once
remote = create_remote("ssh", host="compute-server.example.com", timeout=7200)

# Use with multiple algorithms
scf = create("scf_solver").on_remote(remote)
casci = create("multi_configuration_calculator").on_remote(remote)
```

## How It Works

The remote execution system follows a serialize → transfer → execute → retrieve pattern:

```
1. Serialize inputs    → HDF5 files + JSON manifest
2. Upload to remote    → Backend-specific transfer (SCP, shared filesystem, etc.)
3. Generate script     → Self-contained Python script written to remote
4. Execute remotely    → subprocess / SSH command
5. Download outputs    → Backend-specific transfer
6. Deserialize results → Return native Python objects
```

### Job Directory Layout

```
job_dir/
  manifest.json              # Algorithm type, settings, input metadata
  arg_0.structure.h5         # Serialized input arguments
  kwarg_basis.basis_set.h5   # Serialized keyword arguments
  result_0.wavefunction.h5   # Serialized outputs (after execution)
```

### Manifest Format

```json
{
  "algorithm_type": "scf_solver",
  "algorithm_name": "pyscf",
  "settings": {"convergence_threshold": {"type": "float", "value": 1e-8}},
  "args": [{"type": "Structure", "file": "arg_0.structure.h5"}],
  "kwargs": {"basis_or_guess": {"type": "str", "value": "cc-pvdz"}}
}
```

## Combining with Content Hashing

Use content hashes to avoid re-running expensive remote jobs:

```python
scf = create("scf_solver", "pyscf")
run_hash = scf.hash(structure, charge=0, spin_multiplicity=1, basis_or_guess="cc-pvdz")

if run_hash not in result_cache:
    scf_remote = scf.on_remote("ssh", host="compute-server.example.com")
    energy, wfn = scf_remote.run(structure, 0, 1, "cc-pvdz")
    result_cache[run_hash] = (energy, wfn)
```

## Remote Parameter Sweep Pattern

```python
from qdk_chemistry.algorithms import create
from qdk_chemistry.remote import create_remote
from qdk_chemistry.data import Structure
import numpy as np

remote = create_remote("ssh", host="compute-server.example.com", timeout=3600)

distances = np.linspace(1.0, 5.0, 20)
results = []

for d in distances:
    structure = Structure(
        symbols=["N", "N"],
        coordinates=np.array([[0, 0, 0], [0, 0, d]])
    )
    scf = create("scf_solver").on_remote(remote)
    energy, wfn = scf.run(structure, charge=0, spin_multiplicity=1, basis_or_guess="cc-pvdz")
    results.append((d, energy))
```

## MCP Remote Job Tools

The MCP server exposes remote execution through algorithm-backed `run_*` chemistry tools plus 4 job-management tools. `run_resource_estimation` is local-only because it wraps `Circuit.estimate()` directly. There is **no** standalone `submit_remote_job` tool. To run remotely, call the same supported `run_*` tool you would use locally and add remote parameters.

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `run_*` chemistry tools | Run locally or submit to a remote backend | Normal tool parameters + `remote`, `remote_config?`, `remote_timeout?` |
| `check_remote_job` | Query job status and logs | `project_name`, `job_id` |
| `retrieve_remote_results` | Download results into the project directory | `project_name`, `job_id` |
| `list_remote_jobs` | List all jobs for a project (optionally filtered by status) | `project_name`, `status_filter?` |
| `cancel_remote_job` | Cancel a running job | `project_name`, `job_id` |

When `remote` is set, the tool runs on the named backend. If the computation finishes within `remote_timeout` seconds (default `120`), the tool returns the normal inline result. If it takes longer, the tool returns `{"status": "submitted", ...}` with a `job_id` and related metadata.

### MCP Async Workflow

```
1. Call any run_* tool with remote="backend_name" and remote_timeout=0
   → Returns immediately with:
     {"status": "submitted", "job": {"job_id": "...", "job_file": "...", "backend": "...", "status": "submitted", "run_hash": "...", "submitted_at": "..."}}

2. check_remote_job(project_name, job_id)
   → Returns: {job_id, status, logs, elapsed, submitted_at, error?, run_hash?, input_hashes?, output_hashes?}
   Statuses: "submitted" → "running" → "Succeeded" / "Failed" / "canceled"

3. retrieve_remote_results(project_name, job_id)
   → Downloads output files into the project directory
   → Returns: {status: "retrieved", job_id, downloaded_files, values, output_hashes}
   → Downloaded files are then available to ALL other MCP tools
      (visualize, further computation, etc.)
```

### MCP Submit Example

```
run_scf(
    project_name="n2_study",
    structure_filename="n2.structure.json",
    out_wavefunction_filename="n2_scf.wavefunction.json",
    charge=0,
    spin_multiplicity=1,
    basis_set="cc-pvdz",
    remote="ssh",
    remote_config={"host": "compute-server.example.com", "timeout": 7200},
    remote_timeout=0
)
→ {"status": "submitted", "job": {"job_id": "abc123def456", "backend": "ssh", ...}}
```

Then poll:
```
check_remote_job(project_name="n2_study", job_id="abc123def456")
→ {"status": "running", "elapsed": "0h 12m 34s", "logs": "..."}
```

When done:
```
retrieve_remote_results(project_name="n2_study", job_id="abc123def456")
→ {"status": "retrieved", "downloaded_files": ["n2_scf.wavefunction.json"], ...}
```

The downloaded files are now in the project directory and can be used by any MCP tool — `visualize_orbital_entanglement`, `run_qubit_mapper`, etc.

### Key Details

- **Job persistence**: The submitted response includes a `job_file` path in the jobs cache, containing everything needed to check, cancel, or fetch later — even from a different session.
- **Content hashes**: The initial `run_*` submission records `run_hash`; `check_remote_job` can also report `input_hashes` and `output_hashes`, and `retrieve_remote_results` returns `output_hashes`. These enable cache verification.
- **Input loading**: Use the normal input filename parameters for the chosen `run_*` tool. Inputs are loaded using the same type-marker conventions as every MCP chemistry tool (`.structure.`, `.wavefunction.`, etc.).
- **Backend config**: `remote_config` is passed as a dict matching the backend's constructor options (host, timeout, python_path, identity_file, etc.)

## Error Handling

- Remote errors include both the Python traceback and any remote log files
- SSH backend captures stderr for diagnostics
- Settings are validated locally before remote submission
- Temporary directories are cleaned up after execution (configurable with `keep_workdir`)

## Key Considerations

1. **qdk_chemistry must be installed on the remote** — the generated script imports it
2. **Settings are locked during remote execution** — configure everything before `.on_remote()`
3. **Large data transfers** — HDF5 serialization can produce large files for big molecules; ensure adequate bandwidth
4. **Timeouts** — set realistic timeouts; default is 3600s (1 hour)
5. **Backend availability** — in Python, check with `from qdk_chemistry.remote import available_backends; print(available_backends())`; in MCP workflows, call `list_remote_backends` before selecting a backend.
