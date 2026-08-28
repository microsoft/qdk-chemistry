.. _remote-execution:

Remote Execution & Caching
==========================

QDK/Chemistry supports transparent remote execution and result caching so that
expensive calculations can be offloaded to HPC clusters while results are automatically
persisted for recovery and reuse.

.. contents:: On This Page
   :local:
   :depth: 2


Overview
--------

There are two independent features that combine naturally:

**Caching** stores algorithm results keyed by a deterministic content hash of the inputs.
Identical computations are never repeated — the result is fetched from the cache instantly.

**Remote execution** offloads computation to a backend (SSH server, SLURM cluster, or a local
subprocess) while persisting job state for crash recovery.

Both features are available through all three interfaces (Python SDK, MCP tools, CLI) with
consistent behavior.


Python SDK
----------

.. code-block:: python

   from qdk_chemistry.algorithms import create

   scf = create("scf_solver")

   # Local, no caching
   energy, wfn = scf.run(structure, 0, 1, "cc-pvdz")

   # Local with caching (cache hit skips execution)
   energy, wfn = scf.run(structure, 0, 1, "cc-pvdz",
                          cache="./cache")

   # Remote with caching
   energy, wfn = scf.run(structure, 0, 1, "cc-pvdz",
                          cache="./cache", remote="ssh")

   # Force recomputation despite cache hit
   energy, wfn = scf.run(structure, 0, 1, "cc-pvdz",
                          cache="./cache", force_rerun=True)

The SDK requires a cache when using a remote backend — this ensures job metadata is always
persisted for recovery.


MCP tools & CLI
---------------

All ``run_*`` MCP tools and CLI commands also accept ``cache``, ``remote``,
``remote_config``, and ``remote_timeout`` parameters, with identical behavior
to the Python SDK.  See :doc:`agents` for MCP-specific details.


Job lifecycle
-------------

.. code-block:: text

   ┌─────────┐   submit  ┌───────────┐  poll   ┌───────────┐  fetch  ┌───────────┐
   │  (new)  │──────────►│ submitted │────────►│ Succeeded │────────►│ retrieved │
   └─────────┘           └───────────┘         └───────────┘         └───────────┘
                               │                     │
                               ▼                     ▼
                         ┌──────────┐          ┌──────────┐
                         │  Failed  │          │ canceled │
                         └──────────┘          └──────────┘

Jobs are persisted in the cache as ``<run_hash>.job.json`` files. Each file contains:

- ``job_id``, ``backend``, ``backend_config``, ``backend_state``
- ``algorithm_info`` (type, name, settings)
- ``run_hash`` (deterministic content hash of algorithm + settings + inputs)
- ``input_hashes`` / ``output_hashes``
- ``status``, ``submitted_at``

The ``run_hash`` enables deduplication: if you re-run the same computation, the cache finds
the existing job and either returns the cached result (if done) or resumes polling (if still
running).


Crash recovery
~~~~~~~~~~~~~~

If a session crashes or the MCP connection drops:

1. The job file survives in the cache directory.
2. Re-running the same tool call with the same inputs computes the same ``run_hash``.
3. The cache finds the existing job and resumes where it left off.
4. Alternatively, use ``check_remote_job(job_id)`` or ``list_remote_jobs()`` to find and
   manage jobs directly.


Available backends
------------------

.. list-table::
   :header-rows: 1

   * - Backend
     - Description
     - Configuration
   * - ``local``
     - Runs in a subprocess on the same machine (for testing)
     - ``{}``
   * - ``ssh``
     - Executes on a remote SSH-accessible server
     - ``{"host": "cluster.example.com", "python_path": "/opt/conda/bin/python"}``

Additional backends (e.g., SLURM, cloud HPC) can be added as plugins — see
:ref:`custom-backends`.

Use ``list_remote_backends()`` and ``describe_backend("remote", "ssh")`` to discover and
inspect available backends at runtime.


Cache backends
--------------

.. list-table::
   :header-rows: 1

   * - Backend
     - Description
     - Configuration
   * - ``folder``
     - Content-addressed files in a local directory
     - ``path="/path/to/cache"``
   * - ``tiered``
     - Layered local + remote caches
     - ``local_path="./cache", remote=<CacheBackend>``

Use ``list_cache_backends()`` to discover available cache backends.


Shared caches
~~~~~~~~~~~~~

When local and remote machines share a filesystem (e.g., NFS mount on an HPC cluster),
set ``is_shared=True``:

.. code-block:: python

   from qdk_chemistry.remote.cache.folder import FolderCache

   shared = FolderCache("/mnt/shared/cache", is_shared=True)
   energy, wfn = scf.run(structure, 0, 1, "cc-pvdz",
                          cache=shared, remote="ssh")

With a shared cache:

- **Inputs**: The remote node reads directly from the shared cache — no additional upload needed.
- **Outputs**: The remote node writes results to the shared cache — no additional download needed.
- The client reconstructs the result from the shared cache.


Configuration
-------------

.. list-table::
   :header-rows: 1

   * - Environment variable
     - Default
     - Description
   * - ``QDK_SCRATCH_DIR``
     - ``/scratch``
     - Base directory for project data and cache
   * - ``QDK_JOBS_DIR``
     - ``<scratch>/jobs``
     - Default cache directory for remote jobs (when no explicit cache is provided)
