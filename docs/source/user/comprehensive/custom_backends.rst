.. _custom-backends:

Custom Cache & Remote Backends
==============================

QDK/Chemistry's cache and remote execution systems use a plugin architecture.
You can register your own backends to integrate with cloud databases, job schedulers,
or custom infrastructure.

.. contents:: On This Page
   :local:
   :depth: 2


Writing a custom cache backend
------------------------------

A cache backend stores job metadata and serialized data objects. Implement the
:class:`~qdk_chemistry.remote.cache.base.CacheBackend` interface:

.. code-block:: python

   import json
   import sqlite3
   import tempfile
   from pathlib import Path

   from qdk_chemistry.remote.cache.base import CacheBackend
   from qdk_chemistry.remote.cache import register_cache
   from qdk_chemistry.remote.job import Job


   @register_cache("sqlite")
   class SQLiteCache(CacheBackend):
       """Cache backend backed by a local SQLite database."""

       def __init__(self, db_path: str = "qdk_cache.db", **kwargs):
           super().__init__(**kwargs)
           self._db_path = db_path
           self._conn = sqlite3.connect(db_path)
           self._conn.execute(
               "CREATE TABLE IF NOT EXISTS jobs "
               "(run_hash TEXT PRIMARY KEY, data TEXT)"
           )
           self._conn.execute(
               "CREATE TABLE IF NOT EXISTS blobs "
               "(content_hash TEXT PRIMARY KEY, blob BLOB)"
           )
           self._conn.commit()

       def get_job(self, run_hash: str) -> Job | None:
           """Retrieve job metadata by run_hash, or None on miss."""
           row = self._conn.execute(
               "SELECT data FROM jobs WHERE run_hash = ?", (run_hash,)
           ).fetchone()
           if row is None:
               return None
           return Job(**json.loads(row[0]))

       def put_job(self, run_hash: str, job: Job) -> None:
           """Store or update job metadata keyed by run_hash."""
           self._conn.execute(
               "INSERT OR REPLACE INTO jobs (run_hash, data) VALUES (?, ?)",
               (run_hash, json.dumps(job.to_dict())),
           )
           self._conn.commit()

       def get_data(self, content_hash: str, data_class):
           """Retrieve a serialized data object by content hash."""
           row = self._conn.execute(
               "SELECT blob FROM blobs WHERE content_hash = ?", (content_hash,)
           ).fetchone()
           if row is None:
               return None
           # Write blob to a temporary HDF5 file, then load via data_class
           with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
               tmp.write(row[0])
               tmp_path = tmp.name
           try:
               return data_class.from_hdf5_file(tmp_path)
           finally:
               Path(tmp_path).unlink(missing_ok=True)

       def put_data(self, content_hash: str, data_obj) -> None:
           """Store a serialized data object by content hash (idempotent)."""
           with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
               tmp_path = tmp.name
           try:
               data_obj.to_hdf5_file(tmp_path)
               blob = Path(tmp_path).read_bytes()
           finally:
               Path(tmp_path).unlink(missing_ok=True)
           self._conn.execute(
               "INSERT OR IGNORE INTO blobs (content_hash, blob) VALUES (?, ?)",
               (content_hash, blob),
           )
           self._conn.commit()

       def delete_job(self, run_hash: str) -> None:
           self._conn.execute("DELETE FROM jobs WHERE run_hash = ?", (run_hash,))
           self._conn.commit()

       def delete_data(self, content_hash: str) -> None:
           self._conn.execute(
               "DELETE FROM blobs WHERE content_hash = ?", (content_hash,)
           )
           self._conn.commit()

       def clear(self) -> None:
           self._conn.execute("DELETE FROM jobs")
           self._conn.execute("DELETE FROM blobs")
           self._conn.commit()

After registration, the backend is available everywhere:

.. code-block:: python

   from qdk_chemistry.algorithms import create
   from qdk_chemistry.remote.cache import resolve_cache

   # By name (resolve_cache forwards kwargs to the constructor)
   cache = resolve_cache("sqlite", db_path="./results.db")

   # Or directly
   scf = create("scf_solver")
   energy, wfn = scf.run(structure, 0, 1, "cc-pvdz",
                          cache=SQLiteCache(db_path="./results.db"))

Via MCP, pass the cache path or name as a string:

.. code-block:: text

   {"tool": "run_scf", "arguments": {"cache": "./my_cache", ...}}


Writing a custom remote backend
--------------------------------

A remote backend handles job submission, polling, and result retrieval. Implement the
:class:`~qdk_chemistry.remote.backends.base.RemoteBackend` interface:

.. code-block:: python

   from qdk_chemistry.remote.backends.base import RemoteBackend, JobStatus, register_backend
   from qdk_chemistry.remote.job import Job


   @register_backend("slurm")
   class SlurmBackend(RemoteBackend):
       """Remote backend for SLURM job schedulers."""

       name = "slurm"

       def __init__(self, host: str, partition: str = "default", **kwargs):
           super().__init__(**kwargs)
           self._host = host
           self._partition = partition

       def connect(self) -> None:
           """Establish connection to the SLURM cluster."""
           self._ssh = connect_ssh(self._host)

       def disconnect(self) -> None:
           """Close the connection."""
           self._ssh.close()

       def submit(self, payload: dict, job_dir=None) -> Job:
           """Submit a job to the SLURM scheduler.

           The payload contains algorithm_type, algorithm_name, settings,
           serialized args/kwargs, run_hash, and input_hashes.
           """
           # Upload input files, generate a runner script, sbatch it
           slurm_id = self._sbatch(payload)
           job = Job(
               job_id=slurm_id,
               backend=self.name,
               backend_config={"host": self._host, "partition": self._partition},
               backend_state={"slurm_id": slurm_id, "workdir": "/scratch/..."},
               algorithm_info={
                   "type": payload.get("algorithm_type"),
                   "name": payload.get("algorithm_name"),
                   "settings": payload.get("settings"),
               },
               run_hash=payload.get("run_hash"),
               input_hashes=payload.get("input_hashes"),
           )
           if job_dir:
               job.save(Path(job_dir) / f"job_{slurm_id}.json")
           return job

       def check(self, backend_state: dict) -> JobStatus:
           """Poll SLURM for job status."""
           status = self._squeue(backend_state["slurm_id"])
           return JobStatus(status=status)

       def cancel(self, backend_state: dict) -> None:
           """Cancel the SLURM job."""
           self._scancel(backend_state["slurm_id"])

       def fetch(self, backend_state: dict, local_dir=None):
           """Download results from the SLURM working directory."""
           return self._download_results(backend_state["workdir"], local_dir)

After registration:

.. code-block:: python

   energy, wfn = scf.run(structure, 0, 1, "cc-pvdz",
                          cache="./cache",
                          remote="slurm")

   # Or with configuration
   energy, wfn = scf.run(structure, 0, 1, "cc-pvdz",
                          cache="./cache",
                          remote=SlurmBackend(host="hpc.example.com",
                                              partition="gpu"))


Entry-point discovery
---------------------

For backends distributed as separate packages, register via Python entry points in
``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."qdk_chemistry.cache_backends"]
   sqlite_cache = "my_package.cache:SQLiteCache"

   [project.entry-points."qdk_chemistry.remote_backends"]
   slurm = "my_package.remote:SlurmBackend"

QDK/Chemistry auto-discovers these at import time.


Key contracts
-------------

Cache backends:
  - ``get_job`` / ``put_job``: keyed by ``run_hash`` (16-char hex string)
  - ``get_data`` / ``put_data``: keyed by ``content_hash`` (16-char hex string)
  - ``put_data`` should be idempotent (skip if already exists)
  - ``is_shared``: set ``True`` if the cache is visible from remote compute nodes

Remote backends:
  - ``submit`` returns a ``Job`` with ``backend_state`` containing everything needed to poll/fetch
  - ``check`` returns a ``JobStatus`` with ``status`` string (``"submitted"``, ``"running"``, ``"Succeeded"``, ``"Failed"``)
  - ``fetch`` downloads and deserializes the algorithm result
  - ``backend_config`` must be JSON-serializable (stored in job files for recovery)
