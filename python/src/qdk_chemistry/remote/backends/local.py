"""Local backend for testing remote execution of QDK/Chemistry algorithms.

This backend simulates remote execution by running in a subprocess on the
local machine. Useful for testing and development.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import contextlib
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from .base import JobStatus, RemoteBackend, register_backend


@register_backend("local")
class LocalBackend(RemoteBackend):
    """Backend for local execution (useful for testing remote workflows).

    This backend simulates the remote execution workflow locally:

    - "Upload" copies files to a temporary directory
    - "Execute" runs the script in a subprocess
    - "Download" copies files back from the temporary directory

    This is useful for:

    - Testing remote execution workflows without a remote system
    - Debugging serialization and script generation
    - Running algorithms in isolated subprocesses

    Config options:
        timeout (int): Execution timeout in seconds (default: 3600).
        keep_workdir (bool): If True, don't delete temp workdir (for debugging).

    Example:
        >>> from qdk_chemistry.algorithms import create
        >>>
        >>> scf = create("scf_solver")
        >>> energy, wfn = scf.run(structure, 0, 1, "cc-pvdz",
        ...                       cache="./cache", remote="local")

    """

    def __init__(self, **config):
        """Initialize the local backend.

        Args:
            **config: Configuration options.

        """
        super().__init__(**config)
        self._workdir: Path | None = None

    def connect(self) -> None:
        """Create a temporary working directory."""
        self._workdir = Path(tempfile.mkdtemp(prefix="qdk_local_"))
        self.remote_workdir = str(self._workdir)

    def disconnect(self) -> None:
        """Clean up the temporary working directory."""
        if self._workdir and self._workdir.exists() and not self.config.get("keep_workdir", False):
            shutil.rmtree(self._workdir, ignore_errors=True)

    def upload(self, local_path: str | Path, remote_path: str) -> None:
        """Copy a file to the 'remote' working directory.

        Args:
            local_path: Path to local file.
            remote_path: Destination path (relative to workdir or absolute).

        """
        local_path = Path(local_path)
        dest_path = Path(remote_path)

        # Ensure parent directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(local_path, dest_path)

    def download(self, remote_path: str, local_path: str | Path) -> None:
        """Copy a file from the 'remote' working directory.

        Args:
            remote_path: Path to file in workdir.
            local_path: Destination path on local system.

        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote_path, local_path)

    # ── Async job primitives ─────────────────────────────────────────────

    def _submit(self, payload: dict) -> tuple[str, dict]:
        """Launch a background subprocess and return ``(job_id, backend_state)``."""
        from qdk_chemistry.remote.serialization import serialize_inputs  # noqa: PLC0415

        job_id = uuid.uuid4().hex[:12]
        job_workdir = Path(self.remote_workdir) / f"job_{job_id}"
        input_dir = job_workdir / "input"
        output_dir = job_workdir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        serialize_inputs(
            input_dir,
            args=payload["args"],
            kwargs=payload["kwargs"],
            algorithm_type=payload["algorithm_type"],
            algorithm_name=payload["algorithm_name"],
            settings=payload["settings"],
            remote_cache=payload.get("remote_cache"),
            remote_cache_backend=payload.get("remote_cache_backend"),
        )

        # Launch via CLI command in background subprocess
        proc = subprocess.Popen(
            ["qdk_chem_cli", "remote-run", "--input-dir", str(input_dir), "--output-dir", str(output_dir)],
            cwd=str(job_workdir),
            stdout=open(job_workdir / "stdout.log", "w"),  # noqa: SIM115
            stderr=open(job_workdir / "stderr.log", "w"),  # noqa: SIM115
        )

        backend_state = {
            "pid": proc.pid,
            "output_dir": str(output_dir),
            "job_workdir": str(job_workdir),
        }
        return job_id, backend_state

    def check(self, backend_state: dict) -> JobStatus:
        """Check whether the background subprocess has finished."""
        import os  # noqa: PLC0415

        pid = backend_state["pid"]
        output_dir = backend_state["output_dir"]
        job_workdir = backend_state.get("job_workdir", str(Path(output_dir).parent))

        # Reap zombie processes so os.kill correctly reports them as gone
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)

        # Check if process is still running
        try:
            os.kill(pid, 0)  # signal 0 — just check existence
            status = "running"
        except ProcessLookupError:
            manifest = Path(output_dir) / "manifest.json"
            status = "Succeeded" if manifest.exists() else "Failed"
        except PermissionError:
            status = "running"

        # Read logs if available
        logs = ""
        stderr_log = Path(job_workdir) / "stderr.log"
        if stderr_log.exists():
            logs = stderr_log.read_text()[-2000:]

        return JobStatus(
            job_id="",
            status=status,
            logs=logs,
            metadata={"pid": pid},
        )

    def cancel(self, backend_state: dict) -> None:
        """Kill the background subprocess."""
        import contextlib  # noqa: PLC0415
        import os  # noqa: PLC0415
        import signal  # noqa: PLC0415

        with contextlib.suppress(ProcessLookupError):
            os.kill(backend_state["pid"], signal.SIGTERM)

    def fetch(self, backend_state: dict, local_dir: str | Path | None = None) -> dict:
        """Deserialize results from a completed local job."""
        from qdk_chemistry.remote.serialization import (  # noqa: PLC0415
            deserialize_outputs,
        )

        output_path = Path(backend_state["output_dir"])

        if not (output_path / "manifest.json").exists():
            raise RuntimeError(f"Job output not found at {output_path}. Job may not have completed successfully.")

        if local_dir is not None:
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            for f in output_path.iterdir():
                shutil.copy2(f, local_dir / f.name)
            return deserialize_outputs(local_dir)

        return deserialize_outputs(output_path)
