"""SSH backend for remote execution of QDK/Chemistry algorithms.

This backend transfers files and executes scripts over SSH, making it
suitable for remote servers, HPC login nodes, and cloud VMs.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

from .base import JobStatus, RemoteBackend, register_backend


@register_backend("ssh")
class SSHBackend(RemoteBackend):
    """Backend for execution over SSH.

    This backend uses standard SSH and SCP commands for file transfer and
    script execution. It's suitable for any system accessible via SSH.

    Configuration:

    host (str)
        SSH target in format ``user@hostname`` or just ``hostname``. Required.
    identity_file (str)
        Path to SSH private key file.
    timeout (int)
        Command timeout in seconds (default: 3600).
    remote_workdir (str)
        Working directory on remote (default: ``/tmp/qdk_remote``).
    ssh_options (list[str])
        Additional SSH options (e.g., ``["-o", "StrictHostKeyChecking=no"]``).

    Example:
        >>> from qdk_chemistry.algorithms import create
        >>> from qdk_chemistry.remote import create_remote
        >>>
        >>> ssh = create_remote("ssh",
        ...     host="user@cluster.example.com",
        ...     identity_file="~/.ssh/cluster_key",
        ...     python_path="/opt/conda/bin/python",
        ... )
        >>> scf = create("scf_solver")
        >>> energy, wfn = scf.run(structure, 0, 1, "cc-pvdz",
        ...                       cache="./cache", remote=ssh)

    """

    def __init__(self, **config):
        """Initialize the SSH backend.

        Args:
            **config: Configuration options. Must include 'host'.

        """
        super().__init__(**config)
        host = config.get("host")
        if not host:
            raise ValueError("SSHBackend requires 'host' config option (e.g., 'user@hostname')")
        self.host: str = host

    def connect(self) -> None:
        """Test SSH connection to the remote host."""
        result = subprocess.run(
            self._ssh_cmd(["echo", "connected"]),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise ConnectionError(f"SSH connection failed: {result.stderr}")

        # Ensure remote working directory exists
        subprocess.run(
            self._ssh_cmd(["mkdir", "-p", self.remote_workdir]),
            check=True,
            timeout=30,
        )

    def disconnect(self) -> None:
        """No persistent connection to close for SSH."""

    def upload(self, local_path: str | Path, remote_path: str) -> None:
        """Upload a file to the remote system via SCP.

        Args:
            local_path: Path to local file.
            remote_path: Destination path on remote system.

        """
        local_path = Path(local_path)
        cmd = ["scp", *self._ssh_options(), str(local_path), f"{self._ssh_target()}:{remote_path}"]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=self.timeout)
        if result.returncode != 0:
            raise RuntimeError(f"SCP upload failed: {result.stderr}")

    def download(self, remote_path: str, local_path: str | Path) -> None:
        """Download a file from the remote system via SCP.

        Args:
            remote_path: Path to file on remote system.
            local_path: Destination path on local system.

        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = ["scp", *self._ssh_options(), f"{self._ssh_target()}:{remote_path}", str(local_path)]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=self.timeout)
        if result.returncode != 0:
            raise RuntimeError(f"SCP download failed: {result.stderr}")

    def _ssh_target(self) -> str:
        """Get the SSH target (user@host or just host)."""
        return self.host

    def _ssh_options(self) -> list[str]:
        """Build SSH/SCP options list."""
        opts = []
        if "identity_file" in self.config:
            opts.extend(["-i", str(Path(self.config["identity_file"]).expanduser())])
        if "ssh_options" in self.config:
            opts.extend(self.config["ssh_options"])
        return opts

    def _ssh_cmd(self, remote_cmd: list[str]) -> list[str]:
        """Build a complete SSH command."""
        cmd = ["ssh", *self._ssh_options(), self._ssh_target()]
        # Join remote command as a single string for proper shell execution
        cmd.append(" ".join(remote_cmd))
        return cmd

    def _run_remote(self, command: str, *, timeout: int | None = None) -> subprocess.CompletedProcess:
        """Execute a shell command on the remote host via SSH."""
        cmd = ["ssh", *self._ssh_options(), self._ssh_target(), command]
        return subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout or self.timeout,
        )

    # ── Async job primitives ─────────────────────────────────────────────

    def _submit(self, payload: dict) -> tuple[str, dict]:
        """Upload, launch via ``nohup``, and return ``(job_id, backend_state)``."""
        import shutil  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        from qdk_chemistry.remote.serialization import serialize_inputs  # noqa: PLC0415

        job_id = uuid.uuid4().hex[:12]
        remote_job_dir = f"{self.remote_workdir}/job_{job_id}"
        remote_input_dir = f"{remote_job_dir}/input"
        remote_output_dir = f"{remote_job_dir}/output"

        # Create remote directories
        self._run_remote(f"mkdir -p {remote_input_dir} {remote_output_dir}", timeout=30)

        # Serialize inputs locally
        local_input_dir = Path(tempfile.mkdtemp(prefix="qdk_ssh_input_"))
        try:
            input_files = serialize_inputs(
                local_input_dir,
                args=payload["args"],
                kwargs=payload["kwargs"],
                algorithm_type=payload["algorithm_type"],
                algorithm_name=payload["algorithm_name"],
                settings=payload["settings"],
                remote_cache=payload.get("remote_cache"),
                remote_cache_backend=payload.get("remote_cache_backend"),
            )

            for local_file in input_files:
                self.upload(local_file, f"{remote_input_dir}/{local_file.name}")
        finally:
            shutil.rmtree(local_input_dir, ignore_errors=True)

        # Launch via CLI command in background with nohup; store PID in pidfile
        bg_cmd = (
            f"cd {remote_job_dir} && "
            f"nohup qdk_chem_cli remote-run --input-dir {remote_input_dir} --output-dir {remote_output_dir} "
            f"> {remote_job_dir}/stdout.log 2> {remote_job_dir}/stderr.log & "
            f"echo $! > {remote_job_dir}/pid"
        )
        result = self._run_remote(bg_cmd, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to launch remote job: {result.stderr}")

        backend_state = {
            "remote_job_dir": remote_job_dir,
            "remote_output_dir": remote_output_dir,
        }
        return job_id, backend_state

    def check(self, backend_state: dict) -> JobStatus:
        """Check whether the remote process is still running."""
        remote_job_dir = backend_state["remote_job_dir"]

        # Read PID
        pid_result = self._run_remote(f"cat {remote_job_dir}/pid", timeout=10)
        if pid_result.returncode != 0:
            return JobStatus(job_id="", status="unknown", error="Could not read PID file")

        pid = pid_result.stdout.strip()

        # Check if process is alive
        alive = self._run_remote(f"kill -0 {pid} 2>/dev/null && echo alive || echo done", timeout=10)
        if "alive" in alive.stdout:
            status = "running"
        else:
            manifest_check = self._run_remote(
                f"test -f {remote_job_dir}/output/manifest.json && echo ok || echo missing",
                timeout=10,
            )
            status = "Succeeded" if "ok" in manifest_check.stdout else "Failed"

        # Read tail of stderr log
        logs_result = self._run_remote(f"tail -50 {remote_job_dir}/stderr.log 2>/dev/null", timeout=10)
        logs = logs_result.stdout if logs_result.returncode == 0 else ""

        return JobStatus(
            job_id="",
            status=status,
            logs=logs,
            metadata={"pid": pid, "remote_job_dir": remote_job_dir},
        )

    def cancel(self, backend_state: dict) -> None:
        """Kill the remote background process."""
        remote_job_dir = backend_state["remote_job_dir"]
        pid_result = self._run_remote(f"cat {remote_job_dir}/pid", timeout=10)
        if pid_result.returncode == 0:
            pid = pid_result.stdout.strip()
            self._run_remote(f"kill {pid} 2>/dev/null", timeout=10)

    def fetch(self, backend_state: dict, local_dir: str | Path | None = None) -> dict:
        """Download and deserialize results from the remote host."""
        import json  # noqa: PLC0415
        import shutil  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        from qdk_chemistry.remote.serialization import (  # noqa: PLC0415
            deserialize_outputs,
        )

        remote_output_dir = backend_state["remote_output_dir"]

        own_tmp = local_dir is None
        if own_tmp:
            resolved_dir = Path(tempfile.mkdtemp(prefix="qdk_ssh_fetch_"))
        else:
            assert local_dir is not None
            resolved_dir = Path(local_dir)
            resolved_dir.mkdir(parents=True, exist_ok=True)
        local_dir = resolved_dir

        try:
            manifest_local = local_dir / "manifest.json"
            self.download(f"{remote_output_dir}/manifest.json", manifest_local)

            with open(manifest_local) as f:
                manifest = json.load(f)

            for entry in manifest.get("results", []):
                if entry.get("file"):
                    filename = entry["file"]
                    self.download(f"{remote_output_dir}/{filename}", local_dir / filename)

            return deserialize_outputs(local_dir)
        finally:
            if own_tmp:
                shutil.rmtree(local_dir, ignore_errors=True)
