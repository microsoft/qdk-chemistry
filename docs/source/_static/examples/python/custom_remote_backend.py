"""Example SSH remote backend plugin for QDK/Chemistry.

This example targets a directly SSH-accessible machine with ``ssh``, ``scp``,
``python3``, and QDK/Chemistry available in its default environment. It does
not submit work through a queue scheduler such as SLURM or PBS.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# docs-example: requires-external-service

################################################################################
# start-cell-custom-remote-backend
from __future__ import annotations

import shlex
import subprocess
import uuid
from pathlib import Path, PurePosixPath

from qdk_chemistry.remote.backends import (
    DEFAULT_POLL_INTERVAL,
    DEFAULT_TIMEOUT,
    JobState,
    JobStatus,
    RemoteBackend,
)


def _parse_remote_pid(output: str) -> str | None:
    """Return a normalized positive PID from remote command output.

    Args:
        output: Standard output containing the remote process identifier.

    """
    pid = output.strip()
    if not pid.isascii() or not pid.isdecimal():
        return None
    return pid.lstrip("0") or None


class SSHBackend(RemoteBackend):
    """Run QDK/Chemistry jobs on a directly SSH-accessible machine.

    Only ``connect``, ``disconnect``, ``upload``, and ``download`` are abstract
    in ``RemoteBackend``. The asynchronous job hooks ``_submit``, ``check``,
    ``cancel``, and ``fetch`` have default implementations that raise
    ``NotImplementedError``, so a backend can leave unsupported hooks unchanged.
    This example overrides all five, including ``cleanup_job``, to support the
    complete asynchronous job lifecycle.

    Job directories under ``remote_workdir`` are retained after they reach a
    terminal state. Job cleanup removes only these remote directories; callers
    remain responsible for local job records and result directories.
    """

    # MCP clients may tune polling behavior but cannot select the host,
    # credentials, remote workspace, SSH arguments, or worker executable.
    mcp_safe_config_options = frozenset({"poll_interval", "timeout"})

    def __init__(
        self,
        *,
        host: str,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        timeout: float = DEFAULT_TIMEOUT,
        remote_workdir: str = "/tmp/qdk_remote",
        identity_file: str | Path | None = None,
        ssh_options: list[str] | None = None,
        python_path: str = "python3",
    ) -> None:
        """Initialize the backend with a required SSH host.

        Args:
            host: SSH destination, such as ``"user@hostname"``.
            poll_interval: Seconds between job status checks.
            timeout: Maximum duration for one SSH or SCP operation.
            remote_workdir: Remote directory containing job artifacts.
            identity_file: Optional private-key path passed to SSH and SCP.
            ssh_options: Additional SSH and SCP command-line options.
            python_path: Remote Python executable used to start the worker.

        """
        if not host:
            raise ValueError("SSHBackend requires a host (e.g., 'user@hostname')")
        super().__init__(
            host=host,
            poll_interval=poll_interval,
            timeout=timeout,
            remote_workdir=remote_workdir,
            identity_file=str(identity_file) if identity_file else None,
            ssh_options=list(ssh_options or []),
            python_path=python_path,
        )
        self.host: str = host
        self.timeout = timeout
        self.remote_workdir = remote_workdir
        self.identity_file = identity_file
        self.ssh_options = list(ssh_options or [])
        self.python_path = python_path

    def connect(self) -> None:
        """Implement the abstract connection hook by testing SSH and creating a workdir."""
        result = subprocess.run(
            self._ssh_cmd(["echo", "connected"]),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise ConnectionError(f"SSH connection failed: {result.stderr}")

        subprocess.run(
            self._ssh_cmd(["mkdir", "-p", self.remote_workdir]),
            check=True,
            timeout=30,
        )

    def disconnect(self) -> None:
        """Implement the abstract disconnection hook as a no-op for one-shot commands."""

    def upload(self, local_path: str | Path, remote_path: str) -> None:
        """Implement the abstract upload hook with SCP.

        Args:
            local_path: Source file on the local machine.
            remote_path: Destination file path on the remote machine.

        """
        local_path = Path(local_path)
        command = [
            "scp",
            *self._ssh_options(),
            str(local_path),
            f"{self._ssh_target()}:{remote_path}",
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"SCP upload failed: {result.stderr}")

    def download(self, remote_path: str, local_path: str | Path) -> None:
        """Implement the abstract download hook with SCP.

        Args:
            remote_path: Source file path on the remote machine.
            local_path: Destination file on the local machine.

        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            "scp",
            *self._ssh_options(),
            f"{self._ssh_target()}:{remote_path}",
            str(local_path),
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"SCP download failed: {result.stderr}")

    def _ssh_target(self) -> str:
        """Return the configured SSH target."""
        return self.host

    def _ssh_options(self) -> list[str]:
        """Build the common SSH and SCP options."""
        options = []
        if self.identity_file is not None:
            options.extend(["-i", str(Path(self.identity_file).expanduser())])
        options.extend(self.ssh_options)
        return options

    def _ssh_cmd(self, remote_command: list[str]) -> list[str]:
        """Build an SSH command.

        Args:
            remote_command: Command and arguments to execute on the remote machine.

        """
        command = ["ssh", *self._ssh_options(), self._ssh_target()]
        command.append(shlex.join(remote_command))
        return command

    def _run_remote(
        self, command: str, *, timeout: int | None = None
    ) -> subprocess.CompletedProcess:
        """Run a shell command on the remote machine.

        Args:
            command: Shell command to execute remotely.
            timeout: Optional command timeout in seconds.

        """
        ssh_command = ["ssh", *self._ssh_options(), self._ssh_target(), command]
        return subprocess.run(
            ssh_command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout or self.timeout,
        )

    def _submit(self, payload: dict) -> tuple[str, dict]:
        """Override the optional async submission hook to launch an SSH worker.

        Args:
            payload: Serialized algorithm execution request.

        """
        import shutil
        import tempfile

        from qdk_chemistry.remote.serialization import serialize_inputs

        job_id = uuid.uuid4().hex[:12]
        remote_job_dir = f"{self.remote_workdir}/job_{job_id}"
        remote_input_dir = f"{remote_job_dir}/input"
        remote_output_dir = f"{remote_job_dir}/output"

        self._run_remote(
            f"mkdir -p {shlex.quote(remote_input_dir)} {shlex.quote(remote_output_dir)}",
            timeout=30,
        )

        local_input_dir = Path(tempfile.mkdtemp(prefix="qdk_ssh_input_"))
        try:
            input_files = serialize_inputs(
                local_input_dir,
                args=payload["args"],
                kwargs=payload["kwargs"],
                algorithm_type=payload["algorithm_type"],
                algorithm_name=payload["algorithm_name"],
                settings=payload["settings"],
                run_hash=payload.get("run_hash"),
                input_hashes=payload.get("input_hashes"),
                force_rerun=payload.get("force_rerun", False),
                remote_cache=payload.get("remote_cache"),
                remote_cache_backend=payload.get("remote_cache_backend"),
            )

            for local_file in input_files:
                self.upload(local_file, f"{remote_input_dir}/{local_file.name}")
        finally:
            shutil.rmtree(local_input_dir, ignore_errors=True)

        # A detached worker and PID file are choices made by this direct-SSH
        # transport. A scheduler backend would submit through its scheduler and
        # record the resulting scheduler job ID instead.
        background_command = (
            f"cd {shlex.quote(remote_job_dir)} && "
            f"nohup {shlex.quote(self.python_path)} -m qdk_chemistry.remote.worker "
            f"--input-dir {shlex.quote(remote_input_dir)} --output-dir {shlex.quote(remote_output_dir)} "
            f"> {shlex.quote(f'{remote_job_dir}/stdout.log')} "
            f"2> {shlex.quote(f'{remote_job_dir}/stderr.log')} & "
            f"echo $! > {shlex.quote(f'{remote_job_dir}/pid')}"
        )
        result = self._run_remote(background_command, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to launch remote job: {result.stderr}")

        # Job persists this opaque state in JSON and passes it back to check,
        # cancel, and fetch, so every value must be JSON-serializable.
        backend_state = {
            "job_id": job_id,
            "remote_job_dir": remote_job_dir,
            "remote_output_dir": remote_output_dir,
        }
        return job_id, backend_state

    def check(self, backend_state: dict) -> JobStatus:
        """Override the optional async status hook to inspect the SSH worker.

        ``JobState`` defines the canonical case-insensitive lifecycle states.
        Backend-specific status strings are also supported and remain
        nonterminal until mapped to a terminal state.

        Args:
            backend_state: Persisted state for the submitted remote job.

        """
        remote_job_dir = backend_state["remote_job_dir"]

        pid_path = shlex.quote(f"{remote_job_dir}/pid")
        pid_result = self._run_remote(f"cat {pid_path}", timeout=10)
        if pid_result.returncode != 0:
            return JobStatus(
                job_id=backend_state["job_id"],
                status=JobState.FAILED,
                error="Could not read PID file",
            )

        pid = _parse_remote_pid(pid_result.stdout)
        if pid is None:
            return JobStatus(
                job_id=backend_state["job_id"],
                status=JobState.FAILED,
                error="Invalid PID file",
            )

        # ``kill -0`` is this transport's process-liveness probe. A scheduler
        # backend would query scheduler state using its persisted job ID.
        alive = self._run_remote(
            f"kill -0 {pid} 2>/dev/null && echo alive || echo done", timeout=10
        )
        if "alive" in alive.stdout:
            status = JobState.RUNNING
        else:
            manifest_path = shlex.quote(f"{remote_job_dir}/output/manifest.json")
            manifest_check = self._run_remote(
                f"test -f {manifest_path} && echo ok || echo missing",
                timeout=10,
            )
            status = (
                JobState.SUCCEEDED if "ok" in manifest_check.stdout else JobState.FAILED
            )

        stderr_path = shlex.quote(f"{remote_job_dir}/stderr.log")
        logs_result = self._run_remote(
            f"tail -50 {stderr_path} 2>/dev/null", timeout=10
        )
        logs = logs_result.stdout if logs_result.returncode == 0 else ""

        return JobStatus(
            job_id=backend_state["job_id"],
            status=status,
            logs=logs,
            metadata={"pid": pid, "remote_job_dir": remote_job_dir},
        )

    def cancel(self, backend_state: dict) -> None:
        """Override the optional async cancellation hook by signaling the worker PID.

        Args:
            backend_state: Persisted state for the submitted remote job.

        """
        remote_job_dir = backend_state["remote_job_dir"]
        pid_path = shlex.quote(f"{remote_job_dir}/pid")
        pid_result = self._run_remote(f"cat {pid_path}", timeout=10)
        if pid_result.returncode == 0:
            pid = _parse_remote_pid(pid_result.stdout)
            if pid is not None:
                self._run_remote(f"kill {pid} 2>/dev/null", timeout=10)

    def fetch(
        self,
        backend_state: dict,
        local_dir: str | Path | None = None,
    ) -> dict:
        """Override the optional async fetch hook to download and deserialize results.

        Args:
            backend_state: Persisted state for the completed remote job.
            local_dir: Optional directory for downloaded result files.

        """
        import json
        import shutil
        import tempfile

        from qdk_chemistry.remote.serialization import (
            deserialize_outputs,
            get_serialized_file_names,
        )

        remote_output_dir = backend_state["remote_output_dir"]

        own_temporary_directory = local_dir is None
        if own_temporary_directory:
            resolved_dir = Path(tempfile.mkdtemp(prefix="qdk_ssh_fetch_"))
        else:
            assert local_dir is not None
            resolved_dir = Path(local_dir)
            resolved_dir.mkdir(parents=True, exist_ok=True)
        local_dir = resolved_dir

        try:
            manifest_local = local_dir / "manifest.json"
            self.download(f"{remote_output_dir}/manifest.json", manifest_local)

            with open(manifest_local) as manifest_file:
                manifest = json.load(manifest_file)

            for entry in manifest.get("results", []):
                for filename in get_serialized_file_names(entry):
                    self.download(
                        f"{remote_output_dir}/{filename}", local_dir / filename
                    )

            return deserialize_outputs(local_dir)
        finally:
            if own_temporary_directory:
                shutil.rmtree(local_dir, ignore_errors=True)

    def cleanup_job(self, backend_state: dict) -> None:
        """Remove artifacts owned by one completed SSH job.

        Args:
            backend_state: Persisted state for the terminal remote job.

        """
        remote_workdir = PurePosixPath(self.remote_workdir)
        remote_job_dir = PurePosixPath(backend_state["remote_job_dir"])
        remote_output_dir = PurePosixPath(backend_state["remote_output_dir"])
        if (
            remote_job_dir.parent != remote_workdir
            or remote_output_dir.parent != remote_job_dir
        ):
            raise ValueError(
                "Remote job paths are inconsistent with the configured work directory"
            )

        result = self._run_remote(
            f"rm -rf -- {shlex.quote(str(remote_job_dir))}", timeout=30
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to clean up remote job: {result.stderr}")


# end-cell-custom-remote-backend
################################################################################

################################################################################
# start-cell-custom-remote-registration
from qdk_chemistry.plugins import PluginRegistrar, QdkChemistryPlugin


class SSHRemoteBackendPlugin(QdkChemistryPlugin):
    """Register the illustrative SSH remote backend."""

    def register(self, registrar: PluginRegistrar) -> None:
        """Register the backend with QDK/Chemistry.

        Args:
            registrar: Plugin registrar receiving the SSH backend.

        """
        registrar.register_remote_backend("ssh", SSHBackend)


# end-cell-custom-remote-registration
################################################################################


################################################################################
# start-cell-custom-remote-usage
def create_ssh_backend():
    """Create the automatically discovered SSH backend."""
    from qdk_chemistry.remote import available_backends, create_remote

    assert "ssh" in available_backends()
    return create_remote("ssh", host="user@compute.example.com")


# end-cell-custom-remote-usage
################################################################################

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    structure: Any
    algorithm: Any
    args: tuple[Any, ...]


################################################################################
# start-cell-custom-remote-run
from qdk_chemistry.algorithms import create
from qdk_chemistry.remote import create_remote

scf = create("scf_solver")
remote = create_remote("ssh", host="user@compute.example.com")
try:
    energy, wavefunction = scf.run(
        structure,
        0,
        1,
        "cc-pvdz",
        remote=remote,
        cache="./cache",
    )
finally:
    remote.disconnect()

# end-cell-custom-remote-run
################################################################################


################################################################################
# start-cell-custom-remote-shared-cache
from qdk_chemistry.remote.cache import FolderCache

cache = FolderCache("/mnt/shared/qdk-cache", is_shared=True)
result = algorithm.run(*args, remote=remote, cache=cache)

# end-cell-custom-remote-shared-cache
################################################################################
