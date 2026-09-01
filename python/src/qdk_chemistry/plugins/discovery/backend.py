"""Discovery SDK backend for QDK/Chemistry remote execution."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import base64
import gzip
import json
import os
import shlex
import shutil
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from qdk_chemistry.plugins.discovery._sdk import (
    _OperationIdPolling,
    create_credential,
    create_workspace_client,
    response_mapping,
)
from qdk_chemistry.remote.backends.base import JobStatus, RemoteBackend
from qdk_chemistry.remote.serialization import (
    _manifest_file_path,
    deserialize_outputs,
    get_serialized_file_names,
    serialize_inputs,
)


def _discovery_env_defaults() -> dict[str, Any]:
    """Read Discovery backend defaults from environment variables."""
    defaults: dict[str, Any] = {}
    mappings = {
        "QDK_DISCOVERY_IMAGE": "image",
        "QDK_DISCOVERY_WORKSPACE_ENDPOINT": "workspace_endpoint",
        "QDK_DISCOVERY_PROJECT_NAME": "project_name",
        "QDK_DISCOVERY_TOOL_ID": "tool_id",
        "QDK_DISCOVERY_NODE_POOL_ID": "node_pool_id",
        "QDK_DISCOVERY_CPUS": "cpus",
        "QDK_DISCOVERY_GPUS": "gpus",
        "QDK_DISCOVERY_MEMORY": "memory",
        "QDK_DISCOVERY_AUTH_MODE": "auth_mode",
        "QDK_DISCOVERY_PYTHON_PATH": "python_path",
        "QDK_DISCOVERY_TRANSPORT": "transport",
        "QDK_DISCOVERY_STORAGE_URI": "storage_uri",
        "QDK_DISCOVERY_STORAGE_ACCOUNT_URL": "storage_account_url",
        "QDK_DISCOVERY_STORAGE_CONTAINER": "storage_container",
        "QDK_DISCOVERY_STORAGE_BLOB_PREFIX": "storage_blob_prefix",
        "QDK_DISCOVERY_STORAGE_PREFIX": "storage_prefix",
        "QDK_DISCOVERY_ARTIFACT_RETRY_ATTEMPTS": "artifact_retry_attempts",
        "QDK_DISCOVERY_ARTIFACT_RETRY_DELAY": "artifact_retry_delay",
        "QDK_DISCOVERY_POLL_INTERVAL": "poll_interval",
        "QDK_DISCOVERY_TIMEOUT": "timeout",
    }
    for env_name, config_name in mappings.items():
        if value := os.environ.get(env_name):
            defaults[config_name] = value
    return defaults


class DiscoveryBackend(RemoteBackend):
    """Run QDK/Chemistry jobs through the Azure AI Discovery SDK."""

    name = "discovery"
    mcp_safe_config_options = frozenset({"artifact_retry_attempts", "artifact_retry_delay", "poll_interval", "timeout"})

    def __init__(
        self,
        *,
        workspace_endpoint: str | None = None,
        project_name: str | None = None,
        tool_id: str | None = None,
        node_pool_id: str | None = None,
        transport: str | None = None,
        storage_uri: str | None = None,
        storage_account_url: str | None = None,
        storage_container: str | None = None,
        storage_blob_prefix: str | None = None,
        storage_prefix: str | None = None,
        auth_mode: str | None = None,
        image: str | None = None,
        python_path: str | None = None,
        cpus: int | str | None = None,
        gpus: int | str | None = None,
        memory: str | None = None,
        artifact_retry_attempts: int | None = None,
        artifact_retry_delay: float | None = None,
        poll_interval: float | None = None,
        timeout: float | None = None,
    ):
        """Initialize the Discovery backend."""
        env = _discovery_env_defaults()

        def resolve(name: str, explicit: Any, default: Any = None) -> Any:
            return explicit if explicit is not None else env.get(name, default)

        self.workspace_endpoint = resolve("workspace_endpoint", workspace_endpoint)
        self.project_name = resolve("project_name", project_name)
        self.tool_id = resolve("tool_id", tool_id)
        self.node_pool_id = resolve("node_pool_id", node_pool_id)
        self.transport = str(resolve("transport", transport, "auto")).casefold()
        if self.transport not in {"auto", "blob", "cache"}:
            raise ValueError("transport must be one of: auto, blob, cache")
        self.storage_uri = str(resolve("storage_uri", storage_uri, "")).rstrip("/")
        self.storage_account_url = resolve("storage_account_url", storage_account_url)
        self.storage_container = resolve("storage_container", storage_container)
        self.storage_blob_prefix = str(resolve("storage_blob_prefix", storage_blob_prefix, "")).strip("/")
        self.storage_prefix = str(resolve("storage_prefix", storage_prefix, "qdk_chemistry")).strip("/")
        if not self.storage_prefix:
            raise ValueError("storage_prefix must be a non-empty relative path")

        self.auth_mode = str(resolve("auth_mode", auth_mode, "azure-cli"))
        self.image = resolve("image", image)
        self.python_path = str(resolve("python_path", python_path, "python"))
        self.artifact_retry_attempts = int(resolve("artifact_retry_attempts", artifact_retry_attempts, 5))
        self.artifact_retry_delay = float(resolve("artifact_retry_delay", artifact_retry_delay, 1.0))
        if self.artifact_retry_attempts < 1:
            raise ValueError("artifact_retry_attempts must be at least 1")
        if self.artifact_retry_delay < 0:
            raise ValueError("artifact_retry_delay cannot be negative")
        self.remote_workdir = self.storage_prefix

        self.cpus = resolve("cpus", cpus)
        self.gpus = resolve("gpus", gpus)
        self.memory = resolve("memory", memory)

        resolved_poll_interval = float(resolve("poll_interval", poll_interval, 5.0))
        resolved_timeout = float(resolve("timeout", timeout, 3600.0))
        super().__init__(
            poll_interval=resolved_poll_interval,
            timeout=resolved_timeout,
            workspace_endpoint=self.workspace_endpoint,
            project_name=self.project_name,
            tool_id=self.tool_id,
            node_pool_id=self.node_pool_id,
            transport=self.transport,
            storage_uri=self.storage_uri or None,
            storage_account_url=self.storage_account_url,
            storage_container=self.storage_container,
            storage_blob_prefix=self.storage_blob_prefix or None,
            storage_prefix=self.storage_prefix,
            auth_mode=self.auth_mode,
            image=self.image,
            python_path=self.python_path,
            cpus=self.cpus,
            gpus=self.gpus,
            memory=self.memory,
            artifact_retry_attempts=self.artifact_retry_attempts,
            artifact_retry_delay=self.artifact_retry_delay,
        )
        self.remote_workdir = self.storage_prefix

        self._credential: Any = None
        self._client: Any = None
        self._container_client: Any = None

    def _validate_config(self) -> None:
        """Validate settings required for every Discovery tool run."""
        required = {
            "workspace_endpoint": self.workspace_endpoint,
            "project_name": self.project_name,
            "tool_id": self.tool_id,
            "node_pool_id": self.node_pool_id,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Discovery backend requires: {', '.join(missing)}")

    def _blob_configured(self, *, required: bool = False) -> bool:
        """Validate and report whether Blob Storage transport is configured."""
        values = {
            "storage_uri": self.storage_uri,
            "storage_account_url": self.storage_account_url,
            "storage_container": self.storage_container,
        }
        configured = [name for name, value in values.items() if value]
        if not configured:
            if required:
                raise ValueError(f"Discovery blob transport requires: {', '.join(values)}")
            return False
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ValueError(f"Incomplete Discovery blob configuration; missing: {', '.join(missing)}")
        if not self.storage_uri.startswith("discovery://storageassets/"):
            raise ValueError("storage_uri must be a discovery://storageassets/<storage-asset-ARM-ID> URI")
        if "/paths/" in self.storage_uri:
            raise ValueError("storage_uri must identify the Storage Asset root, not a path within it")
        return True

    def connect(self) -> None:
        """Create authenticated Discovery and optional Blob Storage clients."""
        self._validate_config()
        blob_configured = False
        if self.transport != "cache":
            blob_configured = self._blob_configured(required=self.transport == "blob")
        self._credential = create_credential(self.auth_mode)
        self._client = create_workspace_client(str(self.workspace_endpoint), self._credential)
        if blob_configured:
            from azure.storage.blob import BlobServiceClient  # noqa: PLC0415

            blob_service = BlobServiceClient(account_url=str(self.storage_account_url), credential=self._credential)
            self._container_client = blob_service.get_container_client(str(self.storage_container))

    def disconnect(self) -> None:
        """Close SDK clients."""
        for client in (self._container_client, self._client, self._credential):
            close = getattr(client, "close", None)
            if callable(close):
                close()
        self._container_client = None
        self._client = None
        self._credential = None

    def _require_connection(self, *, storage: bool = False) -> None:
        """Require active Discovery and, when requested, storage clients."""
        if self._client is None:
            raise RuntimeError("Discovery backend is not connected")
        if storage and self._container_client is None:
            raise RuntimeError("Discovery Blob Storage transport is not connected")

    def _relative_path(self, remote_path: str) -> str:
        """Validate and normalize one relative storage path."""
        path = str(PurePosixPath(remote_path))
        if path.startswith("/") or path == "." or ".." in PurePosixPath(path).parts:
            raise ValueError(f"remote storage path must be relative and cannot traverse parents: {remote_path!r}")
        return path

    def _storage_path_uri(self, remote_path: str) -> str:
        """Build a Discovery Storage Asset path URI."""
        return f"{self.storage_uri}/paths/{self._relative_path(remote_path)}"

    def _blob_path(self, remote_path: str) -> str:
        """Map a storage-relative path to its backing blob path."""
        relative_path = self._relative_path(remote_path)
        if not self.storage_blob_prefix:
            return relative_path
        return f"{self.storage_blob_prefix}/{relative_path}"

    def upload(self, local_path: str | Path, remote_path: str) -> None:
        """Upload a file to the linked Azure Blob Storage container."""
        self._require_connection(storage=True)
        local_path = Path(local_path)
        with local_path.open("rb") as data:
            self._container_client.upload_blob(name=self._blob_path(remote_path), data=data, overwrite=True)

    def download(self, remote_path: str, local_path: str | Path) -> None:
        """Download a file from the linked Azure Blob Storage container."""
        from azure.core.exceptions import ResourceNotFoundError  # noqa: PLC0415

        self._require_connection(storage=True)
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob_path = self._blob_path(remote_path)
        for attempt in range(self.artifact_retry_attempts):
            temporary_path: Path | None = None
            try:
                downloader = self._container_client.download_blob(blob_path)
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=local_path.parent,
                    prefix=f".{local_path.name}.",
                    delete=False,
                ) as data:
                    temporary_path = Path(data.name)
                    downloader.readinto(data)
                temporary_path.replace(local_path)
                return
            except ResourceNotFoundError:
                if attempt == self.artifact_retry_attempts - 1:
                    raise
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            time.sleep(self.artifact_retry_delay * (2**attempt))

    def _delete_remote_files(self, remote_paths: list[str]) -> None:
        """Delete remote job artifacts, ignoring paths already removed."""
        from azure.core.exceptions import ResourceNotFoundError  # noqa: PLC0415

        for remote_path in remote_paths:
            try:
                self._container_client.delete_blob(self._blob_path(remote_path))
            except ResourceNotFoundError:
                continue

    def cleanup_job(self, backend_state: dict[str, Any]) -> None:
        """Remove Blob artifacts for one completed Discovery job."""
        if backend_state.get("transport") == "cache":
            return
        self._require_connection(storage=True)
        input_paths = backend_state.get("input_paths", [])
        output_dir = backend_state.get("output_dir")
        if not isinstance(input_paths, list) or not isinstance(output_dir, str):
            raise ValueError("Discovery job state is missing artifact paths")
        output_prefix = f"{self._blob_path(output_dir).rstrip('/')}/"
        output_paths = [blob.name for blob in self._container_client.list_blobs(name_starts_with=output_prefix)]
        self._delete_remote_files(input_paths)
        for blob_path in output_paths:
            self._container_client.delete_blob(blob_path)

    @staticmethod
    def _shared_cache(payload: dict[str, Any]) -> Any:
        """Resolve the shared cache supplied for remote execution."""
        cache = payload.get("remote_cache_backend")
        cache_info = payload.get("remote_cache")
        if cache is None and cache_info and cache_info.get("name"):
            from qdk_chemistry.remote.cache import get_cache  # noqa: PLC0415

            cache_config = {key: value for key, value in cache_info.items() if key != "name"}
            cache = get_cache(cache_info["name"], **cache_config)
        if cache is None or not getattr(cache, "is_shared", False):
            raise ValueError("Discovery cache transport requires a shared cache reachable from the compute node")
        return cache

    def _serialize_job(
        self,
        payload: dict[str, Any],
        *,
        cache: Any = None,
        cache_transport: bool = False,
    ) -> tuple[str, Path, list[Path]]:
        """Serialize one job into a temporary input directory."""
        job_id = uuid.uuid4().hex[:12]
        local_input_dir = Path(tempfile.mkdtemp(prefix="qdk_input_"))
        try:
            input_files = serialize_inputs(
                local_input_dir,
                args=payload["args"],
                kwargs=payload["kwargs"],
                algorithm_type=payload["algorithm_type"],
                algorithm_name=payload["algorithm_name"],
                settings=payload["settings"],
                run_hash=payload.get("run_hash"),
                job_cache_key=payload.get("job_cache_key"),
                input_hashes=payload.get("input_hashes"),
                remote_cache=payload.get("remote_cache"),
                remote_cache_backend=cache,
                remote_cache_transport=cache_transport,
            )
        except BaseException:
            shutil.rmtree(local_input_dir, ignore_errors=True)
            raise
        return job_id, local_input_dir, input_files

    def _prepare_blob_job(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Serialize and upload inputs for a Blob Storage job."""
        job_id, local_input_dir, input_files = self._serialize_job(
            payload,
            cache=payload.get("remote_cache_backend"),
        )
        input_dir = f"{self.remote_workdir}/job_{job_id}/input"
        output_dir = f"{self.remote_workdir}/job_{job_id}/output"
        input_paths: list[str] = []
        try:
            for local_file in input_files:
                remote_path = f"{input_dir}/{local_file.name}"
                self.upload(local_file, remote_path)
                input_paths.append(remote_path)
        except BaseException:
            self._delete_remote_files(input_paths)
            raise
        finally:
            shutil.rmtree(local_input_dir, ignore_errors=True)

        return job_id, {
            "job_id": job_id,
            "input_dir": input_dir,
            "input_paths": input_paths,
            "output_dir": output_dir,
            "transport": "blob",
        }

    def _prepare_cache_job(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Seed a shared cache and prepare an inline input manifest."""
        if not payload.get("run_hash"):
            raise ValueError("Discovery cache transport requires a deterministic run hash")
        if not payload.get("remote_cache"):
            raise ValueError("Discovery cache transport requires serializable shared-cache configuration")
        cache = self._shared_cache(payload)
        job_id, local_input_dir, input_files = self._serialize_job(
            payload,
            cache=cache,
            cache_transport=True,
        )
        try:
            non_manifest_files = [path.name for path in input_files if path.name != "manifest.json"]
            if non_manifest_files:
                raise ValueError(
                    "Discovery cache transport could not place every file-backed input in the shared cache: "
                    + ", ".join(non_manifest_files)
                )
            manifest = (local_input_dir / "manifest.json").read_bytes()
        finally:
            shutil.rmtree(local_input_dir, ignore_errors=True)
        return job_id, {
            "job_id": job_id,
            "inline_manifest": base64.b64encode(gzip.compress(manifest)).decode("ascii"),
            "remote_cache": payload["remote_cache"],
            "run_hash": payload["run_hash"],
            "job_cache_key": payload["job_cache_key"],
            "transport": "cache",
        }

    def _select_transport(self, payload: dict[str, Any]) -> str:
        """Choose cache transport when available, otherwise Blob Storage."""
        if self.transport == "cache":
            self._shared_cache(payload)
            return "cache"
        if self.transport == "blob":
            self._require_connection(storage=True)
            return "blob"
        cache = payload.get("remote_cache_backend")
        if cache is not None and getattr(cache, "is_shared", False) and payload.get("remote_cache"):
            return "cache"
        if self._container_client is not None:
            return "blob"
        raise ValueError("Discovery requires either a shared cache or complete Blob Storage configuration")

    def _submit(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Submit a QDK/Chemistry worker through Azure AI Discovery."""
        from azure.ai.discovery.models import (  # noqa: PLC0415
            InfraOverrides,
            InlineFile,
            InputDataMount,
            OutputDataMount,
        )

        self._require_connection()
        transport = self._select_transport(payload)
        if transport == "cache":
            job_id, state = self._prepare_cache_job(payload)
            input_dir = "/qdk/input"
            output_dir = f"/tmp/qdk_chemistry_{job_id}/output"
            inline_files = [
                InlineFile(
                    mount_path=f"{input_dir}/manifest.json",
                    encoded_file=state["inline_manifest"],
                )
            ]
            input_data: list[Any] = []
            output_data: list[Any] = []
        else:
            job_id, state = self._prepare_blob_job(payload)
            input_dir = "/qdk/input"
            output_dir = "/qdk/output"
            inline_files = []
            input_data = [
                InputDataMount(
                    storage_uri=self._storage_path_uri(state["input_dir"]),
                    mount_path=input_dir,
                )
            ]
            output_data = [
                OutputDataMount(
                    storage_uri=self._storage_path_uri(state["output_dir"]),
                    mount_path=output_dir,
                )
            ]
        command = shlex.join(
            [
                self.python_path,
                "-m",
                "qdk_chemistry.remote.worker",
                "--input-dir",
                input_dir,
                "--output-dir",
                output_dir,
            ]
        )
        polling = _OperationIdPolling()
        try:
            self._client.tools.begin_run(
                project_name=str(self.project_name),
                tool_id=str(self.tool_id),
                node_pool_ids=[str(self.node_pool_id)],
                command=command,
                inline_files=inline_files,
                input_data=input_data,
                output_data=output_data,
                infra_overrides=InfraOverrides(
                    image_uri=str(self.image) if self.image else None,
                    cpu=str(self.cpus) if self.cpus is not None else None,
                    gpu=str(self.gpus) if self.gpus is not None else None,
                    ram=str(self.memory) if self.memory is not None else None,
                    replica_count=1,
                ),
                polling=polling,
            )
            if polling.operation_id is None:
                raise RuntimeError("Discovery run submission did not expose an operation ID")
        except BaseException:
            if transport == "blob":
                self._delete_remote_files(state["input_paths"])
            raise
        state["operation_id"] = polling.operation_id
        state["project_name"] = self.project_name
        state.pop("inline_manifest", None)
        return job_id, state

    def _status_state(self, backend_state: dict[str, Any]) -> dict[str, Any]:
        """Convert one Azure AI Discovery run status to backend state."""
        response = self._client.tools.get_run_status(
            backend_state["project_name"],
            backend_state["operation_id"],
            log_count=100,
        )
        data = response_mapping(response)
        result = response_mapping(data.get("result"))
        report = response_mapping(result.get("toolReport") or result.get("tool_report"))
        status = str(data.get("status", "Unknown"))
        if status.casefold() in {"canceled", "cancelled"}:
            status = "canceled"
        return {
            "status": status,
            "logs": report.get("logs") or "",
            "error": data.get("error"),
            "created_by": result.get("createdBy") or result.get("created_by"),
        }

    def check(self, backend_state: dict[str, Any]) -> JobStatus:
        """Query the current status of a Discovery job."""
        self._require_connection()
        state = self._status_state(backend_state)
        error = state["error"]
        if isinstance(error, dict):
            error = error.get("message") or json.dumps(error)
        return JobStatus(
            job_id=backend_state.get("job_id", ""),
            status=state["status"],
            logs=str(state["logs"]),
            error=str(error) if error else None,
            metadata={
                "operation_id": backend_state["operation_id"],
                "created_by": state["created_by"],
            },
        )

    def cancel(self, backend_state: dict[str, Any]) -> None:
        """Cancel a Discovery job."""
        self._require_connection()
        self._client.tools.begin_cancel_run_lro(
            backend_state["project_name"],
            backend_state["operation_id"],
            polling=False,
        )

    def fetch(self, backend_state: dict[str, Any], local_dir: str | Path | None = None) -> Any:
        """Download and deserialize completed Discovery output."""
        if backend_state.get("transport") == "cache":
            return self._fetch_cached_result(backend_state)
        output_dir = backend_state["output_dir"]
        own_tmp = local_dir is None
        resolved_dir = Path(tempfile.mkdtemp(prefix="qdk_fetch_") if local_dir is None else local_dir)
        resolved_dir.mkdir(parents=True, exist_ok=True)
        try:
            manifest_path = f"{output_dir}/manifest.json"
            manifest_local = resolved_dir / "manifest.json"
            self.download(manifest_path, manifest_local)
            manifest = json.loads(manifest_local.read_text(encoding="utf-8"))
            for entry in manifest.get("results", []):
                for filename in get_serialized_file_names(entry):
                    remote_path = f"{output_dir}/{filename}"
                    self.download(remote_path, _manifest_file_path(resolved_dir, filename))
            return deserialize_outputs(resolved_dir)
        finally:
            if own_tmp:
                shutil.rmtree(resolved_dir, ignore_errors=True)

    @classmethod
    def _fetch_cached_result(cls, backend_state: dict[str, Any]) -> Any:
        """Reconstruct a completed result from the configured shared cache."""
        cache = cls._shared_cache({"remote_cache": backend_state["remote_cache"]})
        run_hash = backend_state["run_hash"]
        job_cache_key = backend_state["job_cache_key"]
        job = cache.get_job(job_cache_key)
        if job is None or job.output_hashes is None or job.output_is_tuple is None:
            raise LookupError(f"Shared cache has no completed result for run {job_cache_key}")
        results: list[Any] = []
        for entry in job.output_hashes:
            if "value" in entry:
                results.append(entry["value"])
                continue
            value = cache.get_data(entry["hash"])
            if value is None:
                raise LookupError(f"Shared cache is missing output {entry['hash']} for run {run_hash}")
            results.append(value)
        if job.output_is_tuple:
            return tuple(results)
        if len(results) == 1:
            return results[0]
        raise LookupError(f"Shared cache has invalid result shape for run {run_hash}")
