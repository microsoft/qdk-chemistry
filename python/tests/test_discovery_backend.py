"""Tests for the bundled Azure AI Discovery backend."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("azure.ai.discovery", reason="azure-ai-discovery is not installed")

from azure.ai.discovery.models import InputDataMount, OutputDataMount

from qdk_chemistry.plugins.discovery.backend import DiscoveryBackend


class _SharedCache:
    """Minimal shared cache used by transport tests."""

    name = "test-shared"
    is_shared = True

    def __init__(self) -> None:
        """Initialize in-memory data and job stores."""
        self.data: dict[str, Any] = {}
        self.jobs: dict[str, Any] = {}

    def has_data(self, content_hash: str) -> bool:
        """Return whether a content hash is stored."""
        return content_hash in self.data

    def put_data(self, content_hash: str, value: Any) -> None:
        """Store a value by content hash."""
        self.data[content_hash] = value

    def get_data(self, content_hash: str) -> Any:
        """Retrieve a value by content hash."""
        return self.data.get(content_hash)

    def get_job(self, run_hash: str) -> Any:
        """Retrieve a job by run hash."""
        return self.jobs.get(run_hash)


class _Tools:
    """Capture one Discovery tool submission."""

    def __init__(self) -> None:
        """Initialize without a captured submission."""
        self.run: dict[str, Any] | None = None

    def begin_run(self, **kwargs: Any) -> None:
        """Capture submission arguments and assign an operation ID."""
        self.run = kwargs
        kwargs["polling"].operation_id = "operation-1"


class _Container:
    """Capture Blob Storage uploads."""

    def __init__(self) -> None:
        """Initialize an empty upload list."""
        self.uploads: list[str] = []

    def upload_blob(self, *, name: str, data: Any, overwrite: bool) -> None:
        """Record one non-empty overwrite upload."""
        assert data.read()
        assert overwrite
        self.uploads.append(name)


def _backend(**kwargs: Any) -> DiscoveryBackend:
    """Create a backend with required Discovery identifiers."""
    config = {
        "workspace_endpoint": "https://workspace.discovery.azure.com",
        "project_name": "project",
        "tool_id": "tool",
        "node_pool_id": "pool",
        **kwargs,
    }
    return DiscoveryBackend(**config)


def _payload() -> dict[str, Any]:
    """Create a deterministic remote execution payload."""
    return {
        "algorithm_type": "test_algorithm",
        "algorithm_name": "test",
        "settings": {},
        "args": ([1, 2],),
        "kwargs": {},
        "run_hash": "run-hash",
        "input_hashes": {"arg_0": "input-hash"},
    }


def test_constructor_arguments_override_qdk_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit constructor arguments override environment defaults."""
    monkeypatch.setenv("QDK_DISCOVERY_PROJECT_NAME", "environment-project")
    monkeypatch.setenv("QDK_DISCOVERY_TRANSPORT", "blob")
    monkeypatch.setenv("QDK_DISCOVERY_CPUS", "8")

    backend = _backend(project_name="argument-project", transport="cache", cpus=4)

    assert backend.project_name == "argument-project"
    assert backend.transport == "cache"
    assert backend.cpus == 4

    restored = DiscoveryBackend(**backend.config)
    assert restored.project_name == "argument-project"
    assert restored.transport == "cache"
    assert restored.remote_workdir == "qdk_chemistry"


def test_cache_transport_seeds_inputs_and_submits_inline_manifest() -> None:
    """Auto mode prefers a shared cache and sends only an inline manifest."""
    cache = _SharedCache()
    container = _Container()
    payload = _payload()
    payload["remote_cache"] = {"name": cache.name}
    payload["remote_cache_backend"] = cache
    tools = _Tools()
    backend = _backend(
        storage_uri="discovery://storageassets/subscriptions/example/storageAssets/data",
        storage_account_url="https://storage.blob.core.windows.net",
        storage_container="container",
    )
    backend._client = SimpleNamespace(tools=tools)
    backend._container_client = container

    _, state = backend._submit(payload)

    assert cache.data["input-hash"] == [1, 2]
    assert container.uploads == []
    assert state["transport"] == "cache"
    assert "input_dir" not in state
    assert tools.run is not None
    assert tools.run["input_data"] == []
    assert tools.run["output_data"] == []
    assert len(tools.run["inline_files"]) == 1
    inline_file = tools.run["inline_files"][0]
    manifest = json.loads(gzip.decompress(base64.b64decode(inline_file.encoded_file)))
    assert inline_file.mount_path == "/qdk/input/manifest.json"
    assert manifest["remote_cache_transport"] is True
    assert manifest["args"][0] == {
        "type": "cached",
        "dataclass_type": "list",
        "content_hash": "input-hash",
    }


def test_blob_transport_submits_discovery_storage_mounts() -> None:
    """Blob mode requests input and output mounts from Discovery."""
    tools = _Tools()
    container = _Container()
    backend = _backend(
        transport="blob",
        storage_uri="discovery://storageassets/subscriptions/example/storageAssets/data",
        storage_account_url="https://storage.blob.core.windows.net",
        storage_container="container",
    )
    backend._client = SimpleNamespace(tools=tools)
    backend._container_client = container

    _, state = backend._submit(_payload())

    assert state["transport"] == "blob"
    assert container.uploads == [f"{state['input_dir']}/manifest.json"]
    assert tools.run is not None
    assert tools.run["inline_files"] == []
    assert len(tools.run["input_data"]) == 1
    assert len(tools.run["output_data"]) == 1
    assert isinstance(tools.run["input_data"][0], InputDataMount)
    assert isinstance(tools.run["output_data"][0], OutputDataMount)
    assert tools.run["input_data"][0].mount_path == "/qdk/input"
    assert tools.run["output_data"][0].mount_path == "/qdk/output"


def test_cache_transport_fetches_result_without_blob(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache transport reconstructs results without Blob Storage."""
    cache = _SharedCache()
    cache.data["output-hash"] = [1, 2]
    cache.jobs["run-hash"] = SimpleNamespace(
        output_hashes=[{"value": -1.5}, {"hash": "output-hash"}],
    )
    monkeypatch.setattr("qdk_chemistry.remote.cache.get_cache", lambda _name, **_config: cache)
    backend = _backend()

    result = backend.fetch(
        {
            "transport": "cache",
            "remote_cache": {"name": cache.name},
            "run_hash": "run-hash",
        }
    )

    assert result == (-1.5, [1, 2])


def test_fetch_rejects_output_file_outside_destination(tmp_path) -> None:
    """Artifact names cannot cause downloads outside the requested destination."""
    backend = _backend()
    downloads: list[tuple[str, Any]] = []

    def download(remote_path: str, local_path: Any) -> None:
        downloads.append((remote_path, local_path))
        Path(local_path).write_text(json.dumps({"results": [{"file": "/outside"}]}))

    backend.download = download

    with pytest.raises(ValueError, match="outside the serialization directory"):
        backend.fetch({"output_dir": "qdk_chemistry/job/output"}, tmp_path / "outputs")

    assert downloads == [("qdk_chemistry/job/output/manifest.json", tmp_path / "outputs" / "manifest.json")]


def test_auto_transport_requires_cache_or_blob() -> None:
    """Auto mode rejects submissions with no usable artifact transport."""
    backend = _backend()
    backend._client = SimpleNamespace(tools=_Tools())

    with pytest.raises(ValueError, match="shared cache or complete Blob Storage"):
        backend._submit(_payload())
