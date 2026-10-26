"""Connection authority for restored job handles."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
from unittest.mock import MagicMock

import pytest

import qdk_chemistry.remote.backends as remote_backends
from qdk_chemistry.remote.backends.base import JobStatus
from qdk_chemistry.remote.job import Job


@pytest.mark.parametrize("discover", [False, True])
@pytest.mark.parametrize(
    "saved_config",
    [
        {"workspace_endpoint": "https://untrusted.invalid/workspace"},
        {"storage_account_url": "https://untrusted.invalid/storage"},
    ],
)
def test_restored_job_uses_runtime_configuration(tmp_path, monkeypatch, saved_config, discover):
    """Saved connection parameters cannot select an authenticated destination."""
    backend = MagicMock()
    backend.check.return_value = JobStatus(job_id="j1", status="running")
    get_backend = MagicMock(return_value=backend)
    monkeypatch.setattr(remote_backends, "get_backend", get_backend)
    path = tmp_path / "j1.job.json"
    path.write_text(
        json.dumps(
            {
                "job_id": "j1",
                "backend": "discovery",
                "backend_config": saved_config,
                "backend_state": {"operation_id": "operation"},
                "status": "submitted",
                "_restored_from_disk": False,
            }
        )
    )

    restored = Job.discover(tmp_path)[0] if discover else Job.load(path)
    status = restored.check()

    get_backend.assert_called_once_with("discovery")
    backend.connect.assert_called_once_with()
    backend.check.assert_called_once_with({"operation_id": "operation"})
    assert status.status == "running"
    assert restored.backend_config == saved_config
    assert "_restored_from_disk" not in restored.to_dict()


def test_restored_job_accepts_attached_backend(tmp_path, monkeypatch):
    """Custom restored connections remain available through an explicit backend."""
    backend = MagicMock()
    backend.check.return_value = JobStatus(job_id="j1", status="running")
    get_backend = MagicMock()
    monkeypatch.setattr(remote_backends, "get_backend", get_backend)
    job = Job(
        job_id="j1",
        backend="custom",
        backend_config={"host": "saved.invalid"},
        backend_state={"operation_id": "operation"},
    )
    restored = Job.load(job.save(tmp_path / "job.json"))
    restored.attach_backend(backend)

    assert restored.check().status == "running"

    get_backend.assert_not_called()
    backend.check.assert_called_once_with({"operation_id": "operation"})


def test_new_job_preserves_explicit_configuration(monkeypatch):
    """In-memory jobs retain connection settings supplied by their caller."""
    backend = MagicMock()
    backend.check.return_value = JobStatus(job_id="j1", status="running")
    get_backend = MagicMock(return_value=backend)
    monkeypatch.setattr(remote_backends, "get_backend", get_backend)
    job = Job(
        job_id="j1",
        backend="custom",
        backend_config={"host": "trusted.invalid"},
        backend_state={"operation_id": "operation"},
    )

    assert job.check().status == "running"

    get_backend.assert_called_once_with("custom", host="trusted.invalid")
