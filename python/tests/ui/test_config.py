"""Unit tests for qdk_chemistry.ui configuration."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import errno
from pathlib import Path

import pytest

from qdk_chemistry.ui.config import QDKMCPConfig


def test_default_scratch_falls_back_to_home_on_read_only_filesystem(monkeypatch, tmp_path):
    monkeypatch.delenv("QDK_SCRATCH_DIR", raising=False)
    monkeypatch.delenv("QDK_CACHE_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    original_setup = QDKMCPConfig._setup_directories

    def fake_setup(self):
        if self.scratch_dir == Path("/scratch"):
            raise OSError(errno.EROFS, "Read-only file system", str(self.scratch_dir))
        original_setup(self)

    monkeypatch.setattr(QDKMCPConfig, "_setup_directories", fake_setup)

    cfg = QDKMCPConfig()

    expected_scratch = tmp_path / ".qdk_chem" / "scratch"
    assert cfg.scratch_dir == expected_scratch
    assert cfg.projects_dir == expected_scratch / "projects"
    assert cfg.cache_dir == expected_scratch / "cache"
    assert cfg.projects_dir.is_dir()
    assert cfg.cache_dir.is_dir()


def test_explicit_scratch_dir_error_is_not_replaced_with_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("QDK_SCRATCH_DIR", "/scratch")
    monkeypatch.delenv("QDK_CACHE_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    def fake_setup(self):
        raise OSError(errno.EROFS, "Read-only file system", str(self.scratch_dir))

    monkeypatch.setattr(QDKMCPConfig, "_setup_directories", fake_setup)

    with pytest.raises(OSError, match="Read-only file system") as exc_info:
        QDKMCPConfig()

    assert exc_info.value.errno == errno.EROFS
