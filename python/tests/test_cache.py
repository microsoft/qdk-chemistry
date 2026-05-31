"""Tests for the cache backend system (FolderCache, registry, resolve_cache)."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json

import numpy as np
import pytest

from qdk_chemistry.data import Orbitals
from qdk_chemistry.remote.cache import (
    CacheBackend,
    FolderCache,
    get_cache,
    register_cache,
    resolve_cache,
)
from qdk_chemistry.remote.job import Job

from .test_helpers import create_test_orbitals

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def cache_dir(tmp_path):
    """Return a temporary directory for cache storage."""
    return tmp_path / "test_cache"


@pytest.fixture
def folder_cache(cache_dir):
    """Return a FolderCache backed by a temporary directory."""
    return FolderCache(path=cache_dir)


@pytest.fixture
def sample_job():
    """Return a minimal Job instance for testing."""
    return Job(
        job_id="test123",
        backend="local",
        backend_config={},
        backend_state={},
        algorithm_info={"type": "scf_solver", "name": "qdk"},
        status="retrieved",
        run_hash="abc123def456",
        input_hashes={"arg_0": "hash_structure"},
        output_hashes=[
            {"hash": "hash_float_result", "type": "float", "value": -75.5},
            {"hash": "hash_wfn_result", "type": "wavefunction"},
        ],
    )


@pytest.fixture
def sample_orbitals():
    """Return a small test Orbitals DataClass."""
    return create_test_orbitals(3)


# ── FolderCache: Job metadata ────────────────────────────────────────────────


class TestFolderCacheJobs:
    """Tests for FolderCache job metadata storage."""

    def test_get_job_miss(self, folder_cache):
        """Cache miss returns None."""
        assert folder_cache.get_job("nonexistent_hash") is None

    def test_put_and_get_job(self, folder_cache, sample_job):
        """Round-trip: put then get returns equivalent job."""
        folder_cache.put_job("run_abc", sample_job)
        loaded = folder_cache.get_job("run_abc")

        assert loaded is not None
        assert loaded.job_id == sample_job.job_id
        assert loaded.backend == sample_job.backend
        assert loaded.status == sample_job.status
        assert loaded.run_hash == sample_job.run_hash
        assert loaded.input_hashes == sample_job.input_hashes
        assert loaded.output_hashes == sample_job.output_hashes
        assert loaded.algorithm_info == sample_job.algorithm_info

    def test_put_job_creates_directory(self, cache_dir, folder_cache, sample_job):
        """put_job creates the cache directory if it doesn't exist."""
        assert not cache_dir.exists()
        folder_cache.put_job("hash1", sample_job)
        assert cache_dir.exists()

    def test_put_job_overwrites(self, folder_cache, sample_job):
        """Storing a second job with the same hash overwrites it."""
        folder_cache.put_job("hash1", sample_job)

        updated_job = Job(
            job_id="updated_id",
            backend="local",
            backend_config={},
            backend_state={},
            status="Failed",
        )
        folder_cache.put_job("hash1", updated_job)

        loaded = folder_cache.get_job("hash1")
        assert loaded.job_id == "updated_id"
        assert loaded.status == "Failed"

    def test_job_file_is_valid_json(self, folder_cache, sample_job, cache_dir):
        """Job file on disk is valid, human-readable JSON."""
        folder_cache.put_job("hash1", sample_job)
        job_path = cache_dir / "hash1.job.json"
        data = json.loads(job_path.read_text())
        assert data["job_id"] == "test123"

    def test_delete_job_existing(self, folder_cache, sample_job):
        """Deleting an existing job returns True and removes the file."""
        folder_cache.put_job("hash1", sample_job)
        assert folder_cache.delete_job("hash1")
        assert folder_cache.get_job("hash1") is None

    def test_delete_job_nonexistent(self, folder_cache):
        """Deleting a nonexistent job returns False."""
        assert not folder_cache.delete_job("no_such_hash")

    def test_delete_job_removes_associated_data(self, folder_cache, sample_job, sample_orbitals):
        """delete_job also removes data blobs referenced in output_hashes."""
        # Store a data object first
        folder_cache.put_data("hash_wfn_result", sample_orbitals)
        assert folder_cache.get_data("hash_wfn_result") is not None

        folder_cache.put_job("run_abc", sample_job)
        folder_cache.delete_job("run_abc")

        # Data blob should be removed (only non-primitive entries)
        assert folder_cache.get_data("hash_wfn_result") is None


# ── FolderCache: DataClass blobs ─────────────────────────────────────────────


class TestFolderCacheData:
    """Tests for FolderCache DataClass blob storage."""

    def test_get_data_miss(self, folder_cache):
        """Cache miss returns None."""
        assert folder_cache.get_data("nonexistent_hash") is None

    def test_put_and_get_data_orbitals(self, folder_cache, sample_orbitals):
        """Round-trip: storing and retrieving Orbitals preserves content."""
        folder_cache.put_data("orb_hash_1", sample_orbitals)
        loaded = folder_cache.get_data("orb_hash_1")
        assert loaded is not None
        assert isinstance(loaded, Orbitals)
        np.testing.assert_array_equal(loaded.get_coefficients(), sample_orbitals.get_coefficients())

    def test_put_data_skips_if_exists(self, folder_cache, sample_orbitals, cache_dir):
        """Second put with same hash is a no-op (doesn't overwrite)."""
        folder_cache.put_data("orb_hash", sample_orbitals)
        first_mtime = next(iter(cache_dir.glob("orb_hash.*.h5"))).stat().st_mtime_ns

        # Put again — should skip
        folder_cache.put_data("orb_hash", sample_orbitals)
        second_mtime = next(iter(cache_dir.glob("orb_hash.*.h5"))).stat().st_mtime_ns
        assert first_mtime == second_mtime

    def test_put_data_creates_directory(self, cache_dir, folder_cache, sample_orbitals):
        """put_data creates the cache directory if it doesn't exist."""
        assert not cache_dir.exists()
        folder_cache.put_data("orb1", sample_orbitals)
        assert cache_dir.exists()

    def test_delete_data_existing(self, folder_cache, sample_orbitals):
        """Deleting existing data returns True."""
        folder_cache.put_data("orb1", sample_orbitals)
        assert folder_cache.delete_data("orb1")
        assert folder_cache.get_data("orb1") is None

    def test_delete_data_nonexistent(self, folder_cache):
        """Deleting nonexistent data returns False."""
        assert not folder_cache.delete_data("nope")

    def test_data_filename_contains_type_name(self, folder_cache, sample_orbitals, cache_dir):
        """Blob file is named <hash>.<type_name>.h5."""
        folder_cache.put_data("myhash", sample_orbitals)
        matches = list(cache_dir.glob("myhash.*.h5"))
        assert len(matches) == 1
        assert "orbitals" in matches[0].name


# ── FolderCache: clear ───────────────────────────────────────────────────────


class TestFolderCacheClear:
    """Tests for FolderCache.clear()."""

    def test_clear_removes_all(self, folder_cache, sample_job, sample_orbitals, cache_dir):
        """clear() removes the entire cache directory."""
        folder_cache.put_job("h1", sample_job)
        folder_cache.put_data("d1", sample_orbitals)
        assert cache_dir.exists()

        folder_cache.clear()
        assert not cache_dir.exists()

    def test_clear_nonexistent_dir(self, folder_cache):
        """clear() on a non-existent directory does not error."""
        folder_cache.clear()  # no-op


# ── Cache registry / resolve_cache ───────────────────────────────────────────


class TestCacheRegistry:
    """Tests for the cache registry and resolve_cache helper."""

    def test_resolve_cache_none(self):
        """None input returns None."""
        assert resolve_cache(None) is None

    def test_resolve_cache_instance(self, folder_cache):
        """A CacheBackend instance passes through unchanged."""
        assert resolve_cache(folder_cache) is folder_cache

    def test_resolve_cache_path_object(self, tmp_path):
        """A Path object creates a FolderCache."""
        cache = resolve_cache(tmp_path / "my_cache")
        assert isinstance(cache, FolderCache)

    def test_resolve_cache_string_path(self, tmp_path):
        """A string that looks like a path creates a FolderCache."""
        cache = resolve_cache(str(tmp_path / "some_cache"))
        assert isinstance(cache, FolderCache)

    def test_resolve_cache_registered_name(self, tmp_path):
        """A registered name with explicit config creates the right backend."""
        cache = get_cache("folder", path=str(tmp_path / "named_cache"))
        assert isinstance(cache, FolderCache)

    def test_get_cache_folder(self, tmp_path):
        """get_cache('folder', path=...) creates a FolderCache."""
        cache = get_cache("folder", path=str(tmp_path / "c"))
        assert isinstance(cache, FolderCache)

    def test_get_cache_unknown_raises(self):
        """get_cache with an unknown name raises ValueError."""
        with pytest.raises(ValueError, match="No cache registered"):
            get_cache("does_not_exist")

    @pytest.mark.usefixtures("tmp_path")
    def test_register_custom_cache(self):
        """A custom cache class can be registered and retrieved."""

        class DummyCache(CacheBackend):
            name = "dummy"

            def get_job(self, _h):
                return None

            def put_job(self, _h, j):
                pass

            def get_data(self, _h):
                return None

            def put_data(self, _h, d):
                pass

            def delete_job(self, _h):
                return False

            def delete_data(self, _h):
                return False

            def clear(self):
                pass

        register_cache("test_dummy")(DummyCache)
        cache = get_cache("test_dummy")
        assert isinstance(cache, DummyCache)
