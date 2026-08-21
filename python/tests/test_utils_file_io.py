"""Tests for cross-platform file helpers."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os
import stat
from pathlib import Path

import pytest

from qdk_chemistry.utils import (
    read_text_file,
    write_file_atomically,
    write_text_file_atomically,
)


def test_write_read_and_replace_text(tmp_path: Path):
    path = tmp_path / "data.txt"

    write_text_file_atomically(path, "first")
    assert read_text_file(path) == "first"

    write_text_file_atomically(path, "second")
    assert read_text_file(path) == "second"


def test_create_parent_directories_when_requested(tmp_path: Path):
    path = tmp_path / "nested" / "directory" / "data.txt"

    write_text_file_atomically(path, "contents", create_parent_directories=True)

    assert read_text_file(path) == "contents"


def test_reject_missing_parent_directory_by_default(tmp_path: Path):
    path = tmp_path / "missing" / "data.txt"

    with pytest.raises(FileNotFoundError, match="Parent directory does not exist"):
        write_text_file_atomically(path, "contents")

    assert not path.exists()


def test_preserve_destination_when_writer_fails(tmp_path: Path):
    path = tmp_path / "data.txt"
    write_text_file_atomically(path, "original")

    def fail_after_write(temporary_path: Path) -> None:
        temporary_path.write_text("incomplete", encoding="utf-8")
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        write_file_atomically(path, fail_after_write)

    assert read_text_file(path) == "original"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.skipif(os.name != "nt", reason="Windows read-only behavior")
def test_clean_up_read_only_temporary_file_when_writer_fails(tmp_path: Path):
    path = tmp_path / "data.txt"
    write_text_file_atomically(path, "original")

    def fail_after_making_temporary_file_read_only(temporary_path: Path) -> None:
        temporary_path.write_text("incomplete", encoding="utf-8")
        temporary_path.chmod(stat.S_IREAD)
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        write_file_atomically(path, fail_after_making_temporary_file_read_only)

    assert read_text_file(path) == "original"
    assert list(tmp_path.iterdir()) == [path]


def test_preserve_line_endings(tmp_path: Path):
    path = tmp_path / "data.txt"
    contents = "a\r\nb\rc\nd"

    write_text_file_atomically(path, contents)

    assert read_text_file(path) == contents


def test_preserve_encoding_error(tmp_path: Path):
    path = tmp_path / "data.txt"
    path.write_text("contents", encoding="utf-8")

    with pytest.raises(LookupError):
        read_text_file(path, encoding="not-a-real-codec")


@pytest.mark.skipif(os.name == "nt", reason="POSIX special files")
def test_reject_non_regular_file(tmp_path: Path):
    fifo = tmp_path / "data.fifo"
    os.mkfifo(fifo)

    with pytest.raises(OSError, match="Path is not a regular file"):
        read_text_file(fifo)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are not portable to Windows")
def test_preserve_destination_permissions(tmp_path: Path):
    path = tmp_path / "data.txt"
    write_text_file_atomically(path, "original")
    path.chmod(0o640)

    write_text_file_atomically(path, "replacement")

    assert stat.S_IMODE(path.stat().st_mode) == 0o640


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are not portable to Windows")
def test_create_new_file_with_owner_only_permissions(tmp_path: Path):
    path = tmp_path / "data.txt"

    def write_and_relax_permissions(temporary_path: Path) -> None:
        temporary_path.write_text("contents", encoding="utf-8")
        temporary_path.chmod(0o666)

    write_file_atomically(path, write_and_relax_permissions)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "nt", reason="Windows read-only behavior")
def test_replace_read_only_destination_on_windows(tmp_path: Path):
    path = tmp_path / "data.txt"
    write_text_file_atomically(path, "original")
    path.chmod(stat.S_IREAD)

    write_text_file_atomically(path, "replacement")

    assert read_text_file(path) == "replacement"
    assert path.stat().st_mode & stat.S_IWRITE == 0


def test_preserve_destination_suffixes_for_writer(tmp_path: Path):
    path = tmp_path / "data.structure.json"
    observed_temporary_path: Path | None = None

    def write_temporary_file(temporary_path: Path) -> None:
        nonlocal observed_temporary_path
        observed_temporary_path = temporary_path
        temporary_path.write_text("contents", encoding="utf-8")

    write_file_atomically(path, write_temporary_file)

    assert observed_temporary_path is not None
    assert observed_temporary_path.suffixes[-2:] == [".structure", ".json"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX component length semantics")
def test_preserve_long_destination_suffix(tmp_path: Path):
    path = tmp_path / f"x.{'a' * 249}"
    observed_temporary_path: Path | None = None

    def write_temporary_file(temporary_path: Path) -> None:
        nonlocal observed_temporary_path
        observed_temporary_path = temporary_path
        temporary_path.write_text("contents", encoding="utf-8")

    write_file_atomically(path, write_temporary_file)

    assert observed_temporary_path is not None
    assert observed_temporary_path.suffix == path.suffix
    assert read_text_file(path) == "contents"
    assert list(tmp_path.iterdir()) == [path]


def test_reject_replaced_temporary_file(tmp_path: Path):
    path = tmp_path / "data.txt"

    def replace_temporary_file(temporary_path: Path) -> None:
        temporary_path.unlink()
        temporary_path.write_text("replacement", encoding="utf-8")

    if os.name == "nt":
        with pytest.raises(PermissionError):
            write_file_atomically(path, replace_temporary_file)
    else:
        with pytest.raises(RuntimeError, match="Temporary file identity changed"):
            write_file_atomically(path, replace_temporary_file)

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_freeze_relative_destination_before_writer_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first_directory.mkdir()
    second_directory.mkdir()
    monkeypatch.chdir(first_directory)

    def change_directory(temporary_path: Path) -> None:
        temporary_path.write_text("contents", encoding="utf-8")
        os.chdir(second_directory)

    write_file_atomically("data.txt", change_directory)

    assert read_text_file(first_directory / "data.txt") == "contents"
    assert not (second_directory / "data.txt").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink traversal semantics")
def test_preserve_symlink_parent_traversal_when_freezing_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    working_directory = tmp_path / "working"
    target_parent = tmp_path / "target-parent"
    target_directory = target_parent / "target"
    working_directory.mkdir()
    target_directory.mkdir(parents=True)
    (working_directory / "link").symlink_to(target_directory, target_is_directory=True)
    monkeypatch.chdir(working_directory)

    write_text_file_atomically("link/../data.txt", "contents")

    assert read_text_file(target_parent / "data.txt") == "contents"
    assert not (working_directory / "data.txt").exists()


def test_support_unicode_paths(tmp_path: Path):
    path = tmp_path / "data-\u6570\u636e.txt"

    write_text_file_atomically(path, "contents")

    assert read_text_file(path) == "contents"
