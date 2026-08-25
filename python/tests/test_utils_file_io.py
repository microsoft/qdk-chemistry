"""Tests for cross-platform file helpers."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import ctypes
import errno
import gc
import os
import stat
import sys
import threading
from ctypes import wintypes
from pathlib import Path

import pytest

from qdk_chemistry.utils import (
    ensure_parent_directory,
    read_text_file,
    write_file_atomically,
    write_text_file_atomically,
)
from qdk_chemistry.utils import file_io as file_io_module


def test_write_read_and_replace_text(tmp_path: Path):
    path = tmp_path / "data.txt"

    write_text_file_atomically(path, "first")
    assert read_text_file(path) == "first"

    write_text_file_atomically(path, "second")
    assert read_text_file(path) == "second"


def test_resolve_pathlike_once(tmp_path: Path):
    path = tmp_path / "data.txt"

    class ChangingPath:
        calls = 0

        def __fspath__(self) -> str:
            self.calls += 1
            return str(path) if self.calls == 1 else "invalid\0path"

    changing_path = ChangingPath()
    write_text_file_atomically(changing_path, "contents")

    assert changing_path.calls == 1
    assert read_text_file(path) == "contents"


def test_create_parent_directories_when_requested(tmp_path: Path):
    path = tmp_path / "nested" / "directory" / "data.txt"

    write_text_file_atomically(path, "contents", create_parent_directories=True)

    assert read_text_file(path) == "contents"
    if os.name != "nt":
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.parent.parent.stat().st_mode) == 0o700


def test_freeze_relative_parent_before_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first_directory.mkdir()
    second_directory.mkdir()
    monkeypatch.chdir(first_directory)

    if os.name == "nt":
        mkdir = Path.mkdir

        def change_directory_then_create(directory: Path, *args, **kwargs) -> None:
            os.chdir(second_directory)
            mkdir(directory, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", change_directory_then_create)
    else:
        create_private_directories = file_io_module._create_private_directories

        def change_directory_then_create(directory: Path) -> None:
            os.chdir(second_directory)
            create_private_directories(directory)

        monkeypatch.setattr(file_io_module, "_create_private_directories", change_directory_then_create)

    ensure_parent_directory("nested/data.txt")

    assert (first_directory / "nested").is_dir()
    assert not (second_directory / "nested").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX umask semantics")
def test_create_private_parent_directories_under_restrictive_umask(tmp_path: Path):
    path = tmp_path / "private" / "nested" / "data.txt"
    original_umask = os.umask(0o777)
    try:
        write_text_file_atomically(path, "contents", create_parent_directories=True)
    finally:
        os.umask(original_umask)

    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert read_text_file(path) == "contents"


@pytest.mark.parametrize("use_setgid_parent", [False, True])
@pytest.mark.parametrize("use_descendant_parent", [False, True])
@pytest.mark.skipif(os.name == "nt", reason="POSIX umask semantics")
def test_serialize_concurrent_parent_creation_under_restrictive_umask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_setgid_parent: bool,
    use_descendant_parent: bool,
):
    root = tmp_path / "root"
    root.mkdir()
    if use_setgid_parent:
        if sys.platform != "linux":
            pytest.skip("setgid directory inheritance is verified on Linux")
        root.chmod(0o2700)
    shared_parent = root / "shared"
    if use_descendant_parent:
        first_path = shared_parent / "first" / "data.txt"
        second_path = shared_parent / "second" / "data.txt"
    else:
        first_path = shared_parent / "first.txt"
        second_path = shared_parent / "second.txt"
    mkdir = file_io_module.os.mkdir
    parent_created = threading.Event()
    release_creator = threading.Event()
    second_finished = threading.Event()
    paused = False
    errors: list[Exception] = []

    def pause_after_creating_parent(path: Path, mode: int) -> None:
        nonlocal paused
        mkdir(path, mode)
        if Path(path) == shared_parent and not paused:
            paused = True
            parent_created.set()
            if not release_creator.wait(timeout=10):
                raise RuntimeError("timed out waiting to release directory creator")

    def write(path: Path, finished: threading.Event | None = None) -> None:
        try:
            write_text_file_atomically(path, "contents", create_parent_directories=True)
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(error)
        finally:
            if finished is not None:
                finished.set()

    monkeypatch.setattr(file_io_module.os, "mkdir", pause_after_creating_parent)
    original_umask = os.umask(0o777)
    first = threading.Thread(target=write, args=(first_path,))
    second = threading.Thread(target=write, args=(second_path, second_finished))
    try:
        first.start()
        assert parent_created.wait(timeout=10)
        second.start()
        assert not second_finished.wait(timeout=0.05)
        release_creator.set()
        first.join(timeout=10)
        second.join(timeout=10)
    finally:
        release_creator.set()
        first.join(timeout=10)
        second.join(timeout=10)
        os.umask(original_umask)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert read_text_file(first_path) == "contents"
    assert read_text_file(second_path) == "contents"
    assert stat.S_IMODE(shared_parent.stat().st_mode) == 0o700


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics")
def test_retry_after_directory_initialization_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    directory = tmp_path / "shared"
    directory.mkdir()
    directory.chmod(0)
    create_private_directories_once = file_io_module._create_private_directories_once
    calls = 0

    def fail_with_stale_permission_error(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            directory.chmod(0o700)
            raise PermissionError(errno.EACCES, "initializing", os.fspath(path))
        create_private_directories_once(path)

    monkeypatch.setattr(
        file_io_module,
        "_create_private_directories_once",
        fail_with_stale_permission_error,
    )

    file_io_module._create_private_directories(directory)

    assert calls == 2


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics")
def test_retry_reservation_after_directory_initialization_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    directory = tmp_path / "shared"
    directory.mkdir()
    directory.chmod(0)
    destination = directory / "data.txt"
    reservation = object()
    calls = 0

    def fail_with_stale_permission_error(path: Path) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            directory.chmod(0o700)
            raise PermissionError(errno.EACCES, "initializing", os.fspath(path))
        assert path == destination
        return reservation

    monkeypatch.setattr(
        file_io_module,
        "_reserve_temporary_file",
        fail_with_stale_permission_error,
    )

    assert file_io_module._reserve_temporary_file_with_parent_retry(destination) is reservation
    assert calls == 2


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics")
def test_retry_after_multiple_directory_state_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    directory = tmp_path / "nested" / "parent"
    states = iter(
        [
            (False, (("root", 1, 1, 0o700),)),
            (False, (("root", 1, 1, 0o700), ("nested", 1, 2, 0o700))),
            (
                False,
                (
                    ("root", 1, 1, 0o700),
                    ("nested", 1, 2, 0o700),
                    ("parent", 1, 3, 0o700),
                ),
            ),
        ]
    )
    calls = 0

    def fail_twice(_: Path) -> None:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError(errno.EACCES, "initializing")

    monkeypatch.setattr(file_io_module, "_directory_initialization_state", lambda _: next(states))
    monkeypatch.setattr(file_io_module, "_create_private_directories_once", fail_twice)

    file_io_module._create_private_directories(directory)

    assert calls == 3


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics")
def test_retry_reservation_after_multiple_directory_state_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "nested" / "parent" / "data.txt"
    reservation = object()
    states = iter(
        [
            (False, (("root", 1, 1, 0o700),)),
            (False, (("root", 1, 1, 0o700), ("nested", 1, 2, 0o700))),
            (
                False,
                (
                    ("root", 1, 1, 0o700),
                    ("nested", 1, 2, 0o700),
                    ("parent", 1, 3, 0o700),
                ),
            ),
        ]
    )
    calls = 0

    def fail_twice(_: Path) -> object:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError(errno.EACCES, "initializing")
        return reservation

    monkeypatch.setattr(file_io_module, "_directory_initialization_state", lambda _: next(states))
    monkeypatch.setattr(file_io_module, "_reserve_temporary_file", fail_twice)

    assert file_io_module._reserve_temporary_file_with_parent_retry(destination) is reservation
    assert calls == 3


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics")
def test_do_not_modify_permanently_inaccessible_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    parent = tmp_path / "inaccessible"
    parent.mkdir()
    parent.chmod(0)
    monkeypatch.setattr(file_io_module, "_DIRECTORY_CREATION_RETRY_TIMEOUT_SECONDS", 0)

    try:
        with pytest.raises(PermissionError):
            ensure_parent_directory(parent / "nested" / "data.txt")
        assert stat.S_IMODE(parent.stat().st_mode) == 0
    finally:
        parent.chmod(0o700)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics")
def test_existing_parent_is_a_noop_without_write_permission(tmp_path: Path):
    parent = tmp_path / "existing"
    parent.mkdir()
    parent.chmod(0o500)

    try:
        ensure_parent_directory(parent / "data.txt")
        assert stat.S_IMODE(parent.stat().st_mode) == 0o500
    finally:
        parent.chmod(0o700)


def test_reject_missing_parent_directory_by_default(tmp_path: Path):
    path = tmp_path / "missing" / "data.txt"

    with pytest.raises(FileNotFoundError, match="Parent directory does not exist"):
        write_text_file_atomically(path, "contents")

    assert not path.exists()


def test_reject_trailing_separator_destination(tmp_path: Path):
    path = tmp_path / "data"

    with pytest.raises(ValueError, match="must name a file"):
        write_text_file_atomically(f"{path}{os.sep}", "contents")

    assert not path.exists()


def test_reject_trailing_separator_in_all_path_helpers(tmp_path: Path):
    path = tmp_path / "data"
    trailing_path = f"{path}{os.sep}"

    with pytest.raises(ValueError, match="must name a file"):
        ensure_parent_directory(trailing_path)
    with pytest.raises(ValueError, match="must name a file"):
        read_text_file(trailing_path)


@pytest.mark.parametrize(
    "path",
    [
        "",
        ".",
        "..",
        f"data{os.sep}.",
        "data\0ignored.txt",
    ],
)
def test_reject_invalid_destination_before_writer_runs(tmp_path: Path, path: str):
    writer_ran = False
    destination = path if not path.startswith("data") else f"{tmp_path}{os.sep}{path}"

    def writer(temporary_path: Path) -> None:
        nonlocal writer_ran
        writer_ran = True
        temporary_path.write_text("contents", encoding="utf-8")

    with pytest.raises(ValueError, match=r"must name a file|embedded NUL"):
        write_file_atomically(destination, writer)
    with pytest.raises(ValueError, match=r"must name a file|embedded NUL"):
        ensure_parent_directory(destination)
    with pytest.raises(ValueError, match=r"must name a file|embedded NUL"):
        read_text_file(destination)

    assert not writer_ran


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


def test_close_failure_does_not_skip_cleanup_or_retry_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "data.txt"
    write_text_file_atomically(path, "original")
    real_close = os.close
    closed_descriptors: list[int] = []

    def close_then_fail(descriptor: int) -> None:
        closed_descriptors.append(descriptor)
        real_close(descriptor)
        raise OSError("close failed")

    def fail_after_write(temporary_path: Path) -> None:
        temporary_path.write_text("incomplete", encoding="utf-8")
        raise RuntimeError("writer failed")

    monkeypatch.setattr(file_io_module.os, "close", close_then_fail)
    with pytest.raises(RuntimeError, match="writer failed") as caught:
        write_file_atomically(path, fail_after_write)
    monkeypatch.undo()

    assert isinstance(caught.value.__cause__, OSError)
    assert str(caught.value.__cause__) == "close failed"
    assert len(closed_descriptors) == 1
    assert read_text_file(path) == "original"
    assert list(tmp_path.iterdir()) == [path]


def test_close_failure_does_not_resurrect_callers_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "data.txt"
    real_close = os.close

    def close_then_fail(descriptor: int) -> None:
        real_close(descriptor)
        raise OSError("close failed")

    monkeypatch.setattr(file_io_module.os, "close", close_then_fail)
    try:
        raise ValueError("caller error")
    except ValueError:
        with pytest.raises(OSError, match="close failed"):
            write_text_file_atomically(path, "contents")
    monkeypatch.undo()

    assert read_text_file(path) == "contents"


def test_reservation_finalizer_closes_and_removes_temporary_file(tmp_path: Path):
    destination = tmp_path / "data.txt"
    reservation = file_io_module._reserve_temporary_file(destination)
    descriptor = reservation.descriptor
    temporary_path = reservation.path

    del reservation
    gc.collect()

    with pytest.raises(OSError, match="(?i)(bad file descriptor|handle is invalid)"):
        os.fstat(descriptor)
    assert not temporary_path.exists()


def test_clean_up_after_initial_reservation_fstat_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "data.txt"
    fstat = file_io_module.os.fstat
    create_exclusive_file = file_io_module._create_exclusive_file
    reservation_descriptor: int | None = None
    reservation_fstat_calls = 0
    closed_descriptors: list[int] = []
    close_descriptor = file_io_module._close_descriptor

    def create_and_track_reservation(temporary_path: Path) -> int:
        nonlocal reservation_descriptor
        reservation_descriptor = create_exclusive_file(temporary_path)
        return reservation_descriptor

    def fail_initial_reservation_fstat(descriptor: int) -> os.stat_result:
        nonlocal reservation_fstat_calls
        if descriptor == reservation_descriptor:
            reservation_fstat_calls += 1
            if reservation_fstat_calls == 1:
                raise OSError("identity snapshot failed")
        return fstat(descriptor)

    def record_close(descriptor: int) -> OSError | None:
        closed_descriptors.append(descriptor)
        return close_descriptor(descriptor)

    monkeypatch.setattr(file_io_module, "_create_exclusive_file", create_and_track_reservation)
    monkeypatch.setattr(file_io_module.os, "fstat", fail_initial_reservation_fstat)
    monkeypatch.setattr(file_io_module, "_close_descriptor", record_close)

    with pytest.raises(OSError, match="identity snapshot failed"):
        write_text_file_atomically(destination, "contents")

    assert reservation_fstat_calls == 2
    assert len(closed_descriptors) == 1
    assert list(tmp_path.iterdir()) == []


def test_failed_reservation_adoption_has_one_descriptor_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    temporary_path = tmp_path / "temporary.txt"
    descriptor = file_io_module._create_exclusive_file(temporary_path)
    reserved_status = os.fstat(descriptor)
    reservations: list[file_io_module._TemporaryFileReservation] = []
    adopt_descriptor = file_io_module._TemporaryFileReservation.adopt_descriptor

    def adopt_then_fail(
        reservation: file_io_module._TemporaryFileReservation,
        owned_descriptor: int,
    ) -> None:
        adopt_descriptor(reservation, owned_descriptor)
        reservations.append(reservation)
        raise MemoryError("adoption failed")

    monkeypatch.setattr(
        file_io_module._TemporaryFileReservation,
        "adopt_descriptor",
        adopt_then_fail,
    )

    with pytest.raises(MemoryError, match="adoption failed"):
        file_io_module._package_reservation(descriptor, temporary_path, reserved_status)

    assert reservations[0].descriptor == -1
    assert not temporary_path.exists()


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


@pytest.mark.skipif(os.name == "nt", reason="POSIX special files")
def test_reject_non_regular_destination(tmp_path: Path):
    fifo = tmp_path / "data.fifo"
    os.mkfifo(fifo)

    with pytest.raises(OSError, match="Destination is not a regular file"):
        write_text_file_atomically(fifo, "replacement")

    assert stat.S_ISFIFO(fifo.lstat().st_mode)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are not portable to Windows")
def test_preserve_destination_permissions(tmp_path: Path):
    path = tmp_path / "data.txt"
    write_text_file_atomically(path, "original")
    path.chmod(0o640)

    write_text_file_atomically(path, "replacement")

    assert stat.S_IMODE(path.stat().st_mode) == 0o640


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are not portable to Windows")
def test_clear_special_permission_bits_on_replacement(tmp_path: Path):
    path = tmp_path / "data.txt"
    write_text_file_atomically(path, "original")
    path.chmod(0o7755)

    write_text_file_atomically(path, "replacement")

    assert stat.S_IMODE(path.stat().st_mode) == 0o755


@pytest.mark.skipif(os.name == "nt", reason="POSIX umask semantics")
def test_restrictive_umask_does_not_prevent_writing(tmp_path: Path):
    path = tmp_path / "data.txt"
    original_umask = os.umask(0o777)
    try:
        write_text_file_atomically(path, "contents")
    finally:
        os.umask(original_umask)

    assert read_text_file(path) == "contents"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
def test_reject_symlink_destination_without_copying_referent_mode(tmp_path: Path):
    target = tmp_path / "target.txt"
    link = tmp_path / "link.txt"
    target.write_text("target", encoding="utf-8")
    target.chmod(0o6755)
    link.symlink_to(target)

    with pytest.raises(ValueError, match="Symlink destinations are not supported"):
        write_text_file_atomically(link, "replacement")

    assert link.is_symlink()
    assert target.read_text(encoding="utf-8") == "target"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are not portable to Windows")
def test_create_new_file_with_owner_only_permissions(tmp_path: Path):
    path = tmp_path / "data.txt"

    def write_and_relax_permissions(temporary_path: Path) -> None:
        temporary_path.write_text("contents", encoding="utf-8")
        temporary_path.chmod(0o666)

    write_file_atomically(path, write_and_relax_permissions)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics")
def test_reject_filesystem_that_ignores_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "data.txt"
    original_fchmod = os.fchmod

    def apply_broader_permissions(descriptor: int, _mode: int) -> None:
        original_fchmod(descriptor, 0o644)

    monkeypatch.setattr(file_io_module.os, "fchmod", apply_broader_permissions)

    with pytest.raises(PermissionError, match="did not apply permissions"):
        write_text_file_atomically(path, "contents")

    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.name != "nt", reason="Windows read-only behavior")
def test_replace_read_only_destination_on_windows(tmp_path: Path):
    if sys.platform != "win32":
        raise AssertionError("Windows-only test ran on another platform")
    path = tmp_path / "data.txt"
    write_text_file_atomically(path, "original")
    set_attributes = ctypes.WinDLL("kernel32", use_last_error=True).SetFileAttributesW
    set_attributes.argtypes = (wintypes.LPCWSTR, wintypes.DWORD)
    set_attributes.restype = wintypes.BOOL
    assert set_attributes(str(path), 0x00000001)

    write_text_file_atomically(path, "replacement")

    assert read_text_file(path) == "replacement"
    assert path.stat().st_mode & stat.S_IWRITE == 0


@pytest.mark.skipif(os.name != "nt", reason="Windows read-only behavior")
def test_reject_read_only_destination_with_surviving_hard_link(tmp_path: Path):
    path = tmp_path / "data.txt"
    alias = tmp_path / "alias.txt"
    write_text_file_atomically(path, "original")
    os.link(path, alias)
    path.chmod(stat.S_IREAD)

    with pytest.raises(RuntimeError, match="multiple hard links"):
        write_text_file_atomically(path, "replacement")

    assert read_text_file(path) == "original"
    assert read_text_file(alias) == "original"
    assert alias.stat().st_mode & stat.S_IWRITE == 0


@pytest.mark.skipif(os.name != "nt", reason="Windows read-only behavior")
def test_restore_read_only_attribute_on_hard_link_created_during_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "data.txt"
    alias = tmp_path / "alias.txt"
    write_text_file_atomically(path, "original")
    path.chmod(stat.S_IREAD)
    real_replace = os.replace
    replace_calls = 0

    def link_before_retry(source: Path, destination: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            os.link(destination, alias)
        real_replace(source, destination)

    monkeypatch.setattr(file_io_module.os, "replace", link_before_retry)
    write_text_file_atomically(path, "replacement")

    assert replace_calls == 2
    assert read_text_file(path) == "replacement"
    assert read_text_file(alias) == "original"
    assert alias.stat().st_mode & stat.S_IWRITE == 0


@pytest.mark.skipif(os.name != "nt", reason="Windows read-only behavior")
def test_preserve_writable_destination_on_windows(tmp_path: Path):
    path = tmp_path / "data.txt"
    write_text_file_atomically(path, "original")

    def write_read_only_temporary_file(temporary_path: Path) -> None:
        temporary_path.write_text("replacement", encoding="utf-8")
        temporary_path.chmod(stat.S_IREAD)

    write_file_atomically(path, write_read_only_temporary_file)

    assert read_text_file(path) == "replacement"
    assert path.stat().st_mode & stat.S_IWRITE != 0


@pytest.mark.skipif(os.name != "nt", reason="Windows file attributes")
def test_strip_temporary_attributes_from_replacement_on_windows(tmp_path: Path):
    if sys.platform != "win32":
        raise AssertionError("Windows-only test ran on another platform")
    path = tmp_path / "data.txt"
    write_text_file_atomically(path, "original")
    set_attributes = ctypes.WinDLL("kernel32", use_last_error=True).SetFileAttributesW
    set_attributes.argtypes = (wintypes.LPCWSTR, wintypes.DWORD)
    set_attributes.restype = wintypes.BOOL

    def write_temporary_file(temporary_path: Path) -> None:
        temporary_path.write_text("replacement", encoding="utf-8")
        assert set_attributes(str(temporary_path), 0x00000002 | 0x00000100)

    write_file_atomically(path, write_temporary_file)

    get_attributes = ctypes.WinDLL("kernel32", use_last_error=True).GetFileAttributesW
    get_attributes.argtypes = (wintypes.LPCWSTR,)
    get_attributes.restype = wintypes.DWORD
    assert get_attributes(str(path)) & (0x00000002 | 0x00000100) == 0


@pytest.mark.skipif(os.name != "nt", reason="Windows descriptor conversion")
def test_remove_created_file_when_windows_descriptor_conversion_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "data.txt"

    class FailingMsvcrt:
        @staticmethod
        def open_osfhandle(_handle: int, _flags: int) -> int:
            raise OSError("descriptor conversion failed")

    monkeypatch.setattr(file_io_module.importlib, "import_module", lambda _name: FailingMsvcrt)

    with pytest.raises(OSError, match="descriptor conversion failed"):
        file_io_module._open_windows_file(
            path,
            desired_access=0,
            creation_disposition=1,
        )

    assert not path.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows descriptor conversion")
def test_resolve_windows_descriptor_converter_before_creating_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "data.txt"

    def fail_import(_name: str) -> None:
        raise OSError("converter unavailable")

    monkeypatch.setattr(file_io_module.importlib, "import_module", fail_import)

    with pytest.raises(OSError, match="converter unavailable"):
        file_io_module._open_windows_file(
            path,
            desired_access=0,
            creation_disposition=1,
        )

    assert not path.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows file-sharing behavior")
def test_allow_exclusive_writer_on_windows(tmp_path: Path):
    if sys.platform != "win32":
        raise AssertionError("Windows-only test ran on another platform")
    path = tmp_path / "data.txt"

    def write_with_exclusive_handle(temporary_path: Path) -> None:
        create_file = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
        create_file.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        create_file.restype = wintypes.HANDLE
        handle = create_file(
            str(temporary_path),
            0x40000000,
            0,
            None,
            3,
            0x00000080,
            None,
        )
        assert handle != wintypes.HANDLE(-1).value
        try:
            written = wintypes.DWORD()
            contents = ctypes.create_string_buffer(b"contents")
            assert ctypes.windll.kernel32.WriteFile(
                handle,
                contents,
                len(contents.value),
                ctypes.byref(written),
                None,
            )
            assert written.value == len(contents.value)
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)

    write_file_atomically(path, write_with_exclusive_handle)

    assert read_text_file(path) == "contents"


@pytest.mark.skipif(os.name != "nt", reason="Windows file-sharing behavior")
def test_reader_does_not_block_atomic_replacement_on_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    if sys.platform != "win32":
        raise AssertionError("Windows-only test ran on another platform")
    path = tmp_path / "data.txt"
    write_text_file_atomically(path, "original")
    reader_opened = threading.Event()
    release_reader = threading.Event()
    original_fdopen = file_io_module.os.fdopen

    def blocking_fdopen(*args, **kwargs):
        stream = original_fdopen(*args, **kwargs)
        reader_opened.set()
        assert release_reader.wait(timeout=10)
        return stream

    monkeypatch.setattr(file_io_module.os, "fdopen", blocking_fdopen)
    result: list[str] = []
    reader = threading.Thread(target=lambda: result.append(read_text_file(path)))
    reader.start()
    assert reader_opened.wait(timeout=10)
    try:
        write_text_file_atomically(path, "replacement")
    finally:
        release_reader.set()
        reader.join(timeout=10)

    assert result == ["original"]
    assert read_text_file(path) == "replacement"


@pytest.mark.skipif(os.name != "nt", reason="Windows path semantics")
def test_reject_alternate_data_stream_destination_on_windows(tmp_path: Path):
    path = tmp_path / "data.txt:stream"

    with pytest.raises(ValueError, match="alternate data streams"):
        write_text_file_atomically(path, "contents")

    assert not (tmp_path / "data.txt").exists()


def test_count_windows_surrogate_code_units(monkeypatch: pytest.MonkeyPatch):
    short_path = Path("data.\ud800")
    long_path = Path("data." + "\ud800" * 256)

    class WindowsOs:
        name = "nt"

    monkeypatch.setattr(file_io_module, "os", WindowsOs())

    assert not file_io_module._component_is_too_long(short_path)
    assert file_io_module._component_is_too_long(long_path)


@pytest.mark.skipif(os.name != "nt", reason="Windows path semantics")
def test_fall_back_for_near_max_path_destination_on_windows(tmp_path: Path):
    parent = tmp_path
    while len(str(parent / "d.txt")) < 220:
        parent /= "segment123"
    current_length = len(str(parent / "d.txt"))
    if current_length < 244:
        parent /= "p" * (243 - current_length)
    parent.mkdir(parents=True)
    path = parent / "d.txt"

    write_text_file_atomically(path, "contents")

    assert read_text_file(path) == "contents"


@pytest.mark.skipif(os.name != "nt", reason="Windows error semantics")
def test_windows_errors_preserve_winerror_and_subclass(tmp_path: Path):
    if sys.platform != "win32":
        raise AssertionError("Windows-only test ran on another platform")
    permission_error = file_io_module._windows_error(tmp_path / "data.txt", 5)
    length_error = file_io_module._windows_error(tmp_path / "data.txt", 206)

    assert isinstance(permission_error, PermissionError)
    assert permission_error.winerror == 5
    assert file_io_module._is_name_too_long(length_error, tmp_path / "data.txt")


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


@pytest.mark.skipif(os.name == "nt", reason="POSIX component length semantics")
def test_compact_temporary_path_never_aliases_destination(tmp_path: Path):
    path = tmp_path / f"qqqq0.{'a' * 249}"
    observed_temporary_path: Path | None = None
    destination_visible = False

    def write_temporary_file(temporary_path: Path) -> None:
        nonlocal destination_visible, observed_temporary_path
        observed_temporary_path = temporary_path
        destination_visible = path.exists()
        temporary_path.write_text("contents", encoding="utf-8")

    write_file_atomically(path, write_temporary_file)

    assert observed_temporary_path != path
    assert not destination_visible
    assert read_text_file(path) == "contents"


def test_compact_temporary_path_uses_distinct_filesystem_identity(tmp_path: Path):
    case_probe = tmp_path / "QdkCaseProbe"
    case_probe.write_text("probe", encoding="utf-8")
    if not (tmp_path / "qdkcaseprobe").exists():
        pytest.skip("Filesystem is case-sensitive")
    case_probe.unlink()
    path = tmp_path / f"Q{'q' * 14}0.{'a' * 230}"
    try:
        path.touch()
    except OSError:
        pytest.skip("Filesystem does not support the long test path")
    path.unlink()
    destination_visible = False

    def write_temporary_file(temporary_path: Path) -> None:
        nonlocal destination_visible
        destination_visible = path.exists()
        temporary_path.write_text("contents", encoding="utf-8")

    write_file_atomically(path, write_temporary_file)

    assert not destination_visible
    assert read_text_file(path) == "contents"


def test_reject_replaced_temporary_file(tmp_path: Path):
    path = tmp_path / "data.txt"
    replacement_path: Path | None = None

    def replace_temporary_file(temporary_path: Path) -> None:
        nonlocal replacement_path
        replacement_path = temporary_path
        temporary_path.unlink()
        temporary_path.write_text("replacement", encoding="utf-8")

    if os.name == "nt":
        with pytest.raises(PermissionError):
            write_file_atomically(path, replace_temporary_file)
    else:
        with pytest.raises(RuntimeError, match="Temporary file identity changed"):
            write_file_atomically(path, replace_temporary_file)

    assert not path.exists()
    if os.name != "nt":
        assert replacement_path is not None
        assert replacement_path.read_text(encoding="utf-8") == "replacement"


@pytest.mark.skipif(os.name == "nt", reason="POSIX hard-link semantics")
def test_clean_up_reserved_path_after_writer_adds_hard_link(tmp_path: Path):
    path = tmp_path / "data.txt"
    extra_link = tmp_path / "extra.txt"
    temporary_path: Path | None = None

    def fail_after_linking(reserved_path: Path) -> None:
        nonlocal temporary_path
        temporary_path = reserved_path
        reserved_path.write_text("sensitive", encoding="utf-8")
        os.link(reserved_path, extra_link)
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        write_file_atomically(path, fail_after_linking)

    assert temporary_path is not None
    assert not temporary_path.exists()
    assert extra_link.read_text(encoding="utf-8") == "sensitive"
    assert not path.exists()


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


@pytest.mark.skipif(
    os.name == "nt" or not all(hasattr(os, name) for name in ("fork", "openpty", "setsid")),
    reason="POSIX controlling-terminal semantics",
)
def test_reading_terminal_does_not_acquire_controlling_terminal():
    master_descriptor, slave_descriptor = os.openpty()
    slave_path = os.ttyname(slave_descriptor)
    child = os.fork()
    if child == 0:
        os.close(master_descriptor)
        os.close(slave_descriptor)
        try:
            os.setsid()
        except OSError:
            os._exit(2)
        try:
            read_text_file(slave_path)
        except OSError as error:
            if "not a regular file" not in str(error):
                os._exit(4)
        else:
            os._exit(3)
        try:
            terminal_descriptor = os.open(
                "/dev/tty",
                os.O_RDONLY | getattr(os, "O_NOCTTY", 0),
            )
        except OSError as error:
            os._exit(0 if error.errno == errno.ENXIO else 5)
        os.close(terminal_descriptor)
        os._exit(1)

    os.close(slave_descriptor)
    _, status = os.waitpid(child, 0)
    os.close(master_descriptor)

    assert os.waitstatus_to_exitcode(status) == 0


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
