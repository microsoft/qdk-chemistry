"""Cross-platform file and path helpers."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import ctypes
import errno
import importlib
import ntpath
import os
import stat
import sys
from collections.abc import Callable
from ctypes import wintypes
from pathlib import Path
from typing import TypeAlias, cast

PathLike: TypeAlias = str | os.PathLike[str]
AtomicFileWriter: TypeAlias = Callable[[Path], None]

__all__ = [
    "AtomicFileWriter",
    "PathLike",
    "ensure_parent_directory",
    "read_text_file",
    "write_file_atomically",
    "write_text_file_atomically",
]


def ensure_parent_directory(path: PathLike) -> None:
    """Create the parent directory of *path* when it does not exist."""
    _validate_destination_path(path)
    parent = Path(path).parent
    if parent != Path("."):
        parent.mkdir(parents=True, exist_ok=True)


def _validate_destination_path(path: PathLike) -> None:
    value = os.fspath(path)
    separators = tuple(separator for separator in (os.sep, os.altsep) if separator)
    if value.endswith(separators):
        raise ValueError(f"Destination path must name a file: '{value}'")
    if sys.platform == "win32" and ":" in ntpath.splitdrive(value)[1]:
        raise ValueError(f"Windows alternate data streams are not supported: '{value}'")


def read_text_file(path: PathLike, *, encoding: str = "utf-8") -> str:
    """Read an entire text file without changing its line endings."""
    _validate_destination_path(path)
    if sys.platform == "win32":
        descriptor = _open_windows_file(
            path,
            desired_access=0x80000000,
            creation_disposition=3,
        )
    else:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f"Path is not a regular file: '{path}'")
        with os.fdopen(descriptor, "r", encoding=encoding, newline="", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def write_file_atomically(
    path: PathLike,
    writer: AtomicFileWriter,
    *,
    create_parent_directories: bool = False,
) -> None:
    """Write through a temporary sibling and atomically replace *path*.

    The writer receives a unique temporary path in the destination directory.
    The path preserves the destination's suffixes for format-sensitive writers.
    The temporary file is removed if the writer raises an exception.

    On POSIX, replacing an existing file preserves its ordinary read, write,
    and execute permission bits. New files are created with owner-only
    permissions. On Windows, replacement preserves the read-only attribute and
    new files use the filesystem's standard access controls. Other file-object
    metadata and hard-link identity are not preserved. Atomic replacement
    prevents partial visibility but does not guarantee durability after power
    loss.

    The destination's parent directory must not be readable or writable by
    principals less privileged than the process performing the write.
    """
    _validate_destination_path(path)
    destination = Path(path)
    if not destination.is_absolute():
        destination = Path(os.path.abspath(destination)) if os.name == "nt" else Path.cwd() / destination
    if create_parent_directories:
        ensure_parent_directory(destination)

    parent = destination.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"Parent directory does not exist for '{destination}'")

    descriptor, temporary_name = _reserve_temporary_file(destination)
    temporary_path = Path(temporary_name)
    reserved_status: os.stat_result | None = None

    try:
        writer(temporary_path)
        reserved_status = os.fstat(descriptor)
        current_status = temporary_path.lstat()
        if not _same_file_identity(reserved_status, current_status):
            raise RuntimeError(f"Temporary file identity changed: '{temporary_path}'")

        try:
            destination_status = destination.lstat()
        except FileNotFoundError:
            destination_mode = None
            if os.name != "nt":
                os.fchmod(descriptor, stat.S_IRUSR | stat.S_IWUSR)
        else:
            if stat.S_ISLNK(destination_status.st_mode):
                raise ValueError(f"Symlink destinations are not supported: '{destination}'")
            existing_mode = stat.S_IMODE(destination_status.st_mode) & 0o777
            destination_mode = existing_mode
            if os.name != "nt" and hasattr(os, "fchmod"):
                os.fchmod(descriptor, existing_mode)
            elif os.name != "nt":
                temporary_path.chmod(existing_mode)

        if os.name == "nt":
            os.close(descriptor)
            descriptor = -1
        _replace_file(temporary_path, destination, destination_mode)
    except BaseException as error:
        if reserved_status is None and descriptor >= 0:
            try:
                reserved_status = os.fstat(descriptor)
            except OSError:
                reserved_status = None
        if descriptor >= 0:
            os.close(descriptor)
            descriptor = -1
        if reserved_status is not None and _temporary_path_matches(temporary_path, reserved_status):
            try:
                _remove_temporary_file(temporary_path)
            except OSError as cleanup_error:
                raise error from cleanup_error
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _reserve_temporary_file(destination: Path) -> tuple[int, str]:
    """Reserve a private temporary sibling and keep its descriptor open."""
    suffix = "".join(destination.suffixes)
    for _ in range(64):
        temporary_path = destination.parent / f".qdk-tmp-{os.urandom(8).hex()}{suffix}"
        if _component_is_too_long(temporary_path):
            break
        try:
            descriptor = _reserve_distinct_temporary_file(destination, temporary_path)
        except FileExistsError:
            continue
        except OSError as error:
            if not _is_name_too_long(error, destination):
                raise
            break
        if descriptor is not None:
            return descriptor, str(temporary_path)

    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_-"
    for stem_length in range(16, 0, -1):
        for attempt in range(64):
            stem = "q" * (stem_length - 1) + alphabet[attempt]
            temporary_path = destination.parent / f"{stem}{suffix}"
            if temporary_path == destination:
                continue
            if _component_is_too_long(temporary_path):
                break
            try:
                descriptor = _reserve_distinct_temporary_file(destination, temporary_path)
            except FileExistsError:
                continue
            except OSError as error:
                if _is_name_too_long(error, destination):
                    break
                raise
            if descriptor is not None:
                return descriptor, str(temporary_path)

    raise FileExistsError(f"Could not create a unique temporary file beside '{destination}'")


def _component_is_too_long(path: Path) -> bool:
    if os.name != "nt":
        return False
    return len(path.name.encode("utf-16-le")) // 2 > 255


def _create_exclusive_file(path: Path) -> int:
    if sys.platform != "win32":
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC, 0o600)
        try:
            os.fchmod(descriptor, stat.S_IRUSR | stat.S_IWUSR)
        except BaseException as error:
            try:
                reserved_status = os.fstat(descriptor)
            except OSError:
                reserved_status = None
            finally:
                os.close(descriptor)
            if reserved_status is not None and _temporary_path_matches(path, reserved_status):
                try:
                    path.unlink()
                except OSError as cleanup_error:
                    raise error from cleanup_error
            raise
        return descriptor

    return _open_windows_file(path, desired_access=0, creation_disposition=1)


def _open_windows_file(
    path: PathLike,
    *,
    desired_access: int,
    creation_disposition: int,
) -> int:

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
        os.fspath(path),
        desired_access,
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        creation_disposition,
        0x00000080,
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        error = ctypes.get_last_error()
        raise _windows_error(path, error)
    close_handle = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    open_osfhandle = cast(
        "Callable[[int, int], int]",
        importlib.import_module("msvcrt").open_osfhandle,
    )
    try:
        return open_osfhandle(
            cast("int", handle),
            os.O_RDONLY | getattr(os, "O_NOINHERIT", 0),
        )
    except BaseException:
        close_handle(handle)
        raise


def _windows_error(path: PathLike, error: int) -> OSError:
    return OSError(0, ctypes.FormatError(error), os.fspath(path), error)


def _reserve_distinct_temporary_file(destination: Path, temporary_path: Path) -> int | None:
    descriptor = _create_exclusive_file(temporary_path)
    try:
        reserved_status = os.fstat(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    if not _temporary_path_matches(destination, reserved_status):
        return descriptor

    os.close(descriptor)
    if _temporary_path_matches(temporary_path, reserved_status):
        _remove_temporary_file(temporary_path)
    return None


def _is_name_too_long(error: OSError, destination: Path) -> bool:
    if error.errno == errno.ENAMETOOLONG:
        return True
    if sys.platform != "win32":
        return False
    winerror = getattr(error, "winerror", None)
    if winerror in (111, 206):
        return True
    return winerror == 3 and destination.parent.is_dir()


def _same_file_identity(reserved_status: os.stat_result, current_status: os.stat_result) -> bool:
    return (
        reserved_status.st_dev == current_status.st_dev
        and reserved_status.st_ino == current_status.st_ino
        and stat.S_ISREG(current_status.st_mode)
        and current_status.st_nlink == 1
    )


def _temporary_path_matches(temporary_path: Path, reserved_status: os.stat_result) -> bool:
    try:
        current_status = temporary_path.lstat()
    except OSError:
        return False
    return _same_file_identity(reserved_status, current_status)


def _remove_temporary_file(temporary_path: Path) -> None:
    """Remove a temporary file, including a Windows read-only file."""
    try:
        temporary_path.unlink(missing_ok=True)
    except PermissionError:
        if os.name != "nt":
            raise
        try:
            mode = temporary_path.stat().st_mode
        except FileNotFoundError:
            return
        temporary_path.chmod(mode | stat.S_IWRITE)
        temporary_path.unlink(missing_ok=True)


def _replace_file(temporary_path: Path, destination: Path, destination_mode: int | None) -> None:
    """Replace a destination, handling Windows read-only files."""
    if os.name == "nt" and destination_mode is not None:
        temporary_path.chmod(destination_mode)
    try:
        os.replace(temporary_path, destination)
        return
    except PermissionError:
        read_only = os.name == "nt" and destination_mode is not None and destination_mode & stat.S_IWRITE == 0
        if not read_only:
            raise

    assert destination_mode is not None
    destination.chmod(destination_mode | stat.S_IWRITE)
    try:
        os.replace(temporary_path, destination)
    except BaseException as replace_error:
        try:
            destination.chmod(destination_mode)
        except OSError as rollback_error:
            raise replace_error from rollback_error
        raise


def write_text_file_atomically(
    path: PathLike,
    contents: str,
    *,
    encoding: str = "utf-8",
    create_parent_directories: bool = False,
) -> None:
    """Write text through an atomic file replacement."""

    def write_temporary_file(temporary_path: Path) -> None:
        with temporary_path.open("w", encoding=encoding, newline="") as stream:
            stream.write(contents)

    write_file_atomically(
        path,
        write_temporary_file,
        create_parent_directories=create_parent_directories,
    )
