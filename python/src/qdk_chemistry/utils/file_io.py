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
import time
from collections.abc import Callable
from ctypes import wintypes
from pathlib import Path
from typing import TypeAlias, cast

PathLike: TypeAlias = str | os.PathLike[str]
AtomicFileWriter: TypeAlias = Callable[[Path], None]

_DIRECTORY_CREATION_RETRY_DELAY_SECONDS = 0.001
_DIRECTORY_CREATION_RETRY_TIMEOUT_SECONDS = 0.1

__all__ = [
    "AtomicFileWriter",
    "PathLike",
    "ensure_parent_directory",
    "read_text_file",
    "write_file_atomically",
    "write_text_file_atomically",
]


class _TemporaryFileReservation:
    def __init__(
        self,
        path: Path,
        status: os.stat_result,
    ) -> None:
        self.descriptor = -1
        self.path = path
        self.status = status
        self.cleanup = True

    def adopt_descriptor(self, descriptor: int) -> None:
        self.descriptor = descriptor

    def take_descriptor(self) -> int:
        descriptor, self.descriptor = self.descriptor, -1
        return descriptor

    def disarm(self) -> None:
        self.cleanup = False

    def __del__(self) -> None:
        try:
            if self.descriptor >= 0:
                _close_descriptor(self.take_descriptor())
            if self.cleanup and _temporary_path_matches(self.path, self.status):
                _remove_temporary_file(self.path)
        except BaseException:  # noqa: BLE001
            pass


def _freeze_path(path_value: str) -> Path:
    destination = Path(path_value)
    if destination.is_absolute():
        return destination
    return Path(os.path.abspath(destination)) if os.name == "nt" else Path.cwd() / destination


def ensure_parent_directory(path: PathLike) -> None:
    """Create the parent directory of *path* when it does not exist.

    Relative paths are frozen to an absolute path before creation begins.
    """
    path_value = os.fspath(path)
    _validate_destination_path(path_value)
    if Path(path_value).parent == Path("."):
        return
    parent = _freeze_path(path_value).parent
    if os.name == "nt":
        parent.mkdir(parents=True, exist_ok=True)
    else:
        _create_private_directories(parent)


def _validate_destination_path(path: PathLike) -> None:
    value = os.fspath(path)
    separators = tuple(separator for separator in (os.sep, os.altsep) if separator)
    path_module = ntpath if sys.platform == "win32" else os.path
    final_component = path_module.basename(value)
    if "\0" in value:
        raise ValueError(f"Path contains an embedded NUL character: '{value}'")
    if not value or value.endswith(separators) or final_component in ("", ".", ".."):
        raise ValueError(f"Destination path must name a file: '{value}'")
    if sys.platform == "win32" and ":" in ntpath.splitdrive(value)[1]:
        raise ValueError(f"Windows alternate data streams are not supported: '{value}'")


def read_text_file(path: PathLike, *, encoding: str = "utf-8") -> str:
    """Read an entire text file without changing its line endings."""
    path_value = os.fspath(path)
    _validate_destination_path(path_value)
    if sys.platform == "win32":
        descriptor = _open_windows_file(
            path_value,
            desired_access=0x80000000,
            creation_disposition=3,
        )
    else:
        descriptor = os.open(
            path_value,
            os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOCTTY", 0),
        )
    operation_error: BaseException | None = None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f"Path is not a regular file: '{path_value}'")
        with os.fdopen(descriptor, "r", encoding=encoding, newline="", closefd=False) as stream:
            return stream.read()
    except BaseException as error:
        operation_error = error
        raise
    finally:
        close_error = _close_descriptor(descriptor)
        if close_error is not None:
            if operation_error is not None:
                raise operation_error from close_error
            raise close_error


def write_file_atomically(
    path: PathLike,
    writer: AtomicFileWriter,
    *,
    create_parent_directories: bool = False,
) -> None:
    """Write through a temporary sibling and atomically replace *path*.

    The writer receives an existing empty temporary file in the destination
    directory. It must write that file in place, close all writes before
    returning, and must not unlink, rename, replace, or hard-link the file.
    Cleanup is guaranteed only while the reserved file remains at the temporary
    path. The path preserves the destination's suffixes for format-sensitive
    writers.

    On POSIX, replacing an existing file preserves its ordinary read, write,
    and execute permission bits. Existing destination ACLs and extended
    attributes are not preserved; the replacement uses metadata inherited when
    its temporary file is created and may therefore grant broader access than
    the file it replaced. Callers that rely on explicit ACLs or extended
    attributes must reapply them after the write. New files are created with
    owner-only permissions. The filesystem must enforce POSIX permission bits;
    the write fails rather than publishing a file with broader mode bits.
    Platform ACLs are not inspected and may grant access beyond those bits. On
    Windows,
    replacement preserves the read-only attribute and new files use the
    filesystem's standard access controls. Existing Windows security
    descriptors and DACLs are not preserved; the replacement uses access
    controls inherited when its temporary file is created and may therefore
    grant broader access than the file it replaced. Callers that rely on
    explicit access-control entries must reapply them after the write.
    Read-only Windows destinations with multiple hard links are rejected.
    Other file-object metadata and hard-link identity are not preserved. The
    named temporary file also inherits the parent directory's access controls
    and may therefore be readable while the writer runs or after cleanup
    fails. Atomic replacement prevents partial visibility at the destination
    path but does not guarantee durability after power loss. Windows alternate
    data streams are not supported.

    The destination's parent directory and mutable ancestors must not be
    writable by principals less privileged than the process performing the
    write. Missing POSIX parent directories are created with owner-only
    permissions. Windows parent directories use inherited filesystem access
    controls.

    On POSIX, relative destinations are frozen to an absolute path before the
    writer runs. A relative path may therefore be rejected when its expanded
    absolute form exceeds the platform pathname limit.
    """
    path_value = os.fspath(path)
    _validate_destination_path(path_value)
    destination = _freeze_path(path_value)
    if create_parent_directories:
        ensure_parent_directory(destination)

    parent = destination.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"Parent directory does not exist for '{destination}'")

    descriptor = -1
    temporary_path: Path | None = None
    reserved_status: os.stat_result | None = None
    reservation: _TemporaryFileReservation | None = None

    try:
        reservation = (
            _reserve_temporary_file_with_parent_retry(destination)
            if create_parent_directories and os.name != "nt"
            else _reserve_temporary_file(destination)
        )
        descriptor = reservation.descriptor
        temporary_path = reservation.path
        reserved_status = reservation.status
        writer(temporary_path)
        current_status = temporary_path.lstat()
        if not _same_file_identity(reserved_status, current_status):
            raise RuntimeError(f"Temporary file identity changed: '{temporary_path}'")

        try:
            existing_status = destination.lstat()
        except FileNotFoundError:
            destination_mode = None
            if os.name != "nt":
                _set_permissions(descriptor, stat.S_IRUSR | stat.S_IWUSR, temporary_path)
        else:
            if stat.S_ISLNK(existing_status.st_mode):
                raise ValueError(f"Symlink destinations are not supported: '{destination}'")
            if not stat.S_ISREG(existing_status.st_mode):
                raise OSError(f"Destination is not a regular file: '{destination}'")
            existing_mode = stat.S_IMODE(existing_status.st_mode) & 0o777
            destination_mode = existing_mode
            if os.name != "nt" and hasattr(os, "fchmod"):
                _set_permissions(descriptor, existing_mode, temporary_path)
            elif os.name != "nt":
                temporary_path.chmod(existing_mode)

        if os.name == "nt":
            owned_descriptor, descriptor = reservation.take_descriptor(), -1
            close_error = _close_descriptor(owned_descriptor)
            if close_error is not None:
                raise close_error
        _replace_file(temporary_path, destination, destination_mode)
        if descriptor >= 0:
            owned_descriptor, descriptor = reservation.take_descriptor(), -1
            close_error = _close_descriptor(owned_descriptor)
            if close_error is not None:
                raise close_error
        reservation.disarm()
    except BaseException as error:
        if reserved_status is None and descriptor >= 0:
            try:
                reserved_status = os.fstat(descriptor)
            except OSError:
                reserved_status = None
        if descriptor >= 0:
            owned_descriptor = reservation.take_descriptor() if reservation is not None else descriptor
            descriptor = -1
            close_error = _close_descriptor(owned_descriptor)
        else:
            close_error = None
        if (
            temporary_path is not None
            and reserved_status is not None
            and _temporary_path_matches(temporary_path, reserved_status)
        ):
            try:
                _remove_temporary_file(temporary_path)
            except OSError as cleanup_error:
                raise error from cleanup_error
        if close_error is not None:
            raise error from close_error
        if reservation is not None:
            reservation.disarm()
        raise


def _reserve_temporary_file(destination: Path) -> _TemporaryFileReservation:
    """Reserve a private temporary sibling and keep its descriptor open."""
    suffix = "".join(destination.suffixes)
    for _ in range(64):
        temporary_path = destination.parent / f".qdk-tmp-{os.urandom(8).hex()}{suffix}"
        if _component_is_too_long(temporary_path):
            break
        try:
            reservation = _reserve_distinct_temporary_file(destination, temporary_path)
        except FileExistsError:
            continue
        except OSError as error:
            if not _is_name_too_long(error, destination):
                raise
            break
        if reservation is not None:
            return reservation

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
                reservation = _reserve_distinct_temporary_file(destination, temporary_path)
            except FileExistsError:
                continue
            except OSError as error:
                if _is_name_too_long(error, destination):
                    break
                raise
            if reservation is not None:
                return reservation

    raise FileExistsError(f"Could not create a unique temporary file beside '{destination}'")


def _package_reservation(
    descriptor: int,
    temporary_path: Path,
    reserved_status: os.stat_result,
) -> _TemporaryFileReservation:
    reservation: _TemporaryFileReservation | None = None
    try:
        reservation = _TemporaryFileReservation(temporary_path, reserved_status)
        reservation.adopt_descriptor(descriptor)
        return reservation
    except BaseException as error:
        if reservation is not None and reservation.descriptor >= 0:
            descriptor = reservation.take_descriptor()
            reservation.disarm()
        close_error = _close_descriptor(descriptor)
        if _temporary_path_matches(temporary_path, reserved_status):
            try:
                _remove_temporary_file(temporary_path)
            except OSError as cleanup_error:
                raise error from cleanup_error
        if close_error is not None:
            raise error from close_error
        raise


def _component_is_too_long(path: Path) -> bool:
    if os.name != "nt":
        return False
    return len(path.name.encode("utf-16-le", errors="surrogatepass")) // 2 > 255


def _create_exclusive_file(path: Path) -> int:
    if sys.platform != "win32":
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC, 0o600)
        try:
            _set_permissions(descriptor, stat.S_IRUSR | stat.S_IWUSR, path)
        except BaseException as error:
            try:
                reserved_status = os.fstat(descriptor)
            except OSError:
                reserved_status = None
            close_error = _close_descriptor(descriptor)
            if reserved_status is not None and _temporary_path_matches(path, reserved_status):
                try:
                    path.unlink()
                except OSError as cleanup_error:
                    raise error from cleanup_error
            if close_error is not None:
                raise error from close_error
            raise
        return descriptor

    return _open_windows_file(path, desired_access=0, creation_disposition=1)


def _open_windows_file(
    path: PathLike,
    *,
    desired_access: int,
    creation_disposition: int,
    flags_and_attributes: int = 0x00000080,
) -> int:
    if sys.platform != "win32":
        raise NotImplementedError("Windows file handles require Windows")

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
    close_handle = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    open_osfhandle = cast(
        "Callable[[int, int], int]",
        importlib.import_module("msvcrt").open_osfhandle,
    )
    handle = create_file(
        os.fspath(path),
        desired_access,
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        creation_disposition,
        flags_and_attributes,
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        error = ctypes.get_last_error()
        raise _windows_error(path, error)
    try:
        return open_osfhandle(
            cast("int", handle),
            os.O_RDONLY | getattr(os, "O_NOINHERIT", 0),
        )
    except BaseException as error:
        cleanup_error: OSError | None = None
        close_handle(handle)
        if creation_disposition == 1:
            try:
                _remove_temporary_file(Path(path))
            except OSError as caught_error:
                cleanup_error = caught_error
        if cleanup_error is not None:
            raise error from cleanup_error
        raise


def _windows_error(path: PathLike, error: int) -> OSError:
    if sys.platform != "win32":
        raise NotImplementedError("Windows errors require Windows")
    return OSError(0, ctypes.FormatError(error), os.fspath(path), error)


class _WindowsFileBasicInfo(ctypes.Structure):
    _fields_ = (
        ("creation_time", ctypes.c_longlong),
        ("last_access_time", ctypes.c_longlong),
        ("last_write_time", ctypes.c_longlong),
        ("change_time", ctypes.c_longlong),
        ("file_attributes", wintypes.DWORD),
    )


def _windows_file_info(descriptor: int, path: Path) -> _WindowsFileBasicInfo:
    if sys.platform != "win32":
        raise NotImplementedError("Windows file attributes require Windows")
    get_osfhandle = cast(
        "Callable[[int], int]",
        importlib.import_module("msvcrt").get_osfhandle,
    )
    handle = get_osfhandle(descriptor)
    info = _WindowsFileBasicInfo()
    get_info = ctypes.WinDLL("kernel32", use_last_error=True).GetFileInformationByHandleEx
    get_info.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    get_info.restype = wintypes.BOOL
    if not get_info(handle, 0, ctypes.byref(info), ctypes.sizeof(info)):
        raise _windows_error(path, ctypes.get_last_error())
    return info


def _normalized_windows_file_attributes(attributes: int) -> int:
    supported = 0x00000001 | 0x00000002 | 0x00000004 | 0x00000020 | 0x00000100 | 0x00001000 | 0x00002000
    result = attributes & supported
    return result or 0x00000080


def _set_windows_file_attributes(descriptor: int, attributes: int, path: Path) -> None:
    if sys.platform != "win32":
        raise NotImplementedError("Windows file attributes require Windows")
    get_osfhandle = cast(
        "Callable[[int], int]",
        importlib.import_module("msvcrt").get_osfhandle,
    )
    handle = get_osfhandle(descriptor)
    info = _windows_file_info(descriptor, path)
    info.file_attributes = _normalized_windows_file_attributes(attributes)
    set_info = ctypes.WinDLL("kernel32", use_last_error=True).SetFileInformationByHandle
    set_info.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    set_info.restype = wintypes.BOOL
    if not set_info(handle, 0, ctypes.byref(info), ctypes.sizeof(info)):
        raise _windows_error(path, ctypes.get_last_error())


def _reserve_distinct_temporary_file(
    destination: Path,
    temporary_path: Path,
) -> _TemporaryFileReservation | None:
    descriptor = _create_exclusive_file(temporary_path)
    reserved_status: os.stat_result | None = None
    try:
        reserved_status = os.fstat(descriptor)
        destination_matches = _path_matches_identity(
            destination,
            reserved_status,
            require_single_link=False,
        )
    except BaseException as error:
        if reserved_status is None:
            try:
                reserved_status = os.fstat(descriptor)
            except OSError:
                reserved_status = None
        close_error = _close_descriptor(descriptor)
        if reserved_status is not None and _path_matches_identity(
            temporary_path,
            reserved_status,
            require_single_link=False,
        ):
            try:
                _remove_temporary_file(temporary_path)
            except OSError as cleanup_error:
                raise error from cleanup_error
        if close_error is not None:
            raise error from close_error
        raise
    assert reserved_status is not None
    if not destination_matches:
        return _package_reservation(descriptor, temporary_path, reserved_status)

    close_error = _close_descriptor(descriptor)
    if _path_matches_identity(temporary_path, reserved_status, require_single_link=False):
        _remove_temporary_file(temporary_path)
    if close_error is not None:
        raise close_error
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


def _path_matches_identity(
    path: Path,
    reserved_status: os.stat_result,
    *,
    require_single_link: bool,
) -> bool:
    return (
        _path_identity_state(
            path,
            reserved_status,
            require_single_link=require_single_link,
        )
        is True
    )


def _path_identity_state(
    path: Path,
    reserved_status: os.stat_result,
    *,
    require_single_link: bool,
) -> bool | None:
    try:
        current_status = path.lstat()
    except (OSError, ValueError):
        return None
    return (
        reserved_status.st_dev == current_status.st_dev
        and reserved_status.st_ino == current_status.st_ino
        and stat.S_ISREG(current_status.st_mode)
        and (not require_single_link or current_status.st_nlink == 1)
    )


def _temporary_path_matches(temporary_path: Path, reserved_status: os.stat_result) -> bool:
    return _path_matches_identity(temporary_path, reserved_status, require_single_link=False)


def _close_descriptor(descriptor: int) -> OSError | None:
    try:
        os.close(descriptor)
    except OSError as error:
        return error
    return None


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


def _replace_file(
    temporary_path: Path,
    destination: Path,
    destination_mode: int | None,
) -> None:
    """Replace a destination, handling Windows read-only files."""
    original_descriptor = -1
    original_attributes: int | None = None
    operation_error: BaseException | None = None
    read_only = False
    if os.name == "nt" and destination_mode is not None:
        try:
            original_descriptor = _open_windows_file(
                destination,
                desired_access=0x00000080 | 0x00000100,
                creation_disposition=3,
                flags_and_attributes=0x00000080 | 0x00200000,
            )
        except OSError:
            original_descriptor = -1
        else:
            try:
                original_attributes = int(_windows_file_info(original_descriptor, destination).file_attributes)
            except BaseException as error:
                owned_descriptor, original_descriptor = original_descriptor, -1
                close_error = _close_descriptor(owned_descriptor)
                if close_error is not None:
                    raise error from close_error
                raise
    try:
        if os.name == "nt" and destination_mode is not None:
            read_only = (
                original_attributes & 0x00000001 != 0
                if original_attributes is not None
                else destination_mode & stat.S_IWRITE == 0
            )
            _set_windows_path_attributes(
                temporary_path,
                0x00000001 if read_only else 0x00000080,
            )
        try:
            os.replace(temporary_path, destination)
            return
        except PermissionError as replace_error:
            if (
                not read_only
                or getattr(replace_error, "winerror", None) != 5
                or original_descriptor < 0
                or original_attributes is None
            ):
                raise

        if os.fstat(original_descriptor).st_nlink != 1:
            raise RuntimeError(
                f"Read-only Windows destinations with multiple hard links are not supported: '{destination}'"
            )
        _set_windows_file_attributes(
            original_descriptor,
            original_attributes & ~0x00000001,
            destination,
        )
        try:
            os.replace(temporary_path, destination)
        except BaseException as replace_error:
            try:
                _set_windows_file_attributes(original_descriptor, original_attributes, destination)
            except OSError as rollback_error:
                raise replace_error from rollback_error
            raise
        try:
            displaced_status = os.fstat(original_descriptor)
        except OSError:
            pass
        else:
            if displaced_status.st_nlink > 0:
                _set_windows_file_attributes(
                    original_descriptor,
                    original_attributes,
                    destination,
                )
    except BaseException as error:
        operation_error = error
        raise
    finally:
        if original_descriptor >= 0:
            owned_descriptor, original_descriptor = original_descriptor, -1
            close_error = _close_descriptor(owned_descriptor)
            if close_error is not None:
                if operation_error is not None:
                    raise operation_error from close_error
                raise close_error


def _set_windows_path_attributes(path: Path, attributes: int) -> None:
    if sys.platform != "win32":
        raise NotImplementedError("Windows file attributes require Windows")
    set_attributes = ctypes.WinDLL("kernel32", use_last_error=True).SetFileAttributesW
    set_attributes.argtypes = (wintypes.LPCWSTR, wintypes.DWORD)
    set_attributes.restype = wintypes.BOOL
    if not set_attributes(os.fspath(path), attributes):
        raise _windows_error(path, ctypes.get_last_error())


def _set_permissions(descriptor: int, mode: int, path: Path) -> None:
    os.fchmod(descriptor, mode)
    actual_mode = stat.S_IMODE(os.fstat(descriptor).st_mode)
    if actual_mode != mode:
        raise PermissionError(
            errno.EPERM,
            f"Filesystem did not apply permissions {mode:#o} to '{path}'",
            os.fspath(path),
        )


def _create_private_directories_once(directory: Path) -> None:
    missing: list[Path] = []
    current = directory
    while not current.is_dir():
        missing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent

    for missing_directory in reversed(missing):
        try:
            os.mkdir(missing_directory, 0o700)
        except FileExistsError:
            if not missing_directory.is_dir():
                raise
            continue
        descriptor = -1
        try:
            os.chmod(missing_directory, 0o700)
            descriptor = os.open(
                missing_directory,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            _set_permissions(descriptor, 0o700, missing_directory)
        except BaseException as error:
            close_error = None
            if descriptor >= 0:
                close_error = _close_descriptor(descriptor)
            try:
                missing_directory.rmdir()
            except OSError as cleanup_error:
                raise error from cleanup_error
            if close_error is not None:
                raise error from close_error
            raise
        close_error = _close_descriptor(descriptor)
        if close_error is not None:
            raise close_error


def _directory_initialization_state(
    directory: Path,
) -> tuple[bool, tuple[tuple[str, int, int, int], ...]]:
    initializing = False
    state: list[tuple[str, int, int, int]] = []
    current = directory
    while True:
        try:
            status = current.stat()
        except FileNotFoundError:
            state.append((os.fspath(current), -1, -1, errno.ENOENT))
        except PermissionError:
            state.append((os.fspath(current), -1, -1, errno.EACCES))
        else:
            permissions = status.st_mode & 0o777
            state.append((os.fspath(current), status.st_dev, status.st_ino, permissions))
            initializing = initializing or (
                stat.S_ISDIR(status.st_mode)
                and status.st_uid == os.geteuid()
                and permissions & (stat.S_IWUSR | stat.S_IXUSR) != (stat.S_IWUSR | stat.S_IXUSR)
            )
        parent = current.parent
        if parent == current:
            return initializing, tuple(state)
        current = parent


def _create_private_directories(directory: Path) -> None:
    deadline = time.monotonic() + _DIRECTORY_CREATION_RETRY_TIMEOUT_SECONDS
    previous_state: tuple[tuple[str, int, int, int], ...] | None = None
    while True:
        permission_error: PermissionError | None = None
        try:
            _create_private_directories_once(directory)
            return
        except PermissionError as error:
            permission_error = error

        if time.monotonic() >= deadline:
            assert permission_error is not None
            raise permission_error
        initializing, current_state = _directory_initialization_state(directory)
        if previous_state is None or current_state != previous_state:
            previous_state = current_state
            time.sleep(_DIRECTORY_CREATION_RETRY_DELAY_SECONDS)
            continue
        if initializing:
            time.sleep(_DIRECTORY_CREATION_RETRY_DELAY_SECONDS)
            continue
        raise permission_error


def _reserve_temporary_file_with_parent_retry(destination: Path) -> _TemporaryFileReservation:
    deadline = time.monotonic() + _DIRECTORY_CREATION_RETRY_TIMEOUT_SECONDS
    previous_state: tuple[tuple[str, int, int, int], ...] | None = None
    while True:
        try:
            return _reserve_temporary_file(destination)
        except PermissionError as error:
            if time.monotonic() >= deadline:
                raise
            initializing, current_state = _directory_initialization_state(destination.parent)
            if previous_state is None or current_state != previous_state:
                previous_state = current_state
                time.sleep(_DIRECTORY_CREATION_RETRY_DELAY_SECONDS)
                continue
            if initializing:
                time.sleep(_DIRECTORY_CREATION_RETRY_DELAY_SECONDS)
                continue
            raise error


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
