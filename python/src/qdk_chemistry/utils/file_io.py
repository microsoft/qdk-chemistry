"""Cross-platform file and path helpers."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

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
    parent = Path(path).parent
    if parent != Path("."):
        parent.mkdir(parents=True, exist_ok=True)


def read_text_file(path: PathLike, *, encoding: str = "utf-8") -> str:
    """Read an entire text file without changing its line endings."""
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

    On POSIX, replacing an existing file preserves its permission bits and new
    files are created with owner-only permissions. On Windows, replacement
    preserves the read-only attribute and new files use the filesystem's
    standard access controls. Other file-object metadata and hard-link identity
    are not preserved. Atomic replacement prevents partial visibility but does
    not guarantee durability after power loss.

    The destination's parent directory must not be writable by principals less
    privileged than the process performing the write.
    """
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

    try:
        writer(temporary_path)
        reserved_status = os.fstat(descriptor)
        current_status = temporary_path.lstat()
        if (
            reserved_status.st_dev != current_status.st_dev
            or reserved_status.st_ino != current_status.st_ino
            or not stat.S_ISREG(current_status.st_mode)
            or current_status.st_nlink != 1
        ):
            raise RuntimeError(f"Temporary file identity changed: '{temporary_path}'")

        try:
            existing_mode = stat.S_IMODE(destination.stat().st_mode)
        except FileNotFoundError:
            destination_mode = None
            if os.name != "nt":
                os.fchmod(descriptor, stat.S_IRUSR | stat.S_IWUSR)
        else:
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
        if descriptor >= 0:
            os.close(descriptor)
            descriptor = -1
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
        try:
            descriptor = os.open(temporary_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError:
            continue
        except OSError as error:
            if error.errno != errno.ENAMETOOLONG:
                raise
            break
        return descriptor, str(temporary_path)

    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_-"
    for stem_length in range(16, 0, -1):
        for attempt in range(64):
            stem = "q" * (stem_length - 1) + alphabet[attempt]
            temporary_path = destination.parent / f"{stem}{suffix}"
            try:
                descriptor = os.open(temporary_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            except FileExistsError:
                continue
            except OSError as error:
                if error.errno == errno.ENAMETOOLONG:
                    break
                raise
            return descriptor, str(temporary_path)

    raise FileExistsError(f"Could not create a unique temporary file beside '{destination}'")


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
    except BaseException:
        destination.chmod(destination_mode)
        raise
    destination.chmod(destination_mode)


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
