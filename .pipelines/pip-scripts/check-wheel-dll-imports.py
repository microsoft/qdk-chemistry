"""Reject Windows wheels with unresolved native DLL dependencies."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import re
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

import pefile

_ALLOWED_SYSTEM_DLLS = frozenset(
    {
        "ADVAPI32.DLL",
        "BCRYPT.DLL",
        "COMBASE.DLL",
        "CRYPT32.DLL",
        "GDI32.DLL",
        "IPHLPAPI.DLL",
        "KERNEL32.DLL",
        "KERNELBASE.DLL",
        "NORMALIZ.DLL",
        "NTDLL.DLL",
        "OLE32.DLL",
        "OLEAUT32.DLL",
        "RPCRT4.DLL",
        "SECUR32.DLL",
        "SHELL32.DLL",
        "SHLWAPI.DLL",
        "UCRTBASE.DLL",
        "USER32.DLL",
        "USERENV.DLL",
        "VERSION.DLL",
        "WINHTTP.DLL",
        "WINMM.DLL",
        "WS2_32.DLL",
    }
)
_PYTHON_DLL = re.compile(r"PYTHON\d+(?:_D)?\.DLL")
_VC_REDIST_DLLS = frozenset(
    {
        "CONCRT140.DLL",
        "MSVCP140.DLL",
        "MSVCP140_1.DLL",
        "MSVCP140_2.DLL",
        "MSVCP140_ATOMIC_WAIT.DLL",
        "MSVCP140_CODECVT_IDS.DLL",
        "VCRUNTIME140.DLL",
        "VCRUNTIME140_1.DLL",
    }
)


def _imports(binary: bytes) -> tuple[list[str], list[str]]:
    """Return regular and delay-loaded DLL imports from a PE image."""
    try:
        image = pefile.PE(data=binary, fast_load=False)
    except pefile.PEFormatError as exc:
        raise ValueError(f"invalid PE image: {exc}") from exc

    regular: set[str] = set()
    delayed: set[str] = set()
    try:
        for entry in getattr(image, "DIRECTORY_ENTRY_IMPORT", ()):
            regular.add(entry.dll.decode("ascii").upper())
        for entry in getattr(image, "DIRECTORY_ENTRY_DELAY_IMPORT", ()):
            delayed.add(entry.dll.decode("ascii").upper())
    finally:
        image.close()
    return sorted(regular), sorted(delayed)


def _is_prerequisite_dll(dll: str) -> bool:
    """Return whether the documented Windows prerequisites supply a DLL."""
    return (
        dll in _ALLOWED_SYSTEM_DLLS
        or dll in _VC_REDIST_DLLS
        or dll.startswith(("API-MS-WIN-", "EXT-MS-WIN-"))
        or _PYTHON_DLL.fullmatch(dll) is not None
    )


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="Windows wheel to inspect")
    return parser.parse_args()


def _main() -> int:
    """Validate the wheel and return a process exit code."""
    wheel = _parse_args().wheel
    try:
        with ZipFile(wheel) as archive:
            binaries = [
                name
                for name in archive.namelist()
                if name.lower().endswith((".dll", ".pyd"))
            ]
            extensions = [
                name
                for name in binaries
                if PurePosixPath(name).name.startswith("_core.")
                and name.lower().endswith(".pyd")
            ]
            if len(extensions) != 1:
                print(
                    f"ERROR: expected exactly one _core*.pyd in {wheel}, found {len(extensions)}"
                )
                return 1
            imports = {name: _imports(archive.read(name)) for name in binaries}
    except (BadZipFile, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: cannot inspect {wheel}: {exc}")
        return 1

    packaged_dlls = {
        PurePosixPath(name).name.upper()
        for name in binaries
        if name.lower().endswith(".dll")
    }
    unresolved: list[tuple[str, str]] = []
    for name, (regular, delayed) in imports.items():
        print(f"DLL imports for {name}:")
        for dll in regular:
            print(f"  {dll}")
            if not _is_prerequisite_dll(dll) and dll not in packaged_dlls:
                unresolved.append((name, dll))
        for dll in delayed:
            print(f"  {dll} (delay-loaded)")
            if not _is_prerequisite_dll(dll) and dll not in packaged_dlls:
                unresolved.append((name, dll))

    core_regular, _ = imports[extensions[0]]
    if "VCRUNTIME140.DLL" not in core_regular:
        print("ERROR: _core is not linked to CPython's dynamic MSVC runtime.")
        return 1

    if unresolved:
        print(
            "ERROR: wheel has DLL imports not supplied by Windows, CPython, "
            "the Visual C++ Redistributable, or the wheel:"
        )
        for name, dll in unresolved:
            print(f"  {name}: {dll}")
        return 1

    print(
        "All native DLL imports are supplied by Windows, CPython, "
        "the Visual C++ Redistributable, or the wheel."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
