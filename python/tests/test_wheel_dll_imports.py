"""Tests for packaged native dependencies in the Windows wheel audit."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile

import pytest


@pytest.mark.parametrize("extension", ["dll", "pyd"])
@pytest.mark.parametrize("delayed", [False, True])
@pytest.mark.parametrize("packaged", [False, True])
def test_wheel_native_dependency_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extension: str,
    delayed: bool,
    packaged: bool,
) -> None:
    """Resolve regular and delay-loaded dependencies only when their binaries are shipped."""
    script = Path(__file__).resolve().parents[2] / ".pipelines" / "pip-scripts" / "check-wheel-dll-imports.py"
    spec = importlib.util.spec_from_file_location("qdk_wheel_dll_imports_test_module", script)
    assert spec is not None
    assert spec.loader is not None
    audit = importlib.util.module_from_spec(spec)
    # PE parsing is mocked; these cases need neither pefile nor native binaries.
    monkeypatch.setitem(sys.modules, "pefile", ModuleType("pefile"))
    spec.loader.exec_module(audit)

    dependency = f"Helper.{extension}"
    wheel = tmp_path / "test.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("qdk_chemistry/_core.pyd", b"core")
        if packaged:
            archive.writestr(f"qdk_chemistry/{dependency}", b"helper")

    def imports(binary: bytes) -> tuple[list[str], list[str]]:
        if binary == b"helper":
            return [], []
        assert binary == b"core"
        if delayed:
            return ["VCRUNTIME140.DLL"], [dependency.upper()]
        return ["VCRUNTIME140.DLL", dependency.upper()], []

    monkeypatch.setattr(audit, "_imports", imports)
    monkeypatch.setattr(sys, "argv", [str(script), str(wheel)])

    assert audit._main() == (0 if packaged else 1)
    output = capsys.readouterr().out
    if packaged:
        assert "All native DLL imports are supplied" in output
    else:
        assert "ERROR: wheel has DLL imports not supplied" in output
        assert dependency.upper() in output
