"""Module lookup and error reporting for stub generation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import subprocess
from pathlib import Path

import pytest

from qdk_chemistry.utils._stubgen import run_stub_generator


def test_target_directory_cannot_shadow_generator(tmp_path, monkeypatch):
    """The installed generator remains authoritative while the target stays importable."""
    target = tmp_path / "package with spaces"
    target.mkdir()
    marker = tmp_path / "shadowed"
    (target / "pybind11_stubgen.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('shadowed')\n"
    )
    (target / "_core.py").write_text("value = 'extension target'\n")
    installed = tmp_path / "installed"
    installed.mkdir()
    (installed / "pybind11_stubgen.py").write_text(
        "def main():\n"
        "    import sys\n"
        "    from pathlib import Path\n"
        "    import _core\n"
        "    output = Path(sys.argv[sys.argv.index('-o') + 1])\n"
        "    (output / 'generated.pyi').write_text(_core.value)\n"
        "    (output / 'working-directory.txt').write_text(str(Path.cwd()))\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(installed))

    run_stub_generator(target)

    assert not marker.exists()
    assert (target / "generated.pyi").read_text() == "extension target"
    working_dir = Path((target / "working-directory.txt").read_text())
    assert working_dir != target
    assert not working_dir.exists()


def test_generator_failure_is_reported(tmp_path, monkeypatch):
    """Generator failures propagate rather than appearing to succeed."""
    target = tmp_path / "package"
    target.mkdir()
    installed = tmp_path / "installed"
    installed.mkdir()
    (installed / "pybind11_stubgen.py").write_text(
        "def main():\n    import sys\n    print('generation failed', file=sys.stderr)\n    sys.exit(7)\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(installed))

    with pytest.raises(subprocess.CalledProcessError) as error:
        run_stub_generator(target)

    assert error.value.returncode == 7
    assert "generation failed" in error.value.stderr
