"""Verify inlined ``from_xyz(...)`` blobs in docs examples match their .xyz files.

The documentation examples hard-code molecular geometries inline so that the
rendered snippet, the executed code, and the tested code are identical.  Each
inlined blob is tagged with a hidden ``docs:xyz <path>`` comment (placed outside
the rendered cell window) pointing at the canonical ``.xyz`` file.  This test
keeps the two in sync: if a ``.xyz`` file changes, the inline blob must be
updated to match (and vice versa).
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import re
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "docs" / "source" / "_static" / "examples"

_TAG_RE = re.compile(r"(?:#|//)\s*docs:xyz\s+(\S+)")
_PY_BLOB_RE = re.compile(r'from_xyz\("""\\?\n(.*?)"""', re.DOTALL)
_CPP_BLOB_RE = re.compile(r'from_xyz\(R"\((.*?)\)"\)', re.DOTALL)


def _example_files() -> list[Path]:
    """Return all Python and C++ example source files."""
    return [p for p in sorted(EXAMPLES_DIR.rglob("*")) if p.suffix.lower() in {".py", ".cpp"}]


def _inlined_examples() -> list[Path]:
    """Example files that contain at least one ``docs:xyz`` tag."""
    return [p for p in _example_files() if _TAG_RE.search(p.read_text(encoding="utf-8"))]


def test_some_examples_are_inlined():
    """Guard against the regexes silently matching nothing."""
    assert _inlined_examples(), "no docs examples with inlined from_xyz blobs were found"


@pytest.mark.parametrize("path", _inlined_examples(), ids=lambda p: str(p.relative_to(EXAMPLES_DIR)))
def test_inline_blobs_match_xyz_files(path: Path):
    """Each inlined from_xyz blob must match its tagged .xyz file byte-for-byte."""
    text = path.read_text(encoding="utf-8")
    tags = _TAG_RE.findall(text)
    blob_re = _PY_BLOB_RE if path.suffix.lower() == ".py" else _CPP_BLOB_RE
    blobs = blob_re.findall(text)

    assert len(tags) == len(blobs), f"{path}: {len(tags)} docs:xyz tag(s) but {len(blobs)} from_xyz blob(s)"

    data_root = (EXAMPLES_DIR / "data").resolve()
    for rel, blob in zip(tags, blobs, strict=True):
        xyz_file = (path.parent / rel).resolve()
        assert xyz_file.is_relative_to(data_root), f"{path}: docs:xyz target must be under {data_root}: {rel}"
        assert xyz_file.is_file(), f"{path}: docs:xyz target not found: {rel}"
        expected = xyz_file.read_text(encoding="utf-8").rstrip("\n")
        assert blob.rstrip("\n") == expected, f"{path}: inline blob does not match {rel}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
