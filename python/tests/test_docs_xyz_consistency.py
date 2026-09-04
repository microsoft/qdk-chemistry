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
from collections.abc import Callable
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


def _find_mismatches(text: str, suffix: str, read_xyz: Callable[[str], str]) -> list[str]:
    """Return descriptions of inlined blobs that disagree with their .xyz data.

    ``read_xyz`` maps a ``docs:xyz`` relative path to the canonical file
    contents.  An empty list means every inlined blob matches its tag.  This is
    the core check, factored out so it can be exercised on deliberately broken
    input by the smoke tests below.
    """
    tags = _TAG_RE.findall(text)
    blob_re = _PY_BLOB_RE if suffix == ".py" else _CPP_BLOB_RE
    blobs = blob_re.findall(text)
    if len(tags) != len(blobs):
        return [f"{len(tags)} docs:xyz tag(s) but {len(blobs)} from_xyz blob(s)"]
    return [
        f"inline blob does not match {rel}"
        for rel, blob in zip(tags, blobs, strict=True)
        if blob.rstrip("\n") != read_xyz(rel).rstrip("\n")
    ]


def test_some_examples_are_inlined():
    """Guard against the regexes silently matching nothing."""
    assert _inlined_examples(), "no docs examples with inlined from_xyz blobs were found"


@pytest.mark.parametrize("path", _inlined_examples(), ids=lambda p: str(p.relative_to(EXAMPLES_DIR)))
def test_inline_blobs_match_xyz_files(path: Path):
    """Each inlined from_xyz blob must match its tagged .xyz file byte-for-byte."""
    text = path.read_text(encoding="utf-8")
    data_root = (EXAMPLES_DIR / "data").resolve()

    def read_xyz(rel: str) -> str:
        xyz_file = (path.parent / rel).resolve()
        assert xyz_file.is_relative_to(data_root), f"{path}: docs:xyz target must be under {data_root}: {rel}"
        assert xyz_file.is_file(), f"{path}: docs:xyz target not found: {rel}"
        return xyz_file.read_text(encoding="utf-8")

    mismatches = _find_mismatches(text, path.suffix.lower(), read_xyz)
    assert not mismatches, f"{path}: " + "; ".join(mismatches)


@pytest.mark.parametrize("suffix", [".py", ".cpp"])
def test_check_detects_corrupted_blob(suffix: str):
    """Falsifiable smoke test: corrupting a real blob must make the check fail.

    This guards against the regexes silently extracting nothing (which would
    make ``test_inline_blobs_match_xyz_files`` pass vacuously).
    """
    path = next(p for p in _inlined_examples() if p.suffix.lower() == suffix)
    text = path.read_text(encoding="utf-8")
    blob_re = _PY_BLOB_RE if suffix == ".py" else _CPP_BLOB_RE
    match = blob_re.search(text)
    assert match is not None, f"expected an inlined blob in {path}"

    # Inject garbage at the start of the blob so it can no longer match the file.
    corrupted = text[: match.start(1)] + "999 CORRUPTED\n" + text[match.start(1) :]

    def read_xyz(rel: str) -> str:
        return (path.parent / rel).read_text(encoding="utf-8")

    assert _find_mismatches(corrupted, suffix, read_xyz), (
        f"consistency check failed to flag a corrupted inline blob in {path}"
    )


def test_check_detects_tag_without_blob():
    """A docs:xyz tag with no matching from_xyz blob must be flagged."""
    assert _find_mismatches("# docs:xyz water.structure.xyz\n", ".py", lambda _rel: "ignored\n")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
