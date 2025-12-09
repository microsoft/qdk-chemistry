"""Test that documentation examples use sensible coordinate values.

Detects Angstrom/Bohr mix-ups by checking bond distances against known ranges.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import re
from pathlib import Path

import numpy as np
import pytest

from qdk_chemistry.constants import BOHR_TO_ANGSTROM

# Global sanity cap - anything beyond this is almost certainly not a bond
MAX_BOND_LENGTH = 2.5

# Approximate covalent radii in Angstrom for testing bond lengths
COVALENT_RADII = {
    # Period 1
    "H": 0.31,
    "He": 0.28,
    # Period 2
    "Li": 1.28,
    "Be": 0.96,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Ne": 0.58,
    # Period 3
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ar": 1.06,
    # Period 4
    "K": 2.03,
    "Ca": 1.76,
    "Sc": 1.70,
    "Ti": 1.60,
    "V": 1.53,
    "Cr": 1.39,
    "Mn": 1.39,
    "Fe": 1.32,
    "Co": 1.26,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Ga": 1.22,
    "Ge": 1.20,
    "As": 1.19,
    "Se": 1.20,
    "Br": 1.20,
    "Kr": 1.16,
    # Period 5
    "Rb": 2.20,
    "Sr": 1.95,
    "Y": 1.90,
    "Zr": 1.75,
    "Nb": 1.64,
    "Mo": 1.54,
    "Tc": 1.47,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Cd": 1.44,
    "In": 1.42,
    "Sn": 1.39,
    "Sb": 1.39,
    "Te": 1.38,
    "I": 1.39,
    "Xe": 1.40,
    # Period 6
    "Cs": 2.44,
    "Ba": 2.15,
    "La": 2.07,
    "Ce": 2.04,
    "Pr": 2.03,
    "Nd": 2.01,
    "Pm": 1.99,
    "Sm": 1.98,
    "Eu": 1.98,
    "Gd": 1.96,
    "Tb": 1.94,
    "Dy": 1.92,
    "Ho": 1.92,
    "Er": 1.89,
    "Tm": 1.90,
    "Yb": 1.87,
    "Lu": 1.87,
    "Hf": 1.75,
    "Ta": 1.70,
    "W": 1.62,
    "Re": 1.51,
    "Os": 1.44,
    "Ir": 1.41,
    "Pt": 1.36,
    "Au": 1.36,
    "Hg": 1.32,
    "Tl": 1.45,
    "Pb": 1.46,
    "Bi": 1.48,
    "Po": 1.40,
    "At": 1.50,
    "Rn": 1.50,
    # Period 7 (actinides)
    "Fr": 2.60,
    "Ra": 2.21,
    "Ac": 2.15,
    "Th": 2.06,
    "Pa": 2.00,
    "U": 1.96,
    "Np": 1.90,
    "Pu": 1.87,
    "Am": 1.80,
    "Cm": 1.69,
}


def get_bond_range(sym1: str, sym2: str) -> tuple[float, float]:
    """Get expected bond range for atom pair from covalent radii.

    Returns (0.7 * d_est, 1.3 * d_est) where d_est = r_cov[A] + r_cov[B].
    This range is wide enough to allow stretched/compressed bonds but catches unit errors.
    """
    r1 = COVALENT_RADII.get(sym1, 1.5)
    r2 = COVALENT_RADII.get(sym2, 1.5)
    d_est = r1 + r2
    return (0.7 * d_est, 1.3 * d_est)


def get_docs_dir() -> Path:
    """Get docs directory path."""
    return Path(__file__).parent.parent.parent / "docs"


def extract_python_coords(content: str) -> list[tuple[list[str], np.ndarray, int]]:
    """Extract (symbols, coordinates, line_number) from Python code."""
    results = []
    lines = content.split("\n")

    # Find coords = ... patterns
    for match in re.finditer(r"(coords|coordinates)\s*=\s*(?:np\.array\s*\(\s*)?\[", content):
        start = match.end() - 1
        line_num = content[:start].count("\n") + 1

        # Balance brackets
        depth, end = 0, start
        for i, c in enumerate(content[start:], start):
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        coord_text = content[start:end]
        coords = _parse_coords(coord_text)
        if not coords:
            continue

        # Find nearby symbols
        region = "\n".join(lines[max(0, line_num - 20) : line_num + 25])
        sym_match = re.search(r"(?:symbols|elements)\s*=\s*\[([^\]]+)\]", region)
        if sym_match:
            symbols = re.findall(r'["\'](\w+)["\']', sym_match.group(1))
            if len(symbols) == len(coords):
                results.append((symbols, np.array(coords), line_num))

    return results


def extract_cpp_coords(content: str) -> list[tuple[list[str], np.ndarray, int]]:
    """Extract (symbols, coordinates, line_number) from C++ code."""
    results = []
    lines = content.split("\n")

    for match in re.finditer(r"(?:std::vector<Eigen::Vector3d>|auto)\s*\w*coords?\w*\s*=\s*\{", content):
        start = match.end() - 1
        line_num = content[:start].count("\n") + 1

        depth, end = 0, start
        for i, c in enumerate(content[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        coord_text = content[start:end]
        coords = _parse_cpp_coords(coord_text)
        if not coords:
            continue

        region = "\n".join(lines[max(0, line_num - 20) : line_num + 25])
        sym_match = re.search(r"symbols\s*=\s*\{([^}]+)\}", region)
        if sym_match:
            symbols = re.findall(r'"(\w+)"', sym_match.group(1))
            if len(symbols) == len(coords):
                results.append((symbols, np.array(coords), line_num))

    return results


def extract_xyz_coords(content: str) -> list[tuple[list[str], np.ndarray, int]]:
    """Extract coordinates from XYZ format."""
    lines = content.strip().split("\n")
    if len(lines) < 3:
        return []
    try:
        n = int(lines[0].strip())
    except ValueError:
        return []

    if len(lines) < n + 2:
        return []

    symbols, coords = [], []
    for line in lines[2 : n + 2]:
        parts = line.split()
        if len(parts) >= 4:
            symbols.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if len(symbols) == n:
        return [(symbols, np.array(coords), 1)]
    return []


def _parse_coords(text: str) -> list[list[float]] | None:
    """Parse Python coordinate arrays."""
    # Normalize whitespace but preserve structure
    text = re.sub(r"\s+", " ", text)
    matches = re.findall(r"\[\s*([\d\.\-+eE]+)\s*,\s*([\d\.\-+eE]+)\s*,\s*([\d\.\-+eE]+)\s*\]", text)
    if not matches:
        return None
    return [[float(x), float(y), float(z)] for x, y, z in matches]


def _parse_cpp_coords(text: str) -> list[list[float]] | None:
    """Parse C++ coordinate arrays."""
    matches = re.findall(
        r"(?:Eigen::Vector3d)?\s*\{\s*([\d\.\-e\+]+)\s*,\s*([\d\.\-e\+]+)\s*,\s*([\d\.\-e\+]+)\s*\}",
        text.replace("\n", " "),
    )
    if not matches:
        return None
    return [[float(x), float(y), float(z)] for x, y, z in matches]


def validate_geometry(symbols: list[str], coords: np.ndarray, assume_bohr: bool = True) -> list[str]:
    """Check bond distances are sensible. Returns list of error messages."""
    errors = []
    n = len(symbols)

    # Convert to Angstrom if needed
    coords_ang = coords * BOHR_TO_ANGSTROM if assume_bohr else coords

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords_ang[i] - coords_ang[j])
            sym_i, sym_j = symbols[i], symbols[j]

            # Check if bonded (within 1.3x sum of covalent radii)
            r_i = COVALENT_RADII.get(sym_i, 1.5)
            r_j = COVALENT_RADII.get(sym_j, 1.5)
            if dist > 1.3 * (r_i + r_j):
                continue  # Not bonded

            # Global sanity check - anything beyond MAX_BOND_LENGTH is broken
            if dist > MAX_BOND_LENGTH:
                errors.append(
                    f"{sym_i}-{sym_j} distance {dist:.3f}Å exceeds max {MAX_BOND_LENGTH}Å - "
                    f"coords may be Bohr treated as Angstrom"
                )
                continue

            # Check against expected range (specific or estimated from covalent radii)
            min_d, max_d = get_bond_range(sym_i, sym_j)
            if dist < min_d:
                errors.append(
                    f"{sym_i}-{sym_j} bond {dist:.3f}Å too short (min {min_d:.2f}Å) - "
                    f"coords may be Angstrom treated as Bohr"
                )
            elif dist > max_d:
                errors.append(
                    f"{sym_i}-{sym_j} bond {dist:.3f}Å too long (max {max_d:.2f}Å) - "
                    f"coords may be Bohr treated as Angstrom"
                )

    return errors


def collect_snippets() -> list[tuple[Path, list[str], np.ndarray, int, str]]:
    """Collect all coordinate snippets from docs."""
    docs = get_docs_dir()
    snippets: list[tuple[Path, list[str], np.ndarray, int, str]] = []

    if not docs.exists():
        return snippets

    for py in docs.rglob("*.py"):
        try:
            content = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for syms, coords, line in extract_python_coords(content):
            snippets.append((py, syms, coords, line, "python"))

    for cpp in docs.rglob("*.cpp"):
        try:
            content = cpp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for syms, coords, line in extract_cpp_coords(content):
            snippets.append((cpp, syms, coords, line, "cpp"))

    for xyz in docs.rglob("*.xyz"):
        try:
            content = xyz.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for syms, coords, line in extract_xyz_coords(content):
            snippets.append((xyz, syms, coords, line, "xyz"))

    return snippets


class TestDocsCoordinates:
    """Validate coordinate geometries in documentation."""

    @pytest.fixture(scope="class")
    def snippets(self):
        return collect_snippets()

    def test_found_snippets(self, snippets):
        """Verify we found coordinate examples."""
        print(f"\n--- Found {len(snippets)} coordinate snippet(s) ---")
        for path, symbols, _coords, line, ftype in snippets:
            rel_path = path.relative_to(get_docs_dir().parent)
            print(f"  {rel_path}:{line} ({ftype}) - {symbols}")
        assert len(snippets) > 0, "No coordinate snippets found in docs"

    def test_geometries_valid(self, snippets):
        """All geometries should have sensible bond lengths."""
        all_errors = []
        print(f"\n--- Validating {len(snippets)} geometry snippet(s) ---")

        for path, symbols, coords, line, ftype in snippets:
            # XYZ files are in Angstrom, code examples assume Bohr
            assume_bohr = ftype != "xyz"
            errors = validate_geometry(symbols, coords, assume_bohr)
            rel_path = path.relative_to(get_docs_dir().parent)
            status = "FAIL" if errors else "OK"
            print(f"  [{status}] {rel_path}:{line} ({ftype}, {'bohr' if assume_bohr else 'angstrom'})")

            if errors:
                all_errors.append(
                    f"\n{rel_path}:{line} ({ftype}, {'bohr' if assume_bohr else 'angstrom'}):\n"
                    f"  Atoms: {symbols}\n" + "\n".join(f"  - {e}" for e in errors)
                )

        if all_errors:
            pytest.fail(f"Found {len(all_errors)} file(s) with bad geometries:" + "".join(all_errors))


if __name__ == "__main__":
    print("Scanning docs for coordinate snippets...")
    docs_root = get_docs_dir().parent
    for path, symbols, coords, line, ftype in collect_snippets():
        assume_bohr = ftype != "xyz"
        errors = validate_geometry(symbols, coords, assume_bohr)
        status = "FAIL" if errors else "OK"
        rel_path = path.relative_to(docs_root)
        print(f"[{status}] {rel_path}:{line} - {symbols}")
        for e in errors:
            print(f"       Line {line}: {e}")
