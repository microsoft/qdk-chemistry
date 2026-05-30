"""Tests for MajoranaMapping data class.

Tests cover:
- Factory construction (JW, BK, parity) for various sizes
- Clifford algebra anticommutation validation
- Custom mapping construction
- from_mode_pairs construction
- Immutability enforcement
- Serialization round-trips (JSON, HDF5, file)
- Invalid input rejection
- Reference output comparison for small systems
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import tempfile
from pathlib import Path

import h5py
import pytest

from qdk_chemistry._core.data import PauliTermAccumulator
from qdk_chemistry.data import MajoranaMapping

# ─── Helpers ─────────────────────────────────────────────────────────────


def verify_clifford_algebra(mapping: MajoranaMapping) -> None:
    """Verify {gamma_i, gamma_j} = 2δ_{ij}·I for all pairs."""
    n = 2 * mapping.num_modes
    for i in range(n):
        wi = mapping.core(i)
        for j in range(i, n):
            wj = mapping.core(j)
            phase_ij, word_ij = PauliTermAccumulator.multiply_uncached(wi, wj)
            phase_ji, word_ji = PauliTermAccumulator.multiply_uncached(wj, wi)

            if i == j:
                assert word_ij == [], f"gamma_{i}² is not identity: word={word_ij}"
                assert abs(phase_ij - 1.0) < 1e-12, f"gamma_{i}² phase={phase_ij}, expected 1"
            else:
                assert word_ij == word_ji, f"gamma_{i}·gamma_{j} and gamma_{j}·gamma_{i} produce different words"
                assert abs(phase_ij + phase_ji) < 1e-12, (
                    f"{{gamma_{i}, gamma_{j}}} != 0: phases {phase_ij} + {phase_ji} = {phase_ij + phase_ji}"
                )


# ─── Factory Tests ───────────────────────────────────────────────────────


class TestJordanWigner:
    """Tests for the Jordan-Wigner factory."""

    @pytest.mark.parametrize("n_modes", [1, 2, 4, 6, 8])
    def test_clifford_algebra(self, n_modes: int) -> None:
        """JW tables satisfy Clifford anticommutation for various sizes."""
        jw = MajoranaMapping.jordan_wigner(num_modes=n_modes)
        verify_clifford_algebra(jw)

    def test_properties(self) -> None:
        """JW factory sets correct properties."""
        jw = MajoranaMapping.jordan_wigner(num_modes=4)
        assert jw.num_modes == 4
        assert jw.num_qubits == 4
        assert jw.name == "jordan-wigner"
        assert len(jw.table) == 8

    def test_reference_n2(self) -> None:
        """JW n=2 matches hand-computed reference (little-endian)."""
        jw = MajoranaMapping.jordan_wigner(num_modes=2)
        # gamma_0 = X_0 → "IX" (qubit 0 rightmost = X, qubit 1 = I)
        # gamma_1 = Y_0 → "IY"
        # gamma_2 = Z_0 X_1 → "XZ" (qubit 0 = Z, qubit 1 = X)
        # gamma_3 = Z_0 Y_1 → "YZ"
        assert jw.table == ("IX", "IY", "XZ", "YZ")

    def test_reference_n4(self) -> None:
        """JW n=4 spot check: gamma_6 = Z_0 Z_1 Z_2 X_3."""
        jw = MajoranaMapping.jordan_wigner(num_modes=4)
        assert jw.table[6] == "XZZZ"  # X_3 Z_2 Z_1 Z_0 in little-endian
        assert jw.table[7] == "YZZZ"  # Y_3 Z_2 Z_1 Z_0


class TestBravyiKitaev:
    """Tests for the Bravyi-Kitaev factory."""

    @pytest.mark.parametrize("n_modes", [1, 2, 4, 6, 8])
    def test_clifford_algebra(self, n_modes: int) -> None:
        """BK tables satisfy Clifford anticommutation for various sizes."""
        bk = MajoranaMapping.bravyi_kitaev(num_modes=n_modes)
        verify_clifford_algebra(bk)

    def test_properties(self) -> None:
        """BK factory sets correct properties."""
        bk = MajoranaMapping.bravyi_kitaev(num_modes=4)
        assert bk.num_modes == 4
        assert bk.num_qubits == 4
        assert bk.name == "bravyi-kitaev"

    @pytest.mark.parametrize("n_modes", [3, 5, 7])
    def test_non_power_of_two(self, n_modes: int) -> None:
        """BK works for non-power-of-2 mode counts."""
        bk = MajoranaMapping.bravyi_kitaev(num_modes=n_modes)
        assert bk.num_modes == n_modes
        verify_clifford_algebra(bk)


class TestParity:
    """Tests for the parity encoding factory."""

    @pytest.mark.parametrize("n_modes", [1, 2, 4, 6, 8])
    def test_clifford_algebra(self, n_modes: int) -> None:
        """Parity tables satisfy Clifford anticommutation for various sizes."""
        par = MajoranaMapping.parity(num_modes=n_modes)
        verify_clifford_algebra(par)

    def test_properties(self) -> None:
        """Parity factory sets correct properties."""
        par = MajoranaMapping.parity(num_modes=4)
        assert par.num_modes == 4
        assert par.num_qubits == 4
        assert par.name == "parity"

    def test_reference_n2(self) -> None:
        """Parity n=2 matches CNOT-derived reference."""
        par = MajoranaMapping.parity(num_modes=2)
        # Derived via CNOT(0,1) conjugation of JW:
        # gamma_0 = X_0 X_1 → "XX"
        # gamma_1 = Y_0 X_1 → "XY" (little-endian: qubit 0=Y, qubit 1=X)
        # gamma_2 = Z_0 X_1 → "XZ"
        # gamma_3 = Y_1     → "YI"
        assert par.table == ("XX", "XY", "XZ", "YI")

    def test_reference_n4(self) -> None:
        """Parity n=4 matches standard-convention reference."""
        par = MajoranaMapping.parity(num_modes=4)
        expected = ("XXXX", "XXXY", "XXXZ", "XXYI", "XXZI", "XYII", "XZII", "YIII")
        assert par.table == expected


# ─── Custom Mapping Tests ────────────────────────────────────────────────


class TestCustomMapping:
    """Tests for custom MajoranaMapping construction."""

    def test_custom_from_table(self) -> None:
        """Custom mapping from table list."""
        custom = MajoranaMapping(table=["IX", "IY", "XZ", "YZ"], name="my-jw")
        assert custom.num_modes == 2
        assert custom.num_qubits == 2
        assert custom.name == "my-jw"
        assert custom.table == ("IX", "IY", "XZ", "YZ")

    def test_custom_unnamed(self) -> None:
        """Custom mapping without name."""
        custom = MajoranaMapping(table=["IX", "IY", "XZ", "YZ"])
        assert custom.name == ""

    def test_from_mode_pairs(self) -> None:
        """from_mode_pairs produces same result as direct table."""
        direct = MajoranaMapping(table=["IX", "IY", "XZ", "YZ"], name="test")
        pairs = MajoranaMapping.from_mode_pairs(pairs=[("IX", "IY"), ("XZ", "YZ")], name="test")
        assert direct.table == pairs.table

    def test_from_mode_pairs_equivalence(self) -> None:
        """from_mode_pairs matches JW factory for n=2."""
        jw = MajoranaMapping.jordan_wigner(num_modes=2)
        pairs = MajoranaMapping.from_mode_pairs(pairs=[("IX", "IY"), ("XZ", "YZ")], name="jordan-wigner")
        assert jw.table == pairs.table


class TestFromBilinears:
    """Tests for the from_bilinears construction."""

    def test_bilinear_only_basic(self) -> None:
        """Bilinear-only mapping stores and retrieves bilinears correctly."""
        jw = MajoranaMapping.jordan_wigner(num_modes=2)
        bilinears: dict[tuple[int, int], tuple[complex, str]] = {}
        for j in range(4):
            for k in range(j + 1, 4):
                coeff, pauli = jw.bilinear(j, k)
                bilinears[(j, k)] = (coeff, pauli)

        bl = MajoranaMapping.from_bilinears(num_modes=2, bilinears=bilinears, name="test-bl")
        assert bl.num_modes == 2
        assert bl.name == "test-bl"
        assert bl.is_majorana_atomic is False
        assert bl.table == ()

    def test_bilinear_lookup_matches_table(self) -> None:
        """Bilinear-only mapping reproduces the same bilinears as the table form."""
        jw = MajoranaMapping.jordan_wigner(num_modes=3)
        bilinears: dict[tuple[int, int], tuple[complex, str]] = {}
        for j in range(6):
            for k in range(j + 1, 6):
                coeff, pauli = jw.bilinear(j, k)
                bilinears[(j, k)] = (coeff, pauli)

        bl = MajoranaMapping.from_bilinears(num_modes=3, bilinears=bilinears)
        for j in range(6):
            for k in range(j + 1, 6):
                c_bl, p_bl = bl.bilinear(j, k)
                c_jw, p_jw = jw.bilinear(j, k)
                assert p_bl == p_jw, f"bilinear({j},{k}): {p_bl} != {p_jw}"
                assert abs(c_bl - c_jw) < 1e-12, f"bilinear({j},{k}) coeff: {c_bl} != {c_jw}"

    def test_bilinear_antisymmetry(self) -> None:
        """bilinear(k,j) = -bilinear(j,k) for bilinear-only mappings."""
        jw = MajoranaMapping.jordan_wigner(num_modes=2)
        bilinears: dict[tuple[int, int], tuple[complex, str]] = {}
        for j in range(4):
            for k in range(j + 1, 4):
                bilinears[(j, k)] = jw.bilinear(j, k)

        bl = MajoranaMapping.from_bilinears(num_modes=2, bilinears=bilinears)
        for j in range(4):
            for k in range(j + 1, 4):
                c_fwd, p_fwd = bl.bilinear(j, k)
                c_rev, p_rev = bl.bilinear(k, j)
                assert p_fwd == p_rev
                assert abs(c_fwd + c_rev) < 1e-12

    def test_majorana_raises_for_bilinear_only(self) -> None:
        """majorana(k) raises ValueError for bilinear-only mappings."""
        jw = MajoranaMapping.jordan_wigner(num_modes=2)
        bilinears: dict[tuple[int, int], tuple[complex, str]] = {}
        for j in range(4):
            for k in range(j + 1, 4):
                bilinears[(j, k)] = jw.bilinear(j, k)
        bl = MajoranaMapping.from_bilinears(num_modes=2, bilinears=bilinears)
        with pytest.raises(ValueError, match="bilinear-only"):
            bl.majorana(0)


# ─── Validation Tests ────────────────────────────────────────────────────


class TestValidation:
    """Tests for input validation."""

    def test_empty_table(self) -> None:
        """Empty table raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            MajoranaMapping(table=[])

    def test_odd_length_table(self) -> None:
        """Odd-length table raises ValueError."""
        with pytest.raises(ValueError, match="even number"):
            MajoranaMapping(table=["IX", "IY", "XZ"])

    def test_invalid_characters(self) -> None:
        """Non-IXYZ characters raise ValueError."""
        with pytest.raises(ValueError, match="Invalid Pauli character"):
            MajoranaMapping(table=["IX", "IA", "XZ", "YZ"])

    def test_inconsistent_lengths(self) -> None:
        """Strings of different lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            MajoranaMapping(table=["IX", "IYZ", "XZ", "YZ"])

    def test_zero_modes(self) -> None:
        """Zero modes raises ValueError in factories."""
        with pytest.raises(ValueError, match="num_modes"):
            MajoranaMapping.jordan_wigner(num_modes=0)
        with pytest.raises(ValueError, match="num_modes"):
            MajoranaMapping.bravyi_kitaev(num_modes=0)
        with pytest.raises(ValueError, match="num_modes"):
            MajoranaMapping.parity(num_modes=0)


# ─── Immutability Tests ─────────────────────────────────────────────────


class TestImmutability:
    """Tests for DataClass immutability."""

    def test_cannot_set_table(self) -> None:
        """Setting table raises AttributeError."""
        jw = MajoranaMapping.jordan_wigner(num_modes=2)
        with pytest.raises(AttributeError, match="Cannot modify"):
            jw.table = ("foo",)  # type: ignore[misc]

    def test_cannot_set_name(self) -> None:
        """Setting name raises AttributeError."""
        jw = MajoranaMapping.jordan_wigner(num_modes=2)
        with pytest.raises(AttributeError, match="Cannot modify"):
            jw.name = "bar"  # type: ignore[misc]

    def test_cannot_delete_attribute(self) -> None:
        """Deleting attribute raises AttributeError."""
        jw = MajoranaMapping.jordan_wigner(num_modes=2)
        with pytest.raises(AttributeError, match="Cannot delete"):
            del jw.table  # type: ignore[misc]


# ─── Serialization Tests ────────────────────────────────────────────────


class TestSerialization:
    """Tests for JSON and HDF5 serialization."""

    def test_json_round_trip(self) -> None:
        """JSON serialize and deserialize."""
        jw = MajoranaMapping.jordan_wigner(num_modes=4)
        data = jw.to_json()
        loaded = MajoranaMapping.from_json(data)
        assert loaded.table == jw.table
        assert loaded.name == jw.name
        assert loaded.num_modes == jw.num_modes

    def test_json_has_version(self) -> None:
        """JSON output includes version field."""
        jw = MajoranaMapping.jordan_wigner(num_modes=2)
        data = jw.to_json()
        assert "version" in data
        assert data["version"] == "0.1.0"

    def test_hdf5_round_trip(self) -> None:
        """HDF5 serialize and deserialize."""
        bk = MajoranaMapping.bravyi_kitaev(num_modes=4)
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            with h5py.File(f.name, "w") as hf:
                bk.to_hdf5(hf)
            with h5py.File(f.name, "r") as hf:
                loaded = MajoranaMapping.from_hdf5(hf)
        assert loaded.table == bk.table
        assert loaded.name == bk.name

    def test_json_file_round_trip(self) -> None:
        """JSON file serialize and deserialize."""
        par = MajoranaMapping.parity(num_modes=4)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.majorana_mapping.json"
            par.to_json_file(str(path))
            loaded = MajoranaMapping.from_json_file(str(path))
        assert loaded.table == par.table
        assert loaded.name == par.name

    def test_hdf5_file_round_trip(self) -> None:
        """HDF5 file serialize and deserialize."""
        jw = MajoranaMapping.jordan_wigner(num_modes=4)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.majorana_mapping.h5"
            jw.to_hdf5_file(str(path))
            loaded = MajoranaMapping.from_hdf5_file(str(path))
        assert loaded.table == jw.table

    def test_custom_serialization(self) -> None:
        """Custom mapping with name survives serialization."""
        custom = MajoranaMapping(table=["IX", "IY", "XZ", "YZ"], name="my-custom")
        data = custom.to_json()
        loaded = MajoranaMapping.from_json(data)
        assert loaded.table == custom.table
        assert loaded.name == "my-custom"

    def test_hdf5_round_trip_with_tapering(self) -> None:
        """HDF5 round-trip preserves tapering for SCBK mappings."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        scbk = MajoranaMapping.symmetry_conserving_bravyi_kitaev(8, Symmetries(2, 2))
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            with h5py.File(f.name, "w") as hf:
                scbk.to_hdf5(hf)
            with h5py.File(f.name, "r") as hf:
                loaded = MajoranaMapping.from_hdf5(hf)
        assert loaded.table == scbk.table
        assert loaded.name == scbk.name
        assert loaded.tapering is not None
        assert loaded.tapering.qubit_indices == scbk.tapering.qubit_indices
        assert loaded.tapering.eigenvalues == scbk.tapering.eigenvalues
        assert loaded.tapering.source_num_qubits == scbk.tapering.source_num_qubits
        assert loaded.tapering.source_encoding == scbk.tapering.source_encoding


# ─── Summary/Repr Tests ─────────────────────────────────────────────────


class TestDisplay:
    """Tests for summary and repr."""

    def test_repr_with_name(self) -> None:
        """Repr includes name when present."""
        jw = MajoranaMapping.jordan_wigner(num_modes=2)
        r = repr(jw)
        assert "jordan-wigner" in r
        assert "num_modes=2" in r

    def test_repr_without_name(self) -> None:
        """Repr works for unnamed custom mappings."""
        custom = MajoranaMapping(table=["IX", "IY", "XZ", "YZ"])
        r = repr(custom)
        assert "num_modes=2" in r

    def test_get_summary(self) -> None:
        """get_summary includes mapping details."""
        jw = MajoranaMapping.jordan_wigner(num_modes=2)
        s = jw.get_summary()
        assert "jordan-wigner" in s
        assert "Modes: 2" in s
        assert "gamma_0" in s
        assert "IX" in s


# ─── Cross-encoding consistency ──────────────────────────────────────────


class TestCrossEncodingConsistency:
    """Tests ensuring all encodings produce valid mappings with same structure."""

    @pytest.mark.parametrize("n_modes", [2, 4, 6])
    def test_all_encodings_same_qubit_count(self, n_modes: int) -> None:
        """JW, BK, and parity all use num_modes qubits."""
        jw = MajoranaMapping.jordan_wigner(num_modes=n_modes)
        bk = MajoranaMapping.bravyi_kitaev(num_modes=n_modes)
        par = MajoranaMapping.parity(num_modes=n_modes)
        assert jw.num_qubits == n_modes
        assert bk.num_qubits == n_modes
        assert par.num_qubits == n_modes

    @pytest.mark.parametrize("n_modes", [2, 4, 6])
    def test_all_encodings_table_length(self, n_modes: int) -> None:
        """All encodings have 2N table entries."""
        jw = MajoranaMapping.jordan_wigner(num_modes=n_modes)
        bk = MajoranaMapping.bravyi_kitaev(num_modes=n_modes)
        par = MajoranaMapping.parity(num_modes=n_modes)
        assert len(jw.table) == 2 * n_modes
        assert len(bk.table) == 2 * n_modes
        assert len(par.table) == 2 * n_modes


# ─── SCBK (symmetry-conserving Bravyi-Kitaev) ────────────────────────────


class TestSymmetryConservingBravyiKitaev:
    """Tests for the SCBK factory method and TaperingSpecification integration."""

    def test_scbk_factory_creates_bk_table(self) -> None:
        """SCBK factory uses a standard BK Majorana table underneath."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        scbk = MajoranaMapping.symmetry_conserving_bravyi_kitaev(8, Symmetries(2, 2))
        bk = MajoranaMapping.bravyi_kitaev(8)
        assert scbk.table == bk.table

    def test_scbk_name_and_base_encoding(self) -> None:
        """SCBK has the right name and base_encoding properties."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        scbk = MajoranaMapping.symmetry_conserving_bravyi_kitaev(8, Symmetries(2, 2))
        assert scbk.name == "symmetry-conserving-bravyi-kitaev"
        assert scbk.base_encoding == "bravyi-kitaev-tree"

    def test_scbk_has_tapering(self) -> None:
        """SCBK mapping has a TaperingSpecification with 2 tapered qubits."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        scbk = MajoranaMapping.symmetry_conserving_bravyi_kitaev(8, Symmetries(2, 2))
        assert scbk.tapering is not None
        assert scbk.tapering.num_tapered == 2
        assert scbk.tapering.source_num_qubits == 8
        assert scbk.tapering.source_encoding == "bravyi-kitaev"

    def test_scbk_eigenvalues_depend_on_alpha_beta(self) -> None:
        """Different (n_alpha, n_beta) produce different eigenvalues."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        scbk_11 = MajoranaMapping.symmetry_conserving_bravyi_kitaev(8, Symmetries(1, 1))
        scbk_20 = MajoranaMapping.symmetry_conserving_bravyi_kitaev(8, Symmetries(2, 0))
        assert scbk_11.tapering.eigenvalues != scbk_20.tapering.eigenvalues

    def test_scbk_num_qubits_is_posttaper_via_property(self) -> None:
        """MajoranaMapping.num_qubits reflects the reduced qubit count."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        scbk = MajoranaMapping.symmetry_conserving_bravyi_kitaev(8, Symmetries(2, 2))
        assert scbk.num_qubits == 6

    def test_scbk_json_roundtrip(self) -> None:
        """SCBK mapping with tapering survives JSON serialization."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        scbk = MajoranaMapping.symmetry_conserving_bravyi_kitaev(8, Symmetries(2, 2))
        data = scbk.to_json()
        loaded = MajoranaMapping.from_json(data)
        assert loaded.name == scbk.name
        assert loaded.table == scbk.table
        assert loaded.tapering is not None
        assert loaded.tapering.qubit_indices == scbk.tapering.qubit_indices
        assert loaded.tapering.eigenvalues == scbk.tapering.eigenvalues

    def test_scbk_odd_modes_raises(self) -> None:
        """SCBK with odd num_modes raises ValueError."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        with pytest.raises(ValueError, match="even"):
            MajoranaMapping.symmetry_conserving_bravyi_kitaev(7, Symmetries(1, 1))

    def test_scbk_too_few_modes_raises(self) -> None:
        """SCBK with num_modes < 4 raises ValueError."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        with pytest.raises(ValueError, match="4"):
            MajoranaMapping.symmetry_conserving_bravyi_kitaev(2, Symmetries(1, 0))

    def test_standard_mappings_have_no_tapering(self) -> None:
        """JW, BK, parity (without symmetries) have tapering=None."""
        assert MajoranaMapping.jordan_wigner(4).tapering is None
        assert MajoranaMapping.bravyi_kitaev(4).tapering is None
        assert MajoranaMapping.parity(4).tapering is None

    def test_scbk_num_qubits_is_posttaper(self) -> None:
        """MajoranaMapping.num_qubits reflects the post-taper qubit count."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        scbk = MajoranaMapping.symmetry_conserving_bravyi_kitaev(8, Symmetries(2, 2))
        assert scbk.num_qubits == 6  # 8 - 2 tapered

    def test_parity_with_symmetries_has_tapering(self) -> None:
        """Parity with symmetries bundles a TaperingSpecification."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        par = MajoranaMapping.parity(8, Symmetries(2, 2))
        assert par.tapering is not None
        assert par.tapering.num_tapered == 2
        assert par.name == "parity-2q-reduced"
        assert par.base_encoding == "parity"
        assert par.num_qubits == 6  # 8 - 2

    def test_parity_without_symmetries_no_tapering(self) -> None:
        """Parity without symmetries has no tapering."""
        par = MajoranaMapping.parity(8)
        assert par.tapering is None
        assert par.num_qubits == 8


# ─── Bilinear primitive ──────────────────────────────────────────────────


def _dense_to_sparse(s: str) -> list[tuple[int, str]]:
    """Convert a dense little-endian Pauli string to a sparse word list.

    The dense representation is ``s[0]`` = highest-index qubit, ``s[-1]`` =
    qubit 0. The sparse representation is a list of ``(qubit_index, gate)``
    pairs sorted by ``qubit_index``, omitting identities.
    """
    n = len(s)
    return [(n - 1 - i, c) for i, c in enumerate(s) if c != "I"]


def _sparse_to_dense(word: list[tuple[int, str]], n_qubits: int) -> str:
    """Inverse of ``_dense_to_sparse``."""
    chars = ["I"] * n_qubits
    for q, g in word:
        chars[q] = g
    return "".join(reversed(chars))


# Single-qubit Pauli multiplication table: (a, b) -> (phase, c) where a*b = phase * c.
_PAULI_MULT = {
    ("I", "I"): (1 + 0j, "I"),
    ("I", "X"): (1 + 0j, "X"),
    ("I", "Y"): (1 + 0j, "Y"),
    ("I", "Z"): (1 + 0j, "Z"),
    ("X", "I"): (1 + 0j, "X"),
    ("X", "X"): (1 + 0j, "I"),
    ("X", "Y"): (1j, "Z"),
    ("X", "Z"): (-1j, "Y"),
    ("Y", "I"): (1 + 0j, "Y"),
    ("Y", "X"): (-1j, "Z"),
    ("Y", "Y"): (1 + 0j, "I"),
    ("Y", "Z"): (1j, "X"),
    ("Z", "I"): (1 + 0j, "Z"),
    ("Z", "X"): (1j, "Y"),
    ("Z", "Y"): (-1j, "X"),
    ("Z", "Z"): (1 + 0j, "I"),
}


def _multiply_dense(a: str, b: str) -> tuple[complex, str]:
    """Multiply two dense little-endian Pauli strings of equal length."""
    assert len(a) == len(b)
    phase: complex = 1 + 0j
    chars: list[str] = []
    for ca, cb in zip(a, b, strict=True):
        p, c = _PAULI_MULT[(ca, cb)]
        phase *= p
        chars.append(c)
    return phase, "".join(chars)


_FACTORIES = [
    ("jordan_wigner", lambda n: MajoranaMapping.jordan_wigner(num_modes=n)),
    ("bravyi_kitaev", lambda n: MajoranaMapping.bravyi_kitaev(num_modes=n)),
    ("parity", lambda n: MajoranaMapping.parity(num_modes=n)),
]


class TestBilinear:
    """Tests for the unified bilinear primitive ``i·gamma_j·gamma_k``.

    Bilinears are the most general primitive across fermion-to-qubit
    encodings: Majorana-atomic encodings expose individual gamma_k as well, but
    redundant encodings (e.g. Bravyi-Kitaev superfast) only admit
    parity-even products. These tests verify the algebraic invariants that
    every backend must satisfy.
    """

    @pytest.mark.parametrize(("name", "factory"), _FACTORIES)
    @pytest.mark.parametrize("n_modes", [2, 4, 6])
    def test_matches_majorana_product(self, name: str, factory, n_modes: int) -> None:
        """``bilinear(j, k)`` equals ``i · majorana(j) · majorana(k)`` exactly."""
        del name
        m = factory(n_modes)
        n = 2 * n_modes
        for j in range(n):
            for k in range(n):
                if j == k:
                    continue
                expected_phase, expected_word = _multiply_dense(m.majorana(j), m.majorana(k))
                expected_coeff = 1j * expected_phase
                bcoeff, bword = m.bilinear(j, k)
                assert bword == expected_word, f"({j},{k}): word {bword} != {expected_word}"
                assert abs(bcoeff - expected_coeff) < 1e-12, f"({j},{k}): coeff {bcoeff} != {expected_coeff}"

    @pytest.mark.parametrize(("name", "factory"), _FACTORIES)
    @pytest.mark.parametrize("n_modes", [2, 4])
    def test_antisymmetry(self, name: str, factory, n_modes: int) -> None:
        """``bilinear(k, j) == -bilinear(j, k)`` for all distinct j, k."""
        del name
        m = factory(n_modes)
        n = 2 * n_modes
        for j in range(n):
            for k in range(j + 1, n):
                cjk, wjk = m.bilinear(j, k)
                ckj, wkj = m.bilinear(k, j)
                assert wjk == wkj
                assert abs(cjk + ckj) < 1e-12, f"({j},{k}): {cjk} + {ckj} != 0"

    @pytest.mark.parametrize(("name", "factory"), _FACTORIES)
    @pytest.mark.parametrize("n_modes", [2, 4])
    def test_squares_to_identity(self, name: str, factory, n_modes: int) -> None:
        """``(i·gamma_j·gamma_k)² == I`` for all distinct j, k."""
        del name
        m = factory(n_modes)
        n = 2 * n_modes
        identity = "I" * len(m.table[0])
        for j in range(n):
            for k in range(n):
                if j == k:
                    continue
                coeff, word = m.bilinear(j, k)
                phase, prod = _multiply_dense(word, word)
                assert prod == identity, f"({j},{k}): word² = {prod}, not identity"
                assert abs((coeff * coeff) * phase - 1.0) < 1e-12, (
                    f"({j},{k}): bilinear² coeff = {coeff * coeff * phase}"
                )

    @pytest.mark.parametrize(("name", "factory"), _FACTORIES)
    @pytest.mark.parametrize("n_modes", [3, 4])
    def test_commutation_relations(self, name: str, factory, n_modes: int) -> None:
        """Disjoint pairs commute; pairs sharing exactly one index anticommute."""
        del name
        m = factory(n_modes)
        n = 2 * n_modes
        pairs = [(j, k) for j in range(n) for k in range(j + 1, n)]
        for j1, k1 in pairs:
            c1, w1 = m.bilinear(j1, k1)
            for j2, k2 in pairs:
                shared = len({j1, k1} & {j2, k2})
                if shared == 2:
                    continue
                c2, w2 = m.bilinear(j2, k2)
                p_ab, w_ab = _multiply_dense(w1, w2)
                p_ba, w_ba = _multiply_dense(w2, w1)
                assert w_ab == w_ba
                ab = c1 * c2 * p_ab
                ba = c2 * c1 * p_ba
                if shared == 0:
                    assert abs(ab - ba) < 1e-12, f"disjoint ({j1},{k1})·({j2},{k2}) does not commute"
                else:
                    assert abs(ab + ba) < 1e-12, f"single-shared ({j1},{k1})·({j2},{k2}) does not anticommute"

    @pytest.mark.parametrize(("name", "factory"), _FACTORIES)
    @pytest.mark.parametrize("n_modes", [2, 4])
    def test_hermitian_real_coefficient(self, name: str, factory, n_modes: int) -> None:
        """Bilinears are Hermitian, so the coefficient is real for current encodings."""
        del name
        m = factory(n_modes)
        n = 2 * n_modes
        for j in range(n):
            for k in range(n):
                if j == k:
                    continue
                coeff, _ = m.bilinear(j, k)
                assert abs(coeff.imag) < 1e-12, f"({j},{k}): coeff {coeff} is not real"
                assert abs(abs(coeff.real) - 1.0) < 1e-12, f"({j},{k}): |coeff| = {abs(coeff.real)}, expected 1"

    def test_jw_n2_known_values(self) -> None:
        """JW(num_modes=2): gamma_0=X_0, gamma_1=Y_0, gamma_2=Z_0 X_1, gamma_3=Z_0 Y_1."""
        m = MajoranaMapping.jordan_wigner(num_modes=2)
        assert m.bilinear(0, 1) == (-1 + 0j, "IZ")
        assert m.bilinear(1, 0) == (1 + 0j, "IZ")
        assert m.bilinear(0, 2) == (1 + 0j, "XY")
        assert m.bilinear(2, 3) == (-1 + 0j, "ZI")

    def test_raises_on_equal_indices(self) -> None:
        """``bilinear(j, j)`` is undefined and must raise ValueError."""
        m = MajoranaMapping.jordan_wigner(num_modes=2)
        with pytest.raises(ValueError, match="distinct"):
            m.bilinear(0, 0)
        with pytest.raises(ValueError, match="distinct"):
            m.bilinear(3, 3)

    def test_raises_on_out_of_range(self) -> None:
        """Out-of-range indices raise IndexError."""
        m = MajoranaMapping.jordan_wigner(num_modes=2)
        with pytest.raises(IndexError):
            m.bilinear(0, 4)
        with pytest.raises(IndexError):
            m.bilinear(99, 0)

    def test_majorana_consistent_with_call(self) -> None:
        """``majorana(k)`` and ``__call__(k)`` describe the same Pauli operator."""
        m = MajoranaMapping.jordan_wigner(num_modes=3)
        n_q = len(m.table[0])
        for k in range(2 * m.num_modes):
            sparse = m.core(k)
            dense = m.majorana(k)
            assert len(dense) == n_q
            non_identity = sum(1 for c in dense if c != "I")
            assert non_identity == len(sparse)

    def test_majorana_out_of_range(self) -> None:
        """``majorana(k)`` raises IndexError on out-of-range k."""
        m = MajoranaMapping.jordan_wigner(num_modes=2)
        with pytest.raises(IndexError):
            m.majorana(4)
        with pytest.raises(IndexError):
            m.majorana(99)


class TestEncodingMetadata:
    """Tests for the new metadata properties on Majorana-atomic encodings."""

    @pytest.mark.parametrize(("name", "factory"), _FACTORIES)
    @pytest.mark.parametrize("n_modes", [2, 4])
    def test_is_majorana_atomic(self, name: str, factory, n_modes: int) -> None:
        """All current encodings are Majorana-atomic."""
        del name
        m = factory(n_modes)
        assert m.is_majorana_atomic is True

    def test_pauli_string_length_untapered(self) -> None:
        """For untapered encodings, bilinear/majorana strings have length ``num_qubits``."""
        m = MajoranaMapping.jordan_wigner(num_modes=4)
        assert len(m.table[0]) == m.num_qubits == 4
        assert len(m.majorana(0)) == 4
        _, w = m.bilinear(0, 1)
        assert len(w) == 4

    def test_pauli_string_length_with_tapering(self) -> None:
        """For tapered SCBK, bilinear/majorana operate in the pre-taper basis."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        scbk = MajoranaMapping.symmetry_conserving_bravyi_kitaev(8, Symmetries(2, 2))
        assert len(scbk.table[0]) == 8
        assert scbk.num_qubits == 6
        for j in range(2 * scbk.num_modes):
            assert len(scbk.majorana(j)) == 8
            for k in range(2 * scbk.num_modes):
                if j == k:
                    continue
                _, w = scbk.bilinear(j, k)
                assert len(w) == 8
