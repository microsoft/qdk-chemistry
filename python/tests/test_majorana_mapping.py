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
        """Parity n=4 matches CNOT-derived reference."""
        par = MajoranaMapping.parity(num_modes=4)
        expected = ("IIXX", "IIXY", "IXXZ", "IXYI", "XXZI", "XYIZ", "XZIZ", "YIZI")
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

    def test_clifford_violation(self) -> None:
        """Table violating Clifford algebra raises ValueError."""
        # gamma_0 = gamma_1 = IX → they commute (shouldn't)
        with pytest.raises(ValueError, match="Clifford"):
            MajoranaMapping(table=["IX", "IX", "XZ", "YZ"])

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
        assert scbk.base_encoding == "bravyi-kitaev"

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
