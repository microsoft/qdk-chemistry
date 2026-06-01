"""Tests for MajoranaMapping data class.

Tests cover:
- Factory construction (JW, BK, parity) for various sizes
- Clifford algebra anticommutation validation
- Custom mapping construction
- from_bilinears construction (bilinear-only mappings)
- Bilinear caching correctness
- Sparse/dense Pauli string conversion helpers
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

import h5py
import pytest

from qdk_chemistry._core.data import (
    PauliTermAccumulator,
    label_to_sparse_pauli_word,
    sparse_pauli_word_to_label,
)
from qdk_chemistry.data import MajoranaMapping

# ─── Helpers ─────────────────────────────────────────────────────────────


def label_to_word(label: str) -> list[tuple[int, int]]:
    """Convert a QubitHamiltonian label to a sparse Pauli word."""
    return label_to_sparse_pauli_word(label)


def word_to_label(word: list[tuple[int, int]], num_qubits: int) -> str:
    """Convert a sparse Pauli word to a QubitHamiltonian label."""
    return sparse_pauli_word_to_label(word, num_qubits)


def table_labels(mapping: MajoranaMapping) -> tuple[str, ...]:
    """Return dense labels for a mapping's sparse table."""
    return tuple(word_to_label(word, mapping.num_qubits) for word in mapping.table)


def bilinear_entries(mapping: MajoranaMapping) -> list[tuple[complex, list[tuple[int, int]]]]:
    """Return upper-triangle bilinear entries in C++ row-major order."""
    return [mapping.bilinear(j, k) for j in range(2 * mapping.num_modes) for k in range(j + 1, 2 * mapping.num_modes)]


def verify_clifford_algebra(mapping: MajoranaMapping) -> None:
    """Verify {gamma_i, gamma_j} = 2δ_{ij}·I for all pairs."""
    n = 2 * mapping.num_modes
    for i in range(n):
        wi = mapping.majorana(i)
        for j in range(i, n):
            wj = mapping.majorana(j)
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

    def test_reference_n2(self) -> None:
        """JW n=2 matches hand-computed reference (little-endian)."""
        jw = MajoranaMapping.jordan_wigner(num_modes=2)
        # gamma_0 = X_0 → "IX" (qubit 0 rightmost = X, qubit 1 = I)
        # gamma_1 = Y_0 → "IY"
        # gamma_2 = Z_0 X_1 → "XZ" (qubit 0 = Z, qubit 1 = X)
        # gamma_3 = Z_0 Y_1 → "YZ"
        assert table_labels(jw) == ("IX", "IY", "XZ", "YZ")

    def test_reference_n4(self) -> None:
        """JW n=4 spot check: gamma_6 = Z_0 Z_1 Z_2 X_3."""
        jw = MajoranaMapping.jordan_wigner(num_modes=4)
        labels = table_labels(jw)
        assert labels[6] == "XZZZ"  # X_3 Z_2 Z_1 Z_0 in little-endian
        assert labels[7] == "YZZZ"  # Y_3 Z_2 Z_1 Z_0


class TestBravyiKitaev:
    """Tests for the Bravyi-Kitaev factory."""

    @pytest.mark.parametrize("n_modes", [1, 2, 4, 6, 8])
    def test_clifford_algebra(self, n_modes: int) -> None:
        """BK tables satisfy Clifford anticommutation for various sizes."""
        bk = MajoranaMapping.bravyi_kitaev(num_modes=n_modes)
        verify_clifford_algebra(bk)

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

    def test_reference_n2(self) -> None:
        """Parity n=2 matches CNOT-derived reference."""
        par = MajoranaMapping.parity(num_modes=2)
        # Derived via CNOT(0,1) conjugation of JW:
        # gamma_0 = X_0 X_1 → "XX"
        # gamma_1 = Y_0 X_1 → "XY" (little-endian: qubit 0=Y, qubit 1=X)
        # gamma_2 = Z_0 X_1 → "XZ"
        # gamma_3 = Y_1     → "YI"
        assert table_labels(par) == ("XX", "XY", "XZ", "YI")

    def test_reference_n4(self) -> None:
        """Parity n=4 matches standard-convention reference."""
        par = MajoranaMapping.parity(num_modes=4)
        expected = ("XXXX", "XXXY", "XXXZ", "XXYI", "XXZI", "XYII", "XZII", "YIII")
        assert table_labels(par) == expected


# ─── Custom Mapping Tests ────────────────────────────────────────────────


class TestCustomMapping:
    """Tests for custom MajoranaMapping construction."""

    def test_custom_from_table(self) -> None:
        """Custom mapping from table list."""
        custom = MajoranaMapping.from_table([label_to_word(s) for s in ("IX", "IY", "XZ", "YZ")], name="my-jw")
        assert custom.num_modes == 2
        assert custom.num_qubits == 2
        assert custom.name == "my-jw"
        assert table_labels(custom) == ("IX", "IY", "XZ", "YZ")


class TestFromBilinears:
    """Tests for the from_bilinears construction."""

    def test_bilinear_lookup_matches_table(self) -> None:
        """Bilinear-only mapping reproduces the same bilinears as the table form."""
        jw = MajoranaMapping.jordan_wigner(num_modes=3)
        bl = MajoranaMapping.from_bilinears(num_modes=3, bilinears=bilinear_entries(jw))
        for j in range(6):
            for k in range(j + 1, 6):
                c_bl, p_bl = bl.bilinear(j, k)
                c_jw, p_jw = jw.bilinear(j, k)
                assert p_bl == p_jw, f"bilinear({j},{k}): {p_bl} != {p_jw}"
                assert abs(c_bl - c_jw) < 1e-12, f"bilinear({j},{k}) coeff: {c_bl} != {c_jw}"

    def test_bilinear_antisymmetry(self) -> None:
        """bilinear(k,j) = -bilinear(j,k) for bilinear-only mappings."""
        jw = MajoranaMapping.jordan_wigner(num_modes=2)
        bl = MajoranaMapping.from_bilinears(num_modes=2, bilinears=bilinear_entries(jw))
        for j in range(4):
            for k in range(j + 1, 4):
                c_fwd, p_fwd = bl.bilinear(j, k)
                c_rev, p_rev = bl.bilinear(k, j)
                assert p_fwd == p_rev
                assert abs(c_fwd + c_rev) < 1e-12

    def test_majorana_raises_for_bilinear_only(self) -> None:
        """majorana(k) raises ValueError for bilinear-only mappings."""
        jw = MajoranaMapping.jordan_wigner(num_modes=2)
        bl = MajoranaMapping.from_bilinears(num_modes=2, bilinears=bilinear_entries(jw))
        with pytest.raises(ValueError, match="bilinear-only"):
            bl.majorana(0)

    def test_num_qubits(self) -> None:
        """num_qubits is derived from the bilinear Pauli words."""
        jw = MajoranaMapping.jordan_wigner(num_modes=4)
        bl = MajoranaMapping.from_bilinears(num_modes=4, bilinears=bilinear_entries(jw))
        assert bl.num_qubits == 4

    def test_wrong_count_raises(self) -> None:
        """from_bilinears raises ValueError if entry count doesn't match num_modes."""
        with pytest.raises(ValueError, match="upper-triangle"):
            MajoranaMapping.from_bilinears(num_modes=2, bilinears=[(1.0, label_to_word("IZ"))])

    def test_missing_entry_raises(self) -> None:
        """from_bilinears raises ValueError if a required (j,k) pair is missing."""
        with pytest.raises(ValueError, match="upper-triangle entries"):
            MajoranaMapping.from_bilinears(
                num_modes=2,
                bilinears=[
                    (1.0, label_to_word("IZ")),
                    (1.0, label_to_word("XZ")),
                    (1.0, label_to_word("YZ")),
                    (1.0, label_to_word("XI")),
                    # One entry intentionally omitted.
                    (1.0, label_to_word("ZI")),
                ],
            )


# ─── Validation Tests ────────────────────────────────────────────────────


class TestValidation:
    """Tests for input validation."""

    def test_empty_table(self) -> None:
        """Empty table raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            MajoranaMapping.from_table([])

    def test_odd_length_table(self) -> None:
        """Odd-length table raises ValueError."""
        with pytest.raises(ValueError, match="even number"):
            MajoranaMapping.from_table([label_to_word(s) for s in ("IX", "IY", "XZ")])

    def test_invalid_characters(self) -> None:
        """Non-IXYZ characters raise ValueError."""
        with pytest.raises(ValueError, match="Invalid Pauli character"):
            label_to_word("IA")

    def test_zero_modes(self) -> None:
        """Zero modes raises ValueError in factories."""
        with pytest.raises(ValueError, match="num_modes"):
            MajoranaMapping.jordan_wigner(num_modes=0)
        with pytest.raises(ValueError, match="num_modes"):
            MajoranaMapping.bravyi_kitaev(num_modes=0)
        with pytest.raises(ValueError, match="num_modes"):
            MajoranaMapping.parity(num_modes=0)


# ─── Immutability Tests ─────────────────────────────────────────────────


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


# ─── SCBK (symmetry-conserving Bravyi-Kitaev) ────────────────────────────


class TestSymmetryConservingBravyiKitaev:
    """Tests for the SCBK factory method and TaperingSpecification integration."""

    def test_scbk_factory_creates_bk_tree_table(self) -> None:
        """SCBK uses the BK-tree table as its base encoding."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        scbk = MajoranaMapping.symmetry_conserving_bravyi_kitaev(8, Symmetries(2, 2))
        bk_tree = MajoranaMapping.bravyi_kitaev_tree(8)
        assert scbk.table == bk_tree.table

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

    def test_scbk_eigenvalues_depend_on_alpha_beta(self) -> None:
        """Different (n_alpha, n_beta) produce different eigenvalues."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        scbk_11 = MajoranaMapping.symmetry_conserving_bravyi_kitaev(8, Symmetries(1, 1))
        scbk_20 = MajoranaMapping.symmetry_conserving_bravyi_kitaev(8, Symmetries(2, 0))
        assert scbk_11.tapering.eigenvalues != scbk_20.tapering.eigenvalues

    def test_scbk_num_qubits_is_base_register_size(self) -> None:
        """MajoranaMapping.num_qubits reflects the base encoding register."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        scbk = MajoranaMapping.symmetry_conserving_bravyi_kitaev(8, Symmetries(2, 2))
        assert scbk.num_qubits == 8

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

    def test_parity_with_symmetries_has_tapering(self) -> None:
        """Parity with symmetries bundles a TaperingSpecification."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        par = MajoranaMapping.parity(8, Symmetries(2, 2))
        assert par.tapering is not None
        assert par.tapering.num_tapered == 2
        assert par.name == "parity-2q-reduced"
        assert par.base_encoding == "parity"
        assert par.num_qubits == 8

    def test_without_tapering(self) -> None:
        """without_tapering strips tapering but preserves the base table."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        scbk = MajoranaMapping.symmetry_conserving_bravyi_kitaev(8, Symmetries(2, 2))
        base = scbk.without_tapering()
        bk_tree = MajoranaMapping.bravyi_kitaev_tree(8)
        assert base.tapering is None
        assert base.table == bk_tree.table
        assert base.num_qubits == 8


# ─── Bilinear primitive ──────────────────────────────────────────────────


def _multiply_words(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> tuple[complex, list[tuple[int, int]]]:
    """Multiply two sparse Pauli words."""
    return PauliTermAccumulator.multiply_uncached(a, b)


_FACTORIES = [
    ("jordan_wigner", lambda n: MajoranaMapping.jordan_wigner(num_modes=n)),
    ("bravyi_kitaev", lambda n: MajoranaMapping.bravyi_kitaev(num_modes=n)),
    ("parity", lambda n: MajoranaMapping.parity(num_modes=n)),
]


class TestBilinear:
    """Tests for the bilinear primitive ``i·gamma_j·gamma_k``."""

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
                expected_phase, expected_word = _multiply_words(m.majorana(j), m.majorana(k))
                expected_coeff = 1j * expected_phase
                bcoeff, bword = m.bilinear(j, k)
                assert bword == expected_word, f"({j},{k}): word {bword} != {expected_word}"
                assert abs(bcoeff - expected_coeff) < 1e-12, f"({j},{k}): coeff {bcoeff} != {expected_coeff}"

    @pytest.mark.parametrize(("name", "factory"), _FACTORIES)
    @pytest.mark.parametrize("n_modes", [2, 4])
    def test_squares_to_identity(self, name: str, factory, n_modes: int) -> None:
        """``(i·gamma_j·gamma_k)² == I`` for all distinct j, k."""
        del name
        m = factory(n_modes)
        n = 2 * n_modes
        for j in range(n):
            for k in range(n):
                if j == k:
                    continue
                coeff, word = m.bilinear(j, k)
                phase, prod = _multiply_words(word, word)
                assert prod == [], f"({j},{k}): word² = {prod}, not identity"
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
                p_ab, w_ab = _multiply_words(w1, w2)
                p_ba, w_ba = _multiply_words(w2, w1)
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
        for (j, k), expected in {
            (0, 1): (-1 + 0j, "IZ"),
            (1, 0): (1 + 0j, "IZ"),
            (0, 2): (1 + 0j, "XY"),
            (2, 3): (-1 + 0j, "ZI"),
        }.items():
            coeff, word = m.bilinear(j, k)
            assert coeff == expected[0]
            assert word_to_label(word, m.num_qubits) == expected[1]

    def test_raises_on_equal_indices(self) -> None:
        """``bilinear(j, j)`` is undefined and must raise ValueError."""
        m = MajoranaMapping.jordan_wigner(num_modes=2)
        with pytest.raises(ValueError, match="distinct"):
            m.bilinear(0, 0)

    def test_raises_on_out_of_range(self) -> None:
        """Out-of-range indices raise IndexError."""
        m = MajoranaMapping.jordan_wigner(num_modes=2)
        with pytest.raises(IndexError):
            m.bilinear(0, 4)

    def test_majorana_out_of_range(self) -> None:
        """``majorana(k)`` raises IndexError on out-of-range k."""
        m = MajoranaMapping.jordan_wigner(num_modes=2)
        with pytest.raises(IndexError):
            m.majorana(4)
        with pytest.raises(IndexError):
            m.majorana(99)


class TestEncodingMetadata:
    """Tests for encoding metadata properties."""

    @pytest.mark.parametrize(("name", "factory"), _FACTORIES)
    @pytest.mark.parametrize("n_modes", [2, 4])
    def test_is_majorana_atomic(self, name: str, factory, n_modes: int) -> None:
        """All current encodings are Majorana-atomic."""
        del name
        m = factory(n_modes)
        assert m.is_majorana_atomic is True

    def test_pauli_string_length_with_tapering(self) -> None:
        """For tapered SCBK, bilinear/majorana operate in the pre-taper basis."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        scbk = MajoranaMapping.symmetry_conserving_bravyi_kitaev(8, Symmetries(2, 2))
        assert scbk.num_qubits == 8
        for j in range(2 * scbk.num_modes):
            assert len(word_to_label(scbk.majorana(j), scbk.num_qubits)) == 8
            for k in range(2 * scbk.num_modes):
                if j == k:
                    continue
                _, w = scbk.bilinear(j, k)
                assert len(word_to_label(w, scbk.num_qubits)) == 8


# ─── TaperingSpecification Serialization ────────────────────────────────


class TestTaperingSpecificationSerialization:
    """Serialization round-trip tests for TaperingSpecification."""

    def test_json_roundtrip_via_mapping(self) -> None:
        """TaperingSpecification survives a JSON round-trip through MajoranaMapping."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        scbk = MajoranaMapping.symmetry_conserving_bravyi_kitaev(8, Symmetries(2, 2))
        data = scbk.to_json()
        loaded = MajoranaMapping.from_json(data)
        assert loaded.tapering is not None
        assert loaded.tapering.qubit_indices == scbk.tapering.qubit_indices
        assert loaded.tapering.eigenvalues == scbk.tapering.eigenvalues
        assert loaded.tapering.num_tapered == scbk.tapering.num_tapered

    def test_hdf5_roundtrip_via_mapping(self) -> None:
        """TaperingSpecification survives an HDF5 round-trip through MajoranaMapping."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        scbk = MajoranaMapping.symmetry_conserving_bravyi_kitaev(8, Symmetries(2, 2))
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            with h5py.File(f.name, "w") as hf:
                scbk.to_hdf5(hf)
            with h5py.File(f.name, "r") as hf:
                loaded = MajoranaMapping.from_hdf5(hf)
        assert loaded.tapering is not None
        assert loaded.tapering.qubit_indices == scbk.tapering.qubit_indices
        assert loaded.tapering.eigenvalues == scbk.tapering.eigenvalues

    def test_json_contains_tapering_fields(self) -> None:
        """TaperingSpecification.to_json() produces the expected structure."""
        import json  # noqa: PLC0415

        from qdk_chemistry.data import Symmetries, TaperingSpecification  # noqa: PLC0415

        tap = TaperingSpecification.symmetry_conserving_bravyi_kitaev(8, Symmetries(2, 2))
        data = json.loads(tap.to_json())
        assert "qubit_indices" in data
        assert "eigenvalues" in data
        reconstructed = TaperingSpecification(
            qubit_indices=list(data["qubit_indices"]),
            eigenvalues=list(data["eigenvalues"]),
        )
        assert reconstructed == tap

    def test_hdf5_standalone_roundtrip(self) -> None:
        """TaperingSpecification survives a standalone HDF5 round-trip."""
        from qdk_chemistry.data import Symmetries, TaperingSpecification  # noqa: PLC0415

        tap = TaperingSpecification.symmetry_conserving_bravyi_kitaev(8, Symmetries(2, 2))
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            tap.to_hdf5_file(f.name)
            with h5py.File(f.name, "r") as hf:
                assert "qubit_indices" in hf
                assert "eigenvalues" in hf
                reconstructed = TaperingSpecification(
                    qubit_indices=[int(x) for x in hf["qubit_indices"][:]],
                    eigenvalues=[int(x) for x in hf["eigenvalues"][:]],
                )
        assert reconstructed == tap

    def test_parity_tapering_json_roundtrip_via_mapping(self) -> None:
        """Parity two-qubit reduction tapering survives a JSON round-trip."""
        from qdk_chemistry.data import Symmetries  # noqa: PLC0415

        par = MajoranaMapping.parity(8, Symmetries(2, 2))
        data = par.to_json()
        loaded = MajoranaMapping.from_json(data)
        assert loaded.tapering is not None
        assert loaded.tapering.qubit_indices == par.tapering.qubit_indices
        assert loaded.tapering.eigenvalues == par.tapering.eigenvalues
