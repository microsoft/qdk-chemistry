"""Tests for phase-1 bosonic Hamiltonian support.

Covers the :class:`~qdk_chemistry.data.BosonicModes` basis, the
:class:`~qdk_chemistry.data.BosonMapping` boson-to-qubit encodings, the
Bose-Hubbard model builder and the ``boson_qubit_mapper`` algorithm surface.

The numerical reference values are the verified fixtures of the boson-encoding
research report; the structural checks additionally rebuild every Hamiltonian
in the occupation basis with an independent NumPy Kronecker construction, so
the library result is cross-checked against an oracle that shares no code with
it.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import itertools
import json
from typing import ClassVar

import numpy as np
import pytest

from qdk_chemistry.algorithms import create, registry
from qdk_chemistry.algorithms.boson_qubit_mapper import QdkBosonQubitMapper
from qdk_chemistry.data import (
    BosonicModes,
    BosonMapping,
    LatticeGraph,
    ModelOrbitals,
    Orbitals,
    QubitOperator,
)
from qdk_chemistry.utils.model_hamiltonians import create_bose_hubbard_hamiltonian

TOL = 1e-10

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_matrix(label: str) -> np.ndarray:
    """Dense matrix of a Pauli label using the library's qubit-0-is-rightmost convention.

    Args:
        label: Pauli string whose leftmost character is the most significant qubit.

    Returns:
        numpy.ndarray: The ``2**len(label)`` square matrix ``kron(label[0], ..., label[-1])``.

    """
    out = np.array([[1.0 + 0.0j]])
    for char in label:
        out = np.kron(out, _PAULI[char])
    return out


def _qubit_operator_matrix(operator: QubitOperator) -> np.ndarray:
    """Dense matrix of a QubitOperator.

    Args:
        operator: The operator to densify.

    Returns:
        numpy.ndarray: Sum of ``coefficient * pauli_matrix(label)`` over all terms.

    """
    nq = operator.num_qubits
    total = np.zeros((2**nq, 2**nq), dtype=complex)
    for label, coefficient in zip(operator.pauli_strings, operator.coefficients, strict=True):
        total += complex(coefficient) * _pauli_matrix(label)
    return total


def _annihilation(dim: int) -> np.ndarray:
    """Truncated single-mode annihilation operator.

    Args:
        dim: Local Fock-space dimension ``d``.

    Returns:
        numpy.ndarray: ``b = sum_{n=1}^{d-1} sqrt(n) |n-1><n|``.

    """
    out = np.zeros((dim, dim))
    for n in range(1, dim):
        out[n - 1, n] = np.sqrt(n)
    return out


def _embed(modes: int, dim: int, factors: dict[int, np.ndarray]) -> np.ndarray:
    """Embed single-mode operators into the multi-mode Fock space.

    Mode 0 is the most significant Kronecker factor, matching the row-major
    occupation ordering ``(n_0, ..., n_{L-1})``.

    Args:
        modes: Number of modes ``L``.
        dim: Local dimension ``d`` of each mode.
        factors: Map from mode index to that mode's single-mode operator.

    Returns:
        numpy.ndarray: The embedded ``d**L`` square matrix.

    """
    out = np.array([[1.0]])
    for i in range(modes):
        out = np.kron(out, factors.get(i, np.eye(dim)))
    return out


def _bose_hubbard_matrix(case: tuple[int, int, float, float, float]) -> np.ndarray:
    """Independent Kronecker construction of the Bose-Hubbard Hamiltonian on a chain.

    Args:
        case: The ``(modes, dim, t, u, mu)`` model parameters.

    Returns:
        numpy.ndarray: The ``d**L`` square Hamiltonian matrix.

    """
    modes, dim, t, u, mu = case
    b = _annihilation(dim)
    bdag = b.T
    number = bdag @ b
    on_site = 0.5 * u * (number @ (number - np.eye(dim))) - mu * number

    total = np.zeros((dim**modes, dim**modes))
    for i in range(modes):
        total += _embed(modes, dim, {i: on_site})
    for i in range(modes - 1):
        hop = _embed(modes, dim, {i: bdag, i + 1: b})
        total += -t * (hop + hop.T)
    return total


NAMED_ENCODINGS = [BosonMapping.standard_binary, BosonMapping.gray_code]


def _mapped(case: tuple[int, int, float, float, float], factory=BosonMapping.standard_binary) -> QubitOperator:
    """Build and map a Bose-Hubbard chain in one step.

    Args:
        case: The ``(modes, dim, t, u, mu)`` model parameters.
        factory: Named ``BosonMapping`` factory selecting the encoding.

    Returns:
        QubitOperator: The encoded Hamiltonian.

    """
    modes, dim, t, u, mu = case
    hamiltonian = create_bose_hubbard_hamiltonian(LatticeGraph.chain(modes), t=t, U=u, mu=mu, mode_dimension=dim)
    return create("boson_qubit_mapper").run(hamiltonian, factory(modes, dim))


def _terms(operator: QubitOperator) -> dict[str, complex]:
    """Collect a QubitOperator into a label to coefficient dictionary.

    Args:
        operator: The operator to collect.

    Returns:
        dict[str, complex]: One entry per distinct Pauli label.

    """
    out: dict[str, complex] = {}
    for label, coefficient in zip(operator.pauli_strings, operator.coefficients, strict=True):
        out[label] = out.get(label, 0.0) + complex(coefficient)
    return out


class TestBosonicModes:
    """Tests for the BosonicModes single-particle basis."""

    def test_construction_and_accessors(self):
        """A uniform-cutoff basis reports its modes and dimensions."""
        modes = BosonicModes(4, 8)
        assert modes.num_modes() == 4
        assert modes.get_num_molecular_orbitals() == 4
        assert modes.mode_dimension(0) == 8
        assert modes.mode_dimension(3) == 8
        assert modes.max_occupation(0) == 7
        assert modes.uniform_dimension() == 8
        assert modes.has_power_of_two_dimensions()
        assert modes.fock_space_dimension() == 8**4
        # The cutoff is attributed per mode: one stored entry per mode.
        assert modes.mode_dimensions() == [8, 8, 8, 8]

    def test_is_an_orbitals_and_model_orbitals(self):
        """BosonicModes participates in the existing single-particle basis hierarchy."""
        modes = BosonicModes(3, 4)
        assert isinstance(modes, ModelOrbitals)
        assert isinstance(modes, Orbitals)

    def test_rejects_invalid_arguments(self):
        """Zero modes and a dimension below two are rejected."""
        with pytest.raises(ValueError, match="num_modes must be at least 1"):
            BosonicModes(0, 4)
        with pytest.raises(ValueError, match="must be at least 2"):
            BosonicModes(2, 1)

    def test_stores_the_cutoff_verbatim(self):
        """The requested cutoff is stored exactly; padding is opt-in."""
        assert BosonicModes(2, 3).mode_dimension(0) == 3
        assert not BosonicModes(2, 3).has_power_of_two_dimensions()

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [(1, 2), (2, 2), (3, 4), (4, 4), (5, 8), (8, 8), (9, 16), (33, 64)],
    )
    def test_padded_dimension(self, requested, expected):
        """Padding rounds up to the next power of two.

        Args:
            requested: Requested local dimension.
            expected: Expected padded dimension.

        """
        assert BosonicModes.padded_dimension(requested) == expected
        padded = BosonicModes.padded_to_power_of_two(2, requested)
        assert padded.mode_dimensions() == [expected, expected]
        # The instance overload pads without mutating the original basis.
        original = BosonicModes(2, requested) if requested >= 2 else None
        if original is not None:
            assert original.with_padded_dimensions().mode_dimensions() == [expected, expected]
            assert original.mode_dimensions() == [requested, requested]

    def test_json_round_trip_through_orbitals_dispatch(self):
        """The serialized payload carries the type tag and reloads as a BosonicModes."""
        modes = BosonicModes(3, 8)
        payload = json.loads(modes.to_json())
        assert payload["type"] == "BosonicModes"
        # Serialized per mode from day one, so heterogeneous bases round-trip
        # later with no schema change and no version bump.
        assert payload["mode_dimensions"] == [8, 8, 8]

        restored = BosonicModes.from_json(modes.to_json())
        assert isinstance(restored, BosonicModes)
        assert restored.num_modes() == 3
        assert restored.mode_dimension(0) == 8

        # NOTE: the generic ``Orbitals.from_json`` binding returns by value and
        # therefore slices; it raises ``IndexError`` for *any* model basis --
        # ``Orbitals.from_json(ModelOrbitals(4).to_json())`` fails identically
        # on an unmodified tree -- so it is not exercised here. The polymorphic
        # C++ ``Orbitals::from_json`` dispatch is covered by the C++ suite
        # (``BosonicModes.JsonRoundTripThroughOrbitalsDispatch``), and
        # ``Hamiltonian.get_orbitals()`` preserves the bosonic type in Python
        # because it returns a shared pointer.
        assert issubclass(BosonicModes, Orbitals)

    def test_existing_model_orbitals_json_still_loads(self):
        """Files without the new type tag continue to deserialize unchanged."""
        payload = json.loads(ModelOrbitals(4).to_json())
        assert payload["type"] == "ModelOrbitals"
        restored = ModelOrbitals.from_json(ModelOrbitals(4).to_json())
        assert isinstance(restored, ModelOrbitals)
        assert not isinstance(restored, BosonicModes)
        assert restored.get_num_molecular_orbitals() == 4

    def test_content_hash_separates_cutoffs(self):
        """Two bases differing only in cutoff hash differently."""
        assert BosonicModes(3, 4).content_hash() != BosonicModes(3, 8).content_hash()
        assert BosonicModes(3, 4).content_hash() == BosonicModes(3, 4).content_hash()


class TestBosonMapping:
    """Tests for the boson-to-qubit mapping primitives."""

    def test_standard_binary_codewords(self):
        """Standard binary maps occupation n to the bit pattern n."""
        mapping = BosonMapping.standard_binary(1, 8)
        assert mapping.codeword_table(0) == [0, 1, 2, 3, 4, 5, 6, 7]
        assert mapping.qubits_per_mode(0) == 3
        assert mapping.num_qubits() == 3
        assert mapping.name == "standard-binary"

    def test_gray_code_codewords(self):
        """Gray code maps occupation n to n XOR (n >> 1)."""
        mapping = BosonMapping.gray_code(1, 8)
        assert mapping.codeword_table(0) == [0, 1, 3, 2, 6, 7, 5, 4]
        assert [mapping.level(0, c) for c in mapping.codeword_table(0)] == list(range(8))

    def test_isometry_is_a_permutation(self):
        """A power-of-two cutoff gives a surjective code, hence no leakage."""
        for factory in NAMED_ENCODINGS:
            iso = factory(1, 8).isometry(0)
            assert iso.shape == (8, 8)
            np.testing.assert_allclose(iso @ iso.T, np.eye(8), atol=TOL)
            np.testing.assert_allclose(iso.T @ iso, np.eye(8), atol=TOL)

    def test_mode_qubit_layout(self):
        """Mode 0 owns the most significant (leftmost) qubit block."""
        mapping = BosonMapping.standard_binary(3, 4)
        assert mapping.mode_qubits(0) == [4, 5]
        assert mapping.mode_qubits(1) == [2, 3]
        assert mapping.mode_qubits(2) == [0, 1]

    def test_rejects_non_power_of_two_cutoff(self):
        """A non-power-of-two cutoff is a hard error naming the padding helper."""
        with pytest.raises(ValueError, match="padded_to_power_of_two"):
            BosonMapping.standard_binary(2, 3)
        # The message must name the mode, the value, the next power of two and
        # the fact that padding costs nothing.
        with pytest.raises(ValueError, match="mode 0"):
            BosonMapping.standard_binary(2, 3)
        with pytest.raises(ValueError, match="32 hopping terms"):
            BosonMapping.standard_binary(2, 3)

    def test_basis_factory_validates_the_cutoff(self):
        """The basis overload reads the cutoff from the basis and rejects a mismatch."""
        modes = BosonicModes(2, 4)
        mapping = BosonMapping.standard_binary(modes)
        assert mapping.mode_dimension(0) == 4
        assert mapping.mode_dimensions() == [4, 4]
        assert mapping.uniform_dimension() == 4
        mapping.validate_basis(modes)
        with pytest.raises(ValueError, match="dimension"):
            mapping.validate_basis(BosonicModes(2, 8))
        with pytest.raises(ValueError, match="mode"):
            mapping.validate_basis(BosonicModes(3, 4))

    def test_number_operator_closed_form(self):
        """n_hat = 1.5 I - 0.5 IZ - 1.0 ZI for standard binary at d = 4 (report section 5.6)."""
        mapping = BosonMapping.standard_binary(1, 4)
        terms = {_label(word, 2): coefficient for word, coefficient in mapping.number(0)}
        assert terms == pytest.approx({"II": 1.5, "IZ": -0.5, "ZI": -1.0}, abs=TOL)

    def test_number_times_number_minus_one_closed_form(self):
        """n(n-1) at d = 8 matches the published seven-term decomposition."""
        mapping = BosonMapping.standard_binary(1, 8)
        terms = {_label(word, 3): coefficient for word, coefficient in mapping.number_times_number_minus_one(0)}
        assert terms == pytest.approx(
            {
                "III": 14.0,
                "IIZ": -3.0,
                "IZI": -6.0,
                "ZII": -12.0,
                "IZZ": 1.0,
                "ZIZ": 2.0,
                "ZZI": 4.0,
            },
            abs=TOL,
        )

    def test_number_times_number_minus_one_vanishes_for_hard_core_bosons(self):
        """At d = 2 the on-site interaction is identically zero."""
        assert BosonMapping.standard_binary(1, 2).number_times_number_minus_one(0) == []

    def test_annihilation_closed_form(self):
        """The annihilation operator at d = 4 matches the eight-term closed form.

        With ``b = |0><1| + sqrt(2)|1><2| + sqrt(3)|2><3|`` and the outer-product
        identities of the encoding report, the standard-binary decomposition is
        ``(1 +- sqrt(3))/4`` on ``IX``/``IY`` and ``ZX``/``ZY`` and ``sqrt(2)/4``
        on the four ``{X,Y}x{X,Y}`` words. Cross-checked against the research
        harness oracle ``paulicore.decompose_dense(paulicore.b_op(4))``.
        """
        mapping = BosonMapping.standard_binary(1, 4)
        terms = {_label(word, 2): coefficient for word, coefficient in mapping.annihilation(0)}
        a = (1.0 + np.sqrt(3.0)) / 4.0
        c = (1.0 - np.sqrt(3.0)) / 4.0
        e = np.sqrt(2.0) / 4.0
        assert terms == pytest.approx(
            {
                "IX": a,
                "IY": 1j * a,
                "ZX": c,
                "ZY": 1j * c,
                "XX": e,
                "YY": e,
                "XY": -1j * e,
                "YX": 1j * e,
            },
            abs=TOL,
        )

    @pytest.mark.parametrize("dim", [2, 4, 8, 16])
    @pytest.mark.parametrize("factory", NAMED_ENCODINGS)
    def test_primitives_reproduce_their_dense_matrices(self, dim, factory):
        """Every single-mode primitive is an exact Pauli decomposition.

        Args:
            dim: Local Fock-space dimension.
            factory: Named ``BosonMapping`` factory selecting the encoding.

        """
        mapping = factory(1, dim)
        nq = mapping.qubits_per_mode(0)
        iso = mapping.isometry(0)

        b = _annihilation(dim)
        number = b.T @ b
        expected = {
            "number": number,
            "creation": b.T,
            "annihilation": b,
            "number_squared": number @ number,
            "number_times_number_minus_one": number @ (number - np.eye(dim)),
        }
        for name, dense in expected.items():
            terms = getattr(mapping, name)(0)
            total = np.zeros((2**nq, 2**nq), dtype=complex)
            for word, coefficient in terms:
                total += complex(coefficient) * _pauli_matrix(_label(word, nq))
            np.testing.assert_allclose(total, iso @ dense @ iso.T, atol=TOL, err_msg=name)

    def test_diagonal_is_general(self):
        """The Walsh-Hadamard diagonal routine handles arbitrary spectra."""
        mapping = BosonMapping.standard_binary(1, 8)
        values = [0.0, -1.5, 2.25, 7.0, -0.125, 3.0, 11.0, -4.0]
        total = np.zeros((8, 8), dtype=complex)
        for word, coefficient in mapping.diagonal(values, 0):
            total += complex(coefficient) * _pauli_matrix(_label(word, 3))
        np.testing.assert_allclose(total, np.diag(values), atol=TOL)

    def test_json_round_trip(self):
        """A mapping survives a JSON round trip."""
        mapping = BosonMapping.gray_code(3, 8)
        restored = BosonMapping.from_json(mapping.to_json())
        assert restored.num_modes() == 3
        assert restored.mode_dimension(0) == 8
        assert restored.mode_dimensions() == [8, 8, 8]
        assert restored.name == "gray-code"
        assert restored.codeword_table(0) == mapping.codeword_table(0)
        assert json.loads(mapping.to_json())["mode_dimensions"] == [8, 8, 8]

    def test_per_mode_cutoffs_are_the_stored_truth(self):
        """The cutoff is attributed per mode even though phase 1 builds it uniformly."""
        # Phase-1 public construction is uniform-only, so ``uniform_dimension``
        # is always engaged; the ``None`` branch and the per-mode block layout
        # are exercised by the C++ suite through a deserialized heterogeneous
        # basis (``BosonMapping.HeterogeneousCutoffsLayOutBlocksByPerModeWidth``).
        mapping = BosonMapping.standard_binary(3, 4)
        assert mapping.mode_dimensions() == [4, 4, 4]
        assert mapping.uniform_dimension() == 4
        assert [mapping.qubits_per_mode(i) for i in range(3)] == [2, 2, 2]
        assert mapping.num_qubits() == sum(mapping.qubits_per_mode(i) for i in range(3))
        with pytest.raises(IndexError):
            mapping.mode_dimension(3)
        with pytest.raises(IndexError):
            mapping.qubits_per_mode(3)


class TestBoseHubbardBuilder:
    """Tests for the Bose-Hubbard model Hamiltonian builder."""

    def test_integrals_use_the_chemist_convention(self):
        """h_ii = -mu, h_ij = -t on bonds, (ii|ii) = U."""
        hamiltonian = create_bose_hubbard_hamiltonian(LatticeGraph.chain(4), t=0.7, U=3.3, mu=0.9, mode_dimension=4)
        h1 = hamiltonian.get_one_body_integrals()[0]
        for i in range(4):
            assert h1[i, i] == pytest.approx(-0.9, abs=TOL)
            assert hamiltonian.get_two_body_element(i, i, i, i) == pytest.approx(3.3, abs=TOL)
        for i in range(3):
            assert h1[i, i + 1] == pytest.approx(-0.7, abs=TOL)
            assert h1[i + 1, i] == pytest.approx(-0.7, abs=TOL)

    def test_carries_a_bosonic_basis(self):
        """The builder attaches a BosonicModes basis holding the cutoff."""
        hamiltonian = create_bose_hubbard_hamiltonian(LatticeGraph.chain(3), t=1.0, U=4.0, mu=0.0, mode_dimension=8)
        orbitals = hamiltonian.get_orbitals()
        assert isinstance(orbitals, BosonicModes)
        assert orbitals.num_modes() == 3
        assert orbitals.mode_dimension(0) == 8


class TestBosonQubitMapperAlgorithm:
    """Tests for the boson_qubit_mapper algorithm surface."""

    def test_registered_in_the_factory_registry(self):
        """The new algorithm type is discoverable through the registry."""
        assert "qdk" in registry.available("boson_qubit_mapper")
        assert registry.show_default("boson_qubit_mapper") == "qdk"
        mapper = create("boson_qubit_mapper")
        assert isinstance(mapper, QdkBosonQubitMapper)
        assert mapper.type_name() == "boson_qubit_mapper"
        assert mapper.name() == "qdk"

    def test_existing_qubit_mapper_type_is_untouched(self):
        """Adding a bosonic type does not disturb the fermionic mapper."""
        mapper = create("qubit_mapper")
        assert mapper.type_name() == "qubit_mapper"
        assert mapper.name() == "qdk"
        assert registry.show_default("qubit_mapper") == "qdk"

    def test_mode_count_mismatch_is_a_hard_error(self):
        """A mapping sized for the wrong number of modes is rejected."""
        hamiltonian = create_bose_hubbard_hamiltonian(LatticeGraph.chain(2), t=1.0, U=4.0, mu=0.0, mode_dimension=4)
        with pytest.raises(ValueError, match="modes"):
            create("boson_qubit_mapper").run(hamiltonian, BosonMapping.standard_binary(3, 4))

    def test_cutoff_mismatch_is_a_hard_error(self):
        """A mapping whose cutoff disagrees with the basis is rejected."""
        hamiltonian = create_bose_hubbard_hamiltonian(LatticeGraph.chain(2), t=1.0, U=4.0, mu=0.0, mode_dimension=4)
        with pytest.raises(ValueError, match="dimension"):
            create("boson_qubit_mapper").run(hamiltonian, BosonMapping.standard_binary(2, 8))

    def test_encoding_metadata_is_recorded(self):
        """The mapped operator records which encoding produced it."""
        operator = _mapped((2, 4, 1.0, 4.0, 0.0), BosonMapping.gray_code)
        assert operator.encoding == "gray-code"
        assert operator.num_qubits == 4


class TestVerifiedFixtures:
    """Fixtures taken verbatim from the verified boson-encoding report."""

    def test_fixture_1_two_hard_core_modes(self):
        """L = 2, d = 2, t = 1, U = 4, mu = 0 gives -0.5 (XX + YY)."""
        terms = _terms(_mapped((2, 2, 1.0, 4.0, 0.0)))
        assert len(terms) == 2
        assert terms == pytest.approx({"XX": -0.5, "YY": -0.5}, abs=TOL)

    def test_fixture_4_three_hard_core_modes(self):
        """L = 3, d = 2, t = 1, U = 8, mu = 0 gives -0.5 on the two nearest-neighbour bonds."""
        terms = _terms(_mapped((3, 2, 1.0, 8.0, 0.0)))
        assert len(terms) == 4
        assert terms == pytest.approx({"IXX": -0.5, "IYY": -0.5, "XXI": -0.5, "YYI": -0.5}, abs=TOL)

    def test_fixture_2_selected_terms_and_count(self):
        """L = 2, d = 4, t = 1, U = 4, mu = 0 gives the published 39-term operator."""
        terms = _terms(_mapped((2, 4, 1.0, 4.0, 0.0)))
        assert len(terms) == 39
        expected = {
            "IIII": 8.0,
            "IIIZ": -2.0,
            "IIZI": -4.0,
            "IIZZ": 2.0,
            "IXIX": -0.933012701892219,
            "ZXZX": -0.066987298107781,
            "ZZII": 2.0,
            "XYYX": 0.25,
        }
        for label, value in expected.items():
            assert terms[label] == pytest.approx(value, abs=TOL), label

    def test_fixture_2_spectrum(self):
        """The encoded operator reproduces the published full-space spectrum."""
        matrix = _qubit_operator_matrix(_mapped((2, 4, 1.0, 4.0, 0.0)))
        np.testing.assert_allclose(matrix, matrix.conj().T, atol=TOL)
        eigenvalues = np.linalg.eigvalsh(matrix)
        expected = [
            -1.0,
            -0.828427124746190,
            0.0,
            1.0,
            1.708497377870826,
            4.0,
            4.828427124746191,
            5.535898384862252,
            6.0,
            12.0,
            12.291502622129185,
            12.464101615137757,
            13.0,
            14.0,
            19.0,
            24.0,
        ]
        np.testing.assert_allclose(eigenvalues, expected, atol=1e-9)


class TestEndToEnd:
    """End-to-end agreement between the encoded operator and exact diagonalization."""

    CASES: ClassVar[list[tuple[int, int, float, float, float]]] = [
        (2, 2, 1.0, 0.0, 0.0),
        (2, 2, 1.0, 4.0, 0.0),
        (2, 2, 0.7, 3.3, 0.9),
        (2, 4, 1.0, 0.0, 0.0),
        (2, 4, 1.0, 4.0, 0.0),
        (2, 4, 1.0, 8.0, 0.0),
        (2, 4, 0.7, 3.3, 0.9),
        (3, 2, 0.7, 3.3, 0.9),
        (3, 4, 1.0, 4.0, 0.0),
        (2, 8, 0.5, 2.0, -1.0),
    ]

    ENERGIES: ClassVar[list[tuple[tuple[int, int, float, float, float], float, float]]] = [
        ((2, 2, 1.0, 0.0, 0.0), -1.0, 0.0),
        ((2, 2, 1.0, 4.0, 0.0), -1.0, 0.0),
        ((2, 2, 0.7, 3.3, 0.9), -1.8, -1.6),
        ((2, 4, 1.0, 0.0, 0.0), -3.464101615137755, -3.0),
        ((2, 4, 1.0, 4.0, 0.0), -1.0, -0.828427124746190),
        ((2, 4, 1.0, 8.0, 0.0), -1.0, -0.472135954999579),
        ((2, 4, 0.7, 3.3, 0.9), -2.313908500838239, -1.6),
        ((3, 2, 1.0, 4.0, 0.0), -1.414213562373095, -1.414213562373095),
        ((3, 2, 0.7, 3.3, 0.9), -2.789949493661166, -2.7),
        ((3, 4, 1.0, 4.0, 0.0), -2.0, -1.723195724029204),
        ((3, 4, 0.7, 3.3, 0.9), -3.763628488935746, -3.154895883864022),
    ]

    PERMUTATION_CASES: ClassVar[list[tuple[int, int, float, float, float]]] = [
        (2, 2, 0.7, 3.3, 0.9),
        (2, 4, 0.7, 3.3, 0.9),
        (3, 4, 0.7, 3.3, 0.9),
        (2, 8, 0.7, 3.3, 0.9),
    ]

    @pytest.mark.parametrize("case", CASES)
    @pytest.mark.parametrize("factory", NAMED_ENCODINGS)
    def test_encoded_operator_matches_the_kronecker_construction(self, case, factory):
        """The encoded matrix equals the occupation-basis Hamiltonian permuted by the code.

        Args:
            case: The ``(modes, dim, t, u, mu)`` model parameters.
            factory: Named ``BosonMapping`` factory selecting the encoding.

        """
        modes, dim = case[0], case[1]
        encoded = _qubit_operator_matrix(_mapped(case, factory))
        np.testing.assert_allclose(encoded, encoded.conj().T, atol=TOL)

        codewords = factory(modes, dim).codeword_table(0)
        # Row-major occupation tuples map to the encoded basis index by
        # concatenating each mode's codeword, mode 0 most significant.
        permutation = []
        for occupations in itertools.product(range(dim), repeat=modes):
            index = 0
            for n in occupations:
                index = index * dim + codewords[n]
            permutation.append(index)
        permutation = np.array(permutation)

        expected = np.zeros_like(encoded)
        expected[np.ix_(permutation, permutation)] = _bose_hubbard_matrix(case)
        np.testing.assert_allclose(encoded, expected, atol=TOL)

    @pytest.mark.parametrize(("case", "e0", "e1"), ENERGIES)
    def test_ground_and_first_excited_energies(self, case, e0, e1):
        """Encoded ground and first excited energies match the verified reference table.

        Args:
            case: The ``(modes, dim, t, u, mu)`` model parameters.
            e0: Reference ground-state energy.
            e1: Reference first excited energy.

        """
        matrix = _qubit_operator_matrix(_mapped(case))
        eigenvalues = np.linalg.eigvalsh(matrix)
        assert eigenvalues[0] == pytest.approx(e0, abs=1e-9)
        assert eigenvalues[1] == pytest.approx(e1, abs=1e-9)

    @pytest.mark.parametrize("case", PERMUTATION_CASES)
    def test_standard_binary_and_gray_are_isospectral(self, case):
        """The two encodings differ only by a basis permutation.

        Args:
            case: The ``(modes, dim, t, u, mu)`` model parameters.

        """
        binary = _qubit_operator_matrix(_mapped(case))
        gray = _qubit_operator_matrix(_mapped(case, BosonMapping.gray_code))
        np.testing.assert_allclose(np.linalg.eigvalsh(binary), np.linalg.eigvalsh(gray), atol=1e-9)

    @pytest.mark.parametrize("case", PERMUTATION_CASES)
    def test_no_leakage_for_power_of_two_cutoffs(self, case):
        """The encoded subspace is the whole Hilbert space, so no state leaks out.

        Args:
            case: The ``(modes, dim, t, u, mu)`` model parameters.

        """
        modes, dim = case[0], case[1]
        mapping = BosonMapping.standard_binary(modes, dim)
        assert mapping.num_qubits() == modes * int(np.log2(dim))
        assert 2 ** mapping.num_qubits() == dim**modes

        encoded = _qubit_operator_matrix(_mapped(case))
        reference = _bose_hubbard_matrix(case)
        # A surjective code is a similarity transform, so the whole spectrum -
        # not merely a physical block - is preserved and nothing leaks.
        np.testing.assert_allclose(
            np.linalg.eigvalsh(encoded), np.linalg.eigvalsh(reference.astype(complex)), atol=1e-9
        )


def _label(word, num_qubits: int) -> str:
    """Render a sparse Pauli word as a label, qubit 0 rightmost.

    Args:
        word: Sparse Pauli word from the C++ engine.
        num_qubits: Total qubit count of the label.

    Returns:
        str: The Pauli label.

    """
    from qdk_chemistry._core.data import sparse_pauli_word_to_label  # noqa: PLC0415

    return sparse_pauli_word_to_label(word, num_qubits)


def _all_primitives(mapping: BosonMapping) -> dict[str, dict[str, complex]]:
    """Every primitive operator a mapping can emit, collected by label.

    Args:
        mapping: The boson-to-qubit mapping to exercise.

    Returns:
        dict[str, dict[str, complex]]: Primitive tag to its Pauli-label expansion.

    """
    nq = mapping.num_qubits()

    def collect(terms) -> dict[str, complex]:
        """Collect ``(word, coefficient)`` pairs into a label dictionary.

        Args:
            terms: Sparse ``(word, coefficient)`` pairs from the mapping.

        Returns:
            dict[str, complex]: One entry per distinct Pauli label.

        """
        out: dict[str, complex] = {}
        for word, coefficient in terms:
            label = _label(word, nq)
            out[label] = out.get(label, 0.0) + complex(coefficient)
        return out

    primitives: dict[str, dict[str, complex]] = {}
    for mode in range(mapping.num_modes()):
        primitives[f"{mode}:a"] = collect(mapping.annihilation(mode))
        primitives[f"{mode}:c"] = collect(mapping.creation(mode))
        primitives[f"{mode}:n"] = collect(mapping.number(mode))
        primitives[f"{mode}:nn1"] = collect(mapping.number_times_number_minus_one(mode))
        primitives[f"{mode}:aa"] = collect(mapping.ladder_product([(mode, False), (mode, False)]))
        primitives[f"{mode}:ccaa"] = collect(
            mapping.ladder_product([(mode, True), (mode, True), (mode, False), (mode, False)])
        )
    # Cross-mode hopping additionally pins the per-mode qubit block layout.
    for i in range(mapping.num_modes() - 1):
        primitives[f"{i}->{i + 1}"] = collect(mapping.ladder_product([(i, True), (i + 1, False)]))
    return primitives


class TestCustomCodewordTable:
    """Tests for the open end of the encoding set, ``BosonMapping.from_codeword_table``."""

    def test_reproduces_the_named_encodings_exactly(self):
        """A table holding the standard-binary or Gray codewords reproduces that mapping."""
        for dim in (2, 4, 8):
            for named, table in (
                (BosonMapping.standard_binary(2, dim), [list(range(dim))] * 2),
                (BosonMapping.gray_code(2, dim), [[n ^ (n >> 1) for n in range(dim)]] * 2),
            ):
                custom = BosonMapping.from_codeword_table(table)
                assert custom.num_qubits() == named.num_qubits()
                assert [custom.codeword_table(i) for i in range(2)] == [named.codeword_table(i) for i in range(2)]
                assert _all_primitives(custom) == _all_primitives(named)

    def test_labels_the_table_without_recognizing_it(self):
        """The name is never inferred from the table; the table is the truth."""
        custom = BosonMapping.from_codeword_table([[0, 1, 2, 3]])
        assert custom.name == "custom"
        assert BosonMapping.from_codeword_table([[0, 1, 2, 3]], "my-encoding").name == "my-encoding"
        assert BosonMapping.standard_binary(1, 4).name == "standard-binary"

    def test_rejects_invalid_tables(self):
        """Every way of breaking the permutation invariant is a hard, actionable error."""
        with pytest.raises(ValueError, match="codeword table is empty"):
            BosonMapping.from_codeword_table([])
        with pytest.raises(ValueError, match="mode 1"):
            BosonMapping.from_codeword_table([[0, 1], [0]])
        with pytest.raises(ValueError, match="power of two"):
            BosonMapping.from_codeword_table([[0, 1, 2]])
        with pytest.raises(ValueError, match="does not fit"):
            BosonMapping.from_codeword_table([[0, 4, 2, 3]])
        with pytest.raises(ValueError, match="injective"):
            BosonMapping.from_codeword_table([[0, 1, 1, 3]])

    def test_json_and_pickle_round_trip_the_table(self):
        """A custom table survives serialization; the table is what is on the wire."""
        import pickle  # noqa: PLC0415

        original = BosonMapping.from_codeword_table([[2, 0, 3, 1], [0, 1, 3, 2]], "mixed")
        document = json.loads(original.to_json())
        assert document["codewords"] == [[2, 0, 3, 1], [0, 1, 3, 2]]
        assert document["name"] == "mixed"

        for restored in (BosonMapping.from_json(original.to_json()), pickle.loads(pickle.dumps(original))):
            assert restored.name == "mixed"
            assert [restored.codeword_table(i) for i in range(2)] == [[2, 0, 3, 1], [0, 1, 3, 2]]
            assert restored.content_hash() == original.content_hash()
            assert _all_primitives(restored) == _all_primitives(original)

    def test_named_encodings_are_written_as_their_table(self):
        """A named encoding is serialized exactly like any other mapping."""
        document = json.loads(BosonMapping.standard_binary(2, 4).to_json())
        assert document["codewords"] == [[0, 1, 2, 3]] * 2
        assert document["name"] == "standard-binary"

    def test_heterogeneous_tables_are_supported(self):
        """Modes may carry different dimensions, exactly as a bosonic basis allows."""
        mapping = BosonMapping.from_codeword_table([[1, 0], [3, 2, 0, 1], [0, 1]])
        assert mapping.mode_dimensions() == [2, 4, 2]
        assert [mapping.qubits_per_mode(i) for i in range(3)] == [1, 2, 1]
        assert mapping.num_qubits() == 4
        # Mode 0 owns the most significant block, as for every other factory.
        assert mapping.mode_qubits(0) == [3]
        assert mapping.mode_qubits(1) == [1, 2]
        assert mapping.mode_qubits(2) == [0]
        assert mapping.uniform_dimension() is None

    def test_maps_a_hamiltonian_identically_to_the_named_encoding(self):
        """End to end: a Gray table produces the Gray operator, and records its own name."""
        hamiltonian = create_bose_hubbard_hamiltonian(LatticeGraph.chain(2), t=1.0, U=4.0, mu=0.0, mode_dimension=4)
        gray_table = [[n ^ (n >> 1) for n in range(4)]] * 2
        custom = create("boson_qubit_mapper").run(hamiltonian, BosonMapping.from_codeword_table(gray_table, "gray-ish"))
        named = create("boson_qubit_mapper").run(hamiltonian, BosonMapping.gray_code(2, 4))
        assert _terms(custom) == pytest.approx(_terms(named), abs=TOL)
        assert custom.encoding == "gray-ish"

    @pytest.mark.parametrize(
        ("case", "codewords"),
        [
            ((2, 4, 0.7, 3.3, 0.9), [2, 0, 3, 1]),
            ((2, 4, 1.0, 4.0, 0.0), [3, 2, 1, 0]),
            ((3, 2, 0.7, 3.3, 0.9), [1, 0]),
            ((2, 8, 0.5, 2.0, -1.0), [5, 2, 7, 0, 3, 6, 1, 4]),
        ],
    )
    def test_custom_permutation_matches_the_kronecker_construction(self, case, codewords):
        """An arbitrary permutation still permutes the occupation-basis Hamiltonian exactly.

        This is the independent check: the reference matrix is built by NumPy
        from the Fock-space definition and shares no code with the mapping.

        Args:
            case: The ``(modes, dim, t, u, mu)`` model parameters.
            codewords: The custom codeword table shared by every mode.

        """
        modes, dim, t, u, mu = case
        hamiltonian = create_bose_hubbard_hamiltonian(LatticeGraph.chain(modes), t=t, U=u, mu=mu, mode_dimension=dim)
        mapping = BosonMapping.from_codeword_table([list(codewords)] * modes)
        encoded = _qubit_operator_matrix(create("boson_qubit_mapper").run(hamiltonian, mapping))
        np.testing.assert_allclose(encoded, encoded.conj().T, atol=TOL)

        permutation = []
        for occupations in itertools.product(range(dim), repeat=modes):
            index = 0
            for n in occupations:
                index = index * dim + codewords[n]
            permutation.append(index)
        expected = np.zeros_like(encoded)
        expected[np.ix_(np.array(permutation), np.array(permutation))] = _bose_hubbard_matrix(case)
        np.testing.assert_allclose(encoded, expected, atol=TOL)


class TestHardCoreBosons:
    """Tests for the ``d = 2`` hard-core limit and its inert on-site interaction."""

    def test_hard_core_basis_is_two_level_and_needs_no_padding(self):
        """``hard_core(L)`` gives every mode ``d = 2``, which is already a power of two."""
        modes = BosonicModes.hard_core(3)
        assert modes.num_modes() == 3
        assert modes.mode_dimensions() == [2, 2, 2]
        assert all(modes.mode_dimension(i) == 2 for i in range(3))
        assert modes.with_padded_dimensions().mode_dimensions() == [2, 2, 2]
        assert BosonMapping.standard_binary(modes).num_qubits() == 3

    def test_annihilation_is_exactly_sigma_minus(self):
        """On two levels ``b`` is the spin lowering operator, with no truncation error."""
        mapping = BosonMapping.standard_binary(BosonicModes.hard_core(1))
        terms = {_label(word, 1): complex(coefficient) for word, coefficient in mapping.annihilation(0)}
        assert terms == pytest.approx({"X": 0.5, "Y": 0.5j}, abs=TOL)
        assert mapping.number_times_number_minus_one(0) == []

    def test_u_is_inert_in_the_hard_core_limit(self):
        """``n (n - 1)`` vanishes identically at ``d = 2``, so any ``U`` gives the same operator."""
        baseline = _terms(_mapped((2, 2, 1.0, 0.0, 0.0)))
        for u in (4.0, 400.0, -17.5):
            assert _terms(_mapped((2, 2, 1.0, u, 0.0))) == baseline
        # Fixture 1 of the research report.
        assert baseline == pytest.approx({"XX": -0.5, "YY": -0.5}, abs=TOL)

    def test_inert_u_is_warned_about(self, capfd):
        """The physics is left alone, but the user is told that ``U`` cannot be felt."""
        from qdk_chemistry.utils import Logger  # noqa: PLC0415

        previous = Logger.get_global_level()
        Logger.set_global_level("warn")
        try:
            capfd.readouterr()
            create_bose_hubbard_hamiltonian(LatticeGraph.chain(2), t=1.0, U=4.0, mu=0.0, mode_dimension=2)
            assert "hard-core limit" in capfd.readouterr().out

            create_bose_hubbard_hamiltonian(LatticeGraph.chain(2), t=1.0, U=0.0, mu=0.0, mode_dimension=2)
            assert "hard-core limit" not in capfd.readouterr().out

            create_bose_hubbard_hamiltonian(LatticeGraph.chain(2), t=1.0, U=4.0, mu=0.0, mode_dimension=4)
            assert "hard-core limit" not in capfd.readouterr().out
        finally:
            Logger.set_global_level(previous)

    def test_the_warning_does_not_alter_the_integrals(self):
        """``U`` is still stored verbatim as ``(ii|ii)``; only the Pauli operator is blind to it."""
        hamiltonian = create_bose_hubbard_hamiltonian(LatticeGraph.chain(2), t=1.0, U=4.0, mu=0.0, mode_dimension=2)
        assert hamiltonian.get_two_body_element(0, 0, 0, 0) == pytest.approx(4.0, abs=TOL)
        assert hamiltonian.get_one_body_element(0, 1) == pytest.approx(-1.0, abs=TOL)
