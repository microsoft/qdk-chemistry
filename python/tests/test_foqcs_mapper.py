"""Tests for the FOQCS-LCU block encoding builder and controlled circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
from qdk import TargetProfile

from qdk_chemistry.algorithms.controlled_circuit_mapper import FoqcsMapper
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.lcu_foqcs import LCUFoqcsBuilder
from qdk_chemistry.data import Circuit, QubitOperator
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.foqcs import FoqcsContainer
from qdk_chemistry.data.unitary_representation.containers.quantum_walk import LCUWalkContainer
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT
from qdk_chemistry.utils.qsharp import create_qsharp_context, use_qsharp_context

if QDK_CHEMISTRY_HAS_QISKIT:
    from qiskit.quantum_info import Operator

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


def _build_unitary_rep(pauli_strings, coefficients, *, quantum_walk=False):
    """Helper: build a FOQCS UnitaryRepresentation from Pauli strings and coefficients."""
    hamiltonian = QubitOperator(pauli_strings=pauli_strings, coefficients=np.array(coefficients, dtype=float))
    return LCUFoqcsBuilder(quantum_walk=quantum_walk).run(hamiltonian)


def _extract_block_encoding_submatrix(full_unitary, num_target, num_ancilla):
    r"""Extract the :math:`H/\lambda` submatrix from a controlled block-encoding unitary.

    Projects onto control=1 and ancilla=``|0>`` to verify the block encoding identity:

    .. math::
        (\langle 0|_\mathrm{anc} \otimes I_\mathrm{sys}) B[H] (|0\rangle_\mathrm{anc} \otimes I_\mathrm{sys})
        = H / \lambda

    """
    n_total = num_target + num_ancilla + 1  # +1 for control
    dim = 2**n_total
    assert full_unitary.shape == (dim, dim)

    indices = []
    for i in range(dim):
        ctrl_bit = (i >> 0) & 1
        anc_bits = i >> (1 + num_target)  # ancilla occupies the highest bits
        if ctrl_bit == 1 and anc_bits == 0:
            indices.append(i)
    return full_unitary[np.ix_(indices, indices)]


def _extract_uncontrolled_submatrix(full_unitary, num_target, num_ancilla):
    """Extract the control=0, ancilla-``|0>`` submatrix, which must be proportional to the identity."""
    dim = 2 ** (num_target + num_ancilla + 1)
    indices = [i for i in range(dim) if (i & 1) == 0 and (i >> (1 + num_target)) == 0]
    return full_unitary[np.ix_(indices, indices)]


def _global_phase_of(block, expected):
    r"""Return the single scalar :math:`e^{i\theta}` relating ``block`` to ``expected``.

    Q# ``R1`` is exported to Qiskit as ``rz``, which drops R1's accompanying
    ``R(PauliI, -theta)`` global-phase term.  Inside the FOQCS controlled
    specialization those rotations are applied unconditionally, so the dropped
    phase multiplies *both* control branches equally and is therefore a genuine
    global phase: unobservable, and harmless for phase estimation.  Tests compare
    up to this factor but still require it to be a single uniform scalar, which is
    what distinguishes it from a per-family (relative) phase error.
    """
    support = np.abs(expected) > 1e-9
    assert support.any(), "Expected matrix is entirely zero."
    ratios = block[support] / expected[support]
    return ratios[np.argmax(np.abs(expected[support]))]


def _assert_matches_up_to_global_phase(block, expected, description):
    """Assert ``block`` equals ``expected`` up to one uniform global phase."""
    phase = _global_phase_of(block, expected)
    assert abs(abs(phase) - 1.0) < float_comparison_absolute_tolerance, (
        f"Block encoding is not unitary-scaled for: {description} (|phase|={abs(phase):.6f})"
    )
    assert np.allclose(
        block,
        phase * expected,
        atol=float_comparison_absolute_tolerance,
        rtol=float_comparison_relative_tolerance,
    ), f"Block encoding identity failed for: {description}"
    return phase


# Hamiltonians inside the FOQCS scope: homogeneous, translationally-invariant
# 1-body (full-chain) and 2-body (fixed-offset nearest-neighbour) families.
#
# Circuit width is 1 + L + (num_families + 2L), and these tests build a dense
# 2**width unitary, so cases are kept small: L=2 unless the case specifically
# needs a longer chain (offset-2 coupling, multi-bond geometry).
_IN_SCOPE_HAMILTONIANS = [
    (["ZZ", "XI", "IX"], [0.5, 0.3, 0.3], "Ising L=2, all positive"),
    (["ZZ", "XI", "IX"], [0.5, -0.3, -0.3], "Ising L=2, negative field"),
    (["XX", "YY", "ZZ"], [0.5, 0.5, 0.5], "Heisenberg L=2, isotropic"),
    (["XX", "YY", "ZZ", "ZI", "IZ"], [0.3, -0.2, 0.5, -0.1, -0.1], "Heisenberg L=2, anisotropic with field"),
    (["YY"], [0.4], "single YY bond"),
    (["YI", "IY"], [0.35, 0.35], "Y field only"),
    (["II", "ZZ", "XI", "IX"], [0.25, 0.5, 0.3, 0.3], "Ising L=2, positive shift"),
    (["II", "ZZ", "XI", "IX"], [-0.6, 0.5, 0.3, 0.3], "Ising L=2, negative shift"),
    (["ZZI", "IZZ", "XII", "IXI", "IIX"], [0.5, 0.5, -0.3, -0.3, -0.3], "Ising L=3, negative field"),
    (["ZIZ", "XII", "IXI", "IIX"], [0.45, 0.2, 0.2, 0.2], "Ising L=3, offset-2 coupling"),
]

_IN_SCOPE_IDS = [
    "ising_l2_positive",
    "ising_l2_neg_field",
    "heisenberg_l2_isotropic",
    "heisenberg_l2_aniso_field",
    "single_yy_bond",
    "y_field_only",
    "ising_l2_positive_shift",
    "ising_l2_negative_shift",
    "ising_l3_neg_field",
    "ising_l3_offset2",
]


class TestLCUFoqcsBuilder:
    """Tests for the FOQCS-LCU block encoding builder."""

    def test_name_and_type(self):
        """Test that name and type_name return the registered values."""
        builder = LCUFoqcsBuilder()
        assert builder.name() == "lcu_foqcs"
        assert builder.type_name() == "hamiltonian_unitary_builder"

    def test_builds_foqcs_container(self):
        """Test that the builder produces a FoqcsContainer with the expected layout."""
        unitary_rep = _build_unitary_rep(["ZZI", "IZZ", "XII", "IXI", "IIX"], [0.5, 0.5, 0.3, 0.3, 0.3])
        container = unitary_rep.get_container()

        assert isinstance(container, FoqcsContainer)
        assert container.num_sites == 3
        assert container.num_target_qubits == 3
        # One X field family and one offset-1 ZZ coupling family.
        assert container.num_families == 2
        assert container.num_prepare_ancillas == container.num_families + 2 * container.num_sites

    def test_quantum_walk_wraps_container(self):
        """Test that quantum_walk=True wraps the block encoding in a walk container."""
        unitary_rep = _build_unitary_rep(["ZZ", "XI", "IX"], [0.5, 0.3, 0.3], quantum_walk=True)
        container = unitary_rep.get_container()

        assert isinstance(container, LCUWalkContainer)
        assert isinstance(container.block_encoding, FoqcsContainer)

    def test_lambda_matches_one_norm(self):
        """Test that lambda equals the 1-norm of the Hamiltonian coefficients."""
        coefficients = [0.5, 0.5, -0.3, -0.3, -0.3]
        unitary_rep = _build_unitary_rep(["ZZI", "IZZ", "XII", "IXI", "IIX"], coefficients)

        assert unitary_rep.get_container().lambda_ == pytest.approx(np.sum(np.abs(coefficients)))

    def test_identity_term_adds_a_family(self):
        """Test that a constant shift is carried as a degenerate identity family."""
        without_shift = _build_unitary_rep(["ZZ", "XI", "IX"], [0.5, 0.3, 0.3]).get_container()
        with_shift = _build_unitary_rep(["II", "ZZ", "XI", "IX"], [0.25, 0.5, 0.3, 0.3]).get_container()

        assert with_shift.num_families == without_shift.num_families + 1
        identity_family = next(f for f in with_shift.families if f.paulis == ())
        assert identity_family.offset == 0
        assert with_shift.lambda_ == pytest.approx(without_shift.lambda_ + 0.25)

    def test_negligible_identity_term_is_dropped(self):
        """Test that an identity term below tolerance does not create a family."""
        container = _build_unitary_rep(["II", "ZZ", "XI", "IX"], [1e-15, 0.5, 0.3, 0.3]).get_container()

        assert all(f.paulis != () for f in container.families)

    @pytest.mark.parametrize(
        ("pauli_strings", "coefficients", "match"),
        [
            (["XYI", "IXY"], [0.5, 0.5], "mixes Pauli letters"),
            (["XXX", "XXX"], [0.5, 0.5], "weight-3"),
            (["XII", "IXI"], [0.5, 0.5], "every site"),
            (["ZZI"], [0.5], "every"),
            (["ZZI", "IZZ"], [0.5, 0.7], "non-uniform coefficients"),
        ],
        ids=["mixed_letters", "weight_three", "partial_field", "partial_coupling", "inhomogeneous"],
    )
    def test_rejects_out_of_scope_hamiltonians(self, pauli_strings, coefficients, match):
        """Test that Hamiltonians outside the FOQCS scope raise a descriptive ValueError."""
        with pytest.raises(ValueError, match=match):
            _build_unitary_rep(pauli_strings, coefficients)

    def test_rejects_zero_hamiltonian(self):
        """Test that a Hamiltonian with vanishing 1-norm is rejected."""
        with pytest.raises(ValueError, match="positive 1-norm"):
            _build_unitary_rep(["XI", "IX"], [0.0, 0.0])


class TestFoqcsMapper:
    """Tests for the FOQCS controlled circuit mapper."""

    def test_name_and_type(self):
        """Test that name and type_name return the registered values."""
        mapper = FoqcsMapper()
        assert mapper.name() == "foqcs"
        assert mapper.type_name() == "controlled_circuit_mapper"

    def test_basic_mapping_produces_circuit_with_factory(self):
        """Test that mapping produces a Circuit with both qsharp_op and qsharp_factory."""
        unitary_rep = _build_unitary_rep(["ZZ", "XI", "IX"], [0.5, 0.3, 0.3])
        circuit = FoqcsMapper().run(unitary_rep)

        assert isinstance(circuit, Circuit)
        assert circuit._qsharp_op is not None
        assert circuit._qsharp_factory is not None

    def test_rejects_non_foqcs_container(self):
        """Test that a non-FOQCS container raises ValueError."""

        class MockContainer:
            """Mock container that is not a FoqcsContainer."""

            @property
            def type(self):
                return "mock"

        unitary_rep = UnitaryRepresentation(container=MockContainer())

        with pytest.raises(ValueError, match="not supported"):
            FoqcsMapper().run(unitary_rep)

    def test_rejects_multiple_control_qubits(self):
        """Test that more than one control qubit raises ValueError."""
        unitary_rep = _build_unitary_rep(["ZZ", "XI", "IX"], [0.5, 0.3, 0.3])

        mapper = FoqcsMapper()
        mapper.settings().set("control_indices", [0, 1])
        with pytest.raises(ValueError, match="single control qubit"):
            mapper.run(unitary_rep)

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
    @pytest.mark.parametrize(
        ("pauli_strings", "coefficients", "description"),
        _IN_SCOPE_HAMILTONIANS,
        ids=_IN_SCOPE_IDS,
    )
    def test_block_encoding_identity(self, pauli_strings, coefficients, description):
        r"""Verify the block encoding identity :math:`\langle 0|_\mathrm{anc} B[H] |0\rangle_\mathrm{anc} = H/\lambda`.

        Covers signed coefficients, ``Y`` families (which need the FOQCS phase
        correction), constant shifts, longer-range couplings, and chain lengths
        L = 2, 3, 4.
        """
        coefficients = np.array(coefficients, dtype=float)
        hamiltonian = QubitOperator(pauli_strings=pauli_strings, coefficients=coefficients)
        num_target = hamiltonian.num_qubits

        unitary_rep = _build_unitary_rep(pauli_strings, coefficients)
        circuit = FoqcsMapper().run(unitary_rep)
        qc = circuit.get_qiskit_circuit()
        full_u = Operator(qc).data

        num_ancilla = qc.num_qubits - 1 - num_target
        block = _extract_block_encoding_submatrix(full_u, num_target=num_target, num_ancilla=num_ancilla)

        lam = unitary_rep.get_container().lambda_
        expected = hamiltonian.to_matrix() / lam

        block_phase = _assert_matches_up_to_global_phase(block, expected, description)

        # The phase must be shared with the control=0 branch, otherwise it is a
        # relative phase and would bias phase estimation.
        identity_block = _extract_uncontrolled_submatrix(full_u, num_target=num_target, num_ancilla=num_ancilla)
        assert np.allclose(
            identity_block,
            block_phase * np.eye(2**num_target),
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        ), f"Control=0 branch is not the identity times the same global phase for: {description}"

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
    def test_ancilla_count_matches_container(self):
        """Test that the generated circuit width matches the container's declared layout."""
        unitary_rep = _build_unitary_rep(["ZZI", "IZZ", "XII", "IXI", "IIX"], [0.5, 0.5, -0.3, -0.3, -0.3])
        container = unitary_rep.get_container()
        qc = FoqcsMapper().run(unitary_rep).get_qiskit_circuit()

        assert qc.num_qubits == 1 + container.num_target_qubits + container.num_prepare_ancillas

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
    def test_block_encoding_is_unaffected_by_a_constant_shift(self):
        r"""Verify a constant shift moves the encoded block by exactly :math:`c I / \lambda`.

        The identity family contributes no Pauli, so SELECT acts as the identity
        on its branch and the shift appears only on the diagonal.
        """
        base_strings = ["ZZ", "XI", "IX"]
        base_coeffs = [0.5, 0.3, 0.3]
        shift = 0.25

        shifted_rep = _build_unitary_rep(["II", *base_strings], [shift, *base_coeffs])
        qc = FoqcsMapper().run(shifted_rep).get_qiskit_circuit()
        full_u = Operator(qc).data

        num_target = 2
        num_ancilla = qc.num_qubits - 1 - num_target
        block = _extract_block_encoding_submatrix(full_u, num_target=num_target, num_ancilla=num_ancilla)

        lam = shifted_rep.get_container().lambda_
        base_matrix = QubitOperator(
            pauli_strings=base_strings, coefficients=np.array(base_coeffs, dtype=float)
        ).to_matrix()
        expected = (base_matrix + shift * np.eye(2**num_target)) / lam

        _assert_matches_up_to_global_phase(block, expected, "constant shift")

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
    def test_quantum_walk_eigenvalues(self):
        r"""Verify the FOQCS walk operator eigenvalues satisfy the arccos relation.

        For a qubitized walk :math:`W = R \cdot B[H]`, each eigenvalue :math:`E_k`
        of :math:`H` maps to a pair of walk eigenvalues
        :math:`e^{\pm i \arccos(E_k / \lambda)}`.

        The walk's reflection lowers to measurement-based AND uncompute under the
        default Adaptive_RIF profile, producing a circuit with classical bits that
        Qiskit cannot turn into an ``Operator``.  The Base profile emits the
        ancilla-free form, so pin this test to it.

        ``H = aX + bZ`` on a single site is used because its normalized spectrum
        :math:`\pm\sqrt{a^2+b^2}/(|a|+|b|)` lies strictly inside :math:`(-1, 1)`,
        so ``arccos`` is not evaluated at a degenerate endpoint.
        """
        pauli_strings = ["X", "Z"]
        coefficients = np.array([0.3, 0.4])

        with use_qsharp_context(create_qsharp_context(TargetProfile.Base)):
            unitary_rep = _build_unitary_rep(pauli_strings, coefficients, quantum_walk=True)
            block = unitary_rep.get_container().block_encoding
            circuit = FoqcsMapper().run(unitary_rep)
            qc = circuit.get_qiskit_circuit()
            full_u = Operator(qc).data

        # The reflection allocates scratch qubits beyond the container's declared
        # layout; they are uncomputed, so restrict to control=1 with scratch |0>.
        num_walk = block.num_target_qubits + block.num_prepare_ancillas
        dim = full_u.shape[0]
        indices = [i for i in range(dim) if (i & 1) == 1 and (i >> (1 + num_walk)) == 0]
        walk_u = full_u[np.ix_(indices, indices)]

        # If the scratch qubits were not correctly uncomputed, this block would not be unitary.
        assert np.allclose(
            walk_u @ walk_u.conj().T,
            np.eye(2**num_walk),
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        ), "Walk operator is not unitary on the control=1, scratch-|0> subspace."

        walk_phases = np.angle(np.linalg.eigvals(walk_u))

        lam = block.lambda_
        hamiltonian = QubitOperator(pauli_strings=pauli_strings, coefficients=coefficients)
        energies = np.linalg.eigvalsh(hamiltonian.to_matrix())

        # Every Hamiltonian eigenvalue must appear as arccos(E/lambda) among the walk phases.
        for energy in energies:
            expected_phase = np.arccos(np.clip(energy / lam, -1.0, 1.0))
            assert np.min(np.abs(np.abs(walk_phases) - expected_phase)) < float_comparison_absolute_tolerance, (
                f"No walk eigenvalue matches arccos({energy:.4f}/{lam:.4f})"
            )
