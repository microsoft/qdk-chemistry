"""Tests for the SOSSA circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import numpy as np
import pytest
from qdk.test_utils import dump_operation_on_state

from qdk_chemistry.algorithms.circuit_mapper import SOSSAMapper
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.sossa import SOSSABuilder
from qdk_chemistry.data import AlgorithmRef, Circuit, FactorizedHamiltonianContainer
from qdk_chemistry.data.circuit import CircuitMetadata
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.sossa import SOSSABlockEncodingContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, get_qsharp_context, use_qsharp_context

from .test_helpers import (
    create_random_factorized_hamiltonian,
    create_test_orbitals,
    factorized_hamiltonian_to_sossa_operator,
)
from .test_phase_estimation_sossa import (
    _build_dfthc_hamiltonian_matrix,
    _python_to_qsharp_permutation,
    _python_to_qsharp_sign,
)


def _build_sossa_unitary(
    num_orbitals: int = 2,
    num_ranks: int = 2,
    num_bases: int = 1,
    num_copies: int = 1,
    *,
    seed: int = 42,
) -> UnitaryRepresentation:
    """Helper: build UnitaryRepresentation with SOSSABlockEncodingContainer from random factorized data."""
    fh = create_random_factorized_hamiltonian(
        num_orbitals=num_orbitals,
        num_ranks=num_ranks,
        num_bases=num_bases,
        num_copies=num_copies,
        seed=seed,
    )
    builder = SOSSABuilder()
    return builder.run(factorized_hamiltonian_to_sossa_operator(fh))


def _make_sossa_mapper(
    outer_algorithm: str = "alias_sampling",
    inner_algorithm: str = "controlled_alias_sampling",
    select_algorithm: str = "qrom_phase_gradient",
    coefficient_bit_precision: int = 10,
    rotation_bit_precision: int = 10,
) -> SOSSAMapper:
    """Create a SOSSAMapper with the given algorithm settings."""
    mapper = SOSSAMapper()
    mapper.settings().set("outer_prepare_algorithm", AlgorithmRef("state_prep", outer_algorithm))
    mapper.settings().set("inner_prepare_algorithm", inner_algorithm)
    mapper.settings().set("select_algorithm", select_algorithm)
    mapper.settings().set("coefficient_bit_precision", coefficient_bit_precision)
    mapper.settings().set("rotation_bit_precision", rotation_bit_precision)
    return mapper


def _reverse_bits(x: int, n: int) -> int:
    """Reverse the bit order of *x* within an *n*-bit field."""
    return int(format(x, f"0{n}b")[::-1], 2)


def _block_encoding_action(circuit, num_system_qubits: int, system_state: np.ndarray) -> np.ndarray:
    r"""Apply ``circuit`` to :math:`|\psi\rangle|0\rangle_\mathrm{anc}` and project on ancilla zero.

    Returns :math:`(\langle 0|_\mathrm{anc} \otimes I) B (|0\rangle_\mathrm{anc} \otimes I) |\psi\rangle`,
    i.e. the top-left block of the encoding applied to ``system_state``. The mapper lays the
    register out as ``[system | ancilla]`` while :func:`dump_operation_on_state` numbers basis
    states big-endian.
    """
    stride = 2 ** (circuit.num_qubits - num_system_qubits)
    dimension = 2**num_system_qubits

    initial_state = [0.0] * ((dimension - 1) * stride + 1)
    for index, amplitude in enumerate(system_state):
        initial_state[_reverse_bits(index, num_system_qubits) * stride] = float(amplitude)

    statevector = dump_operation_on_state(
        circuit._qsharp_op, circuit.num_qubits, initial_state, context=get_qsharp_context()
    )
    return np.array([statevector[_reverse_bits(index, num_system_qubits) * stride] for index in range(dimension)])


def _assert_full_block_matches_hgap(h1, u_matrices, w_matrices, wb_matrix, *, seed=0, atol=1e-9, sweep_basis=False):
    r"""Assert the simulated SOSSA block equals ``H_gap/\Lambda - I`` on a full fixture.

    No global phase is fitted, so a block that is right up to a sign still fails. With
    ``sweep_basis`` the whole block is reconstructed column by column instead of being probed
    along a single random vector.
    """
    num_orbitals = h1.shape[0]
    num_system_qubits = 2 * num_orbitals
    dimension = 2**num_system_qubits

    container = FactorizedHamiltonianContainer(
        one_body_integrals=h1,
        u_matrices=u_matrices.reshape(-1),
        w_matrices=w_matrices.reshape(-1),
        wb_matrix=wb_matrix,
        orbitals=create_test_orbitals(num_orbitals),
        core_energy=0.0,
        inactive_fock_matrix=np.zeros((num_orbitals, num_orbitals)),
    )
    unitary = SOSSABuilder().run(factorized_hamiltonian_to_sossa_operator(container))
    normalization = unitary.get_container().normalization
    circuit = _make_sossa_mapper(
        outer_algorithm="dense_pure_state",
        inner_algorithm="direct",
        select_algorithm="direct",
        coefficient_bit_precision=16,
        rotation_bit_precision=16,
    ).run(unitary)

    permutation = _python_to_qsharp_permutation(num_orbitals)
    signs = _python_to_qsharp_sign(num_orbitals)
    reorder = np.zeros((dimension, dimension))
    for index in range(dimension):
        reorder[permutation[index], index] = signs[index]
    h_gap = _build_dfthc_hamiltonian_matrix(h1, u_matrices, w_matrices, wb_matrix)
    expected_block = reorder @ h_gap @ reorder.T / normalization - np.eye(dimension)

    if sweep_basis:
        actual_block = np.column_stack(
            [
                _block_encoding_action(
                    circuit, num_system_qubits=num_system_qubits, system_state=np.eye(dimension)[column]
                )
                for column in range(dimension)
            ]
        )
        np.testing.assert_allclose(actual_block, expected_block, atol=atol)
        return

    rng = np.random.default_rng(seed)
    system_state = rng.standard_normal(dimension)
    system_state /= np.linalg.norm(system_state)
    np.testing.assert_allclose(
        _block_encoding_action(circuit, num_system_qubits=num_system_qubits, system_state=system_state),
        expected_block @ system_state,
        atol=atol,
    )


def _alias_atol(num_coefficients: int, bits_precision: int) -> float:
    return 1.0 / (num_coefficients * 2**bits_precision)


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-operation builder tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOuterPrep:
    """Tests for SOSSAMapper._build_outer_prepare_circuit."""

    @pytest.mark.parametrize("algorithm", ["dense_pure_state", "qrom"])
    def test_build_outer_prep_fidelity(self, algorithm):
        sossa_unitary = _build_sossa_unitary()
        container = sossa_unitary.get_container()
        mapper = _make_sossa_mapper(outer_algorithm=algorithm)
        prepare_circuit = mapper._build_outer_prepare_circuit(container)
        op = prepare_circuit._qsharp_op
        num_gradient = prepare_circuit.metadata.num_phase_gradient_ancillas

        coefficients = np.asarray(container.outer_prepare.get_coefficients())
        num_qubits = math.ceil(math.log2(len(coefficients))) if len(coefficients) > 1 else 1

        if num_gradient:
            op = QSHARP_UTILS.CircuitComposition.MakeSharedAncillaOp(
                op, QSHARP_UTILS.PhaseGradient.PreparePhaseGradientState, num_gradient
            )
        full_sv = np.array(
            dump_operation_on_state(
                op,
                num_qubits + num_gradient,
                context=get_qsharp_context(),
            )
        )
        actual_sv = full_sv.reshape(2**num_qubits, 2**num_gradient)[:, 0]

        n_states = 2**num_qubits
        expected = np.zeros(n_states)
        for j, amp in enumerate(coefficients):
            if j < n_states:
                expected[_reverse_bits(j, num_qubits)] = amp
        expected /= np.linalg.norm(expected)

        fidelity = abs(np.dot(np.conj(actual_sv), expected))
        assert np.isclose(fidelity, 1.0, atol=1e-3)

    @pytest.mark.slow
    def test_build_outer_prep_alias_sampling_marginal_probs(self):
        sossa_unitary = _build_sossa_unitary()
        container = sossa_unitary.get_container()
        bit_precision = 10
        mapper = _make_sossa_mapper(outer_algorithm="alias_sampling", coefficient_bit_precision=bit_precision)
        prepare_circuit = mapper._build_outer_prepare_circuit(container)
        op = prepare_circuit._qsharp_op
        assert prepare_circuit.metadata.num_phase_gradient_ancillas == 0

        coefficients = np.asarray(container.outer_prepare.get_coefficients())
        num_index_qubits = math.ceil(math.log2(len(coefficients))) if len(coefficients) > 1 else 1
        total_qubits = 2 * num_index_qubits + 2 * bit_precision + 1

        full_sv = np.array(dump_operation_on_state(op, total_qubits, context=get_qsharp_context()))

        n_index = 2**num_index_qubits
        shift = total_qubits - num_index_qubits
        probs = np.zeros(n_index)
        for i in range(len(full_sv)):
            probs[(i >> shift) & (n_index - 1)] += abs(full_sv[i]) ** 2

        squared_coeffs = np.abs(coefficients) ** 2
        expected_probs = np.zeros(n_index)
        for j, p in enumerate(squared_coeffs / np.sum(squared_coeffs)):
            expected_probs[_reverse_bits(j, num_index_qubits)] = p

        atol = _alias_atol(len(coefficients), bit_precision)
        np.testing.assert_allclose(probs, expected_probs, atol=atol)


class TestInnerPrep:
    """Tests for SOSSAMapper._build_inner_prep."""

    @pytest.mark.slow
    @pytest.mark.parametrize("algorithm", ["controlled_alias_sampling", "direct"])
    def test_build_inner_prep_fidelity(self, qsharp_test_context, qsharp_test_utils, algorithm):
        # Use num_bases=2 for a non-trivial inner dimension (B+1=3)
        sossa_unitary = _build_sossa_unitary(num_orbitals=2, num_ranks=2, num_bases=2, num_copies=1)
        container = sossa_unitary.get_container()

        # Build outer prep (exact, dense_pure)
        outer_mapper = _make_sossa_mapper(outer_algorithm="dense_pure_state")
        with use_qsharp_context(qsharp_test_context):
            outer_op = outer_mapper._build_outer_prepare_circuit(container)._qsharp_op

        # Build inner prep
        bit_precision = 6
        inner_mapper = _make_sossa_mapper(inner_algorithm=algorithm, coefficient_bit_precision=bit_precision)
        with use_qsharp_context(qsharp_test_context):
            inner_op, _ = inner_mapper._build_inner_oracles(container)

        # Compute register sizes
        outer_coeffs = np.asarray(container.outer_prepare.get_coefficients())
        num_outer_qubits = math.ceil(math.log2(len(outer_coeffs))) if len(outer_coeffs) > 1 else 1

        inner_coeffs = container.inner_prepare.conditional_coefficients
        n_coeffs = inner_coeffs.shape[1]
        n_index_bits = math.ceil(math.log2(n_coeffs)) if n_coeffs > 1 else 1

        if algorithm == "controlled_alias_sampling":
            fr = container.inner_prepare.free_rider_data
            n_fr = fr.shape[1] if fr is not None and fr.size > 0 else 0
            num_inner_qubits = 2 * n_index_bits + 2 * bit_precision + 3 + n_fr
        else:  # direct
            fr = container.inner_prepare.free_rider_data
            n_fr = fr.shape[1] if fr is not None and fr.size > 0 else 0
            num_inner_qubits = n_index_bits + 1 + n_fr  # + the sign qubit SELECT phases

        # Apply outer + inner prep
        full_sv = np.array(
            dump_operation_on_state(
                qsharp_test_utils.SOSSAWalkTests.TestMakeOuterInnerPrepOp(outer_op, inner_op, num_outer_qubits),
                num_outer_qubits + num_inner_qubits,
                context=qsharp_test_context,
            )
        )

        # Check conditional marginals for each outer value l
        total_qubits = num_outer_qubits + num_inner_qubits
        n_inner_index = 2**n_index_bits

        for ell in range(len(outer_coeffs)):
            if abs(outer_coeffs[ell]) < 1e-10:
                continue

            # Compute conditional marginal probs on inner index register
            probs = np.zeros(n_inner_index)
            for i in range(len(full_sv)):
                amp = full_sv[i]
                if abs(amp) < 1e-15:
                    continue
                bits = format(i, f"0{total_qubits}b")
                outer_be = bits[:num_outer_qubits]
                outer_val = int(outer_be[::-1], 2)  # LE
                if outer_val != ell:
                    continue
                inner_be = bits[num_outer_qubits : num_outer_qubits + n_index_bits]
                inner_val = int(inner_be[::-1], 2)
                probs[inner_val] += abs(amp) ** 2

            # Normalize to conditional probability
            total_prob = np.sum(probs)
            expected_outer_prob = abs(outer_coeffs[ell]) ** 2 / np.sum(np.abs(outer_coeffs) ** 2)
            assert total_prob == pytest.approx(expected_outer_prob, abs=1e-9), (
                f"outer={ell}, algorithm={algorithm}: branch weight {total_prob} != {expected_outer_prob}"
            )
            probs /= total_prob

            # Expected: |c_{l,b}|² / Σ|c_{l,j}|²
            abs_coeffs = np.abs(inner_coeffs[ell])
            expected_probs = abs_coeffs**2 / np.sum(abs_coeffs**2)

            atol = _alias_atol(n_coeffs, bit_precision) if algorithm == "controlled_alias_sampling" else 1e-3
            np.testing.assert_allclose(
                probs[:n_coeffs], expected_probs, atol=atol, err_msg=f"outer={ell}, algorithm={algorithm}"
            )


class TestSOSSAMapper:
    """Tests for the SOSSA block-encoding circuit mapper."""

    def test_rejects_non_sossa_container(self):
        """Verify SOSSAMapper raises ValueError for non-SOSSABlockEncodingContainer containers."""

        class MockContainer:
            """Mock container that is not a SOSSABlockEncodingContainer."""

            @property
            def type(self):
                return "mock"

        unitary_rep = UnitaryRepresentation(container=MockContainer())

        mapper = SOSSAMapper()
        with pytest.raises(ValueError, match="not supported"):
            mapper.run(unitary_rep)

    def test_rejects_non_unit_power(self):
        """The mapper emits a single block encoding, so a powered container must be refused."""
        built = _build_sossa_unitary().get_container()
        powered = SOSSABlockEncodingContainer.from_json({**built.to_json(), "power": 3})

        with pytest.raises(ValueError, match="only unit power"):
            SOSSAMapper().run(UnitaryRepresentation(container=powered))

    @pytest.mark.parametrize(
        ("outer_alg", "inner_alg", "select_alg"),
        [
            ("alias_sampling", "controlled_alias_sampling", "qrom_phase_gradient"),
            ("dense_pure_state", "direct", "direct"),
            ("qrom", "controlled_alias_sampling", "direct"),
            ("alias_sampling", "direct", "qrom_phase_gradient"),
            ("dense_pure_state", "controlled_alias_sampling", "qrom_phase_gradient"),
        ],
        ids=[
            "default_all",
            "dense_direct_direct",
            "qrom_alias_direct",
            "alias_direct_phase",
            "dense_alias_phase",
        ],
    )
    def test_all_algorithm_combinations_produce_circuit(self, outer_alg, inner_alg, select_alg):
        """Test that all valid algorithm combinations produce a Circuit."""
        unitary = _build_sossa_unitary()
        mapper = _make_sossa_mapper(
            outer_algorithm=outer_alg,
            inner_algorithm=inner_alg,
            select_algorithm=select_alg,
        )
        circuit = mapper.run(unitary)

        assert isinstance(circuit, Circuit)
        assert circuit._qsharp_op is not None
        assert circuit._qsharp_factory is not None

    def test_free_rider_placement_costs_inner_prepare_pairs_consistently(self, qsharp_test_utils):
        """One block applies two inner PREPARE pairs against a single free-rider pair."""
        coefficients = [[1.0, 1.0, 1.0, 1.0] for _ in range(6)]
        free_rider_data = [[False, True] for _ in range(6)]

        # Inner pair costs are 27 inline versus 25 split; the free-rider pair costs 5. At the
        # 2:1 pair ratio splitting costs 2*25 + 5 = 55 against 2*27 = 54, so the word stays
        # inline; at 4:1 it was 4*25 + 5 = 105 against 4*27 = 108 and was split out. Both
        # clear by a margin, so the case does not rest on a tie-break.
        load_separately = qsharp_test_utils.SOSSAWalkTests.TestShouldLoadFreeRiderSeparately(
            coefficients,
            free_rider_data,
            1,
        )

        assert load_separately is False

    @pytest.mark.parametrize(
        ("sf_rows", "sf_address_qubits", "dq_rows", "dq_address_qubits"),
        [
            (4, 2, 4, 2),
            (3, 2, 3, 2),
            (5, 3, 3, 2),
            (2, 1, 7, 3),
        ],
    )
    def test_branched_angle_word_erasure_restores_the_address_register(
        self,
        qsharp_test_utils,
        sf_rows: int,
        sf_address_qubits: int,
        dq_rows: int,
        dq_address_qubits: int,
    ) -> None:
        """The measurement-based erasure must phase exactly what the forward load wrote."""
        width = 3
        sf_data = [[(i + j) % 3 == 0 for j in range(width)] for i in range(sf_rows)]
        dq_data = [[bool((i >> j) & 1) for j in range(width - 1)] for i in range(dq_rows)]

        # The erasure measures, so a mismatched row only shows up for the outcomes whose
        # parity it changes; repeat to keep the check from passing by luck.
        for _ in range(8):
            assert qsharp_test_utils.SOSSAWalkTests.TestBranchedRotationWordRoundTrip(
                sf_data,
                dq_data,
                sf_address_qubits,
                dq_address_qubits,
            )

    def test_signed_two_term_block_encoding_matches_hand_calculation(self):
        operator = factorized_hamiltonian_to_sossa_operator(create_random_factorized_hamiltonian(1, 1, 1, 1))
        sossa = operator.get_container()
        sossa.one_body.coeffs[...] = 0.0
        sossa.two_body.coeffs[...] = np.array([[1.0, -0.5]])
        sossa.two_body.angles[...] = 0.0

        unitary = SOSSABuilder().run(operator)
        container = unitary.get_container()
        circuit = _make_sossa_mapper(
            outer_algorithm="dense_pure_state",
            inner_algorithm="direct",
            select_algorithm="direct",
        ).run(unitary)

        identity = np.eye(4)
        spin_z_sum = np.kron(np.diag([1.0, -1.0]), np.eye(2)) + np.kron(np.eye(2), np.diag([1.0, -1.0]))
        generator = 0.5 * (spin_z_sum - identity)
        h_gap = 0.5 * generator @ generator
        assert container.normalization == pytest.approx(9.0 / 16.0)
        expected_block = h_gap / container.normalization - identity

        system_state = np.array([1.0, 2.0, 3.0, 4.0]) / np.sqrt(30.0)
        expected = expected_block @ system_state
        actual = _block_encoding_action(circuit, num_system_qubits=2, system_state=system_state)
        actual *= np.exp(-1j * np.angle(np.vdot(expected, actual)))
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_full_block_matches_hgap_on_two_orbital_general_angle_fixture(self):
        """The 16-dimensional block equals ``H_gap/Lambda - I`` when Givens rotations are active.

        Two spatial orbitals with an off-diagonal one-body term (a general ``D1``/``Q1`` angle)
        and a non-axis-aligned basis vector (a general ``SF`` angle) exercise the neighbour
        rotations that the one-orbital hand calculation cannot reach.
        """
        h1 = np.array([[0.5, 0.3], [0.3, -0.2]])
        u_matrices = np.array([[[0.6, 0.8]]])  # (R=1, B=1, N=2), general angle
        w_matrices = np.array([[[0.7]]])  # (R=1, B=1, C=1)
        wb_matrix = np.array([[0.3]])  # (R=1, C=1)
        _assert_full_block_matches_hgap(h1, u_matrices, w_matrices, wb_matrix, sweep_basis=True)

    def test_full_block_matches_hgap_with_multiple_ranks_copies_mixed_signs(self):
        """The block matches ``H_gap/Lambda - I`` for R>1, C>1, mixed-sign ``w_b`` and ``w_B>0``.

        This is the stress fixture: two ranks, two bases, two copies, mixed-sign two-body
        weights and strictly positive identity weights, so every branch of ``SELECT`` and the
        squared ``SF`` generators contribute.
        """
        rng = np.random.default_rng(11)
        u_matrices = np.zeros((2, 2, 2))
        for r in range(2):
            for b in range(2):
                v = rng.standard_normal(2)
                u_matrices[r, b] = v / np.linalg.norm(v)
        w_matrices = np.array([[[0.5, 0.3], [-0.4, 0.2]], [[0.6, -0.1], [0.25, 0.35]]])  # mixed sign
        wb_matrix = np.array([[0.4, 0.2], [0.3, 0.5]])  # w_B > 0
        h1 = rng.standard_normal((2, 2))
        h1 = 0.3 * (h1 + h1.T)
        _assert_full_block_matches_hgap(h1, u_matrices, w_matrices, wb_matrix)

    def test_declares_the_register_the_walk_reflects_about(self):
        unitary = _build_sossa_unitary()
        container = unitary.get_container()
        mapper = _make_sossa_mapper()
        circuit = mapper.run(unitary)

        num_system_qubits = 2 * container.metadata.num_spatial_orbitals
        num_gradient = circuit.metadata.num_phase_gradient_ancillas
        coefficient_precision = mapper.settings().get("coefficient_bit_precision")
        num_outer_qubits = 2 * container.layout.outer_prep_bits + 2 * coefficient_precision + 1
        num_reflect_inner = container.layout.inner_prep_bits + coefficient_precision + 1
        assert num_gradient == mapper.settings().get("rotation_bit_precision")
        assert circuit.num_qubits == num_system_qubits + num_outer_qubits + num_reflect_inner + 2 + num_gradient
        assert circuit.num_qubits - num_system_qubits - num_gradient > 0

    def test_outer_prepare_layout_uses_declared_circuit_width(self):
        unitary = _build_sossa_unitary()
        container = unitary.get_container()
        mapper = _make_sossa_mapper(select_algorithm="direct")
        outer_extra_bits = 2
        gradient_bits = 3
        outer_prepare_circuit = Circuit(
            qasm="OPENQASM 3.0;",
            num_qubits=container.layout.outer_prep_bits + outer_extra_bits + gradient_bits,
            metadata=CircuitMetadata(num_phase_gradient_ancillas=gradient_bits),
        )

        regs, _ = mapper._compute_register_sizes(container, outer_prepare_circuit)

        assert regs["num_outer_qubits"] == container.layout.outer_prep_bits + outer_extra_bits
        assert regs["num_outer_prepare_gradient_qubits"] == gradient_bits
        assert regs["num_phase_gradient_qubits"] == gradient_bits
        assert regs["num_ancilla_qubits"] == (regs["num_outer_qubits"] + regs["num_reflect_inner"] + 2 + gradient_bits)


def _vector_to_givens_angles(vec: np.ndarray) -> list[float]:
    """Convert a unit vector to Givens rotation angles (same as SOSSABuilder)."""
    N = len(vec)  # noqa: N806
    v = vec.copy().astype(float)
    angles = [0.0] * (N - 1)
    for j in range(N - 2, -1, -1):
        angles[j] = float(np.arctan2(v[j + 1], v[j]))
        v[j] = float(np.sqrt(v[j] ** 2 + v[j + 1] ** 2))
    return angles


class TestSelectFullFidelity:
    """Tests for the full SELECT operation fidelity with known rotation angles."""

    @staticmethod
    def _select_data(
        N: int,  # noqa: N803
        rotation_bit_precision: int,
        num_ranks: int = 1,
        num_bases: int = 1,
        num_copies: int = 1,
    ) -> dict:
        rng = np.random.default_rng(42 + N)

        def unit_angles() -> list[float]:
            v = rng.standard_normal(N)
            return _vector_to_givens_angles(v / np.linalg.norm(v))

        rank_bits = math.ceil(math.log2(num_ranks)) if num_ranks > 1 else 0
        return {
            "numOrbitals": N,
            "numRanks": num_ranks,
            "numBases": num_bases,
            "numCopies": num_copies,
            "numPositiveOneBody": N,
            "OneBodyRotationAngles": [unit_angles() for _ in range(N)],
            # Indexed b * R + r, matching both BuildSFBulkRotationData and the direct path.
            "TwoBodyRotationAngles": [unit_angles() for _ in range(num_ranks * (num_bases + 1))],
            "rotationBitPrecision": rotation_bit_precision,
            "numFreeRiderBits": 2 + rank_bits,
            "signQubitIndex": -1,
        }

    @staticmethod
    def _run_select(
        context_factory, select_data: dict, xo_value: int = 0, b_value: int = 0, use_phase_gradient: bool = False
    ) -> np.ndarray:
        ctx = context_factory()
        ctx.code.QDKChemistry.TestUtils.SOSSAWalkTests.TestSelectDQ(select_data, xo_value, b_value, use_phase_gradient)
        return np.array(ctx.dump_machine().as_dense_state())

    @pytest.mark.parametrize("num_free_rider_bits", [0, 1])
    def test_select_rejects_missing_generator_bits(self, qsharp_test_context_factory, num_free_rider_bits):
        select_data = self._select_data(2, rotation_bit_precision=10)
        select_data["numFreeRiderBits"] = num_free_rider_bits

        with pytest.raises(Exception, match="SelectImpl requires at least two free-rider bits"):
            self._run_select(qsharp_test_context_factory, select_data)

    @pytest.mark.parametrize("N", [2, 3])
    def test_select_dq_applies_the_analytic_rotated_majorana(self, qsharp_test_context_factory, N):  # noqa: N803
        rng = np.random.default_rng(2024)
        u = rng.standard_normal(N)
        u /= np.linalg.norm(u)
        other = rng.standard_normal(N)
        other /= np.linalg.norm(other)
        select_data = {
            "numOrbitals": N,
            "numRanks": 1,
            "numBases": 1,
            "numCopies": 1,
            "numPositiveOneBody": N,
            # x_o = 0 is the generator under test; the rest only need to be well formed.
            "OneBodyRotationAngles": [_vector_to_givens_angles(u)] + [_vector_to_givens_angles(other)] * (N - 1),
            "TwoBodyRotationAngles": [_vector_to_givens_angles(other)] * 2,
            "rotationBitPrecision": 14,
            "numFreeRiderBits": 2,
            "signQubitIndex": -1,
        }
        sv = self._run_select(qsharp_test_context_factory, select_data, xo_value=0, b_value=0)
        assert np.linalg.norm(sv) == pytest.approx(1.0, abs=1e-10)

        total = round(math.log2(len(sv)))
        xo_bits = math.ceil(math.log2(N + 1)) if N + 1 > 1 else 1
        system0 = xo_bits + (1 + 2) + 2  # outer + (b bits + free-rider) + spin

        def amplitude(occupied):
            index = sum(1 << (total - 1 - (system0 + q)) for q in occupied)
            return sv[index].real  # spinDQ = 0 selects the spin-down branch

        vacuum = amplitude([])
        assert abs(vacuum) > 1e-6, "SELECT produced no gamma_0 component"
        measured = [1.0] + [-amplitude([0, p]) / vacuum for p in range(1, N)]
        measured = np.array(measured) * u[0]

        assert measured == pytest.approx(u, abs=1e-3), (
            f"SELECT applied the Majorana of {measured} where the angles encode {u}; "
            f"the Givens chain is not the rotation U(u) of Eq. 93"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("dims", "xo_value", "b_value", "bit_precision"),
        [
            ((2, 1, 1, 1), 0, 0, 10),
            ((2, 2, 1, 1), 2, 0, 10),
            ((3, 2, 2, 1), 0, 0, 7),
            ((3, 2, 2, 1), 0, 1, 7),
        ],
        ids=["N2_dq", "N2_sf", "N3_dq_chain", "N3_dq_nonzero_b"],
    )
    def test_phase_gradient_backend_matches_direct(
        self,
        qsharp_test_context_factory,
        dims: tuple[int, int, int, int],
        xo_value: int,
        b_value: int,
        bit_precision: int,
    ):
        """The two production SELECT backends must implement the same Givens basis change."""
        num_orbitals, num_ranks, num_bases, num_copies = dims
        select_data = self._select_data(
            num_orbitals,
            rotation_bit_precision=bit_precision,
            num_ranks=num_ranks,
            num_bases=num_bases,
            num_copies=num_copies,
        )

        direct = self._run_select(
            qsharp_test_context_factory, select_data, xo_value=xo_value, b_value=b_value, use_phase_gradient=False
        )
        qrom = self._run_select(
            qsharp_test_context_factory, select_data, xo_value=xo_value, b_value=b_value, use_phase_gradient=True
        )[:: 1 << bit_precision]

        # The gradient is conjugated back to |0...0>, so restricting to it keeps the full norm.
        assert len(qrom) == len(direct)
        assert np.linalg.norm(qrom) == pytest.approx(1.0, abs=1e-6)

        fidelity = abs(np.vdot(direct, qrom)) / (np.linalg.norm(direct) * np.linalg.norm(qrom))
        assert fidelity == pytest.approx(1.0, abs=3e-3), (
            f"phase-gradient and direct SELECT backends disagree: fidelity={fidelity}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Walk operator logical resource count tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSOSSAWalkLogicalCounts:
    @pytest.mark.parametrize(
        ("num_orbitals", "num_ranks", "num_bases", "num_copies"),
        [
            (2, 1, 1, 1),
            (2, 2, 1, 1),
            (3, 2, 2, 1),
            (4, 2, 2, 2),
        ],
        ids=["N2R1B1C1", "N2R2B1C1", "N3R2B2C1", "N4R2B2C2"],
    )
    def test_qubit_count_matches_formula(self, num_orbitals, num_ranks, num_bases, num_copies):
        """Verify numQubits matches the paper formula bounds."""
        sossa_unitary = _build_sossa_unitary(
            num_orbitals=num_orbitals,
            num_ranks=num_ranks,
            num_bases=num_bases,
            num_copies=num_copies,
        )
        mapper = _make_sossa_mapper(
            outer_algorithm="dense_pure_state",
            inner_algorithm="direct",
            select_algorithm="direct",
            rotation_bit_precision=10,
        )
        circuit = mapper.run(sossa_unitary)

        factory = circuit._qsharp_factory
        ctx = factory.program._qdk_context
        lc = ctx.logical_counts(factory.program, *factory.parameter.values())

        actual_qubits = lc["numQubits"]

        N = num_orbitals  # noqa: N806
        R, B, C = num_ranks, num_bases, num_copies  # noqa: N806
        Xo = N + R * C  # noqa: N806
        n_xo = math.ceil(math.log2(Xo)) if Xo > 1 else 1
        n_b = math.ceil(math.log2(B + 1)) if B + 1 > 1 else 1
        min_qubits = 2 * N + n_xo + n_b + 2 + 1
        select_ancilla = 3

        assert actual_qubits >= min_qubits + select_ancilla, (
            f"N={N},R={R},B={B},C={C}: qubits={actual_qubits} < min={min_qubits}+select_anc={select_ancilla}"
        )
        max_overhead = n_xo + n_b + N + 10
        assert actual_qubits <= min_qubits + select_ancilla + max_overhead, (
            f"N={N},R={R},B={B},C={C}: qubits={actual_qubits} > max={min_qubits + select_ancilla + max_overhead}"
        )
