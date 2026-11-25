"""Tests for matrix product state preparation with dense and block-sparse unitary synthesis.

Covers settings and input validation, agreement of the dense and block-sparse Q# circuits, a
multi-site preparation with unequal bonds against an independent contraction, the block-sparse site
matrix built from symmetry blocks, fermionic order signs on open-shell states, composition with
Trotterized quantum phase estimation to recover classical eigenvalues, and resource-estimate
regressions. The site isometries are
synthesized by the ``unitary_synthesis_dense`` and ``unitary_synthesis_block_sparse`` circuit mappers, which have
their own tests. Fixture data
(``mps_h2``, ``mps_spin``) lives in ``test_data/mps_h2.mps.json``/``mps_spin.mps.json``: each holds a qubit
Hamiltonian, its exact ground-state energy, and right-canonical MPS tensors of that ground state for the identity
and reversed site orders. ``mps_h2`` is H2/STO-3G at 0.74 Angstrom (RHF orbitals, Jordan-Wigner), and ``mps_spin``
is ``Jx*XX + Jy*YY + hA*Z0 + hB*Z1`` with distinct coefficients. The tensors are stored so resource counts stay
fixed across platforms.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.phase_estimation.standard_phase_estimation import StandardPhaseEstimation
from qdk_chemistry.algorithms.state_preparation.matrix_product_state import MatrixProductStatePreparation
from qdk_chemistry.data import (
    AlgorithmRef,
    Configuration,
    MPSContainer,
    MPSSite,
    Orbitals,
    QubitOperator,
    UnitaryMatrixContainer,
    UnitaryRepresentation,
    Wavefunction,
)
from qdk_chemistry.data import symmetry as sym
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, use_qsharp_context

from .helpers import apply_operation, create_test_basis_set, create_test_orbitals, create_test_wavefunction

_METHODS = pytest.mark.parametrize("method", ["dense", "block_sparse"])
_FIXTURE_DIR = Path(__file__).parent / "test_data"


def make_site(values: np.ndarray) -> MPSSite:
    """Wrap a dense ``(left, physical, right)`` array as one trivial-symmetry MPS block."""
    left, physical, right = values.shape
    product, label = sym.SymmetryProduct([]), sym.SymmetryLabel([])
    tensor_type = sym.SymmetryBlockedTensorRank3Complex if np.iscomplexobj(values) else sym.SymmetryBlockedTensorRank3
    tensor = tensor_type(
        [product] * 3,
        [{label: left}, {label: physical}, {label: right}],
        [((label,) * 3, values.reshape(left * physical, right))],
    )
    return MPSSite(tensor, [label], [label], [label], [])


def make_mps(
    tensors, orthogonality_center: int | None = 0, site_to_orbital_order=None, orbitals: Orbitals | None = None
) -> MPSContainer:
    """Construct a native MPS container from dense test tensors (one site per orbital unless *orbitals* is given)."""
    sites = [make_site(np.asarray(tensor)) for tensor in tensors]
    return MPSContainer(
        sites,
        orbitals if orbitals is not None else create_test_orbitals(max(1, len(sites))),
        orthogonality_center=orthogonality_center,
        site_to_orbital_order=list(site_to_orbital_order or []),
    )


def synthesis_ref(method: str) -> AlgorithmRef:
    """Return the ``unitary_synthesis`` setting that selects the given site synthesis method."""
    return AlgorithmRef("circuit_mapper", f"unitary_synthesis_{method}")


def preparation_data(mps: MPSContainer, method: str = "dense") -> dict:
    """Return the Q# MPS Params fields of a container using the selected site synthesis method."""
    preparer = MatrixProductStatePreparation()
    preparer.settings().update("unitary_synthesis", synthesis_ref(method))
    preparer.settings().update("rotation_bit_precision", _AGREEMENT_PRECISION)
    return preparer._generate_preparation_data(mps)[1]


def simulate_mps_preparation(qsharp_test_utils, params: dict, method: str):
    """Run the preparation with its bond register exposed and post-select that register on ``|0>``."""
    tests = qsharp_test_utils.MPSPreparationTests
    make_op = tests.TestMakeMPSSparseOp if method == "block_sparse" else tests.TestMakeMPSSequentialOp
    num_state_qubits = params["numQubitsPerSite"] * params["numSites"]
    num_bond_qubits = params["numAncillaQubits"]
    initial = np.eye(1, 1 << (num_state_qubits + num_bond_qubits)).ravel()
    output = apply_operation(make_op(params), initial)
    state = output.reshape(1 << num_state_qubits, 1 << num_bond_qubits)[:, 0]
    probability = float(np.sum(np.abs(state) ** 2))
    return probability, state / np.sqrt(probability)


# File-backed mps_h2 / mps_spin fixtures; see the module docstring.
_FIXTURES = {name: json.loads((_FIXTURE_DIR / f"{name}.mps.json").read_text()) for name in ("mps_h2", "mps_spin")}


def _fixture_mps(name: str, order: tuple[int, ...]) -> MPSContainer:
    return make_mps(_fixture_site_tensors(name, order), site_to_orbital_order=order)


def _fixture_site_tensors(name: str, order: tuple[int, ...]):
    """Return the per-order right-canonical MPS tensors for *name*."""
    return _FIXTURES[name]["orders"][",".join(map(str, order))]["site_tensors"]


def _identity_order(name: str) -> tuple[int, ...]:
    return tuple(range(_FIXTURES[name]["num_sites"]))


# A normalized spinless site zero and a right-canonical site one that copies its physical state to its left bond.
_INITIAL_SITE = np.array([[[0.6, 0.0], [0.0, 0.8]]])
_IDENTITY_SITE = np.eye(2).reshape(2, 2, 1)


def test_default_settings():
    """The default site mapper is the dense unitary synthesis mapper, and the rotation precision is range-checked."""
    preparer = create("state_prep", "matrix_product_state")
    ref = preparer.settings().get("unitary_synthesis")
    assert (ref.algorithm_type, ref.algorithm_name) == ("circuit_mapper", "unitary_synthesis_dense")
    with pytest.raises(ValueError, match="out of allowed range"):
        preparer.settings().update("rotation_bit_precision", 0)


@pytest.mark.parametrize("active_space", [False, True], ids=["full-space", "active-space"])
def test_circuit_register_holds_two_qubits_per_site(active_space):
    """The H2 circuit acts on two qubits per site, also when the sites are the active orbitals of a larger space."""
    orbitals = None
    if active_space:
        orbitals = Orbitals(
            np.eye(4),
            np.array([-1.0, -0.5, 0.5, 1.0]),
            None,
            create_test_basis_set(4, "mps-active-space"),
            sym.spin_index_set(4, [1, 2], [1, 2]),
        )
    mps = make_mps(_fixture_site_tensors("mps_h2", (0, 1)), orbitals=orbitals)
    circuit = create("state_prep", "matrix_product_state", rotation_bit_precision=6).run(Wavefunction(mps))
    assert circuit.encoding == "jordan-wigner"
    assert circuit.num_qubits == 4


@pytest.mark.parametrize(
    ("make_wavefunction", "settings", "error", "message"),
    [
        pytest.param(
            lambda: Wavefunction(make_mps([_INITIAL_SITE, _IDENTITY_SITE])),
            {"unitary_synthesis": AlgorithmRef("circuit_mapper", "pauli_sequence")},
            TypeError,
            "requires the unitary_synthesis_dense or unitary_synthesis_block_sparse circuit mapper",
            id="other-circuit-mapper",
        ),
        pytest.param(create_test_wavefunction, {}, TypeError, "requires an MPSContainer", id="non-mps-wavefunction"),
        pytest.param(
            lambda: Wavefunction(make_mps([_INITIAL_SITE, _IDENTITY_SITE], orthogonality_center=None)),
            {},
            ValueError,
            "right-canonical MPS with center zero",
            id="unknown-orthogonality-center",
        ),
        pytest.param(
            lambda: Wavefunction(make_mps([_INITIAL_SITE.astype(complex), _IDENTITY_SITE.astype(complex)])),
            {},
            ValueError,
            "only real-valued",
            id="complex-tensors",
        ),
        pytest.param(
            lambda: Wavefunction(make_mps([np.zeros((1, 2, 2)), _IDENTITY_SITE])),
            {},
            ValueError,
            "finite amplitudes with nonzero norm",
            id="zero-initial-state",
        ),
        pytest.param(
            lambda: Wavefunction(make_mps([_INITIAL_SITE, np.ones((2, 2, 1))])),
            {},
            ValueError,
            "requires isometric sites",
            id="non-isometric-site",
        ),
    ],
)
def test_run_rejects_invalid_input(make_wavefunction, settings, error, message):
    """Each invalid wavefunction or site mapper is rejected with its error."""
    preparer = create("state_prep", "matrix_product_state", **settings)
    with pytest.raises(error, match=message):
        preparer.run(make_wavefunction())


# MPS state preparation composed with Trotterized QPE to recover classical eigenvalues.

# Per fixture, ``(evolution_time, num_divisions)`` keep the second-order Trotter error below 2e-5
# (mps_h2) and 4e-4 (mps_spin) in operator norm. Each tolerance exceeds half the energy bin width
# 2π/(evolution_time·2**_QPE_NUM_BITS), 0.041 and 0.020, plus that Trotter error.
_QPE_NUM_BITS = 7
_QPE_PROBLEM = {
    # Per fixture: evolution_time, trotter_num_divisions, energy_tolerance.
    "mps_h2": (0.6, 8, 0.05),
    "mps_spin": (1.2, 16, 0.03),
}


@_METHODS
@pytest.mark.parametrize("reversed_order", [False, True], ids=["identity-order", "reversed-order"])
@pytest.mark.parametrize("fixture_name", ["mps_h2", "mps_spin"])
def test_mps_trotter_qpe_recovers_eigenvalue(fixture_name, reversed_order, method):
    """Trotterized QPE on the prepared MPS recovers the ground eigenvalue of the fixture Hamiltonian."""
    order = tuple(reversed(_identity_order(fixture_name))) if reversed_order else _identity_order(fixture_name)
    mps = _fixture_mps(fixture_name, order)
    fixture = _FIXTURES[fixture_name]
    hamiltonian = QubitOperator(pauli_strings=fixture["pauli_strings"], coefficients=fixture["coefficients"])
    ground_eigenvalue = float(np.linalg.eigvalsh(hamiltonian.get_container().to_matrix())[0])
    assert ground_eigenvalue == pytest.approx(fixture["reference_eigenvalue"], abs=1e-8)
    circuit = create(
        "state_prep", "matrix_product_state", unitary_synthesis=synthesis_ref(method), rotation_bit_precision=6
    ).run(Wavefunction(mps))
    evolution_time, num_divisions, tolerance = _QPE_PROBLEM[fixture_name]

    qpe = StandardPhaseEstimation(shots=21)
    qpe.settings().set("circuit_executor", AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator", seed=7))
    qpe.settings().set(
        "qpe_circuit_builder",
        AlgorithmRef(
            "qpe_circuit_builder",
            "qdk_standard",
            num_bits=_QPE_NUM_BITS,
            controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
            unitary_builder=AlgorithmRef(
                "hamiltonian_unitary_builder", "trotter", time=evolution_time, order=2, num_divisions=num_divisions
            ),
        ),
    )
    result = qpe.run(qubit_hamiltonian=hamiltonian, state_preparation=circuit)

    assert abs(result.raw_energy - ground_eigenvalue) < tolerance


# Agreement of the dense and block-sparse circuits.

# Each site rotation angle is quantized to rotation_bit_precision bits, which bounds the
# infidelity of either circuit at O(2^-2b) per rotation.
_AGREEMENT_PRECISION = 10
_AGREEMENT_TOLERANCE = 1e-3


@pytest.mark.parametrize("reversed_order", [False, True], ids=["identity-order", "reversed-order"])
@pytest.mark.parametrize("fixture_name", ["mps_h2", "mps_spin"])
def test_dense_and_block_sparse_prepare_same_state(
    qsharp_test_context, qsharp_test_utils, fixture_name, reversed_order
):
    """The dense circuit, with absorbed right factors, and the block-sparse circuit prepare the same state."""
    order = tuple(reversed(_identity_order(fixture_name))) if reversed_order else _identity_order(fixture_name)
    mps = _fixture_mps(fixture_name, order)
    states = {}
    with use_qsharp_context(qsharp_test_context):
        for method in ("dense", "block_sparse"):
            probability, states[method] = simulate_mps_preparation(
                qsharp_test_utils, preparation_data(mps, method), method
            )
            assert probability > 1 - _AGREEMENT_TOLERANCE, method
    assert np.abs(np.vdot(states["dense"], states["block_sparse"])) ** 2 > 1 - _AGREEMENT_TOLERANCE


def test_sparse_site_isometry_scatters_symmetry_blocks():
    """The block-sparse site matrix built from unordered, partly empty symmetry blocks equals the dense isometry."""
    bond_syms = sym.SymmetryProduct([sym.axes.spin(1, False)])
    physical_syms = sym.SymmetryProduct([sym.axes.particle_number(2)])
    alpha, beta = sym.SymmetryLabel([sym.axes.alpha()]), sym.SymmetryLabel([sym.axes.beta()])
    zero, one, two = (sym.SymmetryLabel([sym.axes.particle_number_value(n)]) for n in range(3))
    physical_basis = [Configuration.from_spin_half_string(state) for state in ("u", "d", "0", "2")]
    extents = [{alpha: 2, beta: 1}, {zero: 1, one: 2, two: 1}, {alpha: 1, beta: 2}]
    rng = np.random.default_rng(3)
    blocks = []
    for labels in [(alpha, one, beta), (beta, one, alpha), (alpha, zero, alpha), (beta, two, beta)]:
        left, physical, right = (extents[slot][label] for slot, label in enumerate(labels))
        blocks.append((labels, rng.standard_normal((left * physical, right))))
    site = MPSSite(
        sym.SymmetryBlockedTensorRank3([bond_syms, physical_syms, bond_syms], extents, blocks),
        [beta, alpha],
        [one, zero, two],
        [beta, alpha],
        physical_basis,
    )
    ancilla_dim = 4

    matrix = MatrixProductStatePreparation._sparse_site_isometry(site, ancilla_dim)

    expected = MatrixProductStatePreparation._site_isometry(site, ancilla_dim).reshape(-1, site.left_bond_dimension())
    assert isinstance(matrix, sparse.csc_array)
    assert matrix.nnz == np.count_nonzero(expected) == sum(block.size for _, block in blocks)
    assert np.array_equal(matrix.toarray(), expected)


def _random_right_canonical_tensors(bond_dims: list[int], physical: int, seed: int) -> list[np.ndarray]:
    """Random real right-canonical MPS tensors with the given bond dimensions and a normalized site zero."""
    rng = np.random.default_rng(seed)
    tensors = [rng.standard_normal((left, physical, right)) for left, right in itertools.pairwise(bond_dims)]
    for site in range(len(tensors) - 1, 0, -1):
        left, _, right = tensors[site].shape
        q_matrix, r_matrix = np.linalg.qr(tensors[site].reshape(left, -1).T)
        tensors[site] = q_matrix.T.reshape(left, physical, right)
        tensors[site - 1] = tensors[site - 1] @ r_matrix.T
    tensors[0] /= np.linalg.norm(tensors[0])
    return tensors


def _target_state(mps: MPSContainer) -> np.ndarray:
    """Contract the MPS into the blocked Jordan-Wigner qubit basis of the prepared register.

    With one qubit per site the physical index is the ``('0', '1')`` basis of a spinless mode; with two
    it is ``('0', 'u', 'd', '2')``, whose bits create the alpha and then the beta mode. The mode of
    channel ``c`` on orbital ``o`` is qubit ``c * num_sites + o`` (qubit 0 most significant), and each
    amplitude carries the sign of reordering the creation operators from chain order into qubit order.
    """
    qubits_per_site = 1 if mps.sites()[0].physical_dimension() == 2 else 2
    num_sites, order = mps.num_sites(), list(mps.site_to_orbital_order())
    num_qubits = qubits_per_site * num_sites
    chain = np.ones((1, 1))
    for site in mps.sites():
        tensor = np.asarray(site.to_dense()).reshape(site.shape())
        chain = np.einsum("sa,apb->spb", chain, tensor).reshape(-1, tensor.shape[2])
    chain = chain[:, 0] / np.linalg.norm(chain[:, 0])
    target = np.zeros(1 << num_qubits)
    for index in np.flatnonzero(chain):
        created = []
        for site, orbital in enumerate(order):
            physical = (int(index) >> (qubits_per_site * (num_sites - 1 - site))) & ((1 << qubits_per_site) - 1)
            created += [channel * num_sites + orbital for channel in range(qubits_per_site) if physical >> channel & 1]
        inversions = sum(first > second for i, first in enumerate(created) for second in created[i + 1 :])
        target[sum(1 << (num_qubits - 1 - qubit) for qubit in created)] = (-1) ** inversions * chain[index]
    return target


@_METHODS
def test_multi_site_mps_matches_contraction(qsharp_test_context, qsharp_test_utils, method):
    """A four-site MPS with unequal bonds, a nontrivial right factor and a permuted order prepares its contraction."""
    mps = make_mps(_random_right_canonical_tensors([1, 4, 3, 2, 1], 4, seed=11), site_to_orbital_order=(2, 0, 3, 1))
    if method == "dense":
        last = MatrixProductStatePreparation._site_isometry(mps.sites()[-1], ancilla_dim=4)
        mapper = create("circuit_mapper", "unitary_synthesis_dense", num_input_qubits=2)
        unitary = UnitaryRepresentation(UnitaryMatrixContainer(last.reshape(-1, 2)))
        right_factor = mapper.synthesize(unitary)["rightFactor"]
        assert not np.allclose(np.abs(right_factor), np.eye(2), atol=1e-3)
    params = preparation_data(mps, method)
    with use_qsharp_context(qsharp_test_context):
        probability, state = simulate_mps_preparation(qsharp_test_utils, params, method)
    assert params["numAncillaQubits"] == 2
    assert probability > 1 - _AGREEMENT_TOLERANCE
    assert np.abs(np.vdot(_target_state(mps), state)) ** 2 > 1 - _AGREEMENT_TOLERANCE


@_METHODS
def test_six_site_identity_order_signs_match_contraction(qsharp_test_context, qsharp_test_utils, method):
    """A six-site identity-order MPS, whose signs use the parity ladder, prepares its signed contraction."""
    rng = np.random.default_rng(6)
    tensors = []
    for site in range(6):
        # Alternating (u, d) and (0, 2) superpositions give configurations with both fermionic signs.
        values = np.zeros(4)
        values[[1, 2] if site % 2 == 0 else [0, 3]] = rng.uniform(0.5, 1.0, 2)
        tensors.append((values / np.linalg.norm(values)).reshape(1, 4, 1))
    mps = make_mps(tensors)
    target = _target_state(mps)
    assert (target < 0).any()
    assert (target > 0).any()
    params = preparation_data(mps, method)
    with use_qsharp_context(qsharp_test_context):
        probability, state = simulate_mps_preparation(qsharp_test_utils, params, method)
    assert probability > 1 - _AGREEMENT_TOLERANCE
    assert np.abs(np.vdot(target, state)) ** 2 > 1 - _AGREEMENT_TOLERANCE


@_METHODS
def test_shared_phase_gradient_prepares_target(method):
    """Without its own gradient the circuit acts on [state | gradient], prepares the MPS and restores the gradient."""
    mps = make_mps(_random_right_canonical_tensors([1, 4, 3, 2, 1], 4, seed=11), site_to_orbital_order=(2, 0, 3, 1))
    # Wavefunction takes ownership of the container, so contract it first.
    target = _target_state(mps)
    circuit = create(
        "state_prep",
        "matrix_product_state",
        unitary_synthesis=synthesis_ref(method),
        rotation_bit_precision=_AGREEMENT_PRECISION,
        allocate_phase_gradient=False,
    ).run(Wavefunction(mps))
    assert circuit.metadata.num_phase_gradient_ancillas == _AGREEMENT_PRECISION
    num_state_qubits = circuit.num_qubits - _AGREEMENT_PRECISION
    op = QSHARP_UTILS.CircuitComposition.MakeSharedAncillaOp(
        circuit._qsharp_op, QSHARP_UTILS.PhaseGradient.PreparePhaseGradientState, _AGREEMENT_PRECISION
    )
    output = apply_operation(op, np.eye(1, 1 << circuit.num_qubits).ravel())
    output = output.reshape(1 << num_state_qubits, 1 << _AGREEMENT_PRECISION)
    np.testing.assert_allclose(output[:, 1:], 0.0, atol=1e-8, err_msg="phase gradient was not restored")
    assert np.abs(np.vdot(target, output[:, 0])) ** 2 > 1 - _AGREEMENT_TOLERANCE


def test_operation_leaves_later_system_qubits_untouched():
    """With its own gradient the operation prepares the MPS on the first qubits of a wider system register."""
    mps = _fixture_mps("mps_spin", _identity_order("mps_spin"))
    # Wavefunction takes ownership of the container, so contract it first.
    target = _target_state(mps)
    circuit = create("state_prep", "matrix_product_state", rotation_bit_precision=_AGREEMENT_PRECISION).run(
        Wavefunction(mps)
    )
    output = apply_operation(circuit._qsharp_op, np.eye(1, 2 << circuit.num_qubits).ravel()).reshape(-1, 2)
    np.testing.assert_allclose(output[:, 1], 0.0, atol=1e-8, err_msg="the extra system qubit was changed")
    assert np.abs(np.vdot(target, output[:, 0])) ** 2 > 1 - _AGREEMENT_TOLERANCE


@_METHODS
def test_operation_adjoint_returns_to_zero(qsharp_test_context, qsharp_test_utils, method):
    """Preparing an MPS with unequal bonds and a permuted order, then applying the adjoint, returns to ``|0>``."""
    mps = make_mps(_random_right_canonical_tensors([1, 4, 3, 2, 1], 4, seed=11), site_to_orbital_order=(2, 0, 3, 1))
    with use_qsharp_context(qsharp_test_context):
        circuit = create(
            "state_prep",
            "matrix_product_state",
            unitary_synthesis=synthesis_ref(method),
            rotation_bit_precision=_AGREEMENT_PRECISION,
        ).run(Wavefunction(mps))
        round_trip = qsharp_test_utils.MPSPreparationTests.TestMakeRoundTripOp(circuit._qsharp_op)
        output = apply_operation(round_trip, np.eye(1, 1 << circuit.num_qubits).ravel())
    assert abs(output[0]) ** 2 > 1 - _AGREEMENT_TOLERANCE


def test_initial_state_normalization_ignores_overall_scale():
    """Scaling site zero by any finite nonzero factor leaves the prepared initial state unchanged."""
    tensors = [np.asarray(t) for t in _fixture_site_tensors("mps_spin", _identity_order("mps_spin"))]
    expected = preparation_data(make_mps(tensors))["initialStateVec"]
    for scale in (1e-16, 1e200):
        scaled = preparation_data(make_mps([tensors[0] * scale, *tensors[1:]]))["initialStateVec"]
        np.testing.assert_allclose(scaled, expected, atol=1e-14)


def _two_site_tensors(amplitudes: dict[tuple[int, int], float]):
    """Right-canonical two-site tensors of the state with the given ``(physical0, physical1)`` amplitudes."""
    psi = np.zeros((4, 4))
    for (first, second), value in amplitudes.items():
        psi[first, second] = value
    u, s, vt = np.linalg.svd(psi / np.linalg.norm(psi))
    rank = int(np.sum(s > 1e-12))
    return [(u[:, :rank] * s[:rank]).reshape(1, 4, rank), vt[:rank].reshape(rank, 4, 1)]


# Physical basis (0, u, d, 2) indices of the singly occupied spin-up and spin-down site states.
_UP, _DOWN = 1, 2
_OPEN_SHELL_STATES = {
    "triplet_plus": {(_UP, _UP): 1.0},
    "triplet_zero": {(_UP, _DOWN): 1.0, (_DOWN, _UP): 1.0},
    "open_shell_singlet": {(_UP, _DOWN): 1.0, (_DOWN, _UP): -1.0},
}


@_METHODS
@pytest.mark.parametrize("reversed_order", [False, True], ids=["identity-order", "reversed-order"])
def test_open_shell_states_have_fermionic_signs(qsharp_test_context, qsharp_test_utils, reversed_order, method):
    """The M_s = 0 triplet is degenerate with the sign-free triplet and the open-shell singlet is not."""
    order = (1, 0) if reversed_order else (0, 1)
    fixture = _FIXTURES["mps_h2"]
    hamiltonian = (
        QubitOperator(pauli_strings=fixture["pauli_strings"], coefficients=fixture["coefficients"])
        .get_container()
        .to_matrix()
    )
    energies = {}
    with use_qsharp_context(qsharp_test_context):
        for name, amplitudes in _OPEN_SHELL_STATES.items():
            params = preparation_data(make_mps(_two_site_tensors(amplitudes), site_to_orbital_order=order), method)
            probability, state = simulate_mps_preparation(qsharp_test_utils, params, method)
            assert probability > 1 - _AGREEMENT_TOLERANCE, name
            # The dump has qubit 0 most significant; the Pauli matrix has it least significant.
            state = state.reshape((2,) * 4).transpose().reshape(-1)
            energies[name] = float(np.real(np.vdot(state, hamiltonian @ state)))
    assert energies["triplet_zero"] == pytest.approx(energies["triplet_plus"], abs=_AGREEMENT_TOLERANCE)
    assert energies["open_shell_singlet"] - energies["triplet_plus"] > 0.1


# Resource-estimate regressions pinned from this implementation's own circuits.

# (fixture, method) -> (numQubits, cczCount, ccixCount) at rotation_bit_precision=6 and the identity
# site_to_orbital_order.
_EXPECTED_RESOURCES = {
    ("mps_h2", "dense"): (24, 144, 0),
    ("mps_h2", "block_sparse"): (24, 45, 0),
    ("mps_spin", "dense"): (21, 22, 0),
    ("mps_spin", "block_sparse"): (21, 14, 0),
}


@_METHODS
@pytest.mark.parametrize("fixture_name", ["mps_h2", "mps_spin"])
def test_resource_estimate_regression(fixture_name, method):
    """Pin logical resource counts of the mps_h2/mps_spin circuits against a silent regression."""
    circuit = create(
        "state_prep", "matrix_product_state", unitary_synthesis=synthesis_ref(method), rotation_bit_precision=6
    ).run(Wavefunction(_fixture_mps(fixture_name, _identity_order(fixture_name))))
    counts = circuit.estimate().logical_counts
    actual = (counts["numQubits"], counts["cczCount"], counts["ccixCount"])
    assert actual == _EXPECTED_RESOURCES[(fixture_name, method)]


@_METHODS
def test_circuit_exports_qir(method):
    """The circuit of a permuted-order MPS lowers to QIR, which rejects whole-array comparisons."""
    circuit = create(
        "state_prep", "matrix_product_state", unitary_synthesis=synthesis_ref(method), rotation_bit_precision=6
    ).run(Wavefunction(_fixture_mps("mps_h2", (1, 0))))
    assert str(circuit.get_qir())
