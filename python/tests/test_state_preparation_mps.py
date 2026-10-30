"""Tests for matrix product state preparation with dense and block-sparse unitary synthesis.

Covers settings and input validation, agreement of the dense and block-sparse Q# circuits,
fermionic order signs on open-shell states, composition with Trotterized quantum phase estimation
to recover classical eigenvalues, and resource-estimate regressions. The site isometries are
synthesized by the ``unitary_synthesis`` circuit mapper, which has its own tests. Fixture data
(``mps_h2``, ``mps_spin``) lives in ``test_data/mps_h2.mps.json``/``mps_spin.mps.json``, generated
by ``test_data/make_mps_fixtures.py``.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
from pathlib import Path

import numpy as np
import pytest
from qdk.qsharp import QSharpError

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.phase_estimation.standard_phase_estimation import StandardPhaseEstimation
from qdk_chemistry.algorithms.state_preparation.matrix_product_state import MatrixProductStatePreparation
from qdk_chemistry.data import AlgorithmRef, MPSContainer, MPSSite, Orbitals, QubitOperator, Wavefunction
from qdk_chemistry.data import symmetry as sym
from qdk_chemistry.utils.qsharp import create_qsharp_context, get_qsharp_context, use_qsharp_context

from .helpers import create_test_basis_set, create_test_orbitals, create_test_wavefunction

# Q# operation and site struct of each synthesis method.
_QSHARP_OPERATIONS = {
    "dense": (
        "QDKChemistry.Utils.MPSSequential.MPSSequential",
        "QDKChemistry.Utils.IsometrySynthesis.DenseIsometrySynthesis",
    ),
    "block_sparse": (
        "QDKChemistry.Utils.MPSSparse.MPSSparse",
        "QDKChemistry.Utils.IsometrySynthesis.SparseIsometrySynthesis",
    ),
}
_METHODS = pytest.mark.parametrize("method", list(_QSHARP_OPERATIONS))
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
    return AlgorithmRef("circuit_mapper", "unitary_synthesis", method=method)


def preparation_data(mps: MPSContainer, method: str = "dense"):
    """Generate preparation data from a container using the selected site synthesis method."""
    preparer = MatrixProductStatePreparation()
    preparer.settings().update("unitary_synthesis", synthesis_ref(method))
    return preparer.generate_matrix_product_state_preparation_data(mps)


def simulate_mps_preparation(operation, site_struct, params, num_sites, num_ancilla_qubits):
    """Run an MPS preparation operation and post-select the bond register on ``|0>``."""
    arguments = ", ".join(
        _to_qsharp_literal(params[name], site_struct)
        for name in ("initialStateVec", "numSites", "siteToOrbitalOrder", "siteDecompositions")
    )
    num_state_qubits = params["numQubitsPerSite"] * num_sites
    context = get_qsharp_context()
    context.eval(f"use state = Qubit[{num_state_qubits}];")
    context.eval(f"use ancilla = Qubit[{num_ancilla_qubits}];")
    context.eval(
        f"{{ use phaseGradient = Qubit[{params['rotationBitPrecision']}]; "
        "within { QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState(phaseGradient); } "
        f"apply {{ {operation}({arguments}, state, ancilla, phaseGradient); }} }}"
    )
    dump = context.dump_machine()
    context.eval("ResetAll(state + ancilla);")
    num_relevant, ancilla_mask = num_state_qubits + num_ancilla_qubits, (1 << num_ancilla_qubits) - 1
    amplitudes = np.zeros(1 << num_state_qubits, dtype=complex)
    for idx in dump:
        if idx >> num_relevant == 0 and idx & ancilla_mask == 0:
            amplitudes[idx >> num_ancilla_qubits] = dump[idx]
    probability = float(np.sum(np.abs(amplitudes) ** 2))
    return probability, amplitudes / np.sqrt(probability)


_GIVENS_STRUCT = "QDKChemistry.Utils.UnitarySynthesis.GivensDecomposition"
_GIVENS_FIELDS = {"layerAngles", "layerShifted", "phases"}


def _to_qsharp_literal(value, site_struct: str) -> str:
    """Serialize nested numeric/Boolean data as a Q# literal (``GivensDecomposition`` or ``site_struct``)."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.15f}"
    if isinstance(value, dict):
        fields = ", ".join(f"{name} = {_to_qsharp_literal(item, site_struct)}" for name, item in value.items())
        struct = _GIVENS_STRUCT if value.keys() == _GIVENS_FIELDS else site_struct
        return f"new {struct} {{ {fields} }}"
    if isinstance(value, list):
        return f"[{', '.join(_to_qsharp_literal(item, site_struct) for item in value)}]"
    return str(value)


# File-backed mps_h2 / mps_spin fixtures (see test_data/make_mps_fixtures.py).
_FIXTURES = {name: json.loads((_FIXTURE_DIR / f"{name}.mps.json").read_text()) for name in ("mps_h2", "mps_spin")}


def _fixture_mps(name: str, order: tuple[int, ...]) -> MPSContainer:
    return make_mps(_fixture_site_tensors(name, order), site_to_orbital_order=order)


def _fixture_site_tensors(name: str, order: tuple[int, ...]):
    """Return the per-order right-canonical MPS tensors for *name* (see make_mps_fixtures.py)."""
    return _FIXTURES[name]["orders"][",".join(map(str, order))]["site_tensors"]


def _fixture_hamiltonian(name: str) -> QubitOperator:
    """Return the fixture's one fixed physical Hamiltonian, qubit labels reversed for ``qdk_standard`` QPE.

    The Hamiltonian is the same physical operator for every ``site_to_orbital_order``: only the
    per-order MPS tensors (see ``_fixture_site_tensors``) change, so that each order's circuit still
    prepares the identical physical eigenstate. The standard QPE circuit builder's ``pauli_sequence``
    controlled-unitary mapper pairs state register qubit ``i`` with Hamiltonian qubit
    ``num_qubits - 1 - i`` (the same convention the existing ``four_qubit_phase_problem`` fixture in
    ``test_phase_estimation_standard.py`` handles via ``rowMap=[3, 2, 1, 0]``); reversing each Pauli
    string here keeps the MPS circuit's own (unreversed) wire order aligned with it.
    """
    fixture = _FIXTURES[name]
    reversed_strings = [string[::-1] for string in fixture["pauli_strings"]]
    return QubitOperator(pauli_strings=reversed_strings, coefficients=fixture["coefficients"])


def _identity_order(name: str) -> tuple[int, ...]:
    return tuple(range(_FIXTURES[name]["num_sites"]))


def _zeroed_fixture_mps(name: str) -> MPSContainer:
    """Zero only the loaded initial-state tensor of *name*, keeping later sites valid isometries.

    Site zero becomes the (null) initial state; later sites are still isometric, so the
    zero-norm check fires instead of the earlier isometric-site check.
    """
    tensors = [np.asarray(t) for t in _fixture_site_tensors(name, _identity_order(name))]
    tensors[0] = np.zeros_like(tensors[0])
    return make_mps(tensors)


def _active_space_fixture_mps(name: str) -> MPSContainer:
    """Place the *name* sites on the two active orbitals of a four-orbital space (one core, one virtual)."""
    orbitals = Orbitals(
        np.eye(4),
        np.array([-1.0, -0.5, 0.5, 1.0]),
        None,
        create_test_basis_set(4, "mps-active-space"),
        sym.spin_index_set(4, [1, 2], [1, 2]),
    )
    return make_mps(_fixture_site_tensors(name, _identity_order(name)), orbitals=orbitals)


def _non_isometric_fixture_mps(name: str) -> MPSContainer:
    """Overwrite every loaded site tensor of *name* with an all-ones tensor of the same shape."""
    return make_mps([np.ones_like(np.asarray(t)) for t in _fixture_site_tensors(name, _identity_order(name))])


def _malformed_givens_params(field: str):
    """Build real ``to_qsharp_params`` data from a loaded fixture, then truncate one Givens field."""
    data = preparation_data(_fixture_mps("mps_spin", _identity_order("mps_spin")), "dense")
    params = data.to_qsharp_params(rotation_bit_precision=6)
    block_givens = params["siteDecompositions"][0]["blockGivens"]
    block_givens[field] = block_givens[field][:-1]
    with use_qsharp_context(create_qsharp_context()):
        simulate_mps_preparation(*_QSHARP_OPERATIONS["dense"], params, 2, data.ancilla_bits)


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        pytest.param(
            lambda: create("state_prep", "matrix_product_state").settings().get("unitary_synthesis"),
            lambda ref: (ref.algorithm_type, ref.algorithm_name) == ("circuit_mapper", "unitary_synthesis"),
            id="settings-default-is-unitary-synthesis-mapper",
        ),
        pytest.param(
            lambda: create(
                "state_prep",
                "matrix_product_state",
                unitary_synthesis=AlgorithmRef("circuit_mapper", "pauli_sequence"),
            ).run(Wavefunction(_fixture_mps("mps_h2", _identity_order("mps_h2")))),
            (TypeError, "requires the unitary_synthesis circuit mapper"),
            id="run-rejects-other-circuit-mapper",
        ),
        pytest.param(
            lambda: create("state_prep", "matrix_product_state").settings().update("rotation_bit_precision", 0),
            (ValueError, "out of allowed range"),
            id="settings-rejects-out-of-range-precision",
        ),
        pytest.param(
            lambda: create("state_prep", "matrix_product_state").run(create_test_wavefunction()),
            (TypeError, "requires an MPSContainer"),
            id="run-rejects-non-mps-wavefunction",
        ),
        pytest.param(
            lambda: MatrixProductStatePreparation().generate_matrix_product_state_preparation_data(
                _fixture_mps("mps_h2", _identity_order("mps_h2")).sites
            ),
            (TypeError, "requires an MPSContainer"),
            id="generate-rejects-raw-site-list",
        ),
        pytest.param(
            lambda: create("state_prep", "matrix_product_state", rotation_bit_precision=6).run(
                Wavefunction(_fixture_mps("mps_h2", _identity_order("mps_h2")))
            ),
            lambda circuit: circuit.encoding == "jordan-wigner"
            and circuit.num_qubits == _FIXTURES["mps_h2"]["num_sites"] * _FIXTURES["mps_h2"]["qubits_per_site"],
            id="valid-mps-register-width-matches-sites",
        ),
        pytest.param(
            lambda: create("state_prep", "matrix_product_state", rotation_bit_precision=6).run(
                Wavefunction(_active_space_fixture_mps("mps_h2"))
            ),
            lambda circuit: circuit.num_qubits
            == _FIXTURES["mps_h2"]["num_sites"] * _FIXTURES["mps_h2"]["qubits_per_site"],
            id="active-space-mps-register-width-matches-sites",
        ),
        pytest.param(
            lambda: create("state_prep", "matrix_product_state").run(
                Wavefunction(
                    make_mps(_fixture_site_tensors("mps_h2", _identity_order("mps_h2")), orthogonality_center=None)
                )
            ),
            (ValueError, "right-canonical MPS with center zero"),
            id="rejects-unknown-orthogonality-center",
        ),
        pytest.param(
            lambda: create("state_prep", "matrix_product_state").run(
                Wavefunction(
                    make_mps(
                        [
                            np.asarray(t, dtype=complex)
                            for t in _fixture_site_tensors("mps_h2", _identity_order("mps_h2"))
                        ]
                    )
                )
            ),
            (ValueError, "only real-valued"),
            id="rejects-complex-tensors",
        ),
        pytest.param(
            lambda: create("state_prep", "matrix_product_state").run(Wavefunction(_zeroed_fixture_mps("mps_spin"))),
            (ValueError, "finite amplitudes with nonzero norm"),
            id="rejects-zero-state",
        ),
        pytest.param(
            lambda: create("state_prep", "matrix_product_state").run(
                Wavefunction(_non_isometric_fixture_mps("mps_spin"))
            ),
            (ValueError, "isometric"),
            id="rejects-non-isometric-site",
        ),
        pytest.param(
            lambda: _malformed_givens_params("layerShifted"),
            (QSharpError, "one shift flag per layer"),
            id="rejects-malformed-givens-shift-flags",
        ),
        pytest.param(
            lambda: _malformed_givens_params("phases"),
            (QSharpError, "one entry per basis state of the target register"),
            id="rejects-malformed-givens-phases",
        ),
    ],
)
def test_state_preparation_contract(action, expected):
    """Settings, input-validation, and malformed-Givens-data contract checks, in one place.

    Each case is a zero-argument *action*; ``expected`` is either an ``(exception_type, message)``
    pair (the action must raise it) or a validator called on the action's return value. Invalid
    MPS inputs are built by modifying copies of the real ``mps_h2``/``mps_spin`` fixture tensors
    (zeroed, cast to complex, replaced by same-shaped all-ones arrays, orthogonality center
    cleared) rather than from a standalone random-MPS generator or inline literal tensors.
    """
    if isinstance(expected, tuple):
        exc_type, message = expected
        with pytest.raises(exc_type, match=message):
            action()
    else:
        assert expected(action())


# MPS state preparation composed with Trotterized QPE to recover classical eigenvalues.

# Both Hamiltonians have noncommuting terms (a real H2/STO-3G electronic Hamiltonian, and an
# anisotropic XY chain with a staggered field), so QPE recovers the classical ground eigenvalue
# only approximately. Per fixture, ``(evolution_time, num_divisions)`` were chosen so a
# second-order Trotter unitary matches the exact e^{-iHt} to within <2e-5 (mps_h2) / <4e-4
# (mps_spin) operator norm (negligible next to the phase-estimation bin width below);
# ``_QPE_NUM_BITS`` bits resolve the phase fraction to within half a bin of width
# ``1/(evolution_time * 2**_QPE_NUM_BITS)``, which dominates the energy error. The tolerance below
# is 1.25x that half-bin width plus the Trotter error above, so it is derived from these
# resolution/approximation limits rather than relaxed to force a pass.
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
    """Preparing an MPS and running Trotterized standard QPE recovers the classical eigenvalue.

    ``mps_h2`` is the real H2/STO-3G electronic Hamiltonian (2 chemistry sites, physical dimension
    4, one spatial orbital each); ``mps_spin`` is a small anisotropic XY chain with a staggered
    field (2 spin sites, physical dimension 2). The fixture Hamiltonian is one fixed physical
    operator, independent of ``site_to_orbital_order``: only the MPS tensors differ per order (see
    ``make_mps_fixtures.py``), so a fermionic-sign or bit-order bug in the preparation circuit
    would make one order (but not necessarily both) fail to recover the eigenvalue. The classical
    ground eigenvalue is derived here directly from that same fixed Hamiltonian by diagonalizing
    its dense matrix, rather than solely trusting the fixture's stored scalar.
    """
    order = tuple(reversed(_identity_order(fixture_name))) if reversed_order else _identity_order(fixture_name)
    mps = _fixture_mps(fixture_name, order)
    hamiltonian = _fixture_hamiltonian(fixture_name)
    ground_eigenvalue = float(np.linalg.eigvalsh(hamiltonian.get_container().to_matrix())[0])
    assert ground_eigenvalue == pytest.approx(_FIXTURES[fixture_name]["reference_eigenvalue"], abs=1e-8)
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
def test_dense_and_block_sparse_prepare_same_state(fixture_name, reversed_order):
    """The dense and block-sparse circuits prepare the same state and return the bond register to ``|0>``.

    The dense circuits leave out the right factor of each site isometry, which is absorbed into the
    preceding site or the initial state, while the block-sparse circuits synthesize every isometry in
    full, so their agreement checks the absorption.
    """
    order = tuple(reversed(_identity_order(fixture_name))) if reversed_order else _identity_order(fixture_name)
    mps = _fixture_mps(fixture_name, order)
    states = {}
    with use_qsharp_context(create_qsharp_context()):
        for method, (operation, site_struct) in _QSHARP_OPERATIONS.items():
            data = preparation_data(mps, method)
            params = data.to_qsharp_params(rotation_bit_precision=_AGREEMENT_PRECISION)
            probability, states[method] = simulate_mps_preparation(
                operation, site_struct, params, data.num_sites, data.ancilla_bits
            )
            assert probability > 1 - _AGREEMENT_TOLERANCE, method
    assert np.abs(np.vdot(states["dense"], states["block_sparse"])) ** 2 > 1 - _AGREEMENT_TOLERANCE


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
def test_open_shell_states_have_fermionic_signs(reversed_order, method):
    """Open-shell H2 states land in the right spin multiplet, which needs the fermionic order signs.

    The fixture ground state occupies only ``20``/``02``, which never reorders spin-up past
    spin-down operators. Here ``ud +/- du`` mixes configurations whose Jordan-Wigner order signs
    differ, so dropping or misapplying the sign correction swaps the ``M_s = 0`` triplet and the
    open-shell singlet. The triplet component must be degenerate with the sign-free ``uu``
    state, and the singlet must not be.
    """
    order = (1, 0) if reversed_order else (0, 1)
    fixture = _FIXTURES["mps_h2"]
    hamiltonian = (
        QubitOperator(pauli_strings=fixture["pauli_strings"], coefficients=fixture["coefficients"])
        .get_container()
        .to_matrix()
    )
    operation, site_struct = _QSHARP_OPERATIONS[method]
    energies = {}
    with use_qsharp_context(create_qsharp_context()):
        for name, amplitudes in _OPEN_SHELL_STATES.items():
            data = preparation_data(make_mps(_two_site_tensors(amplitudes), site_to_orbital_order=order), method)
            params = data.to_qsharp_params(rotation_bit_precision=_AGREEMENT_PRECISION)
            probability, state = simulate_mps_preparation(
                operation, site_struct, params, data.num_sites, data.ancilla_bits
            )
            assert probability > 1 - _AGREEMENT_TOLERANCE, name
            energies[name] = float(np.real(np.vdot(state, hamiltonian @ state)))
    assert energies["triplet_zero"] == pytest.approx(energies["triplet_plus"], abs=_AGREEMENT_TOLERANCE)
    assert energies["open_shell_singlet"] - energies["triplet_plus"] > 0.1


# Resource-estimate regressions pinned from this implementation's own circuits.

# (fixture, method) -> (numQubits, cczCount), measured from this repository's own circuits at
# rotation_bit_precision=6 and the identity site_to_orbital_order; other orders are covered by
# the QPE-recovery test above, so only one canonical order is pinned here.
_EXPECTED_RESOURCES = {
    ("mps_h2", "dense"): (30, 135),
    ("mps_h2", "block_sparse"): (30, 79),
    ("mps_spin", "dense"): (27, 22),
    ("mps_spin", "block_sparse"): (27, 14),
}


@_METHODS
@pytest.mark.parametrize("fixture_name", ["mps_h2", "mps_spin"])
def test_resource_estimate_regression(fixture_name, method):
    """Pin logical resource counts of the mps_h2/mps_spin circuits against a silent regression."""
    circuit = create(
        "state_prep", "matrix_product_state", unitary_synthesis=synthesis_ref(method), rotation_bit_precision=6
    ).run(Wavefunction(_fixture_mps(fixture_name, _identity_order(fixture_name))))
    counts = circuit.estimate().logical_counts
    expected_qubits, expected_ccz = _EXPECTED_RESOURCES[(fixture_name, method)]
    assert counts["numQubits"] == expected_qubits
    assert counts["cczCount"] == expected_ccz
