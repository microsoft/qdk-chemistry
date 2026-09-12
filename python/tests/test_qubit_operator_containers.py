"""Tests for the qubit operator wrapper and representation containers."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import h5py
import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.sossa import SOSSABuilder
from qdk_chemistry.algorithms.qubit_mapper.sos import SOSQubitMapper
from qdk_chemistry.data import FactorizedHamiltonianContainer, Hamiltonian, MajoranaMapping, QubitOperator
from qdk_chemistry.data.qubit_operator.containers.base import QubitOperatorContainer
from qdk_chemistry.data.qubit_operator.containers.pauli_lcu import PauliLCUContainer
from qdk_chemistry.data.qubit_operator.containers.sos import (
    FactorizedHamiltonianMetadata,
    RotatedPaulis,
    SOSContainer,
)

from .test_helpers import create_random_factorized_hamiltonian, create_test_orbitals


def test_qubit_operator_wraps_pauli_lcu_container() -> None:
    """Shared metadata and the compatibility Pauli API delegate to the container."""
    container = PauliLCUContainer(["XI", "ZZ"], np.array([0.5, -0.25]), "jordan-wigner", "blocked")
    operator = QubitOperator(container)

    assert isinstance(container, QubitOperatorContainer)
    assert operator.get_container() is container
    assert operator.get_container_type() == "pauli_lcu"
    assert operator.num_qubits == 2
    assert operator.encoding == "jordan-wigner"
    assert operator.fermion_mode_order == "blocked"
    assert operator.pauli_strings is container.pauli_strings
    assert operator.coefficients is container.coefficients
    assert operator.schatten_norm == container.schatten_norm
    np.testing.assert_allclose(operator.to_matrix(), container.to_matrix())
    assert operator.is_hermitian()
    assert operator.equiv(QubitOperator(PauliLCUContainer(["ZZ", "XI"], np.array([-0.25, 0.5]))))

    scaled = 2 * operator
    added = operator + operator
    assert isinstance(scaled, QubitOperator)
    assert isinstance(added, QubitOperator)
    np.testing.assert_allclose(scaled.coefficients, np.array([1.0, -0.5]))
    assert added.pauli_strings == ["XI", "ZZ", "XI", "ZZ"]

    restored = QubitOperator.from_json(operator.to_json())
    assert restored.get_container_type() == "pauli_lcu"
    assert restored.content_hash() == operator.content_hash()


def test_qubit_operator_requires_coefficients_with_pauli_strings() -> None:
    """Pauli strings alone are neither a container nor a complete legacy call."""
    with pytest.raises(TypeError, match="QubitOperator requires a QubitOperatorContainer"):
        QubitOperator(["X"])  # type: ignore[arg-type]


def test_qubit_operator_still_accepts_the_legacy_positional_constructor() -> None:
    """``QubitOperator(pauli_strings, coefficients)`` is a shipped signature and still works."""
    operator = QubitOperator(["XI", "ZZ"], np.array([0.5, -0.25]))

    assert operator.get_container_type() == "pauli_lcu"
    assert operator.pauli_strings == ["XI", "ZZ"]
    np.testing.assert_allclose(operator.coefficients, np.array([0.5, -0.25]))


def test_pauli_lcu_json_preserves_coefficient_shape_and_dtype() -> None:
    """Pauli LCU and SOS coefficients use the same complex-array wire format."""
    coefficients = np.array([0.5 + 0.25j, -0.25j], dtype=np.complex64)
    container = PauliLCUContainer(["XI", "ZZ"], coefficients)

    json_data = container.to_json()
    restored = PauliLCUContainer.from_json(json_data)

    assert json_data["coefficients"] == {
        "real": [0.5, 0.0],
        "imag": [0.25, -0.25],
        "shape": [2],
        "dtype": "complex64",
    }
    np.testing.assert_array_equal(restored.coefficients, coefficients)
    assert restored.coefficients.dtype == coefficients.dtype


def test_pauli_lcu_reads_complex_coefficients_without_shape() -> None:
    """Read coefficient dictionaries written before the shared array codec added shape."""
    json_data = PauliLCUContainer(["X"], np.array([0.5 + 0.25j])).to_json()
    del json_data["coefficients"]["shape"]

    restored = PauliLCUContainer.from_json(json_data)

    np.testing.assert_array_equal(restored.coefficients, np.array([0.5 + 0.25j]))


def test_qubit_operator_reads_documents_written_before_container_dispatch() -> None:
    """A hand-authored pre-container document loads as a Pauli LCU operator.

    Pre-container releases wrote ``pauli_strings``/``coefficients`` at version ``0.1.0``
    with no ``container_type`` and no ``shape``/``dtype`` on the coefficient dict. The
    missing key must default to ``pauli_lcu`` and the container's ``0.1.0`` guard must
    accept the document unchanged; a fixture (not a stripped round-trip of current output)
    is what pins that legacy schema.
    """
    legacy = {
        "pauli_strings": ["XI", "ZZ"],
        "coefficients": {"real": [0.5, -0.25], "imag": [0.0, 0.0]},
        "encoding": "jordan-wigner",
        "version": "0.1.0",
    }

    restored = QubitOperator.from_json(legacy)

    assert restored.get_container_type() == "pauli_lcu"
    assert restored.pauli_strings == ["XI", "ZZ"]
    np.testing.assert_allclose(restored.coefficients, np.array([0.5, -0.25]))
    assert restored.encoding == "jordan-wigner"


def test_qubit_operator_reads_hdf5_groups_written_before_container_dispatch(tmp_path) -> None:
    """A hand-authored pre-container HDF5 group loads with the same missing-key default."""
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("operator")
        group.attrs["version"] = "0.1.0"
        group.attrs["encoding"] = "jordan-wigner"
        group.create_dataset("pauli_strings", data=np.array(["XI", "ZZ"], dtype="S"))
        group.create_dataset("coefficients", data=np.array([0.5, -0.25], dtype=complex))

    with h5py.File(path, "r") as handle:
        restored = QubitOperator.from_hdf5(handle["operator"])

    assert restored.get_container_type() == "pauli_lcu"
    assert restored.pauli_strings == ["XI", "ZZ"]
    np.testing.assert_allclose(restored.coefficients, np.array([0.5, -0.25]))
    assert restored.encoding == "jordan-wigner"


def test_sos_container_json_roundtrip_preserves_complex_coefficients() -> None:
    """Complex LCU coefficients and Givens angles survive a JSON round-trip.

    The SOS generators carry the D1/Q1 ``+/-i`` sign in the imaginary part, so a
    serializer that silently drops it would still produce a well-formed container
    while flipping particle generators into hole generators.
    """
    one_body_coeffs = np.array([[0.2, 0.2j], [0.3, -0.3j]])
    container = SOSContainer(
        one_body=RotatedPaulis(np.array([[0.1], [0.2]]), one_body_coeffs, ("X", "Y")),
        two_body=RotatedPaulis(np.array([[0.3]]), np.array([[0.3, 0.7]]), ("Z",)),
        encoding="jordan-wigner",
        fermion_mode_order="blocked",
        metadata=FactorizedHamiltonianMetadata(
            num_spatial_orbitals=2,
            num_ranks=1,
            num_bases=1,
            num_copies=1,
            num_positive_one_body_terms=1,
            energy_shift=-1.5,
        ),
    )

    json_data = QubitOperator(container).to_json()
    assert json_data["one_body_coeffs"]["shape"] == [2, 2]
    assert json_data["one_body_coeffs"]["dtype"] == "complex128"
    del json_data["one_body_coeffs"]["dtype"]
    del json_data["two_body_coeffs"]["dtype"]
    restored = QubitOperator.from_json(json_data).get_container()

    np.testing.assert_allclose(restored.one_body.coeffs, one_body_coeffs)
    np.testing.assert_allclose(restored.one_body.angles, container.one_body.angles)
    np.testing.assert_allclose(restored.two_body.coeffs, container.two_body.coeffs)
    np.testing.assert_allclose(restored.two_body.angles, container.two_body.angles)
    assert restored.metadata.num_positive_one_body_terms == 1
    assert restored.metadata.energy_shift == pytest.approx(-1.5)


def _factorized_with_one_body(h1: np.ndarray) -> FactorizedHamiltonianContainer:
    """Build a small factorized Hamiltonian carrying a chosen one-body matrix."""
    n = h1.shape[0]
    rng = np.random.default_rng(1)
    u = np.zeros(2 * n)
    for b in range(2):
        v = rng.standard_normal(n)
        u[b * n : (b + 1) * n] = v / np.linalg.norm(v)
    return FactorizedHamiltonianContainer(
        one_body_integrals=h1,
        u_matrices=u,
        w_matrices=rng.standard_normal(2),
        wb_matrix=rng.standard_normal((1, 1)),
        orbitals=create_test_orbitals(n),
        core_energy=0.0,
        inactive_fock_matrix=np.zeros((n, n)),
    )


def _one_body_block(h1: np.ndarray):
    """Map a one-body matrix through the SOS mapper and return the one-body generator block."""
    n = h1.shape[0]
    operator = SOSQubitMapper().run(Hamiltonian(_factorized_with_one_body(h1)), MajoranaMapping.jordan_wigner(2 * n))
    return operator.get_container()


def test_screened_one_body_modes_keep_their_outer_register_slot() -> None:
    """A vanishing eigenvalue stays a generator carrying no amplitude."""
    n = 3
    basis, _ = np.linalg.qr(np.random.default_rng(1).standard_normal((n, n)))
    h1 = basis @ np.diag([1.5, -0.7, 0.3]) @ basis.T
    h1 = 0.5 * (h1 + h1.T)

    # h1' is h1 plus a two-body correction, so subtracting an eigenvalue of h1' drives that
    # mode to zero without disturbing the others.
    spectrum = np.linalg.eigvalsh(np.asarray(_factorized_with_one_body(h1).get_h1_prime(), dtype=float))
    container = _one_body_block(h1 - spectrum[np.argmin(np.abs(spectrum))] * np.eye(n))

    amplitudes = np.abs(container.one_body.coeffs).sum(axis=1)
    assert container.one_body.angles.shape == (n, n - 1)
    assert np.count_nonzero(amplitudes == 0.0) == 1
    assert container.metadata.num_positive_one_body_terms == 2


@pytest.mark.parametrize("threshold", [-1e-12, -0.5, float("nan")])
def test_rejects_a_threshold_that_is_not_non_negative(threshold) -> None:
    """A negative threshold would emit the same mode as both D1 and Q1.

    The two masks overlap once the threshold goes negative, so a mode in
    ``(threshold, -threshold)`` is counted twice: the generator block outgrows the ``N``
    slots the register layout reserves, and the Q1 copy takes the square root of a
    positive eigenvalue's negation.
    """
    mapper = SOSQubitMapper()
    mapper.settings().set("threshold", threshold)
    factorized = create_random_factorized_hamiltonian(num_orbitals=3, num_ranks=1, num_bases=2, num_copies=1)

    with pytest.raises(ValueError, match="non-negative"):
        mapper.run(Hamiltonian(factorized), MajoranaMapping.jordan_wigner(6))


def test_nullspace_noise_does_not_become_a_generator() -> None:
    """Eigensolver noise below the threshold carries no amplitude, so it cannot ride a random eigenvector."""
    n = 3
    # h1' is h1 plus a two-body correction, so negating the correction leaves only rounding error.
    correction = np.asarray(_factorized_with_one_body(np.zeros((n, n))).get_h1_prime(), dtype=float)
    container = _one_body_block(-correction)

    spectrum = np.linalg.eigvalsh(np.asarray(_factorized_with_one_body(-correction).get_h1_prime(), dtype=float))
    assert np.abs(spectrum).max() < 1e-12, "fixture is meant to leave h1' at rounding error"
    assert container.one_body.angles.shape == (n, n - 1)
    np.testing.assert_array_equal(np.abs(container.one_body.coeffs).sum(axis=1), np.zeros(n))


@pytest.mark.parametrize(
    ("mapping", "match"),
    [
        (MajoranaMapping.bravyi_kitaev(4), "jordan-wigner"),
        (MajoranaMapping.parity(4), "jordan-wigner"),
        (MajoranaMapping.jordan_wigner(2), "spin orbitals"),
        (MajoranaMapping.jordan_wigner(8), "spin orbitals"),
    ],
)
def test_rejects_a_mapping_it_cannot_honour(mapping, match) -> None:
    """An unsupported encoding or mode count fails rather than being silently replaced."""
    factorized = create_random_factorized_hamiltonian(num_orbitals=2, num_ranks=1, num_bases=2, num_copies=1)

    with pytest.raises(ValueError, match=match):
        SOSQubitMapper().run(Hamiltonian(factorized), mapping)


def test_maps_factorized_hamiltonian_to_sos_qubit_operator() -> None:
    """The mapper owns factorized conversion and returns the unified wrapper."""
    factorized = create_random_factorized_hamiltonian(num_orbitals=2, num_ranks=1, num_bases=2, num_copies=1)
    expected_normalization = factorized.get_lambda()

    result = SOSQubitMapper().run(Hamiltonian(factorized), MajoranaMapping.jordan_wigner(4))

    assert isinstance(result, QubitOperator)
    container = result.get_container()
    assert isinstance(container, SOSContainer)
    assert result.get_container_type() == "sos"
    meta = container.metadata
    assert container.one_body.angles.shape[1] == meta.num_spatial_orbitals - 1
    assert container.two_body.coeffs.shape == (meta.num_ranks * meta.num_copies, meta.num_bases + 1)
    # The block-encoding normalization is derived by the builder from the container generators.
    walk = SOSSABuilder().run(result).get_container()
    assert walk.normalization == pytest.approx(expected_normalization)


def test_sos_qubit_mapper_is_reachable_through_the_registry() -> None:
    """``create`` is the supported entry point, so the mapper has to be registered under it."""
    assert isinstance(create("qubit_mapper", "sum_of_squares"), SOSQubitMapper)
