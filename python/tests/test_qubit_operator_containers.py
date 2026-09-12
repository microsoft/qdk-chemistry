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
from qdk_chemistry.algorithms.qubit_mapper.sum_of_squares import SumOfSquaresQubitMapper
from qdk_chemistry.data import FactorizedHamiltonianContainer, Hamiltonian, MajoranaMapping, QubitOperator
from qdk_chemistry.data.qubit_operator.containers.base import QubitOperatorContainer
from qdk_chemistry.data.qubit_operator.containers.pauli_decomposition import PauliDecompositionContainer
from qdk_chemistry.data.qubit_operator.containers.sum_of_squares import (
    RotatedPaulis,
    SumOfSquaresContainer,
    SumOfSquaresMetadata,
)

from .test_helpers import create_random_factorized_hamiltonian, create_test_orbitals


def _sum_of_squares_container() -> SumOfSquaresContainer:
    """Build a small SOS container whose one-body block carries D1/Q1 ``+/-i`` phases."""
    return SumOfSquaresContainer(
        one_body=RotatedPaulis(np.array([[0.1], [0.2]]), np.array([[0.2, 0.2j], [0.3, -0.3j]]), ("X", "Y")),
        two_body=RotatedPaulis(np.array([[0.3]]), np.array([[0.3, 0.7]]), ("Z",)),
        encoding="jordan-wigner",
        fermion_mode_order="blocked",
        metadata=SumOfSquaresMetadata(
            num_spatial_orbitals=2,
            num_ranks=1,
            num_bases=1,
            num_copies=1,
            num_positive_one_body_terms=1,
            energy_shift=-1.5,
        ),
    )


def _assert_sum_of_squares_containers_match(restored, container) -> None:
    """Compare every field a serializer could silently drop or reorder."""
    np.testing.assert_allclose(restored.one_body.angles, container.one_body.angles)
    np.testing.assert_allclose(restored.one_body.coeffs, container.one_body.coeffs)
    assert restored.one_body.paulis == container.one_body.paulis
    np.testing.assert_allclose(restored.two_body.angles, container.two_body.angles)
    np.testing.assert_allclose(restored.two_body.coeffs, container.two_body.coeffs)
    assert restored.two_body.paulis == container.two_body.paulis
    assert restored.metadata == container.metadata
    assert restored.encoding == container.encoding
    assert restored.fermion_mode_order == container.fermion_mode_order


class TestPauliDecompositionContainer:
    """Serialization and the pre-container public surface of the Pauli decomposition."""

    def test_pre_container_surface_still_works(self, tmp_path) -> None:
        """Everything that shipped before container dispatch keeps working unchanged.

        Pre-container releases exposed ``QubitOperator(pauli_strings, coefficients)`` and
        read Pauli attributes straight off the operator, and they wrote documents at
        version ``0.1.0`` with no ``container_type`` and no ``shape``/``dtype`` on the
        coefficient dict. The missing key must default to ``pauli_decomposition``, and
        hand-authored fixtures (not stripped round-trips of current output) are what pin
        that legacy schema.
        """
        operator = QubitOperator(["XI", "ZZ"], np.array([0.5, -0.25]))
        container = operator.get_container()

        assert isinstance(container, QubitOperatorContainer)
        assert QubitOperator(container).get_container() is container
        assert operator.get_container_type() == "pauli_decomposition"
        assert operator.num_qubits == 2
        assert operator.pauli_strings is container.pauli_strings
        assert operator.coefficients is container.coefficients
        assert operator.schatten_norm == container.schatten_norm
        np.testing.assert_allclose(operator.to_matrix(), container.to_matrix())
        assert operator.is_hermitian()

        # equiv and arithmetic are defined on the wrapper rather than forwarded, so they
        # have to unwrap the operand and rewrap the result.
        assert operator.equiv(QubitOperator(PauliDecompositionContainer(["ZZ", "XI"], np.array([-0.25, 0.5]))))
        scaled = 2 * operator
        added = operator + operator
        assert isinstance(scaled, QubitOperator)
        assert isinstance(added, QubitOperator)
        np.testing.assert_allclose(scaled.coefficients, np.array([1.0, -0.5]))
        assert added.pauli_strings == ["XI", "ZZ", "XI", "ZZ"]

        legacy_json = {
            "pauli_strings": ["XI", "ZZ"],
            "coefficients": {"real": [0.5, -0.25], "imag": [0.0, 0.0]},
            "encoding": "jordan-wigner",
            "version": "0.1.0",
        }
        restored = QubitOperator.from_json(legacy_json)
        assert restored.get_container_type() == "pauli_decomposition"
        assert restored.pauli_strings == ["XI", "ZZ"]
        np.testing.assert_allclose(restored.coefficients, np.array([0.5, -0.25]))
        assert restored.encoding == "jordan-wigner"

        path = tmp_path / "legacy.h5"
        with h5py.File(path, "w") as handle:
            group = handle.create_group("operator")
            group.attrs["version"] = "0.1.0"
            group.attrs["encoding"] = "jordan-wigner"
            group.create_dataset("pauli_strings", data=np.array(["XI", "ZZ"], dtype="S"))
            group.create_dataset("coefficients", data=np.array([0.5, -0.25], dtype=complex))
        with h5py.File(path, "r") as handle:
            restored = QubitOperator.from_hdf5(handle["operator"])
        assert restored.get_container_type() == "pauli_decomposition"
        assert restored.pauli_strings == ["XI", "ZZ"]
        np.testing.assert_allclose(restored.coefficients, np.array([0.5, -0.25]))
        assert restored.encoding == "jordan-wigner"

    def test_json_roundtrip(self) -> None:
        """Complex coefficients keep their shape and dtype across the shared array codec."""
        coefficients = np.array([0.5 + 0.25j, -0.25j], dtype=np.complex64)
        container = PauliDecompositionContainer(["XI", "ZZ"], coefficients, "jordan-wigner", "blocked")
        operator = QubitOperator(container)

        json_data = operator.to_json()
        assert json_data["coefficients"] == {
            "real": [0.5, 0.0],
            "imag": [0.25, -0.25],
            "shape": [2],
            "dtype": "complex64",
        }

        restored = QubitOperator.from_json(json_data)
        assert restored.get_container_type() == "pauli_decomposition"
        assert restored.pauli_strings == container.pauli_strings
        np.testing.assert_array_equal(restored.coefficients, coefficients)
        assert restored.coefficients.dtype == coefficients.dtype
        assert restored.encoding == container.encoding
        assert restored.fermion_mode_order == container.fermion_mode_order
        assert restored.content_hash() == operator.content_hash()

        # ``shape`` postdates the first release of the codec, so it stays optional.
        del json_data["coefficients"]["shape"]
        np.testing.assert_array_equal(QubitOperator.from_json(json_data).coefficients, coefficients)

    def test_hdf5_roundtrip(self, tmp_path) -> None:
        """The container reloads from HDF5 through the qubit operator's container dispatch."""
        coefficients = np.array([0.5 + 0.25j, -0.25j], dtype=np.complex64)
        container = PauliDecompositionContainer(["XI", "ZZ"], coefficients, "jordan-wigner", "blocked")
        operator = QubitOperator(container)
        path = tmp_path / "pauli.h5"

        with h5py.File(path, "w") as handle:
            operator.to_hdf5(handle.create_group("operator"))

        with h5py.File(path, "r") as handle:
            assert handle["operator"].attrs["container_type"] == "pauli_decomposition"
            restored = QubitOperator.from_hdf5(handle["operator"])

        assert isinstance(restored.get_container(), PauliDecompositionContainer)
        assert restored.pauli_strings == container.pauli_strings
        np.testing.assert_array_equal(restored.coefficients, coefficients)
        assert restored.coefficients.dtype == coefficients.dtype
        assert restored.encoding == container.encoding
        assert restored.fermion_mode_order == container.fermion_mode_order
        assert restored.content_hash() == operator.content_hash()


class TestSumOfSquaresContainer:
    """Serialization and the validation the SOS blocks and metadata enforce."""

    def test_json_roundtrip(self) -> None:
        """Complex LCU coefficients and Givens angles survive a JSON round-trip.

        The SOS generators carry the D1/Q1 ``+/-i`` sign in the imaginary part, so a
        serializer that silently drops it would still produce a well-formed container
        while flipping particle generators into hole generators.
        """
        container = _sum_of_squares_container()

        json_data = QubitOperator(container).to_json()
        assert json_data["one_body"]["coeffs"]["shape"] == [2, 2]
        assert json_data["one_body"]["coeffs"]["dtype"] == "complex128"
        assert json_data["one_body"]["paulis"] == ["X", "Y"]
        del json_data["one_body"]["coeffs"]["dtype"]
        del json_data["two_body"]["coeffs"]["dtype"]
        restored = QubitOperator.from_json(json_data).get_container()

        _assert_sum_of_squares_containers_match(restored, container)
        assert restored.metadata.energy_shift == pytest.approx(-1.5)

    def test_hdf5_roundtrip(self, tmp_path) -> None:
        """The container reloads from HDF5 through the qubit operator's container dispatch."""
        container = _sum_of_squares_container()
        path = tmp_path / "sos.h5"

        with h5py.File(path, "w") as handle:
            QubitOperator(container).to_hdf5(handle.create_group("operator"))

        with h5py.File(path, "r") as handle:
            assert handle["operator"].attrs["container_type"] == "sum_of_squares"
            operator = QubitOperator.from_hdf5(handle["operator"])

        restored = operator.get_container()
        assert isinstance(restored, SumOfSquaresContainer)
        _assert_sum_of_squares_containers_match(restored, container)
        assert operator.content_hash() == QubitOperator(container).content_hash()

    def test_rejects_inputs_that_are_not_internally_consistent(self) -> None:
        """A block owns the shape facts every block shares; the container owns metadata agreement."""
        with pytest.raises(TypeError, match="QubitOperator requires a QubitOperatorContainer"):
            QubitOperator(["X"])  # type: ignore[arg-type]

        # Negative R/B/C survive the ``.size``-guarded shape checks, so metadata rejects
        # them itself.
        for num_ranks, num_bases, num_copies in ((-1, 1, 1), (1, -1, 1), (1, 1, -1)):
            with pytest.raises(ValueError, match="must not be negative"):
                SumOfSquaresMetadata(
                    num_spatial_orbitals=2,
                    num_ranks=num_ranks,
                    num_bases=num_bases,
                    num_copies=num_copies,
                    num_positive_one_body_terms=0,
                    energy_shift=0.0,
                )

        with pytest.raises(ValueError, match="2-D"):
            RotatedPaulis(np.array([0.1, 0.2]), np.array([0.3, 0.4]), ("X",))

        with pytest.raises(ValueError, match="at least one Pauli word"):
            RotatedPaulis(np.array([[0.1]]), np.array([[0.3]]), ())

        metadata = SumOfSquaresMetadata(
            num_spatial_orbitals=3,
            num_ranks=1,
            num_bases=1,
            num_copies=1,
            num_positive_one_body_terms=1,
            energy_shift=0.0,
        )
        two_body = RotatedPaulis(np.array([[0.3, 0.4]]), np.array([[0.3, 0.7]]), ("Z",))

        with pytest.raises(ValueError, match="matching generator counts"):
            SumOfSquaresContainer(
                RotatedPaulis(np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([[0.2, 0.2j]]), ("X", "Y")),
                two_body,
                None,
                None,
                metadata,
            )

        with pytest.raises(ValueError, match="one column per Pauli label"):
            SumOfSquaresContainer(
                RotatedPaulis(np.array([[0.1, 0.2]]), np.array([[0.2, 0.2j, 0.1]]), ("X", "Y")),
                two_body,
                None,
                None,
                metadata,
            )

        with pytest.raises(ValueError, match="num_spatial_orbitals - 1 columns"):
            SumOfSquaresContainer(
                RotatedPaulis(np.array([[0.1]]), np.array([[0.2, 0.2j]]), ("X", "Y")),
                two_body,
                None,
                None,
                metadata,
            )

        # Positive control: two-body angles are per ``(rank, basis)`` and coefficients per
        # ``(rank, copy)``, so whenever ``num_bases != num_copies`` the two legitimately
        # disagree on row count and row alignment cannot be a shared block invariant.
        container = SumOfSquaresContainer(
            RotatedPaulis(np.array([[0.1, 0.2]]), np.array([[0.2, 0.2j]]), ("X", "Y")),
            RotatedPaulis(np.zeros((2, 2)), np.zeros((1, 3)), ("Z",)),
            None,
            None,
            SumOfSquaresMetadata(
                num_spatial_orbitals=3,
                num_ranks=1,
                num_bases=2,
                num_copies=1,
                num_positive_one_body_terms=1,
                energy_shift=0.0,
            ),
        )
        assert container.two_body.angles.shape != container.two_body.coeffs.shape


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
    operator = SumOfSquaresQubitMapper().run(
        Hamiltonian(_factorized_with_one_body(h1)), MajoranaMapping.jordan_wigner(2 * n)
    )
    return operator.get_container()


class TestSumOfSquaresQubitMapper:
    """The mapper that produces :class:`SumOfSquaresContainer` from a factorized Hamiltonian."""

    def test_maps_factorized_hamiltonian_to_sos_qubit_operator(self) -> None:
        """The mapper owns factorized conversion and returns the unified wrapper."""
        assert isinstance(create("qubit_mapper", "sum_of_squares"), SumOfSquaresQubitMapper)

        factorized = create_random_factorized_hamiltonian(num_orbitals=2, num_ranks=1, num_bases=2, num_copies=1)
        expected_normalization = factorized.get_lambda()

        result = SumOfSquaresQubitMapper().run(Hamiltonian(factorized), MajoranaMapping.jordan_wigner(4))

        assert isinstance(result, QubitOperator)
        container = result.get_container()
        assert isinstance(container, SumOfSquaresContainer)
        assert result.get_container_type() == "sum_of_squares"
        meta = container.metadata
        assert container.one_body.angles.shape[1] == meta.num_spatial_orbitals - 1
        assert container.two_body.coeffs.shape == (meta.num_ranks * meta.num_copies, meta.num_bases + 1)
        # The block-encoding normalization is derived by the builder from the container generators.
        walk = SOSSABuilder().run(result).get_container()
        assert walk.normalization == pytest.approx(expected_normalization)

    def test_screened_one_body_modes_keep_their_outer_register_slot(self) -> None:
        """A vanishing eigenvalue stays a generator carrying no amplitude.

        The register layout reserves ``N`` one-body slots unconditionally, so a screened
        mode must keep its slot rather than shift its neighbours, and eigensolver noise
        below the threshold must not ride an arbitrary eigenvector into a generator.
        """
        n = 3
        basis, _ = np.linalg.qr(np.random.default_rng(1).standard_normal((n, n)))
        h1 = basis @ np.diag([1.5, -0.7, 0.3]) @ basis.T
        h1 = 0.5 * (h1 + h1.T)

        # h1' is h1 plus a two-body correction, so subtracting an eigenvalue of h1' drives
        # that mode to zero without disturbing the others.
        spectrum = np.linalg.eigvalsh(np.asarray(_factorized_with_one_body(h1).get_h1_prime(), dtype=float))
        container = _one_body_block(h1 - spectrum[np.argmin(np.abs(spectrum))] * np.eye(n))

        amplitudes = np.abs(container.one_body.coeffs).sum(axis=1)
        assert container.one_body.angles.shape == (n, n - 1)
        assert np.count_nonzero(amplitudes == 0.0) == 1
        assert container.metadata.num_positive_one_body_terms == 2

        # Negating the correction instead leaves h1' at nothing but rounding error.
        correction = np.asarray(_factorized_with_one_body(np.zeros((n, n))).get_h1_prime(), dtype=float)
        noise_spectrum = np.linalg.eigvalsh(
            np.asarray(_factorized_with_one_body(-correction).get_h1_prime(), dtype=float)
        )
        assert np.abs(noise_spectrum).max() < 1e-12, "fixture is meant to leave h1' at rounding error"

        noise_container = _one_body_block(-correction)
        assert noise_container.one_body.angles.shape == (n, n - 1)
        np.testing.assert_array_equal(np.abs(noise_container.one_body.coeffs).sum(axis=1), np.zeros(n))

    @pytest.mark.parametrize("threshold", [-1e-12, -0.5, float("nan")])
    def test_rejects_a_threshold_that_is_not_non_negative(self, threshold) -> None:
        """A negative threshold would emit the same mode as both D1 and Q1.

        The two masks overlap once the threshold goes negative, so a mode in
        ``(threshold, -threshold)`` is counted twice: the generator block outgrows the ``N``
        slots the register layout reserves, and the Q1 copy takes the square root of a
        positive eigenvalue's negation.
        """
        mapper = SumOfSquaresQubitMapper()
        mapper.settings().set("threshold", threshold)
        factorized = create_random_factorized_hamiltonian(num_orbitals=3, num_ranks=1, num_bases=2, num_copies=1)

        with pytest.raises(ValueError, match="non-negative"):
            mapper.run(Hamiltonian(factorized), MajoranaMapping.jordan_wigner(6))

    @pytest.mark.parametrize(
        ("mapping", "match"),
        [
            (MajoranaMapping.bravyi_kitaev(4), "jordan-wigner"),
            (MajoranaMapping.parity(4), "jordan-wigner"),
            (MajoranaMapping.jordan_wigner(2), "spin orbitals"),
            (MajoranaMapping.jordan_wigner(8), "spin orbitals"),
        ],
    )
    def test_rejects_a_mapping_it_cannot_honour(self, mapping, match) -> None:
        """An unsupported encoding or mode count fails rather than being silently replaced."""
        factorized = create_random_factorized_hamiltonian(num_orbitals=2, num_ranks=1, num_bases=2, num_copies=1)

        with pytest.raises(ValueError, match=match):
            SumOfSquaresQubitMapper().run(Hamiltonian(factorized), mapping)
