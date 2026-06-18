"""Tests for the SOSSA controlled circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms.controlled_circuit_mapper import (
    InnerPrepareMapper,
    OuterPrepareMapper,
    SelectMapper,
    SOSSAMapper,
)
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.sossa import SOSSABuilder
from qdk_chemistry.data import Circuit, FactorizedHamiltonianContainer
from qdk_chemistry.data.controlled_unitary import ControlledUnitary
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation

from .test_helpers import create_test_orbitals

# ═══════════════════════════════════════════════════════════════════════════════
# Test helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _make_random_factorized_hamiltonian(
    num_orbitals: int = 2,
    num_ranks: int = 2,
    num_bases: int = 1,
    num_copies: int = 1,
    *,
    seed: int = 42,
):
    """Create a random FactorizedHamiltonianContainer for testing."""
    rng = np.random.default_rng(seed)
    n, r, b, c = num_orbitals, num_ranks, num_bases, num_copies

    h1 = rng.standard_normal((n, n))
    h1 = (h1 + h1.T) / 2

    u_matrices = np.zeros(r * b * n)
    for ri in range(r):
        for bi in range(b):
            v = rng.standard_normal(n)
            v /= np.linalg.norm(v)
            u_matrices[ri * b * n + bi * n : ri * b * n + (bi + 1) * n] = v

    w_matrices = rng.standard_normal(r * b * c)
    wb_matrix = rng.standard_normal((r, c))

    orbitals = create_test_orbitals(n)
    inactive_fock = np.zeros((n, n))

    return FactorizedHamiltonianContainer(
        h1, u_matrices, w_matrices, wb_matrix,
        r, b, c,
        orbitals, 0.0, inactive_fock,
    )


def _build_controlled_unitary(
    num_orbitals: int = 2,
    num_ranks: int = 2,
    num_bases: int = 1,
    num_copies: int = 1,
    *,
    seed: int = 42,
    quantum_walk: bool = True,
):
    """Helper: build ControlledUnitary with SOSSAContainer from random factorized data."""
    fh = _make_random_factorized_hamiltonian(
        num_orbitals=num_orbitals,
        num_ranks=num_ranks,
        num_bases=num_bases,
        num_copies=num_copies,
        seed=seed,
    )
    builder = SOSSABuilder(quantum_walk=quantum_walk)
    unitary_rep = builder.run(fh)
    return ControlledUnitary(unitary=unitary_rep, control_indices=[0])


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-operation mapper tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOuterPrepareMapper:
    """Tests for the OuterPrepareMapper dataclass."""

    def test_default_algorithm(self):
        """Test default algorithm is alias_sampling."""
        mapper = OuterPrepareMapper()
        assert mapper.algorithm == "alias_sampling"
        assert mapper.coefficient_bit_precision == 10

    @pytest.mark.parametrize("algorithm", ["alias_sampling", "dense_pure", "qrom"])
    def test_valid_algorithms(self, algorithm):
        """Test all valid algorithms are accepted."""
        mapper = OuterPrepareMapper(algorithm=algorithm)
        assert mapper.algorithm == algorithm

    def test_invalid_algorithm_raises(self):
        """Test invalid algorithm name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown outer prepare algorithm"):
            OuterPrepareMapper(algorithm="bogus")

    def test_needs_alias_reflection(self):
        """Test needs_alias_reflection is True only for alias_sampling."""
        assert OuterPrepareMapper(algorithm="alias_sampling").needs_alias_reflection is True
        assert OuterPrepareMapper(algorithm="dense_pure").needs_alias_reflection is False
        assert OuterPrepareMapper(algorithm="qrom").needs_alias_reflection is False


class TestInnerPrepareMapper:
    """Tests for the InnerPrepareMapper dataclass."""

    def test_default_algorithm(self):
        """Test default algorithm is controlled_alias_sampling."""
        mapper = InnerPrepareMapper()
        assert mapper.algorithm == "controlled_alias_sampling"
        assert mapper.coefficient_bit_precision == 10

    @pytest.mark.parametrize("algorithm", ["controlled_alias_sampling", "direct"])
    def test_valid_algorithms(self, algorithm):
        """Test all valid algorithms are accepted."""
        mapper = InnerPrepareMapper(algorithm=algorithm)
        assert mapper.algorithm == algorithm

    def test_invalid_algorithm_raises(self):
        """Test invalid algorithm name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown inner prepare algorithm"):
            InnerPrepareMapper(algorithm="bogus")

    def test_needs_alias_reflection(self):
        """Test needs_alias_reflection is True only for controlled_alias_sampling."""
        assert InnerPrepareMapper(algorithm="controlled_alias_sampling").needs_alias_reflection is True
        assert InnerPrepareMapper(algorithm="direct").needs_alias_reflection is False


class TestSelectMapper:
    """Tests for the SelectMapper dataclass."""

    def test_default_algorithm(self):
        """Test default algorithm is qrom_phase_gradient."""
        mapper = SelectMapper()
        assert mapper.multiplexed_rotation == "qrom_phase_gradient"
        assert mapper.rotation_bit_precision == 10

    @pytest.mark.parametrize("algorithm", ["qrom_phase_gradient", "direct"])
    def test_valid_algorithms(self, algorithm):
        """Test all valid algorithms are accepted."""
        mapper = SelectMapper(multiplexed_rotation=algorithm)
        assert mapper.multiplexed_rotation == algorithm

    def test_invalid_algorithm_raises(self):
        """Test invalid algorithm name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown multiplexed rotation"):
            SelectMapper(multiplexed_rotation="bogus")

    def test_needs_phase_gradient_register(self):
        """Test needs_phase_gradient_register is True only for qrom_phase_gradient."""
        assert SelectMapper(multiplexed_rotation="qrom_phase_gradient").needs_phase_gradient_register is True
        assert SelectMapper(multiplexed_rotation="direct").needs_phase_gradient_register is False


# ═══════════════════════════════════════════════════════════════════════════════
# Main SOSSAMapper tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSOSSAMapper:
    """Tests for the SOSSA controlled circuit mapper."""

    def test_name_and_type(self):
        """Test that name and type_name return correct values."""
        mapper = SOSSAMapper()
        assert mapper.name() == "sossa"
        assert mapper.type_name() == "controlled_circuit_mapper"

    def test_default_sub_mappers(self):
        """Test default sub-mapper types are created."""
        mapper = SOSSAMapper()
        assert isinstance(mapper.outer_prepare_mapper, OuterPrepareMapper)
        assert isinstance(mapper.inner_prepare_mapper, InnerPrepareMapper)
        assert isinstance(mapper.select_mapper, SelectMapper)
        assert mapper.outer_prepare_mapper.algorithm == "alias_sampling"
        assert mapper.inner_prepare_mapper.algorithm == "controlled_alias_sampling"
        assert mapper.select_mapper.multiplexed_rotation == "qrom_phase_gradient"

    def test_custom_sub_mappers(self):
        """Test custom sub-mappers are accepted."""
        outer = OuterPrepareMapper(algorithm="dense_pure")
        inner = InnerPrepareMapper(algorithm="direct")
        select = SelectMapper(multiplexed_rotation="direct")
        mapper = SOSSAMapper(
            outer_prepare_mapper=outer,
            inner_prepare_mapper=inner,
            select_mapper=select,
        )
        assert mapper.outer_prepare_mapper.algorithm == "dense_pure"
        assert mapper.inner_prepare_mapper.algorithm == "direct"
        assert mapper.select_mapper.multiplexed_rotation == "direct"

    def test_basic_mapping_produces_circuit_with_factory(self):
        """Test that mapping produces a Circuit with both qsharp_op and qsharp_factory."""
        controlled_unitary = _build_controlled_unitary()
        mapper = SOSSAMapper()
        circuit = mapper.run(controlled_unitary)

        assert isinstance(circuit, Circuit)
        assert circuit._qsharp_op is not None
        assert circuit._qsharp_factory is not None

    def test_rejects_non_sossa_container(self):
        """Verify SOSSAMapper raises ValueError for non-SOSSAContainer containers."""

        class MockContainer:
            """Mock container that is not a SOSSAContainer."""

            @property
            def type(self):
                return "mock"

        unitary_rep = UnitaryRepresentation(container=MockContainer())
        controlled_unitary = ControlledUnitary(unitary=unitary_rep, control_indices=[0])

        mapper = SOSSAMapper()
        with pytest.raises(ValueError, match="not supported"):
            mapper.run(controlled_unitary)

    def test_rejects_multiple_control_qubits(self):
        """Verify SOSSAMapper raises ValueError for multiple control qubits."""
        controlled_unitary = _build_controlled_unitary()
        controlled_unitary = ControlledUnitary(
            unitary=controlled_unitary.unitary, control_indices=[0, 1]
        )

        mapper = SOSSAMapper()
        with pytest.raises(ValueError, match="single control qubit"):
            mapper.run(controlled_unitary)

    @pytest.mark.parametrize(
        ("outer_alg", "inner_alg", "select_alg"),
        [
            ("alias_sampling", "controlled_alias_sampling", "qrom_phase_gradient"),
            ("dense_pure", "direct", "direct"),
            ("qrom", "controlled_alias_sampling", "direct"),
            ("alias_sampling", "direct", "qrom_phase_gradient"),
            ("dense_pure", "controlled_alias_sampling", "qrom_phase_gradient"),
        ],
        ids=[
            "default_all",
            "dense_direct_direct",
            "qrom_alias_direct",
            "alias_direct_phase",
            "dense_alias_phase",
        ],
    )
    def test_all_algorithm_combinations_produce_circuit(
        self, outer_alg, inner_alg, select_alg
    ):
        """Test that all valid algorithm combinations produce a Circuit."""
        controlled_unitary = _build_controlled_unitary()
        mapper = SOSSAMapper(
            outer_prepare_mapper=OuterPrepareMapper(algorithm=outer_alg),
            inner_prepare_mapper=InnerPrepareMapper(algorithm=inner_alg),
            select_mapper=SelectMapper(multiplexed_rotation=select_alg),
        )
        circuit = mapper.run(controlled_unitary)

        assert isinstance(circuit, Circuit)
        assert circuit._qsharp_op is not None
        assert circuit._qsharp_factory is not None

    @pytest.mark.parametrize(
        ("num_orbitals", "num_ranks", "num_bases", "num_copies"),
        [
            (2, 1, 1, 1),
            (2, 2, 1, 1),
            (3, 2, 2, 1),
        ],
        ids=["N2R1B1C1", "N2R2B1C1", "N3R2B2C1"],
    )
    def test_mapping_parametrized_dimensions(
        self, num_orbitals, num_ranks, num_bases, num_copies
    ):
        """Test mapping for various (N, R, B, C) configurations."""
        controlled_unitary = _build_controlled_unitary(
            num_orbitals=num_orbitals,
            num_ranks=num_ranks,
            num_bases=num_bases,
            num_copies=num_copies,
        )
        mapper = SOSSAMapper()
        circuit = mapper.run(controlled_unitary)

        assert isinstance(circuit, Circuit)
        assert circuit._qsharp_op is not None

    def test_circuit_has_qsharp_factory_program_and_parameter(self):
        """Test that the circuit's qsharp_factory has correct structure."""
        controlled_unitary = _build_controlled_unitary()
        mapper = SOSSAMapper()
        circuit = mapper.run(controlled_unitary)

        factory = circuit._qsharp_factory
        assert factory is not None
        assert factory.program is not None
        assert isinstance(factory.parameter, dict)

        # Verify expected keys in walk_params
        expected_keys = {
            "outerPrepareOp", "innerPrepareOp", "selectOp",
            "numSystemQubits", "numOuterQubits", "numInnerQubits",
            "power", "outerReflectionIncludesKeep", "innerReflectionIncludesKeep",
            "needsPhaseGradient", "phaseGradientBits",
        }
        assert expected_keys == set(factory.parameter.keys())

    def test_walk_params_reflect_mapper_settings(self):
        """Test that walk_params correctly reflect sub-mapper settings."""
        outer = OuterPrepareMapper(algorithm="dense_pure")
        inner = InnerPrepareMapper(algorithm="direct")
        select = SelectMapper(multiplexed_rotation="direct", rotation_bit_precision=12)

        controlled_unitary = _build_controlled_unitary()
        mapper = SOSSAMapper(
            outer_prepare_mapper=outer,
            inner_prepare_mapper=inner,
            select_mapper=select,
        )
        circuit = mapper.run(controlled_unitary)

        params = circuit._qsharp_factory.parameter
        assert params["outerReflectionIncludesKeep"] is False  # dense_pure
        assert params["innerReflectionIncludesKeep"] is False  # direct
        assert params["needsPhaseGradient"] is False  # direct rotation
        assert params["phaseGradientBits"] == 12

    def test_walk_params_alias_sampling_flags(self):
        """Test that alias sampling algorithms set reflection flags correctly."""
        controlled_unitary = _build_controlled_unitary()
        mapper = SOSSAMapper(
            outer_prepare_mapper=OuterPrepareMapper(algorithm="alias_sampling"),
            inner_prepare_mapper=InnerPrepareMapper(algorithm="controlled_alias_sampling"),
            select_mapper=SelectMapper(multiplexed_rotation="qrom_phase_gradient"),
        )
        circuit = mapper.run(controlled_unitary)

        params = circuit._qsharp_factory.parameter
        assert params["outerReflectionIncludesKeep"] is True
        assert params["innerReflectionIncludesKeep"] is True
        assert params["needsPhaseGradient"] is True

    def test_walk_params_system_qubit_count(self):
        """Test that numSystemQubits = 2 * num_orbitals."""
        n = 3
        controlled_unitary = _build_controlled_unitary(num_orbitals=n)
        mapper = SOSSAMapper()
        circuit = mapper.run(controlled_unitary)

        params = circuit._qsharp_factory.parameter
        assert params["numSystemQubits"] == 2 * n

    def test_walk_params_power(self):
        """Test that power passes through to walk_params."""
        fh = _make_random_factorized_hamiltonian()
        builder = SOSSABuilder(power=5)
        unitary_rep = builder.run(fh)
        controlled_unitary = ControlledUnitary(unitary=unitary_rep, control_indices=[0])

        mapper = SOSSAMapper()
        circuit = mapper.run(controlled_unitary)

        params = circuit._qsharp_factory.parameter
        assert params["power"] == 5
