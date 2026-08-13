"""Tests for the PREPARE-SELECT (non-controlled) block-encoding circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms.circuit_mapper.psp_mapper import PSPMapper
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.lcu import LCUBuilder
from qdk_chemistry.data import Circuit, QubitOperator
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation


def _build_unitary_rep(pauli_strings, coefficients, *, quantum_walk=False):
    """Helper: build UnitaryRepresentation from Pauli strings and coefficients."""
    hamiltonian = QubitOperator(pauli_strings=pauli_strings, coefficients=coefficients)
    builder = LCUBuilder(quantum_walk=quantum_walk)
    return builder.run(hamiltonian)


class TestPSPMapperQsharpSurface:
    """The mapper resolves its Q# entry points inside ``run()``, not at import.

    ``PSPMapper`` is importable, registered and exported without ever touching Q#: every
    ``QSHARP_UTILS.PrepSelPrep.*`` dereference happens inside ``run()``. Importing the package,
    enumerating the registry and reading ``__all__`` therefore all succeed even if the Q# surface
    the mapper dispatches to is absent, and the first evidence of a missing operation is an
    ``AttributeError`` raised at call time by user code.

    These tests call ``run()`` on both dispatch paths so that surface is exercised directly:

    * both paths reach ``NoOpPrepare``, ``MakePrepSelPrepOp`` and ``MakePrepSelPrepCircuit``;
    * the walk path additionally reaches ``MakeWalkOp`` and ``MakeAncillaReflectionOp``.
    """

    def test_name_and_type(self):
        """The mapper reports the identity the registry registers it under."""
        mapper = PSPMapper()

        assert mapper.name() == "prepare_select_prepare"
        assert mapper.type_name() == "circuit_mapper"

    def test_block_encoding_path_builds_a_circuit(self):
        """A plain ``LCUContainer`` maps to a circuit carrying a Q# factory."""
        unitary_rep = _build_unitary_rep(["XX", "ZZ"], np.array([0.25, 0.5]))

        circuit = PSPMapper().run(unitary_rep)

        assert isinstance(circuit, Circuit)
        assert circuit._qsharp_op is not None
        assert circuit._qsharp_factory is not None

    def test_walk_path_builds_a_circuit(self):
        """An ``LCUWalkContainer`` maps to a circuit spanning the register it declares.

        The walk path pairs the block encoding with a reflection about the ancilla register, so
        this is the only path that reaches ``MakeWalkOp`` and ``MakeAncillaReflectionOp``.
        """
        unitary_rep = _build_unitary_rep(["X", "Z"], np.array([0.5, 0.5]), quantum_walk=True)

        circuit = PSPMapper().run(unitary_rep)

        assert isinstance(circuit, Circuit)
        assert circuit.num_qubits == unitary_rep.get_num_qubits()

    def test_rejects_a_container_that_is_neither_lcu_nor_walk(self):
        """Containers the mapper cannot unwrap are rejected by name."""

        class MockContainer:
            """Mock container that is neither an LCU nor an LCU walk."""

            @property
            def type(self):
                return "mock"

        unitary_rep = UnitaryRepresentation(container=MockContainer())

        with pytest.raises(ValueError, match="LCUContainer or LCUWalkContainer"):
            PSPMapper().run(unitary_rep)
