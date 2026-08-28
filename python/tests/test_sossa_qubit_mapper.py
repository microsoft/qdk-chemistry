"""Tests for the SOSSA qubit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.sossa import SOSSABuilder
from qdk_chemistry.algorithms.qubit_mapper.sos import SOSQubitMapper
from qdk_chemistry.data import Hamiltonian, MajoranaMapping, QubitOperator
from qdk_chemistry.data.qubit_operator.containers.sossa import SOSContainer

from .test_helpers import create_random_factorized_hamiltonian


def test_maps_factorized_hamiltonian_to_sos_qubit_operator() -> None:
    """The mapper owns factorized conversion and returns the unified wrapper."""
    factorized = create_random_factorized_hamiltonian(num_orbitals=2, num_ranks=1, num_bases=2, num_copies=1)
    expected_normalization = factorized.get_lambda()

    result = SOSQubitMapper().run(Hamiltonian(factorized), MajoranaMapping.jordan_wigner(4))

    assert isinstance(result, QubitOperator)
    container = result.get_container()
    assert isinstance(container, SOSContainer)
    assert result.get_container_type() == "sossa"
    meta = container.metadata
    assert container.one_body.angles.shape[1] == meta.num_spatial_orbitals - 1
    assert container.two_body.coeffs.shape == (meta.num_ranks * meta.num_copies, meta.num_bases + 1)
    # The block-encoding normalization is derived by the builder from the container generators.
    walk = SOSSABuilder().run(result).get_container()
    assert walk.metadata.normalization == pytest.approx(expected_normalization)


def test_sossa_qubit_mapper_is_reachable_through_the_registry() -> None:
    """``create`` is the supported entry point, so the mapper has to be registered under it."""
    assert isinstance(create("qubit_mapper", "sossa"), SOSQubitMapper)
