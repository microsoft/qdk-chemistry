"""Tests for the rotated Jordan-Wigner qubit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Hamiltonian, MajoranaMapping, QubitOperator, SOSContainer

from .test_helpers import create_random_factorized_hamiltonian


def test_maps_factorized_hamiltonian_to_sos_qubit_operator() -> None:
    """The mapper owns factorized conversion and returns the unified wrapper."""
    factorized = create_random_factorized_hamiltonian(num_orbitals=2, num_ranks=1, num_bases=2, num_copies=1)
    expected_normalization = factorized.get_lambda()

    result = create("qubit_mapper").run(Hamiltonian(factorized), MajoranaMapping.jordan_wigner(4))

    assert isinstance(result, QubitOperator)
    assert isinstance(result.get_container(), SOSContainer)
    assert result.get_container_type() == "sos"
    assert result.get_container().normalization == pytest.approx(expected_normalization)
    assert all(isinstance(generator.operator, QubitOperator) for generator in result.get_container().generators)


@pytest.mark.parametrize("tolerance", [-1.0, float("inf"), float("nan")])
def test_rejects_invalid_tolerance(tolerance: float) -> None:
    """Tolerance must be finite and non-negative."""
    factorized = create_random_factorized_hamiltonian(num_orbitals=2)
    mapper = create("qubit_mapper", factorized_tolerance=tolerance)

    with pytest.raises(ValueError, match="tolerance must be finite and non-negative"):
        mapper.run(Hamiltonian(factorized), MajoranaMapping.jordan_wigner(4))


def test_rotated_mapper_is_not_a_registered_variant() -> None:
    """Factorized conversion is an option of QubitMapper, not a separate algorithm."""
    with pytest.raises(KeyError, match="rotated-jordan-wigner"):
        create("qubit_mapper", "rotated-jordan-wigner")
