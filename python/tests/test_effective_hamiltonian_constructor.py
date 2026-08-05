"""Tests for the effective-Hamiltonian constructor interface."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest

from qdk_chemistry.algorithms import EffectiveHamiltonianConstructor, registry
from qdk_chemistry.data.symmetry import spin_index_set

from .test_helpers import create_test_wavefunction


class _TestEffectiveHamiltonianConstructor(EffectiveHamiltonianConstructor):
    def __init__(self, expected_reference, expected_indices):
        super().__init__()
        self.expected_reference = expected_reference
        self.expected_indices = expected_indices

    def name(self):
        return "test"

    def _run_impl(self, reference, hamiltonian, p_space_indices):
        assert reference is self.expected_reference
        assert p_space_indices is self.expected_indices
        return hamiltonian


@pytest.mark.parametrize(
    "p_space_indices",
    [
        pytest.param(spin_index_set(4, [1, 2], [1, 2]), id="restricted"),
        pytest.param(spin_index_set(4, [0, 2], [1, 3], equivalent=False), id="unrestricted"),
    ],
)
def test_run_forwards_p_space_indices(p_space_indices):
    """The fixed run signature forwards restricted and unrestricted P-spaces."""
    reference = create_test_wavefunction()
    constructor = _TestEffectiveHamiltonianConstructor(reference, p_space_indices)

    assert constructor.run(reference, None, p_space_indices) is None


def test_custom_constructor_registry_round_trip():
    """A custom implementation can be registered and created by type."""
    reference = create_test_wavefunction()
    p_space_indices = spin_index_set(4, [1, 2], [1, 2])

    registry.register(lambda: _TestEffectiveHamiltonianConstructor(reference, p_space_indices))
    try:
        constructor = registry.create("effective_hamiltonian_constructor", "test")

        assert constructor.type_name() == "effective_hamiltonian_constructor"
        assert constructor.run(reference, None, p_space_indices) is None
    finally:
        registry.unregister("effective_hamiltonian_constructor", "test")
