"""Tests for algorithms registry functionality."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms import registry


class TestRegistryShowDefault:
    """Test the show_default function in the registry module."""

    def test_show_default_returns_dict_when_no_type_specified(self):
        """Test that show_default returns a dict when called without arguments."""
        defaults = registry.show_default()

        # Should return a dictionary
        assert isinstance(defaults, dict)

        # Should have entries for all known algorithm types
        expected_types = [
            "active_space_selector",
            "coupled_cluster_calculator",
            "hamiltonian_constructor",
            "orbital_localizer",
            "multi_configuration_calculator",
            "multi_configuration_scf",
            "projected_multi_configuration_calculator",
            "scf_solver",
            "stability_checker",
            "energy_estimator",
            "state_prep",
            "qubit_mapper",
        ]

        for algorithm_type in expected_types:
            assert algorithm_type in defaults, f"Expected algorithm type '{algorithm_type}' not found in defaults"
            assert isinstance(defaults[algorithm_type], str), f"Default for '{algorithm_type}' should be a string"
            assert len(defaults[algorithm_type]) > 0, f"Default for '{algorithm_type}' should not be empty"

    def test_show_default_returns_string_for_specific_type(self):
        """Test that show_default returns a string when called with a specific type."""
        # Test for active space selector
        default_active_space_selector = registry.show_default("active_space_selector")
        assert isinstance(default_active_space_selector, str)
        assert default_active_space_selector == "qdk_autocas_eos"

        # Test for hamiltonian constructor
        default_hamiltonian_constructor = registry.show_default("hamiltonian_constructor")
        assert isinstance(default_hamiltonian_constructor, str)
        assert default_hamiltonian_constructor == "qdk"

        # Test for multi configuration SCF
        default_mcscf = registry.show_default("multi_configuration_scf")
        assert isinstance(default_mcscf, str)
        assert default_mcscf == "pyscf"

        # Test for qubit mapper
        default_qubit_mapper = registry.show_default("qubit_mapper")
        assert isinstance(default_qubit_mapper, str)
        assert default_qubit_mapper == "qiskit"

        # Test for SCF solver
        default_scf = registry.show_default("scf_solver")
        assert isinstance(default_scf, str)
        assert default_scf == "qdk"

    def test_show_default_returns_empty_string_for_unknown_type(self):
        """Test that show_default returns empty string for unknown algorithm type."""
        result = registry.show_default("nonexistent_type")
        assert result == ""

    def test_show_default_consistent_with_available(self):
        """Test that default algorithms are actually available in the registry."""
        defaults = registry.show_default()
        available = registry.available()

        for algorithm_type, default_name in defaults.items():
            # Each default should be in the available list for that type
            assert algorithm_type in available, f"Algorithm type '{algorithm_type}' not in available algorithms"
            assert default_name in available[algorithm_type], (
                f"Default algorithm '{default_name}' for type '{algorithm_type}' "
                f"not found in available algorithms: {available[algorithm_type]}"
            )

    def test_default_algorithm_can_be_created(self):
        """Test that default algorithms can actually be instantiated."""
        defaults = registry.show_default()

        for algorithm_type, default_name in defaults.items():
            # Should be able to create the default algorithm
            algorithm = registry.create(algorithm_type, default_name)
            assert algorithm is not None
            assert algorithm.type_name() == algorithm_type
