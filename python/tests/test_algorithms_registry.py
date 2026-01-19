"""Tests for algorithms registry functionality."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest

from qdk_chemistry._core._algorithms import ScfSolverFactory
from qdk_chemistry.algorithms import ScfSolver, registry

try:
    import pyscf  # noqa: F401

    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False


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
            "dynamical_correlation_calculator",
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
        if PYSCF_AVAILABLE:
            default_mcscf = registry.show_default("multi_configuration_scf")
            assert isinstance(default_mcscf, str)
            assert default_mcscf == "pyscf"

        # Test for qubit mapper
        default_qubit_mapper = registry.show_default("qubit_mapper")
        assert isinstance(default_qubit_mapper, str)
        assert default_qubit_mapper == "qdk"

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
            if default_name == "pyscf" and not PYSCF_AVAILABLE:
                continue  # Skip check if pyscf is not available
            assert default_name in available[algorithm_type], (
                f"Default algorithm '{default_name}' for type '{algorithm_type}' "
                f"not found in available algorithms: {available[algorithm_type]}"
            )

    def test_default_algorithm_can_be_created(self):
        """Test that default algorithms can actually be instantiated."""
        defaults = registry.show_default()

        for algorithm_type, default_name in defaults.items():
            if default_name == "pyscf" and not PYSCF_AVAILABLE:
                continue  # Skip check if pyscf is not available
            # Should be able to create the default algorithm
            algorithm = registry.create(algorithm_type, default_name)
            assert algorithm is not None
            assert algorithm.type_name() == algorithm_type


class TestRegistryCreate:
    """Test the create function in the registry module."""

    def test_create_default_algorithm(self):
        """Test creating the default algorithm without specifying name."""
        # Create using None for algorithm_name
        scf = registry.create("scf_solver", None)
        assert scf is not None
        assert scf.type_name() == "scf_solver"

        # Create using empty string
        scf2 = registry.create("scf_solver", "")
        assert scf2 is not None
        assert scf2.type_name() == "scf_solver"

    def test_create_named_algorithm(self):
        """Test creating a specific algorithm by name."""
        # Create a QDK SCF solver
        scf = registry.create("scf_solver", "qdk")
        assert scf is not None
        assert scf.name() == "qdk"
        assert scf.type_name() == "scf_solver"

    def test_create_with_kwargs(self):
        """Test creating an algorithm with settings passed as kwargs."""
        # Create an SCF solver with custom settings
        scf = registry.create("scf_solver", "qdk", max_iterations=100)
        assert scf is not None
        settings = scf.settings().to_dict()
        assert settings["max_iterations"] == 100

    def test_create_with_multiple_kwargs(self):
        """Test creating an algorithm with multiple settings."""
        scf = registry.create(
            "scf_solver",
            "qdk",
            max_iterations=75,
            convergence_threshold=1e-8,
        )
        assert scf is not None
        settings = scf.settings().to_dict()
        assert settings["max_iterations"] == 75
        assert settings["convergence_threshold"] == 1e-8

    def test_create_invalid_algorithm_type(self):
        """Test that creating an algorithm with invalid type raises KeyError."""
        with pytest.raises(KeyError) as excinfo:
            registry.create("nonexistent_type", "some_name")
        assert "nonexistent_type" in str(excinfo.value)
        assert "not registered" in str(excinfo.value)

    def test_create_invalid_algorithm_name(self):
        """Test that creating an algorithm with invalid name raises KeyError."""
        with pytest.raises(KeyError) as excinfo:
            registry.create("scf_solver", "nonexistent_algorithm")
        assert "nonexistent_algorithm" in str(excinfo.value)
        assert "not found" in str(excinfo.value)

    def test_create_different_algorithm_types(self):
        """Test creating algorithms of different types."""
        # Test active space selector
        active_space = registry.create("active_space_selector", "qdk_autocas_eos")
        assert active_space is not None
        assert active_space.type_name() == "active_space_selector"

        # Test hamiltonian constructor
        ham = registry.create("hamiltonian_constructor", "qdk")
        assert ham is not None
        assert ham.type_name() == "hamiltonian_constructor"

        # Test qubit mapper
        mapper = registry.create("qubit_mapper", "qiskit")
        assert mapper is not None
        assert mapper.type_name() == "qubit_mapper"


class TestRegistryAvailable:
    """Test the available function in the registry module."""

    def test_available_returns_dict_when_no_type_specified(self):
        """Test that available returns a dict when called without arguments."""
        all_algorithms = registry.available()
        assert isinstance(all_algorithms, dict)
        assert len(all_algorithms) > 0

        # Check some expected algorithm types exist
        expected_types = ["scf_solver", "active_space_selector", "hamiltonian_constructor"]
        for expected_type in expected_types:
            assert expected_type in all_algorithms

    def test_available_returns_list_for_specific_type(self):
        """Test that available returns a list when called with a specific type."""
        scf_solvers = registry.available("scf_solver")
        assert isinstance(scf_solvers, list)
        assert len(scf_solvers) > 0
        assert "qdk" in scf_solvers

    def test_available_returns_empty_list_for_unknown_type(self):
        """Test that available returns empty list for unknown algorithm type."""
        result = registry.available("nonexistent_type")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_available_all_types_have_algorithms(self):
        """Test that all algorithm types have at least one algorithm available."""
        all_algorithms = registry.available()
        for algorithm_type, algorithms in all_algorithms.items():
            assert isinstance(algorithms, list), f"Expected list for {algorithm_type}"
            assert len(algorithms) > 0, f"No algorithms available for {algorithm_type}"

    def test_available_algorithms_can_be_created(self):
        """Test that all available algorithms can actually be created."""
        # Test a few algorithm types
        test_types = ["scf_solver", "hamiltonian_constructor", "qubit_mapper"]
        for algorithm_type in test_types:
            algorithms = registry.available(algorithm_type)
            for algorithm_name in algorithms:
                # Should be able to create each algorithm
                alg = registry.create(algorithm_type, algorithm_name)
                assert alg is not None
                assert alg.type_name() == algorithm_type


class TestRegistryPrintSettings:
    """Test the print_settings function in the registry module."""

    def test_print_settings_produces_output(self, capsys):
        """Test that print_settings produces output."""
        registry.print_settings("scf_solver", "qdk")
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        # Check for table formatting
        assert "-" in captured.out

    def test_print_settings_with_custom_width(self, capsys):
        """Test print_settings with custom character width."""
        registry.print_settings("scf_solver", "qdk", characters=80)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_print_settings_invalid_type(self):
        """Test that print_settings raises KeyError for invalid algorithm type."""
        with pytest.raises(KeyError) as excinfo:
            registry.print_settings("nonexistent_type", "some_name")
        assert "nonexistent_type" in str(excinfo.value)
        assert "not registered" in str(excinfo.value)

    def test_print_settings_different_algorithms(self, capsys):
        """Test print_settings for different algorithm types."""
        # Test for hamiltonian constructor
        registry.print_settings("hamiltonian_constructor", "qdk")
        captured = capsys.readouterr()
        assert len(captured.out) > 0

        # Test for active space selector
        registry.print_settings("active_space_selector", "qdk_autocas_eos")
        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestRegistryInspectSettings:
    """Test the inspect_settings function in the registry module."""

    def test_inspect_settings_returns_list_of_tuples(self):
        """Test that inspect_settings returns correct data structure."""
        settings_info = registry.inspect_settings("scf_solver", "qdk")
        assert isinstance(settings_info, list)
        assert len(settings_info) > 0

        # Check tuple structure
        for item in settings_info:
            assert isinstance(item, tuple)
            assert len(item) == 5
            name, python_type, _default, _description, _limits = item
            assert isinstance(name, str)
            assert isinstance(python_type, str)
            # default can be any type
            # description can be str or None
            # limits can be any type or None

    def test_inspect_settings_has_expected_fields(self):
        """Test that inspect_settings returns settings with expected fields."""
        settings_info = registry.inspect_settings("scf_solver", "qdk")
        setting_names = [item[0] for item in settings_info]
        # Check for some expected settings
        expected_settings = ["max_iterations", "convergence_threshold"]
        for expected in expected_settings:
            assert expected in setting_names, f"Expected setting '{expected}' not found"

    def test_inspect_settings_invalid_type(self):
        """Test that inspect_settings raises KeyError for invalid algorithm type."""
        with pytest.raises(KeyError) as excinfo:
            registry.inspect_settings("nonexistent_type", "some_name")
        assert "nonexistent_type" in str(excinfo.value)

    def test_inspect_settings_different_algorithms(self):
        """Test inspect_settings for different algorithm types."""
        # Test hamiltonian constructor
        ham_settings = registry.inspect_settings("hamiltonian_constructor", "qdk")
        assert len(ham_settings) > 0
        # Test active space selector
        active_space_settings = registry.inspect_settings("active_space_selector", "qdk_autocas_eos")
        assert len(active_space_settings) > 0

    def test_inspect_settings_limits_format(self):
        """Test that limits in inspect_settings have correct format."""
        settings_info = registry.inspect_settings("scf_solver", "qdk")
        for name, _python_type, _default, _description, limits in settings_info:
            if limits is not None:
                # Limits can be tuple (for ranges) or list (for allowed values)
                assert isinstance(limits, tuple | list), f"Unexpected limits type for {name}: {type(limits)}"


class TestRegistryRegisterUnregister:
    """Test the register and unregister functions in the registry module."""

    def test_register_and_create_custom_algorithm(self):
        """Test registering a custom algorithm and creating it."""

        class CustomTestScf(ScfSolver):
            def name(self):
                return "custom_test_scf_v1"

            def _run_impl(self, structure, charge, spin_multiplicity):
                # Minimal implementation for testing
                pass

        # Register the custom algorithm
        registry.register(lambda: CustomTestScf())

        # Should now be able to create it
        scf = registry.create("scf_solver", "custom_test_scf_v1")
        assert scf is not None
        assert scf.name() == "custom_test_scf_v1"

        # Clean up
        registry.unregister("scf_solver", "custom_test_scf_v1")

    def test_register_custom_algorithm_appears_in_available(self):
        """Test that registered custom algorithm appears in available list."""

        class CustomTestScf2(ScfSolver):
            def name(self):
                return "custom_test_scf_v2"

            def _run_impl(self, structure, charge, spin_multiplicity):
                pass

        # Register
        registry.register(lambda: CustomTestScf2())

        # Should appear in available list
        available_scf = registry.available("scf_solver")
        assert "custom_test_scf_v2" in available_scf

        # Clean up
        registry.unregister("scf_solver", "custom_test_scf_v2")

    def test_unregister_custom_algorithm(self):
        """Test unregistering a custom algorithm."""

        class CustomTestScf3(ScfSolver):
            def name(self):
                return "custom_test_scf_v3"

            def _run_impl(self, structure, charge, spin_multiplicity):
                pass

        # Register and verify it exists
        registry.register(lambda: CustomTestScf3())
        available_before = registry.available("scf_solver")
        assert "custom_test_scf_v3" in available_before

        # Unregister
        registry.unregister("scf_solver", "custom_test_scf_v3")

        # Should no longer be available
        available_after = registry.available("scf_solver")
        assert "custom_test_scf_v3" not in available_after

    def test_unregister_invalid_type(self):
        """Test that unregister raises KeyError for invalid algorithm type."""
        with pytest.raises(KeyError) as excinfo:
            registry.unregister("nonexistent_type", "some_name")
        assert "nonexistent_type" in str(excinfo.value)

    def test_register_invalid_algorithm_type(self):
        """Test that registering an algorithm with invalid type raises KeyError."""

        class FakeAlgorithm:
            def type_name(self):
                return "nonexistent_type"

        with pytest.raises(KeyError):
            registry.register(lambda: FakeAlgorithm())


class TestRegistryFactoryRegistration:
    """Test the register_factory and unregister_factory functions."""

    def test_register_factory_duplicate(self):
        """Test that registering a duplicate factory raises ValueError."""
        # ScfSolverFactory is already registered
        with pytest.raises(ValueError, match="already registered"):
            registry.register_factory(ScfSolverFactory)

    def test_unregister_factory_invalid_type(self):
        """Test that unregister_factory raises KeyError for invalid type."""
        with pytest.raises(KeyError) as excinfo:
            registry.unregister_factory("nonexistent_factory_type")
        assert "not registered" in str(excinfo.value)
