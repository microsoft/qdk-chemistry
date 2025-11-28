"""Tests for the algorithms module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry import algorithms
from qdk_chemistry.algorithms import (
    ActiveSpaceSelector,
    CoupledClusterCalculator,
    HamiltonianConstructor,
    MultiConfigurationCalculator,
    MultiConfigurationScf,
    OrbitalLocalizer,
    ProjectedMultiConfigurationCalculator,
    ScfSolver,
    StabilityChecker,
    StatePreparation,
)
from qdk_chemistry.data import (
    Ansatz,
    CasWavefunctionContainer,
    Configuration,
    CoupledClusterAmplitudes,
    Hamiltonian,
    Orbitals,
    Settings,
    SlaterDeterminantContainer,
    StabilityResult,
    Structure,
    Wavefunction,
)

from .test_helpers import create_test_basis_set, create_test_hamiltonian, create_test_orbitals


class MockLocalizationPy(OrbitalLocalizer):
    """A dummy localizer for testing purposes in Python."""

    def __init__(self):
        super().__init__()
        self._settings = Settings()

    def _run_impl(self, orbitals, loc_indices_a, loc_indices_b):  # noqa: ARG002
        """Fake localize orbitals in Python."""
        return orbitals

    def name(self) -> str:
        """Return the algorithm name."""
        return "mock_localizer"


class MockStabilityChecker(StabilityChecker):
    """A dummy stability checker for testing purposes in Python."""

    def __init__(self):
        super().__init__()
        self._settings = Settings()
        # Define default settings for stability checking
        self._settings._set_default("tolerance", "double", 1e-6)
        self._settings._set_default("max_iterations", "int", 100)

    def _run_impl(self, wavefunction: Wavefunction) -> tuple[bool, StabilityResult]:  # noqa: ARG002
        """Fake stability check implementation."""
        # Create some mock eigenvalues and eigenvectors for internal and external
        internal_eigenvalues = np.array([0.1, 0.5, 1.0])  # All positive = stable
        external_eigenvalues = np.array([0.2, 0.8])  # All positive = stable
        internal_eigenvectors = np.eye(3)
        external_eigenvectors = np.eye(2)
        internal_stable = np.all(internal_eigenvalues > 0)
        external_stable = np.all(external_eigenvalues > 0)
        result = StabilityResult(
            internal_stable,
            external_stable,
            internal_eigenvalues,
            internal_eigenvectors,
            external_eigenvalues,
            external_eigenvectors,
        )
        return (result.is_stable(), result)

    def name(self) -> str:
        """Return the algorithm name."""
        return "mock_stability_checker"


class MockMultiConfigurationCalculator(MultiConfigurationCalculator):
    def __init__(self):
        super().__init__()
        self._settings = Settings()
        # Define default settings
        self._settings._set_default("test_param", "int", 42)
        self._settings._set_default("test_parameter", "string", "")
        self._settings._set_default("numeric_param", "double", 0.0)
        self._settings._set_default("list_param", "vector<int>", [])

    def _run_impl(self, hamiltonian, _: int, __: int):
        """A simple test implementation of the calculate method."""
        sd = SlaterDeterminantContainer(Configuration("20"), hamiltonian.get_orbitals())
        return 0.0, Wavefunction(sd)

    def name(self) -> str:
        """Return the algorithm name."""
        return "mock_mc_calculator"


class MockProjectedMultiConfigurationCalculator(ProjectedMultiConfigurationCalculator):
    def __init__(self):
        super().__init__()
        self._settings = Settings()
        # Define default settings
        self._settings._set_default("test_param", "int", 42)
        self._settings._set_default("h_el_tol", "double", 1e-12)
        self._settings._set_default("calculate_one_rdm", "bool", False)
        self._settings._set_default("calculate_two_rdm", "bool", False)

    def _run_impl(self, hamiltonian, configurations):
        """A simple test implementation of the calculate method."""
        # Use first configuration for the wavefunction determinant
        if not configurations:
            raise RuntimeError("Empty configuration set")

        sd = SlaterDeterminantContainer(configurations[0], hamiltonian.get_orbitals())
        return 0.0, Wavefunction(sd)

    def name(self) -> str:
        """Return the algorithm name."""
        return "mock_pmc_calculator"


class MockHamiltonianConstructor(HamiltonianConstructor):
    def __init__(self):
        super().__init__()
        self._settings = Settings()
        # Define default settings
        self._settings._set_default("basis_set", "string", "def2-svp")
        self._settings._set_default("test_parameter", "string", "")
        self._settings._set_default("numeric_param", "double", 0.0)
        self._settings._set_default("list_param", "vector<int>", [])

    def _run_impl(self, orbitals):
        """A simple test implementation of the construct method."""
        # Simple test implementation - create basic Hamiltonian

        # Use valid orbitals or create test ones
        if orbitals.get_num_molecular_orbitals() == 0:
            # Create minimal valid orbitals for testing
            coeffs = np.eye(2)  # 2x2 identity matrix
            energies = np.array([0.0, 1.0])
            orbitals = Orbitals(coeffs, np.array([2.0, 0.0]), energies)

        size = orbitals.get_num_molecular_orbitals() if orbitals.get_num_molecular_orbitals() > 0 else 2

        # Create identity matrices for one-body and two-body integrals
        one_body = np.eye(size)
        two_body = np.zeros(size**4)
        fock = np.eye(0)

        # Create Hamiltonian with proper constructor
        return Hamiltonian(one_body, two_body, orbitals, 0.0, fock)

    def name(self) -> str:
        """Return the algorithm name."""
        return "mock_hamiltonian_constructor"


class MockScfSolver(ScfSolver):
    def __init__(self):
        super().__init__()
        self._settings = Settings()
        # Define and set default settings
        self._settings._set_default("max_iterations", "int", 100)
        self._settings._set_default("convergence_threshold", "double", 1e-8)
        self._settings._set_default("basis_set", "string", "def2-svp")
        self._settings._set_default("test_parameter", "string", "")
        self._settings._set_default("numeric_param", "double", 0.0)
        self._settings._set_default("list_param", "vector<int>", [])

    def _run_impl(self, structure, charge: int, multiplicity: int, initial_guess: Orbitals | None = None):  # noqa: ARG002
        """A simple test implementation of the solve method."""
        # Simple test implementation - create valid orbitals
        coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2x2 identity
        energies = np.array([0.0, 1.0])
        basis_set = create_test_basis_set(2)
        orbitals = Orbitals(coeffs, energies, None, basis_set)
        hf_config = Configuration("20")
        wavefunction = Wavefunction(SlaterDeterminantContainer(hf_config, orbitals))
        return 0.0, wavefunction

    def name(self) -> str:
        """Return the algorithm name."""
        return "mock_scf_solver"


class MockStatePreparation(StatePreparation):
    def __init__(self):
        super().__init__()
        self._settings = Settings()
        self._settings._set_default("test_parameter", "string", "")
        self._settings._set_default("numeric_param", "double", 0.0)
        self._settings._set_default("list_param", "vector<int>", [])

    def _run_impl(self, _) -> str:
        """A simple test implementation of the prepare_state method."""
        return "mock_circuit_representation"

    def name(self) -> str:
        """Return the algorithm name."""
        return "mock_state_preparation"


class MockActiveSpaceSelector(ActiveSpaceSelector):
    def __init__(self):
        super().__init__()
        self._settings = Settings()
        # Define and set default settings
        self._settings._set_default("num_active_orbitals", "int", 4)
        self._settings._set_default("selection_method", "string", "highest_occupied")
        self._settings._set_default("threshold", "double", 1e-6)
        self._settings._set_default("test_parameter", "string", "")
        self._settings._set_default("numeric_param", "double", 0.0)
        self._settings._set_default("list_param", "vector<int>", [])

    def _run_impl(self, wavefunction):
        """A simple test implementation of the select_active_space method."""
        # Simple test implementation - return first few orbital indices
        num_active = self._settings.get_or_default("num_active_orbitals", 4)
        orbitals = wavefunction.get_orbitals()
        indices = list(
            range(
                min(
                    num_active,
                    orbitals.get_num_molecular_orbitals() if orbitals.get_num_molecular_orbitals() > 0 else num_active,
                )
            )
        )
        # Create a new orbitals object with active space data
        coeffs_data = orbitals.get_coefficients()
        energies_data = orbitals.get_energies() if orbitals.get_energies() is not None else None

        # Check if this is unrestricted (returns tuples) or restricted (returns single arrays)
        if isinstance(coeffs_data, tuple):
            # Unrestricted case - use alpha/beta constructor
            coeffs_alpha, coeffs_beta = coeffs_data
            if energies_data is not None:
                energies_alpha, energies_beta = energies_data
            else:
                energies_alpha, energies_beta = None, None
            new_orbitals = Orbitals(
                coeffs_alpha,
                coeffs_beta,
                energies_alpha,
                energies_beta,
                None,
                orbitals.get_basis_set(),
                [indices, indices, [], []],
            )
            return Wavefunction(SlaterDeterminantContainer(wavefunction.get_active_determinants()[0], new_orbitals))
        # Restricted case - use restricted constructor
        new_orbitals = Orbitals(
            coeffs_data,
            energies_data,
            None,
            orbitals.get_basis_set(),
            [indices, []],
        )
        return Wavefunction(SlaterDeterminantContainer(wavefunction.get_active_determinants()[0], new_orbitals))

    def name(self) -> str:
        """Return the algorithm name."""
        return "mock_cas_selector"


class MockMultiConfigurationScf(MultiConfigurationScf):
    def __init__(self):
        super().__init__()
        self._settings = Settings()
        # Define and set default settings
        self._settings._set_default("max_iterations", "int", 50)
        self._settings._set_default("convergence_threshold", "double", 1e-6)
        self._settings._set_default("active_space_size", "int", 4)
        self._settings._set_default("basis_set", "string", "def2-svp")
        self._settings._set_default("test_parameter", "string", "")
        self._settings._set_default("numeric_param", "double", 0.0)
        self._settings._set_default("list_param", "vector<int>", [])

    def _run_impl(self, _orbs, _hamil_constr, _mc_solver, __: int, ___: int):
        """A simple test implementation of the solve method."""
        # Simple test implementation - return basic energy and wavefunction
        energy = -1.5  # Mock energy value
        hf_config_str = "20"
        hf_config = Configuration(hf_config_str)
        orbitals = create_test_orbitals(2)
        wavefunction = Wavefunction(SlaterDeterminantContainer(hf_config, orbitals))
        return energy, wavefunction

    def name(self) -> str:
        """Return the algorithm name."""
        return "mock_mcscf_solver"


class MockCoupledClusterCalculator(CoupledClusterCalculator):
    """A test calculator for coupled cluster methods."""

    def __init__(self):
        super().__init__()
        self._settings = Settings()
        # Define default settings
        self._settings._set_default("cc_type", "string", "CCSD")
        self._settings._set_default("test_parameter", "string", "")
        self._settings._set_default("numeric_param", "double", 0.0)
        self._settings._set_default("list_param", "vector<int>", [])

    def _run_impl(self, ansatz):
        """Implement the calculate method."""
        # Create a properly set up orbital object with canonical and restricted characteristics
        # Create minimal valid orbitals for CC result
        coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])
        # Set orbital energies (important for canonical orbitals)
        # The occupied orbital should have lower energy than the virtual
        energies = np.array([-1.0, 0.5])  # First orbital has lower energy
        # Set occupations for restricted case (2.0 = doubly occupied, 0.0 = unoccupied)
        # This makes it both restricted (same alpha/beta) and canonical (integer occupations)
        orbs = Orbitals(coeffs, energies, None, create_test_basis_set(2))

        # Create dummy amplitudes
        # For 1 occupied and 1 virtual orbital:
        # T1 size = num_occupied * num_virtual = 1 * 1 = 1
        t1 = np.array([0.01])

        # T2 size = (num_occupied * num_virtual)^2 = (1 * 1)^2 = 1
        t2 = np.array([0.005])

        num_alpha, num_beta = ansatz.get_wavefunction().get_total_num_electrons()

        # Create a dummy coupled cluster result
        cc = CoupledClusterAmplitudes(orbs, t1, t2, num_alpha, num_beta)

        # Return energy and coupled cluster amplitudes object
        return -10.0, cc

    def name(self) -> str:
        """Return the algorithm name."""
        return "mock_coupled_cluster_calculator"


class TestAlgorithmClasses:
    """Test cases for the algorithm base classes."""

    @pytest.fixture
    def basic_orbitals(self):
        """Create basic orbitals for testing."""
        coeffs = np.array([[0.9, 0.1], [0.1, -0.9], [0.0, 0.0]])
        basis_set = create_test_basis_set(3, "test-ansatz")
        return Orbitals(coeffs, None, None, basis_set)

    @pytest.fixture
    def test_structure(self):
        """Create a test structure (H2 molecule)."""
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
        elements = ["H", "H"]
        return Structure(positions, elements)

    @pytest.fixture
    def test_basis_set(self, test_structure):
        """Create a test basis set."""
        return create_test_basis_set(2, "test-ansatz-basis", test_structure)

    @pytest.fixture
    def test_wavefunction(self, basic_orbitals):
        """Create a test wavefunction."""
        det1 = Configuration("20")
        coeffs = np.array([1.0])  # Single determinant with coefficient 1.0

        container = CasWavefunctionContainer(coeffs, [det1], basic_orbitals)
        return Wavefunction(container)

    @pytest.fixture
    def test_ansatz(self, test_wavefunction):
        """Create a test ansatz."""
        # Create a hamiltonian using the same orbitals as the wavefunction
        orbitals = test_wavefunction.get_orbitals()
        num_orbitals = orbitals.get_num_molecular_orbitals()

        # Create simple one and two body integrals
        one_body = np.eye(num_orbitals)
        two_body = np.zeros(num_orbitals**4)
        fock = np.eye(0)

        test_hamiltonian = Hamiltonian(one_body, two_body, orbitals, 0.0, fock)
        return Ansatz(test_hamiltonian, test_wavefunction)

    def test_multi_configuration_calculator_inheritance(self):
        """Test that MultiConfigurationCalculator can be inherited from Python."""
        # Create instance
        mc_calc = MockMultiConfigurationCalculator()
        assert isinstance(mc_calc, MultiConfigurationCalculator)

        # Test settings method
        settings = mc_calc.settings()
        assert isinstance(settings, Settings)

        # Test setting and getting values
        settings["test_param"] = 42
        assert settings["test_param"] == 42

        # Test calculate method
        h = create_test_hamiltonian(2)
        energy, result = mc_calc.run(h, 1, 1)
        assert isinstance(energy, float)
        assert isinstance(result, Wavefunction)

    def test_pmc_calculator_inheritance(self):
        """Test that ProjectedMultiConfigurationCalculator can be inherited from Python."""
        # Create instance
        pmc_calc = MockProjectedMultiConfigurationCalculator()
        assert isinstance(pmc_calc, ProjectedMultiConfigurationCalculator)

        # Test settings method
        settings = pmc_calc.settings()
        assert isinstance(settings, Settings)

        # Test setting and getting values
        settings["test_param"] = 42
        assert settings["test_param"] == 42

        # Test calculate method with configurations
        h = create_test_hamiltonian(2)
        configurations = [Configuration("20"), Configuration("02")]
        energy, result = pmc_calc.run(h, configurations)
        assert isinstance(energy, float)
        assert isinstance(result, Wavefunction)

    def test_state_prep_inheritance(self):
        """Test that StatePreparation can be inherited from Python."""
        # Create instance
        state_prep = MockStatePreparation()
        assert isinstance(state_prep, StatePreparation)

        # Test settings method
        settings = state_prep.settings()
        assert isinstance(settings, Settings)

        # Test prepare_state method
        wavefunction = Wavefunction(SlaterDeterminantContainer(Configuration("20"), create_test_orbitals(2)))
        circuit = state_prep.run(wavefunction)
        assert isinstance(circuit, str)
        assert circuit == "mock_circuit_representation"

    def test_hamiltonian_constructor_inheritance(self):
        """Test that HamiltonianConstructor can be inherited from Python."""
        # Create instance
        ham_constructor = MockHamiltonianConstructor()
        assert isinstance(ham_constructor, HamiltonianConstructor)

        # Test settings method
        settings = ham_constructor.settings()
        assert isinstance(settings, Settings)

        # Test setting and getting values
        settings["basis_set"] = "6-31G"
        assert settings["basis_set"] == "6-31G"

        # Test construct method with basic orbitals
        coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])
        energies = np.array([0.0, 1.0])
        orbitals = Orbitals(coeffs, energies, None, create_test_basis_set(2))

        result = ham_constructor.run(orbitals)
        assert isinstance(result, Hamiltonian)
        # For legacy constructor, check that it has one-body integrals set
        assert result.has_one_body_integrals()

    def test_scf_solver_inheritance(self):
        """Test that ScfSolver can be inherited from Python."""
        # Create instance
        scf_solver = MockScfSolver()
        assert isinstance(scf_solver, ScfSolver)

        # Test settings method
        settings = scf_solver.settings()
        assert isinstance(settings, Settings)

        # Test default settings
        assert settings["max_iterations"] == 100
        assert settings["convergence_threshold"] == 1e-8

        # Test modifying settings
        settings["basis_set"] = "STO-3G"
        assert settings["basis_set"] == "STO-3G"

        # Test solve method with simple structure
        coords = np.array([[0.0, 0.0, 0.0]])
        symbols = ["H"]
        structure = Structure(coords, symbols)

        energy, result = scf_solver.run(structure, 0, 1)
        assert isinstance(energy, float)
        assert isinstance(result, Wavefunction)

    def test_active_space_selector_inheritance(self):
        """Test that ActiveSpaceSelector can be inherited from Python."""
        # Create instance
        selector = MockActiveSpaceSelector()
        assert isinstance(selector, ActiveSpaceSelector)

        # Test settings method
        settings = selector.settings()
        assert isinstance(settings, Settings)

        # Test default settings
        assert settings["num_active_orbitals"] == 4
        assert settings["selection_method"] == "highest_occupied"

        # Test modifying settings
        settings["threshold"] = 0.01
        assert settings["threshold"] == 0.01

        # Test select_active_space method with basic orbitals
        coeffs = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        energies = np.array([0.0, 1.0, 2.0, 3.0])
        orbitals = Orbitals(coeffs, energies, None, create_test_basis_set(4))

        wavefunction = Wavefunction(SlaterDeterminantContainer(Configuration("2200"), orbitals))

        selected_wfn = selector.run(wavefunction)
        active_orbitals = selected_wfn.get_orbitals()
        assert isinstance(active_orbitals, Orbitals)
        assert active_orbitals.has_active_space()
        active_indices_pair = active_orbitals.get_active_space_indices()
        # For restricted case (which we're using here), the first element has the indices
        active_indices = active_indices_pair[0]  # alpha indices
        assert len(active_indices) <= 4  # Should not exceed num_active_orbitals
        assert all(isinstance(idx, int) for idx in active_indices)

    def test_mcscf_inheritance(self):
        """Test that MultiConfigurationScf can be inherited from Python."""
        # Create instance
        mcscf_solver = MockMultiConfigurationScf()
        assert isinstance(mcscf_solver, MultiConfigurationScf)

        # Test settings method
        settings = mcscf_solver.settings()
        assert isinstance(settings, Settings)

        # Test default settings
        assert settings["max_iterations"] == 50
        assert settings["convergence_threshold"] == 1e-6
        assert settings["active_space_size"] == 4

        # Test modifying settings
        settings["basis_set"] = "6-31G"
        assert settings["basis_set"] == "6-31G"

        # Test solve method with basic hamiltonian and mc calculator
        orbitals = create_test_orbitals(2)
        hamiltonian_creator = MockHamiltonianConstructor()
        cas_solver = MockMultiConfigurationCalculator()
        energy, wavefunction = mcscf_solver.run(orbitals, hamiltonian_creator, cas_solver, 1, 1)
        assert isinstance(energy, float)
        assert isinstance(wavefunction, Wavefunction)

    def test_coupled_cluster_calculator_inheritance(self, test_ansatz):
        """Test that CoupledClusterCalculator can be inherited from Python."""
        # Create instance
        coupled_cluster_calculator = MockCoupledClusterCalculator()
        assert isinstance(coupled_cluster_calculator, CoupledClusterCalculator)

        # Test settings method
        settings = coupled_cluster_calculator.settings()
        assert isinstance(settings, Settings)

        # Test adding settings
        settings["cc_type"] = "CCSD(T)"
        assert settings["cc_type"] == "CCSD(T)"

        # Test calculate method with basic hamiltonian
        energy, cc_result = coupled_cluster_calculator.run(test_ansatz)

        # Check results
        assert isinstance(energy, float)
        assert isinstance(cc_result, CoupledClusterAmplitudes)
        assert cc_result.has_t1_amplitudes()
        assert cc_result.has_t2_amplitudes()

        # Check orbital counts to verify restricted orbitals setup
        alpha_occ, beta_occ = cc_result.get_num_occupied()
        alpha_virt, beta_virt = cc_result.get_num_virtual()

        # For restricted orbitals, alpha and beta counts should match
        assert alpha_occ == beta_occ
        assert alpha_virt == beta_virt

        # Check that we have 1 occupied and 1 virtual orbital as set in the mock
        assert alpha_occ == 1
        assert alpha_virt == 1

        # Verify the orbitals are properly set
        orbs_pair = cc_result.get_num_occupied()
        assert orbs_pair[0] == 1  # One alpha occupied
        assert orbs_pair[1] == 1  # One beta occupied

        virtual_pair = cc_result.get_num_virtual()
        assert virtual_pair[0] == 1  # One alpha virtual
        assert virtual_pair[1] == 1  # One beta virtual

    def test_scf_solver_registration(self):
        """Test that SCF solver can be registered and used."""

        def _test_register_scf_solver():
            """Dummy function to simulate registration."""
            return MockScfSolver()

        # Register the solver
        key = "mock_scf_solver"
        algorithms.register(_test_register_scf_solver)

        # Verify registration worked
        assert key in algorithms.available("scf_solver")

        # Test that the correct instance is created
        scf_solver = algorithms.create("scf_solver", key)
        assert isinstance(scf_solver, MockScfSolver)

    def test_hamiltonian_constructor_registration(self):
        """Test that Hamiltonian constructor can be registered and used."""

        def _test_register_hamiltonian_constructor():
            """Dummy function to simulate registration."""
            return MockHamiltonianConstructor()

        # Register the constructor
        key = "mock_hamiltonian_constructor"
        algorithms.register(_test_register_hamiltonian_constructor)

        # Verify registration worked
        assert key in algorithms.available("hamiltonian_constructor")

        # Test that the correct instance is created
        constructor = algorithms.create("hamiltonian_constructor", key)
        assert isinstance(constructor, MockHamiltonianConstructor)

    def test_localizer_registration(self):
        """Test that Localizer can be registered and used."""

        def _test_register_localizer():
            """Dummy function to simulate registration."""
            # Use the existing MockLocalizationPy class from the module
            return MockLocalizationPy()

        # Register the localizer
        key = "mock_localizer"
        if key in algorithms.available("orbital_localizer"):
            algorithms.unregister("orbital_localizer", key)  # Ensure clean state
        assert key not in algorithms.available("orbital_localizer")
        algorithms.register(_test_register_localizer)

        # Verify registration worked
        assert key in algorithms.available("orbital_localizer")

        # Test that the correct instance is created
        localizer = algorithms.create("orbital_localizer", key)
        assert localizer is not None
        # MockLocalizationPy should have a specific docstring

    def test_multi_configuration_calculator_registration(self):
        """Test that MC calculator can be registered and used."""

        def _test_register_multi_configuration_calculator():
            """Dummy function to simulate registration."""
            return MockMultiConfigurationCalculator()

        # Register the calculator
        key = "mock_mc_calculator"
        algorithms.register(_test_register_multi_configuration_calculator)

        # Verify registration worked
        assert key in algorithms.available("multi_configuration_calculator")

        # Test that the correct instance is created
        calculator = algorithms.create("multi_configuration_calculator", key)
        assert isinstance(calculator, MockMultiConfigurationCalculator)

    def test_pmc_calculator_registration(self):
        """Test that PMC calculator can be registered and used."""

        def _test_register_pmc_calculator():
            """Dummy function to simulate registration."""
            return MockProjectedMultiConfigurationCalculator()

        # Register the calculator
        key = "mock_pmc_calculator"
        algorithms.register(_test_register_pmc_calculator)

        # Verify registration worked
        assert key in algorithms.available("projected_multi_configuration_calculator")

        # Test that the correct instance is created
        calculator = algorithms.create("projected_multi_configuration_calculator", key)
        assert isinstance(calculator, MockProjectedMultiConfigurationCalculator)

    def test_state_preparation_registration(self):
        """Test that State Preparation can be registered and used."""

        def _test_register_state_preparation():
            """Dummy function to simulate registration."""
            return MockStatePreparation()

        # Register the state preparation
        key = "mock_state_preparation"
        algorithms.register(_test_register_state_preparation)

        # Verify registration worked
        assert key in algorithms.available("state_prep")

        # Test that the correct instance is created
        state_prep = algorithms.create("state_prep", key)
        assert isinstance(state_prep, MockStatePreparation)

        algorithms.unregister("state_prep", key)  # Clean up after test

    def test_active_space_selector_registration(self):
        """Test that Active Space Selector can be registered and used."""

        def _test_register_active_space_selector():
            """Dummy function to simulate registration."""
            return MockActiveSpaceSelector()

        # Register the selector
        key = "mock_cas_selector"
        algorithms.register(_test_register_active_space_selector)

        # Verify registration worked
        assert key in algorithms.available("active_space_selector")

        # Test that the correct instance is created
        selector = algorithms.create("active_space_selector", key)
        assert isinstance(selector, MockActiveSpaceSelector)

    def test_mcscf_registration(self):
        """Test that MultiConfigurationScf can be registered and used."""

        def _test_register_mcscf():
            """Dummy function to simulate registration."""
            return MockMultiConfigurationScf()

        # Register the solver
        key = "mock_mcscf_solver"
        algorithms.register(_test_register_mcscf)

        # Verify registration worked
        assert key in algorithms.available("multi_configuration_scf")

        # Test that the correct instance is created
        mcscf_solver = algorithms.create("multi_configuration_scf", key)
        assert isinstance(mcscf_solver, MockMultiConfigurationScf)

    def test_coupled_cluster_calculator_registration(self):
        """Test that CoupledClusterCalculator can be registered and used."""

        def _test_register_coupled_cluster_calculator():
            """Dummy function to simulate registration."""
            return MockCoupledClusterCalculator()

        # Register the calculator
        key = "mock_coupled_cluster_calculator"
        if key in algorithms.available("coupled_cluster_calculator"):
            algorithms.unregister("coupled_cluster_calculator", key)  # Ensure clean state
        assert key not in algorithms.available("coupled_cluster_calculator")
        algorithms.register(_test_register_coupled_cluster_calculator)

        # Verify registration worked
        assert key in algorithms.available("coupled_cluster_calculator")

        # Test that the correct instance is created
        coupled_cluster_calculator = algorithms.create("coupled_cluster_calculator", key)
        assert isinstance(coupled_cluster_calculator, MockCoupledClusterCalculator)

    def test_algorithm_repr(self):
        """Test string representations of algorithm classes."""
        mc = MockMultiConfigurationCalculator()
        ham_constructor = MockHamiltonianConstructor()
        scf = MockScfSolver()
        selector = MockActiveSpaceSelector()
        mcscf = MockMultiConfigurationScf()
        cc = MockCoupledClusterCalculator()
        sp = MockStatePreparation()

        # Test that repr works (exact string may vary)
        assert "MultiConfigurationCalculator" in repr(mc)
        assert "HamiltonianConstructor" in repr(ham_constructor)
        assert "ScfSolver" in repr(scf)
        assert "ActiveSpaceSelector" in repr(selector)
        assert "MultiConfigurationScf" in repr(mcscf)
        assert "CoupledClusterCalculator" in repr(cc)
        assert "StatePreparation" in repr(sp)

    def test_settings_interface(self) -> None:
        """Test that all algorithms provide settings interface."""
        # Test all algorithm types have settings
        algorithms: list[
            MockMultiConfigurationCalculator
            | MockHamiltonianConstructor
            | MockScfSolver
            | MockActiveSpaceSelector
            | MockMultiConfigurationScf
            | MockCoupledClusterCalculator
            | MockStatePreparation
        ] = [
            MockMultiConfigurationCalculator(),
            MockHamiltonianConstructor(),
            MockScfSolver(),
            MockActiveSpaceSelector(),
            MockMultiConfigurationScf(),
            MockCoupledClusterCalculator(),
            MockStatePreparation(),
        ]

        for alg in algorithms:
            # Test settings method exists and returns Settings object
            settings = alg.settings()
            assert isinstance(settings, Settings)

            # Test settings can be configured
            settings["test_parameter"] = "test_value"
            assert settings["test_parameter"] == "test_value"

            # Test numeric settings
            settings["numeric_param"] = 42.5
            assert settings["numeric_param"] == 42.5

            # Test list settings
            settings["list_param"] = [1, 2, 3]
            assert settings["list_param"] == [1, 2, 3]

    def test_abstract_methods_required(self):
        """Test that abstract methods must be implemented."""
        # Note: These classes are not currently implemented as abstract in the C++ bindings
        # so they can be instantiated directly. This test documents current behavior.
        try:
            mc = MultiConfigurationCalculator()
            assert mc is not None
        except TypeError:
            pass  # Would be expected behavior for abstract classes

        try:
            hc = HamiltonianConstructor()
            assert hc is not None
        except TypeError:
            pass  # Would be expected behavior for abstract classes

        try:
            scf = ScfSolver()
            assert scf is not None
        except TypeError:
            pass  # Would be expected behavior for abstract classes

        try:
            sp = StatePreparation()
            assert sp is not None
        except TypeError:
            pass  # Would be expected behavior for abstract classes
        try:
            selector = ActiveSpaceSelector()
            assert selector is not None
        except TypeError:
            pass  # Would be expected behavior for abstract classes

        try:
            mcscf = MultiConfigurationScf()
            assert mcscf is not None
        except TypeError:
            pass  # Would be expected behavior for abstract classes

    def test_localizer_comprehensive(self):
        """Test comprehensive Localizer functionality including trampoline methods."""
        # Test MockLocalizationPy which inherits from OrbitalLocalizer
        localizer = MockLocalizationPy()
        assert isinstance(localizer, OrbitalLocalizer)

        # Test trampoline methods are properly overridden
        # Test localize method
        coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])
        basis_set = create_test_basis_set(2)
        orbitals = Orbitals(coeffs, None, None, basis_set)
        wavefunction = Wavefunction(SlaterDeterminantContainer(Configuration("20"), orbitals))

        # Test with orbital indices
        result = localizer.run(wavefunction, [0, 1], [0, 1])
        assert isinstance(result, Wavefunction)

        # Test settings method
        settings = localizer.settings()
        assert isinstance(settings, Settings)

        # Test __repr__ method
        repr_str = repr(localizer)
        assert "<qdk_chemistry.algorithms.OrbitalLocalizer>" in repr_str

    def test_localizer_factory_functions(self):
        """Test Localizer factory functions and unregistration."""

        def _test_factory_localizer():
            """Dummy function for factory testing."""
            return MockLocalizationPy()

        # Register a test localizer
        key = "mock_localizer"
        if key in algorithms.available("orbital_localizer"):
            algorithms.unregister("orbital_localizer", key)  # Ensure clean state
        assert key not in algorithms.available("orbital_localizer")

        algorithms.register(_test_factory_localizer)

        # Test that the localizer was registered
        assert key in algorithms.available("orbital_localizer")

    def test_localizer_base_class(self):
        """Test Localizer base class functionality."""
        # Test that base Localizer can be instantiated (though it's abstract)
        try:
            # This might not work if the class is properly abstract
            localizer = OrbitalLocalizer()

            # Test __repr__ method
            repr_str = repr(localizer)
            assert "<qdk_chemistry.algorithms.OrbitalLocalizer>" in repr_str
        except TypeError:
            # Expected behavior for abstract classes
            pass

    def test_stability_checker_inheritance(self):
        """Test StabilityChecker class inheritance."""
        checker = MockStabilityChecker()
        assert isinstance(checker, StabilityChecker)
        assert hasattr(checker, "run")
        assert hasattr(checker, "settings")

    def test_stability_checker_registration(self):
        """Test StabilityChecker registration and factory functions."""

        def _test_factory_stability_checker():
            """Dummy function for factory testing."""
            return MockStabilityChecker()

        # Get initial list of stability checkers
        initial_checkers = algorithms.available("stability_checker")

        # Register a test stability checker
        key = "mock_stability_checker"
        if key in initial_checkers:
            algorithms.unregister("stability_checker", key)  # Ensure clean state
        assert key not in algorithms.available("stability_checker")
        algorithms.register(_test_factory_stability_checker)

        # Test that the stability checker was registered
        current_checkers = algorithms.available("stability_checker")
        assert key in current_checkers
        assert len(current_checkers) == len(initial_checkers) + 1

        # Test creating stability checker by key
        created_checker = algorithms.create("stability_checker", key)
        assert isinstance(created_checker, StabilityChecker)

    def test_stability_result_functionality(self):
        """Test StabilityResult data class functionality."""
        # Test empty StabilityResult
        empty_result = StabilityResult()
        assert empty_result.internal_size() == 0
        assert empty_result.external_size() == 0
        assert empty_result.is_internal_stable()  # Default should be True
        assert empty_result.is_external_stable()  # Default should be True
        assert empty_result.is_stable()  # Both True = overall True
        assert "empty" in empty_result.get_summary()
        assert str(empty_result) == repr(empty_result)  # Should be identical

        # Test StabilityResult with stable data
        internal_eigenvals = np.array([0.1, 0.5, 1.0])
        external_eigenvals = np.array([0.2, 0.8])
        internal_eigenvecs = np.eye(3)
        external_eigenvecs = np.eye(2)
        stable_result = StabilityResult(
            True, True, internal_eigenvals, internal_eigenvecs, external_eigenvals, external_eigenvecs
        )

        assert stable_result.is_stable()
        assert stable_result.is_internal_stable()
        assert stable_result.is_external_stable()
        assert stable_result.internal_size() == 3
        assert stable_result.external_size() == 2
        assert stable_result.get_smallest_internal_eigenvalue() == 0.1
        assert stable_result.get_smallest_external_eigenvalue() == 0.2
        assert stable_result.get_smallest_eigenvalue() == 0.1  # Overall smallest
        assert "stable" in stable_result.get_summary()
        assert "internal" in stable_result.get_summary()
        assert "external" in stable_result.get_summary()
        assert str(stable_result) == repr(stable_result)  # Should be identical

        # Test StabilityResult with unstable data
        unstable_internal_eigenvals = np.array([-0.5, 0.1, 1.0])
        unstable_external_eigenvals = np.array([0.3, -0.2])
        unstable_result = StabilityResult(
            False,
            False,
            unstable_internal_eigenvals,
            internal_eigenvecs,
            unstable_external_eigenvals,
            external_eigenvecs,
        )

        assert not unstable_result.is_stable()
        assert not unstable_result.is_internal_stable()
        assert not unstable_result.is_external_stable()
        assert unstable_result.internal_size() == 3
        assert unstable_result.external_size() == 2
        assert unstable_result.get_smallest_internal_eigenvalue() == -0.5
        assert unstable_result.get_smallest_external_eigenvalue() == -0.2
        assert unstable_result.get_smallest_eigenvalue() == -0.5  # Overall smallest (most negative)
        assert "unstable" in unstable_result.get_summary() or "internal" in unstable_result.get_summary()

        # Test setters
        unstable_result.set_internal_stable(True)
        assert unstable_result.is_internal_stable()
        assert not unstable_result.is_stable()  # Still false because external is false

        unstable_result.set_external_stable(True)
        assert unstable_result.is_external_stable()
        assert unstable_result.is_stable()  # Now both are true

        # Test eigenvalue setters
        new_internal_eigenvals = np.array([0.2, 0.6, 1.2])
        unstable_result.set_internal_eigenvalues(new_internal_eigenvals)
        assert np.array_equal(unstable_result.get_internal_eigenvalues(), new_internal_eigenvals)
        assert unstable_result.get_smallest_internal_eigenvalue() == 0.2

        new_external_eigenvals = np.array([0.4, 0.9])
        unstable_result.set_external_eigenvalues(new_external_eigenvals)
        assert np.array_equal(unstable_result.get_external_eigenvalues(), new_external_eigenvals)
        assert unstable_result.get_smallest_external_eigenvalue() == 0.4
        assert unstable_result.get_smallest_eigenvalue() == 0.2  # Overall smallest

        # Test eigenvector setters
        rng = np.random.default_rng(42)
        new_internal_eigenvecs = rng.random((3, 3))
        unstable_result.set_internal_eigenvectors(new_internal_eigenvecs)
        assert np.array_equal(unstable_result.get_internal_eigenvectors(), new_internal_eigenvecs)

        new_external_eigenvecs = rng.random((2, 2))
        unstable_result.set_external_eigenvectors(new_external_eigenvecs)
        assert np.array_equal(unstable_result.get_external_eigenvectors(), new_external_eigenvecs)

        # Test eigenvalue-eigenvector pair methods
        internal_val, internal_vec = unstable_result.get_smallest_internal_eigenvalue_and_vector()
        assert internal_val == 0.2
        assert len(internal_vec) == 3

        external_val, external_vec = unstable_result.get_smallest_external_eigenvalue_and_vector()
        assert external_val == 0.4
        assert len(external_vec) == 2

        overall_val, overall_vec = unstable_result.get_smallest_eigenvalue_and_vector()
        assert overall_val == 0.2
        assert len(overall_vec) == 3  # Should be from internal since it's smaller

    def test_stability_checker_comprehensive(self):
        """Test comprehensive StabilityChecker functionality including trampoline methods."""
        # Test MockStabilityChecker which inherits from StabilityChecker
        checker = MockStabilityChecker()
        assert isinstance(checker, StabilityChecker)

        # Test settings method
        settings = checker.settings()
        assert isinstance(settings, Settings)

        # Test check method with a dummy wavefunction
        # Create minimal test components
        coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])
        basis_set = create_test_basis_set(2)
        orbitals = Orbitals(coeffs, None, None, basis_set)
        sd_container = SlaterDeterminantContainer(Configuration("20"), orbitals)
        wavefunction = Wavefunction(sd_container)

        # Test run method
        is_stable, result = checker.run(wavefunction)
        assert isinstance(result, StabilityResult)
        assert is_stable is True  # MockStabilityChecker returns stable result
        assert result.is_stable()  # MockStabilityChecker returns stable result
        assert result.internal_size() == 3  # MockStabilityChecker creates 3 internal eigenvalues
        assert result.external_size() == 2  # MockStabilityChecker creates 2 external eigenvalues

        # Test __repr__ method
        repr_str = repr(checker)
        assert "<qdk_chemistry.algorithms.StabilityChecker>" in repr_str

    def test_stability_checker_factory_functions(self):
        """Test StabilityChecker factory functions and error handling."""
        # Test that empty list is returned when no checkers are registered initially
        checkers = algorithms.available("stability_checker")
        assert isinstance(checkers, list)

        # Test creating stability checker with invalid key fails appropriately
        with pytest.raises(KeyError, match=r"Available algorithms for this type"):
            algorithms.create("stability_checker", "invalid_key_12345")
