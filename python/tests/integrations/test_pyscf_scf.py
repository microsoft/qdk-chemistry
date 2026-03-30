"""Tests for PySCF SCF solver, DFT, and plugin utilities."""

# --
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --

from .pyscf_helpers import *  # noqa: F401, F403

class TestPyscfPlugin:
    """Test class for PySCF plugin functionality."""

    def test_pyscf_plugin_registration(self):
        """Test that PySCF plugin is properly registered."""
        available_solvers = algorithms.available("scf_solver")
        assert "pyscf" in available_solvers

        available_localizers = algorithms.available("orbital_localizer")
        assert "pyscf_multi" in available_localizers

        available_selectors = algorithms.available("active_space_selector")
        assert "pyscf_avas" in available_selectors

        available_stability_checkers = algorithms.available("stability_checker")
        assert "pyscf" in available_stability_checkers

    def test_pyscf_scf_solver_creation(self):
        """Test creating PySCF SCF solver."""
        scf_solver = algorithms.create("scf_solver", "pyscf")
        assert scf_solver is not None

    def test_pyscf_localizer_creation(self):
        """Test creating PySCF localizer."""
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")
        assert localizer is not None

    def test_pyscf_avas_selector_creation(self):
        """Test creating PySCF AVAS Active Space Selector."""
        avas_selector = algorithms.create("active_space_selector", "pyscf_avas")
        assert avas_selector is not None

    def test_pyscf_cc_calculator(self):
        """Test creating PySCF Coupled Cluster module."""
        cc = algorithms.create("dynamical_correlation_calculator", "pyscf_coupled_cluster")
        assert cc is not None

    def test_pyscf_stability_checker_creation(self):
        """Test creating PySCF stability checker."""
        stability_checker = algorithms.create("stability_checker", "pyscf")
        assert stability_checker is not None

    def test_pyscf_scf_solver_settings(self):
        """Test PySCF SCF solver settings interface."""
        scf_solver = algorithms.create("scf_solver", "pyscf")
        settings = scf_solver.settings()

        # Test that settings object exists
        assert settings is not None

        assert settings.get("max_iterations") == 50

        # Test setting max iterations
        settings.set("max_iterations", 100)
        assert settings.get("max_iterations") == 100

        # Test setting other parameters
        settings.set("scf_type", "restricted")
        assert settings.get("scf_type") == "restricted"

    def test_pyscf_localizer_settings(self):
        """Test PySCF localizer settings interface."""
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")
        settings = localizer.settings()

        # Test that settings object exists
        assert settings is not None

        # Test default values
        assert settings.get("method") == "pipek-mezey"
        assert settings.get("population_method") == "mulliken"

        # Test setting method
        settings.set("method", "foster-boys")
        assert settings.get("method") == "foster-boys"

        # Test setting population method
        settings.set("population_method", "lowdin")
        assert settings.get("population_method") == "lowdin"

    def test_pyscf_avas_selector_settings(self):
        """Test PySCF AVAS selector settings interface."""
        avas_selector = algorithms.create("active_space_selector", "pyscf_avas")
        settings = avas_selector.settings()

        # Test that settings object exists
        assert settings is not None

        ao_labels = settings.get("ao_labels")
        assert len(ao_labels) == 0

        canonicalize = settings.get("canonicalize")
        assert canonicalize is False

        ref_labels = ["1s", "2s", "2p"]
        settings.set("ao_labels", ref_labels)
        ao_labels = settings.get("ao_labels")
        assert ao_labels == ref_labels

    def test_pyscf_cc_settings(self):
        """Test PySCF Coupled Cluster settings interface."""
        cc = algorithms.create("dynamical_correlation_calculator", "pyscf_coupled_cluster")
        settings = cc.settings()

        # Test that settings object exists
        assert settings is not None

        # Since the (T) setting was removed, these assertions are no longer valid
        # Instead, let's test that settings is a proper Settings object
        assert isinstance(settings, Settings)

        # Add a test setting
        settings.set("conv_tol", 1e-8)
        assert settings.get("conv_tol") == 1e-8

    def test_pyscf_stability_checker_settings(self):
        """Test PySCF stability checker settings interface."""
        stability_checker = algorithms.create("stability_checker", "pyscf")
        settings = stability_checker.settings()

        # Test that settings object exists
        assert settings is not None

        # Test default settings
        assert settings.get("internal") is True
        assert settings.get("external") is True
        assert settings.get("with_symmetry") is False
        assert settings.get("nroots") == 3
        assert settings.get("davidson_tolerance") == 1e-8
        assert settings.get("stability_tolerance") == -1e-4

        # Test setting parameters
        settings.set("internal", False)
        assert settings.get("internal") is False

        settings.set("nroots", 5)
        assert settings.get("nroots") == 5

        settings.set("stability_tolerance", 1e-6)
        assert settings.get("stability_tolerance") == 1e-6

    def test_pyscf_water_scf_def2svp(self):
        """Test PySCF SCF solver on water molecule with def2-svp basis."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        energy, wavefunction = scf_solver.run(water, 0, 1, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable results
        assert isinstance(energy, float)
        assert np.isclose(
            energy, -75.9229032346701, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )
        assert orbitals is not None
        assert orbitals.has_basis_set()
        assert orbitals.has_overlap_matrix()
        assert orbitals.is_restricted()

        # Check occupations
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        assert np.isclose(
            np.sum(occ_a), 5.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.isclose(
            np.sum(occ_b), 5.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

        # Check basis set populated
        basis_set = orbitals.get_basis_set()
        assert basis_set is not None

    def test_pyscf_water_scf_def2tzvp(self):
        """Test PySCF SCF solver on water molecule with def2-tzvp basis."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        energy, wavefunction = scf_solver.run(water, 0, 1, "def2-tzvp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable results
        assert isinstance(energy, float)
        assert np.isclose(
            energy, -76.02057765181318, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )
        assert orbitals is not None
        assert orbitals.has_basis_set()
        assert orbitals.has_overlap_matrix()
        assert orbitals.is_restricted()

        # Check occupations
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        assert np.isclose(
            np.sum(occ_a), 5.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.isclose(
            np.sum(occ_b), 5.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

        # Check basis set populated
        basis_set = orbitals.get_basis_set()
        assert basis_set is not None

    def test_pyscf_li_scf_def2svp(self):
        """Test PySCF SCF solver on lithium atom with def2-svp basis."""
        lithium = create_li_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        energy, wavefunction = scf_solver.run(lithium, 0, 2, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable results
        assert isinstance(energy, float)
        assert np.isclose(energy, -7.4250663561, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance)
        assert orbitals is not None
        assert orbitals.has_basis_set()
        assert orbitals.has_overlap_matrix()
        assert orbitals.is_restricted() is False

        # Check occupations
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        assert np.isclose(
            np.sum(occ_a), 2.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.isclose(
            np.sum(occ_b), 1.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_pyscf_li_rohf_def2svp(self):
        """Test PySCF SCF solver on lithium atom with ROHF/def2-svp basis."""
        lithium = create_li_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("scf_type", "restricted")

        energy, wavefunction = scf_solver.run(lithium, 0, 2, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable results
        assert isinstance(energy, float)
        assert np.isclose(
            energy, -7.42506404463744, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )
        assert orbitals is not None
        assert orbitals.has_basis_set()
        assert orbitals.has_overlap_matrix()
        assert orbitals.is_restricted()

        # Check occupations
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        assert np.isclose(
            np.sum(occ_a), 2.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.isclose(
            np.sum(occ_b), 1.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_pyscf_li_plus_scf_def2svp(self):
        """Test PySCF SCF solver on lithium ion (Li+) with def2-svp basis."""
        lithium = create_li_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        energy, wavefunction = scf_solver.run(lithium, 1, 1, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable results
        assert isinstance(energy, float)
        assert np.isclose(
            energy, -7.23289811389006, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )
        assert orbitals is not None
        assert orbitals.has_basis_set()
        assert orbitals.has_overlap_matrix()
        assert orbitals.is_restricted()

        # Check occupations
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        assert np.isclose(
            np.sum(occ_a), 1.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.isclose(
            np.sum(occ_b), 1.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_pyscf_o2_triplet_scf_def2svp(self):
        """Test PySCF SCF solver on O2 triplet state with def2-svp basis."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        energy, wavefunction = scf_solver.run(o2, 0, 3, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable results
        assert isinstance(energy, float)
        assert np.isclose(
            energy, -149.49029917454197, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )
        assert orbitals is not None
        assert orbitals.has_basis_set()
        assert orbitals.has_overlap_matrix()
        assert orbitals.is_restricted() is False

        # Check occupations
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        assert np.isclose(
            np.sum(occ_a), 9.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.isclose(
            np.sum(occ_b), 7.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_pyscf_water_dft_b3lyp_def2svp(self):
        """Test PySCF DFT solver on water molecule with B3LYP/def2-svp."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("method", "b3lyp")

        energy, wavefunction = scf_solver.run(water, 0, 1, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable DFT results
        assert isinstance(energy, float)
        assert np.isclose(
            energy, -76.33342033646656, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )
        assert orbitals is not None
        assert orbitals.has_basis_set()
        assert orbitals.has_overlap_matrix()
        assert orbitals.is_restricted()

        # Check occupations
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        assert np.isclose(
            np.sum(occ_a), 5.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.isclose(
            np.sum(occ_b), 5.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_pyscf_water_dft_pbe_def2svp(self):
        """Test PySCF DFT solver on water molecule with PBE/def2-svp."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("method", "pbe")

        energy, wavefunction = scf_solver.run(water, 0, 1, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable DFT results
        assert isinstance(energy, float)
        assert np.isclose(
            energy, -76.2511269787294, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )
        assert orbitals is not None
        assert orbitals.has_basis_set()
        assert orbitals.has_overlap_matrix()
        assert orbitals.is_restricted()

    def test_pyscf_li_dft_b3lyp_def2svp(self):
        """Test PySCF DFT solver on lithium atom with B3LYP/def2-svp (UKS)."""
        lithium = create_li_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("method", "b3lyp")

        energy, wavefunction = scf_solver.run(lithium, 0, 2, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable DFT results
        assert isinstance(energy, float)
        assert np.isclose(
            energy, -7.484980651804635, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )
        assert orbitals is not None
        assert orbitals.has_basis_set()
        assert orbitals.has_overlap_matrix()
        assert orbitals.is_restricted() is False  # Should be UKS (unrestricted)

        # Check occupations
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        assert np.isclose(
            np.sum(occ_a), 2.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.isclose(
            np.sum(occ_b), 1.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_pyscf_li_dft_roks_b3lyp_def2svp(self):
        """Test PySCF DFT solver on lithium atom with ROKS B3LYP/def2-svp."""
        lithium = create_li_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("method", "b3lyp")
        scf_solver.settings().set("scf_type", "restricted")

        energy, wavefunction = scf_solver.run(lithium, 0, 2, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable ROKS DFT results
        assert isinstance(energy, float)
        assert np.isclose(
            energy, -7.484979697016255, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )
        assert orbitals is not None
        assert orbitals.has_basis_set()
        assert orbitals.has_overlap_matrix()
        assert orbitals.is_restricted()  # Should be restricted (ROKS)

        # Check occupations
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        assert np.isclose(
            np.sum(occ_a), 2.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.isclose(
            np.sum(occ_b), 1.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_pyscf_o2_triplet_dft_b3lyp_def2svp(self):
        """Test PySCF DFT solver on O2 triplet state with B3LYP/def2-svp."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("method", "b3lyp")

        energy, wavefunction = scf_solver.run(o2, 0, 3, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable DFT results
        assert isinstance(energy, float)
        assert np.isclose(
            energy, -150.204697358644, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )
        assert orbitals is not None
        assert orbitals.has_basis_set()
        assert orbitals.has_overlap_matrix()
        assert orbitals.is_restricted() is False  # Should be UKS
        assert orbitals.is_unrestricted()

        # Check occupations
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        assert np.isclose(
            np.sum(occ_a), 9.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.isclose(
            np.sum(occ_b), 7.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_pyscf_dft_method_case_insensitive(self):
        """Test that DFT method names are case insensitive."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        # Test uppercase
        scf_solver.settings().set("method", "B3LYP")
        energy_upper, _ = scf_solver.run(water, 0, 1, "sto-3g")

        # Test lowercase
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("method", "b3lyp")
        energy_lower, _ = scf_solver.run(water, 0, 1, "sto-3g")

        # Should give the same result
        assert np.isclose(
            energy_upper, energy_lower, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )

    def test_pyscf_uo2_lanl2dz(self):
        """Test PySCF SCF solver on UO2 with LANL2DZ basis and ECP."""
        uo2 = create_uo2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        energy, _ = scf_solver.run(uo2, 0, 1, "lanl2dz")
        ref_energy = -200.29749139183
        assert np.isclose(energy, ref_energy, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance)

    def test_pyscf_scf_solver_initial_guess_restart(self):
        """Test PySCF SCF solver with initial guess from converged orbitals."""
        # Water as restricted test
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("method", "hf")

        # First calculation - let it converge normally
        energy_first, wfn_first = scf_solver.run(water, 0, 1, "def2-tzvp")
        orbitals_first = wfn_first.get_orbitals()

        # Verify we get the expected energy for HF/def2-tzvp
        assert np.isclose(
            energy_first, -76.0205776518, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )

        # Now restart with the converged orbitals as initial guess
        # Create a new solver instance since settings are locked after run
        scf_solver2 = algorithms.create("scf_solver", "pyscf")
        scf_solver2.settings().set("method", "hf")
        scf_solver2.settings().set("max_iterations", 2)  # 2 is minimum as need to check energy difference

        # Second calculation with initial guess
        energy_second, _ = scf_solver2.run(water, 0, 1, orbitals_first)

        # Should get the same energy (within tight tolerance)
        assert np.isclose(
            energy_first, energy_second, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )

        # Oxygen Triplet Initial Guess Test
        o2 = create_o2_structure()
        scf_solver3 = algorithms.create("scf_solver", "pyscf")
        scf_solver3.settings().set("method", "hf")

        # First calculation - let triplet converge normally
        energy_o2_first, wfn_o2_first = scf_solver3.run(o2, 0, 3, "sto-3g")
        orbitals_o2_first = wfn_o2_first.get_orbitals()

        # Verify we get the expected energy for HF/STO-3G triplet
        assert np.isclose(
            energy_o2_first, -147.633969608498, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )

        # Now restart with the converged orbitals as initial guess
        # Create a new solver instance since settings are locked after run
        scf_solver4 = algorithms.create("scf_solver", "pyscf")
        scf_solver4.settings().set("method", "hf")
        scf_solver4.settings().set("max_iterations", 2)  # 2 is minimum as need to check energy difference

        # Second calculation with initial guess
        energy_o2_second, _ = scf_solver4.run(o2, 0, 3, orbitals_o2_first)

        # Should get the same energy (within tight tolerance)
        assert np.isclose(
            energy_o2_first, energy_o2_second, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )

    def test_pyscf_occupations_from_n_electrons_and_multiplicity(self):
        """Test occupations from n_electrons and multiplicity on water with def2-svp basis."""
        # Get orbitals and Hamiltonian
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        _, wavefunction = scf_solver.run(water, 0, 1, "def2-svp")
        orbitals = wavefunction.get_orbitals()
        hamiltonian_calculator = algorithms.create("hamiltonian_constructor")
        hamiltonian = hamiltonian_calculator.run(orbitals)

        # Check orbitals to SCF for singlet state
        occupation_singlet = [np.concatenate((np.ones(5), np.zeros(19))), np.concatenate((np.ones(5), np.zeros(19)))]
        scf_1 = orbitals_to_scf(orbitals, occupation_singlet[0], occupation_singlet[1])
        scf_2 = orbitals_to_scf_from_n_electrons_and_multiplicity(orbitals, 10, 1)
        assert np.allclose(
            scf_1.mo_occ,
            scf_2.mo_occ,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Check orbitals to SCF for doublet state
        occupation_doublet = [np.concatenate((np.ones(6), np.zeros(18))), np.concatenate((np.ones(5), np.zeros(19)))]
        scf_1 = orbitals_to_scf(orbitals, occupation_doublet[0], occupation_doublet[1])
        scf_2 = orbitals_to_scf_from_n_electrons_and_multiplicity(orbitals, 11, 2)
        assert np.allclose(
            scf_1.mo_occ,
            scf_2.mo_occ,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Check orbitals to SCF for triplet state
        occupation_triplet = [np.concatenate((np.ones(6), np.zeros(18))), np.concatenate((np.ones(4), np.zeros(20)))]
        scf_1 = orbitals_to_scf(orbitals, occupation_triplet[0], occupation_triplet[1])
        scf_2 = orbitals_to_scf_from_n_electrons_and_multiplicity(orbitals, 10, 3)
        assert np.allclose(
            scf_1.mo_occ,
            scf_2.mo_occ,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Check Hamiltonian to SCF for singlet state
        scf_1 = hamiltonian_to_scf(hamiltonian, occupation_singlet[0], occupation_singlet[1])
        scf_2 = hamiltonian_to_scf_from_n_electrons_and_multiplicity(hamiltonian, 10, 1)
        assert np.allclose(
            scf_1.mo_occ,
            scf_2.mo_occ,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_orbitals_to_scf_charge_and_multiplicity_handling(self):
        """Test that orbitals_to_scf correctly sets charge and multiplicity in the PySCF molecule."""
        scf_solver = algorithms.create("scf_solver", "pyscf")

        # Test singlet state (closed-shell)
        water = create_water_structure()
        _, wavefunction_singlet = scf_solver.run(water, 0, 1, "sto-3g")
        orbitals_singlet = wavefunction_singlet.get_orbitals()
        n_orbitals_singlet = orbitals_singlet.get_num_molecular_orbitals()

        # Singlet: 5 alpha, 5 beta electrons (charge = 0, multiplicity = 1)
        occ_alpha_singlet = np.concatenate((np.ones(5), np.zeros(n_orbitals_singlet - 5)))
        occ_beta_singlet = np.concatenate((np.ones(5), np.zeros(n_orbitals_singlet - 5)))
        scf_singlet = orbitals_to_scf(orbitals_singlet, occ_alpha_singlet, occ_beta_singlet)

        assert scf_singlet.mol.charge == 0
        assert scf_singlet.mol.spin == 0
        assert scf_singlet.mol.multiplicity == 1

        # Test doublet state (open-shell)
        lithium = create_li_structure()
        _, wavefunction_doublet = scf_solver.run(lithium, 0, 2, "sto-3g")
        orbitals_doublet = wavefunction_doublet.get_orbitals()
        n_orbitals_doublet = orbitals_doublet.get_num_molecular_orbitals()

        # Doublet: 2 alpha, 1 beta electrons (charge = 0, multiplicity = 2)
        occ_alpha_doublet = np.concatenate((np.ones(2), np.zeros(n_orbitals_doublet - 2)))
        occ_beta_doublet = np.concatenate((np.ones(1), np.zeros(n_orbitals_doublet - 1)))
        scf_doublet = orbitals_to_scf(orbitals_doublet, occ_alpha_doublet, occ_beta_doublet)

        assert scf_doublet.mol.charge == 0
        assert scf_doublet.mol.spin == 1
        assert scf_doublet.mol.multiplicity == 2

        # Test triplet state (open-shell)
        o2 = create_o2_structure()
        _, wavefunction_triplet = scf_solver.run(o2, 0, 3, "sto-3g")
        orbitals_triplet = wavefunction_triplet.get_orbitals()
        n_orbitals_triplet = orbitals_triplet.get_num_molecular_orbitals()

        # Triplet: 9 alpha, 7 beta electrons (charge = 0, multiplicity = 3)
        occ_alpha_triplet = np.concatenate((np.ones(9), np.zeros(n_orbitals_triplet - 9)))
        occ_beta_triplet = np.concatenate((np.ones(7), np.zeros(n_orbitals_triplet - 7)))
        scf_triplet = orbitals_to_scf(orbitals_triplet, occ_alpha_triplet, occ_beta_triplet)

        assert scf_triplet.mol.charge == 0
        assert scf_triplet.mol.spin == 2
        assert scf_triplet.mol.multiplicity == 3

        # Test cation (open-shell)
        water = create_water_structure()
        _, wavefunction_cation = scf_solver.run(water, 1, 2, "sto-3g")
        orbitals_cation = wavefunction_cation.get_orbitals()
        n_orbitals_cation = orbitals_cation.get_num_molecular_orbitals()

        # Doublet: 5 alpha, 4 beta electrons (charge = 1, multiplicity = 2)
        occ_alpha_cation = np.concatenate((np.ones(5), np.zeros(n_orbitals_cation - 5)))
        occ_beta_cation = np.concatenate((np.ones(4), np.zeros(n_orbitals_cation - 4)))
        scf_cation = orbitals_to_scf(orbitals_cation, occ_alpha_cation, occ_beta_cation)

        assert scf_cation.mol.charge == 1
        assert scf_cation.mol.spin == 1
        assert scf_cation.mol.multiplicity == 2
        assert scf_cation.mol.nelectron == 9

        # Test with ECP electrons
        ag = Structure(["Ag"], np.array([[0.0, 0.0, 0.0]]))
        _, wavefunction_ecp = scf_solver.run(ag, 0, 2, "lanl2dz")
        orbitals_ecp = wavefunction_ecp.get_orbitals()
        n_orbitals_ecp = orbitals_ecp.get_num_molecular_orbitals()

        # Doublet: 10 alpha, 9 beta electrons (charge = 0, multiplicity = 2)
        occ_alpha_ecp = np.concatenate((np.ones(10), np.zeros(n_orbitals_ecp - 10)))
        occ_beta_ecp = np.concatenate((np.ones(9), np.zeros(n_orbitals_ecp - 9)))
        scf_ecp = orbitals_to_scf(orbitals_ecp, occ_alpha_ecp, occ_beta_ecp)

        assert hasattr(scf_ecp.mol, "ecp")
        assert scf_ecp.mol.ecp
        assert scf_ecp.mol.charge == 0
        assert scf_ecp.mol.spin == 1
        assert scf_ecp.mol.multiplicity == 2
        assert scf_ecp.mol.nelectron == 19

    def test_hamiltonian_to_scf_rerouting_and_error_handling(self):
        """Test hamiltonian_to_scf rerouting and error handling.

        This test validates three scenarios:
        1. Rerouting: Non-model Hamiltonians should reroute to orbitals_to_scf
        2. Error throwing: Invalid configurations should raise ValueError
        3. Non-rerouting: Valid model Hamiltonians should create fake SCF objects
        """
        # 1: Rerouting for non-model Hamiltonian
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        _, wavefunction = scf_solver.run(water, 0, 1, "sto-3g")
        orbitals = wavefunction.get_orbitals()

        # Create a Hamiltonian from these orbitals (non-model)
        hamiltonian_calculator = algorithms.create("hamiltonian_constructor")
        hamiltonian = hamiltonian_calculator.run(orbitals)

        # Verify the orbitals have coefficients (non-model)
        coeff_a, coeff_b = orbitals.get_coefficients()
        assert coeff_a is not None
        assert coeff_b is not None

        # Create occupation arrays for a singlet state
        norb = orbitals.get_num_molecular_orbitals()
        occupation_alpha = np.concatenate((np.ones(5), np.zeros(norb - 5)))
        occupation_beta = np.concatenate((np.ones(5), np.zeros(norb - 5)))

        # Call hamiltonian_to_scf - should re-route to orbitals_to_scf
        scf_from_hamiltonian = hamiltonian_to_scf(hamiltonian, occupation_alpha, occupation_beta)

        # MO coefficients should NOT be identity matrix
        assert not np.allclose(scf_from_hamiltonian.mo_coeff, np.eye(norb)), (
            "MO coefficients should not be identity matrix for non-model Hamiltonian (rerouted case)"
        )

        # 2. Unrestricted model Hamiltonian should throw
        model_orbitals_unrestricted = data.ModelOrbitals(4, False)  # unrestricted
        one_body_alpha = np.eye(4)
        one_body_beta = np.eye(4) * 1.1
        two_body_aaaa = np.zeros(4**4)
        two_body_aabb = np.zeros(4**4)
        two_body_bbbb = np.zeros(4**4)
        h_unrestricted_model = data.Hamiltonian(
            data.CanonicalFourCenterHamiltonianContainer(
                one_body_alpha,
                one_body_beta,
                two_body_aaaa,
                two_body_aabb,
                two_body_bbbb,
                model_orbitals_unrestricted,
                0.0,
                np.eye(4),
                np.eye(4),
            )
        )

        occupation_alpha_test = np.array([1.0, 1.0, 0.0, 0.0])
        occupation_beta_test = np.array([1.0, 1.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="You cannot pass an unrestricted model Hamiltonian here"):
            hamiltonian_to_scf(h_unrestricted_model, occupation_alpha_test, occupation_beta_test)

        # 3. Non-rerouting for valid model Hamiltonian

        # Create a model Hamiltonian (restricted, closed-shell, full active space)
        model_orbitals_proper = data.ModelOrbitals(4, True)  # All orbitals are active by default
        one_body_model = np.eye(4) * 0.5
        two_body_model = np.zeros(4**4)
        h_model = data.Hamiltonian(
            data.CanonicalFourCenterHamiltonianContainer(
                one_body_model, two_body_model, model_orbitals_proper, 0.5, np.eye(4)
            )
        )

        # Closed-shell occupations
        occupation_alpha_closed = np.array([1.0, 1.0, 0.0, 0.0])
        occupation_beta_closed = np.array([1.0, 1.0, 0.0, 0.0])

        # Call hamiltonian_to_scf - should create fake SCF
        scf_from_model = hamiltonian_to_scf(h_model, occupation_alpha_closed, occupation_beta_closed)

        # For model Hamiltonian - MO coefficients should be identity matrix (fake SCF)
        assert np.allclose(scf_from_model.mo_coeff, np.eye(4)), (
            "MO coefficients should be identity matrix for model Hamiltonian (fake SCF object)"
        )

        # Verify core energy matches
        assert np.isclose(scf_from_model.energy_nuc(), 0.5), "Core energy should match the model Hamiltonian"

        # Verify occupations are set correctly (total = alpha + beta for restricted)
        expected_total_occ = occupation_alpha_closed + occupation_beta_closed
        assert np.allclose(scf_from_model.mo_occ, expected_total_occ), (
            "Occupations should be correctly set in fake SCF object"
        )

        # Verify electron count
        assert scf_from_model.mol.nelectron == 4, "Total electron count should be 4"

