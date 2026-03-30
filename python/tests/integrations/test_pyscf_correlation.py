"""Tests for PySCF correlation methods (CCSD, MCSCF, AVAS)."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from .pyscf_helpers import *  # noqa: F401, F403


class TestPyscfCorrelation:
    """PySCF correlation method tests."""


    def test_pyscf_avas_selector_water_def2svp(self):
        """Test PySCF AVAS selector on water molecule with def2-svp basis."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        _, wavefunction = scf_solver.run(water, 0, 1, "def2-svp")

        # Select active space using AVAS
        avas_selector = algorithms.create("active_space_selector", "pyscf_avas")
        avas_selector.settings().set("ao_labels", ["O 2s", "O 2p", "H 1s"])
        active_wfn = avas_selector.run(wavefunction)

        act_a, act_b = active_wfn.get_orbitals().get_active_space_indices()
        assert act_a == act_b
        assert act_a == [0, 1, 2, 3, 4]

        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        assert np.isclose(
            np.sum(occ_a), 5.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.allclose(
            occ_a, occ_b, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_pyscf_avas_selector_o2_triplet_def2svp(self):
        """Test PySCF AVAS selector on O2 molecule (triplet ROHF) with def2-svp basis."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("scf_type", "restricted")

        _, wavefunction = scf_solver.run(o2, 0, 3, "def2-svp")

        # Select active space using AVAS
        avas_selector = algorithms.create("active_space_selector", "pyscf_avas")
        avas_selector.settings().set("ao_labels", ["O 2s", "O 2p"])
        active_wfn = avas_selector.run(wavefunction)

        act_a, act_b = active_wfn.get_orbitals().get_active_space_indices()
        assert act_a == act_b
        assert act_a == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_pyscf_ccsd_water_def2svp(self):
        """Test PySCF CCSD on water with def2-svp basis."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        _, wavefunction = scf_solver.run(water, 0, 1, "def2-svp")

        ham_calculator = algorithms.create("hamiltonian_constructor", "qdk")
        hamiltonian = ham_calculator.run(wavefunction.get_orbitals())

        # Compute CC energy
        cc_calculator = algorithms.create("dynamical_correlation_calculator", "pyscf_coupled_cluster")
        cc_calculator.settings().set("store_amplitudes", True)
        ansatz_object = Ansatz(hamiltonian, wavefunction)
        cc_energy, updated_wavefunction, _ = cc_calculator.run(ansatz_object)
        reference_energy = -76.14613724756676
        assert np.isclose(cc_energy, reference_energy), f"{cc_energy=} should match total energy {reference_energy=}"

        # Get amplitudes from the wavefunction container
        cc_container = updated_wavefunction.get_container()
        assert cc_container.has_t1_amplitudes()
        assert cc_container.has_t2_amplitudes()
        t1_amplitudes = cc_container.get_t1_amplitudes()
        t2_amplitudes = cc_container.get_t2_amplitudes()
        assert t1_amplitudes is not None
        assert t2_amplitudes is not None

    def test_pyscf_uccsd_o2_triplet_def2svp(self):
        """Test PySCF UCCSD on O2 triplet with def2-svp basis."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        _, wavefunction = scf_solver.run(o2, 0, 3, "def2-svp")

        # Verify we have unrestricted orbitals
        orbitals = wavefunction.get_orbitals()
        assert orbitals.is_unrestricted(), "O2 triplet should have unrestricted orbitals"
        assert wavefunction.get_container_type() == "sd"
        assert wavefunction.size() == 1, "single determinant"

        # Create Hamiltonian
        ham_calculator = algorithms.create("hamiltonian_constructor", "qdk")
        hamiltonian = ham_calculator.run(orbitals)

        # Compute UCCSD energy
        cc_calculator = algorithms.create("dynamical_correlation_calculator", "pyscf_coupled_cluster")
        cc_calculator.settings().set("store_amplitudes", True)
        ansatz_object = Ansatz(hamiltonian, wavefunction)
        cc_energy, updated_wavefunction, _ = cc_calculator.run(ansatz_object)
        reference_energy = -149.8417973596817
        assert np.isclose(cc_energy, reference_energy), (
            f"cc energy {cc_energy} should match reference {reference_energy}"
        )

        # Get amplitudes from the wavefunction container
        cc_container = updated_wavefunction.get_container()
        assert cc_container.has_t1_amplitudes(), "should have T1 amplitudes"
        assert cc_container.has_t2_amplitudes(), "should have T2 amplitudes"

        # For unrestricted, we should get separate alpha and beta amplitudes
        t1_alpha, t1_beta = cc_container.get_t1_amplitudes()
        t2_abab, t2_aaaa, t2_bbbb = cc_container.get_t2_amplitudes()

        # Verify all amplitudes are present
        assert t1_alpha is not None, "T1 alpha amplitudes should not be None"
        assert t1_beta is not None, "T1 beta amplitudes should not be None"
        assert t2_abab is not None, "T2 alpha-beta amplitudes should not be None"
        assert t2_aaaa is not None, "T2 alpha-alpha amplitudes should not be None"
        assert t2_bbbb is not None, "T2 beta-beta amplitudes should not be None"

        # Verify the amplitudes have the expected shapes (stored as column vectors)
        # T1: (nocc * nvirt, 1) for each spin
        # T2: (nocc * nocc * nvirt * nvirt, 1) for each component

        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        nocc_alpha = int(np.sum(occ_a))
        nocc_beta = int(np.sum(occ_b))
        nvirt_alpha = orbitals.get_num_molecular_orbitals() - nocc_alpha
        nvirt_beta = orbitals.get_num_molecular_orbitals() - nocc_beta

        t1_alpha_array = np.array(t1_alpha) if not isinstance(t1_alpha, np.ndarray) else t1_alpha
        t1_beta_array = np.array(t1_beta) if not isinstance(t1_beta, np.ndarray) else t1_beta

        assert t1_alpha_array.shape[0] == nocc_alpha * nvirt_alpha, "T1 alpha shape mismatch"
        assert t1_beta_array.shape[0] == nocc_beta * nvirt_beta, "T1 beta shape mismatch"

    def test_pyscf_uccsd_wavefunction_serialization_roundtrip(self, tmp_path):
        """Test that UCCSD wavefunction can be serialized and deserialized via JSON and HDF5."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        _, wavefunction = scf_solver.run(o2, 0, 3, "sto-3g")

        ham_calculator = algorithms.create("hamiltonian_constructor", "qdk")
        hamiltonian = ham_calculator.run(wavefunction.get_orbitals())

        # Compute UCCSD energy with amplitudes stored
        cc_calculator = algorithms.create("dynamical_correlation_calculator", "pyscf_coupled_cluster")
        cc_calculator.settings().set("store_amplitudes", True)
        ansatz_object = Ansatz(hamiltonian, wavefunction)
        _, cc_wavefunction, _ = cc_calculator.run(ansatz_object)

        # Verify original wavefunction properties
        assert cc_wavefunction.get_container_type() == "coupled_cluster"

        # Get original container and check it has amplitudes
        original_container = cc_wavefunction.get_container()
        assert original_container.has_t1_amplitudes()
        assert original_container.has_t2_amplitudes()

        # Get original amplitudes - unrestricted has separate alpha/beta
        orig_t1_alpha, orig_t1_beta = original_container.get_t1_amplitudes()
        orig_t2_abab, orig_t2_aaaa, orig_t2_bbbb = original_container.get_t2_amplitudes()

        # Verify all amplitudes are present
        assert orig_t1_alpha is not None
        assert orig_t1_beta is not None
        assert orig_t2_abab is not None
        assert orig_t2_aaaa is not None
        assert orig_t2_bbbb is not None

        # Get original orbitals properties
        orig_orbitals = cc_wavefunction.get_orbitals()
        orig_num_orbs = orig_orbitals.get_num_molecular_orbitals()
        orig_is_unrestricted = orig_orbitals.is_unrestricted()

        orig_num_elec = cc_wavefunction.get_total_num_electrons()

        # Test 1: JSON serialization
        wf_json = cc_wavefunction.to_json()
        restored_json = data.Wavefunction.from_json(wf_json)

        # Verify JSON restored wavefunction
        assert restored_json.get_container_type() == "coupled_cluster"

        json_container = restored_json.get_container()
        assert json_container.has_t1_amplitudes()
        assert json_container.has_t2_amplitudes()

        # Verify amplitudes are preserved - unrestricted version
        json_t1_alpha, json_t1_beta = json_container.get_t1_amplitudes()
        json_t2_abab, json_t2_aaaa, json_t2_bbbb = json_container.get_t2_amplitudes()

        assert np.allclose(
            np.array(orig_t1_alpha),
            np.array(json_t1_alpha),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            np.array(orig_t1_beta),
            np.array(json_t1_beta),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            np.array(orig_t2_abab),
            np.array(json_t2_abab),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            np.array(orig_t2_aaaa),
            np.array(json_t2_aaaa),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            np.array(orig_t2_bbbb),
            np.array(json_t2_bbbb),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Verify orbitals properties preserved
        json_orbitals = restored_json.get_orbitals()
        assert json_orbitals.get_num_molecular_orbitals() == orig_num_orbs
        assert json_orbitals.is_unrestricted() == orig_is_unrestricted

        # Verify electron count preserved
        json_num_elec = restored_json.get_total_num_electrons()
        assert json_num_elec == orig_num_elec

        # Test 2: HDF5 serialization
        filename = tmp_path / "uccsd_wf.wavefunction.hdf5"
        cc_wavefunction.to_hdf5_file(str(filename))
        restored_hdf5 = data.Wavefunction.from_hdf5_file(str(filename))

        # Verify HDF5 restored wavefunction
        assert restored_hdf5.get_container_type() == "coupled_cluster"

        hdf5_container = restored_hdf5.get_container()
        assert hdf5_container.has_t1_amplitudes()
        assert hdf5_container.has_t2_amplitudes()

        # Verify amplitudes are preserved - unrestricted version
        hdf5_t1_alpha, hdf5_t1_beta = hdf5_container.get_t1_amplitudes()
        hdf5_t2_abab, hdf5_t2_aaaa, hdf5_t2_bbbb = hdf5_container.get_t2_amplitudes()

        assert np.allclose(
            np.array(orig_t1_alpha),
            np.array(hdf5_t1_alpha),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            np.array(orig_t1_beta),
            np.array(hdf5_t1_beta),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            np.array(orig_t2_abab),
            np.array(hdf5_t2_abab),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            np.array(orig_t2_aaaa),
            np.array(hdf5_t2_aaaa),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            np.array(orig_t2_bbbb),
            np.array(hdf5_t2_bbbb),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Verify orbitals properties preserved
        hdf5_orbitals = restored_hdf5.get_orbitals()
        assert hdf5_orbitals.get_num_molecular_orbitals() == orig_num_orbs
        assert hdf5_orbitals.is_unrestricted() == orig_is_unrestricted

        # Verify electron count preserved
        hdf5_num_elec = restored_hdf5.get_total_num_electrons()
        assert hdf5_num_elec == orig_num_elec

    def test_pyscf_ccsd_wavefunction_serialization_roundtrip(self, tmp_path):
        """Test that CCSD wavefunction can be serialized and deserialized with json and hdf5."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver")
        _, wavefunction = scf_solver.run(water, 0, 1, "sto-3g")

        ham_calculator = algorithms.create("hamiltonian_constructor", "qdk")
        hamiltonian = ham_calculator.run(wavefunction.get_orbitals())

        # Compute CC energy with amplitudes stored
        cc_calculator = algorithms.create("dynamical_correlation_calculator", "pyscf_coupled_cluster")
        cc_calculator.settings().set("store_amplitudes", True)
        ansatz_object = Ansatz(hamiltonian, wavefunction)
        _, cc_wavefunction, _ = cc_calculator.run(ansatz_object)

        # Verify original wavefunction properties
        assert cc_wavefunction.get_container_type() == "coupled_cluster"

        # Get original container and check it has amplitudes
        original_container = cc_wavefunction.get_container()
        assert original_container.has_t1_amplitudes()
        assert original_container.has_t2_amplitudes()

        # Get original amplitudes
        orig_t1 = original_container.get_t1_amplitudes()
        orig_t2 = original_container.get_t2_amplitudes()
        assert orig_t1 is not None
        assert orig_t2 is not None

        # Get original orbitals properties
        orig_orbitals = cc_wavefunction.get_orbitals()
        orig_num_orbs = orig_orbitals.get_num_molecular_orbitals()
        orig_is_restricted = orig_orbitals.is_restricted()

        orig_num_elec = cc_wavefunction.get_total_num_electrons()

        # Test 1: JSON serialization
        wf_json = cc_wavefunction.to_json()
        restored_json = data.Wavefunction.from_json(wf_json)

        # Verify JSON restored wavefunction
        assert restored_json.get_container_type() == "coupled_cluster"

        json_container = restored_json.get_container()
        assert json_container.has_t1_amplitudes()
        assert json_container.has_t2_amplitudes()

        # Verify amplitudes are preserved
        json_t1 = json_container.get_t1_amplitudes()
        json_t2 = json_container.get_t2_amplitudes()
        assert np.allclose(
            np.array(orig_t1),
            np.array(json_t1),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            np.array(orig_t2),
            np.array(json_t2),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Verify orbitals properties preserved
        json_orbitals = restored_json.get_orbitals()
        assert json_orbitals.get_num_molecular_orbitals() == orig_num_orbs
        assert json_orbitals.is_restricted() == orig_is_restricted

        # Verify electron count preserved
        json_num_elec = restored_json.get_total_num_electrons()
        assert json_num_elec == orig_num_elec

        # Test 2: HDF5 serialization
        filename = tmp_path / "cc_wf.wavefunction.hdf5"
        cc_wavefunction.to_hdf5_file(str(filename))
        restored_hdf5 = data.Wavefunction.from_hdf5_file(str(filename))

        # Verify HDF5 restored wavefunction
        assert restored_hdf5.get_container_type() == "coupled_cluster"

        hdf5_container = restored_hdf5.get_container()
        assert hdf5_container.has_t1_amplitudes()
        assert hdf5_container.has_t2_amplitudes()

        # Verify amplitudes are preserved
        hdf5_t1 = hdf5_container.get_t1_amplitudes()
        hdf5_t2 = hdf5_container.get_t2_amplitudes()
        assert np.allclose(
            np.array(orig_t1),
            np.array(hdf5_t1),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            np.array(orig_t2),
            np.array(hdf5_t2),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Verify orbitals properties preserved
        hdf5_orbitals = restored_hdf5.get_orbitals()
        assert hdf5_orbitals.get_num_molecular_orbitals() == orig_num_orbs
        assert hdf5_orbitals.is_restricted() == orig_is_restricted

        # Verify electron count preserved
        hdf5_num_elec = restored_hdf5.get_total_num_electrons()
        assert hdf5_num_elec == orig_num_elec

    def test_pyscf_mcscf_singlet(self):
        """Test PySCF MCSCF for n2 with cc-pvdz basis and CAS(6,6)."""
        # Create N2 structure
        n2 = create_n2_structure()

        # Perform SCF calculation with qdk-chemistry
        scf_solver = algorithms.create("scf_solver")
        _, wavefunction = scf_solver.run(n2, 0, 1, "cc-pvdz")

        # Construct qdk-chemistry Hamiltonian for active space
        ham_calculator = algorithms.create("hamiltonian_constructor")

        # Create MACIS calculator
        macis_calc = algorithms.create("multi_configuration_calculator", "macis_cas")
        macis_calc.settings().set("calculate_one_rdm", True)
        macis_calc.settings().set("calculate_two_rdm", True)

        # Select active space: 6 orbitals, 6 electrons
        valence_selector = algorithms.create("active_space_selector", "qdk_valence")
        valence_selector.settings().set("num_active_electrons", 6)
        valence_selector.settings().set("num_active_orbitals", 6)
        active_orbitals_sd = valence_selector.run(wavefunction)

        # Calculate with qdk-chemistry/MACIS
        pyscf_mcscf = algorithms.create("multi_configuration_scf", "pyscf")
        pyscf_mcscf_energy, _ = pyscf_mcscf.run(active_orbitals_sd.get_orbitals(), ham_calculator, macis_calc, 3, 3)

        assert np.isclose(
            pyscf_mcscf_energy,
            -108.78966139913287,
            rtol=float_comparison_relative_tolerance,
            atol=mcscf_energy_tolerance,
        )

    def test_pyscf_mcscf_triplet(self):
        """Test PySCF MCSCF for o2 triplet with cc-pvdz basis and CAS(6,6)."""
        # Create O2 structure
        o2 = create_o2_structure()

        # Perform SCF calculation with qdk-chemistry
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("scf_type", "restricted")
        _, wavefunction = scf_solver.run(o2, 0, 3, "cc-pvdz")

        # Construct qdk-chemistry Hamiltonian for active space
        ham_calculator = algorithms.create("hamiltonian_constructor")

        # Create MACIS calculator
        macis_calc = algorithms.create("multi_configuration_calculator", "macis_cas")
        macis_calc.settings().set("calculate_one_rdm", True)
        macis_calc.settings().set("calculate_two_rdm", True)
        macis_calc.settings().set("ci_residual_tolerance", 1e-10)

        # Select active space: 6 orbitals, 6 electrons
        valence_selector = algorithms.create("active_space_selector", "qdk_valence")
        valence_selector.settings().set("num_active_electrons", 6)
        valence_selector.settings().set("num_active_orbitals", 6)
        active_orbitals_sd = valence_selector.run(wavefunction)

        # Calculate with qdk-chemistry/MACIS
        pyscf_mcscf = algorithms.create("multi_configuration_scf", "pyscf")
        pyscf_mcscf_energy, _ = pyscf_mcscf.run(active_orbitals_sd.get_orbitals(), ham_calculator, macis_calc, 4, 2)

        assert np.isclose(
            pyscf_mcscf_energy,
            -149.68131616317658,
            rtol=float_comparison_relative_tolerance,
            atol=mcscf_energy_tolerance,
        )

    def test_pyscf_fciwrapper_casci_singlet(self):
        """Test MC wrapper for n2 with cc-pvdz basis and CAS(6,6)."""
        # Create N2 structure
        n2 = create_n2_structure()
        pyscf_mol = pyscf.gto.M(
            atom=structure_to_pyscf_atom_labels(n2)[0], basis="cc-pvdz", unit="Bohr", charge=0, spin=0
        )

        # Perform SCF calculation with qdk-chemistry
        scf_solver = algorithms.create("scf_solver")
        _, wavefunction = scf_solver.run(n2, 0, 1, "cc-pvdz")

        # Get pyscf object from hf orbitals
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        pyscf_scf = orbitals_to_scf(wavefunction.get_orbitals(), occ_alpha=occ_a, occ_beta=occ_b, scf_type="restricted")

        # Create MACIS calculator
        macis_calc = algorithms.create("multi_configuration_calculator", "macis_cas")
        macis_calc.settings().set("calculate_one_rdm", True)
        macis_calc.settings().set("calculate_two_rdm", True)

        # Create PySCF CASCI calculation with macis
        casci = pyscf.mcscf.CASCI(pyscf_scf, 6, 6)
        casci.fcisolver = _mcsolver_to_fcisolver(pyscf_mol, macis_calc)
        casci.verbose = 0
        casci_energy = casci.kernel()[0]

        assert np.isclose(
            casci_energy, -108.74113344655625, rtol=float_comparison_relative_tolerance, atol=mcscf_energy_tolerance
        )

    def test_pyscf_fciwrapper_casci_triplet(self):
        """Test MC wrapper for o2 triplet with cc-pvdz basis and CAS(6,6)."""
        # Create O2 structure
        o2 = create_o2_structure()
        pyscf_mol = pyscf.gto.M(atom=structure_to_pyscf_atom_labels(o2)[0], basis="cc-pvdz", charge=0, spin=2)

        # Perform SCF calculation with qdk-chemistry
        scf_solver = algorithms.create("scf_solver", "pyscf")
        _, wavefunction = scf_solver.run(o2, 0, 3, "cc-pvdz")

        # Get pyscf object from hf orbitals
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        pyscf_scf = orbitals_to_scf(wavefunction.get_orbitals(), occ_alpha=occ_a, occ_beta=occ_b, scf_type="restricted")

        # Create MACIS calculator
        macis_calc = algorithms.create("multi_configuration_calculator", "macis_cas")
        macis_calc.settings().set("calculate_one_rdm", True)
        macis_calc.settings().set("calculate_two_rdm", True)

        # Create PySCF CASCI calculation with macis
        casci = pyscf.mcscf.CASCI(pyscf_scf, 8, 6)
        casci.fcisolver = _mcsolver_to_fcisolver(pyscf_mol, macis_calc)
        casci.verbose = 0
        casci_energy = casci.kernel()[0]

        assert np.isclose(
            casci_energy, -149.661310389037, rtol=float_comparison_relative_tolerance, atol=mcscf_energy_tolerance
        )

    def test_pyscf_fciwrapper_casscf_singlet(self):
        """Test MC wrapper in casscf for n2 with cc-pvdz basis and CAS(6,6)."""
        # Create N2 structure
        n2 = create_n2_structure()
        pyscf_mol = pyscf.gto.M(
            atom=structure_to_pyscf_atom_labels(n2)[0], basis="cc-pvdz", unit="Bohr", charge=0, spin=0
        )

        # Perform SCF calculation with qdk-chemistry
        scf_solver = algorithms.create("scf_solver")
        _, wavefunction = scf_solver.run(n2, 0, 1, "cc-pvdz")

        # Get pyscf object from hf orbitals
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        pyscf_scf = orbitals_to_scf(wavefunction.get_orbitals(), occ_alpha=occ_a, occ_beta=occ_b, scf_type="restricted")

        # Create MACIS calculator
        macis_calc = algorithms.create("multi_configuration_calculator", "macis_cas")
        macis_calc.settings().set("calculate_one_rdm", True)
        macis_calc.settings().set("calculate_two_rdm", True)

        # Create PySCF CASSCF calculation with macis
        casscf = pyscf.mcscf.CASSCF(pyscf_scf, 6, 6)
        casscf.fcisolver = _mcsolver_to_fcisolver(pyscf_mol, macis_calc)
        casscf.verbose = 0
        casscf_energy = casscf.kernel()[0]

        assert np.isclose(
            casscf_energy, -108.78966139913287, rtol=float_comparison_relative_tolerance, atol=mcscf_energy_tolerance
        )

    def test_pyscf_fciwrapper_casscf_triplet(self):
        """Test MC wrapper in casscf for o2 triplet with cc-pvdz basis and CAS(6,6)."""
        # Create O2 structure
        o2 = create_o2_structure()
        pyscf_mol = pyscf.gto.M(atom=structure_to_pyscf_atom_labels(o2)[0], basis="cc-pvdz", charge=0, spin=2)

        # Perform SCF calculation with qdk-chemistry
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("scf_type", "restricted")
        _, wavefunction = scf_solver.run(o2, 0, 3, "cc-pvdz")

        # Get pyscf object from hf orbitals
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        pyscf_scf = orbitals_to_scf(wavefunction.get_orbitals(), occ_alpha=occ_a, occ_beta=occ_b, scf_type="restricted")

        # Create MACIS calculator
        macis_calc = algorithms.create("multi_configuration_calculator", "macis_cas")
        macis_calc.settings().set("calculate_one_rdm", True)
        macis_calc.settings().set("calculate_two_rdm", True)
        macis_calc.settings().set("ci_residual_tolerance", 1e-10)

        # Create PySCF CASSCF calculation with macis
        casscf = pyscf.mcscf.CASSCF(pyscf_scf, 6, 6)
        casscf.fcisolver = _mcsolver_to_fcisolver(pyscf_mol, macis_calc)
        casscf.verbose = 0
        casscf_energy = casscf.kernel()[0]

        assert np.isclose(
            casscf_energy, -149.68131616317658, rtol=float_comparison_relative_tolerance, atol=mcscf_energy_tolerance
        )
