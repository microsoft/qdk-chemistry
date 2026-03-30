"""Tests for PySCF orbital localization."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from .pyscf_helpers import *  # noqa: F401, F403


class TestPyscfLocalization:
    """PySCF orbital localization tests."""

"""Tests for PySCF orbital localization."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from .pyscf_helpers import *  # noqa: F401, F403


class TestPyscfLocalization:
    """PySCF orbital localization tests."""

    def test_pyscf_water_pm_localization_def2svp(self):
        """Test PySCF Pipek-Mezey localization on water molecule with def2-svp basis."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        _, wavefunction = scf_solver.run(water, 0, 1, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Compute the objective function for the canonical orbitals
        can_objective_value = pipek_objective_function(orbitals, orbitals.get_coefficients()[0])

        # Get orbital occupancy information
        num_elec = wavefunction.get_total_num_electrons()
        num_occupied_orbitals = num_elec[0]  # Number of occupied orbitals

        # Prepare for Test 2: Calculate can_random_objective_value before localizer creation
        random_occ_indices = [1, 3, 4]  # Random subset of occupied orbitals
        ca_can, _ = orbitals.get_coefficients()
        ca_selected = ca_can[:, random_occ_indices]
        can_random_objective_value = pipek_objective_function(orbitals, ca_selected)

        # Localize orbitals
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")

        # Test 1: Localize occupied first, then virtual orbitals on the localized occupied orbitals
        occ_indices = list(range(num_occupied_orbitals))
        virt_indices = list(range(num_occupied_orbitals, orbitals.get_num_molecular_orbitals()))

        # Localize occupied orbitals first
        localized_occ_wfn = localizer.run(wavefunction, occ_indices, occ_indices)
        assert localized_occ_wfn is not None

        # Localize virtual orbitals on the occupied-localized orbitals
        localized_virt_wfn = localizer.run(localized_occ_wfn, virt_indices, virt_indices)
        assert localized_virt_wfn is not None

        # Check that the objective function improved for the final orbitals
        localized_virt = localized_virt_wfn.get_orbitals()
        mos_final, _ = localized_virt.get_coefficients()
        final_objective_value = pipek_objective_function(localized_virt, mos_final)
        assert final_objective_value > can_objective_value

        # Test 2: Randomly choose indices from occupied orbitals only
        localized_random = localizer.run(wavefunction, random_occ_indices, random_occ_indices)
        mos_rand, _ = localized_random.get_orbitals().get_coefficients()

        # Extract the submatrix for the localized indices
        s_matrix = orbitals.get_overlap_matrix()
        ca_loc_selected = mos_rand[:, random_occ_indices]

        # Check that the transformation for selected indices is unitary
        u_selected = ca_selected.T @ s_matrix @ ca_loc_selected
        unitarity_error = np.linalg.norm(u_selected @ u_selected.T - np.eye(len(random_occ_indices)))
        assert unitarity_error < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal
        overlap_check = ca_loc_selected.T @ s_matrix @ ca_loc_selected
        orthonormality_error = np.linalg.norm(overlap_check - np.eye(len(random_occ_indices)))
        assert orthonormality_error < orthonormality_error_tolerance

        # Test that the objective function improved for the random selection
        random_objective_value = pipek_objective_function(localized_random.get_orbitals(), ca_loc_selected)
        assert random_objective_value >= can_random_objective_value

    def test_pyscf_water_fb_localization_def2svp(self):
        """Test PySCF Foster-Boys localization on water molecule with def2-svp basis."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        _, wavefunction = scf_solver.run(water, 0, 1, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Compute the objective function for the canonical orbitals
        can_objective_value = boys_objective_function(orbitals, orbitals.get_coefficients()[0])

        # Get orbital occupancy information
        num_elec = wavefunction.get_total_num_electrons()
        num_occupied_orbitals = num_elec[0]  # Number of occupied orbitals

        # Prepare for Test 2: Calculate can_random_objective_value before localizer creation
        random_virt_indices = [5, 7, 9]  # Random subset of virtual orbitals (indices >= num_occupied_orbitals)
        ca_can, _ = orbitals.get_coefficients()
        ca_selected = ca_can[:, random_virt_indices]
        can_random_objective_value = boys_objective_function(orbitals, ca_selected)

        # Localize orbitals
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")
        localizer.settings().set("method", "foster-boys")

        # Test 1: Localize occupied first, then virtual orbitals on the localized occupied orbitals
        occ_indices = list(range(num_occupied_orbitals))
        virt_indices = list(range(num_occupied_orbitals, orbitals.get_num_molecular_orbitals()))

        # Localize occupied orbitals first
        localized_occ_wfn = localizer.run(wavefunction, occ_indices, occ_indices)
        assert localized_occ_wfn is not None

        # Localize virtual orbitals on the occupied-localized orbitals
        localized_virt_wfn = localizer.run(localized_occ_wfn, virt_indices, virt_indices)
        assert localized_virt_wfn is not None

        # Check that the objective function improved for the final orbitals
        localized_virt = localized_virt_wfn.get_orbitals()
        mos_final, _ = localized_virt.get_coefficients()
        final_objective_value = boys_objective_function(localized_virt, mos_final)
        assert final_objective_value < can_objective_value

        # Test 2: Randomly choose indices from virtual orbitals only
        localized_random = localizer.run(wavefunction, random_virt_indices, random_virt_indices)
        mos_rand, _ = localized_random.get_orbitals().get_coefficients()

        # Extract the submatrix for the localized indices
        s_matrix = orbitals.get_overlap_matrix()
        ca_loc_selected = mos_rand[:, random_virt_indices]

        # Check that the transformation for selected indices is unitary
        u_selected = ca_selected.T @ s_matrix @ ca_loc_selected
        unitarity_error = np.linalg.norm(u_selected @ u_selected.T - np.eye(len(random_virt_indices)))
        assert unitarity_error < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal
        overlap_check = ca_loc_selected.T @ s_matrix @ ca_loc_selected
        orthonormality_error = np.linalg.norm(overlap_check - np.eye(len(random_virt_indices)))
        assert orthonormality_error < orthonormality_error_tolerance

        # Test that the objective function improved for the random selection
        random_objective_value = boys_objective_function(localized_random.get_orbitals(), ca_loc_selected)
        assert random_objective_value <= can_random_objective_value

    def test_pyscf_water_er_localization_def2svp(self):
        """Test PySCF Edmiston-Ruedenberg localization on water molecule with def2-svp basis."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        _, wavefunction = scf_solver.run(water, 0, 1, "def2-svp")
        orbitals = wavefunction.get_orbitals()
        ca_can, _ = orbitals.get_coefficients()
        # Compute the objective function for the canonical orbitals
        can_objective_value = er_objective_function(orbitals, orbitals.get_coefficients()[0])
        # Random subset of occupied orbitals, must include 0 (O 1s), possibly due to numerical instability
        random_occ_indices = [0, 1, 4]
        # Prepare for Test 2: Calculate can_random_objective_value before localizer creation
        ca_selected = ca_can[:, random_occ_indices]
        can_random_objective_value = er_objective_function(orbitals, ca_selected)

        # Localize orbitals
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")
        localizer.settings().set("method", "edmiston-ruedenberg")

        # Get orbital occupancy information
        num_elec = wavefunction.get_total_num_electrons()
        num_occupied_orbitals = num_elec[0]  # Number of occupied orbitals
        # Test 1: Localize occupied first, then virtual orbitals on the localized occupied orbitals
        occ_indices = list(range(num_occupied_orbitals))
        virt_indices = list(range(num_occupied_orbitals, orbitals.get_num_molecular_orbitals()))

        # Localize occupied orbitals first
        localized_occ_wfn = localizer.run(wavefunction, occ_indices, occ_indices)
        assert localized_occ_wfn is not None

        # Localize virtual orbitals on the occupied-localized orbitals
        localized_virt_wfn = localizer.run(localized_occ_wfn, virt_indices, virt_indices)
        assert localized_virt_wfn is not None

        # Check that the objective function improved for the final orbitals
        localized_virt = localized_virt_wfn.get_orbitals()
        mos_final, _ = localized_virt.get_coefficients()
        final_objective_value = er_objective_function(localized_virt, mos_final)
        assert final_objective_value > can_objective_value

        # Test 2: Randomly choose indices from occupied orbitals only
        localized_random = localizer.run(wavefunction, random_occ_indices, random_occ_indices)
        mos_rand, _ = localized_random.get_orbitals().get_coefficients()

        # Extract the submatrix for the localized indices
        s_matrix = orbitals.get_overlap_matrix()
        ca_loc_selected = mos_rand[:, random_occ_indices]
        # Test that the objective function improved for the random selection
        random_objective_value = er_objective_function(orbitals, ca_loc_selected)
        assert random_objective_value >= can_random_objective_value

        # Check that the transformation for selected indices is unitary
        u_selected = ca_selected.T @ s_matrix @ ca_loc_selected
        unitarity_error = np.linalg.norm(u_selected @ u_selected.T - np.eye(len(random_occ_indices)))
        assert unitarity_error < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal
        overlap_check = ca_loc_selected.T @ s_matrix @ ca_loc_selected
        orthonormality_error = np.linalg.norm(overlap_check - np.eye(len(random_occ_indices)))
        assert orthonormality_error < orthonormality_error_tolerance

    def test_pyscf_o2_pm_localization_def2svp_uhf(self):
        """Test PySCF Pipek-Mezey localization on O2 molecule with UHF/def2-svp."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        _, wavefunction = scf_solver.run(o2, 0, 3, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        [can_a, can_b] = orbitals.get_coefficients()
        can_objective_value_a = pipek_objective_function(orbitals, can_a)
        can_objective_value_b = pipek_objective_function(orbitals, can_b)

        # Get orbital occupancy information
        num_elec = wavefunction.get_total_num_electrons()
        num_alpha, num_beta = num_elec[0], num_elec[1]

        # Prepare for Test 2: Randomly choose indices from occupied orbitals only for both spin channels
        random_occ_indices_alpha = [2, 3]  # Random subset of occupied alpha orbitals
        random_occ_indices_beta = [0, 1]  # Random subset of occupied beta orbitals
        ca_selected = can_a[:, random_occ_indices_alpha]
        cb_selected = can_b[:, random_occ_indices_beta]
        can_random_objective_value_a = pipek_objective_function(orbitals, ca_selected)
        can_random_objective_value_b = pipek_objective_function(orbitals, cb_selected)

        # Localize orbitals
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")

        # Test 1: Localize occupied first, then virtual orbitals on the localized occupied orbitals
        occ_indices_alpha = list(range(num_alpha))
        virt_indices_alpha = list(range(num_alpha, orbitals.get_num_molecular_orbitals()))
        occ_indices_beta = list(range(num_beta))
        virt_indices_beta = list(range(num_beta, orbitals.get_num_molecular_orbitals()))

        # Localize occupied orbitals first
        localized_occ_wfn = localizer.run(wavefunction, occ_indices_alpha, occ_indices_beta)
        assert localized_occ_wfn is not None

        # Localize virtual orbitals on the occupied-localized orbitals
        localized_virt_wfn = localizer.run(localized_occ_wfn, virt_indices_alpha, virt_indices_beta)
        assert localized_virt_wfn is not None

        # Check that the objective function improved for the final orbitals
        localized_virt = localized_virt_wfn.get_orbitals()
        mos_final_a, mos_final_b = localized_virt.get_coefficients()
        final_objective_value_a = pipek_objective_function(localized_virt, mos_final_a)
        final_objective_value_b = pipek_objective_function(localized_virt, mos_final_b)
        assert final_objective_value_a > can_objective_value_a
        assert final_objective_value_b > can_objective_value_b

        # Test 2: Randomly choose indices to localize for both spin channels
        localized_random = localizer.run(wavefunction, random_occ_indices_alpha, random_occ_indices_beta)
        mos_rand_a, mos_rand_b = localized_random.get_orbitals().get_coefficients()

        # Test alpha channel
        s_matrix = orbitals.get_overlap_matrix()
        ca_loc_selected = mos_rand_a[:, random_occ_indices_alpha]

        # Check that the transformation for selected indices is unitary - alpha
        u_selected_a = ca_selected.T @ s_matrix @ ca_loc_selected
        unitarity_error_a = np.linalg.norm(u_selected_a @ u_selected_a.T - np.eye(len(random_occ_indices_alpha)))
        assert unitarity_error_a < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal - alpha
        overlap_check_a = ca_loc_selected.T @ s_matrix @ ca_loc_selected
        orthonormality_error_a = np.linalg.norm(overlap_check_a - np.eye(len(random_occ_indices_alpha)))
        assert orthonormality_error_a < orthonormality_error_tolerance

        # Test beta channel
        cb_loc_selected = mos_rand_b[:, random_occ_indices_beta]

        # Check that the transformation for selected indices is unitary - beta
        u_selected_b = cb_selected.T @ s_matrix @ cb_loc_selected
        unitarity_error_b = np.linalg.norm(u_selected_b @ u_selected_b.T - np.eye(len(random_occ_indices_beta)))
        assert unitarity_error_b < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal - beta
        overlap_check_b = cb_loc_selected.T @ s_matrix @ cb_loc_selected
        orthonormality_error_b = np.linalg.norm(overlap_check_b - np.eye(len(random_occ_indices_beta)))
        assert orthonormality_error_b < orthonormality_error_tolerance

        # Test that the objective function improved for the random selection
        random_objective_value_a = pipek_objective_function(localized_random.get_orbitals(), ca_loc_selected)
        random_objective_value_b = pipek_objective_function(localized_random.get_orbitals(), cb_loc_selected)
        # Allow tiny numerical tolerance due to floating point precision
        assert random_objective_value_a >= can_random_objective_value_a - 5e-14
        assert random_objective_value_b >= can_random_objective_value_b - 5e-14

    def test_pyscf_o2_fb_localization_def2svp_uhf(self):
        """Test PySCF Foster-Boys localization on O2 molecule with UHF/def2-svp."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        _, wavefunction = scf_solver.run(o2, 0, 3, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Compute the objective function for the canonical orbitals
        [can_a, can_b] = orbitals.get_coefficients()
        can_objective_value_a = boys_objective_function(orbitals, can_a)
        can_objective_value_b = boys_objective_function(orbitals, can_b)

        # Get orbital occupancy information
        num_elec = wavefunction.get_total_num_electrons()
        num_alpha, num_beta = num_elec[0], num_elec[1]

        # Prepare for Test 2: Randomly choose indices from virtual orbitals only for both spin channels
        random_virt_indices_alpha = [
            num_alpha + 1,
            num_alpha + 3,
        ]  # Random subset of virtual alpha orbitals
        random_virt_indices_beta = [
            num_beta + 2,
            num_beta + 4,
        ]  # Random subset of virtual beta orbitals
        ca_selected = can_a[:, random_virt_indices_alpha]
        cb_selected = can_b[:, random_virt_indices_beta]
        can_random_objective_value_a = boys_objective_function(orbitals, ca_selected)
        can_random_objective_value_b = boys_objective_function(orbitals, cb_selected)

        # Localize orbitals
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")
        localizer.settings().set("method", "foster-boys")

        # Test 1: Localize occupied first, then virtual orbitals on the localized occupied orbitals
        occ_indices_alpha = list(range(num_alpha))
        virt_indices_alpha = list(range(num_alpha, orbitals.get_num_molecular_orbitals()))
        occ_indices_beta = list(range(num_beta))
        virt_indices_beta = list(range(num_beta, orbitals.get_num_molecular_orbitals()))

        # Localize occupied orbitals first
        localized_occ_wfn = localizer.run(wavefunction, occ_indices_alpha, occ_indices_beta)
        assert localized_occ_wfn is not None

        # Localize virtual orbitals on the occupied-localized orbitals
        localized_virt_wfn = localizer.run(localized_occ_wfn, virt_indices_alpha, virt_indices_beta)
        assert localized_virt_wfn is not None

        # Check that the objective function improved for the final orbitals
        localized_virt = localized_virt_wfn.get_orbitals()
        mos_final_a, mos_final_b = localized_virt.get_coefficients()
        final_objective_value_a = boys_objective_function(localized_virt, mos_final_a)
        final_objective_value_b = boys_objective_function(localized_virt, mos_final_b)
        assert final_objective_value_a < can_objective_value_a
        assert final_objective_value_b < can_objective_value_b

        # Test 2: Randomly choose indices from virtual orbitals only for both spin channels
        localized_random = localizer.run(wavefunction, random_virt_indices_alpha, random_virt_indices_beta)
        mos_rand_a, mos_rand_b = localized_random.get_orbitals().get_coefficients()

        # Test alpha channel
        s_matrix = orbitals.get_overlap_matrix()
        ca_loc_selected = mos_rand_a[:, random_virt_indices_alpha]

        # Check that the transformation for selected indices is unitary - alpha
        u_selected_a = ca_selected.T @ s_matrix @ ca_loc_selected
        unitarity_error_a = np.linalg.norm(u_selected_a @ u_selected_a.T - np.eye(len(random_virt_indices_alpha)))
        assert unitarity_error_a < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal - alpha
        overlap_check_a = ca_loc_selected.T @ s_matrix @ ca_loc_selected
        orthonormality_error_a = np.linalg.norm(overlap_check_a - np.eye(len(random_virt_indices_alpha)))
        assert orthonormality_error_a < orthonormality_error_tolerance

        # Test beta channel
        cb_loc_selected = mos_rand_b[:, random_virt_indices_beta]

        # Check that the transformation for selected indices is unitary - beta
        u_selected_b = cb_selected.T @ s_matrix @ cb_loc_selected
        unitarity_error_b = np.linalg.norm(u_selected_b @ u_selected_b.T - np.eye(len(random_virt_indices_beta)))
        assert unitarity_error_b < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal - beta
        overlap_check_b = cb_loc_selected.T @ s_matrix @ cb_loc_selected
        orthonormality_error_b = np.linalg.norm(overlap_check_b - np.eye(len(random_virt_indices_beta)))
        assert orthonormality_error_b < orthonormality_error_tolerance

        # Test that the objective function improved for the random selection
        random_objective_value_a = boys_objective_function(localized_random.get_orbitals(), ca_loc_selected)
        random_objective_value_b = boys_objective_function(localized_random.get_orbitals(), cb_loc_selected)
        # Allow tiny numerical tolerance due to floating point precision
        assert random_objective_value_a <= can_random_objective_value_a + 5e-14
        assert random_objective_value_b <= can_random_objective_value_b + 5e-14

    def test_pyscf_o2_er_localization_def2svp_uhf(self):
        """Test PySCF Edmiston-Ruedenberg localization on O2 molecule with UHF/def2-svp."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        _, wavefunction = scf_solver.run(o2, 0, 3, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Compute the objective function for the canonical orbitals
        [can_a, can_b] = orbitals.get_coefficients()
        can_objective_value_a = er_objective_function(orbitals, can_a)
        can_objective_value_b = er_objective_function(orbitals, can_b)

        # Get orbital occupancy information
        num_elec = wavefunction.get_total_num_electrons()
        num_alpha, num_beta = num_elec[0], num_elec[1]

        # Prepare for Test 2: Randomly choose indices from occupied orbitals only for both spin channels
        random_occ_indices_alpha = [2, 3]  # Random subset of occupied alpha orbitals
        random_occ_indices_beta = [0, 1, 4]  # Random subset of occupied beta orbitals
        ca_selected = can_a[:, random_occ_indices_alpha]
        cb_selected = can_b[:, random_occ_indices_beta]
        can_random_objective_value_a = er_objective_function(orbitals, ca_selected)
        can_random_objective_value_b = er_objective_function(orbitals, cb_selected)

        # Localize orbitals
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")

        # Test 1: Localize occupied first, then virtual orbitals on the localized occupied orbitals
        occ_indices_alpha = list(range(num_alpha))
        virt_indices_alpha = list(range(num_alpha, orbitals.get_num_molecular_orbitals()))
        occ_indices_beta = list(range(num_beta))
        virt_indices_beta = list(range(num_beta, orbitals.get_num_molecular_orbitals()))

        # Localize occupied orbitals first
        localized_occ_wfn = localizer.run(wavefunction, occ_indices_alpha, occ_indices_beta)
        assert localized_occ_wfn is not None

        # Localize virtual orbitals on the occupied-localized orbitals
        localized_virt_wfn = localizer.run(localized_occ_wfn, virt_indices_alpha, virt_indices_beta)
        assert localized_virt_wfn is not None

        # Check that the objective function improved for the final orbitals
        localized_virt = localized_virt_wfn.get_orbitals()
        mos_final_a, mos_final_b = localized_virt.get_coefficients()
        final_objective_value_a = er_objective_function(localized_virt, mos_final_a)
        final_objective_value_b = er_objective_function(localized_virt, mos_final_b)
        assert final_objective_value_a > can_objective_value_a
        assert final_objective_value_b > can_objective_value_b

        # Test 2: Randomly choose indices to localize for both spin channels
        localized_random = localizer.run(wavefunction, random_occ_indices_alpha, random_occ_indices_beta)
        mos_rand_a, mos_rand_b = localized_random.get_orbitals().get_coefficients()

        # Test alpha channel
        s_matrix = orbitals.get_overlap_matrix()
        ca_loc_selected = mos_rand_a[:, random_occ_indices_alpha]

        # Check that the transformation for selected indices is unitary - alpha
        u_selected_a = ca_selected.T @ s_matrix @ ca_loc_selected
        unitarity_error_a = np.linalg.norm(u_selected_a @ u_selected_a.T - np.eye(len(random_occ_indices_alpha)))
        assert unitarity_error_a < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal - alpha
        overlap_check_a = ca_loc_selected.T @ s_matrix @ ca_loc_selected
        orthonormality_error_a = np.linalg.norm(overlap_check_a - np.eye(len(random_occ_indices_alpha)))
        assert orthonormality_error_a < orthonormality_error_tolerance

        # Test beta channel
        cb_loc_selected = mos_rand_b[:, random_occ_indices_beta]

        # Check that the transformation for selected indices is unitary - beta
        u_selected_b = cb_selected.T @ s_matrix @ cb_loc_selected
        unitarity_error_b = np.linalg.norm(u_selected_b @ u_selected_b.T - np.eye(len(random_occ_indices_beta)))
        assert unitarity_error_b < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal - beta
        overlap_check_b = cb_loc_selected.T @ s_matrix @ cb_loc_selected
        orthonormality_error_b = np.linalg.norm(overlap_check_b - np.eye(len(random_occ_indices_beta)))
        assert orthonormality_error_b < orthonormality_error_tolerance

        # Test that the objective function improved for the random selection
        random_objective_value_a = er_objective_function(localized_random.get_orbitals(), ca_loc_selected)
        random_objective_value_b = er_objective_function(localized_random.get_orbitals(), cb_loc_selected)
        # Allow tiny numerical tolerance due to floating point precision
        assert random_objective_value_a >= can_random_objective_value_a - 5e-14
        assert random_objective_value_b >= can_random_objective_value_b - 5e-14

    def test_pyscf_o2_pm_localization_def2svp_rohf(self):
        """Test PySCF Pipek-Mezey localization on O2 molecule with ROHF/def2-svp."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("scf_type", "restricted")

        _, wavefunction = scf_solver.run(o2, 0, 3, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Compute the objective function for the canonical orbitals
        can_mos, _ = orbitals.get_coefficients()
        can_objective_value = pipek_objective_function(orbitals, can_mos)

        # Get orbital occupancy information
        num_elec = wavefunction.get_total_num_electrons()
        num_occupied_orbitals = num_elec[0]  # Number of occupied orbitals (same as alpha for ROHF)

        # Prepare for Test 2: Randomly choose indices from occupied orbitals only
        random_occ_indices = [0, 3, 4, 7]  # Random subset of occupied orbitals
        ca_selected = can_mos[:, random_occ_indices]
        can_random_objective_value = pipek_objective_function(orbitals, ca_selected)

        # Localize orbitals
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")

        # Test 1: Localize occupied first, then virtual orbitals on the localized occupied orbitals
        occ_indices = list(range(num_occupied_orbitals))
        virt_indices = list(range(num_occupied_orbitals, orbitals.get_num_molecular_orbitals()))

        # Localize occupied orbitals first
        localized_occ_wfn = localizer.run(wavefunction, occ_indices, occ_indices)
        assert localized_occ_wfn is not None

        # Localize virtual orbitals on the occupied-localized orbitals
        localized_virt_wfn = localizer.run(localized_occ_wfn, virt_indices, virt_indices)
        assert localized_virt_wfn is not None

        # Check that the objective function improved for the final orbitals
        localized_virt = localized_virt_wfn.get_orbitals()
        mos_final, _ = localized_virt.get_coefficients()
        final_objective_value = pipek_objective_function(localized_virt, mos_final)
        assert final_objective_value > can_objective_value

        # Test 2: Randomly choose indices from occupied orbitals only
        localized_random = localizer.run(wavefunction, random_occ_indices, random_occ_indices)
        mos_rand, _ = localized_random.get_orbitals().get_coefficients()

        # Extract the submatrix for the localized indices
        s_matrix = orbitals.get_overlap_matrix()
        ca_loc_selected = mos_rand[:, random_occ_indices]

        # Check that the transformation for selected indices is unitary
        u_selected = ca_selected.T @ s_matrix @ ca_loc_selected
        unitarity_error = np.linalg.norm(u_selected @ u_selected.T - np.eye(len(random_occ_indices)))
        assert unitarity_error < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal
        overlap_check = ca_loc_selected.T @ s_matrix @ ca_loc_selected
        orthonormality_error = np.linalg.norm(overlap_check - np.eye(len(random_occ_indices)))
        assert orthonormality_error < orthonormality_error_tolerance

        # Test that the objective function improved for the random selection
        random_objective_value = pipek_objective_function(localized_random.get_orbitals(), ca_loc_selected)
        assert random_objective_value >= can_random_objective_value

    def test_pyscf_o2_fb_localization_def2svp_rohf(self):
        """Test PySCF Foster-Boys localization on O2 molecule with ROHF/def2-svp."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("scf_type", "restricted")

        _, wavefunction = scf_solver.run(o2, 0, 3, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Compute the objective function for the canonical orbitals
        can_mos, _ = orbitals.get_coefficients()
        can_objective_value = boys_objective_function(orbitals, can_mos)

        # Get orbital occupancy information
        num_elec = wavefunction.get_total_num_electrons()
        num_occupied_orbitals = num_elec[0]  # Number of occupied orbitals (same as alpha for ROHF)

        # Prepare for Test 2: Randomly choose indices from virtual orbitals only
        random_virt_indices = [
            num_occupied_orbitals + 1,
            num_occupied_orbitals + 3,
            num_occupied_orbitals + 5,
            num_occupied_orbitals + 7,
        ]  # Random subset of virtual orbitals
        ca_selected = can_mos[:, random_virt_indices]
        can_random_objective_value = boys_objective_function(orbitals, ca_selected)

        # Localize orbitals
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")
        localizer.settings().set("method", "foster-boys")

        # Test 1: Localize occupied first, then virtual orbitals on the localized occupied orbitals
        occ_indices = list(range(num_occupied_orbitals))
        virt_indices = list(range(num_occupied_orbitals, orbitals.get_num_molecular_orbitals()))

        # Localize occupied orbitals first
        localized_occ_wfn = localizer.run(wavefunction, occ_indices, occ_indices)
        assert localized_occ_wfn is not None

        # Localize virtual orbitals on the occupied-localized orbitals
        localized_virt_wfn = localizer.run(localized_occ_wfn, virt_indices, virt_indices)
        assert localized_virt_wfn is not None

        # Check that the objective function improved for the final orbitals
        localized_virt = localized_virt_wfn.get_orbitals()
        mos_final, _ = localized_virt.get_coefficients()
        final_objective_value = boys_objective_function(localized_virt, mos_final)
        assert final_objective_value < can_objective_value

        # Test 2: Randomly choose indices from virtual orbitals only
        localized_random = localizer.run(wavefunction, random_virt_indices, random_virt_indices)
        mos_rand, _ = localized_random.get_orbitals().get_coefficients()

        # Extract the submatrix for the localized indices
        s_matrix = orbitals.get_overlap_matrix()
        ca_loc_selected = mos_rand[:, random_virt_indices]

        # Check that the transformation for selected indices is unitary
        u_selected = ca_selected.T @ s_matrix @ ca_loc_selected
        unitarity_error = np.linalg.norm(u_selected @ u_selected.T - np.eye(len(random_virt_indices)))
        assert unitarity_error < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal
        overlap_check = ca_loc_selected.T @ s_matrix @ ca_loc_selected
        orthonormality_error = np.linalg.norm(overlap_check - np.eye(len(random_virt_indices)))
        assert orthonormality_error < orthonormality_error_tolerance

        # Test that the objective function improved for the random selection
        random_objective_value = boys_objective_function(localized_random.get_orbitals(), ca_loc_selected)
        assert random_objective_value <= can_random_objective_value

    def test_pyscf_o2_er_localization_def2svp_rohf(self):
        """Test PySCF Edmiston-Ruedenberg localization on O2 molecule with ROHF/def2-svp."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("scf_type", "restricted")

        _, wavefunction = scf_solver.run(o2, 0, 3, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Compute the objective function for the canonical orbitals
        can_mos, _ = orbitals.get_coefficients()
        can_objective_value = er_objective_function(orbitals, can_mos)

        # Get orbital occupancy information
        num_elec = wavefunction.get_total_num_electrons()
        num_occupied_orbitals = num_elec[0]  # Number of occupied orbitals (same as alpha for ROHF)

        # Prepare for Test 2: Randomly choose indices from occupied orbitals only
        random_occ_indices = [0, 3, 4]  # Random subset of occupied orbitals
        ca_selected = can_mos[:, random_occ_indices]
        can_random_objective_value = er_objective_function(orbitals, ca_selected)

        # Localize orbitals
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")
        localizer.settings().set("method", "edmiston-ruedenberg")

        # Test 1: Localize occupied first, then virtual orbitals on the localized occupied orbitals
        occ_indices = list(range(num_occupied_orbitals))
        virt_indices = list(range(num_occupied_orbitals, orbitals.get_num_molecular_orbitals()))

        # Localize occupied orbitals first
        localized_occ_wfn = localizer.run(wavefunction, occ_indices, occ_indices)
        assert localized_occ_wfn is not None

        # Localize virtual orbitals on the occupied-localized orbitals
        localized_virt_wfn = localizer.run(localized_occ_wfn, virt_indices, virt_indices)
        assert localized_virt_wfn is not None

        # Check that the objective function improved for the final orbitals
        localized_virt = localized_virt_wfn.get_orbitals()
        mos_final, _ = localized_virt.get_coefficients()
        final_objective_value = er_objective_function(localized_virt, mos_final)
        assert final_objective_value > can_objective_value

        # Test 2: Randomly choose indices from occupied orbitals only
        localized_random = localizer.run(wavefunction, random_occ_indices, random_occ_indices)
        mos_rand, _ = localized_random.get_orbitals().get_coefficients()

        # Extract the submatrix for the localized indices
        s_matrix = orbitals.get_overlap_matrix()
        ca_loc_selected = mos_rand[:, random_occ_indices]

        # Check that the transformation for selected indices is unitary
        u_selected = ca_selected.T @ s_matrix @ ca_loc_selected
        unitarity_error = np.linalg.norm(u_selected @ u_selected.T - np.eye(len(random_occ_indices)))
        assert unitarity_error < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal
        overlap_check = ca_loc_selected.T @ s_matrix @ ca_loc_selected
        orthonormality_error = np.linalg.norm(overlap_check - np.eye(len(random_occ_indices)))
        assert orthonormality_error < orthonormality_error_tolerance

        # Test that the objective function improved for the random selection
        random_objective_value = er_objective_function(localized_random.get_orbitals(), ca_loc_selected)
        assert random_objective_value >= can_random_objective_value

    # =============================================================================
    # Tests for active space preservation after localization
    # Regression tests for bug: active space indices lost after orbital localization
    # =============================================================================

    def _verify_active_space_preserved(self, wfn_before, wfn_after, localizer_name):
        """Helper to verify active space indices are preserved after localization."""
        orbitals_before = wfn_before.get_orbitals()
        orbitals_after = wfn_after.get_orbitals()

        assert orbitals_before.has_active_space()
        assert orbitals_after.has_active_space(), f"Active space lost after {localizer_name} localization"

        alpha_before, beta_before = orbitals_before.get_active_space_indices()
        alpha_after, beta_after = orbitals_after.get_active_space_indices()

        assert list(alpha_before) == list(alpha_after), f"{localizer_name}: alpha indices changed"
        assert list(beta_before) == list(beta_after), f"{localizer_name}: beta indices changed"

    @pytest.mark.parametrize("method", ["pipek-mezey", "foster-boys", "edmiston-ruedenberg", "cholesky"])
    def test_pyscf_localization_preserves_active_space_restricted(self, method):
        """Test that PySCF localization preserves active space indices (restricted)."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        _, wavefunction = scf_solver.run(water, 0, 1, "sto-3g")

        # Select an active space
        selector = algorithms.create("active_space_selector", "qdk_valence")
        selector.settings().set("num_active_electrons", 6)
        selector.settings().set("num_active_orbitals", 5)
        active_wfn = selector.run(wavefunction)

        active_alpha, active_beta = active_wfn.get_orbitals().get_active_space_indices()

        # Localize
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")
        localizer.settings().set("method", method)
        localized_wfn = localizer.run(active_wfn, list(active_alpha), list(active_beta))

        self._verify_active_space_preserved(active_wfn, localized_wfn, f"pyscf_{method}")

    def test_pyscf_localization_preserves_active_space_unrestricted(self):
        """Test that PySCF localization preserves active space indices (unrestricted)."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("scf_type", "unrestricted")
        _, wavefunction = scf_solver.run(water, 0, 1, "sto-3g")

        # Manually set active space indices (ValenceActiveSpaceSelector doesn't support UHF)
        orbitals = wavefunction.get_orbitals()
        num_mo = orbitals.get_num_molecular_orbitals()

        # Define active space: frozen core (first 2 are inactive), rest are active
        # Must include all occupied orbitals in active space for SlaterDeterminantContainer
        active_alpha = list(range(2, num_mo))
        active_beta = list(range(2, num_mo))
        inactive_alpha = [0, 1]
        inactive_beta = [0, 1]

        # Create orbitals with active space
        coeffs_alpha, coeffs_beta = orbitals.get_coefficients()
        active_orbitals = data.Orbitals(
            coefficients_alpha=coeffs_alpha,
            coefficients_beta=coeffs_beta,
            ao_overlap=orbitals.get_overlap_matrix() if orbitals.has_overlap_matrix() else None,
            basis_set=orbitals.get_basis_set(),
            indices=(active_alpha, active_beta, inactive_alpha, inactive_beta),
        )

        active_wfn = data.Wavefunction(
            data.SlaterDeterminantContainer(wavefunction.get_active_determinants()[0], active_orbitals)
        )

        # Localize only the active orbitals
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")
        localizer.settings().set("method", "pipek-mezey")
        localized_wfn = localizer.run(active_wfn, active_alpha, active_beta)

        self._verify_active_space_preserved(active_wfn, localized_wfn, "pyscf_pipek_mezey_unrestricted")

