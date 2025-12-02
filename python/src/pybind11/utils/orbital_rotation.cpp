// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/utils/orbital_rotation.hpp>

namespace py = pybind11;

void bind_orbital_rotation(py::module& m) {
  m.def("rotate_orbitals", &qdk::chemistry::utils::rotate_orbitals,
        R"(
            Rotate molecular orbitals using a rotation vector.

            This function takes QATK orbitals and applies orbital rotations using a
            rotation vector, typically taken from stability analysis eigenvectors.

            The rotation is performed by:
            1. Unpacking the rotation vector into an anti-Hermitian matrix
            2. Computing the unitary rotation matrix via matrix exponential
            3. Applying the rotation to the molecular orbital coefficients

            Parameters
            ----------
            orbitals : qdk_chemistry.data.Orbitals
                The QATK orbitals to rotate
            rotation_vector : numpy.ndarray
                The rotation vector (typically from stability analysis,
                corresponding to the lowest eigenvalue)
            num_alpha_occupied_orbitals : int
                Number of alpha occupied orbitals
            num_beta_occupied_orbitals : int
                Number of beta occupied orbitals
            restricted_external : bool, optional
                If True and orbitals are restricted, creates unrestricted orbitals with
                rotated coefficients for alpha spin and unrotated coefficients for beta spin.
                Default is False.

            Returns
            -------
            qdk_chemistry.data.Orbitals
                A new QATK Orbitals object with rotated molecular orbital coefficients

            Notes
            -----
            - restricted_external can break spin symmetry and solve external
              instabilities of RHF/RKS.
            - For unrestricted calculations, the rotation vector should contain
              alpha rotations first (n_occ_alpha * n_vir_alpha elements),
              then beta rotations (n_occ_beta * n_vir_beta elements).
            - This function assumes aufbau filling for occupation numbers.
            - Orbital energies are invalidated by rotation and set to null.

            Raises
            ------
            RuntimeError
                If rotation vector size is invalid

            Examples
            --------
            >>> # After stability analysis that finds instability
            >>> rotation_vector = ...  # eigenvector from stability analysis
            >>> rotated_orbitals = rotate_orbitals(orbitals, rotation_vector,
            ...                                     num_alpha_occupied_orbitals,
            ...                                     num_beta_occupied_orbitals)
        )",
        py::arg("orbitals"), py::arg("rotation_vector"),
        py::arg("num_alpha_occupied_orbitals"),
        py::arg("num_beta_occupied_orbitals"),
        py::arg("restricted_external") = false);

  m.def("run_scf_with_stability_workflow",
        &qdk::chemistry::utils::run_scf_with_stability_workflow,
        R"(
            Run SCF with iterative stability checking and orbital rotation workflow.

            This workflow function iteratively performs:
            1. Run SCF calculation
            2. Check stability of resulting wavefunction (internal and external for restricted)
            3. If internally unstable, rotate orbitals and restart SCF
            4. If externally unstable (restricted only), break spin symmetry and switch to unrestricted
            5. Repeat until stable or max_stability_iterations is reached

            For restricted calculations (RHF), both internal and external stability
            are checked. If external instability is detected, the workflow automatically
            switches to unrestricted (UHF) by rotating alpha orbitals while keeping beta
            orbitals unchanged, effectively breaking spin symmetry.

            Parameters
            ----------
            structure : qdk_chemistry.data.Structure
                The molecular structure
            charge : int
                The molecular charge
            spin_multiplicity : int
                The spin multiplicity
            scf_solver_name : str
                Name of the SCF solver algorithm to create (e.g., "qdk", "pyscf")
            stability_checker_name : str
                Name of the stability checker algorithm to create (e.g., "pyscf")
            initial_guess : qdk_chemistry.data.Orbitals, optional
                Optional initial orbital guess for the first SCF calculation
            max_stability_iterations : int, optional
                Maximum number of stability check and rotation cycles (default: 5)
            stability_tolerance : float, optional
                Tolerance threshold for considering eigenvalues as indicating instability.
                Eigenvalues above this threshold are considered stable (default: -1e-4)
            reference_type : str, optional
                Reference type for initial SCF calculation: "auto" (default), "restricted"
                (RHF for closed-shell, ROHF for open-shell), or "unrestricted" (UHF for
                both closed- and open-shell). Note: if external instability is detected,
                the workflow will automatically switch to "unrestricted" regardless of
                this setting (default: "auto")

            Returns:
                tuple[float, qdk_chemistry.data.Wavefunction, bool, qdk_chemistry.data.StabilityResult]:
                    Final SCF energy, converged wavefunction, stability status, and detailed stability result.

            Raises
            ------
            ValueError
                If structure is None or max_stability_iterations is less than 1
            RuntimeError
                If SCF solver or stability checker creation fails

            Notes
            -----
            - For restricted wavefunctions, external stability checks are automatically enabled
            - Internal instabilities are resolved first before checking external stability
            - External instabilities trigger automatic RHF→UHF transition
            - After switching to unrestricted, only internal stability is checked

            Examples
            --------
            >>> from qdk_chemistry.data import Structure
            >>> import numpy as np
            >>> # Create a molecular structure
            >>> structure = Structure(["O", "H", "H"], coords)
            >>> # Run stability workflow (will check external stability for restricted)
            >>> energy, wfn, is_stable, result = run_scf_with_stability_workflow(
            ...     structure, 0, 1, "qdk", "pyscf",
            ...     max_stability_iterations=5, stability_tolerance=-1e-4)
            >>> # Check convergence
            >>> print(f"Stability check converged: {is_stable}")
            >>> print(f"Final energy: {energy} Hartree")
        )",
        py::arg("structure"), py::arg("charge"), py::arg("spin_multiplicity"),
        py::arg("scf_solver_name"), py::arg("stability_checker_name"),
        py::arg("initial_guess") = std::nullopt,
        py::arg("max_stability_iterations") = 5,
        py::arg("stability_tolerance") = -1e-4,
        py::arg("reference_type") = "auto");
}
