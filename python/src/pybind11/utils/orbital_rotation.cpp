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

            This function takes Orbitals and applies orbital rotations using a
            rotation vector, typically taken from stability analysis eigenvectors.

            The rotation is performed by:
            1. Unpacking the rotation vector into an anti-Hermitian matrix
            2. Computing the unitary rotation matrix via matrix exponential
            3. Applying the rotation to the molecular orbital coefficients

            Args
            ----
            orbitals : qdk_chemistry.data.Orbitals
                The Orbitals to rotate
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
                A new Orbitals object with rotated molecular orbital coefficients

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
}
