// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/utils/valence_space.hpp>

namespace py = pybind11;

void bind_valence_space(py::module& m) {
  m.def("compute_valence_space_parameters",
        &qdk::chemistry::utils::compute_valence_space_parameters,
        R"(
Get the default number of active electrons, active orbitals, which are obtained from the valence electrons and orbitals of the atomic element types in the structure.
The structure is automatically extracted from the wavefunction.

Args:
    wavefunction: The input wavefunction (the molecular structure is taken
        from ``wavefunction.orbitals.basis_set``).
    charge: The total charge of the molecular system. Should match the charge
        used in the upstream SCF calculation.
    include_double_d_shell: When ``True``, add 5 correlating d' orbitals per
        d-block atom (Sc-Zn, Y-Cd, Hf-Hg) to capture the strong nd / (n+1)d'
        radial correlation in transition metals (the "double d-shell"
        effect). Defaults to ``False`` to preserve the historical sizing.
        Mirrors the ``include_double_d_shell`` setting on
        :class:`QdkValenceActiveSpaceSelector` and should be kept consistent
        with it when the returned values are used to populate that selector.

Returns:
    tuple: Pair of ( n_active_electrons, n_active_orbitals )

Examples:
    (active_electrons, active_orbitals) = compute_valence_space_parameters(wavefunction, charge)
    (active_electrons, active_orbitals) = compute_valence_space_parameters(
        wavefunction, charge, include_double_d_shell=True)
)",
        py::arg("wavefunction"), py::arg("charge"),
        py::arg("include_double_d_shell") = false);
}
