// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <nanobind/nanobind.h>

#include <qdk/chemistry/utils/valence_space.hpp>

#include "stl.hpp"

namespace nb = nanobind;

void bind_valence_space(nb::module_& m) {
  m.def("compute_valence_space_parameters",
        &qdk::chemistry::utils::compute_valence_space_parameters,
        R"(
Get the default number of active electrons and active orbitals, which are obtained from the valence electrons and orbitals of the atomic element types in the structure.
The structure is automatically extracted from the wavefunction.

Args:
    wavefunction: The input wavefunction (the molecular structure is taken
        from ``wavefunction.orbitals.basis_set.structure``).
    charge: The total charge of the molecular system. Should match the charge
        used in the upstream SCF calculation.
    include_double_d_shell: When ``True``, add 5 correlating d' orbitals per
        d-block atom (Sc-Zn, Y-Cd, Hf-Hg) to capture the strong nd / (n+1)d'
        radial correlation in transition metals (the "double d-shell"
        effect). Defaults to ``False`` to preserve the historical sizing.

Returns:
    tuple: Pair of ( n_active_electrons, n_active_orbitals )

Examples:
    >>> nele, norb = compute_valence_space_parameters(wavefunction, charge)
    >>> nele, norb = compute_valence_space_parameters(wavefunction, charge, include_double_d_shell=True)
)",
        nb::arg("wavefunction"), nb::arg("charge"),
        nb::arg("include_double_d_shell") = false);
}
