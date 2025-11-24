// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/utils/valence_space.hpp>

namespace py = pybind11;

void bind_utils(py::module& m) {
  m.def("compute_valence_space_parameters",
        &qdk::chemistry::utils::compute_valence_space_parameters,
        R"(
Get the default number of active electrons, active orbitals, which are obtained from the valence electrons and orbitals of the atomic element types in the structure.
The structure is automatically extracted from the wavefunction.

Returns:
    tuple: Pair of ( n_active_electrons, n_active_orbitals )

Examples:
    (active_electrons, active_orbitals) = compute_valence_space(wavefunction, charge)
)",
        py::arg("wavefunction"), py::arg("charge"));
}
