// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <qdk/chemistry/data/majorana_mapping.hpp>
#include <qdk/chemistry/data/pauli_operator.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

void bind_majorana_mapping(pybind11::module& data) {
  using namespace qdk::chemistry::data;

  py::class_<MajoranaMapping> mapping(data, "MajoranaMapping", R"(
Immutable fermion-to-qubit encoding.

Stores a 2N-entry table mapping each Majorana operator gamma_k to a sparse
Pauli word. The bilinear ``i*gamma_j*gamma_k`` is the unified primitive and
is computed on demand from the table. Sparse Pauli words use the little-endian
convention of QubitHamiltonian (qubit 0 has the smallest index).
)");

  mapping.def(py::init([](const std::vector<SparsePauliWord>& table,
                          const std::string& name) {
                try {
                  return MajoranaMapping(table, name);
                } catch (const std::invalid_argument& e) {
                  throw py::value_error(e.what());
                }
              }),
              py::arg("table"), py::arg("name") = "",
              "Construct from a list of 2N sparse Pauli words.");

  mapping.def_property_readonly(
      "num_modes", [](const MajoranaMapping& self) { return self.num_modes(); },
      "Number of fermionic modes (spin-orbitals).");

  mapping.def_property_readonly(
      "num_qubits",
      [](const MajoranaMapping& self) { return self.num_qubits(); },
      "Number of qubits required by this encoding.");

  mapping.def_property_readonly(
      "name", [](const MajoranaMapping& self) { return self.name(); },
      "Encoding name (may be empty).");

  mapping.def_property_readonly(
      "table", [](const MajoranaMapping& self) { return self.table(); },
      "List of 2N sparse Pauli words [(qubit_idx, op_type), ...].");

  mapping.def_property_readonly(
      "is_majorana_atomic",
      [](const MajoranaMapping& self) { return self.is_majorana_atomic(); },
      "True if individual Majorana operators have a Pauli image.");

  mapping.def(
      "__call__",
      [](const MajoranaMapping& self, std::size_t k) -> SparsePauliWord {
        try {
          return self(k);
        } catch (const std::out_of_range& e) {
          throw py::index_error(e.what());
        }
      },
      py::arg("k"), "Sparse Pauli word for Majorana operator gamma_k.");

  mapping.def(
      "majorana",
      [](const MajoranaMapping& self, std::size_t k) -> SparsePauliWord {
        try {
          return self.majorana(k);
        } catch (const std::out_of_range& e) {
          throw py::index_error(e.what());
        }
      },
      py::arg("k"), "Sparse Pauli word for Majorana operator gamma_k.");

  mapping.def(
      "bilinear",
      [](const MajoranaMapping& self, std::size_t j,
         std::size_t k) -> py::tuple {
        try {
          auto [coeff, word] = self.bilinear(j, k);
          return py::make_tuple(py::cast(coeff), py::cast(word));
        } catch (const std::invalid_argument& e) {
          throw py::value_error(e.what());
        } catch (const std::out_of_range& e) {
          throw py::index_error(e.what());
        }
      },
      py::arg("j"), py::arg("k"),
      "Pauli image (coeff, word) of the bilinear i*gamma_j*gamma_k.");

  mapping.def("__repr__", [](const MajoranaMapping& self) -> std::string {
    std::string repr = "MajoranaMapping(";
    if (!self.name().empty()) {
      repr += "'" + self.name() + "', ";
    }
    repr += "num_modes=" + std::to_string(self.num_modes()) +
            ", num_qubits=" + std::to_string(self.num_qubits()) + ")";
    return repr;
  });

  mapping.def_static(
      "jordan_wigner",
      [](std::size_t num_modes) {
        try {
          return MajoranaMapping::jordan_wigner(num_modes);
        } catch (const std::invalid_argument& e) {
          throw py::value_error(e.what());
        }
      },
      py::arg("num_modes"), "Construct a Jordan-Wigner encoding.");

  mapping.def_static(
      "bravyi_kitaev",
      [](std::size_t num_modes) {
        try {
          return MajoranaMapping::bravyi_kitaev(num_modes);
        } catch (const std::invalid_argument& e) {
          throw py::value_error(e.what());
        }
      },
      py::arg("num_modes"), "Construct a Bravyi-Kitaev encoding.");

  mapping.def_static(
      "bravyi_kitaev_tree",
      [](std::size_t num_modes) {
        try {
          return MajoranaMapping::bravyi_kitaev_tree(num_modes);
        } catch (const std::invalid_argument& e) {
          throw py::value_error(e.what());
        }
      },
      py::arg("num_modes"),
      "Construct a balanced binary-tree Bravyi-Kitaev encoding.");

  mapping.def_static(
      "parity",
      [](std::size_t num_modes) {
        try {
          return MajoranaMapping::parity(num_modes);
        } catch (const std::invalid_argument& e) {
          throw py::value_error(e.what());
        }
      },
      py::arg("num_modes"), "Construct a parity encoding.");

  data.def(
      "majorana_map_hamiltonian",
      [](const MajoranaMapping& mapping, double core_energy,
         py::array_t<double, py::array::c_style | py::array::forcecast>
             h1_alpha,
         py::array_t<double, py::array::c_style | py::array::forcecast> h1_beta,
         py::array_t<double, py::array::c_style | py::array::forcecast>
             eri_aaaa,
         py::array_t<double, py::array::c_style | py::array::forcecast>
             eri_aabb,
         py::array_t<double, py::array::c_style | py::array::forcecast>
             eri_bbbb,
         std::size_t n_spatial, bool is_spin_free, double threshold,
         double integral_threshold) -> py::tuple {
        auto result = majorana_map_hamiltonian(
            mapping, core_energy, h1_alpha.data(), h1_beta.data(),
            eri_aaaa.data(), eri_aabb.data(), eri_bbbb.data(), n_spatial,
            is_spin_free, threshold, integral_threshold);
        return py::make_tuple(py::cast(result.words),
                              py::cast(result.coefficients));
      },
      py::arg("mapping"), py::arg("core_energy"), py::arg("h1_alpha"),
      py::arg("h1_beta"), py::arg("eri_aaaa"), py::arg("eri_aabb"),
      py::arg("eri_bbbb"), py::arg("n_spatial"), py::arg("is_spin_free"),
      py::arg("threshold"), py::arg("integral_threshold"),
      R"(
Map a fermionic Hamiltonian to qubit Pauli terms using Majorana loops.

Limitation: only valid for spin-free Hamiltonians (spin-independent
integrals). Returns ``(words, coefficients)`` where ``words`` is a list of
sparse Pauli words.
)");
}
