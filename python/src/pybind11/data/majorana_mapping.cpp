// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/cholesky.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/sparse.hpp>
#include <qdk/chemistry/data/majorana_mapping.hpp>
#include <qdk/chemistry/data/pauli_operator.hpp>
#include <qdk/chemistry/data/tapering.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "path_utils.hpp"

namespace py = pybind11;

void bind_majorana_mapping(pybind11::module& data) {
  using namespace qdk::chemistry::data;

  py::class_<TaperingSpecification, DataClass, py::smart_holder> tapering(
      data, "TaperingSpecification", R"(
Immutable specification for post-mapping qubit tapering.
)");

  tapering
      .def(py::init<std::vector<std::size_t>, std::vector<int>>(),
           py::arg("qubit_indices"), py::arg("eigenvalues"))
      .def_property_readonly("qubit_indices",
                             &TaperingSpecification::qubit_indices)
      .def_property_readonly("eigenvalues", &TaperingSpecification::eigenvalues)
      .def_property_readonly("num_tapered", &TaperingSpecification::num_tapered)
      .def_static(
          "symmetry_conserving_bravyi_kitaev",
          [](std::size_t num_modes, const py::object& symmetries) {
            auto n_alpha = symmetries.attr("n_alpha").cast<std::size_t>();
            auto n_beta = symmetries.attr("n_beta").cast<std::size_t>();
            return TaperingSpecification::symmetry_conserving_bravyi_kitaev(
                num_modes, n_alpha, n_beta);
          },
          py::arg("num_modes"), py::arg("symmetries"))
      .def_static(
          "parity_two_qubit_reduction",
          [](std::size_t num_modes, const py::object& symmetries) {
            auto n_alpha = symmetries.attr("n_alpha").cast<std::size_t>();
            auto n_beta = symmetries.attr("n_beta").cast<std::size_t>();
            return TaperingSpecification::parity_two_qubit_reduction(
                num_modes, n_alpha, n_beta);
          },
          py::arg("num_modes"), py::arg("symmetries"))
      .def("__eq__", &TaperingSpecification::operator==, py::arg("other"))
      .def("__hash__",
           [](const TaperingSpecification& self) {
             return py::hash(
                 py::make_tuple(py::tuple(py::cast(self.qubit_indices())),
                                py::tuple(py::cast(self.eigenvalues()))));
           })
      .def(
          "to_json",
          [](const TaperingSpecification& self) {
            py::module_ json = py::module_::import("json");
            return json.attr("loads")(self.to_json().dump());
          },
          "Serialize to a JSON-compatible dictionary.")
      .def_static(
          "from_json",
          [](const py::object& json_data) {
            py::module_ json = py::module_::import("json");
            return TaperingSpecification::from_json(nlohmann::json::parse(
                json.attr("dumps")(json_data).cast<std::string>()));
          },
          py::arg("json_data"),
          "Deserialize from a JSON-compatible dictionary.")
      .def(
          "to_hdf5",
          [](const TaperingSpecification& self, const py::object& group) {
            group.attr("attrs").attr("__setitem__")("json",
                                                    self.to_json().dump());
          },
          py::arg("group"), "Serialize to an HDF5 group.")
      .def_static(
          "from_hdf5",
          [](const py::object& group) {
            auto json = group.attr("attrs")
                            .attr("__getitem__")("json")
                            .cast<std::string>();
            return TaperingSpecification::from_json(
                nlohmann::json::parse(json));
          },
          py::arg("group"), "Deserialize from an HDF5 group.")
      .def(
          "to_json_file",
          [](const TaperingSpecification& self, const py::object& filename) {
            self.to_json_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          py::arg("filename"))
      .def_static(
          "from_json_file",
          [](const py::object& filename) {
            return TaperingSpecification::from_json_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          py::arg("filename"))
      .def(
          "to_hdf5_file",
          [](const TaperingSpecification& self, const py::object& filename) {
            self.to_hdf5_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          py::arg("filename"))
      .def_static(
          "from_hdf5_file",
          [](const py::object& filename) {
            return TaperingSpecification::from_hdf5_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          py::arg("filename"));

  py::class_<MajoranaMapping, DataClass, py::smart_holder> mapping(
      data, "MajoranaMapping", R"(
Fermion-to-qubit encoding.

Majorana-atomic mappings store a 2N-entry Pauli table for individual gamma_k;
bilinear-only mappings (via from_bilinears) store the bilinear images directly.
bilinear(j, k) is available on both forms.
)");

  mapping
      .def_static("from_table", &MajoranaMapping::from_table, py::arg("table"),
                  py::arg("name") = "",
                  "Construct a Majorana-atomic mapping from sparse Pauli "
                  "words.")
      .def_static("from_bilinears", &MajoranaMapping::from_bilinears,
                  py::arg("num_modes"), py::arg("bilinears"),
                  py::arg("name") = "",
                  "Construct a bilinear-only mapping from pre-computed "
                  "bilinears.")
      .def_property_readonly("num_modes", &MajoranaMapping::num_modes)
      .def_property_readonly("num_qubits", &MajoranaMapping::num_qubits)
      .def_property_readonly("name", &MajoranaMapping::name)
      .def_property_readonly("base_encoding", &MajoranaMapping::base_encoding)
      .def_property_readonly("table", &MajoranaMapping::table)
      .def_property_readonly("tapering",
                             [](const MajoranaMapping& self)
                                 -> std::optional<TaperingSpecification> {
                               return self.tapering();
                             })
      .def_property_readonly("is_majorana_atomic",
                             &MajoranaMapping::is_majorana_atomic)
      .def(
          "__call__",
          [](const MajoranaMapping& self,
             std::size_t k) -> const SparsePauliWord& {
            try {
              return self.majorana(k);
            } catch (const std::out_of_range& e) {
              throw py::index_error(e.what());
            } catch (const std::logic_error& e) {
              throw py::value_error(e.what());
            }
          },
          py::arg("k"), py::return_value_policy::reference_internal)
      .def(
          "majorana",
          [](const MajoranaMapping& self,
             std::size_t k) -> const SparsePauliWord& {
            try {
              return self.majorana(k);
            } catch (const std::out_of_range& e) {
              throw py::index_error(e.what());
            } catch (const std::logic_error& e) {
              throw py::value_error(e.what());
            }
          },
          py::arg("k"), py::return_value_policy::reference_internal)
      .def(
          "bilinear",
          [](const MajoranaMapping& self, std::size_t j, std::size_t k) {
            auto [coeff, word] = self.bilinear(j, k);
            return py::make_tuple(coeff, word);
          },
          py::arg("j"), py::arg("k"))
      .def("without_tapering", &MajoranaMapping::without_tapering)
      .def("__repr__", [](const MajoranaMapping& self) {
        std::string repr = "MajoranaMapping(";
        if (!self.name().empty()) {
          repr += "'" + self.name() + "', ";
        }
        repr += "num_modes=" + std::to_string(self.num_modes()) +
                ", num_qubits=" + std::to_string(self.num_qubits()) + ")";
        return repr;
      });

  mapping
      .def_static("jordan_wigner", &MajoranaMapping::jordan_wigner,
                  py::arg("num_modes"), "Construct a Jordan-Wigner encoding.")
      .def_static("bravyi_kitaev", &MajoranaMapping::bravyi_kitaev,
                  py::arg("num_modes"), "Construct a Bravyi-Kitaev encoding.")
      .def_static("bravyi_kitaev_tree", &MajoranaMapping::bravyi_kitaev_tree,
                  py::arg("num_modes"),
                  "Construct a balanced binary-tree Bravyi-Kitaev encoding.")
      .def_static(
          "parity",
          [](std::size_t num_modes) {
            return MajoranaMapping::parity(num_modes);
          },
          py::arg("num_modes"), "Construct a parity encoding.")
      .def_static(
          "parity",
          [](std::size_t num_modes, const py::object& symmetries) {
            auto n_alpha = symmetries.attr("n_alpha").cast<std::size_t>();
            auto n_beta = symmetries.attr("n_beta").cast<std::size_t>();
            return MajoranaMapping::parity(num_modes, n_alpha, n_beta);
          },
          py::arg("num_modes"), py::arg("symmetries"),
          "Construct a parity encoding with two-qubit reduction metadata.")
      .def_static(
          "symmetry_conserving_bravyi_kitaev",
          [](std::size_t num_modes, const py::object& symmetries) {
            auto n_alpha = symmetries.attr("n_alpha").cast<std::size_t>();
            auto n_beta = symmetries.attr("n_beta").cast<std::size_t>();
            return MajoranaMapping::symmetry_conserving_bravyi_kitaev(
                num_modes, n_alpha, n_beta);
          },
          py::arg("num_modes"), py::arg("symmetries"),
          "Construct a symmetry-conserving Bravyi-Kitaev encoding.");

  mapping
      .def(
          "to_json",
          [](const MajoranaMapping& self) {
            py::module_ json = py::module_::import("json");
            return json.attr("loads")(self.to_json().dump());
          },
          "Serialize to a JSON-compatible dictionary.")
      .def_static(
          "from_json",
          [](const py::object& json_data) {
            py::module_ json = py::module_::import("json");
            return MajoranaMapping::from_json(nlohmann::json::parse(
                json.attr("dumps")(json_data).cast<std::string>()));
          },
          py::arg("json_data"))
      .def(
          "to_hdf5",
          [](const MajoranaMapping& self, const py::object& group) {
            group.attr("attrs").attr("__setitem__")("json",
                                                    self.to_json().dump());
          },
          py::arg("group"))
      .def_static(
          "from_hdf5",
          [](const py::object& group) {
            auto json = group.attr("attrs")
                            .attr("__getitem__")("json")
                            .cast<std::string>();
            return MajoranaMapping::from_json(nlohmann::json::parse(json));
          },
          py::arg("group"))
      .def(
          "to_json_file",
          [](const MajoranaMapping& self, const py::object& filename) {
            self.to_json_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          py::arg("filename"))
      .def_static(
          "from_json_file",
          [](const py::object& filename) {
            return MajoranaMapping::from_json_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          py::arg("filename"))
      .def(
          "to_hdf5_file",
          [](const MajoranaMapping& self, const py::object& filename) {
            self.to_hdf5_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          py::arg("filename"))
      .def_static(
          "from_hdf5_file",
          [](const py::object& filename) {
            return MajoranaMapping::from_hdf5_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          py::arg("filename"));

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
         std::size_t n_spatial, bool spin_symmetric, double threshold,
         double integral_threshold) -> py::tuple {
        auto result = majorana_map_hamiltonian(
            mapping, core_energy, h1_alpha.data(), h1_beta.data(),
            eri_aaaa.data(), eri_aabb.data(), eri_bbbb.data(), n_spatial,
            spin_symmetric, threshold, integral_threshold);
        return py::make_tuple(py::cast(result.words),
                              py::cast(result.coefficients));
      },
      py::arg("mapping"), py::arg("core_energy"), py::arg("h1_alpha"),
      py::arg("h1_beta"), py::arg("eri_aaaa"), py::arg("eri_aabb"),
      py::arg("eri_bbbb"), py::arg("n_spatial"), py::arg("spin_symmetric"),
      py::arg("threshold"), py::arg("integral_threshold"),
      R"(
Map a fermionic Hamiltonian to qubit Pauli terms using Majorana loops.

When ``spin_symmetric`` is true, uses a spin-summed fast path that assumes
identical integrals across spin channels (restricted orbitals). When false,
handles each spin channel independently (unrestricted orbitals).

Returns ``(words, coefficients)`` where ``words`` is a list of sparse
Pauli words.
)");

  // Overload taking a Hamiltonian directly.  The two-body integrals are
  // consumed in the native storage format of the underlying container:
  // Cholesky containers feed their three-center factors and sparse
  // containers their non-zero integral list straight into the engine, so
  // the dense N^4 two-body tensor is never materialized.  Any other
  // container uses the dense engine path.  The result is numerically
  // equivalent to the dense path for the same integrals.
  data.def(
      "majorana_map_hamiltonian",
      [](const MajoranaMapping& mapping, const Hamiltonian& hamiltonian,
         bool spin_symmetric, double threshold,
         double integral_threshold) -> py::tuple {
        auto one_body = hamiltonian.get_one_body_integrals();
        const Eigen::MatrixXd& h1a = std::get<0>(one_body);
        const Eigen::MatrixXd& h1b = std::get<1>(one_body);
        const std::size_t n = static_cast<std::size_t>(h1a.rows());

        // Flatten one-body integrals to row-major [n*n] as the engine expects.
        std::vector<double> h1a_flat(n * n);
        std::vector<double> h1b_flat(n * n);
        for (std::size_t p = 0; p < n; ++p) {
          for (std::size_t s = 0; s < n; ++s) {
            h1a_flat[p * n + s] =
                h1a(static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(s));
            h1b_flat[p * n + s] =
                h1b(static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(s));
          }
        }

        MajoranaMapResult result;
        if (hamiltonian.has_container_type<CholeskyHamiltonianContainer>()) {
          const auto& container =
              hamiltonian.get_container<CholeskyHamiltonianContainer>();
          auto three_center = container.get_three_center_integrals();
          const Eigen::MatrixXd& three_center_aa = three_center.first;
          const Eigen::MatrixXd& three_center_bb = three_center.second;
          const std::size_t naux =
              static_cast<std::size_t>(three_center_aa.cols());
          result = majorana_map_hamiltonian_cholesky(
              mapping, 0.0, h1a_flat.data(), h1b_flat.data(),
              three_center_aa.data(), three_center_bb.data(), n, naux,
              spin_symmetric, threshold, integral_threshold);
        } else if (hamiltonian
                       .has_container_type<SparseHamiltonianContainer>()) {
          const auto& container =
              hamiltonian.get_container<SparseHamiltonianContainer>();
          const auto& two_body_map = container.sparse_two_body_integrals();
          std::vector<int> indices;
          std::vector<double> values;
          indices.reserve(two_body_map.size() * 4);
          values.reserve(two_body_map.size());
          for (const auto& [idx, val] : two_body_map) {
            const auto& [p, q, r, s] = idx;
            indices.push_back(p);
            indices.push_back(q);
            indices.push_back(r);
            indices.push_back(s);
            values.push_back(val);
          }
          result = majorana_map_hamiltonian_sparse(
              mapping, 0.0, h1a_flat.data(), h1b_flat.data(), indices.data(),
              values.data(), values.size(), n, spin_symmetric, threshold,
              integral_threshold);
        } else {
          // Dense path for canonical four-center (and any other) container.
          auto two_body = hamiltonian.get_two_body_integrals();
          const Eigen::VectorXd& aaaa = std::get<0>(two_body);
          const Eigen::VectorXd& aabb = std::get<1>(two_body);
          const Eigen::VectorXd& bbbb = std::get<2>(two_body);
          result = majorana_map_hamiltonian(
              mapping, 0.0, h1a_flat.data(), h1b_flat.data(), aaaa.data(),
              aabb.data(), bbbb.data(), n, spin_symmetric, threshold,
              integral_threshold);
        }

        return py::make_tuple(py::cast(result.words),
                              py::cast(result.coefficients));
      },
      py::arg("mapping"), py::arg("hamiltonian"), py::arg("spin_symmetric"),
      py::arg("threshold"), py::arg("integral_threshold"),
      R"(
Map a fermionic Hamiltonian to qubit Pauli terms.

The two-body integrals are consumed in the native storage format of the
Hamiltonian's container: Cholesky (three-center) and sparse containers are
read directly without materializing the dense N^4 two-body tensor; any
other container uses the dense path.  The returned ``(words,
coefficients)`` are numerically equivalent to the dense path for the same
underlying integrals.

Args:
    mapping (MajoranaMapping): The Majorana-to-Pauli encoding (no tapering).
    hamiltonian (Hamiltonian): The fermionic Hamiltonian.
    spin_symmetric (bool): Use the spin-summed restricted fast path.
    threshold (float): Drop Pauli terms with magnitude below this value.
    integral_threshold (float): Skip integrals with magnitude below this value.

Returns:
    tuple: ``(words, coefficients)`` where ``words`` is a list of sparse Pauli
    words and ``coefficients`` is a list of complex coefficients.
)");
}
