// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>

#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/lattice_graph.hpp>
#include <qdk/chemistry/data/majorana_mapping.hpp>
#include <qdk/chemistry/data/pauli_operator.hpp>
#include <qdk/chemistry/data/tapering.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "path_utils.hpp"
#include "stl.hpp"

namespace nb = nanobind;

void bind_majorana_mapping(nb::module_& data) {
  using namespace qdk::chemistry::data;

  nb::class_<TaperingSpecification, DataClass> tapering(
      data, "TaperingSpecification", R"(
Immutable specification for post-mapping qubit tapering.
)");

  tapering
      .def(nb::init<std::vector<std::size_t>, std::vector<int>>(),
           nb::arg("qubit_indices"), nb::arg("eigenvalues"))
      .def_prop_ro("qubit_indices", &TaperingSpecification::qubit_indices)
      .def_prop_ro("eigenvalues", &TaperingSpecification::eigenvalues)
      .def_prop_ro("num_tapered", &TaperingSpecification::num_tapered)
      .def_static(
          "symmetry_conserving_bravyi_kitaev",
          [](std::size_t num_modes, const nb::object& symmetries) {
            auto n_alpha = nb::cast<std::size_t>(symmetries.attr("n_alpha"));
            auto n_beta = nb::cast<std::size_t>(symmetries.attr("n_beta"));
            return TaperingSpecification::symmetry_conserving_bravyi_kitaev(
                num_modes, n_alpha, n_beta);
          },
          nb::arg("num_modes"), nb::arg("symmetries"))
      .def_static(
          "parity_two_qubit_reduction",
          [](std::size_t num_modes, const nb::object& symmetries) {
            auto n_alpha = nb::cast<std::size_t>(symmetries.attr("n_alpha"));
            auto n_beta = nb::cast<std::size_t>(symmetries.attr("n_beta"));
            return TaperingSpecification::parity_two_qubit_reduction(
                num_modes, n_alpha, n_beta);
          },
          nb::arg("num_modes"), nb::arg("symmetries"))
      .def("__eq__", &TaperingSpecification::operator==, nb::arg("other"))
      .def("__hash__",
           [](const TaperingSpecification& self) {
             return nb::hash(
                 nb::make_tuple(nb::tuple(nb::cast(self.qubit_indices())),
                                nb::tuple(nb::cast(self.eigenvalues()))));
           })
      .def(
          "to_json",
          [](const TaperingSpecification& self) {
            nb::module_ json = nb::module_::import_("json");
            return json.attr("loads")(self.to_json().dump());
          },
          "Serialize to a JSON-compatible dictionary.")
      .def_static(
          "from_json",
          [](const nb::object& json_data) {
            nb::module_ json = nb::module_::import_("json");
            return TaperingSpecification::from_json(nlohmann::json::parse(
                nb::cast<std::string>(json.attr("dumps")(json_data))));
          },
          nb::arg("json_data"),
          "Deserialize from a JSON-compatible dictionary.")
      .def(
          "to_hdf5",
          [](const TaperingSpecification& self, const nb::object& group) {
            group.attr("attrs").attr("__setitem__")("json",
                                                    self.to_json().dump());
          },
          nb::arg("group"), "Serialize to an HDF5 group.")
      .def_static(
          "from_hdf5",
          [](const nb::object& group) {
            auto json = nb::cast<std::string>(
                group.attr("attrs").attr("__getitem__")("json"));
            return TaperingSpecification::from_json(
                nlohmann::json::parse(json));
          },
          nb::arg("group"), "Deserialize from an HDF5 group.")
      .def(
          "to_json_file",
          [](const TaperingSpecification& self, const nb::object& filename) {
            self.to_json_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          nb::arg("filename"))
      .def_static(
          "from_json_file",
          [](const nb::object& filename) {
            return TaperingSpecification::from_json_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          nb::arg("filename"))
      .def(
          "to_hdf5_file",
          [](const TaperingSpecification& self, const nb::object& filename) {
            self.to_hdf5_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          nb::arg("filename"))
      .def_static(
          "from_hdf5_file",
          [](const nb::object& filename) {
            return TaperingSpecification::from_hdf5_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          nb::arg("filename"));

  tapering.def_static("data_type_name", &TaperingSpecification::data_type_name,
                      R"(
Return the wire-format identifier for tapering specifications.

Returns:
    str: ``"tapering_specification"``

)");

  nb::class_<MajoranaMapping, DataClass> mapping(data, "MajoranaMapping", R"(
Fermion-to-qubit encoding.

Majorana-atomic mappings store a 2N-entry Pauli table for individual gamma_k;
bilinear-only mappings (via from_bilinears) store the bilinear images directly.
bilinear(j, k) is available on both forms.
)");

  mapping
      .def_static("from_table", &MajoranaMapping::from_table, nb::arg("table"),
                  nb::arg("name") = "",
                  "Construct a Majorana-atomic mapping from sparse Pauli "
                  "words.")
      .def_static("from_bilinears", &MajoranaMapping::from_bilinears,
                  nb::arg("num_modes"), nb::arg("bilinears"),
                  nb::arg("name") = "",
                  "Construct a bilinear-only mapping from pre-computed "
                  "bilinears.")
      .def_prop_ro("num_modes", &MajoranaMapping::num_modes)
      .def_prop_ro("num_qubits", &MajoranaMapping::num_qubits)
      .def_prop_ro("name", &MajoranaMapping::name)
      .def_prop_ro("base_encoding", &MajoranaMapping::base_encoding)
      .def_prop_ro("table", &MajoranaMapping::table)
      .def_prop_ro("tapering",
                   [](const MajoranaMapping& self)
                       -> std::optional<TaperingSpecification> {
                     return self.tapering();
                   })
      .def_prop_ro("stabilizers", &MajoranaMapping::stabilizers)
      .def_prop_ro("is_majorana_atomic", &MajoranaMapping::is_majorana_atomic)
      .def(
          "__call__",
          [](const MajoranaMapping& self,
             std::size_t k) -> const SparsePauliWord& {
            try {
              return self.majorana(k);
            } catch (const std::out_of_range& e) {
              throw nb::index_error(std::string(e.what()).c_str());
            } catch (const std::logic_error& e) {
              throw nb::value_error(std::string(e.what()).c_str());
            }
          },
          nb::arg("k"), nb::rv_policy::reference_internal)
      .def(
          "majorana",
          [](const MajoranaMapping& self,
             std::size_t k) -> const SparsePauliWord& {
            try {
              return self.majorana(k);
            } catch (const std::out_of_range& e) {
              throw nb::index_error(std::string(e.what()).c_str());
            } catch (const std::logic_error& e) {
              throw nb::value_error(std::string(e.what()).c_str());
            }
          },
          nb::arg("k"), nb::rv_policy::reference_internal)
      .def(
          "bilinear",
          [](const MajoranaMapping& self, std::size_t j, std::size_t k) {
            auto [coeff, word] = self.bilinear(j, k);
            return nb::make_tuple(coeff, word);
          },
          nb::arg("j"), nb::arg("k"))
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
                  nb::arg("num_modes"), "Construct a Jordan-Wigner encoding.")
      .def_static("bravyi_kitaev", &MajoranaMapping::bravyi_kitaev,
                  nb::arg("num_modes"), "Construct a Bravyi-Kitaev encoding.")
      .def_static("bravyi_kitaev_tree", &MajoranaMapping::bravyi_kitaev_tree,
                  nb::arg("num_modes"),
                  "Construct a balanced binary-tree Bravyi-Kitaev encoding.")
      .def_static(
          "parity",
          [](std::size_t num_modes) {
            return MajoranaMapping::parity(num_modes);
          },
          nb::arg("num_modes"), "Construct a parity encoding.")
      .def_static(
          "parity",
          [](std::size_t num_modes, const nb::object& symmetries) {
            auto n_alpha = nb::cast<std::size_t>(symmetries.attr("n_alpha"));
            auto n_beta = nb::cast<std::size_t>(symmetries.attr("n_beta"));
            return MajoranaMapping::parity(num_modes, n_alpha, n_beta);
          },
          nb::arg("num_modes"), nb::arg("symmetries"),
          "Construct a parity encoding with two-qubit reduction "
          "metadata.")
      .def_static(
          "symmetry_conserving_bravyi_kitaev",
          [](std::size_t num_modes, const nb::object& symmetries) {
            auto n_alpha = nb::cast<std::size_t>(symmetries.attr("n_alpha"));
            auto n_beta = nb::cast<std::size_t>(symmetries.attr("n_beta"));
            return MajoranaMapping::symmetry_conserving_bravyi_kitaev(
                num_modes, n_alpha, n_beta);
          },
          nb::arg("num_modes"), nb::arg("symmetries"),
          "Construct a symmetry-conserving Bravyi-Kitaev encoding.")
      .def_static("verstraete_cirac", &MajoranaMapping::verstraete_cirac,
                  nb::arg("lattice"), "Construct a Verstraete-Cirac encoding.");

  mapping
      .def(
          "to_json",
          [](const MajoranaMapping& self) {
            nb::module_ json = nb::module_::import_("json");
            return json.attr("loads")(self.to_json().dump());
          },
          "Serialize to a JSON-compatible dictionary.")
      .def_static(
          "from_json",
          [](const nb::object& json_data) {
            nb::module_ json = nb::module_::import_("json");
            return MajoranaMapping::from_json(nlohmann::json::parse(
                nb::cast<std::string>(json.attr("dumps")(json_data))));
          },
          nb::arg("json_data"))
      .def(
          "to_hdf5",
          [](const MajoranaMapping& self, const nb::object& group) {
            group.attr("attrs").attr("__setitem__")("json",
                                                    self.to_json().dump());
          },
          nb::arg("group"))
      .def_static(
          "from_hdf5",
          [](const nb::object& group) {
            auto json = nb::cast<std::string>(
                group.attr("attrs").attr("__getitem__")("json"));
            return MajoranaMapping::from_json(nlohmann::json::parse(json));
          },
          nb::arg("group"))
      .def(
          "to_json_file",
          [](const MajoranaMapping& self, const nb::object& filename) {
            self.to_json_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          nb::arg("filename"))
      .def_static(
          "from_json_file",
          [](const nb::object& filename) {
            return MajoranaMapping::from_json_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          nb::arg("filename"))
      .def(
          "to_hdf5_file",
          [](const MajoranaMapping& self, const nb::object& filename) {
            self.to_hdf5_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          nb::arg("filename"))
      .def_static(
          "from_hdf5_file",
          [](const nb::object& filename) {
            return MajoranaMapping::from_hdf5_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          nb::arg("filename"));

  mapping.def_static("data_type_name", &MajoranaMapping::data_type_name, R"(
Return the wire-format identifier for Majorana mappings.

Returns:
    str: ``"majorana_mapping"``

)");

  data.def(
      "majorana_map_hamiltonian",
      [](const MajoranaMapping& mapping, double core_energy,
         nb::ndarray<nb::numpy, const double, nb::c_contig> h1_alpha,
         nb::ndarray<nb::numpy, const double, nb::c_contig> h1_beta,
         nb::ndarray<nb::numpy, const double, nb::c_contig> eri_aaaa,
         nb::ndarray<nb::numpy, const double, nb::c_contig> eri_aabb,
         nb::ndarray<nb::numpy, const double, nb::c_contig> eri_bbbb,
         std::size_t n_spatial, bool spin_symmetric, double threshold,
         double integral_threshold) -> nb::tuple {
        auto result = majorana_map_hamiltonian(
            mapping, core_energy, h1_alpha.data(), h1_beta.data(),
            eri_aaaa.data(), eri_aabb.data(), eri_bbbb.data(), n_spatial,
            spin_symmetric, threshold, integral_threshold);
        return nb::make_tuple(nb::cast(result.words),
                              nb::cast(result.coefficients));
      },
      nb::arg("mapping"), nb::arg("core_energy"), nb::arg("h1_alpha"),
      nb::arg("h1_beta"), nb::arg("eri_aaaa"), nb::arg("eri_aabb"),
      nb::arg("eri_bbbb"), nb::arg("n_spatial"), nb::arg("spin_symmetric"),
      nb::arg("threshold"), nb::arg("integral_threshold"),
      R"(
Map a fermionic Hamiltonian to qubit Pauli terms using Majorana loops.

When ``spin_symmetric`` is true, uses a spin-summed fast path that assumes
identical integrals across spin channels (restricted orbitals). When false,
handles each spin channel independently (unrestricted orbitals).

Returns ``(words, coefficients)`` where ``words`` is a list of sparse
Pauli words.
)");

  // Overload taking a Hamiltonian directly.  Container dispatch and
  // sparse index marshalling live in the C++ library overload.
  data.def(
      "majorana_map_hamiltonian",
      [](const MajoranaMapping& mapping, const Hamiltonian& hamiltonian,
         bool spin_symmetric, double threshold,
         double integral_threshold) -> nb::tuple {
        const MajoranaMapResult result =
            majorana_map_hamiltonian(mapping, hamiltonian, spin_symmetric,
                                     threshold, integral_threshold);
        return nb::make_tuple(nb::cast(result.words),
                              nb::cast(result.coefficients));
      },
      nb::arg("mapping"), nb::arg("hamiltonian"), nb::arg("spin_symmetric"),
      nb::arg("threshold"), nb::arg("integral_threshold"),
      R"(
Map a fermionic Hamiltonian to qubit Pauli terms.

The two-body integrals are consumed in the native storage format of the
Hamiltonian's container: Cholesky (three-center) and sparse containers are
read directly without materializing the dense N^4 two-body tensor; any
other container uses the dense path.

For Cholesky and dense containers, the returned ``(words, coefficients)``
match the dense path for the same underlying integrals.  For sparse
containers, stored entries are canonicalized and symmetry-expanded before
mapping, so results agree with the dense path when integrals are stored
canonically or symmetry-complete (as for in-repo model builders); the
sparse fast path is more robust when only non-canonical symmetry
representatives are stored.

The Hamiltonian's constant energy shift (nuclear repulsion / frozen core)
is intentionally **not** included in the mapped operator, matching the
buffer-based overload as invoked by ``QdkQubitMapper`` (which always
passes ``core_energy=0.0``); callers that need the constant term must add
it separately.

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
