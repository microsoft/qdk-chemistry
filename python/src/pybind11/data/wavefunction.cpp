// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <nlohmann/json.hpp>
#include <numeric>
#include <qdk/chemistry.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/cas.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sci.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>

#include "property_binding_helpers.hpp"

namespace py = pybind11;

// Helper functions to convert variant types to Python objects
py::object variant_to_python(
    const qdk::chemistry::data::ContainerTypes::ScalarVariant& var) {
  return std::visit(
      [](const auto& value) -> py::object { return py::cast(value); }, var);
}

py::object variant_to_python(
    const qdk::chemistry::data::ContainerTypes::MatrixVariant& var) {
  return std::visit(
      [](const auto& value) -> py::object { return py::cast(value); }, var);
}

py::object variant_to_python(
    const qdk::chemistry::data::ContainerTypes::VectorVariant& var) {
  return std::visit(
      [](const auto& value) -> py::object { return py::cast(value); }, var);
}

void bind_wavefunction(pybind11::module& data) {
  using namespace qdk::chemistry::algorithms;
  using namespace qdk::chemistry::data;
  using qdk::chemistry::python::utils::bind_getter_as_property;

  // Bind WavefunctionType enum
  py::enum_<WavefunctionType>(data, "WavefunctionType",
                              R"(
Enum to distinguish between different wavefunction representations.

This enum allows tagging wavefunctions based on their mathematical role:

* SelfDual: Wavefunctions that can be used as both bra and ket
* NotSelfDual: Wavefunctions that are strictly bra or ket
)")
      .value("SelfDual", WavefunctionType::SelfDual,
             "Wavefunction that can be used as both bra and ket")
      .value("NotSelfDual", WavefunctionType::NotSelfDual,
             "Wavefunction that is strictly bra or ket")
      .export_values();

  // Bind abstract WavefunctionContainer base class
  py::class_<WavefunctionContainer, py::smart_holder>(data,
                                                      "WavefunctionContainer",
                                                      R"(
Abstract base class for wavefunction containers.

This class provides the interface for different types of wavefunction representations (e.g., CI, MCSCF, coupled cluster).
It uses variant types to support both real and complex arithmetic.
    )")
      .def("get_coefficient", &WavefunctionContainer::get_coefficient,
           "Get coefficient for a specific determinant", py::arg("det"))
      .def("get_active_determinants",
           &WavefunctionContainer::get_active_determinants,
           "Get all determinants in the wavefunction",
           py::return_value_policy::reference_internal)
      .def("size", &WavefunctionContainer::size, "Get number of determinants")
      .def("norm", &WavefunctionContainer::norm,
           "Calculate norm of the wavefunction")
      .def("get_orbitals", &WavefunctionContainer::get_orbitals,
           "Get reference to orbital basis set")
      .def("get_type", &WavefunctionContainer::get_type,
           "Get the wavefunction type (Bra, Ket, or Both)")
      .def("get_active_num_electrons",
           &WavefunctionContainer::get_active_num_electrons,
           "Get number of active alpha and beta electrons")
      .def("get_total_num_electrons",
           &WavefunctionContainer::get_total_num_electrons,
           "Get total number of alpha and beta electrons")
      .def("get_orbital_occupations",
           &WavefunctionContainer::get_active_orbital_occupations,
           "Get orbital occupations for active orbitals")
      .def("get_active_orbital_occupations",
           &WavefunctionContainer::get_total_orbital_occupations,
           "Get orbital occupations for all orbitals")
      .def("get_active_orbital_occupations",
           &WavefunctionContainer::get_active_orbital_occupations,
           "Get orbital occupations for active orbitals only")
      .def("has_one_rdm_spin_dependent",
           &WavefunctionContainer::has_one_rdm_spin_dependent,
           "Check if spin-dependent one-particle RDMs for active orbitals are "
           "available")
      .def("has_one_rdm_spin_traced",
           &WavefunctionContainer::has_one_rdm_spin_traced,
           "Check if spin-traced one-particle RDM for active orbitals is "
           "available")
      .def("has_two_rdm_spin_dependent",
           &WavefunctionContainer::has_two_rdm_spin_dependent,
           "Check if spin-dependent two-particle RDMs for active orbitals are "
           "available")
      .def("has_two_rdm_spin_traced",
           &WavefunctionContainer::has_two_rdm_spin_traced,
           "Check if spin-traced two-particle RDM for active orbitals is "
           "available")
      .def("is_complex", &WavefunctionContainer::is_complex,
           "Check if the wavefunction is complex-valued");

  // Wavefunction class
  py::class_<Wavefunction, DataClass, py::smart_holder> wavefunction(
      data, "Wavefunction",
      R"(
Represents a wavefunction with associated properties.

This class encapsulates wavefunction data including:

* Configuration interaction coefficients
* Reduced density matrices (RDMs) in spin-dependent and spin-traced forms
* Methods for computing properties

Supports both restricted and unrestricted wavefunctions with real or complex coefficients.
    )");

  // Constructor with container
  wavefunction.def(py::init<std::unique_ptr<WavefunctionContainer>>(),
                   R"(
Constructs a wavefunction with a container implementation.

Args:
    container (WavefunctionContainer): The container holding the wavefunction implementation

Examples:
    >>> container = qdk_chemistry.SciWavefunctionContainer(coeffs, dets, orbitals)
    >>> wf = qdk_chemistry.Wavefunction(container)
)",
                   py::arg("container"));

  // Orbital and container access
  bind_getter_as_property(wavefunction, "get_orbitals",
                          &Wavefunction::get_orbitals,
                          R"(
Get reference to orbital basis set.

Returns:
    Orbitals: Shared pointer to orbital basis set

Examples:
    >>> orbitals = wf.get_orbitals()
)",
                          py::return_value_policy::reference_internal);

  wavefunction.def(
      "get_container",
      [](const Wavefunction& self) -> const WavefunctionContainer& {
        if (!self.has_container_type<WavefunctionContainer>()) {
          throw std::runtime_error(
              "The wavefunction does not have a valid container.");
        }
        if (self.has_container_type<SciWavefunctionContainer>()) {
          return self.get_container<SciWavefunctionContainer>();
        } else if (self.has_container_type<CasWavefunctionContainer>()) {
          return self.get_container<CasWavefunctionContainer>();
        } else if (self.has_container_type<SlaterDeterminantContainer>()) {
          return self.get_container<SlaterDeterminantContainer>();
        } else {
          throw std::runtime_error("Unknown container type.");
        }
      },
      R"(
Get the underlying wavefunciton container.

Returns:
    WavefunctionContainer: The underlying wavefunction container

Raises:
    RuntimeError: If the container is not available

Examples:
    >>> container = wf.get_container()
)",
      py::return_value_policy::reference_internal);

  // Electron and occupation methods
  wavefunction.def("get_active_num_electrons",
                   &Wavefunction::get_active_num_electrons,
                   R"(
Get number of active alpha and beta electrons.

Returns:
    tuple: Pair of ``(n_alpha, n_beta)`` electrons

Examples:
    >>> n_alpha, n_beta = wf.get_active_num_electrons()
)");

  wavefunction.def("get_total_num_electrons",
                   &Wavefunction::get_total_num_electrons,
                   R"(
Get total number of alpha and beta electrons.

Returns:
    tuple: Pair of ``(n_alpha, n_beta)`` electrons

Examples:
    >>> n_alpha, n_beta = wf.get_total_num_electrons()
)");

  wavefunction.def("get_total_orbital_occupations",
                   &Wavefunction::get_total_orbital_occupations,
                   R"(
Get orbital occupations for all orbitals.

Returns:
    tuple: Pair of ``(alpha_occupations, beta_occupations)``

Examples:
    >>> alpha_occ, beta_occ = wf.get_total_orbital_occupations()
)");

  wavefunction.def("get_active_orbital_occupations",
                   &Wavefunction::get_active_orbital_occupations,
                   R"(
Get orbital occupations for active orbitals only.

Returns:
    tuple: Pair of ``(alpha_active_occupations, beta_active_occupations)``

Examples:
    >>> alpha_active, beta_active = wf.get_active_orbital_occupations()
)");

  // Coefficient access methods
  wavefunction.def(
      "get_coefficient",
      [](const Wavefunction& self, const Configuration& det) {
        return variant_to_python(self.get_coefficient(det));
      },
      R"(
Get coefficient for a specific determinant.

Args:
    det (Configuration): The determinant configuration to look up

Returns:
    complex | float: The coefficient for the determinant if found, zero otherwise

Examples:
    >>> coeff = wf.get_coefficient(qdk_chemistry.Configuration("33221100"))
)",
      py::arg("det"));

  wavefunction.def(
      "get_coefficients",
      [](const Wavefunction& self) {
        return variant_to_python(self.get_coefficients());
      },
      R"(
Get coefficients for all determinants as a vector, in which the sequence of coefficients is consistent with the vector from ``get_active_determinants()``

Returns:
    numpy.ndarray: Vector of all coefficients (real or complex)

Examples:
    >>> coeffs = wf.get_coefficients()
    >>> print(f"Number of coefficients: {len(coeffs)}")
)");

  wavefunction.def("get_active_determinants",
                   &Wavefunction::get_active_determinants,
                   R"(
Get all determinants in the wavefunction (active space only).

Returns:
    list[Configuration]: Vector of all configurations/determinants representing only the active space

Notes:
    The determinants only include the active space orbitals.
    To get determinants with full orbital space (including inactive and virtual orbitals), use ``get_total_determinants()``.

Examples:
    >>> dets = wf.get_active_determinants()
)",
                   py::return_value_policy::reference_internal);

  wavefunction.def("get_total_determinants",
                   &Wavefunction::get_total_determinants,
                   R"(
Get all determinants in the wavefunction with full orbital space.

Returns:
    list[Configuration]: Vector of all configurations/determinants including inactive and virtual orbitals

Notes:
    Converts active-space-only determinants to full orbital space by prepending doubly occupied inactive orbitals and appending unoccupied virtual orbitals.

Examples:
    >>> total_dets = wf.get_total_determinants()
    >>> # Each determinant now includes inactive (doubly occupied) and virtual (unoccupied) orbitals
)");

  wavefunction.def("get_active_determinant",
                   &Wavefunction::get_active_determinant,
                   R"(
Extract active space determinant from a full orbital space determinant.

Args:
    total_determinant (Configuration): Configuration representing full orbital space

Returns:
    Configuration: Configuration representing only the active space portion

Notes:
    Removes inactive and virtual orbital information, keeping only the active space orbitals.

Examples:
    >>> total_det = qdk_chemistry.Configuration("2222uudd0000")  # 4 inactive, 4 active, 4 virtual
    >>> active_det = wf.get_active_determinant(total_det)
    >>> # active_det now contains only "uudd" (the 4 active orbitals)
)",
                   py::arg("total_determinant"));

  wavefunction.def("get_total_determinant",
                   &Wavefunction::get_total_determinant,
                   R"(
Convert active space determinant to full orbital space determinant.

Args:
    active_determinant (Configuration): Configuration representing only active space

Returns:
    Configuration: Configuration representing full orbital space

Notes:
    Expands active-space-only determinant to full orbital space by prepending doubly occupied inactive orbitals and appending unoccupied virtual orbitals.

Examples:
    >>> active_det = qdk_chemistry.Configuration("uudd")  # 4 active orbitals
    >>> total_det = wf.get_total_determinant(active_det)
    >>> # If there are 4 inactive and 4 virtual orbitals, total_det will be "2222uudd0000"
)",
                   py::arg("active_determinant"));

  wavefunction.def("size", &Wavefunction::size,
                   R"(
Get number of determinants.

Returns:
    int: Number of determinants in the wavefunction

Examples:
    >>> dim = wf.size()
    >>> print(f"Wavefunction dimension: {dim}")
)");

  wavefunction.def("norm", &Wavefunction::norm,
                   R"(
Calculate norm of the wavefunction.

Returns:
    float: Norm (always real)

Examples:
    >>> norm_value = wf.norm()
    >>> print(f"Wavefunction norm: {norm_value}")
)");

  wavefunction.def(
      "overlap",
      [](const Wavefunction& self, const Wavefunction& other) {
        return variant_to_python(self.overlap(other));
      },
      R"(
Calculate overlap with another wavefunction.

Args:
    other (Wavefunction): Other wavefunction

Returns:
    complex | float: Overlap value (real or complex)

Examples:
    >>> wf1 = qdk_chemistry.Wavefunction(container1)
    >>> wf2 = qdk_chemistry.Wavefunction(container2)
    >>> overlap = wf1.overlap(wf2)
    >>> print(f"Overlap: {overlap}")
)",
      py::arg("other"));

  // RDM methods
  wavefunction.def(
      "get_active_one_rdm_spin_dependent",
      [](const Wavefunction& self) {
        auto [aa, bb] = self.get_active_one_rdm_spin_dependent();
        return py::make_tuple(variant_to_python(aa), variant_to_python(bb));
      },
      R"(
Get spin-dependent one-particle reduced density matrices (RDMs) for active orbitals only.

Returns:
    tuple: Tuple of (alpha-alpha, beta-beta) one-particle RDMs for active orbitals

Raises:
    RuntimeError: If the spin-dependent 1-RDM is not available

Examples:
    >>> rdm_aa, rdm_bb = wf.get_active_one_rdm_spin_dependent()
)");

  wavefunction.def(
      "get_active_two_rdm_spin_dependent",
      [](const Wavefunction& self) {
        auto [aaaa, aabb, bbbb] = self.get_active_two_rdm_spin_dependent();
        return py::make_tuple(variant_to_python(aaaa), variant_to_python(aabb),
                              variant_to_python(bbbb));
      },
      R"(
Get spin-dependent two-particle reduced density matrices (RDMs) for active orbitals only.

Returns:
    tuple: Tuple of (aaaa, aabb, bbbb) two-particle RDMs for active orbitals

Raises:
    RuntimeError: If the spin-dependent 2-RDM is not available

Examples:
    >>> aaaa, aabb, bbbb = wf.get_active_two_rdm_spin_dependent()
)");

  wavefunction.def(
      "get_active_one_rdm_spin_traced",
      [](const Wavefunction& self) {
        return variant_to_python(self.get_active_one_rdm_spin_traced());
      },
      R"(
Get spin-traced one-particle reduced density matrix (RDM) for active orbitals only.

Returns:
    numpy.ndarray: Spin-traced one-particle RDM for active orbitals

Raises:
    RuntimeError: If the 1-RDM is not available

Examples:
    >>> rdm = wf.get_active_one_rdm_spin_traced()
)");

  wavefunction.def(
      "get_active_two_rdm_spin_traced",
      [](const Wavefunction& self) {
        return variant_to_python(self.get_active_two_rdm_spin_traced());
      },
      R"(
Get spin-traced two-particle reduced density matrix (RDM) for active orbitals only.

Returns:
    numpy.ndarray: Spin-traced two-particle RDM for active orbitals

Raises:
    RuntimeError: If the 2-RDM is not available

Examples:
    >>> two_rdm = wf.get_active_two_rdm_spin_traced()
)");

  // TODO (NAB): it would be helpful to explain how to mark or check whether
  // orbitals are active. Same comment applies to other methods that refer to
  // "active orbitals". Workitem: 41398

  wavefunction.def("get_single_orbital_entropies",
                   &Wavefunction::get_single_orbital_entropies,
                   R"(
Calculate single orbital entropies for active orbitals only.

Returns:
    numpy.ndarray: Vector of orbital entropies for active orbitals (always real)

Raises:
    RuntimeError: If the required reduced density matrices (RDMs) are not available

Examples:
    >>> entropies = wf.get_single_orbital_entropies()
)");

  // RDM availability check methods
  wavefunction.def("has_one_rdm_spin_dependent",
                   &Wavefunction::has_one_rdm_spin_dependent,
                   R"(
Check if spin-dependent one-particle RDMs for active orbitals are available.

Returns:
    bool: True if available

Examples:
    >>> if wf.has_one_rdm_spin_dependent():
    ...     rdm_aa, rdm_bb = wf.get_active_one_rdm_spin_dependent()
)");

  wavefunction.def("has_one_rdm_spin_traced",
                   &Wavefunction::has_one_rdm_spin_traced,
                   R"(
Check if spin-traced one-particle RDM for active orbitals is available.

Returns:
    bool: True if available

Examples:
    >>> if wf.has_one_rdm_spin_traced():
    ...     rdm = wf.get_active_one_rdm_spin_traced()
)");

  wavefunction.def("has_two_rdm_spin_dependent",
                   &Wavefunction::has_two_rdm_spin_dependent,
                   R"(
Check if spin-dependent two-particle RDMs for active orbitals are available.

Returns:
    bool: True if available

Examples:
    >>> if wf.has_two_rdm_spin_dependent():
    ...     aaaa, aabb, bbbb = wf.get_active_two_rdm_spin_dependent()
)");

  wavefunction.def("has_two_rdm_spin_traced",
                   &Wavefunction::has_two_rdm_spin_traced,
                   R"(
Check if spin-traced two-particle RDM for active orbitals is available.

Returns:
    bool: True if available

Examples:
    >>> if wf.has_two_rdm_spin_traced():
    ...     two_rdm = wf.get_active_two_rdm_spin_traced()
)");

  // Type checking methods
  wavefunction.def("is_complex", &Wavefunction::is_complex,
                   R"(
Check if the wavefunction is complex-valued.

Returns:
    bool: True if the wavefunction uses complex coefficients, False if real

Examples:
    >>> is_complex = wf.is_complex()
    >>> print(f"Wavefunction uses complex coefficients: {is_complex}")
)");

  // Type access methods
  wavefunction.def("get_type", &Wavefunction::get_type,
                   R"(
Get the wavefunction type (Bra, Ket, or Both).

Returns:
    WavefunctionType: Enum value representing the wavefunction type

Examples:
    >>> wf_type = wf.get_type()
)");

  // Serialization methods
  wavefunction.def(
      "to_json",
      [](const Wavefunction& self) -> std::string {
        return self.to_json().dump();
      },
      R"(
Convert wavefunction to JSON string format.

Returns:
    str: JSON string containing wavefunction data

Examples:
    >>> json_str = wf.to_json()
)");

  wavefunction.def_static(
      "from_json",
      [](const std::string& json_str) -> std::shared_ptr<Wavefunction> {
        return Wavefunction::from_json(nlohmann::json::parse(json_str));
      },
      R"(
Load wavefunction from JSON string format.

Args:
    json_str (str): JSON string containing wavefunction data

Returns:
    Wavefunction: Wavefunction object created from JSON string

Examples:
    >>> wf = qdk_chemistry.Wavefunction.from_json(json_str)
)",
      py::arg("json_str"));

  wavefunction.def("to_json_file", &Wavefunction::to_json_file,
                   R"(
Save wavefunction to JSON file.

Args:
    filename (str): Path to JSON file to create/overwrite

Examples:
    >>> wf.to_json_file("wavefunction.json")
)",
                   py::arg("filename"));

  wavefunction.def_static("from_json_file", &Wavefunction::from_json_file,
                          R"(
Load wavefunction from JSON file.

Args:
    filename (str): Path to JSON file to read

Returns:
    Wavefunction: Wavefunction object created from JSON file

Examples:
    >>> wf = qdk_chemistry.Wavefunction.from_json_file("wavefunction.json")
)",
                          py::arg("filename"));

  wavefunction.def("to_hdf5_file", &Wavefunction::to_hdf5_file,
                   R"(
Save wavefunction to HDF5 file.

Args:
    filename (str): Path to HDF5 file to create/overwrite

Examples:
    >>> wf.to_hdf5_file("wavefunction.h5")
)",
                   py::arg("filename"));

  wavefunction.def_static("from_hdf5_file", &Wavefunction::from_hdf5_file,
                          R"(
Load wavefunction from HDF5 file.

Args:
    filename (str): Path to HDF5 file to read

Returns:
    Wavefunction: Wavefunction object created from HDF5 file

Examples:
    >>> wf = qdk_chemistry.Wavefunction.from_hdf5_file("wavefunction.h5")
)",
                          py::arg("filename"));

  wavefunction.def("to_file", &Wavefunction::to_file,
                   R"(
Save wavefunction to file in specified format.

Args:
    filename (str): Path to file to create/overwrite
    format (str): Format type ("json" or "hdf5")

Examples:
    >>> wf.to_file("wavefunction.json", "json")
    >>> wf.to_file("wavefunction.h5", "hdf5")
)",
                   py::arg("filename"), py::arg("format"));

  wavefunction.def_static("from_file", &Wavefunction::from_file,
                          R"(
Load wavefunction from file in specified format.

Args:
    filename (str): Path to file to read
    format (str): Format type ("json" or "hdf5")

Returns:
    Wavefunction: Wavefunction object created from file

Examples:
    >>> wf = qdk_chemistry.Wavefunction.from_file("wavefunction.json", "json")
    >>> wf = qdk_chemistry.Wavefunction.from_file("wavefunction.h5", "hdf5")
)",
                          py::arg("filename"), py::arg("format"));

  wavefunction.def(
      "__repr__",
      [](const Wavefunction& w) {
        return "<qdk_chemistry.Wavefunction size=" + std::to_string(w.size()) +
               " norm=" + std::to_string(w.norm()) + ">";
      },
      R"(
Returns a string representation of the Wavefunction object.

The representation includes the size of the wavefunction and its norm.

Returns:
    str: String representation of the Wavefunction object
)");

  wavefunction.def("get_container_type", &Wavefunction::get_container_type,
                   R"(
Get the type of the underlying wavefunction container.

Returns:
      str: Type name of the underlying wavefunction container

Examples:
    >>> container_type = wf.get_container_type()
    >>> print(f"Container type: {container_type}")
)");

  wavefunction.def("__str__", [](const Wavefunction& w) {
    return "<qdk_chemistry.Wavefunction size=" + std::to_string(w.size()) +
           " norm=" + std::to_string(w.norm()) + ">";
  });

  // Pickling support using JSON serialization
  wavefunction.def(py::pickle(
      [](const Wavefunction& w) {
        // __getstate__ - serialize to JSON string
        return w.to_json().dump();
      },
      [](const std::string& json_str) {
        // __setstate__ - deserialize from JSON string
        nlohmann::json parsed_json = nlohmann::json::parse(json_str);
        std::shared_ptr<Wavefunction> wf_ptr =
            Wavefunction::from_json(parsed_json);
        return *wf_ptr;
      }));

  // Bind SciWavefunctionContainer
  py::class_<SciWavefunctionContainer, WavefunctionContainer, py::smart_holder>(
      data, "SciWavefunctionContainer",
      R"(
Selected CI wavefunction container implementation.

This container represents wavefunctions obtained from selected configuration interaction (SCI) methods or full configuration interaction (FCI).
)")
      // Basic constructor: coeffs, dets, orbitals, type
      .def(py::init<const ContainerTypes::VectorVariant&,
                    const ContainerTypes::DeterminantVector&,
                    std::shared_ptr<Orbitals>, WavefunctionType>(),
           R"(
Constructs a basic SCI wavefunction container.

Args:
    coeffs (numpy.ndarray): The vector of CI coefficients (real or complex)
    dets (list[Configuration]): The vector of determinants
    orbitals (Orbitals): Shared pointer to orbital basis set
    type (WavefunctionType | None): Type of wavefunction (default: SelfDual)

Examples:
    >>> import numpy as np
    >>> coeffs = np.array([0.9, 0.1])
    >>> dets = [qdk_chemistry.Configuration("33221100"), qdk_chemistry.Configuration("33221001")]
    >>> container = qdk_chemistry.SciWavefunctionContainer(coeffs, dets, orbitals)
)",
           py::arg("coeffs"), py::arg("dets"), py::arg("orbitals"),
           py::arg("type") = WavefunctionType::SelfDual)
      // Constructor with spin-traced RDMs: coeffs, dets, orbitals,
      // one_rdm_spin_traced, two_rdm_spin_traced, type
      .def(py::init<const ContainerTypes::VectorVariant&,
                    const ContainerTypes::DeterminantVector&,
                    std::shared_ptr<Orbitals>,
                    const std::optional<ContainerTypes::MatrixVariant>&,
                    const std::optional<ContainerTypes::VectorVariant>&,
                    WavefunctionType>(),
           R"(
Constructs a SCI wavefunction container with spin-traced RDMs.

Args:
    coeffs (numpy.ndarray): The vector of CI coefficients (real or complex)
    dets (list[Configuration]): The vector of determinants
    orbitals (Orbitals): Shared pointer to orbital basis set
    one_rdm_spin_traced (numpy.ndarray | None): Spin-traced one-particle reduced density matrix
    two_rdm_spin_traced (numpy.ndarray | None): Spin-traced two-particle reduced density matrix
    type (WavefunctionType | None): Type of wavefunction (default: SelfDual)

Examples:
    >>> import numpy as np
    >>> coeffs = np.array([0.9, 0.1])
    >>> dets = [qdk_chemistry.Configuration("33221100"), qdk_chemistry.Configuration("33221001")]
    >>> one_rdm = np.eye(4)  # Example 1-RDM
    >>> container = qdk_chemistry.SciWavefunctionContainer(coeffs, dets, orbitals, one_rdm, None)
)",
           py::arg("coeffs"), py::arg("dets"), py::arg("orbitals"),
           py::arg("one_rdm_spin_traced") = std::nullopt,
           py::arg("two_rdm_spin_traced") = std::nullopt,
           py::arg("type") = WavefunctionType::SelfDual)
      // Full constructor with all RDM components
      .def(py::init<const ContainerTypes::VectorVariant&,
                    const ContainerTypes::DeterminantVector&,
                    std::shared_ptr<Orbitals>,
                    const std::optional<ContainerTypes::MatrixVariant>&,
                    const std::optional<ContainerTypes::MatrixVariant>&,
                    const std::optional<ContainerTypes::MatrixVariant>&,
                    const std::optional<ContainerTypes::VectorVariant>&,
                    const std::optional<ContainerTypes::VectorVariant>&,
                    const std::optional<ContainerTypes::VectorVariant>&,
                    const std::optional<ContainerTypes::VectorVariant>&,
                    WavefunctionType>(),
           R"(
Constructs a SCI wavefunction container with full RDM data.

Args:
    coeffs (numpy.ndarray): The vector of CI coefficients (real or complex)
    dets (list[Configuration]): The vector of determinants
    orbitals (Orbitals): Shared pointer to orbital basis set
    one_rdm_spin_traced (numpy.ndarray | None): Spin-traced one-particle reduced density matrix
    one_rdm_aa (numpy.ndarray | None): Alpha-alpha block of one-particle RDM
    one_rdm_bb (numpy.ndarray | None): Beta-beta block of one-particle RDM
    two_rdm_spin_traced (numpy.ndarray | None): Spin-traced two-particle reduced density matrix
    two_rdm_aabb (numpy.ndarray | None): Alpha-beta-beta-alpha block of two-particle RDM
    two_rdm_aaaa (numpy.ndarray | None): Alpha-alpha-alpha-alpha block of two-particle RDM
    two_rdm_bbbb (numpy.ndarray | None): Beta-beta-beta-beta block of two-particle RDM
    type (WavefunctionType | None): Type of wavefunction (default: SelfDual)

Examples:
    >>> import numpy as np
    >>> coeffs = np.array([0.9, 0.1])
    >>> dets = [qdk_chemistry.Configuration("33221100"), qdk_chemistry.Configuration("33221001")]
    >>> container = qdk_chemistry.SciWavefunctionContainer(coeffs, dets, orbitals,
    ...                                          one_rdm, one_rdm_aa, one_rdm_bb,
    ...                                          two_rdm, two_rdm_aabb, two_rdm_aaaa, two_rdm_bbbb)
)",
           py::arg("coeffs"), py::arg("dets"), py::arg("orbitals"),
           py::arg("one_rdm_spin_traced") = std::nullopt,
           py::arg("one_rdm_aa") = std::nullopt,
           py::arg("one_rdm_bb") = std::nullopt,
           py::arg("two_rdm_spin_traced") = std::nullopt,
           py::arg("two_rdm_aabb") = std::nullopt,
           py::arg("two_rdm_aaaa") = std::nullopt,
           py::arg("two_rdm_bbbb") = std::nullopt,
           py::arg("type") = WavefunctionType::SelfDual)
      .def("get_coefficients", &SciWavefunctionContainer::get_coefficients,
           "Get the coefficients of the wavefunction",
           py::return_value_policy::reference_internal);

  // Bind CasWavefunctionContainer
  // TODO (NAB): explain what makes this different from the generic wavefunction
  // class 41400

  py::class_<CasWavefunctionContainer, WavefunctionContainer, py::smart_holder>(
      data, "CasWavefunctionContainer",
      R"(
Complete Active Space (CAS) wavefunction container implementation.

This container represents wavefunctions obtained from complete active space self-consistent field (CASSCF) or complete active space configuration interaction (CASCI) methods.
)")
      // Basic constructor: coeffs, dets, orbitals, type
      .def(py::init<const ContainerTypes::VectorVariant&,
                    const ContainerTypes::DeterminantVector&,
                    std::shared_ptr<Orbitals>, WavefunctionType>(),
           R"(
Constructs a basic CAS wavefunction container.

Args:
    coeffs (numpy.ndarray): The vector of CI coefficients (real or complex)
    dets (list[Configuration]): The vector of determinants
    orbitals (Orbitals): Shared pointer to orbital basis set
    type (WavefunctionType | None): Type of wavefunction (default: SelfDual)

Examples:
    >>> import numpy as np
    >>> coeffs = np.array([0.9, 0.1])
    >>> dets = [qdk_chemistry.Configuration("33221100"), qdk_chemistry.Configuration("33221001")]
    >>> container = qdk_chemistry.CasWavefunctionContainer(coeffs, dets, orbitals)
)",
           py::arg("coeffs"), py::arg("dets"), py::arg("orbitals"),
           py::arg("type") = WavefunctionType::SelfDual)
      // Constructor with spin-traced RDMs: coeffs, dets, orbitals,
      // one_rdm_spin_traced, two_rdm_spin_traced, type
      .def(py::init<const ContainerTypes::VectorVariant&,
                    const ContainerTypes::DeterminantVector&,
                    std::shared_ptr<Orbitals>,
                    const std::optional<ContainerTypes::MatrixVariant>&,
                    const std::optional<ContainerTypes::VectorVariant>&,
                    WavefunctionType>(),
           R"(
Constructs a CAS wavefunction container with spin-traced RDMs.

Args:
    coeffs (numpy.ndarray): The vector of CI coefficients (real or complex)
    dets (list[Configuration]): The vector of determinants
    orbitals (Orbitals): Shared pointer to orbital basis set
    one_rdm_spin_traced (numpy.ndarray | None): Spin-traced one-particle reduced density matrix
    two_rdm_spin_traced (numpy.ndarray | None): Spin-traced two-particle reduced density matrix
    type (WavefunctionType | None): Type of wavefunction (default: SelfDual)

Examples:
    >>> import numpy as np
    >>> coeffs = np.array([0.9, 0.1])
    >>> dets = [qdk_chemistry.Configuration("33221100"), qdk_chemistry.Configuration("33221001")]
    >>> one_rdm = np.eye(4)  # Example 1-RDM
    >>> container = qdk_chemistry.CasWavefunctionContainer(coeffs, dets, orbitals, one_rdm, None)
)",
           py::arg("coeffs"), py::arg("dets"), py::arg("orbitals"),
           py::arg("one_rdm_spin_traced") = std::nullopt,
           py::arg("two_rdm_spin_traced") = std::nullopt,
           py::arg("type") = WavefunctionType::SelfDual)
      // Full constructor with all RDM components
      .def(py::init<const ContainerTypes::VectorVariant&,
                    const ContainerTypes::DeterminantVector&,
                    std::shared_ptr<Orbitals>,
                    const std::optional<ContainerTypes::MatrixVariant>&,
                    const std::optional<ContainerTypes::MatrixVariant>&,
                    const std::optional<ContainerTypes::MatrixVariant>&,
                    const std::optional<ContainerTypes::VectorVariant>&,
                    const std::optional<ContainerTypes::VectorVariant>&,
                    const std::optional<ContainerTypes::VectorVariant>&,
                    const std::optional<ContainerTypes::VectorVariant>&,
                    WavefunctionType>(),
           R"(
Constructs a CAS wavefunction container with full RDM data.

Args:
    coeffs (numpy.ndarray): The vector of CI coefficients (real or complex)
    dets (list[Configuration]): The vector of determinants
    orbitals (Orbitals): Shared pointer to orbital basis set
    one_rdm_spin_traced (numpy.ndarray | None): Spin-traced one-particle reduced density matrix
    one_rdm_aa (numpy.ndarray | None): Alpha-alpha block of one-particle RDM
    one_rdm_bb (numpy.ndarray | None): Beta-beta block of one-particle RDM
    two_rdm_spin_traced (numpy.ndarray | None): Spin-traced two-particle reduced density matrix
    two_rdm_aabb (numpy.ndarray | None): Alpha-beta-beta-alpha block of two-particle RDM
    two_rdm_aaaa (numpy.ndarray | None): Alpha-alpha-alpha-alpha block of two-particle RDM
    two_rdm_bbbb (numpy.ndarray | None): Beta-beta-beta-beta block of two-particle RDM
    type (WavefunctionType | None): Type of wavefunction (default: SelfDual)

Examples:
    >>> import numpy as np
    >>> coeffs = np.array([0.9, 0.1])
    >>> dets = [qdk_chemistry.Configuration("33221100"), qdk_chemistry.Configuration("33221001")]
    >>> container = qdk_chemistry.CasWavefunctionContainer(coeffs, dets, orbitals,
    ...                                          one_rdm, one_rdm_aa, one_rdm_bb,
    ...                                          two_rdm, two_rdm_aabb, two_rdm_aaaa, two_rdm_bbbb)
)",
           py::arg("coeffs"), py::arg("dets"), py::arg("orbitals"),
           py::arg("one_rdm_spin_traced") = std::nullopt,
           py::arg("one_rdm_aa") = std::nullopt,
           py::arg("one_rdm_bb") = std::nullopt,
           py::arg("two_rdm_spin_traced") = std::nullopt,
           py::arg("two_rdm_aabb") = std::nullopt,
           py::arg("two_rdm_aaaa") = std::nullopt,
           py::arg("two_rdm_bbbb") = std::nullopt,
           py::arg("type") = WavefunctionType::SelfDual)
      .def("get_coefficients", &CasWavefunctionContainer::get_coefficients,
           "Get the coefficients of the wavefunction",
           py::return_value_policy::reference_internal);

  // Bind SlaterDeterminantContainer
  py::class_<SlaterDeterminantContainer, WavefunctionContainer,
             py::smart_holder>(data, "SlaterDeterminantContainer",
                               R"(
Single Slater determinant wavefunction container implementation.

This container represents the simplest wavefunction - a single Slater determinant with coefficient 1.0.
It provides efficient storage and computation for single-determinant wavefunctions such as Hartree-Fock reference states.
)")
      .def(py::init<const Configuration&, std::shared_ptr<Orbitals>,
                    WavefunctionType>(),
           R"(
Constructs a single Slater determinant wavefunction container.

Args:
    det (Configuration): The single determinant configuration
    orbitals (Orbitals): Shared pointer to orbital basis set
    type (WavefunctionType | None): Type of wavefunction (default: Both)

Examples:
    >>> det = qdk_chemistry.Configuration("33221100")
    >>> container = qdk_chemistry.SlaterDeterminantContainer(det, orbitals)
)",
           py::arg("det"), py::arg("orbitals"),
           py::arg("type") = WavefunctionType::SelfDual)
      .def("contains_determinant",
           &SlaterDeterminantContainer::contains_determinant,
           "Check if a determinant matches the stored one", py::arg("det"));
}
