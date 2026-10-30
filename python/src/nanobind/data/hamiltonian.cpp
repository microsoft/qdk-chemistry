// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>

#include <qdk/chemistry.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/cholesky.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/dfthc.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/sparse.hpp>

#include "eigen.hpp"
#include "path_utils.hpp"
#include "property_binding_helpers.hpp"
#include "stl.hpp"

namespace nb = nanobind;

// Wrapper functions that accept both strings and pathlib.Path objects
namespace {

void hamiltonian_to_file_wrapper(qdk::chemistry::data::Hamiltonian& self,
                                 const nb::object& filename,
                                 const std::string& format_type) {
  self.to_file(qdk::chemistry::python::utils::to_string_path(filename),
               format_type);
}
std::shared_ptr<qdk::chemistry::data::Hamiltonian>
hamiltonian_from_file_wrapper(const nb::object& filename,
                              const std::string& format_type) {
  return qdk::chemistry::data::Hamiltonian::from_file(
      qdk::chemistry::python::utils::to_string_path(filename), format_type);
}

void hamiltonian_to_json_file_wrapper(qdk::chemistry::data::Hamiltonian& self,
                                      const nb::object& filename) {
  self.to_json_file(qdk::chemistry::python::utils::to_string_path(filename));
}

std::shared_ptr<qdk::chemistry::data::Hamiltonian>
hamiltonian_from_json_file_wrapper(const nb::object& filename) {
  return qdk::chemistry::data::Hamiltonian::from_json_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

void hamiltonian_to_hdf5_file_wrapper(qdk::chemistry::data::Hamiltonian& self,
                                      const nb::object& filename) {
  self.to_hdf5_file(qdk::chemistry::python::utils::to_string_path(filename));
}

std::shared_ptr<qdk::chemistry::data::Hamiltonian>
hamiltonian_from_hdf5_file_wrapper(const nb::object& filename) {
  return qdk::chemistry::data::Hamiltonian::from_hdf5_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

void hamiltonian_to_fcidump_file_wrapper(
    qdk::chemistry::data::Hamiltonian& self, const nb::object& filename,
    size_t nalpha, size_t nbeta) {
  self.to_fcidump_file(qdk::chemistry::python::utils::to_string_path(filename),
                       nalpha, nbeta);
}

}  // namespace

void bind_hamiltonian(nb::module_& data) {
  using namespace qdk::chemistry::data;
  using qdk::chemistry::python::utils::bind_getter_as_property;

  // HamiltonianType enumeration
  nb::enum_<HamiltonianType>(data, "HamiltonianType", R"(
Enumeration for Hamiltonian types.

Values:
    Hermitian: Standard Hermitian Hamiltonian
    NonHermitian: Non-Hermitian Hamiltonian
)")
      .value("Hermitian", HamiltonianType::Hermitian)
      .value("NonHermitian", HamiltonianType::NonHermitian);

  // SpinChannel enumeration
  nb::enum_<SpinChannel>(data, "SpinChannel", R"(
Enumeration for different spin channels in unrestricted calculations.

Values:
    aa: Alpha-alpha spin channel (for one-body integrals)
    bb: Beta-beta spin channel (for one-body integrals)
    aaaa: Alpha-alpha-alpha-alpha spin channel (for two-body integrals)
    aabb: Alpha-beta-alpha-beta spin channel (for two-body integrals)
    bbbb: Beta-beta-beta-beta spin channel (for two-body integrals)
)")
      .value("aa", SpinChannel::aa)
      .value("bb", SpinChannel::bb)
      .value("aaaa", SpinChannel::aaaa)
      .value("aabb", SpinChannel::aabb)
      .value("bbbb", SpinChannel::bbbb);

  // ============================================================================
  // HamiltonianContainer - Abstract base class (not directly constructible)
  // ============================================================================
  nb::class_<HamiltonianContainer> hamiltonian_container(
      data, "HamiltonianContainer", R"(
Abstract base class for Hamiltonian container implementations.

This class defines the interface for storing molecular Hamiltonian data
for quantum chemistry calculations. It contains:

* One-electron integrals (kinetic + nuclear attraction) in MO representation
* Molecular orbital information for the active space
* A constant (zero-body) energy term

Derived classes implement specific storage formats for two-electron integrals
(e.g., canonical 4-center, density-fitted, etc.).

Note:
    This class cannot be instantiated directly. Use a derived class like
    CanonicalFourCenterHamiltonianContainer instead.
)");

  // HamiltonianContainer methods (read-only accessors)
  hamiltonian_container.def("get_one_body_integrals",
                            &HamiltonianContainer::get_one_body_integrals,
                            nb::rv_policy::reference_internal,
                            R"(
Deprecated: use one_body_integrals() instead.

Returns:
    tuple[numpy.ndarray, numpy.ndarray]: One-electron integral matrices [norb x norb]
    for alpha and beta spin channels.
)");

  hamiltonian_container.def("has_one_body_integrals",
                            &HamiltonianContainer::has_one_body_integrals,
                            R"(
Check if one-body integrals are available.

Returns:
    bool: True if one-body integrals have been set
)");

  hamiltonian_container.def(
      "get_one_body_element", &HamiltonianContainer::get_one_body_element,
      R"(
Get specific one-electron integral element <ij>.

Args:
    i (int): First orbital index
    j (int): Second orbital index
    channel (SpinChannel): Spin channel (aa or bb), defaults to aa

Returns:
    float: Value of the one-electron integral <ij>
)",
      nb::arg("i"), nb::arg("j"), nb::arg("channel") = SpinChannel::aa);

  hamiltonian_container.def("get_inactive_fock_matrix",
                            &HamiltonianContainer::get_inactive_fock_matrix,
                            nb::rv_policy::reference_internal,
                            R"(
Deprecated: use inactive_fock() instead.

Returns:
    tuple[numpy.ndarray, numpy.ndarray]: Inactive Fock matrices for the active space
)");

  hamiltonian_container.def("has_inactive_fock_matrix",
                            &HamiltonianContainer::has_inactive_fock_matrix,
                            R"(
Check if inactive Fock matrix is available.

Returns:
    bool: True if inactive Fock matrix has been set
)");

  // SBT-native accessors
  hamiltonian_container.def("one_body_integrals",
                            &HamiltonianContainer::one_body_integrals,
                            nb::rv_policy::reference_internal,
                            R"(
One-body integrals as a rank-2 symmetry-blocked tensor.

Returns:
    qdk_chemistry.data.symmetry.SymmetryBlockedTensorRank2: The h1 SBT.
)");

  hamiltonian_container.def("one_body_integrals_block",
                            &HamiltonianContainer::one_body_integrals_block,
                            nb::rv_policy::reference_internal,
                            R"(
One-body integral block for the given row/column symmetry labels.

Args:
    row (SymmetryLabel): Row symmetry label.
    col (SymmetryLabel): Column symmetry label.

Returns:
    numpy.ndarray: The matrix block.
)",
                            nb::arg("row"), nb::arg("col"));

  hamiltonian_container.def("inactive_fock",
                            &HamiltonianContainer::inactive_fock,
                            nb::rv_policy::reference_internal,
                            R"(
Inactive Fock matrix as a rank-2 symmetry-blocked tensor.

Returns:
    qdk_chemistry.data.symmetry.SymmetryBlockedTensorRank2: The inactive Fock SBT.
)");

  hamiltonian_container.def("inactive_fock_block",
                            &HamiltonianContainer::inactive_fock_block,
                            nb::rv_policy::reference_internal,
                            R"(
Inactive Fock block for the given row/column symmetry labels.

Args:
    row (SymmetryLabel): Row symmetry label.
    col (SymmetryLabel): Column symmetry label.

Returns:
    numpy.ndarray: The matrix block.
)",
                            nb::arg("row"), nb::arg("col"));

  bind_getter_as_property(hamiltonian_container, "get_orbitals",
                          &HamiltonianContainer::get_orbitals,
                          R"(
Get molecular orbital data.

Returns:
    Orbitals: The orbital data associated with this container
)",
                          nb::rv_policy::reference_internal);

  hamiltonian_container.def("has_orbitals", &HamiltonianContainer::has_orbitals,
                            R"(
Check if orbital data is available.

Returns:
    bool: True if orbital data has been set
)");

  bind_getter_as_property(hamiltonian_container, "get_core_energy",
                          &HamiltonianContainer::get_core_energy,
                          R"(
Get the constant (zero-body) energy term in atomic units.

Returns:
    float: Constant energy term in Hartree
)");

  bind_getter_as_property(hamiltonian_container, "get_type",
                          &HamiltonianContainer::get_type,
                          R"(
Get the type of Hamiltonian (Hermitian or NonHermitian).

Returns:
    HamiltonianType: The Hamiltonian type
)");

  hamiltonian_container.def("is_hermitian", &HamiltonianContainer::is_hermitian,
                            R"(
Check if the Hamiltonian is Hermitian.

Returns:
    bool: True if the Hamiltonian type is Hermitian
)");

  hamiltonian_container.def("is_unrestricted",
                            &HamiltonianContainer::is_unrestricted,
                            R"(
Check if Hamiltonian is unrestricted.

Returns:
    bool: True if alpha and beta integrals are different
)");

  hamiltonian_container.def("get_container_type",
                            &HamiltonianContainer::get_container_type,
                            R"(
Get the type of this container as a string.

Returns:
    str: Container type identifier (e.g., "canonical_four_center")
)");
  // ============================================================================
  // CholeskyHamiltonianContainer - Concrete implementation
  // ============================================================================
  nb::class_<CholeskyHamiltonianContainer, HamiltonianContainer>
      cholesky_container(data, "CholeskyHamiltonianContainer", R"(
Represents a molecular Hamiltonian using Cholesky-decomposed three-center integrals.
This class stores molecular Hamiltonian data for quantum chemistry calculations,
specifically designed for active space methods. It contains:

* One-electron integrals (kinetic + nuclear attraction) in MO representation
* Three-center two-electron integrals (ij|Q) in MO representation
* Molecular orbital information for the active space
* A constant (zero-body) energy term

Four-center integrals are lazily computed from three-center integrals on first access.

Examples:
    >>> import numpy as np
    >>> # Create restricted Hamiltonian
    >>> one_body = np.random.rand(4, 4)  # 4 orbitals
    >>> three_center = np.random.rand(16, 10)  # (norb^2) x naux
    >>> inactive_fock_matrix = np.random.rand(4, 4)
    >>> container = CholeskyHamiltonianContainer(
    ...     one_body, three_center, orbitals, 10.5, inactive_fock_matrix
    ... )
    >>> # Wrap in Hamiltonian interface
    >>> hamiltonian = Hamiltonian(container)
)");

  // Restricted constructor
  cholesky_container.def(
      nb::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&,
               std::shared_ptr<Orbitals>, double, const Eigen::MatrixXd&,
               std::optional<Eigen::MatrixXd>, HamiltonianType>(),
      R"(
Constructor for restricted active space Hamiltonian with Cholesky-decomposed integrals.

Args:
    one_body_integrals (numpy.ndarray): One-electron integrals matrix [norb x norb]
    three_center_integrals (numpy.ndarray): Three-center two-electron integrals
        in MO basis [(norb*norb) x naux]
    orbitals (Orbitals): Molecular orbital data
    core_energy (float): Constant (zero-body) energy term
    inactive_fock_matrix (numpy.ndarray): Inactive Fock matrix [norb x norb]
    ao_cholesky_vectors (numpy.ndarray or None, optional): AO Cholesky vectors
        for potential reuse. Defaults to None.
    type (HamiltonianType, optional): Type of Hamiltonian (Hermitian by default)

Examples:
    >>> import numpy as np
    >>> one_body = np.random.rand(4, 4)
    >>> three_center = np.random.rand(16, 10)  # 16 = 4^2, 10 auxiliary basis functions
    >>> inactive_fock_matrix = np.random.rand(4, 4)
    >>> container = CholeskyHamiltonianContainer(
    ...     one_body, three_center, orbitals, 10.5, inactive_fock_matrix
    ... )
)",
      nb::arg("one_body_integrals"), nb::arg("three_center_integrals"),
      nb::arg("orbitals"), nb::arg("core_energy"),
      nb::arg("inactive_fock_matrix"),
      nb::arg("ao_cholesky_vectors") = nb::none(),
      nb::arg("type") = HamiltonianType::Hermitian);

  // Unrestricted constructor
  cholesky_container.def(
      nb::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&,
               const Eigen::MatrixXd&, const Eigen::MatrixXd&,
               std::shared_ptr<Orbitals>, double, const Eigen::MatrixXd&,
               const Eigen::MatrixXd&, std::optional<Eigen::MatrixXd>,
               HamiltonianType>(),
      R"(
Constructor for unrestricted active space Hamiltonian with Cholesky-decomposed integrals.

Args:
    one_body_integrals_alpha (numpy.ndarray): Alpha one-electron integrals [norb x norb]
    one_body_integrals_beta (numpy.ndarray): Beta one-electron integrals [norb x norb]
    three_center_integrals_aa (numpy.ndarray): Alpha-alpha three-center integrals
        [(norb*norb) x naux], orbital pair index in row-major order
    three_center_integrals_bb (numpy.ndarray): Beta-beta three-center integrals
        [(norb*norb) x naux], orbital pair index in row-major order
    orbitals (Orbitals): Molecular orbital data
    core_energy (float): Constant (zero-body) energy term
    inactive_fock_matrix_alpha (numpy.ndarray): Alpha inactive Fock matrix [norb x norb]
    inactive_fock_matrix_beta (numpy.ndarray): Beta inactive Fock matrix [norb x norb]
    ao_cholesky_vectors (numpy.ndarray or None, optional): AO Cholesky vectors
        for potential reuse. Defaults to None.
    type (HamiltonianType, optional): Type of Hamiltonian (Hermitian by default)

Examples:
    >>> import numpy as np
    >>> one_body_a = np.random.rand(4, 4)
    >>> one_body_b = np.random.rand(4, 4)
    >>> three_center_aa = np.random.rand(16, 10)  # 16 = 4^2, 10 aux
    >>> three_center_bb = np.random.rand(16, 10)
    >>> fock_a = np.random.rand(4, 4)
    >>> fock_b = np.random.rand(4, 4)
    >>> container = CholeskyHamiltonianContainer(
    ...     one_body_a, one_body_b,
    ...     three_center_aa, three_center_bb,
    ...     orbitals, 10.5, fock_a, fock_b
    ... )
)",
      nb::arg("one_body_integrals_alpha"), nb::arg("one_body_integrals_beta"),
      nb::arg("three_center_integrals_aa"),
      nb::arg("three_center_integrals_bb"), nb::arg("orbitals"),
      nb::arg("core_energy"), nb::arg("inactive_fock_matrix_alpha"),
      nb::arg("inactive_fock_matrix_beta"),
      nb::arg("ao_cholesky_vectors") = nb::none(),
      nb::arg("type") = HamiltonianType::Hermitian);

  // Three-center integral access
  cholesky_container.def(
      "get_three_center_integrals",
      &CholeskyHamiltonianContainer::get_three_center_integrals,
      nb::rv_policy::reference_internal,
      R"(
Deprecated: use three_center() instead.

Returns:
    tuple[numpy.ndarray, numpy.ndarray]: Pair of (aa, bb) three-center
    two-electron integral matrices, each of dimension [(norb*norb) x naux]
    with the orbital pair index stored in row-major order.

Raises:
    RuntimeError: If three-center integrals are not set
)");

  // AO Cholesky vectors access
  cholesky_container.def(
      "get_ao_cholesky_vectors",
      [](const CholeskyHamiltonianContainer& self) -> nb::object {
        const auto& opt = self.get_ao_cholesky_vectors();
        if (!opt) return nb::none();
        const Eigen::MatrixXd& mat = *opt;
        return nb::ndarray<nb::numpy, const double, nb::f_contig>(
                   mat.data(),
                   {static_cast<size_t>(mat.rows()),
                    static_cast<size_t>(mat.cols())},
                   nb::cast(&self, nb::rv_policy::reference))
            .cast();
      },
      R"(
Get the optional AO Cholesky vectors (zero-copy view).

Returns:
    numpy.ndarray or None: AO Cholesky vectors matrix [nao^2 x nchol],
    or None if not stored. The returned array shares memory with the
    internal storage and should not be modified.
)");

  // Two-body integral access (lazily computed from three-center integrals)
  cholesky_container.def("get_two_body_integrals",
                         &CholeskyHamiltonianContainer::get_two_body_integrals,
                         nb::rv_policy::reference_internal,
                         R"(
Deprecated: use three_center() instead.

Returns:
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Tuple of two-electron
    integral vectors [norb^4] for aaaa, aabb, and bbbb spin channels.

Notes:
    Integrals are stored as flattened vectors in chemist notation <ij|kl>
    where indices are ordered as i + j*norb + k*norb^2 + l*norb^3
)");

  cholesky_container.def("get_two_body_element",
                         &CholeskyHamiltonianContainer::get_two_body_element,
                         R"(
Get specific two-electron integral element <ij|kl>.

Args:
    i, j, k, l (int): Orbital indices
    channel (SpinChannel): Spin channel (aaaa, aabb, or bbbb), defaults to aaaa

Returns:
    float: Value of the two-electron integral <ij|kl>
)",
                         nb::arg("i"), nb::arg("j"), nb::arg("k"), nb::arg("l"),
                         nb::arg("channel") = SpinChannel::aaaa);

  cholesky_container.def("has_two_body_integrals",
                         &CholeskyHamiltonianContainer::has_two_body_integrals,
                         R"(
Check if two-body integrals are available.

Returns:
    bool: True if two-body integrals have been set
)");

  cholesky_container.def("is_restricted",
                         &CholeskyHamiltonianContainer::is_restricted,
                         R"(
Check if Hamiltonian is restricted (alpha == beta).

Returns:
    bool: True if alpha and beta integrals are identical
)");

  // SBT-native three-center accessor
  cholesky_container.def("three_center",
                         &CholeskyHamiltonianContainer::three_center,
                         nb::rv_policy::reference_internal,
                         R"(
Three-center integrals as a rank-2 symmetry-blocked tensor.

Row axis keyed by MO spin symmetries (extent = norb^2 per spin),
column axis has no symmetry (extent = naux).

Returns:
    qdk_chemistry.data.symmetry.SymmetryBlockedTensorRank2: The three-center SBT.
)");

  cholesky_container.def("is_valid", &CholeskyHamiltonianContainer::is_valid,
                         R"(
Check if the Hamiltonian data is complete and consistent.

Returns:
    bool: True if all required data is set and dimensions are consistent
)");

  cholesky_container.def(
      "to_json",
      [](const CholeskyHamiltonianContainer& self) -> std::string {
        return self.to_json().dump();
      },
      R"(
Convert container to JSON string.

Returns:
    str: JSON representation of the container
)");

  cholesky_container.def(
      "to_fcidump_file", &CholeskyHamiltonianContainer::to_fcidump_file,
      R"(
Save Hamiltonian to FCIDUMP file.

Args:
    filename (str): Path to FCIDUMP file to create/overwrite
    nalpha (int): Number of alpha electrons
    nbeta (int): Number of beta electrons
)",
      nb::arg("filename"), nb::arg("nalpha"), nb::arg("nbeta"));

  // ============================================================================
  // CanonicalFourCenterHamiltonianContainer - Concrete implementation
  // ============================================================================
  nb::class_<CanonicalFourCenterHamiltonianContainer, HamiltonianContainer>
      canonical_four_center(data, "CanonicalFourCenterHamiltonianContainer", R"(
Represents a molecular Hamiltonian with canonical 4-center two-electron integrals.

This class stores molecular Hamiltonian data for quantum chemistry calculations,
specifically designed for active space methods. It contains:

* One-electron integrals (kinetic + nuclear attraction) in MO representation
* Two-electron integrals (electron-electron repulsion) in MO representation
* Molecular orbital information for the active space
* A constant (zero-body) energy term

This is the standard full integral storage format where two-electron integrals
are stored as a flattened [norb^4] vector.

Examples:
    >>> import numpy as np
    >>> # Create restricted Hamiltonian
    >>> one_body = np.random.rand(4, 4)  # 4 orbitals
    >>> two_body = np.random.rand(256)   # 4^4 elements
    >>> fock_matrix = np.random.rand(4, 4)
    >>> container = CanonicalFourCenterHamiltonianContainer(
    ...     one_body, two_body, orbitals, 10.5, fock_matrix
    ... )
    >>> # Wrap in Hamiltonian interface
    >>> hamiltonian = Hamiltonian(container)
)");

  // Restricted constructor
  canonical_four_center.def(
      nb::init<const Eigen::MatrixXd&, const Eigen::VectorXd&,
               std::shared_ptr<Orbitals>, double, const Eigen::MatrixXd&,
               HamiltonianType>(),
      R"(
Constructor for restricted active space Hamiltonian with 4-center integrals.

Args:
    one_body_integrals (numpy.ndarray): One-electron integrals matrix [norb x norb]
    two_body_integrals (numpy.ndarray): Two-electron integrals vector [norb^4]
    orbitals (Orbitals): Molecular orbital data
    core_energy (float): Constant (zero-body) energy term
    inactive_fock_matrix (numpy.ndarray): Inactive Fock matrix [norb x norb]
    type (HamiltonianType, optional): Type of Hamiltonian (Hermitian by default)

Examples:
    >>> import numpy as np
    >>> one_body = np.random.rand(4, 4)
    >>> two_body = np.random.rand(256)
    >>> fock_matrix = np.random.rand(4, 4)
    >>> container = CanonicalFourCenterHamiltonianContainer(
    ...     one_body, two_body, orbitals, 10.5, fock_matrix
    ... )
)",
      nb::arg("one_body_integrals"), nb::arg("two_body_integrals"),
      nb::arg("orbitals"), nb::arg("core_energy"),
      nb::arg("inactive_fock_matrix"),
      nb::arg("type") = HamiltonianType::Hermitian);

  // Unrestricted constructor
  canonical_four_center.def(
      nb::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&,
               const Eigen::VectorXd&, const Eigen::VectorXd&,
               const Eigen::VectorXd&, std::shared_ptr<Orbitals>, double,
               const Eigen::MatrixXd&, const Eigen::MatrixXd&,
               HamiltonianType>(),
      R"(
Constructor for unrestricted active space Hamiltonian with 4-center integrals.

Args:
    one_body_integrals_alpha (numpy.ndarray): Alpha one-electron integrals [norb x norb]
    one_body_integrals_beta (numpy.ndarray): Beta one-electron integrals [norb x norb]
    two_body_integrals_aaaa (numpy.ndarray): Alpha-alpha-alpha-alpha integrals [norb^4]
    two_body_integrals_aabb (numpy.ndarray): Alpha-beta-alpha-beta integrals [norb^4]
    two_body_integrals_bbbb (numpy.ndarray): Beta-beta-beta-beta integrals [norb^4]
    orbitals (Orbitals): Molecular orbital data
    core_energy (float): Constant (zero-body) energy term
    inactive_fock_matrix_alpha (numpy.ndarray): Alpha inactive Fock matrix [norb x norb]
    inactive_fock_matrix_beta (numpy.ndarray): Beta inactive Fock matrix [norb x norb]
    type (HamiltonianType, optional): Type of Hamiltonian (Hermitian by default)

Examples:
    >>> import numpy as np
    >>> one_body_a = np.random.rand(4, 4)
    >>> one_body_b = np.random.rand(4, 4)
    >>> two_body_aaaa = np.random.rand(256)
    >>> two_body_aabb = np.random.rand(256)
    >>> two_body_bbbb = np.random.rand(256)
    >>> fock_a = np.random.rand(4, 4)
    >>> fock_b = np.random.rand(4, 4)
    >>> container = CanonicalFourCenterHamiltonianContainer(
    ...     one_body_a, one_body_b,
    ...     two_body_aaaa, two_body_aabb, two_body_bbbb,
    ...     orbitals, 10.5, fock_a, fock_b
    ... )
)",
      nb::arg("one_body_integrals_alpha"), nb::arg("one_body_integrals_beta"),
      nb::arg("two_body_integrals_aaaa"), nb::arg("two_body_integrals_aabb"),
      nb::arg("two_body_integrals_bbbb"), nb::arg("orbitals"),
      nb::arg("core_energy"), nb::arg("inactive_fock_matrix_alpha"),
      nb::arg("inactive_fock_matrix_beta"),
      nb::arg("type") = HamiltonianType::Hermitian);

  // Two-body integral access (specific to
  // CanonicalFourCenterHamiltonianContainer)
  canonical_four_center.def(
      "get_two_body_integrals",
      &CanonicalFourCenterHamiltonianContainer::get_two_body_integrals,
      nb::rv_policy::reference_internal,
      R"(
Deprecated: use two_body_integrals() instead.

Returns:
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Tuple of two-electron
    integral vectors [norb^4] for aaaa, aabb, and bbbb spin channels.

Notes:
    Integrals are stored as flattened vectors in chemist notation <ij|kl>
    where indices are ordered as i + j*norb + k*norb^2 + l*norb^3
)");

  canonical_four_center.def(
      "get_two_body_element",
      &CanonicalFourCenterHamiltonianContainer::get_two_body_element,
      R"(
Get specific two-electron integral element <ij|kl>.

Args:
    i, j, k, l (int): Orbital indices
    channel (SpinChannel): Spin channel (aaaa, aabb, or bbbb), defaults to aaaa

Returns:
    float: Value of the two-electron integral <ij|kl>
)",
      nb::arg("i"), nb::arg("j"), nb::arg("k"), nb::arg("l"),
      nb::arg("channel") = SpinChannel::aaaa);

  canonical_four_center.def(
      "has_two_body_integrals",
      &CanonicalFourCenterHamiltonianContainer::has_two_body_integrals,
      R"(
Check if two-body integrals are available.

Returns:
    bool: True if two-body integrals have been set
)");

  canonical_four_center.def(
      "is_restricted", &CanonicalFourCenterHamiltonianContainer::is_restricted,
      R"(
Check if Hamiltonian is restricted (alpha == beta).

Returns:
    bool: True if alpha and beta integrals are identical
)");

  // SBT-native h2 accessors
  canonical_four_center.def(
      "two_body_integrals",
      &CanonicalFourCenterHamiltonianContainer::two_body_integrals,
      nb::rv_policy::reference_internal,
      R"(
Two-body integrals as a rank-4 symmetry-blocked tensor.

Returns:
    qdk_chemistry.data.symmetry.SymmetryBlockedTensorRank4: The h2 SBT.
)");

  canonical_four_center.def(
      "two_body_integrals_block",
      &CanonicalFourCenterHamiltonianContainer::two_body_integrals_block,
      nb::rv_policy::reference_internal,
      R"(
Two-body integral block for the given symmetry labels.

Args:
    p (SymmetryLabel): First symmetry label.
    q (SymmetryLabel): Second symmetry label.
    r (SymmetryLabel): Third symmetry label.
    s (SymmetryLabel): Fourth symmetry label.

Returns:
    numpy.ndarray: The flat-packed vector block.
)",
      nb::arg("p"), nb::arg("q"), nb::arg("r"), nb::arg("s"));

  canonical_four_center.def("is_valid",
                            &CanonicalFourCenterHamiltonianContainer::is_valid,
                            R"(
Check if the Hamiltonian data is complete and consistent.

Returns:
    bool: True if all required data is set and dimensions are consistent
)");

  canonical_four_center.def(
      "to_json",
      [](const CanonicalFourCenterHamiltonianContainer& self) -> std::string {
        return self.to_json().dump();
      },
      R"(
Convert container to JSON string.

Returns:
    str: JSON representation of the container
)");

  canonical_four_center.def(
      "to_fcidump_file",
      &CanonicalFourCenterHamiltonianContainer::to_fcidump_file,
      R"(
Save Hamiltonian to FCIDUMP file.

Args:
    filename (str): Path to FCIDUMP file to create/overwrite
    nalpha (int): Number of alpha electrons
    nbeta (int): Number of beta electrons
)",
      nb::arg("filename"), nb::arg("nalpha"), nb::arg("nbeta"));

  // ============================================================================
  // SparseHamiltonianContainer — sparse lattice-model container
  // ============================================================================
  nb::class_<SparseHamiltonianContainer, HamiltonianContainer> sparse_container(
      data, "SparseHamiltonianContainer", R"(
Hamiltonian container for lattice model Hamiltonians (Hückel, Hubbard, PPP, etc.)
with sparse internal storage.

Stores one-body integrals as a sparse matrix and two-body integrals as a
sparse dictionary of (p, q, r, s) index tuples.

Internally uses ``ModelOrbitals`` so that the full ``HamiltonianContainer``
interface (``get_one_body_integrals``, ``get_orbitals``, etc.) works.

Examples:
    >>> import numpy as np
    >>> from qdk_chemistry.data import SparseHamiltonianContainer, Hamiltonian
    >>> h1 = np.diag([-0.5, -0.5, -0.5])
    >>> container = SparseHamiltonianContainer(h1)
    >>> hamiltonian = Hamiltonian(container)
    >>> hamiltonian.has_one_body_integrals()
    True
)");

  // -- Sparse + two-body constructor --
  sparse_container.def(nb::init<Eigen::SparseMatrix<double>,
                                SparseHamiltonianContainer::TwoBodyMap, double,
                                HamiltonianType>(),
                       R"(
Construct from sparse one-body integrals and a sparse two-body map.

Args:
    one_body_integrals (scipy.sparse matrix): Sparse one-body integral matrix [n x n].
    two_body_integrals (dict[tuple[int,int,int,int], float]): Sparse two-body
        integral map from (p, q, r, s) index tuples to values.
    core_energy (float, optional): Scalar energy offset. Defaults to 0.0.
    type (HamiltonianType, optional): Hamiltonian type. Defaults to Hermitian.
)",
                       nb::arg("one_body_integrals"),
                       nb::arg("two_body_integrals"),
                       nb::arg("core_energy") = 0.0,
                       nb::arg("type") = HamiltonianType::Hermitian);

  // -- Sparse one-body only --
  sparse_container.def(
      nb::init<Eigen::SparseMatrix<double>, double, HamiltonianType>(),
      R"(
Construct from sparse one-body integrals only.

Args:
    one_body_integrals (scipy.sparse matrix): Sparse one-body integral matrix [n x n].
    core_energy (float, optional): Scalar energy offset. Defaults to 0.0.
    type (HamiltonianType, optional): Hamiltonian type. Defaults to Hermitian.
)",
      nb::arg("one_body_integrals"), nb::arg("core_energy") = 0.0,
      nb::arg("type") = HamiltonianType::Hermitian);

  // -- Dense + two-body constructor --
  sparse_container.def(nb::init<const Eigen::MatrixXd&, const Eigen::VectorXd&,
                                double, HamiltonianType>(),
                       R"(
Construct from dense one-body and two-body integrals.

Args:
    one_body_integrals (numpy.ndarray): Dense one-body integral matrix [n x n].
    two_body_integrals (numpy.ndarray): Dense two-body integrals [n^4].
    core_energy (float, optional): Scalar energy offset. Defaults to 0.0.
    type (HamiltonianType, optional): Hamiltonian type. Defaults to Hermitian.
)",
                       nb::arg("one_body_integrals"),
                       nb::arg("two_body_integrals"),
                       nb::arg("core_energy") = 0.0,
                       nb::arg("type") = HamiltonianType::Hermitian);

  // -- Dense one-body only --
  sparse_container.def(
      nb::init<const Eigen::MatrixXd&, double, HamiltonianType>(),
      R"(
Construct from dense one-body integrals only (no two-body).

Args:
    one_body_integrals (numpy.ndarray): Dense one-body integral matrix [n x n].
    core_energy (float, optional): Scalar energy offset. Defaults to 0.0.
    type (HamiltonianType, optional): Hamiltonian type. Defaults to Hermitian.
)",
      nb::arg("one_body_integrals"), nb::arg("core_energy") = 0.0,
      nb::arg("type") = HamiltonianType::Hermitian);

  // -- Base-class overrides --
  sparse_container.def("get_two_body_integrals",
                       &SparseHamiltonianContainer::get_two_body_integrals,
                       nb::rv_policy::reference_internal,
                       R"(
Deprecated: use h2_sparse() instead.

Get two-electron integrals as dense vectors for all spin channels.

Materialises the sparse map into a dense n^4 vector on first access (cached).
Model Hamiltonians are restricted so all three channels are identical.

Returns:
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: (aaaa, aabb, bbbb) integrals.

Raises:
    RuntimeError: If no two-body integrals are stored.
)");

  sparse_container.def("get_two_body_element",
                       &SparseHamiltonianContainer::get_two_body_element,
                       R"(
Get a specific two-electron integral element.

Args:
    i, j, k, l (int): Orbital indices.
    channel (SpinChannel, optional): Spin channel (ignored; model Hamiltonians
        are restricted). Defaults to aaaa.

Returns:
    float: Two-electron integral (ij|kl), or 0 if not stored.
)",
                       nb::arg("i"), nb::arg("j"), nb::arg("k"), nb::arg("l"),
                       nb::arg("channel") = SpinChannel::aaaa);

  sparse_container.def("has_two_body_integrals",
                       &SparseHamiltonianContainer::has_two_body_integrals,
                       R"(
Check if two-body integrals are available.

Returns:
    bool: True if the two-body map is non-empty.
)");

  sparse_container.def("is_restricted",
                       &SparseHamiltonianContainer::is_restricted, R"(
Model Hamiltonians are always restricted.

Returns:
    bool: Always True.
)");

  sparse_container.def("is_valid", &SparseHamiltonianContainer::is_valid, R"(
Check if the container data is consistent.

Returns:
    bool: True if the one-body matrix is square and non-empty.
)");

  // -- Sparse-specific accessors --
  sparse_container.def("sparse_one_body_integrals",
                       &SparseHamiltonianContainer::sparse_one_body_integrals,
                       R"(
Direct access to the sparse one-body integral matrix.

Returns:
    scipy.sparse.csc_matrix: Sparse one-body integrals [n x n].
)",
                       nb::rv_policy::reference_internal);

  sparse_container.def("sparse_two_body_integrals",
                       &SparseHamiltonianContainer::sparse_two_body_integrals,
                       R"(
Deprecated: use two_body_integrals_sparse() for SymmetryBlockedSparseMap access.

Returns:
    dict[tuple[int,int,int,int], float]: Sparse two-body integral map,
    reconstructed from the canonical SymmetryBlockedSparseMap block.
)");

  sparse_container.def("two_body_integrals_sparse",
                       &SparseHamiltonianContainer::two_body_integrals_sparse,
                       nb::rv_policy::reference_internal,
                       R"(
Two-body integrals as a rank-4 symmetry-blocked sparse map.

Returns:
    SymmetryBlockedSparseMapRank4: The two-body sparse SBT.

Raises:
    RuntimeError: If two-body integrals are not set.
)");

  sparse_container.def("one_body_element",
                       &SparseHamiltonianContainer::one_body_element, R"(
Access a single one-body integral element.

Args:
    i (int): Row index.
    j (int): Column index.

Returns:
    float: Value of one-body integral (i, j).
)",
                       nb::arg("i"), nb::arg("j"));

  // -- Serialization --
  sparse_container.def(
      "to_json",
      [](const SparseHamiltonianContainer& self) -> std::string {
        return self.to_json().dump();
      },
      R"(
Convert the container to a JSON string.

Serialises one-body integrals as sparse COO triplets and two-body integrals
as sparse [p, q, r, s, value] entries.

Returns:
    str: JSON string with all container data.
)");

  sparse_container.def(
      "to_fcidump_file", &SparseHamiltonianContainer::to_fcidump_file, R"(
Save the Hamiltonian to an FCIDUMP file.

Args:
    filename (str): Path to the output file.
    nalpha (int): Number of alpha electrons.
    nbeta (int): Number of beta electrons.
)",
      nb::arg("filename"), nb::arg("nalpha"), nb::arg("nbeta"));

  // ============================================================================
  // DFTHCHamiltonianContainer - double-factorized tensor hypercontraction
  // ============================================================================
  nb::class_<DFTHCHamiltonianContainer, HamiltonianContainer>
      factorized_container(data, "DFTHCHamiltonianContainer", R"(
Restricted, spin-free, double-factorized tensor hypercontraction Hamiltonian.
)");

  factorized_container.def(
      nb::init<const Eigen::MatrixXd&, const Eigen::VectorXd&,
               const Eigen::VectorXd&, const Eigen::MatrixXd&,
               std::shared_ptr<Orbitals>, double, const Eigen::MatrixXd&,
               HamiltonianType>(),
      R"(
Constructor for a DFTHC Hamiltonian.

Args:
    one_body_integrals (numpy.ndarray): One-body integrals with shape [N,N] over the N active spatial orbitals.
    u_matrices (numpy.ndarray): U factors flattened in [R,B,N] order. Each basis row must be a unit vector.
    w_matrices (numpy.ndarray): W factors flattened in [R,B,C] order.
    wb_matrix (numpy.ndarray): Identity weights with shape [R,C].
    orbitals (Orbitals): Restricted orbitals with N active spatial orbitals.
    core_energy (float): Nuclear and inactive-core energy.
    inactive_fock_matrix (numpy.ndarray): Inactive Fock matrix over the full molecular-orbital space.
    type (HamiltonianType, optional): Hamiltonian type; defaults to Hermitian.

Raises:
    ValueError: If required data, dimensions, or restrictedness are inconsistent, or if a basis row of U is not normalized.
)",
      nb::arg("one_body_integrals"), nb::arg("u_matrices"),
      nb::arg("w_matrices"), nb::arg("wb_matrix"), nb::arg("orbitals"),
      nb::arg("core_energy"), nb::arg("inactive_fock_matrix"),
      nb::arg("type") = HamiltonianType::Hermitian);

  factorized_container.def(
      nb::init<const Eigen::MatrixXd&, const Eigen::VectorXd&,
               const Eigen::VectorXd&, std::shared_ptr<Orbitals>, double,
               const Eigen::MatrixXd&, HamiltonianType>(),
      R"(
Construct an ordinary double-factorized Hamiltonian with zero WB.

Args:
    one_body_integrals (numpy.ndarray): Conventional one-body integrals with shape [N,N].
    u_matrices (numpy.ndarray): U factors flattened in [R,N,N] order. Each basis row must be a unit vector.
    w_matrices (numpy.ndarray): W factors flattened in [R,N] order.
    orbitals (Orbitals): Restricted orbitals with N active spatial orbitals.
    core_energy (float): Nuclear and inactive-core energy.
    inactive_fock_matrix (numpy.ndarray): Inactive Fock matrix over the full molecular-orbital space.
    type (HamiltonianType, optional): Hamiltonian type; defaults to Hermitian.

Raises:
    ValueError: If the factors do not have ordinary DF dimensions or the general constructor rejects the data.
)",
      nb::arg("one_body_integrals"), nb::arg("u_matrices"),
      nb::arg("w_matrices"), nb::arg("orbitals"), nb::arg("core_energy"),
      nb::arg("inactive_fock_matrix"),
      nb::arg("type") = HamiltonianType::Hermitian);

  factorized_container.def("get_u_matrices",
                           &DFTHCHamiltonianContainer::get_u_matrices,
                           nb::rv_policy::reference_internal, R"(
Get U matrices as flat vector [R*B*N].

Returns:
    numpy.ndarray: U factors flattened in [R,B,N] order.
)");

  factorized_container.def("get_w_matrices",
                           &DFTHCHamiltonianContainer::get_w_matrices,
                           nb::rv_policy::reference_internal, R"(
Get W matrices as flat vector [R*B*C].

Returns:
    numpy.ndarray: W factors flattened in [R,B,C] order.
)");

  factorized_container.def("get_wb_matrix",
                           &DFTHCHamiltonianContainer::get_wb_matrix,
                           nb::rv_policy::reference_internal, R"(
Get WB identity weight matrix [R x C].

Returns:
    numpy.ndarray: Identity weights with shape [R,C].
)");

  factorized_container.def("get_num_orbitals",
                           &DFTHCHamiltonianContainer::get_num_orbitals,
                           R"(
Number of spatial orbitals (N).

Returns:
    int: Number of active spatial orbitals.
)");

  factorized_container.def("get_num_ranks",
                           &DFTHCHamiltonianContainer::get_num_ranks, R"(
Number of ranks (R).

Returns:
    int: Number of ranks in the factorization.
)");

  factorized_container.def("get_num_bases",
                           &DFTHCHamiltonianContainer::get_num_bases, R"(
Number of bases per rank (B).

Returns:
    int: Number of bases per rank.
)");

  factorized_container.def("get_num_copies",
                           &DFTHCHamiltonianContainer::get_num_copies, R"(
Number of copies per rank (C).

Returns:
    int: Number of copies per rank.
)");

  factorized_container.def("get_lambda", &DFTHCHamiltonianContainer::get_lambda,
                           R"(
Compute the normalization from the adjusted one-body matrix and factors.

Returns:
    float: Block-encoding normalization.

Raises:
    RuntimeError: If the adjusted one-body matrix is not symmetric, or if it cannot be diagonalized.
)");

  factorized_container.def("get_h1_prime",
                           &DFTHCHamiltonianContainer::get_h1_prime, R"(
Return the adjusted one-body matrix used by the DFTHC block encoding.

Returns:
    numpy.ndarray: Adjusted one-body matrix with shape [N,N].
)");

  factorized_container.def(
      "reconstruct_two_body_integrals",
      &DFTHCHamiltonianContainer::reconstruct_two_body_integrals, R"(
Reconstruct the two-body integrals from U and W.

Returns:
    numpy.ndarray: New flat N**4 array in [p,q,r,s] order.
)");

  factorized_container.def("get_two_body_integrals",
                           &DFTHCHamiltonianContainer::get_two_body_integrals,
                           nb::rv_policy::reference_internal, R"(
Return the lazily reconstructed two-body integrals.

Returns:
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: The aaaa, aabb, and bbbb views of one cached N**4 vector.
)");

  factorized_container.def(
      "get_two_body_element", &DFTHCHamiltonianContainer::get_two_body_element,
      R"(
Get specific two-electron integral element <ij|kl>.

Args:
    i, j, k, l (int): Orbital indices.
    channel (SpinChannel, optional): Ignored because the container is restricted; defaults to aaaa.

Returns:
    float: Two-electron integral ``(ij|kl)``.

Raises:
    IndexError: If any orbital index is outside [0,N).
)",
      nb::arg("i"), nb::arg("j"), nb::arg("k"), nb::arg("l"),
      nb::arg("channel") = SpinChannel::aaaa);

  factorized_container.def("has_two_body_integrals",
                           &DFTHCHamiltonianContainer::has_two_body_integrals,
                           R"(
Check if two-body integrals are available.

Returns:
    bool: Whether U and W factors are available.
)");

  factorized_container.def("is_restricted",
                           &DFTHCHamiltonianContainer::is_restricted, R"(
Return whether the container is restricted.

Returns:
    bool: Always ``True``.
)");

  factorized_container.def("is_valid", &DFTHCHamiltonianContainer::is_valid, R"(
Check if the Hamiltonian data is complete and consistent.

Returns:
    bool: Whether required data and dimensions are valid.
)");

  factorized_container.def(
      "to_json",
      [](const DFTHCHamiltonianContainer& self) -> std::string {
        return self.to_json().dump();
      },
      R"(
Convert container to JSON string.

Returns:
    str: JSON representation of the container.
)");

  factorized_container.def(
      "__repr__", [](const DFTHCHamiltonianContainer& self) -> std::string {
        return "<DFTHCHamiltonianContainer N=" +
               std::to_string(self.get_num_orbitals()) +
               " R=" + std::to_string(self.get_num_ranks()) +
               " B=" + std::to_string(self.get_num_bases()) +
               " C=" + std::to_string(self.get_num_copies()) + ">";
      });

  // ============================================================================
  // Hamiltonian - Interface class
  // ============================================================================
  nb::class_<Hamiltonian, DataClass> hamiltonian(data, "Hamiltonian", R"(
Interface class for molecular Hamiltonians in the molecular orbital basis.

This class provides a unified interface to molecular Hamiltonian data by
wrapping a HamiltonianContainer implementation. It supports:

* One-electron integrals (kinetic + nuclear attraction) in MO representation
* Two-electron integrals (electron-electron repulsion) in MO representation
* Molecular orbital information for the active space
* A constant (zero-body) energy term

The actual integral storage is handled by the underlying container, which
can use different representations (canonical 4-center, density-fitted, etc.).

Examples:
    >>> # Create a Hamiltonian from a CanonicalFourCenterHamiltonianContainer container
    >>> container = CanonicalFourCenterHamiltonianContainer(h1, h2, orbitals, e_core, fock)
    >>> hamiltonian = Hamiltonian(container)
    >>>
    >>> # Access integrals through the interface
    >>> h1_alpha, h1_beta = hamiltonian.get_one_body_integrals()
    >>> core_energy = hamiltonian.get_core_energy
)");

  // Constructor from container
  hamiltonian.def(nb::init<std::shared_ptr<HamiltonianContainer>>(),
                  R"(
Construct a Hamiltonian from a HamiltonianContainer.

Args:
    container (HamiltonianContainer): The container holding the Hamiltonian data.
        The immutable container is shared with the Hamiltonian.

Examples:
    >>> container = CanonicalFourCenterHamiltonianContainer(h1, h2, orbitals, e_core, fock)
    >>> hamiltonian = Hamiltonian(container)
)",
                  nb::arg("container"));

  // One-body integral access
  hamiltonian.def("get_one_body_integrals",
                  &Hamiltonian::get_one_body_integrals,
                  nb::rv_policy::reference_internal,
                  R"(
Deprecated: use the underlying container's one_body_integrals() instead.

Returns:
    tuple[numpy.ndarray, numpy.ndarray]: One-electron integral matrices [norb x norb]
    for alpha and beta spin channels.

Raises:
    RuntimeError: If one-body integrals have not been set

Examples:
    >>> h1_alpha, h1_beta = hamiltonian.get_one_body_integrals()
    >>> print(f"One-body matrix shape: {h1_alpha.shape}")
)");

  hamiltonian.def("has_one_body_integrals",
                  &Hamiltonian::has_one_body_integrals,
                  R"(
Check if one-body integrals are available.

Returns:
    bool: True if one-body integrals have been set
)");

  hamiltonian.def("get_one_body_element", &Hamiltonian::get_one_body_element,
                  R"(
Get specific one-electron integral element <ij>.

Args:
    i (int): First orbital index
    j (int): Second orbital index
    channel (SpinChannel): Spin channel (aa or bb), defaults to aa

Returns:
    float: Value of the one-electron integral <ij>
)",
                  nb::arg("i"), nb::arg("j"),
                  nb::arg("channel") = SpinChannel::aa);

  // Two-body integral access
  hamiltonian.def("get_two_body_integrals",
                  &Hamiltonian::get_two_body_integrals,
                  nb::rv_policy::reference_internal,
                  R"(
Deprecated: use the underlying container's two_body_integrals() instead when available.

Returns:
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Tuple of two-electron
    integral vectors [norb^4] for aaaa, aabb, and bbbb spin channels.

Raises:
    RuntimeError: If two-body integrals have not been set

Notes:
    Integrals are stored as flattened vectors in chemist notation <ij|kl>
)");

  hamiltonian.def("get_two_body_element", &Hamiltonian::get_two_body_element,
                  R"(
Get specific two-electron integral element <ij|kl>.

Args:
    i, j, k, l (int): Orbital indices
    channel (SpinChannel): Spin channel (aaaa, aabb, or bbbb), defaults to aaaa

Returns:
    float: Value of the two-electron integral <ij|kl>
)",
                  nb::arg("i"), nb::arg("j"), nb::arg("k"), nb::arg("l"),
                  nb::arg("channel") = SpinChannel::aaaa);

  hamiltonian.def("has_two_body_integrals",
                  &Hamiltonian::has_two_body_integrals,
                  R"(
Check if two-body integrals are available.

Returns:
    bool: True if two-body integrals have been set
)");

  // Orbital information
  bind_getter_as_property(hamiltonian, "get_orbitals",
                          &Hamiltonian::get_orbitals,
                          R"(
Get molecular orbital data.

Returns:
    Orbitals: The orbital data associated with this Hamiltonian

Raises:
    RuntimeError: If orbital data has not been set
)",
                          nb::rv_policy::reference_internal);

  hamiltonian.def("has_orbitals", &Hamiltonian::has_orbitals,
                  R"(
Check if orbital data is available.

Returns:
    bool: True if orbital data has been set
)");

  // Core energy and Fock matrix
  bind_getter_as_property(hamiltonian, "get_core_energy",
                          &Hamiltonian::get_core_energy,
                          R"(
Get the constant (zero-body) energy term in atomic units.

Returns:
    float: Constant energy term in Hartree
)");

  hamiltonian.def("has_inactive_fock_matrix",
                  &Hamiltonian::has_inactive_fock_matrix,
                  R"(
Check if inactive Fock matrix is available.

Returns:
    bool: True if inactive Fock matrix has been set
)");

  hamiltonian.def("get_inactive_fock_matrix",
                  &Hamiltonian::get_inactive_fock_matrix,
                  nb::rv_policy::reference_internal,
                  R"(
Deprecated: use the underlying container's inactive_fock() instead.

Returns:
    tuple[numpy.ndarray, numpy.ndarray]: Inactive Fock matrices for the active space
)");

  // Type and restriction checks
  bind_getter_as_property(hamiltonian, "get_type", &Hamiltonian::get_type,
                          R"(
Get the type of Hamiltonian (Hermitian or NonHermitian).

Returns:
    HamiltonianType: The Hamiltonian type
)");

  hamiltonian.def("is_hermitian", &Hamiltonian::is_hermitian,
                  R"(
Check if the Hamiltonian is Hermitian.

Returns:
    bool: True if the Hamiltonian type is Hermitian
)");

  hamiltonian.def("is_restricted", &Hamiltonian::is_restricted,
                  R"(
Check if Hamiltonian is restricted (alpha == beta).

Returns:
    bool: True if alpha and beta integrals are identical
)");

  hamiltonian.def("is_unrestricted", &Hamiltonian::is_unrestricted,
                  R"(
Check if Hamiltonian is unrestricted (alpha != beta).

Returns:
    bool: True if alpha and beta integrals are different
)");

  // Container access
  hamiltonian.def("get_container_type", &Hamiltonian::get_container_type,
                  R"(
Get the type of the underlying container.

Returns:
    str: Container type identifier (e.g., "canonical_four_center")
)");

  hamiltonian.def(
      "get_container",
      [](const Hamiltonian& self) -> const HamiltonianContainer& {
        return self.get_container<HamiltonianContainer>();
      },
      nb::rv_policy::reference_internal,
      R"(
Get the underlying container.

Returns:
    HamiltonianContainer: The container holding this Hamiltonian's data.
)");

  // Summary
  hamiltonian.def("get_summary", &Hamiltonian::get_summary,
                  R"(
Get a human-readable summary of the Hamiltonian data.

Returns:
    str: Multi-line string describing the Hamiltonian properties
)");

  // Generic file I/O
  hamiltonian.def("to_file", hamiltonian_to_file_wrapper, R"(
Save Hamiltonian to file based on type parameter.

Args:
    filename (str or pathlib.Path): Path to file to create/overwrite
    type (str): File format type ('json' or 'hdf5')

Raises:
    ValueError: If format_type is not supported
    RuntimeError: If I/O error occurs

Examples:
    >>> hamiltonian.to_file("water.hamiltonian.json", "json")
    >>> hamiltonian.to_file("molecule.hamiltonian.h5", "hdf5")
)",
                  nb::arg("filename"), nb::arg("type"));

  hamiltonian.def_static("from_file", hamiltonian_from_file_wrapper, R"(
Load Hamiltonian from file based on type parameter.

Args:
    filename (str or pathlib.Path): Path to file to read
    type (str): File format type ('json' or 'hdf5')

Returns:
    Hamiltonian: New Hamiltonian loaded from file

Raises:
    ValueError: If format_type is not supported
    RuntimeError: If file doesn't exist or I/O error occurs

Examples:
    >>> hamiltonian = Hamiltonian.from_file("water.hamiltonian.json", "json")
    >>> hamiltonian = Hamiltonian.from_file("molecule.hamiltonian.h5", "hdf5")
)",
                         nb::arg("filename"), nb::arg("type"));

  // JSON file I/O
  hamiltonian.def("to_json_file", hamiltonian_to_json_file_wrapper, R"(
Save Hamiltonian to JSON file (with validation).

Args:
    filename (str or pathlib.Path): Path to JSON file to create/overwrite.
        Must have ``.hamiltonian`` before the file extension.

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If an I/O error occurs

Examples:
    >>> hamiltonian.to_json_file("water.hamiltonian.json")
)",
                  nb::arg("filename"));

  hamiltonian.def_static("from_json_file", hamiltonian_from_json_file_wrapper,
                         R"(
Load Hamiltonian from JSON file.

Args:
    filename (str or pathlib.Path): Path to JSON file to read.
        Must have ``.hamiltonian`` before the file extension.

Returns:
    Hamiltonian: New Hamiltonian loaded from file

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If file doesn't exist or I/O error occurs

Examples:
    >>> hamiltonian = Hamiltonian.from_json_file("water.hamiltonian.json")
)",
                         nb::arg("filename"));

  // HDF5 file I/O
  hamiltonian.def("to_hdf5_file", hamiltonian_to_hdf5_file_wrapper, R"(
Save Hamiltonian to HDF5 file (with validation).

Args:
    filename (str or pathlib.Path): Path to HDF5 file to create/overwrite.
        Must have ``.hamiltonian`` before the file extension.

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If I/O error occurs

Examples:
    >>> hamiltonian.to_hdf5_file("water.hamiltonian.h5")
)",
                  nb::arg("filename"));

  hamiltonian.def_static("from_hdf5_file", hamiltonian_from_hdf5_file_wrapper,
                         R"(
Load Hamiltonian from HDF5 file.

Args:
    filename (str or pathlib.Path): Path to HDF5 file to read.
        Must have ``.hamiltonian`` before the file extension.

Returns:
    Hamiltonian: New Hamiltonian loaded from file

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If file doesn't exist or I/O error occurs

Examples:
    >>> hamiltonian = Hamiltonian.from_hdf5_file("water.hamiltonian.h5")
)",
                         nb::arg("filename"));

  // FCIDUMP file I/O
  hamiltonian.def("to_fcidump_file", hamiltonian_to_fcidump_file_wrapper,
                  R"(
Save Hamiltonian to FCIDUMP file.

Args:
    filename (str or pathlib.Path): Path to FCIDUMP file to create/overwrite
    nalpha (int): Number of alpha electrons
    nbeta (int): Number of beta electrons

Raises:
    RuntimeError: If I/O error occurs

Examples:
    >>> hamiltonian.to_fcidump_file("water.fcidump", 5, 5)

Notes:
    FCIDUMP format is a standard quantum chemistry format for storing
    molecular integrals and is widely supported by quantum chemistry codes.
)",
                  nb::arg("filename"), nb::arg("nalpha"), nb::arg("nbeta"));

  // JSON serialization
  hamiltonian.def(
      "to_json",
      [](const Hamiltonian& self) -> std::string {
        return self.to_json().dump();
      },
      R"(
Convert Hamiltonian to JSON string.

Returns:
    str: JSON representation of the Hamiltonian
)");

  hamiltonian.def_static(
      "from_json",
      [](const std::string& json_str) -> Hamiltonian {
        return *Hamiltonian::from_json(nlohmann::json::parse(json_str));
      },
      R"(
Load Hamiltonian from JSON string.

Args:
    json_data (str): JSON string containing Hamiltonian data

Returns:
    Hamiltonian: New Hamiltonian loaded from JSON

Raises:
    RuntimeError: If JSON is malformed or contains invalid data

Examples:
    >>> json_str = hamiltonian.to_json()
    >>> loaded = Hamiltonian.from_json(json_str)
)",
      nb::arg("json_data"));

  // String representation
  hamiltonian.def("__repr__",
                  [](const Hamiltonian& h) { return h.get_summary(); });

  hamiltonian.def("__str__",
                  [](const Hamiltonian& h) { return h.get_summary(); });

  // Pickling support using JSON serialization
  hamiltonian
      .def("__getstate__",
           [](const Hamiltonian& h) {
             // __getstate__ - serialize to JSON string
             return h.to_json().dump();
           })
      .def("__setstate__", [](Hamiltonian* self, const std::string& json_str) {
        // __setstate__ - deserialize from JSON string
        new (self) Hamiltonian(
            *Hamiltonian::from_json(nlohmann::json::parse(json_str)));
      });

  // Data type name class attribute
  hamiltonian.def_static("data_type_name", &Hamiltonian::data_type_name, R"(
Return the wire-format identifier for Hamiltonians.

Returns:
        str: ``"hamiltonian"``

)");
}
