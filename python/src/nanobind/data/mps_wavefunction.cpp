// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <nanobind/nanobind.h>

#include <complex>
#include <memory>
#include <optional>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <qdk/chemistry/data/wavefunction_containers/mps_wavefunction.hpp>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include "eigen.hpp"
#include "stl.hpp"

namespace nb = nanobind;
using namespace qdk::chemistry::data;

namespace {

template <typename Scalar>
std::shared_ptr<MPSSite> site_from_tensor(
    std::shared_ptr<const SymmetryBlockedTensor<3, Scalar>> tensor,
    std::vector<SymmetryLabel> left_order,
    std::vector<SymmetryLabel> physical_order,
    std::vector<SymmetryLabel> right_order,
    std::vector<Configuration> physical_basis) {
  if (!tensor) {
    throw std::invalid_argument("MPS site tensor must not be null.");
  }
  return std::make_shared<MPSSite>(
      std::make_shared<const MPSSite::TensorVariant>(*tensor),
      MPSSite::SectorOrders{std::move(left_order), std::move(physical_order),
                            std::move(right_order)},
      std::move(physical_basis));
}

}  // namespace

void bind_mps_wavefunction(nb::module_& data) {
  nb::class_<MPSSite>(
      data, "MPSSite",
      "An explicit rank-3 MPS tensor with (left, physical, right) slots. "
      "Each block packs local_left * physical_extent + local_physical into "
      "rows and local_right into columns.")
      .def(nb::new_(&site_from_tensor<double>), nb::arg("tensor").none(),
           nb::arg("left_sector_order"), nb::arg("physical_sector_order"),
           nb::arg("right_sector_order"),
           nb::arg("physical_basis") = std::vector<Configuration>{})
      .def(nb::new_(&site_from_tensor<std::complex<double>>),
           nb::arg("tensor").none(), nb::arg("left_sector_order"),
           nb::arg("physical_sector_order"), nb::arg("right_sector_order"),
           nb::arg("physical_basis") = std::vector<Configuration>{})
      .def(
          "tensor",
          [](const MPSSite& site) {
            return std::visit(
                [](const auto& value) {
                  return nb::cast(value, nb::rv_policy::copy);
                },
                site.tensor());
          },
          "A copy of the bound real or complex rank-3 symmetry-blocked tensor.")
      .def("sector_orders", &MPSSite::sector_orders)
      .def("left_sector_order", &MPSSite::left_sector_order)
      .def("physical_sector_order", &MPSSite::physical_sector_order)
      .def("right_sector_order", &MPSSite::right_sector_order)
      .def("physical_basis", &MPSSite::physical_basis,
           "One-mode configurations in physical-index order.")
      .def("physical_dimension", &MPSSite::physical_dimension)
      .def("left_bond_dimension", &MPSSite::left_bond_dimension)
      .def("right_bond_dimension", &MPSSite::right_bond_dimension)
      .def("is_complex", &MPSSite::is_complex)
      .def("shape",
           [](const MPSSite& site) {
             return nb::make_tuple(site.left_bond_dimension(),
                                   site.physical_dimension(),
                                   site.right_bond_dimension());
           })
      .def("to_dense", &MPSSite::to_dense,
           "Return a packed matrix of shape (left * physical, right). "
           "Use site.to_dense().reshape(site.shape()) for three-dimensional "
           "indexing.");

  nb::class_<MPSContainer, WavefunctionContainer>(
      data, "MPSContainer",
      "Container to store MPS over an orbital or model-mode basis. Sites may "
      "have different local bases. Scale, phase and gauge are preserved; "
      "particle counts and canonical center are optional, unverified producer "
      "metadata.")
      .def(nb::init<std::vector<MPSContainer::SitePtr>,
                    std::shared_ptr<Orbitals>,
                    std::shared_ptr<const MPSContainer::ParticleCount>,
                    std::optional<std::size_t>, std::vector<std::size_t>,
                    const std::optional<ContainerTypes::MatrixVariant>&,
                    const std::optional<ContainerTypes::MatrixVariant>&,
                    const std::optional<ContainerTypes::MatrixVariant>&,
                    const std::optional<ContainerTypes::VectorVariant>&,
                    const std::optional<ContainerTypes::VectorVariant>&,
                    const std::optional<ContainerTypes::VectorVariant>&,
                    const std::optional<ContainerTypes::VectorVariant>&>(),
           nb::arg("sites"), nb::arg("orbitals").none(),
           nb::arg("active_num_particles").none() = nullptr,
           nb::arg("orthogonality_center") = std::nullopt,
           nb::arg("site_to_orbital_order") = std::vector<std::size_t>{},
           nb::arg("one_rdm_spin_traced") = std::nullopt,
           nb::arg("one_rdm_aa") = std::nullopt,
           nb::arg("one_rdm_bb") = std::nullopt,
           nb::arg("two_rdm_spin_traced") = std::nullopt,
           nb::arg("two_rdm_aaaa") = std::nullopt,
           nb::arg("two_rdm_aabb") = std::nullopt,
           nb::arg("two_rdm_bbbb") = std::nullopt,
           "Store supplied sites and optional real/complex RDM arrays. RDMs "
           "use active-orbital order. ")
      .def("sites", &MPSContainer::sites)
      .def("has_active_num_particles", &MPSContainer::has_active_num_particles)
      .def("orthogonality_center", &MPSContainer::orthogonality_center,
           "Asserted canonical center, or None if unknown.")
      .def("site_to_orbital_order", &MPSContainer::site_to_orbital_order,
           "Permutation of active-mode slots, not full orbital indices.")
      .def("num_sites", &MPSContainer::num_sites)
      .def("max_bond_dimension", &MPSContainer::max_bond_dimension);
}
