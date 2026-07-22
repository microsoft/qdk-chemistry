// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <complex>
#include <map>
#include <memory>
#include <numeric>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <qdk/chemistry/data/wavefunction_containers/abelian_mps_wavefunction.hpp>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace py = pybind11;
using namespace qdk::chemistry::data;

namespace {

template <typename Scalar>
AbelianMPSSite::PhysicalSlicePtr make_trivial_slice(
    const Eigen::Ref<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>& matrix) {
  using Slice = SymmetryBlockedTensor<2, Scalar>;
  auto trivial =
      std::make_shared<const SymmetryProduct>(SymmetryProduct::trivial());
  typename Slice::ExtentsArray extents;
  extents[0][SymmetryLabel{}] = static_cast<std::size_t>(matrix.rows());
  extents[1][SymmetryLabel{}] = static_cast<std::size_t>(matrix.cols());
  typename Slice::BlockMap blocks;
  blocks[{SymmetryLabel{}, SymmetryLabel{}}] = std::make_shared<
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>(matrix);
  Slice slice({trivial, trivial}, std::move(extents), std::move(blocks));
  return std::make_shared<const AbelianMPSSite::PhysicalSlice>(std::move(slice));
}

template <typename Scalar>
std::shared_ptr<AbelianMPSSite> site_from_slices(
    std::vector<std::shared_ptr<const SymmetryBlockedTensor<2, Scalar>>>
        physical_slices,
    std::vector<SymmetryLabel> left_sector_order,
    std::vector<SymmetryLabel> right_sector_order) {
  std::vector<AbelianMPSSite::PhysicalSlicePtr> variants;
  variants.reserve(physical_slices.size());
  for (auto& slice : physical_slices) {
    if (!slice) {
      throw std::invalid_argument(
          "MPS physical slice pointers must not be null.");
    }
    variants.push_back(std::make_shared<const AbelianMPSSite::PhysicalSlice>(*slice));
  }
  return std::make_shared<AbelianMPSSite>(std::move(variants),
                                   std::move(left_sector_order),
                                   std::move(right_sector_order));
}

template <typename Scalar>
std::shared_ptr<AbelianMPSSite> site_from_dense(
    py::array_t<Scalar, py::array::c_style | py::array::forcecast> tensor) {
  const auto values = tensor.template unchecked<3>();
  std::vector<AbelianMPSSite::PhysicalSlicePtr> slices;
  slices.reserve(static_cast<std::size_t>(values.shape(1)));
  for (py::ssize_t physical = 0; physical < values.shape(1); ++physical) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix(
        values.shape(0), values.shape(2));
    for (py::ssize_t left = 0; left < values.shape(0); ++left) {
      for (py::ssize_t right = 0; right < values.shape(2); ++right) {
        matrix(left, right) = values(left, physical, right);
      }
    }
    slices.push_back(make_trivial_slice<Scalar>(matrix));
  }
  return std::make_shared<AbelianMPSSite>(std::move(slices),
                                   std::vector<SymmetryLabel>{SymmetryLabel{}},
                                   std::vector<SymmetryLabel>{SymmetryLabel{}});
}

std::shared_ptr<AbelianMPSSite> site_from_dense_dispatch(const py::array& tensor) {
  if (tensor.dtype().kind() == 'c') {
    return site_from_dense<std::complex<double>>(tensor);
  }
  return site_from_dense<double>(tensor);
}

/**
 * @brief Create a particle-number-blocked AbelianMPSSite from a dense tensor.
 *
 * Given a tensor of shape (chi_left, d, chi_right), sector size maps for left
 * and right bonds, and the per-physical-state particle count delta, extracts
 * the appropriate sub-blocks from the dense tensor.
 *
 * @param tensor Dense tensor array of shape (chi_left, d, chi_right).
 * @param left_sector_sizes Map from particle number to sector dimension (left).
 * @param right_sector_sizes Map from particle number to sector dimension
 * (right).
 * @param delta_n Per-physical-state particle-number change. Physical state
 *        @c p connects left sector @c n to right sector
 *        <tt>n + delta_n[p]</tt>. Must have length @c d.
 * @param max_particle_number Maximum particle number for the axis.
 */
template <typename Scalar>
std::shared_ptr<AbelianMPSSite> site_from_dense_abelian(
    py::array_t<Scalar, py::array::c_style | py::array::forcecast> tensor,
    std::unordered_map<std::size_t, std::size_t> left_sector_sizes,
    std::unordered_map<std::size_t, std::size_t> right_sector_sizes,
    std::vector<std::size_t> delta_n, std::size_t max_particle_number) {
  const auto values = tensor.template unchecked<3>();
  const auto chi_left = static_cast<std::size_t>(values.shape(0));
  const auto d = static_cast<std::size_t>(values.shape(1));
  const auto chi_right = static_cast<std::size_t>(values.shape(2));

  if (delta_n.size() != d) {
    throw std::invalid_argument(
        "delta_n length must equal the physical dimension.");
  }

  // Build symmetry product.
  auto symmetries = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::particle_number(max_particle_number)}));

  // Compute cumulative offsets for left and right sectors.
  // Sectors are ordered by particle number.
  std::map<std::size_t, std::size_t> left_offsets, right_offsets;
  std::size_t offset = 0;
  for (auto& [n, dim] : std::map<std::size_t, std::size_t>(
           left_sector_sizes.begin(), left_sector_sizes.end())) {
    left_offsets[n] = offset;
    offset += dim;
  }
  if (offset != chi_left) {
    throw std::invalid_argument(
        "Sum of left sector sizes must equal chi_left.");
  }
  offset = 0;
  for (auto& [n, dim] : std::map<std::size_t, std::size_t>(
           right_sector_sizes.begin(), right_sector_sizes.end())) {
    right_offsets[n] = offset;
    offset += dim;
  }
  if (offset != chi_right) {
    throw std::invalid_argument(
        "Sum of right sector sizes must equal chi_right.");
  }

  // Build extents.
  using Slice = SymmetryBlockedTensor<2, Scalar>;
  typename Slice::ExtentsArray extents;
  for (auto& [n, dim] : left_sector_sizes) {
    extents[0][SymmetryLabel({axes::particle_number_value(n)})] = dim;
  }
  for (auto& [n, dim] : right_sector_sizes) {
    extents[1][SymmetryLabel({axes::particle_number_value(n)})] = dim;
  }

  // Build sector order vectors.
  std::vector<SymmetryLabel> left_sector_order, right_sector_order;
  for (auto& [n, _] : std::map<std::size_t, std::size_t>(
           left_sector_sizes.begin(), left_sector_sizes.end())) {
    left_sector_order.push_back(
        SymmetryLabel({axes::particle_number_value(n)}));
  }
  for (auto& [n, _] : std::map<std::size_t, std::size_t>(
           right_sector_sizes.begin(), right_sector_sizes.end())) {
    right_sector_order.push_back(
        SymmetryLabel({axes::particle_number_value(n)}));
  }

  // Extract blocks for each physical state.
  std::vector<AbelianMPSSite::PhysicalSlicePtr> slices;
  for (std::size_t p = 0; p < d; ++p) {
    typename Slice::BlockMap blocks;
    for (auto& [n_left, left_dim] : left_sector_sizes) {
      std::size_t n_right = n_left + delta_n[p];
      auto it = right_sector_sizes.find(n_right);
      if (it == right_sector_sizes.end()) {
        continue;
      }
      std::size_t right_dim = it->second;
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> block(
          static_cast<Eigen::Index>(left_dim),
          static_cast<Eigen::Index>(right_dim));
      for (std::size_t i = 0; i < left_dim; ++i) {
        for (std::size_t j = 0; j < right_dim; ++j) {
          block(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
              values(static_cast<py::ssize_t>(left_offsets[n_left] + i),
                     static_cast<py::ssize_t>(p),
                     static_cast<py::ssize_t>(right_offsets[n_right] + j));
        }
      }
      SymmetryLabel left_label({axes::particle_number_value(n_left)});
      SymmetryLabel right_label({axes::particle_number_value(n_right)});
      blocks[{left_label, right_label}] = std::make_shared<
          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>(
          std::move(block));
    }
    slices.push_back(std::make_shared<const AbelianMPSSite::PhysicalSlice>(
        Slice({symmetries, symmetries}, extents, std::move(blocks))));
  }
  return std::make_shared<AbelianMPSSite>(std::move(slices),
                                   std::move(left_sector_order),
                                   std::move(right_sector_order));
}

std::shared_ptr<AbelianMPSSite> site_from_dense_abelian_dispatch(
    const py::array& tensor,
    std::unordered_map<std::size_t, std::size_t> left_sector_sizes,
    std::unordered_map<std::size_t, std::size_t> right_sector_sizes,
    std::vector<std::size_t> delta_n, std::size_t max_particle_number) {
  if (tensor.dtype().kind() == 'c') {
    return site_from_dense_abelian<std::complex<double>>(
        tensor, std::move(left_sector_sizes), std::move(right_sector_sizes),
        std::move(delta_n), max_particle_number);
  }
  return site_from_dense_abelian<double>(
      tensor, std::move(left_sector_sizes), std::move(right_sector_sizes),
      std::move(delta_n), max_particle_number);
}

template <typename Scalar>
py::array_t<Scalar> unpack_dense(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& packed,
    std::size_t physical_dimension) {
  const auto left_dimension =
      packed.rows() / static_cast<Eigen::Index>(physical_dimension);
  py::array_t<Scalar> result({left_dimension,
                              static_cast<Eigen::Index>(physical_dimension),
                              packed.cols()});
  auto values = result.template mutable_unchecked<3>();
  for (Eigen::Index left = 0; left < left_dimension; ++left) {
    for (std::size_t physical = 0; physical < physical_dimension; ++physical) {
      for (Eigen::Index right = 0; right < packed.cols(); ++right) {
        values(left, static_cast<py::ssize_t>(physical), right) =
            packed(left * static_cast<Eigen::Index>(physical_dimension) +
                       static_cast<Eigen::Index>(physical),
                   right);
      }
    }
  }
  return result;
}

py::object site_to_dense(const AbelianMPSSite& site) {
  return std::visit(
      [&](const auto& dense) -> py::object {
        return unpack_dense(dense, site.physical_dimension());
      },
      site.to_dense());
}

template <typename Extents>
std::unordered_map<SymmetryLabel, std::size_t> sector_offsets(
    const Extents& extents, const std::vector<SymmetryLabel>& order) {
  std::unordered_map<SymmetryLabel, std::size_t> offsets;
  std::size_t offset = 0;
  for (const auto& label : order) {
    offsets.emplace(label, offset);
    offset += extents.at(label);
  }
  return offsets;
}

template <typename Slice>
py::object slice_to_csc(const Slice& slice,
                        const std::vector<SymmetryLabel>& left_order,
                        const std::vector<SymmetryLabel>& right_order) {
  using Scalar = typename Slice::BlockPtr::element_type::Scalar;
  std::vector<Scalar> values;
  std::vector<std::size_t> rows;
  std::vector<std::size_t> columns;
  const auto left_offsets = sector_offsets(slice.extents()[0], left_order);
  const auto right_offsets = sector_offsets(slice.extents()[1], right_order);
  for (const auto& [labels, block] : slice.blocks()) {
    for (Eigen::Index left = 0; left < block->rows(); ++left) {
      for (Eigen::Index right = 0; right < block->cols(); ++right) {
        if ((*block)(left, right) != Scalar{}) {
          values.push_back((*block)(left, right));
          rows.push_back(left_offsets.at(labels[0]) +
                         static_cast<std::size_t>(left));
          columns.push_back(right_offsets.at(labels[1]) +
                            static_cast<std::size_t>(right));
        }
      }
    }
  }
  auto total_extent = [](const auto& extents) {
    return std::accumulate(extents.begin(), extents.end(), std::size_t{},
                           [](std::size_t total, const auto& item) {
                             return total + item.second;
                           });
  };
  auto coordinates = py::make_tuple(py::cast(rows), py::cast(columns));
  auto shape = py::make_tuple(total_extent(slice.extents()[0]),
                              total_extent(slice.extents()[1]));
  return py::module_::import("scipy.sparse")
      .attr("csc_array")(py::make_tuple(py::cast(values), coordinates),
                         py::arg("shape") = shape);
}

py::list physical_slices(const AbelianMPSSite& site) {
  py::list result;
  for (const auto& slice : site.physical_slices()) {
    result.append(std::visit(
        [&](const auto& value) {
          return slice_to_csc(value, site.left_sector_order(),
                              site.right_sector_order());
        },
        *slice));
  }
  return result;
}

}  // namespace

void bind_mps_wavefunction(py::module& data) {
  py::class_<AbelianMPSSite, py::smart_holder>(
      data, "AbelianMPSSite",
      "One MPS site stored as a symmetry-blocked matrix for each physical "
      "state. All slices share common left and right bond spaces.")
      .def(py::init(&site_from_slices<double>), py::arg("physical_slices"),
           py::arg("left_sector_order"), py::arg("right_sector_order"),
           "Construct a real-valued site from symmetry-blocked physical "
           "slices. Sector-order arguments specify how sector-local rows and "
           "columns are concatenated when forming dense bond indices.")
      .def(py::init(&site_from_slices<std::complex<double>>),
           py::arg("physical_slices"), py::arg("left_sector_order"),
           py::arg("right_sector_order"),
           "Construct a complex-valued site from symmetry-blocked physical "
           "slices. Sector-order arguments specify how sector-local rows and "
           "columns are concatenated when forming dense bond indices.")
      .def_static(
          "from_dense", &site_from_dense_dispatch, py::arg("tensor"),
          "Construct an unsymmetrized site from an array with shape "
          "(chi_left, physical_dimension, chi_right). The result has one "
          "trivial block per physical state and cannot be used to construct "
          "an AbelianMPSContainer.")
      .def_static("from_dense_complex", &site_from_dense<std::complex<double>>,
                  py::arg("tensor"),
                  "Construct an unsymmetrized complex-valued site from an "
                  "array with shape (chi_left, physical_dimension, "
                  "chi_right). Prefer from_dense for automatic dtype "
                  "dispatch.")
      .def_static("from_dense_abelian", &site_from_dense_abelian_dispatch,
                  py::arg("tensor"), py::arg("left_sector_sizes"),
                  py::arg("right_sector_sizes"), py::arg("delta_n"),
                  py::arg("max_particle_number"),
                  "Construct a particle-number-blocked site from a dense "
                  "tensor and bond-sector sizes. For physical state p, only "
                  "the block connecting particle-number sectors n and "
                  "n + delta_n[p] is extracted. Sectors are packed in "
                  "ascending particle-number order.")
      .def_property_readonly(
          "physical_slices", &physical_slices,
          "Per-physical-state matrices as scipy.sparse.csc_array objects. "
          "Rows and columns follow left_sector_order and "
          "right_sector_order; absent symmetry blocks and zero block entries "
          "appear as sparse zeros.")
      .def_property_readonly(
          "left_sector_order", &AbelianMPSSite::left_sector_order,
          "Left-bond sector labels in dense row-packing order.")
      .def_property_readonly(
          "right_sector_order", &AbelianMPSSite::right_sector_order,
          "Right-bond sector labels in dense column-packing order.")
      .def_property_readonly("physical_dimension", &AbelianMPSSite::physical_dimension,
                             "Number of physical-state slices at this site.")
      .def_property_readonly("left_bond_dimension",
                             &AbelianMPSSite::left_bond_dimension,
                             "Total left-bond dimension across all sectors.")
      .def_property_readonly("right_bond_dimension",
                             &AbelianMPSSite::right_bond_dimension,
                             "Total right-bond dimension across all sectors.")
      .def_property_readonly("is_complex", &AbelianMPSSite::is_complex,
                             "Whether all site amplitudes use complex scalars.")
      .def_property_readonly(
          "shape",
          [](const AbelianMPSSite& self) {
            return py::make_tuple(self.left_bond_dimension(),
                                  self.physical_dimension(),
                                  self.right_bond_dimension());
          },
          "Dense tensor shape (chi_left, physical_dimension, chi_right).")
      .def("to_dense", &site_to_dense,
           "Return an array with shape (chi_left, physical_dimension, "
           "chi_right), inserting zeros for absent symmetry blocks.");

  py::class_<MPSContainer, WavefunctionContainer, py::smart_holder>(
      data, "MPSContainer")
      .def_property_readonly("orbitals", &MPSContainer::get_orbitals)
      .def_property_readonly("total_num_particles",
                             &MPSContainer::total_num_particles)
      .def_property_readonly("active_num_particles",
                             &MPSContainer::active_num_particles)
      .def_property_readonly("orthogonality_center",
                             &MPSContainer::orthogonality_center)
      .def_property_readonly("physical_basis", &MPSContainer::physical_basis)
      .def_property_readonly("site_to_orbital_order",
                             &MPSContainer::site_to_orbital_order)
      .def_property_readonly("num_sites", &MPSContainer::num_sites)
      .def_property_readonly("is_complex", &MPSContainer::is_complex);

  py::class_<AbelianMPSContainer, MPSContainer, py::smart_holder>(
      data, "AbelianMPSContainer",
      "Immutable MPS whose left and right bond spaces are partitioned by "
      "particle number.")
      .def(py::init<std::vector<AbelianMPSContainer::SitePtr>,
                    std::shared_ptr<Orbitals>,
                    std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>,
                    std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>,
                    std::optional<std::size_t>, std::vector<Configuration>,
                    std::vector<std::size_t>>(),
           py::arg("sites"), py::arg("orbitals"),
           py::arg("total_num_particles") = nullptr,
           py::arg("active_num_particles") = nullptr,
           py::arg("orthogonality_center") = std::size_t{0},
           py::arg("physical_basis") = std::vector<Configuration>{},
           py::arg("site_to_orbital_order") = std::vector<std::size_t>{},
           "Construct from particle-number-blocked sites in chain order. "
           "Adjacent sites must describe the same shared bond space.")
      .def_property_readonly("sites", &AbelianMPSContainer::sites,
                             "Immutable MPS sites in chain order.")
      .def_property_readonly("max_bond_dimension",
                             &AbelianMPSContainer::max_bond_dimension,
                             "Largest total left or right bond dimension.")
      .def_property_readonly(
          "physical_dimension",
          [](const AbelianMPSContainer& self) {
            return self.sites().front()->physical_dimension();
          },
          "Number of physical states per site, uniform across the chain.");
}
