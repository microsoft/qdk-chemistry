// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <complex>
#include <memory>
#include <numeric>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <qdk/chemistry/data/wavefunction_containers/mps_wavefunction.hpp>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace py = pybind11;
using namespace qdk::chemistry::data;

namespace {

template <typename Scalar>
MPSSite::PhysicalSlicePtr make_trivial_slice(
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
  return std::make_shared<const MPSSite::PhysicalSlice>(std::move(slice));
}

template <typename Scalar>
std::shared_ptr<MPSSite> site_from_slices(
    std::vector<std::shared_ptr<const SymmetryBlockedTensor<2, Scalar>>>
        physical_slices,
    std::vector<SymmetryLabel> left_sector_order,
    std::vector<SymmetryLabel> right_sector_order) {
  std::vector<MPSSite::PhysicalSlicePtr> variants;
  variants.reserve(physical_slices.size());
  for (auto& slice : physical_slices) {
    if (!slice) {
      throw std::invalid_argument(
          "MPS physical slice pointers must not be null.");
    }
    variants.push_back(std::make_shared<const MPSSite::PhysicalSlice>(*slice));
  }
  return std::make_shared<MPSSite>(std::move(variants),
                                   std::move(left_sector_order),
                                   std::move(right_sector_order));
}

template <typename Scalar>
std::shared_ptr<MPSSite> site_from_dense(
    py::array_t<Scalar, py::array::c_style | py::array::forcecast> tensor) {
  const auto values = tensor.template unchecked<3>();
  std::vector<MPSSite::PhysicalSlicePtr> slices;
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
  return std::make_shared<MPSSite>(std::move(slices),
                                   std::vector<SymmetryLabel>{SymmetryLabel{}},
                                   std::vector<SymmetryLabel>{SymmetryLabel{}});
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

py::object site_to_dense(const MPSSite& site) {
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

py::list physical_slices(const MPSSite& site) {
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
  py::enum_<MPSCanonicalForm>(data, "MPSCanonicalForm")
      .value("Unspecified", MPSCanonicalForm::Unspecified)
      .value("LeftNormalized", MPSCanonicalForm::LeftNormalized)
      .value("RightNormalized", MPSCanonicalForm::RightNormalized)
      .value("Mixed", MPSCanonicalForm::Mixed);

  py::class_<MPSSite, py::smart_holder>(data, "MPSSite")
      .def(py::init(&site_from_slices<double>), py::arg("physical_slices"),
           py::arg("left_sector_order"), py::arg("right_sector_order"))
      .def(py::init(&site_from_slices<std::complex<double>>),
           py::arg("physical_slices"), py::arg("left_sector_order"),
           py::arg("right_sector_order"))
      .def_static("from_dense", &site_from_dense<double>, py::arg("tensor"))
      .def_static("from_dense_complex", &site_from_dense<std::complex<double>>,
                  py::arg("tensor"))
      .def_property_readonly("physical_slices", &physical_slices)
      .def_property_readonly("left_sector_order", &MPSSite::left_sector_order)
      .def_property_readonly("right_sector_order", &MPSSite::right_sector_order)
      .def_property_readonly("physical_dimension", &MPSSite::physical_dimension)
      .def_property_readonly("left_bond_dimension",
                             &MPSSite::left_bond_dimension)
      .def_property_readonly("right_bond_dimension",
                             &MPSSite::right_bond_dimension)
      .def_property_readonly("is_complex", &MPSSite::is_complex)
      .def_property_readonly("shape",
                             [](const MPSSite& self) {
                               return py::make_tuple(
                                   self.left_bond_dimension(),
                                   self.physical_dimension(),
                                   self.right_bond_dimension());
                             })
      .def("to_dense", &site_to_dense);

  py::class_<MPSContainer, WavefunctionContainer, py::smart_holder>(
      data, "MPSContainer")
      .def_property_readonly("orbitals", &MPSContainer::orbitals)
      .def_property_readonly("total_num_particles",
                             &MPSContainer::total_num_particles)
      .def_property_readonly("active_num_particles",
                             &MPSContainer::active_num_particles)
      .def_property_readonly("canonical_form", &MPSContainer::canonical_form)
      .def_property_readonly("canonical_center",
                             &MPSContainer::canonical_center)
      .def_property_readonly("discarded_weight",
                             &MPSContainer::discarded_weight)
      .def_property_readonly("physical_basis", &MPSContainer::physical_basis)
      .def_property_readonly("num_sites", &MPSContainer::num_sites)
      .def_property_readonly("is_complex", &MPSContainer::is_complex);

  py::class_<AbelianMPSContainer, MPSContainer, py::smart_holder>(
      data, "AbelianMPSContainer")
      .def(py::init<std::vector<AbelianMPSContainer::SitePtr>,
                    std::shared_ptr<Orbitals>,
                    std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>,
                    std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>,
                    MPSCanonicalForm, std::optional<std::size_t>, double,
                    std::vector<std::string>>(),
           py::arg("sites"), py::arg("orbitals"),
           py::arg("total_num_particles") = nullptr,
           py::arg("active_num_particles") = nullptr,
           py::arg("canonical_form") = MPSCanonicalForm::Unspecified,
           py::arg("canonical_center") = std::nullopt,
           py::arg("discarded_weight") = 0.0,
           py::arg("physical_basis") = std::vector<std::string>{})
      .def_property_readonly("sites", &AbelianMPSContainer::sites)
      .def_property_readonly("max_bond_dimension",
                             &AbelianMPSContainer::max_bond_dimension)
      .def_property_readonly(
          "physical_dimension", [](const AbelianMPSContainer& self) {
            return self.sites().front()->physical_dimension();
          });
}
