/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

#include <algorithm>
#include <limits>
#include <numeric>
#include <qdk/chemistry/data/wavefunction_containers/mps_wavefunction.hpp>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

namespace qdk::chemistry::data {
namespace {

template <typename Extents>
std::size_t total_extent(const Extents& extents) {
  std::size_t total = 0;
  for (const auto& [label, extent] : extents) {
    total += extent;
  }
  return total;
}

template <typename Extents>
std::unordered_map<SymmetryLabel, std::size_t> sector_offsets(
    const Extents& extents, const std::vector<SymmetryLabel>& order) {
  if (order.size() != extents.size()) {
    throw std::invalid_argument(
        "MPS bond-sector order must contain every sector exactly once.");
  }
  std::unordered_map<SymmetryLabel, std::size_t> offsets;
  std::size_t offset = 0;
  for (const auto& label : order) {
    const auto extent = extents.find(label);
    if (extent == extents.end() || !offsets.emplace(label, offset).second) {
      throw std::invalid_argument(
          "MPS bond-sector order contains a missing or duplicate sector.");
    }
    offset += extent->second;
  }
  return offsets;
}

void validate_bond_symmetry(const SymmetryProduct& symmetry) {
  if (symmetry.has_axis(AxisName::Spin) &&
      !symmetry.has_axis(AxisName::ParticleNumber)) {
    throw std::invalid_argument(
        "Spin-resolved MPS bond sectors must also carry particle number.");
  }
}

std::vector<std::size_t> normalized_site_to_orbital_order(
    std::vector<std::size_t> order, std::size_t site_count) {
  if (order.empty()) {
    order.resize(site_count);
    std::iota(order.begin(), order.end(), std::size_t{});
  }
  return order;
}

template <typename Matrix>
bool is_canonical_site(const Matrix& packed, std::size_t physical_dimension,
                       bool left_normalized) {
  using Scalar = typename Matrix::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  const auto left_dimension =
      packed.rows() / static_cast<Eigen::Index>(physical_dimension);
  const auto right_dimension = packed.cols();
  const auto scale = static_cast<RealScalar>(
      std::max({left_dimension, right_dimension,
                static_cast<Eigen::Index>(physical_dimension)}));
  const auto tolerance =
      RealScalar{100} * std::numeric_limits<RealScalar>::epsilon() * scale;

  if (left_normalized) {
    const auto gram = packed.adjoint() * packed;
    return gram.isApprox(Matrix::Identity(right_dimension, right_dimension),
                         tolerance);
  }

  Matrix gram = Matrix::Zero(left_dimension, left_dimension);
  for (std::size_t physical = 0; physical < physical_dimension; ++physical) {
    Matrix slice(left_dimension, right_dimension);
    for (Eigen::Index left = 0; left < left_dimension; ++left) {
      slice.row(left) =
          packed.row(left * static_cast<Eigen::Index>(physical_dimension) +
                     static_cast<Eigen::Index>(physical));
    }
    gram.noalias() += slice * slice.adjoint();
  }
  return gram.isApprox(Matrix::Identity(left_dimension, left_dimension),
                       tolerance);
}

}  // namespace

MPSSite::MPSSite(std::vector<PhysicalSlicePtr> physical_slices,
                 std::vector<SymmetryLabel> left_sector_order,
                 std::vector<SymmetryLabel> right_sector_order)
    : _physical_slices(std::move(physical_slices)),
      _left_sector_order(std::move(left_sector_order)),
      _right_sector_order(std::move(right_sector_order)) {
  _validate();
}

void MPSSite::_validate() const {
  if (_physical_slices.empty()) {
    throw std::invalid_argument(
        "MPS site must contain at least one physical slice.");
  }
  if (!_physical_slices.front()) {
    throw std::invalid_argument(
        "MPS physical slice pointers must not be null.");
  }

  const auto scalar_index = _physical_slices.front()->index();
  std::visit(
      [&](const auto& reference) {
        validate_bond_symmetry(*reference.symmetries()[0]);
        validate_bond_symmetry(*reference.symmetries()[1]);
        sector_offsets(reference.extents()[0], _left_sector_order);
        sector_offsets(reference.extents()[1], _right_sector_order);
        for (const auto& slice : _physical_slices) {
          if (!slice) {
            throw std::invalid_argument(
                "MPS physical slice pointers must not be null.");
          }
          if (slice->index() != scalar_index) {
            throw std::invalid_argument(
                "MPS physical slices must use one scalar type.");
          }
          std::visit(
              [&](const auto& candidate) {
                using Reference = std::decay_t<decltype(reference)>;
                using Candidate = std::decay_t<decltype(candidate)>;
                if constexpr (std::is_same_v<Reference, Candidate>) {
                  if (*candidate.symmetries()[0] !=
                          *reference.symmetries()[0] ||
                      *candidate.symmetries()[1] !=
                          *reference.symmetries()[1] ||
                      candidate.extents() != reference.extents()) {
                    throw std::invalid_argument(
                        "MPS physical slices must share their bond spaces.");
                  }
                }
              },
              *slice);
        }
      },
      *_physical_slices.front());
  if (left_bond_dimension() == 0 || right_bond_dimension() == 0) {
    throw std::invalid_argument("MPS bond dimensions must be positive.");
  }
}

std::size_t MPSSite::left_bond_dimension() const {
  return std::visit(
      [](const auto& slice) { return total_extent(slice.extents()[0]); },
      *_physical_slices.front());
}

std::size_t MPSSite::right_bond_dimension() const {
  return std::visit(
      [](const auto& slice) { return total_extent(slice.extents()[1]); },
      *_physical_slices.front());
}

bool MPSSite::is_complex() const {
  return _physical_slices.front()->index() == 1;
}

MPSSite::DenseMatrixVariant MPSSite::to_dense() const {
  return std::visit(
      [&](const auto& first_slice) -> DenseMatrixVariant {
        using Slice = std::decay_t<decltype(first_slice)>;
        using Scalar = typename Slice::BlockPtr::element_type::Scalar;
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> dense =
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(
                static_cast<Eigen::Index>(left_bond_dimension() *
                                          physical_dimension()),
                static_cast<Eigen::Index>(right_bond_dimension()));
        const auto left_offsets =
            sector_offsets(first_slice.extents()[0], _left_sector_order);
        const auto right_offsets =
            sector_offsets(first_slice.extents()[1], _right_sector_order);

        for (std::size_t physical = 0; physical < physical_dimension();
             ++physical) {
          const auto& slice = std::get<Slice>(*_physical_slices[physical]);
          for (const auto& [labels, block] : slice.blocks()) {
            const auto left_offset = left_offsets.at(labels[0]);
            const auto right_offset = right_offsets.at(labels[1]);
            for (Eigen::Index left = 0; left < block->rows(); ++left) {
              const auto packed_row = static_cast<Eigen::Index>(
                  (left_offset + static_cast<std::size_t>(left)) *
                      physical_dimension() +
                  physical);
              dense.block(packed_row, static_cast<Eigen::Index>(right_offset),
                          1, block->cols()) = block->row(left);
            }
          }
        }
        return dense;
      },
      *_physical_slices.front());
}

MPSContainer::MPSContainer(
    std::shared_ptr<Orbitals> orbitals,
    std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
        total_num_particles,
    std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
        active_num_particles,
    std::optional<std::size_t> orthogonality_center,
    std::vector<Configuration> physical_basis,
    std::vector<std::size_t> site_to_orbital_order)
    : _orbitals(std::move(orbitals)),
      _total_num_particles(std::move(total_num_particles)),
      _active_num_particles(std::move(active_num_particles)),
      _orthogonality_center(orthogonality_center),
      _physical_basis(std::move(physical_basis)),
      _site_to_orbital_order(std::move(site_to_orbital_order)) {}

void MPSContainer::_validate_common(std::size_t site_count,
                                    std::size_t physical_dimension) const {
  if (site_count == 0) {
    throw std::invalid_argument(
        "MPS wavefunction must contain at least one site.");
  }
  if (!_orbitals) {
    throw std::invalid_argument("MPS wavefunction requires orbitals.");
  }
  if (_orthogonality_center && *_orthogonality_center >= site_count) {
    throw std::invalid_argument(
        "MPS orthogonality center must be a valid site index.");
  }
  if (!_physical_basis.empty() &&
      _physical_basis.size() != physical_dimension) {
    throw std::invalid_argument(
        "MPS physical basis size must match the number of physical slices.");
  }
  if (std::any_of(_physical_basis.begin(), _physical_basis.end(),
                  [](const Configuration& state) {
                    return state.bits_per_mode() != 2;
                  })) {
    throw std::invalid_argument(
        "MPS physical basis states must be one-orbital spin-half "
        "configurations.");
  }
  if (_site_to_orbital_order.size() != site_count) {
    throw std::invalid_argument(
        "MPS site-to-orbital order size must match the number of sites.");
  }
  auto sorted_order = _site_to_orbital_order;
  std::ranges::sort(sorted_order);
  if (std::ranges::adjacent_find(sorted_order) != sorted_order.end() ||
      sorted_order.back() >= _orbitals->get_num_molecular_orbitals()) {
    throw std::invalid_argument(
        "MPS site-to-orbital order must contain unique molecular-orbital "
        "indices.");
  }
}

AbelianMPSContainer::AbelianMPSContainer(
    std::vector<SitePtr> sites, std::shared_ptr<Orbitals> orbitals,
    std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
        total_num_particles,
    std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
        active_num_particles,
    std::optional<std::size_t> orthogonality_center,
    std::vector<Configuration> physical_basis,
    std::vector<std::size_t> site_to_orbital_order)
    : MPSContainer(std::move(orbitals), std::move(total_num_particles),
                   std::move(active_num_particles), orthogonality_center,
                   std::move(physical_basis),
                   normalized_site_to_orbital_order(
                       std::move(site_to_orbital_order), sites.size())),
      _sites(std::move(sites)) {
  _validate();
}

void AbelianMPSContainer::_validate() const {
  const auto physical_dimension = _sites.empty() || !_sites.front()
                                      ? 0
                                      : _sites.front()->physical_dimension();
  _validate_common(_sites.size(), physical_dimension);

  const auto is_complex = _sites.front() && _sites.front()->is_complex();
  for (std::size_t index = 0; index < _sites.size(); ++index) {
    if (!_sites[index]) {
      throw std::invalid_argument("MPS site pointers must not be null.");
    }
    if (_sites[index]->physical_dimension() != physical_dimension) {
      throw std::invalid_argument("MPS sites must use one physical dimension.");
    }
    if (_sites[index]->is_complex() != is_complex) {
      throw std::invalid_argument("MPS sites must use one scalar type.");
    }
    if (orthogonality_center() && index != *orthogonality_center()) {
      const auto left_normalized = index < *orthogonality_center();
      const auto canonical = std::visit(
          [&](const auto& dense) {
            return is_canonical_site(dense, physical_dimension,
                                     left_normalized);
          },
          _sites[index]->to_dense());
      if (!canonical) {
        throw std::invalid_argument(left_normalized
                                        ? "MPS site is not left-normalized."
                                        : "MPS site is not right-normalized.");
      }
    }
  }
  for (std::size_t index = 0; index + 1 < _sites.size(); ++index) {
    const auto& left_slice = *_sites[index]->physical_slices().front();
    const auto& right_slice = *_sites[index + 1]->physical_slices().front();
    std::visit(
        [&](const auto& left, const auto& right) {
          using Left = std::decay_t<decltype(left)>;
          using Right = std::decay_t<decltype(right)>;
          if constexpr (std::is_same_v<Left, Right>) {
            if (*left.symmetries()[1] != *right.symmetries()[0] ||
                left.extents()[1] != right.extents()[0] ||
                _sites[index]->right_sector_order() !=
                    _sites[index + 1]->left_sector_order()) {
              throw std::invalid_argument(
                  "Adjacent MPS sites have incompatible bond spaces.");
            }
          }
        },
        left_slice, right_slice);
  }
}

std::size_t AbelianMPSContainer::max_bond_dimension() const {
  std::size_t maximum = 0;
  for (const auto& site : _sites) {
    maximum = std::max(maximum, site->left_bond_dimension());
    maximum = std::max(maximum, site->right_bond_dimension());
  }
  return maximum;
}

bool AbelianMPSContainer::is_complex() const {
  return _sites.front()->is_complex();
}

std::unique_ptr<WavefunctionContainer> AbelianMPSContainer::clone() const {
  return std::make_unique<AbelianMPSContainer>(*this);
}

MPSContainer::ScalarVariant MPSContainer::overlap(
    const WavefunctionContainer&) const {
  throw std::runtime_error(
      "overlap() is not implemented for MPS wavefunctions.");
}

double MPSContainer::norm() const {
  throw std::runtime_error("norm() is not implemented for MPS wavefunctions.");
}

std::shared_ptr<const SymmetryBlockedTensor<1>>
MPSContainer::total_orbital_occupations() const {
  throw std::runtime_error(
      "Orbital occupations are not implemented for MPS wavefunctions.");
}

std::shared_ptr<const SymmetryBlockedTensor<1>>
MPSContainer::active_orbital_occupations() const {
  throw std::runtime_error(
      "Orbital occupations are not implemented for MPS wavefunctions.");
}

void MPSContainer::clear_caches() const {}

nlohmann::json MPSContainer::to_json() const {
  throw std::runtime_error(
      "to_json() is not implemented for MPS wavefunctions.");
}

std::string MPSContainer::get_container_type() const { return "mps"; }

std::vector<std::string> MPSContainer::sectors() const {
  return {Wavefunction::DEFAULT_SECTOR};
}

std::shared_ptr<const Orbitals> MPSContainer::sector_basis(
    const std::string& name) const {
  if (name != Wavefunction::DEFAULT_SECTOR) {
    throw std::out_of_range("Unknown MPS wavefunction sector: " + name);
  }
  return _orbitals;
}

}  // namespace qdk::chemistry::data
