// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <qdk/chemistry/data/wavefunction_containers/mps_wavefunction.hpp>
#include <stdexcept>

namespace qdk::chemistry::data {

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
                    return state.bits_per_mode() != 2 || state.capacity() != 1;
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

void MPSContainer::hash_update(
    qdk::chemistry::utils::HashContext& ctx) const {
  WavefunctionContainer::hash_update(ctx);
  hash_value(ctx, get_container_type());
  hash_value(ctx, _orbitals->content_hash());
  hash_value(ctx, _total_num_particles);
  hash_value(ctx, _active_num_particles);
  hash_value(ctx, _orthogonality_center);
  hash_value(ctx, _physical_basis);
  hash_value(ctx, _site_to_orbital_order);
}

}  // namespace qdk::chemistry::data
