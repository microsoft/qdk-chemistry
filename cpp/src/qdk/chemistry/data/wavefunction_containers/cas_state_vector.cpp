// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <bitset>
#include <numeric>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/wavefunction_containers/cas_state_vector.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>

namespace qdk::chemistry::data {
using ScalarVariant = ContainerTypes::ScalarVariant;

namespace {

// Number of k-element subsets of n elements
size_t binomial(size_t n, size_t k) {
  if (k > n) return 0;
  k = std::min(k, n - k);
  size_t result = 1;
  for (size_t i = 1; i <= k; ++i) result = result * (n - k + i) / i;
  return result;
}

// All determinants of n_alpha and n_beta electrons in n <= N / 2 orbitals,
// alpha-major, each spin's strings in lexicographic order of their sorted
// occupied orbitals
template <size_t N>
ContainerTypes::DeterminantVector complete_space_determinants(size_t n,
                                                              size_t n_alpha,
                                                              size_t n_beta) {
  auto strings = [n](size_t k, size_t offset) {
    std::vector<std::bitset<N>> result;
    std::vector<size_t> occupied(k);
    std::iota(occupied.begin(), occupied.end(), size_t{0});
    while (true) {
      std::bitset<N> bits;
      for (size_t p : occupied) bits.set(offset + p);
      result.push_back(bits);
      size_t i = k;
      while (i > 0 && occupied[i - 1] == n - k + i - 1) --i;
      if (i == 0) return result;
      ++occupied[i - 1];
      for (size_t j = i; j < k; ++j) occupied[j] = occupied[j - 1] + 1;
    }
  };
  const auto alpha = strings(n_alpha, 0);
  const auto beta = strings(n_beta, N / 2);

  ContainerTypes::DeterminantVector dets;
  dets.reserve(alpha.size() * beta.size());
  for (const auto& a : alpha) {
    for (const auto& b : beta) {
      dets.push_back(Configuration::from_spin_half_bitset(a | b, n));
    }
  }
  return dets;
}

}  // namespace

CasStateVectorContainer::CasStateVectorContainer(
    VectorVariant coeffs, size_t n_alpha, size_t n_beta,
    std::shared_ptr<Orbitals> orbitals, const std::string& sector,
    WavefunctionType type)
    : CasStateVectorContainer(std::move(coeffs), n_alpha, n_beta, orbitals,
                              std::nullopt,  // one_rdm_spin_traced
                              std::nullopt,  // one_rdm_aa
                              std::nullopt,  // one_rdm_bb
                              std::nullopt,  // two_rdm_spin_traced
                              std::nullopt,  // two_rdm_aaaa
                              std::nullopt,  // two_rdm_aabb
                              std::nullopt,  // two_rdm_bbbb
                              sector, OrbitalEntropies{}, type) {
  QDK_LOG_TRACE_ENTERING();
}

CasStateVectorContainer::CasStateVectorContainer(
    VectorVariant coeffs, size_t n_alpha, size_t n_beta,
    std::shared_ptr<Orbitals> orbitals,
    const std::optional<MatrixVariant>& one_rdm_spin_traced,
    const std::optional<MatrixVariant>& one_rdm_aa,
    const std::optional<MatrixVariant>& one_rdm_bb,
    const std::optional<VectorVariant>& two_rdm_spin_traced,
    const std::optional<VectorVariant>& two_rdm_aaaa,
    const std::optional<VectorVariant>& two_rdm_aabb,
    const std::optional<VectorVariant>& two_rdm_bbbb, const std::string& sector,
    const OrbitalEntropies& entropies, WavefunctionType type)
    : StateVectorContainer(std::move(coeffs), orbitals, one_rdm_spin_traced,
                           one_rdm_aa, one_rdm_bb, two_rdm_spin_traced,
                           two_rdm_aaaa, two_rdm_aabb, two_rdm_bbbb, sector,
                           entropies, type),
      _num_orbitals([&] {
        const auto active =
            spin_channel_indices(orbitals->active_indices(), axes::alpha());
        return active.empty() ? orbitals->get_num_molecular_orbitals()
                              : active.size();
      }()),
      _n_alpha(n_alpha),
      _n_beta(n_beta) {
  QDK_LOG_TRACE_ENTERING();
  // Determinants pair the alpha and beta active orbitals position by position
  const auto beta_active =
      spin_channel_indices(orbitals->active_indices(), axes::beta());
  if (!beta_active.empty() && beta_active.size() != _num_orbitals) {
    throw std::invalid_argument(
        "CasStateVectorContainer: the alpha and beta active spaces differ in "
        "size (" +
        std::to_string(_num_orbitals) + " and " +
        std::to_string(beta_active.size()) + " orbitals).");
  }
  if (n_alpha > _num_orbitals || n_beta > _num_orbitals) {
    throw std::invalid_argument(
        "CasStateVectorContainer: " + std::to_string(n_alpha) + " alpha and " +
        std::to_string(n_beta) + " beta electrons do not fit in " +
        std::to_string(_num_orbitals) + " active orbitals.");
  }
  const size_t num_determinants =
      binomial(_num_orbitals, n_alpha) * binomial(_num_orbitals, n_beta);
  if (size() != num_determinants) {
    throw std::invalid_argument(
        "CasStateVectorContainer: coefficient vector size (" +
        std::to_string(size()) +
        ") does not match the number of determinants (" +
        std::to_string(num_determinants) + ").");
  }
}

std::unique_ptr<WavefunctionContainer> CasStateVectorContainer::clone() const {
  QDK_LOG_TRACE_ENTERING();
  return std::make_unique<CasStateVectorContainer>(*this);
}

std::optional<size_t> CasStateVectorContainer::_index_of(
    const Configuration& det) const {
  // Equal to a stored determinant only with its encoding and capacity (2 bits
  // per mode, 4 modes per byte, as from_spin_half_bitset builds them)
  if (det.bits_per_mode() != 2 ||
      det.capacity() != 4 * ((_num_orbitals + 3) / 4)) {
    return std::nullopt;
  }
  // Lexicographic rank of a string o_0 < ... < o_{k-1} of n orbitals:
  // C(n, k) - 1 - sum_i C(n - 1 - o_i, k - i)
  size_t rank_alpha = binomial(_num_orbitals, _n_alpha) - 1;
  size_t rank_beta = binomial(_num_orbitals, _n_beta) - 1;
  size_t k_alpha = 0, k_beta = 0;
  for (size_t p = 0; p < det.capacity(); ++p) {
    const uint8_t state = det.get_mode_state(p);
    if (state == 0) continue;
    if (p >= _num_orbitals) return std::nullopt;
    if (state & 1) {
      if (k_alpha == _n_alpha) return std::nullopt;
      rank_alpha -= binomial(_num_orbitals - 1 - p, _n_alpha - k_alpha++);
    }
    if (state & 2) {
      if (k_beta == _n_beta) return std::nullopt;
      rank_beta -= binomial(_num_orbitals - 1 - p, _n_beta - k_beta++);
    }
  }
  if (k_alpha != _n_alpha || k_beta != _n_beta) return std::nullopt;
  return rank_alpha * binomial(_num_orbitals, _n_beta) + rank_beta;
}

ScalarVariant CasStateVectorContainer::get_coefficient(
    const Configuration& det) const {
  QDK_LOG_TRACE_ENTERING();
  const auto index = _index_of(det);
  return std::visit(
      [&](const auto& coeffs) -> ScalarVariant {
        using Scalar = typename std::decay_t<decltype(coeffs)>::Scalar;
        return index ? coeffs[*index] : Scalar(0);
      },
      get_coefficients());
}

bool CasStateVectorContainer::contains_determinant(
    const Configuration& det) const {
  QDK_LOG_TRACE_ENTERING();
  return _index_of(det).has_value();
}

const CasStateVectorContainer::DeterminantVector&
CasStateVectorContainer::get_active_determinants() const {
  QDK_LOG_TRACE_ENTERING();
  return get_configuration_set().get_configurations();
}

const ConfigurationSet& CasStateVectorContainer::get_configuration_set() const {
  QDK_LOG_TRACE_ENTERING();
  if (!_determinants) {
    DeterminantVector dets;
    if (_num_orbitals <= 32) {
      dets = complete_space_determinants<64>(_num_orbitals, _n_alpha, _n_beta);
    } else if (_num_orbitals <= 64) {
      dets = complete_space_determinants<128>(_num_orbitals, _n_alpha, _n_beta);
    } else if (_num_orbitals <= 128) {
      dets = complete_space_determinants<256>(_num_orbitals, _n_alpha, _n_beta);
    } else if (_num_orbitals <= 2048) {
      dets =
          complete_space_determinants<4096>(_num_orbitals, _n_alpha, _n_beta);
    } else {
      throw std::runtime_error(
          "CasStateVectorContainer: determinants of more than 2048 active "
          "orbitals are not supported.");
    }
    _determinants = std::make_shared<const ConfigurationSet>(
        std::move(dets), get_orbitals(), sectors().front());
  }
  return *_determinants;
}

size_t CasStateVectorContainer::size() const {
  QDK_LOG_TRACE_ENTERING();
  return std::visit(
      [](const auto& coeffs) { return static_cast<size_t>(coeffs.size()); },
      get_coefficients());
}

ScalarVariant CasStateVectorContainer::overlap(
    const WavefunctionContainer& other) const {
  QDK_LOG_TRACE_ENTERING();
  const auto* other_cas = dynamic_cast<const CasStateVectorContainer*>(&other);
  if (!other_cas || other_cas->_num_orbitals != _num_orbitals ||
      other_cas->_n_alpha != _n_alpha || other_cas->_n_beta != _n_beta ||
      other_cas->get_orbitals() != get_orbitals()) {
    return StateVectorContainer::overlap(other);
  }
  // Same determinants in the same order: the overlap is the coefficient dot
  // product, conjugating the bra
  return std::visit(
      [](const auto& bra, const auto& ket) -> ScalarVariant {
        using Bra = std::decay_t<decltype(bra)>;
        using Ket = std::decay_t<decltype(ket)>;
        if constexpr (std::is_same_v<Bra, Eigen::VectorXd> &&
                      std::is_same_v<Ket, Eigen::VectorXd>) {
          return bra.dot(ket);
        } else {
          return bra.template cast<std::complex<double>>().dot(
              ket.template cast<std::complex<double>>());
        }
      },
      get_coefficients(), other.get_coefficients());
}

std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
CasStateVectorContainer::total_num_particles() const {
  QDK_LOG_TRACE_ENTERING();
  const auto inactive = get_orbitals()->inactive_indices();
  return _make_particle_count(
      _n_alpha + spin_channel_indices(inactive, axes::alpha()).size(),
      _n_beta + spin_channel_indices(inactive, axes::beta()).size());
}

std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
CasStateVectorContainer::active_num_particles() const {
  QDK_LOG_TRACE_ENTERING();
  return _make_particle_count(_n_alpha, _n_beta);
}

void CasStateVectorContainer::clear_caches() const {
  QDK_LOG_TRACE_ENTERING();
  StateVectorContainer::clear_caches();
  _determinants.reset();
}

}  // namespace qdk::chemistry::data
