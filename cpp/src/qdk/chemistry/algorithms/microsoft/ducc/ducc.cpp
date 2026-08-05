// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "ducc.hpp"

#include <btas/btas.h>
#include <btas/tensor.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_tensor.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/amplitude_container.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft {

namespace {

using BTensor = btas::Tensor<double>;
using IndexSet = data::SymmetryBlockedIndexSet;
using SBT2 = data::SymmetryBlockedTensor<2>;
using SBT4 = data::SymmetryBlockedTensor<4>;

template <typename Variant>
const Eigen::VectorXd& real_vector(const Variant& value) {
  return std::get<Eigen::VectorXd>(value);
}

std::size_t index4(std::size_t n, std::size_t p, std::size_t q, std::size_t r,
                   std::size_t s) {
  return ((p * n + q) * n + r) * n + s;
}

std::shared_ptr<data::Orbitals> output_orbitals(
    const data::Orbitals& input, std::shared_ptr<const IndexSet> p_space) {
  const std::optional<Eigen::MatrixXd> overlap =
      input.has_overlap_matrix()
          ? std::optional<Eigen::MatrixXd>(input.get_overlap_matrix())
          : std::nullopt;
  const auto energies = input.has_energies() ? input.energies() : nullptr;
  const auto basis = input.has_basis_set() ? input.get_basis_set() : nullptr;

  if (input.is_unrestricted()) {
    return std::make_shared<data::Orbitals>(input.coefficients(), energies,
                                            overlap, basis, std::move(p_space),
                                            nullptr);
  }

  const auto& coefficients =
      input.coefficients()->block({data::axes::alpha(), data::axes::alpha()});
  const std::optional<Eigen::VectorXd> energy =
      energies ? std::optional<Eigen::VectorXd>(
                     energies->block({data::axes::alpha()}))
               : std::nullopt;
  return std::make_shared<data::Orbitals>(coefficients, coefficients, energy,
                                          energy, overlap, basis,
                                          std::move(p_space), nullptr);
}

class TensorProvider {
 public:
  TensorProvider(
      const data::CanonicalFourCenterHamiltonianContainer& hamiltonian,
      const data::Wavefunction& wavefunction,
      std::shared_ptr<const IndexSet> p_space)
      : _hamiltonian(hamiltonian),
        _amplitudes(wavefunction.get_container<data::AmplitudeContainer>()),
        _restricted(hamiltonian.is_restricted()),
        _p_space(std::move(p_space)) {
    const auto [nocc_a, nocc_b] = wavefunction.get_total_num_electrons();
    _nocc_a = nocc_a;
    _nocc_b = nocc_b;
    const auto& h_a = _hamiltonian.one_body_integrals().block(
        {data::axes::alpha(), data::axes::alpha()});
    _nmo = static_cast<std::size_t>(h_a.rows());
    _nvir_a = _nmo - _nocc_a;
    _nvir_b = _nmo - _nocc_b;
    const auto alpha = p_space_indices(/*beta=*/false);
    const auto beta = p_space_indices(/*beta=*/true);
    _nao_a = static_cast<std::size_t>(
        std::count_if(alpha.begin(), alpha.end(),
                      [this](std::uint32_t p) { return p < _nocc_a; }));
    _nao_b = static_cast<std::size_t>(
        std::count_if(beta.begin(), beta.end(),
                      [this](std::uint32_t p) { return p < _nocc_b; }));
    _nav_a = alpha.size() - _nao_a;
    _nav_b = beta.size() - _nao_b;
    _nact = alpha.size();
    if (beta.size() != _nact)
      throw std::runtime_error(
          "ducc: alpha and beta P-spaces must have equal size");
  }

  BTensor get(const std::string& label, const std::string& tags,
              const std::string& mask) {
    const std::string key = label + "_" + tags + "#" + mask;
    if (auto it = _store.find(key); it != _store.end()) return it->second;
    auto tensor = make(label, tags, mask);
    _store.emplace(key, tensor);
    return tensor;
  }

  void put(const std::string& label, const std::string& tags,
           const std::string& mask, const BTensor& value) {
    _store[label + "_" + tags + "#" + mask] = value;
  }

  double get_scalar(const std::string& label) { return _scalars[label]; }
  void put_scalar(const std::string& label, double value) {
    _scalars[label] = value;
  }

  std::size_t nmo() const { return _nmo; }
  std::size_t nocc_a() const { return _nocc_a; }
  std::size_t nocc_b() const { return _nocc_b; }
  std::size_t nvir_a() const { return _nvir_a; }
  std::size_t nvir_b() const { return _nvir_b; }
  std::size_t nact() const { return _nact; }
  std::size_t nao_a() const { return _nao_a; }
  std::size_t nao_b() const { return _nao_b; }
  std::size_t nav_a() const { return _nav_a; }
  std::size_t nav_b() const { return _nav_b; }

  BTensor raw_one_body(bool beta) const {
    const auto spin = beta ? data::axes::beta() : data::axes::alpha();
    const auto& source = _hamiltonian.one_body_integrals().block({spin, spin});
    BTensor result{btas::Range{_nmo, _nmo}};
    for (std::size_t p = 0; p < _nmo; ++p)
      for (std::size_t q = 0; q < _nmo; ++q) result(p, q) = source(p, q);
    return result;
  }

  BTensor raw_two_body(int channel) const {
    const auto& two_body = _hamiltonian.two_body_integrals();
    const Eigen::VectorXd* source = nullptr;
    if (channel == 0)
      source = &two_body.block({data::axes::alpha(), data::axes::alpha(),
                                data::axes::alpha(), data::axes::alpha()});
    else if (channel == 1)
      source = &two_body.block({data::axes::alpha(), data::axes::alpha(),
                                data::axes::beta(), data::axes::beta()});
    else if (channel == 2)
      source = &two_body.block({data::axes::beta(), data::axes::beta(),
                                data::axes::beta(), data::axes::beta()});
    else
      throw std::invalid_argument("ducc: invalid two-body spin channel");

    BTensor result{btas::Range{_nmo, _nmo, _nmo, _nmo}};
    std::copy(source->begin(), source->end(), result.begin());
    return result;
  }

  BTensor occupied_projector(bool beta) const {
    BTensor result{btas::Range{_nmo, _nmo}};
    result.fill(0.0);
    const std::size_t nocc = beta ? _nocc_b : _nocc_a;
    for (std::size_t i = 0; i < nocc; ++i) result(i, i) = 1.0;
    return result;
  }

  BTensor active_occupied_projector(bool beta) const {
    BTensor result{btas::Range{_nact, _nact}};
    result.fill(0.0);
    const std::size_t nocc = beta ? _nocc_b : _nocc_a;
    const auto active = p_space_indices(beta);
    for (std::size_t i = 0; i < active.size(); ++i)
      if (active[i] < nocc) result(i, i) = 1.0;
    return result;
  }

  double core_energy() const { return _hamiltonian.get_core_energy(); }
  double reference_energy() const { return _reference_energy; }
  void set_reference_energy(double value) { _reference_energy = value; }

  void set_fock(BTensor alpha, BTensor beta) {
    _fock_a = std::move(alpha);
    _fock_b = std::move(beta);
  }

  BTensor assemble_one_body(bool beta) const {
    BTensor result{btas::Range{_nact, _nact}};
    result.fill(0.0);
    for (const auto& [key, block] : _store) {
      const auto [label, tags] = parse_key(key);
      if (label != "Fbar" || is_beta(tags.front()) != beta) continue;
      for (std::size_t i = 0; i < block.extent(0); ++i)
        for (std::size_t j = 0; j < block.extent(1); ++j)
          result(output_index(tags[0], i), output_index(tags[1], j)) +=
              block(i, j);
    }
    return result;
  }

  BTensor assemble_two_body(int channel) const {
    if (channel < 0 || channel > 2)
      throw std::invalid_argument("ducc: invalid two-body spin channel");

    BTensor result{btas::Range{_nact, _nact, _nact, _nact}};
    result.fill(0.0);
    for (const auto& [key, block] : _store) {
      const auto [label, tags] = parse_key(key);
      if (label != "Vbar") continue;
      const auto alpha_count = static_cast<std::size_t>(std::count_if(
          tags.begin(), tags.end(), [](char tag) { return !is_beta(tag); }));
      if ((channel == 0 && alpha_count != 4) ||
          (channel == 1 && alpha_count != 2) ||
          (channel == 2 && alpha_count != 0))
        continue;

      for (std::size_t i = 0; i < block.extent(0); ++i)
        for (std::size_t j = 0; j < block.extent(1); ++j)
          for (std::size_t k = 0; k < block.extent(2); ++k)
            for (std::size_t l = 0; l < block.extent(3); ++l) {
              std::size_t p = output_index(tags[0], i);
              std::size_t q = output_index(tags[1], j);
              std::size_t r = output_index(tags[2], k);
              std::size_t s = output_index(tags[3], l);
              if (channel == 1 && is_beta(tags[0])) {
                std::swap(p, q);
                std::swap(r, s);
              }
              result(p, q, r, s) += block(i, j, k, l);
            }
    }
    return result;
  }

  void put_gamma(const BTensor&, const BTensor&, const BTensor&, const BTensor&,
                 const BTensor&, double) {}

  void put_final(BTensor one_a, BTensor one_b, BTensor two_aa, BTensor two_ab,
                 BTensor two_bb, double scalar) {
    _final_one_a = std::move(one_a);
    _final_one_b = std::move(one_b);
    _final_two_aa = std::move(two_aa);
    _final_two_ab = std::move(two_ab);
    _final_two_bb = std::move(two_bb);
    _final_scalar = scalar;
  }

  SBT2 one_body() const {
    Eigen::MatrixXd alpha = Eigen::MatrixXd::Zero(_nact, _nact);
    Eigen::MatrixXd beta = Eigen::MatrixXd::Zero(_nact, _nact);
    for (std::size_t i = 0; i < _nact; ++i)
      for (std::size_t j = 0; j < _nact; ++j) {
        alpha(i, j) = _final_one_a(i, j);
        beta(i, j) = _final_one_b(i, j);
      }
    return data::make_spin_diagonal_rank2_sbt(alpha, beta,
                                              /*restricted=*/false);
  }

  SBT4 two_body() const {
    const Eigen::Index size =
        static_cast<Eigen::Index>(_nact * _nact * _nact * _nact);
    Eigen::VectorXd aaaa(size);
    Eigen::VectorXd aabb(size);
    Eigen::VectorXd bbbb(size);
    for (Eigen::Index i = 0; i < size; ++i) {
      aaaa[i] = _final_two_aa.data()[i];
      aabb[i] = _final_two_ab.data()[i];
      bbbb[i] = _final_two_bb.data()[i];
    }
    return data::make_spin_diagonal_rank4_sbt(aaaa, aabb, bbbb,
                                              /*restricted=*/false);
  }

  double final_scalar() const { return _final_scalar; }

 private:
  static bool is_beta(char space) {
    return space == 'O' || space == 'V' || space == 'P';
  }

  static std::pair<std::string, std::string> parse_key(const std::string& key) {
    const auto underscore = key.find('_');
    const auto hash = key.find('#');
    return {key.substr(0, underscore),
            key.substr(underscore + 1, hash - underscore - 1)};
  }

  std::span<const std::uint32_t> p_space_indices(bool beta) const {
    return _p_space->indices(
        data::SymmetryLabel{beta ? data::axes::beta() : data::axes::alpha()});
  }

  std::size_t full_extent(char space) const {
    if (space == 'o') return _nocc_a;
    if (space == 'O') return _nocc_b;
    if (space == 'v') return _nvir_a;
    if (space == 'V') return _nvir_b;
    return _nmo;
  }

  std::vector<std::size_t> active_indices(char space) const {
    const bool beta = is_beta(space);
    const std::size_t nocc = beta ? _nocc_b : _nocc_a;
    std::vector<std::size_t> indices;
    for (std::uint32_t orbital : p_space_indices(beta)) {
      if (space == 'p' || space == 'P')
        indices.push_back(orbital);
      else if (space == 'o' || space == 'O') {
        if (orbital < nocc) indices.push_back(orbital);
      } else if (orbital >= nocc) {
        indices.push_back(orbital - nocc);
      }
    }
    return indices;
  }

  std::size_t output_index(char space, std::size_t local) const {
    if (space == 'v') return _nao_a + local;
    if (space == 'V') return _nao_b + local;
    return local;
  }

  std::size_t spatial_index(char space, std::size_t local) const {
    if (space == 'v') return _nocc_a + local;
    if (space == 'V') return _nocc_b + local;
    return local;
  }

  bool is_active(char space, std::size_t local) const {
    const bool beta = is_beta(space);
    const auto active = p_space_indices(beta);
    const auto spatial = spatial_index(space, local);
    return std::binary_search(active.begin(), active.end(), spatial);
  }

  double chemist_two_body_element(const std::string& tags, std::size_t p,
                                  std::size_t q, std::size_t r,
                                  std::size_t s) const {
    const bool first_beta = is_beta(tags[0]);
    const bool second_beta = is_beta(tags[2]);
    const auto& two_body = _hamiltonian.two_body_integrals();
    if (first_beta == second_beta) {
      const auto spin = first_beta ? data::axes::beta() : data::axes::alpha();
      const auto& block = two_body.block({spin, spin, spin, spin});
      return block[static_cast<Eigen::Index>(index4(_nmo, p, q, r, s))];
    }
    const auto& mixed =
        two_body.block({data::axes::alpha(), data::axes::alpha(),
                        data::axes::beta(), data::axes::beta()});
    return mixed[static_cast<Eigen::Index>(
        first_beta ? index4(_nmo, r, s, p, q) : index4(_nmo, p, q, r, s))];
  }

  BTensor make(const std::string& label, const std::string& tags,
               const std::string& mask) const {
    std::vector<std::vector<std::size_t>> source(tags.size());
    std::vector<std::size_t> extents(tags.size());
    for (std::size_t i = 0; i < tags.size(); ++i) {
      if (i < mask.size() && mask[i] == 'A') {
        source[i] = active_indices(tags[i]);
      } else {
        source[i].resize(full_extent(tags[i]));
        std::iota(source[i].begin(), source[i].end(), std::size_t{0});
      }
      extents[i] = source[i].size();
    }

    BTensor result{btas::Range{extents}};
    result.fill(0.0);
    if (label == "Fbar" || label == "Vbar") return result;

    std::vector<std::size_t> index(tags.size(), 0);
    for (std::size_t flat = 0; flat < result.size(); ++flat) {
      if (label == "f") {
        if (is_beta(tags[0]) == is_beta(tags[1])) {
          const auto& fock = is_beta(tags[0]) ? _fock_b : _fock_a;
          result.data()[flat] =
              fock(spatial_index(tags[0], source[0][index[0]]),
                   spatial_index(tags[1], source[1][index[1]]));
        }
      } else if (label == "v") {
        result.data()[flat] = chemist_two_body_element(
            tags, spatial_index(tags[0], source[0][index[0]]),
            spatial_index(tags[1], source[1][index[1]]),
            spatial_index(tags[2], source[2][index[2]]),
            spatial_index(tags[3], source[3][index[3]]));
      } else if (label == "t" || label == "t_") {
        result.data()[flat] = amplitude(tags, source, index);
      } else {
        throw std::runtime_error("ducc: unsupported tensor '" + label + "'");
      }

      for (std::size_t i = tags.size(); i-- > 0;) {
        if (++index[i] < extents[i]) break;
        index[i] = 0;
      }
    }
    return result;
  }

  double amplitude(const std::string& tags,
                   const std::vector<std::vector<std::size_t>>& source,
                   const std::vector<std::size_t>& index) const {
    std::vector<std::size_t> local(tags.size());
    bool fully_active = true;
    for (std::size_t i = 0; i < tags.size(); ++i) {
      local[i] = source[i][index[i]];
      fully_active &= is_active(tags[i], local[i]);
    }
    if (fully_active) return 0.0;

    const bool virtual_first = tags[0] == 'v' || tags[0] == 'V';
    if (tags.size() == 2) {
      const char v = tags[virtual_first ? 0 : 1];
      const char o = tags[virtual_first ? 1 : 0];
      if (is_beta(v) != is_beta(o)) return 0.0;
      const auto [t1_a, t1_b] = _amplitudes.get_t1_amplitudes();
      const auto& t1 = real_vector(is_beta(v) ? t1_b : t1_a);
      const std::size_t nvir = is_beta(v) ? _nvir_b : _nvir_a;
      const std::size_t a = local[virtual_first ? 0 : 1];
      const std::size_t i = local[virtual_first ? 1 : 0];
      return t1[static_cast<Eigen::Index>(i * nvir + a)];
    }

    const char va = tags[virtual_first ? 0 : 2];
    const char vb = tags[virtual_first ? 1 : 3];
    const char oi = tags[virtual_first ? 2 : 0];
    const char oj = tags[virtual_first ? 3 : 1];
    if (is_beta(va) != is_beta(oi) || is_beta(vb) != is_beta(oj)) return 0.0;

    const std::size_t a = local[virtual_first ? 0 : 2];
    const std::size_t b = local[virtual_first ? 1 : 3];
    const std::size_t i = local[virtual_first ? 2 : 0];
    const std::size_t j = local[virtual_first ? 3 : 1];
    if (is_beta(va) != is_beta(vb))
      return is_beta(va) ? mixed_t2(j, i, b, a) : mixed_t2(i, j, a, b);

    if (_restricted) return mixed_t2(i, j, a, b) - mixed_t2(i, j, b, a);
    const auto [t2_ab, t2_aa, t2_bb] = _amplitudes.get_t2_amplitudes();
    const auto& t2 = real_vector(is_beta(va) ? t2_bb : t2_aa);
    const std::size_t nocc = is_beta(va) ? _nocc_b : _nocc_a;
    const std::size_t nvir = is_beta(va) ? _nvir_b : _nvir_a;
    (void)t2_ab;
    return t2[static_cast<Eigen::Index>(((i * nocc + j) * nvir + a) * nvir +
                                        b)];
  }

  double mixed_t2(std::size_t i, std::size_t j, std::size_t a,
                  std::size_t b) const {
    const auto [t2_ab, t2_aa, t2_bb] = _amplitudes.get_t2_amplitudes();
    (void)t2_aa;
    (void)t2_bb;
    return real_vector(t2_ab)[static_cast<Eigen::Index>(
        ((i * _nocc_b + j) * _nvir_a + a) * _nvir_b + b)];
  }

  const data::CanonicalFourCenterHamiltonianContainer& _hamiltonian;
  const data::AmplitudeContainer& _amplitudes;
  bool _restricted;
  std::shared_ptr<const IndexSet> _p_space;
  BTensor _fock_a;
  BTensor _fock_b;
  double _reference_energy = 0.0;
  BTensor _final_one_a;
  BTensor _final_one_b;
  BTensor _final_two_aa;
  BTensor _final_two_ab;
  BTensor _final_two_bb;
  double _final_scalar = 0.0;
  std::size_t _nmo = 0;
  std::size_t _nocc_a = 0;
  std::size_t _nocc_b = 0;
  std::size_t _nvir_a = 0;
  std::size_t _nvir_b = 0;
  std::size_t _nact = 0;
  std::size_t _nao_a = 0;
  std::size_t _nao_b = 0;
  std::size_t _nav_a = 0;
  std::size_t _nav_b = 0;
  std::map<std::string, BTensor> _store;
  std::map<std::string, double> _scalars;
};

// Generated DUCC equations and the qdk_btas adapters they use.
#include "ducc_equations.inc"

void run_generated(TensorProvider& provider, int level) {
  const auto run = [&provider](auto evaluator) {
    evaluator(provider, provider.nocc_a(), provider.nocc_b(), provider.nvir_a(),
              provider.nvir_b(), provider.nmo(), provider.nao_a(),
              provider.nao_b(), provider.nav_a(), provider.nav_b(),
              provider.nact());
  };
  switch (level) {
    case 0:
      run(run_all_L0);
      return;
    case 1:
      run(run_all_L1);
      return;
    case 2:
      run(run_all_L2);
      return;
    default:
      throw std::runtime_error("ducc: unsupported ducc_level " +
                               std::to_string(level));
  }
}

std::shared_ptr<data::Hamiltonian> make_hamiltonian(
    SBT2 one_body, SBT4 two_body, double core,
    std::shared_ptr<data::Orbitals> orbitals) {
  auto container =
      std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
          std::move(one_body), std::move(two_body), std::move(orbitals), core,
          nullptr);
  return std::make_shared<data::Hamiltonian>(std::move(container));
}

std::shared_ptr<data::Hamiltonian> run_ducc(
    const data::CanonicalFourCenterHamiltonianContainer& hamiltonian,
    const data::Wavefunction& wavefunction, int level,
    std::shared_ptr<const IndexSet> p_space,
    std::shared_ptr<data::Orbitals> orbitals) {
  TensorProvider provider(hamiltonian, wavefunction, std::move(p_space));
  run_generated(provider, level);
  return make_hamiltonian(provider.one_body(), provider.two_body(),
                          provider.final_scalar(), std::move(orbitals));
}

}  // namespace

std::shared_ptr<data::Hamiltonian> DuccSolver::_run_impl(
    std::shared_ptr<data::Hamiltonian> hamiltonian,
    std::shared_ptr<data::Wavefunction> wavefunction,
    std::shared_ptr<const IndexSet> p_space) const {
  if (!p_space)
    throw std::invalid_argument("ducc: p_space_indices must not be null");

  if (hamiltonian->has_inactive_fock_matrix())
    throw std::runtime_error(
        "ducc: input Hamiltonian must span the full orbital space");

  const auto& canonical =
      hamiltonian
          ->get_container<data::CanonicalFourCenterHamiltonianContainer>();
  auto orbitals = output_orbitals(*hamiltonian->get_orbitals(), p_space);
  return run_ducc(canonical, *wavefunction,
                  static_cast<int>(_settings->get<int64_t>("ducc_level")),
                  std::move(p_space), std::move(orbitals));
}

}  // namespace qdk::chemistry::algorithms::microsoft
