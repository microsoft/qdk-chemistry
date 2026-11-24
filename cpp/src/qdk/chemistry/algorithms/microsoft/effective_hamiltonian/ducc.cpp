// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "ducc.hpp"

#include <Eigen/Dense>
#include <SeQuant/core/context.hpp>
#include <SeQuant/core/expr.hpp>
#include <SeQuant/core/wick.hpp>
#include <SeQuant/domain/mbpt/context.hpp>
#include <SeQuant/domain/mbpt/convention.hpp>
#include <SeQuant/domain/mbpt/op.hpp>
#include <SeQuant/domain/mbpt/vac_av.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/wavefunction_containers/amplitude_container.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <span>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "../sequant_evaluation.hpp"

namespace qdk::chemistry::algorithms::microsoft {
namespace {

using sequant_evaluation::index4;
using sequant_evaluation::tensor_size;

// Spin orbitals are ordered occupied alpha, occupied beta, virtual alpha,
// virtual beta; spin 0 is alpha, so a.spin + c.spin picks the aa/ab/bb block.
struct Orbital {
  std::size_t window_index, amplitude_index;
  int spin;
};

struct Input {
  std::size_t nmo = 0, nocc = 0;
  std::array<std::size_t, 2> occupied{}, virtuals{};
  std::array<Eigen::MatrixXd, 2> fock_blocks;
  std::array<const Eigen::VectorXd*, 3> eri{}, doubles{};
  std::array<const Eigen::VectorXd*, 2> singles{};
  std::vector<Orbital> orbitals;
  std::vector<bool> active;
  std::array<std::vector<std::size_t>, 2> kept, inactive;
  std::vector<std::size_t> output;
  bool restricted = false;
  double scalar = 0.0;

  double chemist_eri(std::size_t p, std::size_t q, std::size_t r,
                     std::size_t s) const {
    const auto &a = orbitals[p], &b = orbitals[q], &c = orbitals[r],
               &d = orbitals[s];
    if (a.spin != b.spin || c.spin != d.spin) return 0.0;
    if (a.spin == 1 && c.spin == 0)
      return (*eri[1])[index4(c.window_index, d.window_index, a.window_index,
                              b.window_index, nmo)];
    return (*eri[a.spin + c.spin])[index4(a.window_index, b.window_index,
                                          c.window_index, d.window_index, nmo)];
  }

  double antisymmetrized_eri(std::size_t p, std::size_t q, std::size_t r,
                             std::size_t s) const {
    return chemist_eri(p, r, q, s) - chemist_eri(p, s, q, r);
  }

  double fock(std::size_t p, std::size_t q) const {
    const auto &a = orbitals[p], &b = orbitals[q];
    return a.spin == b.spin
               ? fock_blocks[a.spin](a.window_index, b.window_index)
               : 0.0;
  }

  double t1(std::size_t a, std::size_t i) const {
    if (active[a] && active[i]) return 0.0;
    const auto &v = orbitals[a], &o = orbitals[i];
    if (v.spin != o.spin) return 0.0;
    return (*singles[o.spin])[o.amplitude_index * virtuals[o.spin] +
                              v.amplitude_index];
  }

  double t2(std::size_t a, std::size_t b, std::size_t i, std::size_t j) const {
    if (active[a] && active[b] && active[i] && active[j]) return 0.0;
    const auto &v = orbitals[a], &w = orbitals[b], &o = orbitals[i],
               &p = orbitals[j];
    if (v.spin + w.spin != o.spin + p.spin) return 0.0;
    const auto ab = [&](std::size_t ia, std::size_t ib, std::size_t va,
                        std::size_t vb) {
      return (*doubles[1])[((ia * occupied[1] + ib) * virtuals[0] + va) *
                               virtuals[1] +
                           vb];
    };
    const auto i_index = o.amplitude_index, j_index = p.amplitude_index,
               a_index = v.amplitude_index, b_index = w.amplitude_index;
    if (o.spin == p.spin) {
      if (restricted)
        return ab(i_index, j_index, a_index, b_index) -
               ab(i_index, j_index, b_index, a_index);
      if (i_index == j_index || a_index == b_index) return 0.0;
      // Same-spin doubles may be stored only for i < j and a < b.
      const auto [first, second] = std::minmax(i_index, j_index);
      const auto [lower, upper] = std::minmax(a_index, b_index);
      const double value = (*doubles[2 * o.spin])
          [((first * occupied[o.spin] + second) * virtuals[o.spin] + lower) *
               virtuals[o.spin] +
           upper];
      return (i_index < j_index) == (a_index < b_index) ? value : -value;
    }
    if (o.spin == v.spin)
      return o.spin == 0 ? ab(i_index, j_index, a_index, b_index)
                         : ab(j_index, i_index, b_index, a_index);
    return o.spin == 0 ? -ab(i_index, j_index, b_index, a_index)
                       : -ab(j_index, i_index, a_index, b_index);
  }

  double leaf(std::wstring_view label,
              std::span<const std::size_t> slots) const {
    if (slots.size() != 2 && slots.size() != 4)
      throw std::logic_error(
          "DUCC supports only rank-one and rank-two leaves.");
    if (label == L"f") return fock(slots[0], slots[1]);
    if (label == L"g")
      return antisymmetrized_eri(slots[0], slots[1], slots[2], slots[3]);
    if (label != L"t")
      throw std::logic_error("DUCC encountered an unsupported tensor label.");
    // The adjoint amplitude lists its occupied indices first.
    const bool adjoint = slots[0] < nocc;
    if (slots.size() == 2)
      return adjoint ? t1(slots[1], slots[0]) : t1(slots[0], slots[1]);
    return adjoint ? t2(slots[2], slots[3], slots[0], slots[1])
                   : t2(slots[0], slots[1], slots[2], slots[3]);
  }
};

sequant::ExprPtr normal_ordered_bch(int level) {
  using namespace sequant;
  static std::mutex mutex;
  static std::array<ExprPtr, 3> cache;
  const std::lock_guard lock(mutex);
  if (!cache[level]) {
    const auto scope = set_scoped_default_context(
        {.index_basis_registry_shared_ptr =
             mbpt::make_min_sr_spaces(mbpt::SpinConvention::None),
         .vacuum = Vacuum::SingleProduct});
    const auto mbpt_scope = mbpt::set_scoped_default_mbpt_context(
        {.op_registry_ptr = mbpt::make_minimal_registry()});
    const auto cluster = mbpt::T(2);
    const auto sigma = cluster - adjoint(cluster);
    auto nested = mbpt::H();
    ExprPtr correction = ex<Constant>(0);
    std::int64_t factorial = 1;
    for (int order = 1; order <= level; ++order) {
      nested = nested * sigma - sigma * nested;
      simplify(nested);
      factorial *= order;
      correction += ex<Constant>(rational{1, factorial}) * nested;
    }
    nested = mbpt::F();
    for (int order = 1; order <= level + 1; ++order) {
      nested = nested * sigma - sigma * nested;
      simplify(nested);
    }
    correction += ex<Constant>(rational{1, factorial * (level + 1)}) * nested;
    simplify(correction);
    mbpt::op::lower_to_tensor_form(correction);
    simplify(correction);
    FWickTheorem wick{correction};
    // Cap emitted operators, not intermediate contraction paths.
    wick.full_contractions(false).use_topology(true).max_ops(4);
    auto result = wick.compute(false, true);
    simplify(result);
    cache[level] = std::move(result);
  }
  return cache[level]->clone();
}

Input assemble_input(
    const data::Wavefunction& reference,
    const data::AmplitudeContainer& amplitudes,
    const data::Hamiltonian& hamiltonian,
    const std::shared_ptr<const data::SymmetryBlockedIndexSet>& p_indices) {
  if (!amplitudes.has_t1_amplitudes() || !amplitudes.has_t2_amplitudes())
    throw std::invalid_argument(
        "DUCC requires stored T1 and T2 amplitudes over the Hamiltonian "
        "window.");
  const auto determinant_reference = amplitudes.get_wavefunction();
  const auto& determinants = determinant_reference->get_active_determinants();
  if (determinants.size() != 1)
    throw std::invalid_argument(
        "DUCC amplitudes must be relative to a single determinant.");
  const auto& determinant = determinants.front();
  const auto orbitals = hamiltonian.get_orbitals();
  Input input;
  input.restricted = hamiltonian.is_restricted();
  std::array<std::vector<std::size_t>, 2> window, kept, occupied, virtuals;
  const std::array spins{data::axes::alpha(), data::axes::beta()};
  for (int spin = 0; spin < 2; ++spin) {
    auto& inactive = input.inactive[spin];
    window[spin] =
        data::spin_channel_indices(orbitals->active_indices(), spins[spin]);
    kept[spin] = data::spin_channel_indices(p_indices, spins[spin]);
    inactive =
        data::spin_channel_indices(orbitals->inactive_indices(), spins[spin]);
    for (const auto& other :
         {reference.get_orbitals(), determinant_reference->get_orbitals()}) {
      if (data::spin_channel_indices(other->active_indices(), spins[spin]) !=
              window[spin] ||
          data::spin_channel_indices(other->inactive_indices(), spins[spin]) !=
              inactive)
        throw std::invalid_argument(
            "DUCC Hamiltonian, amplitudes, and determinant must have identical "
            "active windows and inactive cores.");
    }
    for (std::size_t i = 0; i < window[spin].size(); ++i) {
      const bool filled = spin == 0 ? determinant.has_alpha_electron(i)
                                    : determinant.has_beta_electron(i);
      (filled ? occupied[spin] : virtuals[spin]).push_back(i);
      if (filled && !std::binary_search(kept[spin].begin(), kept[spin].end(),
                                        window[spin][i]))
        inactive.push_back(window[spin][i]);
    }
    std::sort(inactive.begin(), inactive.end());
    input.occupied[spin] = occupied[spin].size();
    input.virtuals[spin] = virtuals[spin].size();
  }
  if (window[0].size() != window[1].size() || kept[0].size() != kept[1].size())
    throw std::invalid_argument(
        "DUCC requires equal alpha/beta orbital counts in W and P.");
  if (kept[0].empty())
    throw std::invalid_argument("DUCC requires a nonempty target space P.");
  if (input.restricted && (occupied[0] != occupied[1] || kept[0] != kept[1]))
    throw std::invalid_argument(
        "Restricted DUCC requires a closed-shell determinant and a common "
        "P-space; use unrestricted orbitals for an open-shell reference.");
  input.nmo = window[0].size();
  input.nocc = occupied[0].size() + occupied[1].size();
  const auto two_size =
      tensor_size({input.nmo, input.nmo, input.nmo, input.nmo});
  for (const auto& partition : {occupied, virtuals})
    for (int spin = 0; spin < 2; ++spin)
      for (std::size_t i = 0; i < partition[spin].size(); ++i)
        input.orbitals.push_back({partition[spin][i], i, spin});
  input.active.resize(input.orbitals.size(), false);
  for (int spin = 0; spin < 2; ++spin)
    for (const auto global : kept[spin]) {
      const auto window_index = static_cast<std::size_t>(
          std::lower_bound(window[spin].begin(), window[spin].end(), global) -
          window[spin].begin());
      for (std::size_t i = 0; i < input.orbitals.size(); ++i)
        if (input.orbitals[i].spin == spin &&
            input.orbitals[i].window_index == window_index) {
          input.kept[spin].push_back(i);
          input.active[i] = true;
        }
    }
  for (std::size_t i = 0; i < input.orbitals.size(); ++i)
    if (input.active[i]) input.output.push_back(i);
  const auto [h1a, h1b] = hamiltonian.get_one_body_integrals();
  const auto [gaa, gab, gbb] = hamiltonian.get_two_body_integrals();
  input.eri = {&gaa, &gab, &gbb};
  for (const auto* block : input.eri)
    if (static_cast<std::size_t>(block->size()) != two_size ||
        !block->allFinite())
      throw std::invalid_argument(
          "DUCC requires finite two-body integrals matching W.");
  for (const auto* block : {&h1a, &h1b})
    if (block->rows() != static_cast<Eigen::Index>(input.nmo) ||
        block->cols() != block->rows() || !block->allFinite())
      throw std::invalid_argument(
          "DUCC requires finite one-body integrals matching W.");
  const auto [t1a, t1b] = amplitudes.get_t1_amplitudes();
  const auto [t2ab, t2aa, t2bb] = amplitudes.get_t2_amplitudes();
  const auto real = [](const data::ContainerTypes::VectorVariant& value,
                       std::size_t size) {
    const auto* vector = std::get_if<Eigen::VectorXd>(&value);
    if (!vector || static_cast<std::size_t>(vector->size()) != size ||
        !vector->allFinite())
      throw std::invalid_argument(
          "DUCC requires finite real amplitudes with dimensions matching the "
          "reference window.");
    return vector;
  };
  input.singles = {
      real(t1a, tensor_size({input.occupied[0], input.virtuals[0]})),
      real(t1b, tensor_size({input.occupied[1], input.virtuals[1]}))};
  input.doubles = {
      real(t2aa, tensor_size({input.occupied[0], input.occupied[0],
                              input.virtuals[0], input.virtuals[0]})),
      real(t2ab, tensor_size({input.occupied[0], input.occupied[1],
                              input.virtuals[0], input.virtuals[1]})),
      real(t2bb, tensor_size({input.occupied[1], input.occupied[1],
                              input.virtuals[1], input.virtuals[1]}))};
  if (input.restricted) {
    if (((*input.singles[0] - *input.singles[1]).array().abs() > 1e-12).any())
      throw std::invalid_argument(
          "Restricted DUCC requires equal alpha/beta singles.");
    const auto no = input.occupied[0], nv = input.virtuals[0];
    std::array<bool, 2> spatial{true, true}, antisymmetric{true, true};
    const auto ab = [&](std::size_t i, std::size_t j, std::size_t a,
                        std::size_t b) {
      return (*input.doubles[1])[((i * no + j) * nv + a) * nv + b];
    };
    for (std::size_t i = 0; i < no; ++i)
      for (std::size_t j = 0; j < no; ++j)
        for (std::size_t a = 0; a < nv; ++a)
          for (std::size_t b = 0; b < nv; ++b) {
            const auto value = ab(i, j, a, b);
            if (std::abs(value - ab(j, i, b, a)) > 1e-12)
              throw std::invalid_argument(
                  "Restricted DUCC requires pair-symmetric spatial doubles.");
            for (int spin = 0; spin < 2; ++spin) {
              const auto same =
                  (*input.doubles[2 * spin])[((i * no + j) * nv + a) * nv + b];
              spatial[spin] = spatial[spin] && std::abs(same - value) <= 1e-12;
              antisymmetric[spin] =
                  antisymmetric[spin] &&
                  std::abs(same - (value - ab(i, j, b, a))) <= 1e-12;
            }
          }
    for (int spin = 0; spin < 2; ++spin)
      if (!spatial[spin] && !antisymmetric[spin])
        throw std::invalid_argument(
            "Restricted DUCC same-spin doubles must match the spatial "
            "amplitudes or their antisymmetrization.");
  } else {
    for (int spin = 0; spin < 2; ++spin) {
      const auto no = input.occupied[spin], nv = input.virtuals[spin];
      const auto& same_spin = *input.doubles[2 * spin];
      const auto at = [&](std::size_t i, std::size_t j, std::size_t a,
                          std::size_t b) {
        return same_spin[((i * no + j) * nv + a) * nv + b];
      };
      for (std::size_t i = 0; i < no; ++i)
        for (std::size_t j = 0; j < no; ++j)
          for (std::size_t a = 0; a < nv; ++a)
            for (std::size_t b = 0; b < nv; ++b) {
              const auto value = at(i, j, a, b);
              if ((i < j && a < b) || std::abs(value) <= 1e-12) continue;
              const double image = i == j || a == b
                                       ? 0.0
                                       : ((i < j) == (a < b) ? 1.0 : -1.0) *
                                             at(std::min(i, j), std::max(i, j),
                                                std::min(a, b), std::max(a, b));
              if (std::abs(value - image) > 1e-12)
                throw std::invalid_argument(
                    "Unrestricted DUCC same-spin doubles must be "
                    "antisymmetric or stored only for i < j and a < b.");
            }
    }
  }
  input.fock_blocks = {h1a, h1b};
  for (std::size_t p = 0; p < input.orbitals.size(); ++p)
    for (std::size_t q = 0; q < input.orbitals.size(); ++q) {
      const auto &a = input.orbitals[p], &b = input.orbitals[q];
      if (a.spin != b.spin) continue;
      for (std::size_t m = 0; m < input.nocc; ++m)
        input.fock_blocks[a.spin](a.window_index, b.window_index) +=
            input.antisymmetrized_eri(p, m, q, m);
    }
  input.scalar = hamiltonian.get_core_energy();
  for (std::size_t m = 0; m < input.nocc; ++m) {
    const auto& orbital = input.orbitals[m];
    const auto& h1 = orbital.spin == 0 ? h1a : h1b;
    input.scalar += 0.5 * (h1(orbital.window_index, orbital.window_index) +
                           input.fock(m, m));
  }
  return input;
}

sequant_evaluation::NormalOrderedOperator dress(const Input& input, int level) {
  const auto n = input.output.size();
  sequant_evaluation::NormalOrderedOperator total;
  if (level > 0) {
    sequant_evaluation::SpinOrbitalSpace space;
    for (std::size_t i = 0; i < input.orbitals.size(); ++i)
      (i < input.nocc ? space.occupied : space.virtuals)[input.orbitals[i].spin]
          .push_back(i);
    space.target = input.output;
    total = sequant_evaluation::evaluate(
        normal_ordered_bch(level), space,
        [&input](std::wstring_view label, std::span<const std::size_t> slots) {
          return input.leaf(label, slots);
        });
  } else {
    total.one_body.assign(tensor_size({n, n}), 0.0);
    total.two_body.assign(tensor_size({n, n, n, n}), 0.0);
  }
  total.scalar += input.scalar;
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q) {
      total.one_body[p * n + q] += input.fock(input.output[p], input.output[q]);
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t s = 0; s < n; ++s)
          total.two_body[index4(p, q, r, s, n)] +=
              0.25 * input.antisymmetrized_eri(input.output[p], input.output[q],
                                               input.output[r],
                                               input.output[s]);
    }
  const auto finite = [](double x) { return std::isfinite(x); };
  if (!finite(total.scalar) ||
      !std::all_of(total.one_body.begin(), total.one_body.end(), finite) ||
      !std::all_of(total.two_body.begin(), total.two_body.end(), finite))
    throw std::overflow_error("DUCC evaluation produced non-finite integrals.");
  return total;
}

std::shared_ptr<data::Hamiltonian> assemble_output(
    const sequant_evaluation::NormalOrderedOperator& dressed,
    const Input& input, const std::shared_ptr<data::Orbitals>& orbitals,
    const std::shared_ptr<const data::SymmetryBlockedIndexSet>& p_indices) {
  const auto n = input.output.size();
  std::vector<std::size_t> local(input.orbitals.size(), n), active_occupied;
  for (std::size_t i = 0; i < n; ++i) {
    local[input.output[i]] = i;
    if (input.output[i] < input.nocc) active_occupied.push_back(i);
  }
  const auto dressed_two_body = [&](std::size_t p, std::size_t q, std::size_t r,
                                    std::size_t s) {
    return dressed.two_body[index4(p, q, r, s, n)] -
           dressed.two_body[index4(q, p, r, s, n)] -
           dressed.two_body[index4(p, q, s, r, n)] +
           dressed.two_body[index4(q, p, s, r, n)];
  };
  const auto dressed_one_body = [&](std::size_t p, std::size_t q) {
    double value = dressed.one_body[p * n + q];
    for (const auto m : active_occupied) value -= dressed_two_body(p, m, q, m);
    return value;
  };
  double core = dressed.scalar;
  for (const auto m : active_occupied) {
    core -= dressed_one_body(m, m);
    for (const auto k : active_occupied)
      core -= 0.5 * dressed_two_body(m, k, m, k);
  }
  std::array<std::vector<std::size_t>, 2> kept;
  for (int spin = 0; spin < 2; ++spin)
    for (const auto i : input.kept[spin]) kept[spin].push_back(local[i]);
  const auto np = kept[0].size();
  std::array<Eigen::MatrixXd, 2> one_body;
  std::array<Eigen::VectorXd, 3> two_body;
  for (int spin = 0; spin < (input.restricted ? 1 : 2); ++spin) {
    one_body[spin].resize(np, np);
    for (std::size_t p = 0; p < np; ++p)
      for (std::size_t q = 0; q < np; ++q)
        one_body[spin](p, q) = dressed_one_body(kept[spin][p], kept[spin][q]);
  }
  // Only the antisymmetric part of same-spin interactions is defined.
  for (int channel = 0; channel < 3; ++channel) {
    if (input.restricted && channel != 1) continue;
    const auto &left = kept[channel == 2 ? 1 : 0],
               &right = kept[channel == 0 ? 0 : 1];
    two_body[channel].resize(tensor_size({np, np, np, np}));
    for (std::size_t p = 0; p < np; ++p)
      for (std::size_t q = 0; q < np; ++q)
        for (std::size_t r = 0; r < np; ++r)
          for (std::size_t s = 0; s < np; ++s)
            two_body[channel][index4(p, q, r, s, np)] =
                (channel == 1 ? 1.0 : 0.5) *
                dressed_two_body(left[p], right[r], left[q], right[s]);
  }
  if (!std::isfinite(core) || !one_body[0].allFinite() ||
      !one_body[1].allFinite() || !two_body[0].allFinite() ||
      !two_body[1].allFinite() || !two_body[2].allFinite())
    throw std::overflow_error(
        "DUCC output assembly produced non-finite integrals.");
  const std::array spins{data::axes::alpha(), data::axes::beta()};
  std::unordered_map<data::SymmetryLabel, std::vector<std::uint32_t>>
      inactive_map;
  if (p_indices->symmetries()->has_axis(data::AxisName::Spin)) {
    for (int spin = 0; spin < 2; ++spin)
      inactive_map[spins[spin]] = std::vector<std::uint32_t>(
          input.inactive[spin].begin(), input.inactive[spin].end());
  } else {
    inactive_map[data::SymmetryLabel{}] = std::vector<std::uint32_t>(
        input.inactive[0].begin(), input.inactive[0].end());
  }
  const auto inactive_indices =
      std::make_shared<const data::SymmetryBlockedIndexSet>(
          p_indices->symmetries(), p_indices->extents(),
          std::move(inactive_map));
  std::shared_ptr<data::Orbitals> output_orbitals;
  if (std::dynamic_pointer_cast<data::ModelOrbitals>(orbitals)) {
    output_orbitals =
        std::make_shared<data::ModelOrbitals>(p_indices, inactive_indices);
  } else {
    std::optional<Eigen::MatrixXd> overlap;
    if (orbitals->has_overlap_matrix())
      overlap = orbitals->get_overlap_matrix();
    output_orbitals = std::make_shared<data::Orbitals>(
        orbitals->coefficients(),
        orbitals->has_energies() ? orbitals->energies() : nullptr, overlap,
        orbitals->has_basis_set() ? orbitals->get_basis_set() : nullptr,
        p_indices, inactive_indices);
  }
  const Eigen::MatrixXd empty_fock;
  if (input.restricted)
    return std::make_shared<data::Hamiltonian>(
        std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
            one_body[0], two_body[1], output_orbitals, core, empty_fock));
  return std::make_shared<data::Hamiltonian>(
      std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
          one_body[0], one_body[1], two_body[0], two_body[1], two_body[2],
          output_orbitals, core, empty_fock, empty_fock));
}

}  // namespace

std::shared_ptr<data::Hamiltonian> DuccConstructor::_run_impl(
    std::shared_ptr<data::Wavefunction> reference,
    std::shared_ptr<data::Hamiltonian> hamiltonian,
    std::shared_ptr<const data::SymmetryBlockedIndexSet> p_indices) const {
  QDK_LOG_TRACE_ENTERING();
  const auto level = _settings->get<int64_t>("level");
  if (level < 0 || level > 2)
    throw std::invalid_argument("DUCC level must be between 0 and 2.");
  _validate_inputs(reference, hamiltonian, p_indices);
  if (!reference->has_container_type<data::AmplitudeContainer>())
    throw std::invalid_argument(
        "DUCC requires an AmplitudeContainer reference.");
  if (!hamiltonian->is_hermitian())
    throw std::invalid_argument("DUCC requires a Hermitian Hamiltonian.");
  const auto& amplitudes = reference->get_container<data::AmplitudeContainer>();
  _validate_inputs(amplitudes.get_wavefunction(), hamiltonian, p_indices);
  const auto input =
      assemble_input(*reference, amplitudes, *hamiltonian, p_indices);
  return assemble_output(dress(input, static_cast<int>(level)), input,
                         hamiltonian->get_orbitals(), p_indices);
}

}  // namespace qdk::chemistry::algorithms::microsoft
