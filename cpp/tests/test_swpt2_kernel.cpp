// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <SeQuant/core/context.hpp>
#include <SeQuant/domain/mbpt/convention.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "qdk/chemistry/algorithms/microsoft/effective_hamiltonian/swpt2_kernel.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

namespace sw = qdk::chemistry::algorithms::microsoft::swpt2;

struct OneBodyCase {
  std::vector<int> active, inactive, virtuals;
  std::array<Eigen::MatrixXd, 2> one_body;
  Eigen::VectorXd energies;
  Eigen::VectorXd integrals;
  // Unequal spin blocks; when empty, `energies` and `integrals` serve both.
  Eigen::VectorXd beta_energies;
  std::array<Eigen::VectorXd, 2> same_spin_integrals;

  int norb() const { return energies.size(); }
  const Eigen::VectorXd& spin_energies(int spin) const {
    return spin == 1 && beta_energies.size() != 0 ? beta_energies : energies;
  }
  const Eigen::VectorXd& same_spin(int spin) const {
    return same_spin_integrals[spin].size() != 0 ? same_spin_integrals[spin]
                                                 : integrals;
  }
};

OneBodyCase make_case(std::vector<int> active, std::vector<int> inactive,
                      std::vector<int> virtuals, bool degenerate = false) {
  const int norb = active.size() + inactive.size() + virtuals.size();
  OneBodyCase inputs{std::move(active), std::move(inactive),
                     std::move(virtuals)};
  inputs.energies.resize(norb);
  for (int orbital = 0; orbital < norb; ++orbital)
    inputs.energies(orbital) = -1.5 + 0.31 * orbital;
  if (degenerate && norb > 1)
    inputs.energies(1) = inputs.energies(0) + 0.5 * sw::bare_denominator_floor;
  for (int spin = 0; spin < 2; ++spin) {
    inputs.one_body[spin].resize(norb, norb);
    for (int row = 0; row < norb; ++row)
      for (int col = 0; col < norb; ++col)
        inputs.one_body[spin](row, col) =
            0.1 * std::cos(0.7 * (row + col) + spin);
  }
  inputs.integrals = Eigen::VectorXd::Zero(norb * norb * norb * norb);
  return inputs;
}

void populate_two_body(OneBodyCase& inputs) {
  const int norb = inputs.norb();
  Eigen::VectorXd integrals(norb * norb * norb * norb);
  std::mt19937 generator(42);
  std::uniform_real_distribution<double> coupling(-0.02, 0.02);
  for (int first = 0; first < norb; ++first)
    for (int second = first; second < norb; ++second)
      for (int third = 0; third < norb; ++third)
        for (int fourth = third; fourth < norb; ++fourth) {
          const double value = coupling(generator);
          const std::array<std::array<int, 4>, 8> permutations{
              {{first, second, third, fourth},
               {second, first, third, fourth},
               {first, second, fourth, third},
               {second, first, fourth, third},
               {third, fourth, first, second},
               {fourth, third, first, second},
               {third, fourth, second, first},
               {fourth, third, second, first}}};
          for (const auto& orbital : permutations)
            integrals(sw::idx4(orbital[0], orbital[1], orbital[2], orbital[3],
                               norb)) = value;
        }
  inputs.integrals = std::move(integrals);
}

// Distinct aaaa, aabb and bbbb blocks and beta energies. Same-spin blocks keep
// all eight permutation symmetries, aabb only those within each particle.
void make_unrestricted(OneBodyCase& inputs) {
  const int norb = inputs.norb();
  std::mt19937 generator(7);
  std::uniform_real_distribution<double> coupling(-0.02, 0.02);
  const auto random_block = [&](bool same_spin) {
    Eigen::VectorXd block(norb * norb * norb * norb);
    for (int first = 0; first < norb; ++first)
      for (int second = first; second < norb; ++second)
        for (int third = 0; third < norb; ++third)
          for (int fourth = third; fourth < norb; ++fourth) {
            const double value = coupling(generator);
            for (const auto& [p, q] :
                 {std::pair{first, second}, std::pair{second, first}})
              for (const auto& [r, s] :
                   {std::pair{third, fourth}, std::pair{fourth, third}}) {
                block(sw::idx4(p, q, r, s, norb)) = value;
                if (same_spin) block(sw::idx4(r, s, p, q, norb)) = value;
              }
          }
    return block;
  };
  inputs.same_spin_integrals = {random_block(true), random_block(true)};
  inputs.integrals = random_block(false);
  inputs.beta_energies = inputs.energies;
  for (int orbital = 0; orbital < norb; ++orbital)
    inputs.beta_energies(orbital) += 0.07 * std::cos(1.3 * orbital);
}

Eigen::MatrixXd make_density(const OneBodyCase& inputs, int kind) {
  const int norb = inputs.norb();
  Eigen::MatrixXd density = Eigen::MatrixXd::Zero(norb, norb);
  for (std::size_t axis = 0; axis < inputs.active.size(); ++axis) {
    const int orbital = inputs.active[axis];
    if (kind == 1) density(orbital, orbital) = axis < 2 ? 2.0 : 0.0;
    if (kind == 2 || kind == 3) density(orbital, orbital) = 0.35 + 0.5 * axis;
    if (kind == 4) density(orbital, orbital) = axis == 0 ? 2.0 : 1.0;
  }
  if (kind == 3) {
    Eigen::MatrixXd rotation = Eigen::MatrixXd::Identity(norb, norb);
    for (std::size_t axis = 1; axis < inputs.active.size(); ++axis) {
      Eigen::MatrixXd step = Eigen::MatrixXd::Identity(norb, norb);
      const int first = inputs.active[axis - 1], second = inputs.active[axis];
      step(first, first) = step(second, second) = std::cos(0.37);
      step(first, second) = std::sin(0.37);
      step(second, first) = -std::sin(0.37);
      rotation *= step;
    }
    density = sw::rotate_one_body(density, rotation);
  }
  return density;
}

// A spin-traced `reference_density` splits evenly unless `beta_density` is
// given, in which case `reference_density` is the alpha density.
std::array<Eigen::MatrixXd, 2> spin_densities(
    const Eigen::MatrixXd& reference_density,
    const Eigen::MatrixXd& beta_density) {
  if (beta_density.size() != 0) return {reference_density, beta_density};
  std::array<Eigen::MatrixXd, 2> density;
  if (reference_density.size() != 0)
    density.fill(Eigen::MatrixXd(0.5 * reference_density));
  return density;
}

sw::DownfoldResult run_native(const OneBodyCase& inputs,
                              const sw::RegularizerOptions& regularizer,
                              const Eigen::MatrixXd& reference_density = {},
                              double core_energy = 0.17,
                              const Eigen::MatrixXd& beta_density = {}) {
  return sw::downfold(
      inputs.one_body,
      {std::cref(inputs.same_spin(0)), std::cref(inputs.integrals),
       std::cref(inputs.same_spin(1))},
      {inputs.spin_energies(0), inputs.spin_energies(1)}, inputs.active,
      inputs.inactive, inputs.virtuals, regularizer, core_energy,
      spin_densities(reference_density, beta_density));
}

// Integrals and energies are shared by both spins; one-body blocks may differ.
sw::DownfoldResult shared_integral_downfold(
    const Eigen::MatrixXd& alpha, const Eigen::MatrixXd& beta,
    const Eigen::VectorXd& integrals, const Eigen::VectorXd& energies,
    const std::vector<int>& active, const std::vector<int>& inactive,
    const std::vector<int>& virtuals, const sw::RegularizerOptions& regularizer,
    double core_energy) {
  return sw::downfold(
      {alpha, beta},
      {std::cref(integrals), std::cref(integrals), std::cref(integrals)},
      {energies, energies}, active, inactive, virtuals, regularizer,
      core_energy);
}

// The generalized Fock matrix of a spin-traced density and spin-free integrals.
Eigen::MatrixXd spin_free_fock(const Eigen::MatrixXd& one_body,
                               const Eigen::VectorXd& integrals,
                               const Eigen::MatrixXd& density, int norb) {
  const Eigen::MatrixXd half = 0.5 * density;
  return sw::generalized_fock_matrices(
      {one_body, one_body},
      {std::cref(integrals), std::cref(integrals), std::cref(integrals)},
      {half, half}, norb)[0];
}

struct FockParts {
  Eigen::MatrixXd diagonal, off_diagonal, generator;
  double min_denominator = std::numeric_limits<double>::infinity();
  double max_amplitude = 0.0;
};

template <typename Scalar, typename Emit>
void visit_matrix_elements(Scalar coupling, const std::vector<int>& creators,
                           const std::vector<int>& annihilators, int dimension,
                           Emit&& emit) {
  if (coupling == 0.0) return;
  unsigned creator_mask = 0, annihilator_mask = 0;
  for (const int index : creators) {
    const unsigned mask = 1u << index;
    if (creator_mask & mask) return;
    creator_mask |= mask;
  }
  for (const int index : annihilators) {
    const unsigned mask = 1u << index;
    if (annihilator_mask & mask) return;
    annihilator_mask |= mask;
  }
  const unsigned empty_mask = creator_mask & ~annihilator_mask;
  for (int ket = 0; ket < dimension; ++ket) {
    if ((ket & annihilator_mask) != annihilator_mask || (ket & empty_mask))
      continue;
    unsigned bra = ket;
    Scalar value = coupling;
    const auto apply = [&](int index, bool creation) {
      const unsigned mask = 1u << index;
      if (static_cast<bool>(bra & mask) == creation) {
        value = 0.0;
        return;
      }
      if (std::popcount(bra & (mask - 1u)) % 2 != 0) value = -value;
      bra ^= mask;
    };
    for (auto index = annihilators.rbegin(); index != annihilators.rend();
         ++index)
      apply(*index, false);
    for (auto index = creators.rbegin(); index != creators.rend(); ++index)
      apply(*index, true);
    if (value != 0.0) emit(bra, ket, value);
  }
}

FockParts fock_parts(const OneBodyCase& inputs,
                     const sw::RegularizerOptions& regularizer,
                     double core_energy = 0.17) {
  const int norb = inputs.norb(), dimension = 1 << (2 * norb);
  FockParts result{
      core_energy * Eigen::MatrixXd::Identity(dimension, dimension),
      Eigen::MatrixXd::Zero(dimension, dimension),
      Eigen::MatrixXd::Zero(dimension, dimension)};
  std::vector<char> roles(norb, 'u');
  for (const int orbital : inputs.inactive) roles[orbital] = 'i';
  for (const int orbital : inputs.virtuals) roles[orbital] = 'a';
  const auto append = [&](double coupling, const std::vector<int>& creators,
                          const std::vector<int>& annihilators) {
    if (coupling == 0.0) return;
    bool off_diagonal = false;
    double delta = 0.0;
    for (const char role : {'i', 'a'}) {
      int change = 0;
      for (const int index : creators) change += roles[index / 2] == role;
      for (const int index : annihilators) change -= roles[index / 2] == role;
      off_diagonal = off_diagonal || change != 0;
    }
    for (const int index : creators)
      delta += inputs.spin_energies(index % 2)(index / 2);
    for (const int index : annihilators)
      delta -= inputs.spin_energies(index % 2)(index / 2);
    const double inverse =
        regularizer.sigma2 == 0.0
            ? (std::abs(delta) < 1e-8 ? 0.0 : 1.0 / delta)
            : (delta == 0.0
                   ? 0.0
                   : -std::expm1(-regularizer.sigma2 * delta * delta) / delta);
    if (off_diagonal) {
      result.min_denominator =
          std::min(result.min_denominator, std::abs(delta));
      const double raw_coupling =
          creators.size() == 1 ? coupling : 4.0 * coupling;
      result.max_amplitude =
          std::max(result.max_amplitude,
                   std::abs(raw_coupling) / std::max(std::abs(delta), 1e-8));
    }
    visit_matrix_elements(coupling, creators, annihilators, dimension,
                          [&](int bra, int ket, double value) {
                            if (off_diagonal) {
                              result.off_diagonal(bra, ket) += value;
                              result.generator(bra, ket) += value * inverse;
                            } else
                              result.diagonal(bra, ket) += value;
                          });
  };
  for (int first = 0; first < norb; ++first)
    for (int second = 0; second < norb; ++second)
      for (int spin = 0; spin < 2; ++spin)
        append(inputs.one_body[spin](first, second), {2 * first + spin},
               {2 * second + spin});
  const auto ordered = [&](int first, int second, int third, int fourth) {
    if (first % 2 != fourth % 2 || second % 2 != third % 2) return 0.0;
    const bool same_spin = first % 2 == second % 2;
    if (!same_spin && first % 2 != 0) return 0.0;
    const double integral =
        (same_spin ? inputs.same_spin(first % 2) : inputs.integrals)(
            sw::idx4(first / 2, fourth / 2, second / 2, third / 2, norb));
    return same_spin ? 0.5 * integral : integral;
  };
  const int spin_count = 2 * norb;
  for (int first = 0; first < spin_count; ++first)
    for (int second = 0; second < spin_count; ++second)
      for (int third = 0; third < spin_count; ++third)
        for (int fourth = 0; fourth < spin_count; ++fourth)
          append(0.25 * (ordered(first, second, third, fourth) -
                         ordered(second, first, third, fourth) -
                         ordered(first, second, fourth, third) +
                         ordered(second, first, fourth, third)),
                 {first, second}, {third, fourth});
  return result;
}

sw::DownfoldResult project_fock_reference(
    const OneBodyCase& inputs, const FockParts& parts,
    const Eigen::MatrixXd& density = {},
    const Eigen::MatrixXd& beta_density = {}) {
  using Matrix = Eigen::MatrixXd;
  const auto spin_density = spin_densities(density, beta_density);
  const int count = inputs.active.size(), spin_count = 2 * count;
  const int dimension = 1 << spin_count;
  unsigned external = 0;
  for (int orbital : inputs.inactive) external |= 3u << (2 * orbital);
  const auto embed = [&](unsigned compact) {
    unsigned full = external;
    for (int axis = 0; axis < count; ++axis)
      full |= ((compact >> (2 * axis)) & 3u) << (2 * inputs.active[axis]);
    return full;
  };
  const auto occupied = [&](unsigned mask) {
    std::vector<int> orbitals;
    for (int orbital = 0; orbital < spin_count; ++orbital)
      if (mask & (1u << orbital)) orbitals.push_back(orbital);
    return orbitals;
  };
  std::array<std::vector<unsigned>, 4> sectors;
  for (unsigned mask = 0; mask < static_cast<unsigned>(dimension); ++mask)
    if (std::popcount(mask) <= 3) sectors[std::popcount(mask)].push_back(mask);
  Matrix projected = Matrix::Zero(dimension, dimension);
  for (const auto& sector : sectors)
    for (unsigned bra : sector)
      for (unsigned ket : sector) {
        const unsigned full_bra = embed(bra), full_ket = embed(ket);
        projected(bra, ket) = parts.diagonal(full_bra, full_ket) +
                              0.5 * (parts.generator.row(full_bra).dot(
                                         parts.off_diagonal.col(full_ket)) -
                                     parts.off_diagonal.row(full_bra).dot(
                                         parts.generator.col(full_ket)));
      }
  Matrix lower = projected(0, 0) * Matrix::Identity(dimension, dimension);
  const auto append = [&](Matrix& matrix, double coupling,
                          const std::vector<int>& creators,
                          const std::vector<int>& annihilators) {
    visit_matrix_elements(
        coupling, creators, annihilators, dimension,
        [&](int bra, int ket, double value) { matrix(bra, ket) += value; });
  };
  for (int rank : {1, 2})
    for (unsigned bra : sectors[rank])
      for (unsigned ket : sectors[rank]) {
        const double coupling = projected(bra, ket) - lower(bra, ket);
        auto annihilators = occupied(ket);
        std::reverse(annihilators.begin(), annihilators.end());
        append(lower, coupling, occupied(bra), annihilators);
      }
  Matrix folded = lower;
  if (density.size() != 0) {
    for (unsigned bra : sectors[3])
      for (unsigned ket : sectors[3]) {
        const double coupling = projected(bra, ket) - lower(bra, ket);
        if (coupling == 0.0) continue;
        std::vector<std::pair<int, bool>> ladders;
        for (int orbital : occupied(bra)) ladders.emplace_back(orbital, true);
        auto annihilators = occupied(ket);
        std::reverse(annihilators.begin(), annihilators.end());
        for (int orbital : annihilators) ladders.emplace_back(orbital, false);
        const auto contract =
            [&](auto&& self, const std::vector<std::pair<int, bool>>& remaining,
                std::size_t cursor, double coefficient, int pairs) -> void {
          if (cursor == remaining.size() || !remaining[cursor].second) {
            if (pairs == 0) return;
            std::vector<int> creators, annihilators;
            for (const auto& [orbital, creation] : remaining)
              (creation ? creators : annihilators).push_back(orbital);
            append(folded, pairs % 2 == 0 ? -coefficient : coefficient,
                   creators, annihilators);
            return;
          }
          self(self, remaining, cursor + 1, coefficient, pairs);
          const int creator = remaining[cursor].first;
          for (std::size_t partner = cursor + 1; partner < remaining.size();
               ++partner) {
            const int annihilator = remaining[partner].first;
            if (remaining[partner].second || creator % 2 != annihilator % 2)
              continue;
            const double contraction = spin_density[creator % 2](
                inputs.active[creator / 2], inputs.active[annihilator / 2]);
            if (contraction == 0.0) continue;
            auto reduced = remaining;
            reduced.erase(reduced.begin() + partner);
            reduced.erase(reduced.begin() + cursor);
            const double sign = (partner - cursor - 1) % 2 == 0 ? 1.0 : -1.0;
            self(self, reduced, cursor, coefficient * contraction * sign,
                 pairs + 1);
          }
        };
        contract(contract, ladders, 0, coupling, 0);
      }
  }
  sw::DownfoldResult result;
  result.scalar = folded(0, 0);
  Matrix one_body = folded(0, 0) * Matrix::Identity(dimension, dimension);
  for (int spin = 0; spin < 2; ++spin) {
    result.one_body[spin].resize(count, count);
    for (int row = 0; row < count; ++row)
      for (int col = 0; col < count; ++col) {
        const int creator = 2 * row + spin, annihilator = 2 * col + spin;
        const double coefficient = folded(1u << creator, 1u << annihilator) -
                                   (row == col ? folded(0, 0) : 0.0);
        result.one_body[spin](row, col) = coefficient;
        append(one_body, coefficient, {creator}, {annihilator});
      }
  }
  result.two_body.resize(count * count * count * count);
  for (int first = 0; first < count; ++first)
    for (int second = 0; second < count; ++second)
      for (int third = 0; third < count; ++third)
        for (int fourth = 0; fourth < count; ++fourth) {
          const unsigned bra = (1u << (2 * first)) | (1u << (2 * third + 1));
          const unsigned ket = (1u << (2 * second)) | (1u << (2 * fourth + 1));
          const double sign =
              (first <= third ? 1.0 : -1.0) * (second <= fourth ? 1.0 : -1.0);
          result.two_body(sw::idx4(first, second, third, fourth, count)) =
              sign * (folded(bra, ket) - one_body(bra, ket));
        }
  // Same-spin elements fix only the exchange-antisymmetrized combination.
  for (int spin = 0; spin < 2; ++spin) {
    auto& block = result.same_spin_two_body[spin];
    block = Eigen::VectorXd::Zero(count * count * count * count);
    for (int first = 0; first < count; ++first)
      for (int second = 0; second < count; ++second)
        for (int third = 0; third < count; ++third)
          for (int fourth = 0; fourth < count; ++fourth) {
            if (first == third || second == fourth) continue;
            const unsigned bra =
                (1u << (2 * first + spin)) | (1u << (2 * third + spin));
            const unsigned ket =
                (1u << (2 * second + spin)) | (1u << (2 * fourth + spin));
            const double sign =
                (first < third ? 0.5 : -0.5) * (second < fourth ? 1.0 : -1.0);
            block(sw::idx4(first, second, third, fourth, count)) =
                sign * (folded(bra, ket) - one_body(bra, ket));
          }
  }
  result.min_denominator = parts.min_denominator;
  result.max_amplitude = parts.max_amplitude;
  return result;
}

sw::DownfoldResult fock_reference(const OneBodyCase& inputs,
                                  const sw::RegularizerOptions& regularizer,
                                  const Eigen::MatrixXd& density = {},
                                  double core_energy = 0.17,
                                  const Eigen::MatrixXd& beta_density = {}) {
  return project_fock_reference(inputs,
                                fock_parts(inputs, regularizer, core_energy),
                                density, beta_density);
}

// The particle-exchange-antisymmetrized same-spin block, which unlike the
// block itself does not depend on the representation.
Eigen::VectorXd antisymmetrized(const Eigen::VectorXd& block, int count) {
  Eigen::VectorXd result(block.size());
  for (int first = 0; first < count; ++first)
    for (int second = 0; second < count; ++second)
      for (int third = 0; third < count; ++third)
        for (int fourth = 0; fourth < count; ++fourth)
          result(sw::idx4(first, second, third, fourth, count)) =
              block(sw::idx4(first, second, third, fourth, count)) -
              block(sw::idx4(first, fourth, third, second, count));
  return result;
}

void compare(const sw::DownfoldResult& actual,
             const sw::DownfoldResult& expected) {
  EXPECT_NEAR(actual.scalar, expected.scalar,
              5e-12 * (1.0 + std::abs(expected.scalar)));
  for (int spin = 0; spin < 2; ++spin)
    EXPECT_LT(
        (actual.one_body[spin] - expected.one_body[spin]).cwiseAbs().maxCoeff(),
        5e-12 * (1.0 + expected.one_body[spin].cwiseAbs().maxCoeff()));
  EXPECT_LT((actual.two_body - expected.two_body).cwiseAbs().maxCoeff(),
            5e-12 * (1.0 + expected.two_body.cwiseAbs().maxCoeff()));
  const int count = expected.one_body[0].rows();
  for (int spin = 0; spin < 2; ++spin) {
    const auto reference =
        antisymmetrized(expected.same_spin_two_body[spin], count);
    EXPECT_LT(
        (antisymmetrized(actual.same_spin_two_body[spin], count) - reference)
            .cwiseAbs()
            .maxCoeff(),
        5e-12 * (1.0 + reference.cwiseAbs().maxCoeff()));
  }
  EXPECT_DOUBLE_EQ(actual.min_denominator, expected.min_denominator);
  EXPECT_DOUBLE_EQ(actual.max_amplitude, expected.max_amplitude);
}

Eigen::MatrixXd fock_matrix(const sw::DownfoldResult& result) {
  std::vector<int> active;
  for (int orbital = 0; orbital < result.one_body[0].rows(); ++orbital)
    active.push_back(orbital);
  auto inputs = make_case(active, {}, {});
  inputs.one_body = result.one_body;
  inputs.integrals = result.two_body;
  inputs.same_spin_integrals = result.same_spin_two_body;
  return fock_parts(inputs, {}, result.scalar).diagonal;
}

TEST(Swpt2KernelTest, RestoresCallerContextAndReusesPlans) {
  auto caller_context = sequant::set_scoped_default_context(
      {.index_space_registry_shared_ptr = sequant::mbpt::make_min_sr_spaces(
           sequant::mbpt::SpinConvention::None),
       .vacuum = sequant::Vacuum::Physical});
  const auto original_context = sequant::get_default_context();
  Eigen::MatrixXd one_body(2, 2);
  one_body << -0.5, 0.1, 0.1, 2.0;
  const Eigen::VectorXd integrals = Eigen::VectorXd::Zero(16);
  const Eigen::VectorXd energies = one_body.diagonal();
  const auto identity = shared_integral_downfold(
      one_body, one_body, integrals, energies, {0, 1}, {}, {}, {}, 0.4);
  EXPECT_EQ(sequant::get_default_context(), original_context);
  EXPECT_DOUBLE_EQ(identity.scalar, 0.4);
  EXPECT_LT((identity.one_body[0] - one_body).norm(), 1e-14);
  const sw::RegularizerOptions regularizer{0.8};
  const auto first = shared_integral_downfold(
      one_body, one_body, integrals, energies, {0}, {}, {1}, regularizer, 0.4);
  EXPECT_EQ(sequant::get_default_context(), original_context);
  EXPECT_NEAR(first.one_body[0](0, 0),
              -0.5 + 0.01 * sw::regularized_inverse(-2.5, regularizer), 1e-14);
  auto alternate_context = sequant::set_scoped_default_context(
      {.index_space_registry_shared_ptr = sequant::mbpt::make_min_mr_spaces(
           sequant::mbpt::SpinConvention::Legacy),
       .vacuum = sequant::Vacuum::SingleProduct});
  const auto alternate = sequant::get_default_context();
  const auto second = shared_integral_downfold(
      one_body, one_body, integrals, energies, {0}, {}, {1}, regularizer, 0.4);
  EXPECT_EQ(sequant::get_default_context(), alternate);
  EXPECT_NEAR(second.scalar, first.scalar, 1e-14);
  EXPECT_LT((second.one_body[0] - first.one_body[0]).norm(), 1e-14);
  EXPECT_LT((second.two_body - first.two_body).norm(), 1e-14);
}

TEST(Swpt2KernelTest, DerivationStatistics) {
  for (const auto& [inactive, active, virtuals] :
       {std::array<std::size_t, 3>{10, 10, 10}, {2, 4, 60}, {30, 6, 200}}) {
    const auto statistics = sw::downfold_statistics(inactive, active, virtuals);
    EXPECT_EQ(statistics.projected_terms, 102);
    EXPECT_EQ(statistics.contractions, 984);
    EXPECT_EQ(statistics.maximum_rank, 4);
  }
}

TEST(Swpt2KernelTest, ThreadCountDoesNotChangeResults) {
#ifndef _OPENMP
  GTEST_SKIP() << "OpenMP is disabled";
#else
  // Large enough that the Fock, diagnostic and operand loops all run threaded.
  std::vector<int> virtuals;
  for (int orbital = 8; orbital < 36; ++orbital) virtuals.push_back(orbital);
  auto inputs = make_case({4, 5, 6, 7}, {0, 1, 2, 3}, virtuals);
  populate_two_body(inputs);
  const Eigen::MatrixXd density = make_density(inputs, 3);
  const sw::RegularizerOptions regularizer{0.8};
  const int threads = omp_get_max_threads();
  omp_set_num_threads(1);
  const auto serial = run_native(inputs, regularizer, density);
  const Eigen::MatrixXd serial_fock = spin_free_fock(
      inputs.one_body[0], inputs.integrals, density, inputs.norb());
  omp_set_num_threads(4);
  const auto parallel = run_native(inputs, regularizer, density);
  const Eigen::MatrixXd parallel_fock = spin_free_fock(
      inputs.one_body[0], inputs.integrals, density, inputs.norb());
  omp_set_num_threads(threads);
  EXPECT_EQ(serial.scalar, parallel.scalar);
  for (int spin = 0; spin < 2; ++spin)
    EXPECT_TRUE(serial.one_body[spin] == parallel.one_body[spin]);
  EXPECT_TRUE(serial.two_body == parallel.two_body);
  for (int spin = 0; spin < 2; ++spin)
    EXPECT_TRUE(serial.same_spin_two_body[spin] ==
                parallel.same_spin_two_body[spin]);
  EXPECT_EQ(serial.min_denominator, parallel.min_denominator);
  EXPECT_EQ(serial.max_amplitude, parallel.max_amplitude);
  EXPECT_TRUE(serial_fock == parallel_fock);
#endif
}

TEST(Swpt2KernelTest, RejectsMalformedDimensionsAndPartitions) {
  const auto inputs = make_case({0, 2}, {1}, {});
  auto malformed = inputs;
  malformed.one_body[1] = Eigen::MatrixXd::Zero(2, 2);
  EXPECT_THROW(run_native(malformed, {}), std::invalid_argument);
  EXPECT_THROW(run_native(inputs, {}, Eigen::MatrixXd::Zero(2, 2)),
               std::invalid_argument);
  malformed = inputs;
  malformed.active = {0, 1};
  EXPECT_THROW(run_native(malformed, {}), std::invalid_argument);
  malformed.active = {0, 3};
  EXPECT_THROW(run_native(malformed, {}), std::invalid_argument);
}

TEST(Swpt2KernelTest, SpatialOneBodyMatchesFockSpace) {
  for (const auto& inputs :
       {make_case({0, 3}, {2}, {1, 4}), make_case({0, 2}, {}, {1}),
        make_case({0, 2}, {1}, {}), make_case({0, 1}, {}, {}),
        make_case({0, 3}, {2}, {1, 4}, true)}) {
    for (const double sigma : {0.0, 0.8}) {
      const sw::RegularizerOptions regularizer{sigma};
      compare(run_native(inputs, regularizer),
              fock_reference(inputs, regularizer));
    }
  }
}

TEST(Swpt2KernelTest, BareCutoffMatchesRawDoubleExcitations) {
  auto inputs = make_case({0}, {1, 2}, {3, 4});
  for (auto& block : inputs.one_body) block.setZero();
  inputs.integrals.setZero();
  for (const auto& orbital :
       {std::array{1, 3, 2, 4}, std::array{3, 1, 2, 4}, std::array{1, 3, 4, 2},
        std::array{3, 1, 4, 2}, std::array{2, 4, 1, 3}, std::array{4, 2, 1, 3},
        std::array{2, 4, 3, 1}, std::array{4, 2, 3, 1}})
    inputs.integrals(sw::idx4(orbital[0], orbital[1], orbital[2], orbital[3],
                              inputs.norb())) = 0.013;
  inputs.energies << 0.2, -1.0, -0.5, -1.0, -0.5;
  for (const double fraction : {0.125, 0.25, 0.5, 0.75, 0.875})
    for (const int step : {-2, -1, 0, 1, 2}) {
      inputs.energies(3) = -1.0 + fraction * sw::bare_denominator_floor;
      inputs.energies(4) = -0.5 + (1.0 - fraction) * sw::bare_denominator_floor;
      for (int offset = 0; offset < std::abs(step); ++offset)
        inputs.energies(4) =
            std::nextafter(inputs.energies(4), step < 0 ? -1.0 : 0.0);
      SCOPED_TRACE("fraction=" + std::to_string(fraction) +
                   " ulp_step=" + std::to_string(step));
      const auto actual = run_native(inputs, {});
      compare(actual, fock_reference(inputs, {}));
    }
}

TEST(Swpt2KernelTest, BareCutoffPreservesExternalTrace) {
  auto inputs = make_case({0}, {1}, {2});
  for (auto& block : inputs.one_body) block.setZero();
  inputs.integrals.setZero();
  for (const auto& orbital : {std::array{0, 2, 1, 1}, std::array{2, 0, 1, 1},
                              std::array{1, 1, 0, 2}, std::array{1, 1, 2, 0}})
    inputs.integrals(sw::idx4(orbital[0], orbital[1], orbital[2], orbital[3],
                              inputs.norb())) = 0.001;
  for (const double spectator : {-2.0, -10.0, -100.0})
    for (const int step : {-2, -1, 0, 1, 2}) {
      inputs.energies << -0.5, spectator, -0.5 + sw::bare_denominator_floor;
      for (int offset = 0; offset < std::abs(step); ++offset)
        inputs.energies(2) =
            std::nextafter(inputs.energies(2), step < 0 ? -1.0 : 0.0);
      SCOPED_TRACE("spectator=" + std::to_string(spectator) +
                   " ulp_step=" + std::to_string(step));
      for (const double sigma : {0.0, 0.8}) {
        const sw::RegularizerOptions regularizer{sigma};
        compare(run_native(inputs, regularizer),
                fock_reference(inputs, regularizer));
      }
    }
}

TEST(Swpt2KernelTest, CutoffBoundaryPreservesAllOperatorRanks) {
  int cases = 0;
  for (const double energy_offset : {-64.0, 0.0, 64.0, 1e6})
    for (const double sign : {-1.0, 1.0})
      for (const double spectator : {-2.0, -100.0})
        for (const int step : {-1, 0, 1}) {
          auto inputs = make_case({0, 2, 4}, {1}, {3});
          populate_two_body(inputs);
          inputs.one_body[1] = inputs.one_body[0];
          inputs.energies << -0.5, spectator, 0.25,
              -0.5 + sign * sw::bare_denominator_floor, 1.6;
          inputs.energies.array() += energy_offset;
          if (step != 0)
            inputs.energies(3) = std::nextafter(
                inputs.energies(3),
                step < 0 ? -std::numeric_limits<double>::infinity()
                         : std::numeric_limits<double>::infinity());
          for (const double sigma : {0.0, 0.8}) {
            const sw::RegularizerOptions regularizer{sigma};
            const auto parts = fock_parts(inputs, regularizer);
            for (const bool folded : {false, true}) {
              ++cases;
              SCOPED_TRACE("energy_offset=" + std::to_string(energy_offset) +
                           " sign=" + std::to_string(sign) +
                           " spectator=" + std::to_string(spectator) +
                           " ulp_step=" + std::to_string(step) +
                           " folded=" + std::to_string(folded) +
                           " sigma=" + std::to_string(sigma));
              const auto density =
                  folded ? make_density(inputs, 3) : Eigen::MatrixXd{};
              compare(run_native(inputs, regularizer, density),
                      project_fock_reference(inputs, parts, density));
            }
          }
        }
  EXPECT_EQ(cases, 192);
}

TEST(Swpt2KernelTest, ZeroCouplingsKeepOnlyTheBaseline) {
  auto inputs = make_case({0, 2, 4}, {1}, {3});
  for (int spin = 0; spin < 2; ++spin) {
    inputs.one_body[spin].setZero();
    inputs.one_body[spin].diagonal() = (0.3 + 0.1 * spin) * inputs.energies;
  }
  for (const bool folded : {false, true})
    for (const double sigma : {0.0, 0.8}) {
      const auto density = folded ? make_density(inputs, 3) : Eigen::MatrixXd{};
      const sw::RegularizerOptions regularizer{sigma};
      const auto actual = run_native(inputs, regularizer, density);
      compare(actual, fock_reference(inputs, regularizer, density));
      EXPECT_TRUE(std::isinf(actual.min_denominator));
      EXPECT_DOUBLE_EQ(actual.max_amplitude, 0.0);
      EXPECT_DOUBLE_EQ(actual.two_body.squaredNorm(), 0.0);
    }
}

TEST(Swpt2KernelTest, ArbitraryOrbitalRelabelingIsCovariant) {
  auto inputs = make_case({0, 3}, {1, 4}, {2, 5});
  populate_two_body(inputs);
  const std::array<int, 6> renamed{4, 3, 5, 2, 1, 0};
  auto permuted = inputs;
  for (auto* role : {&permuted.active, &permuted.inactive, &permuted.virtuals})
    for (auto& orbital : *role) orbital = renamed[orbital];
  const auto density = make_density(inputs, 3);
  Eigen::MatrixXd permuted_density(inputs.norb(), inputs.norb());
  for (int first = 0; first < inputs.norb(); ++first) {
    permuted.energies(renamed[first]) = inputs.energies(first);
    for (int second = 0; second < inputs.norb(); ++second) {
      for (int spin = 0; spin < 2; ++spin)
        permuted.one_body[spin](renamed[first], renamed[second]) =
            inputs.one_body[spin](first, second);
      permuted_density(renamed[first], renamed[second]) =
          density(first, second);
      for (int third = 0; third < inputs.norb(); ++third)
        for (int fourth = 0; fourth < inputs.norb(); ++fourth)
          permuted.integrals(sw::idx4(renamed[first], renamed[second],
                                      renamed[third], renamed[fourth],
                                      inputs.norb())) =
              inputs.integrals(
                  sw::idx4(first, second, third, fourth, inputs.norb()));
    }
  }
  for (const bool folded : {false, true})
    for (const double sigma : {0.0, 0.8}) {
      const sw::RegularizerOptions regularizer{sigma};
      const auto original =
          run_native(inputs, regularizer, folded ? density : Eigen::MatrixXd{});
      const auto actual = run_native(
          permuted, regularizer, folded ? permuted_density : Eigen::MatrixXd{});
      EXPECT_NEAR(actual.scalar, original.scalar, 1e-12);
      for (int spin = 0; spin < 2; ++spin)
        EXPECT_LT((actual.one_body[spin] - original.one_body[spin])
                      .cwiseAbs()
                      .maxCoeff(),
                  1e-12);
      EXPECT_LT((actual.two_body - original.two_body).cwiseAbs().maxCoeff(),
                1e-12);
      for (int spin = 0; spin < 2; ++spin)
        EXPECT_LT((antisymmetrized(actual.same_spin_two_body[spin], 2) -
                   antisymmetrized(original.same_spin_two_body[spin], 2))
                      .cwiseAbs()
                      .maxCoeff(),
                  1e-12);
      EXPECT_DOUBLE_EQ(actual.min_denominator, original.min_denominator);
      EXPECT_DOUBLE_EQ(actual.max_amplitude, original.max_amplitude);
    }
}

TEST(Swpt2KernelTest, DensityFoldMatchesFockSpace) {
  double fold_signal = 0.0;
  int cases = 0;
  for (auto inputs :
       {make_case({0, 2}, {}, {1}), make_case({0, 2}, {1}, {}),
        make_case({0, 2, 4}, {1}, {3}), make_case({0, 2, 3, 4}, {}, {1}),
        make_case({0, 1, 2}, {}, {}), make_case({0, 2}, {1}, {3}, true)}) {
    populate_two_body(inputs);
    inputs.one_body[1] = inputs.one_body[0];
    for (const double sigma : {0.0, 0.8}) {
      const sw::RegularizerOptions regularizer{sigma};
      const auto parts = fock_parts(inputs, regularizer);
      const auto discarded = run_native(inputs, regularizer);
      compare(discarded, project_fock_reference(inputs, parts));
      for (int kind = 0; kind < 5; ++kind) {
        SCOPED_TRACE("case=" + std::to_string(cases++) + " density=" +
                     std::to_string(kind) + " sigma=" + std::to_string(sigma));
        const auto density = make_density(inputs, kind);
        const auto actual = run_native(inputs, regularizer, density);
        compare(actual, project_fock_reference(inputs, parts, density));
        fold_signal = std::max(
            fold_signal,
            (actual.two_body - discarded.two_body).cwiseAbs().maxCoeff());
      }
    }
  }
  EXPECT_EQ(cases, 60);
  EXPECT_GT(fold_signal, 1e-8);
}

TEST(Swpt2KernelTest, MultipleExternalOrbitalsMatchFockSpace) {
  int cases = 0;
  for (auto inputs :
       {make_case({0, 3}, {1, 4}, {2}), make_case({0, 3}, {1}, {2, 4}),
        make_case({0}, {1, 3}, {2, 4})}) {
    populate_two_body(inputs);
    inputs.one_body[1] = inputs.one_body[0];
    for (const bool folded : {false, true})
      for (const double sigma : {0.0, 0.8}) {
        SCOPED_TRACE("case=" + std::to_string(cases++) +
                     " norb=" + std::to_string(inputs.norb()) +
                     " fold=" + std::to_string(folded) +
                     " sigma=" + std::to_string(sigma));
        const auto density =
            folded ? make_density(inputs, 3) : Eigen::MatrixXd{};
        const sw::RegularizerOptions regularizer{sigma};
        compare(run_native(inputs, regularizer, density),
                fock_reference(inputs, regularizer, density));
      }
  }
  EXPECT_EQ(cases, 12);
}

TEST(Swpt2KernelTest, UnrestrictedBlocksMatchFockSpace) {
  int cases = 0;
  double spin_asymmetry = 0.0;
  for (auto inputs :
       {make_case({0, 2}, {1}, {3}), make_case({0, 2, 4}, {1}, {3}),
        make_case({0, 3}, {1, 4}, {2}), make_case({0, 2, 3, 4}, {}, {1})}) {
    make_unrestricted(inputs);
    const Eigen::MatrixXd alpha = 0.5 * make_density(inputs, 2);
    const Eigen::MatrixXd beta = 0.5 * make_density(inputs, 3);
    for (const bool folded : {false, true})
      for (const double sigma : {0.0, 0.8}) {
        SCOPED_TRACE("case=" + std::to_string(cases++) +
                     " fold=" + std::to_string(folded) +
                     " sigma=" + std::to_string(sigma));
        const sw::RegularizerOptions regularizer{sigma};
        const Eigen::MatrixXd density = folded ? alpha : Eigen::MatrixXd{};
        const Eigen::MatrixXd beta_density = folded ? beta : Eigen::MatrixXd{};
        const auto actual =
            run_native(inputs, regularizer, density, 0.17, beta_density);
        compare(actual, fock_reference(inputs, regularizer, density, 0.17,
                                       beta_density));
        const int count = inputs.active.size();
        // Every block is Hermitian; the same-spin blocks are also symmetric
        // under electron exchange, as the restricted output is.
        const auto violation = [count](const Eigen::VectorXd& block,
                                       bool exchange) {
          double worst = 0.0;
          for (int p = 0; p < count; ++p)
            for (int q = 0; q < count; ++q)
              for (int r = 0; r < count; ++r)
                for (int s = 0; s < count; ++s) {
                  const double value = block(sw::idx4(p, q, r, s, count));
                  worst = std::max(
                      worst,
                      std::abs(value - block(sw::idx4(q, p, s, r, count))));
                  if (exchange)
                    worst = std::max(
                        worst,
                        std::abs(value - block(sw::idx4(r, s, p, q, count))));
                }
          return worst;
        };
        EXPECT_LT(violation(actual.two_body, false), 1e-12);
        for (const auto& block : actual.same_spin_two_body)
          EXPECT_LT(violation(block, true), 1e-12);
        spin_asymmetry =
            std::max(spin_asymmetry,
                     (antisymmetrized(actual.same_spin_two_body[0], count) -
                      antisymmetrized(actual.same_spin_two_body[1], count))
                         .cwiseAbs()
                         .maxCoeff());
      }
  }
  EXPECT_EQ(cases, 16);
  EXPECT_GT(spin_asymmetry, 1e-6);
}

TEST(Swpt2KernelTest, IndependentFockSpace) {
  for (auto inputs :
       {make_case({0, 2}, {1}, {3}), make_case({0, 2, 3}, {}, {1})}) {
    inputs.one_body[1] = inputs.one_body[0];
    populate_two_body(inputs);
    unsigned external = 0;
    for (const int orbital : inputs.inactive) external |= 3u << (2 * orbital);
    const auto embed = [&](unsigned compact) {
      unsigned full = external;
      for (std::size_t axis = 0; axis < inputs.active.size(); ++axis)
        full |= ((compact >> (2 * axis)) & 3u) << (2 * inputs.active[axis]);
      return full;
    };
    for (const double sigma : {0.0, 0.8}) {
      const sw::RegularizerOptions regularizer{sigma};
      const auto parts = fock_parts(inputs, regularizer);
      const Eigen::MatrixXd exact =
          parts.diagonal + 0.5 * (parts.generator * parts.off_diagonal -
                                  parts.off_diagonal * parts.generator);
      EXPECT_LT((exact - exact.transpose()).cwiseAbs().maxCoeff(), 1e-12);
      const auto matrix = fock_matrix(run_native(inputs, regularizer));
      double error = 0.0, correction = 0.0, truncation_signal = 0.0;
      for (unsigned bra = 0; bra < static_cast<unsigned>(matrix.rows()); ++bra)
        for (unsigned ket = 0; ket < static_cast<unsigned>(matrix.cols());
             ++ket) {
          const double difference =
              std::abs(matrix(bra, ket) - exact(embed(bra), embed(ket)));
          if (std::popcount(bra) <= 2 && std::popcount(ket) <= 2) {
            error = std::max(error, difference);
            correction = std::max(
                correction, std::abs(exact(embed(bra), embed(ket)) -
                                     parts.diagonal(embed(bra), embed(ket))));
          }
          if (std::popcount(bra) == 3 && std::popcount(ket) == 3)
            truncation_signal = std::max(truncation_signal, difference);
        }
      EXPECT_LT(error, 1e-11);
      EXPECT_GT(correction, 1e-6);
      EXPECT_GT(truncation_signal, 1e-6);
      Eigen::MatrixXd density =
          Eigen::MatrixXd::Zero(inputs.norb(), inputs.norb());
      for (const int orbital : inputs.active) density(orbital, orbital) = 2.0;
      const auto folded = fock_matrix(run_native(inputs, regularizer, density));
      const unsigned reference = folded.rows() - 1;
      EXPECT_NEAR(folded(reference, reference),
                  exact(embed(reference), embed(reference)), 1e-11);
      EXPECT_GT(
          std::abs(folded(reference, reference) - matrix(reference, reference)),
          1e-6);
    }
  }
}

TEST(Swpt2KernelTest, IndependentSecondOrderSingles) {
  const auto inputs = make_case({0, 2}, {1}, {3});
  const auto embed = [&](unsigned compact) {
    unsigned full = 3u << 2;
    for (std::size_t axis = 0; axis < inputs.active.size(); ++axis)
      full |= ((compact >> (2 * axis)) & 3u) << (2 * inputs.active[axis]);
    return full;
  };
  for (const double sigma : {0.0, 0.8}) {
    const sw::RegularizerOptions regularizer{sigma};
    const auto parts = fock_parts(inputs, regularizer);
    const Eigen::MatrixXd exact =
        parts.diagonal + 0.5 * (parts.generator * parts.off_diagonal -
                                parts.off_diagonal * parts.generator);
    const auto matrix = fock_matrix(run_native(inputs, regularizer));
    double error = 0.0, correction = 0.0;
    for (unsigned bra = 0; bra < static_cast<unsigned>(matrix.rows()); ++bra)
      for (unsigned ket = 0; ket < static_cast<unsigned>(matrix.cols());
           ++ket) {
        error = std::max(
            error, std::abs(matrix(bra, ket) - exact(embed(bra), embed(ket))));
        correction = std::max(correction,
                              std::abs(exact(embed(bra), embed(ket)) -
                                       parts.diagonal(embed(bra), embed(ket))));
      }
    EXPECT_LT(error, 1e-11);
    EXPECT_GT(correction, 1e-6);
  }
}

TEST(Swpt2KernelTest, ClosedShellMp2Limit) {
  auto inputs = make_case({0, 1, 2}, {}, {3, 4});
  populate_two_body(inputs);
  Eigen::MatrixXd density = Eigen::MatrixXd::Zero(inputs.norb(), inputs.norb());
  for (const int orbital : inputs.active) density(orbital, orbital) = 2.0;
  const auto mean_field =
      spin_free_fock(Eigen::MatrixXd::Zero(inputs.norb(), inputs.norb()),
                     inputs.integrals, density, inputs.norb());
  inputs.one_body[0] =
      Eigen::MatrixXd(inputs.energies.asDiagonal()) - mean_field;
  inputs.one_body[1] = inputs.one_body[0];
  const auto integral = [&](int first, int second, int third, int fourth) {
    return inputs.integrals(
        sw::idx4(first, second, third, fourth, inputs.norb()));
  };
  double hf_energy = 0.17, mp2 = 0.0;
  for (const int occupied : inputs.active) {
    hf_energy += 2 * inputs.one_body[0](occupied, occupied);
    for (const int other : inputs.active) {
      hf_energy += 2 * integral(occupied, occupied, other, other) -
                   integral(occupied, other, other, occupied);
      for (const int empty : inputs.virtuals)
        for (const int second_empty : inputs.virtuals) {
          const double direct = integral(occupied, empty, other, second_empty);
          mp2 += direct *
                 (2 * direct - integral(occupied, second_empty, other, empty)) /
                 (inputs.energies(occupied) + inputs.energies(other) -
                  inputs.energies(empty) - inputs.energies(second_empty));
        }
    }
  }
  const auto expectation = [&](const sw::DownfoldResult& result) {
    double value =
        result.scalar + result.one_body[0].trace() + result.one_body[1].trace();
    const int count = inputs.active.size();
    for (int first = 0; first < count; ++first)
      for (int second = 0; second < count; ++second)
        value +=
            2 * result.two_body(sw::idx4(first, first, second, second, count)) -
            result.two_body(sw::idx4(first, second, second, first, count));
    return value;
  };
  const double actual = expectation(run_native(inputs, {}, density));
  EXPECT_LT(mp2, -1e-4);
  EXPECT_NEAR(actual, hf_energy + mp2, 1e-11);
  EXPECT_GT(std::abs(expectation(run_native(inputs, {})) - actual), 1e-6);
}

TEST(Swpt2KernelTest, VirtualExternalSpaceConvergesAsCouplingShrinks) {
  const auto residual = [&](double coupling_scale) {
    auto inputs = make_case({0, 1}, {}, {2});
    inputs.one_body[0].setZero();
    inputs.one_body[0].diagonal() << 0.0, 0.5, 3.0;
    inputs.one_body[0](0, 2) = inputs.one_body[0](2, 0) = -0.2 * coupling_scale;
    inputs.one_body[1] = inputs.one_body[0];
    const auto set_integral = [&](int first, int second, int third, int fourth,
                                  double value) {
      for (const auto& orbital : {std::array{first, second, third, fourth},
                                  std::array{second, first, third, fourth},
                                  std::array{first, second, fourth, third},
                                  std::array{second, first, fourth, third},
                                  std::array{third, fourth, first, second},
                                  std::array{fourth, third, first, second},
                                  std::array{third, fourth, second, first},
                                  std::array{fourth, third, second, first}})
        inputs.integrals(sw::idx4(orbital[0], orbital[1], orbital[2],
                                  orbital[3], 3)) = value;
    };
    set_integral(0, 0, 0, 0, 0.60);
    set_integral(1, 1, 1, 1, 0.50);
    set_integral(0, 0, 1, 1, 0.30);
    set_integral(2, 2, 2, 2, 0.40);
    set_integral(0, 0, 2, 2, 0.20);
    set_integral(1, 1, 2, 2, 0.20);
    set_integral(0, 2, 0, 0, 0.10 * coupling_scale);
    set_integral(0, 2, 1, 1, 0.10 * coupling_scale);
    Eigen::MatrixXd density = Eigen::MatrixXd::Zero(3, 3);
    density(0, 0) = 2.0;
    inputs.energies =
        spin_free_fock(inputs.one_body[0], inputs.integrals, density, 3)
            .diagonal();
    const auto ground = [](const Eigen::MatrixXd& matrix, int count) {
      Eigen::MatrixXd sector(count * count, count * count);
      for (int first = 0; first < count; ++first)
        for (int second = 0; second < count; ++second)
          for (int third = 0; third < count; ++third)
            for (int fourth = 0; fourth < count; ++fourth)
              sector(first * count + second, third * count + fourth) =
                  matrix((1u << (2 * first)) | (1u << (2 * second + 1)),
                         (1u << (2 * third)) | (1u << (2 * fourth + 1)));
      return Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(sector)
          .eigenvalues()(0);
    };
    const auto full = fock_parts(inputs, {}, 0.0);
    const double exact = ground(full.diagonal + full.off_diagonal, 3);
    const double effective =
        ground(fock_matrix(run_native(inputs, {}, {}, 0.0)), 2);
    return std::abs(exact - effective);
  };
  const double largest = residual(0.4), middle = residual(0.2),
               smallest = residual(0.1);
  EXPECT_GT(largest, middle);
  EXPECT_GT(middle, smallest);
  EXPECT_LT(middle, 0.3 * largest);
  EXPECT_LT(smallest, 0.3 * middle);
}

TEST(Swpt2KernelTest, OpenShellFoldStaysSpinFree) {
  auto inputs = make_case({0, 1, 2, 3}, {}, {4});
  populate_two_body(inputs);
  inputs.one_body[1] = inputs.one_body[0];
  Eigen::MatrixXd density = Eigen::MatrixXd::Zero(5, 5);
  density.diagonal() << 2.0, 1.0, 1.0, 0.0, 0.0;
  const auto matrix = fock_matrix(run_native(inputs, {}, density));
  const auto ground = [&](int alpha, int beta) {
    std::vector<unsigned> states;
    for (unsigned occupation = 0; occupation < 256; ++occupation) {
      const int alpha_count = std::popcount(occupation & 0x55u);
      const int beta_count = std::popcount(occupation & 0xaau);
      if (alpha_count == alpha && beta_count == beta)
        states.push_back(occupation);
    }
    Eigen::MatrixXd sector(states.size(), states.size());
    for (std::size_t bra = 0; bra < states.size(); ++bra)
      for (std::size_t ket = 0; ket < states.size(); ++ket)
        sector(bra, ket) = matrix(states[bra], states[ket]);
    return Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(sector).eigenvalues()(
        0);
  };
  for (const auto& [alpha, beta] :
       {std::pair{3, 1}, std::pair{3, 2}, std::pair{4, 2}, std::pair{4, 1}})
    EXPECT_NEAR(ground(alpha, beta), ground(beta, alpha), 1e-10);
}

TEST(Swpt2KernelTest, SpinExchangeAndFourFoldSymmetry) {
  auto inputs = make_case({0, 2, 4}, {1}, {3});
  populate_two_body(inputs);
  const auto density = make_density(inputs, 3);
  const sw::RegularizerOptions regularizer{0.8};
  const auto original = run_native(inputs, regularizer, density);
  std::swap(inputs.one_body[0], inputs.one_body[1]);
  const auto exchanged = run_native(inputs, regularizer, density);
  EXPECT_NEAR(original.scalar, exchanged.scalar, 1e-12);
  EXPECT_LT(
      (original.one_body[0] - exchanged.one_body[1]).cwiseAbs().maxCoeff(),
      1e-12);
  EXPECT_LT(
      (original.one_body[1] - exchanged.one_body[0]).cwiseAbs().maxCoeff(),
      1e-12);
  const int count = inputs.active.size();
  for (int first = 0; first < count; ++first)
    for (int second = 0; second < count; ++second)
      for (int third = 0; third < count; ++third)
        for (int fourth = 0; fourth < count; ++fourth)
          EXPECT_NEAR(
              original.two_body(sw::idx4(first, second, third, fourth, count)),
              exchanged.two_body(sw::idx4(third, fourth, first, second, count)),
              1e-12);
  inputs.one_body[1] = inputs.one_body[0];
  const auto result = run_native(inputs, regularizer, density);
  for (const auto& one_body : result.one_body)
    EXPECT_LT((one_body - one_body.transpose()).cwiseAbs().maxCoeff(), 1e-12);
  const auto integral = [&](int first, int second, int third, int fourth) {
    return result.two_body(sw::idx4(first, second, third, fourth, count));
  };
  double eight_fold_violation = 0.0;
  for (int first = 0; first < count; ++first)
    for (int second = 0; second < count; ++second)
      for (int third = 0; third < count; ++third)
        for (int fourth = 0; fourth < count; ++fourth) {
          const double value = integral(first, second, third, fourth);
          EXPECT_NEAR(value, integral(third, fourth, first, second), 1e-12);
          EXPECT_NEAR(value, integral(second, first, fourth, third), 1e-12);
          EXPECT_NEAR(value, integral(fourth, third, second, first), 1e-12);
          eight_fold_violation = std::max(
              eight_fold_violation,
              std::abs(value - integral(second, first, third, fourth)));
        }
  EXPECT_GT(eight_fold_violation, 1e-6);
}

TEST(Swpt2KernelTest, SemicanonicalCovariance) {
  auto inputs = make_case({2, 3}, {0, 1}, {4, 5});
  populate_two_body(inputs);
  inputs.one_body[0].diagonal() = inputs.energies;
  inputs.one_body[1] = inputs.one_body[0];
  auto density = make_density(inputs, 3);
  for (const int orbital : inputs.inactive) density(orbital, orbital) = 2.0;
  const auto run = [&](const Eigen::MatrixXd& one_body,
                       const Eigen::VectorXd& integrals,
                       const Eigen::MatrixXd& reference_density, double sigma) {
    const auto fock =
        spin_free_fock(one_body, integrals, reference_density, inputs.norb());
    const auto rotation = sw::semicanonical_rotation(
        fock, {inputs.inactive, inputs.active, inputs.virtuals}, 1e-14);
    auto prepared = inputs;
    prepared.one_body[0] = sw::rotate_one_body(one_body, rotation);
    prepared.one_body[1] = prepared.one_body[0];
    prepared.integrals =
        sw::rotate_two_body(integrals, rotation, inputs.norb());
    prepared.energies = sw::rotate_one_body(fock, rotation).diagonal();
    const auto rotated_density =
        sw::rotate_one_body(reference_density, rotation);
    auto result =
        run_native(prepared, sw::RegularizerOptions{sigma}, rotated_density);
    const Eigen::MatrixXd active_rotation = rotation.block(2, 2, 2, 2);
    for (auto& block : result.one_body)
      block = sw::rotate_one_body(block, active_rotation.transpose());
    result.two_body =
        sw::rotate_two_body(result.two_body, active_rotation.transpose(), 2);
    for (auto& block : result.same_spin_two_body)
      block = sw::rotate_two_body(block, active_rotation.transpose(), 2);
    return result;
  };
  Eigen::MatrixXd basis =
      Eigen::MatrixXd::Identity(inputs.norb(), inputs.norb());
  for (const auto& [first, angle] :
       {std::pair{0, 0.19}, std::pair{2, 0.31}, std::pair{4, -0.27}}) {
    basis(first, first) = basis(first + 1, first + 1) = std::cos(angle);
    basis(first, first + 1) = -std::sin(angle);
    basis(first + 1, first) = std::sin(angle);
  }
  for (const double sigma : {0.0, 0.8}) {
    const auto original =
        run(inputs.one_body[0], inputs.integrals, density, sigma);
    auto rotated =
        run(sw::rotate_one_body(inputs.one_body[0], basis),
            sw::rotate_two_body(inputs.integrals, basis, inputs.norb()),
            sw::rotate_one_body(density, basis), sigma);
    const Eigen::MatrixXd active_basis = basis.block(2, 2, 2, 2);
    for (auto& block : rotated.one_body)
      block = sw::rotate_one_body(block, active_basis.transpose());
    rotated.two_body =
        sw::rotate_two_body(rotated.two_body, active_basis.transpose(), 2);
    for (auto& block : rotated.same_spin_two_body)
      block = sw::rotate_two_body(block, active_basis.transpose(), 2);
    EXPECT_NEAR(original.scalar, rotated.scalar, 1e-11);
    for (int spin = 0; spin < 2; ++spin)
      EXPECT_LT((original.one_body[spin] - rotated.one_body[spin])
                    .cwiseAbs()
                    .maxCoeff(),
                1e-11);
    EXPECT_LT((original.two_body - rotated.two_body).cwiseAbs().maxCoeff(),
              1e-11);
    for (int spin = 0; spin < 2; ++spin)
      EXPECT_LT((antisymmetrized(original.same_spin_two_body[spin], 2) -
                 antisymmetrized(rotated.same_spin_two_body[spin], 2))
                    .cwiseAbs()
                    .maxCoeff(),
                1e-11);
    EXPECT_NEAR(original.min_denominator, rotated.min_denominator, 1e-11);
    EXPECT_NEAR(original.max_amplitude, rotated.max_amplitude, 1e-11);
  }
}

TEST(Swpt2KernelTest, DISABLED_FullKernelBenchmark) {
  using Clock = std::chrono::steady_clock;
  const auto elapsed = [](const auto& start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start)
        .count();
  };
  std::vector<int> active_counts{2, 3, 4};
  if (const char* active_setting = std::getenv("SWPT2_BENCH_ACTIVE"))
    active_counts = {std::stoi(active_setting)};
  const char* external_setting = std::getenv("SWPT2_BENCH_EXTERNAL");
  const int external_count = external_setting ? std::stoi(external_setting) : 1;
  ASSERT_GE(external_count, 1);
  ASSERT_LE(external_count, 4);
  const auto setup_start = Clock::now();
  const auto statistics = sw::downfold_statistics(
      external_count, active_counts.front(), external_count);
  const double setup_ms = elapsed(setup_start);
  ASSERT_EQ(statistics.projected_terms, 102);
  ASSERT_EQ(statistics.contractions, 984);
  ASSERT_EQ(statistics.maximum_rank, 4);
  std::cout << "full_kernel_setup_ms=" << setup_ms
            << " projected_terms=" << statistics.projected_terms
            << " contractions=" << statistics.contractions
            << " maximum_rank=" << statistics.maximum_rank << '\n';
  for (const int active_count : active_counts) {
    ASSERT_GE(active_count, 2);
    ASSERT_LE(active_count, 4);
    std::vector<int> active;
    for (int orbital = 0; orbital < active_count; ++orbital)
      active.push_back(orbital + external_count);
    std::vector<int> inactive, virtuals;
    for (int orbital = 0; orbital < external_count; ++orbital) {
      inactive.push_back(orbital);
      virtuals.push_back(orbital + external_count + active_count);
    }
    auto inputs = make_case(active, inactive, virtuals);
    populate_two_body(inputs);
    const sw::RegularizerOptions regularizer{0.8};
    // make_case's alpha and beta one-body blocks differ; equal blocks take the
    // spin-free path.
    auto spin_free_inputs = inputs;
    spin_free_inputs.one_body[1] = spin_free_inputs.one_body[0];
    for (const bool spin_free : {true, false})
      for (const auto& [mode, kind] :
           {std::pair{"discard", 0}, std::pair{"determinant", 1},
            std::pair{"dense", 3}}) {
        const auto& case_inputs = spin_free ? spin_free_inputs : inputs;
        const Eigen::MatrixXd density =
            kind == 0 ? Eigen::MatrixXd{} : make_density(case_inputs, kind);
        const auto timed_run = [&] {
          const auto begin = Clock::now();
          auto result = run_native(case_inputs, regularizer, density);
          const double milliseconds = elapsed(begin);
          return std::pair{std::move(result), milliseconds};
        };
        const auto [cold_result, first_ms] = timed_run();
        std::vector<double> timings;
        double checksum = 0.0;
        for (int repeat = 0; repeat < 3; ++repeat) {
          const auto [result, milliseconds] = timed_run();
          timings.push_back(milliseconds);
          checksum += result.scalar + result.one_body[0].squaredNorm() +
                      result.one_body[1].squaredNorm() +
                      result.two_body.squaredNorm();
          compare(result, cold_result);
        }
        EXPECT_TRUE(std::isfinite(checksum));
        std::sort(timings.begin(), timings.end());
        std::cout << std::fixed << std::setprecision(4)
                  << "full_kernel_active=" << active_count
                  << " external=" << external_count << " mode=" << mode
                  << " spin_free=" << spin_free << " sequant_ms=" << timings[1]
                  << " sequant_first_ms=" << first_ms
                  << " checksum=" << checksum << '\n';
      }
  }
}

TEST(Swpt2KernelTest, BareInverseUsesGuardedPseudoinverse) {
  const sw::RegularizerOptions bare;
  EXPECT_DOUBLE_EQ(sw::regularized_inverse(2.0, bare), 0.5);
  EXPECT_DOUBLE_EQ(sw::regularized_inverse(0.5e-8, bare), 0.0);
  EXPECT_DOUBLE_EQ(sw::regularized_inverse(sw::bare_denominator_floor, bare),
                   1e8);
  const sw::RegularizerOptions damped{1.0};
  EXPECT_DOUBLE_EQ(sw::regularized_inverse(0.0, damped), 0.0);
  EXPECT_NEAR(sw::regularized_inverse(1e-10, damped), 1e-10, 1e-24);
}

TEST(Swpt2KernelTest, RegularizedInverseSurvivesDenominatorCancellation) {
  for (double sigma2 : {1.0, 1e6})
    for (double delta = 1e-9; delta < 10.0; delta *= 1.3) {
      const long double expected =
          -std::expm1l(-static_cast<long double>(sigma2) * delta * delta) /
          delta;
      EXPECT_NEAR(
          sw::regularized_inverse(delta, sw::RegularizerOptions{sigma2}),
          static_cast<double>(expected), 1e-13 * std::abs(expected));
    }
}

TEST(Swpt2KernelTest, GeneralizedFockFromDefinition) {
  constexpr int count = 3;
  Eigen::MatrixXd one_body(count, count), density(count, count);
  one_body << -0.50, 0.10, 0.20, 0.10, -1.20, -0.15, 0.20, -0.15, 3.00;
  density << 1.0, 0.2, -0.1, 0.2, 2.0, 0.15, -0.1, 0.15, 0.4;
  Eigen::VectorXd integrals(count * count * count * count);
  for (int index = 0; index < integrals.size(); ++index)
    integrals(index) = 0.2 * std::sin(0.37 * index);
  const auto result = spin_free_fock(one_body, integrals, density, count);
  Eigen::MatrixXd expected = one_body;
  for (int row = 0; row < count; ++row)
    for (int col = 0; col < count; ++col)
      for (int first = 0; first < count; ++first)
        for (int second = 0; second < count; ++second)
          expected(row, col) +=
              density(first, second) *
              (integrals(sw::idx4(row, col, first, second, count)) -
               0.5 * integrals(sw::idx4(row, first, second, col, count)));
  EXPECT_LT((result - 0.5 * (expected + expected.transpose()).eval())
                .cwiseAbs()
                .maxCoeff(),
            1e-12);

  // Per spin: Coulomb with both densities, exchange with its own.
  std::array<Eigen::VectorXd, 3> blocks;
  for (int block = 0; block < 3; ++block) {
    blocks[block].resize(integrals.size());
    for (int index = 0; index < integrals.size(); ++index)
      blocks[block](index) = 0.2 * std::sin(0.37 * index + 0.4 * block);
  }
  const std::array<Eigen::MatrixXd, 2> spin_one_body{one_body, 1.3 * one_body};
  const std::array<Eigen::MatrixXd, 2> spin_density{0.6 * density,
                                                    0.4 * density.transpose()};
  const auto spin_result = sw::generalized_fock_matrices(
      spin_one_body,
      {std::cref(blocks[0]), std::cref(blocks[1]), std::cref(blocks[2])},
      spin_density, count);
  const auto chemist = [&](int left, int right, int p, int q, int r, int s) {
    if (left == right) return blocks[2 * left](sw::idx4(p, q, r, s, count));
    return left == 0 ? blocks[1](sw::idx4(p, q, r, s, count))
                     : blocks[1](sw::idx4(r, s, p, q, count));
  };
  for (int spin = 0; spin < 2; ++spin) {
    Eigen::MatrixXd spin_expected = spin_one_body[spin];
    for (int row = 0; row < count; ++row)
      for (int col = 0; col < count; ++col)
        for (int first = 0; first < count; ++first)
          for (int second = 0; second < count; ++second) {
            for (int other = 0; other < 2; ++other)
              spin_expected(row, col) +=
                  spin_density[other](first, second) *
                  chemist(spin, other, row, col, first, second);
            spin_expected(row, col) -=
                spin_density[spin](first, second) *
                chemist(spin, spin, row, first, second, col);
          }
    EXPECT_LT((spin_result[spin] -
               0.5 * (spin_expected + spin_expected.transpose()).eval())
                  .cwiseAbs()
                  .maxCoeff(),
              1e-12);
  }
  EXPECT_GT((spin_result[0] - spin_result[1]).cwiseAbs().maxCoeff(), 1e-3);
}

TEST(Swpt2KernelTest, InactiveFockFoldMeanField) {
  Eigen::MatrixXd one_body = Eigen::MatrixXd::Zero(2, 2);
  one_body.diagonal() << 0.3, -0.5;
  Eigen::VectorXd integrals = Eigen::VectorXd::Zero(16);
  integrals(sw::idx4(0, 0, 1, 1, 2)) = 1.0;
  integrals(sw::idx4(1, 1, 0, 0, 2)) = 1.0;
  integrals(sw::idx4(0, 1, 1, 0, 2)) = 0.2;
  integrals(sw::idx4(1, 0, 0, 1, 2)) = 0.2;
  integrals(sw::idx4(1, 1, 1, 1, 2)) = 0.5;
  const Eigen::VectorXd energies = one_body.diagonal();
  const auto result = shared_integral_downfold(one_body, one_body, integrals,
                                               energies, {0}, {1}, {}, {}, 0.7);
  EXPECT_NEAR(result.scalar, 0.7 - 1.0 + 0.5, 1e-12);
  for (const auto& block : result.one_body)
    EXPECT_NEAR(block(0, 0), 0.3 + 2.0 - 0.2, 1e-12);
}

TEST(Swpt2KernelTest, EmptyExternalSpaceIsIdentity) {
  Eigen::MatrixXd alpha(2, 2), beta(2, 2);
  alpha << 0.3, -0.4, -0.4, 0.9;
  beta = 1.1 * alpha;
  Eigen::VectorXd integrals(16), energies(2);
  energies << -0.5, 0.7;
  for (int index = 0; index < integrals.size(); ++index)
    integrals(index) = 0.03 * std::cos(index);
  const std::vector<int> active{1, 0};
  const auto result = shared_integral_downfold(alpha, beta, integrals, energies,
                                               active, {}, {}, {}, 0.7);
  EXPECT_DOUBLE_EQ(result.scalar, 0.7);
  for (int row = 0; row < 2; ++row)
    for (int col = 0; col < 2; ++col) {
      EXPECT_DOUBLE_EQ(result.one_body[0](row, col),
                       alpha(active[row], active[col]));
      EXPECT_DOUBLE_EQ(result.one_body[1](row, col),
                       beta(active[row], active[col]));
      for (int third = 0; third < 2; ++third)
        for (int fourth = 0; fourth < 2; ++fourth)
          EXPECT_DOUBLE_EQ(
              result.two_body(sw::idx4(row, col, third, fourth, 2)),
              integrals(sw::idx4(active[row], active[col], active[third],
                                 active[fourth], 2)));
    }
  EXPECT_TRUE(std::isinf(result.min_denominator));
  EXPECT_DOUBLE_EQ(result.max_amplitude, 0.0);
}

TEST(Swpt2KernelTest, SecondOrderLevelShifts) {
  for (const bool virtual_external : {false, true}) {
    Eigen::MatrixXd one_body(2, 2);
    const double active_energy = virtual_external ? -0.5 : 3.0;
    const double external_energy = virtual_external ? 3.0 : -0.5;
    one_body << active_energy, 0.2, 0.2, external_energy;
    const Eigen::VectorXd energies = one_body.diagonal();
    const std::vector<int> inactive =
        virtual_external ? std::vector<int>{} : std::vector<int>{1};
    const std::vector<int> virtuals =
        virtual_external ? std::vector<int>{1} : std::vector<int>{};
    const auto result =
        shared_integral_downfold(one_body, one_body, Eigen::VectorXd::Zero(16),
                                 energies, {0}, inactive, virtuals, {}, 0.0);
    for (const auto& block : result.one_body)
      EXPECT_NEAR(block(0, 0),
                  active_energy + 0.04 / (active_energy - external_energy),
                  1e-12);
  }
}

TEST(Swpt2KernelTest, PartitionWindowRoundsFoldedOccupations) {
  const std::vector<std::size_t> window{10, 11, 12, 13};
  const std::unordered_set<std::size_t> kept{11, 12};
  const auto integer =
      sw::partition_window({2.0, 1.0, 1.0, 0.0}, window, kept, 4, 0.5);
  EXPECT_EQ(integer.active_spatial, (std::vector<int>{1, 2}));
  EXPECT_EQ(integer.inactive_spatial, (std::vector<int>{0}));
  EXPECT_EQ(integer.virtual_spatial, (std::vector<int>{3}));
  EXPECT_EQ(integer.active_electrons, 2);
  EXPECT_DOUBLE_EQ(integer.worst_deviation, 0.0);
  EXPECT_DOUBLE_EQ(integer.folded_charge_error, 0.0);
  const auto fractional =
      sw::partition_window({1.98, 1.0, 1.0, 0.02}, window, kept, 4, 0.5);
  EXPECT_EQ(fractional.inactive_spatial, (std::vector<int>{0}));
  EXPECT_EQ(fractional.virtual_spatial, (std::vector<int>{3}));
  EXPECT_EQ(fractional.active_electrons, 2);
  EXPECT_NEAR(fractional.worst_deviation, 0.02, 1e-12);
  EXPECT_EQ(fractional.worst_orbital, 10u);
  EXPECT_NEAR(fractional.worst_occupation, 1.98, 1e-12);
  EXPECT_NEAR(fractional.folded_charge_error, 0.0, 1e-12);
}

TEST(Swpt2KernelTest, PartitionWindowChargeErrorAccumulates) {
  const auto result =
      sw::partition_window({1.9, 1.9, 1.9, 1.3}, {0, 1, 2, 3}, {3}, 7, 0.5);
  EXPECT_EQ(result.inactive_spatial, (std::vector<int>{0, 1, 2}));
  EXPECT_TRUE(result.virtual_spatial.empty());
  EXPECT_NEAR(result.worst_deviation, 0.1, 1e-12);
  EXPECT_NEAR(result.folded_charge_error, 0.3, 1e-12);
  EXPECT_EQ(result.active_electrons, 1);
}

TEST(Swpt2KernelTest, PartitionWindowRejectsAmbiguousFolds) {
  const std::vector<std::size_t> window{0, 1, 2, 3};
  const std::unordered_set<std::size_t> kept{1, 2};
  EXPECT_THROW(sw::partition_window({1.0, 1.0, 1.0, 1.0}, window, kept, 4, 0.5),
               std::invalid_argument);
  EXPECT_NO_THROW(
      sw::partition_window({2.0, 1.0, 1.0, 0.0}, window, kept, 4, 0.0));
  EXPECT_THROW(
      sw::partition_window({1.98, 1.0, 1.0, 0.02}, window, kept, 4, 0.0),
      std::invalid_argument);
  for (double tolerance : {-0.1, 1.0})
    EXPECT_THROW(
        sw::partition_window({2.0, 1.0, 1.0, 0.0}, window, kept, 4, tolerance),
        std::invalid_argument);
  EXPECT_THROW(sw::partition_window({2.0, 0.0}, {0, 1}, {}, 2, 0.5),
               std::invalid_argument);
  EXPECT_THROW(sw::partition_window({2.0, 0.0}, {0, 1, 2}, {0}, 2, 0.5),
               std::invalid_argument);
  for (int electrons : {1, 5})
    EXPECT_THROW(sw::partition_window({2.0, 0.0}, {0, 1}, {1}, electrons, 0.5),
                 std::invalid_argument);
}

TEST(Swpt2KernelTest, SemicanonicalPrimitivesAreCovariantAndReversible) {
  constexpr int count = 4;
  Eigen::MatrixXd one_body(count, count);
  one_body << -1.2, 0.08, -0.03, 0.02, 0.08, -0.4, 0.11, -0.01, -0.03, 0.11,
      0.5, 0.07, 0.02, -0.01, 0.07, 1.4;
  Eigen::VectorXd integrals(count * count * count * count);
  for (int index = 0; index < integrals.size(); ++index)
    integrals(index) = 0.02 * std::sin(0.37 * index + 0.2);
  Eigen::MatrixXd density = Eigen::MatrixXd::Zero(count, count);
  density.diagonal() << 2.0, 1.4, 0.6, 0.0;
  density(1, 2) = density(2, 1) = 0.15;
  Eigen::MatrixXd basis = Eigen::MatrixXd::Identity(count, count);
  basis(1, 1) = basis(2, 2) = std::cos(0.37);
  basis(1, 2) = -std::sin(0.37);
  basis(2, 1) = std::sin(0.37);
  const auto fock = spin_free_fock(one_body, integrals, density, count);
  const auto rotated =
      spin_free_fock(sw::rotate_one_body(one_body, basis),
                     sw::rotate_two_body(integrals, basis, count),
                     sw::rotate_one_body(density, basis), count);
  EXPECT_LT((rotated - basis.transpose() * fock * basis).cwiseAbs().maxCoeff(),
            1e-11);
  const auto semicanonical =
      sw::semicanonical_rotation(rotated, {{0}, {1, 2}, {3}}, 1e-14);
  const auto diagonal = sw::rotate_one_body(rotated, semicanonical);
  EXPECT_NEAR(diagonal(1, 2), 0.0, 1e-12);
  EXPECT_LT((semicanonical.transpose() * semicanonical -
             Eigen::MatrixXd::Identity(count, count))
                .cwiseAbs()
                .maxCoeff(),
            1e-12);
  EXPECT_LT((sw::rotate_one_body(sw::rotate_one_body(one_body, basis),
                                 basis.transpose()) -
             one_body)
                .cwiseAbs()
                .maxCoeff(),
            1e-12);
  EXPECT_LT((sw::rotate_two_body(sw::rotate_two_body(integrals, basis, count),
                                 basis.transpose(), count) -
             integrals)
                .cwiseAbs()
                .maxCoeff(),
            1e-12);
  EXPECT_EQ(sw::semicanonical_rotation(diagonal, {{0}, {1, 2}, {3}}, 1e-10),
            Eigen::MatrixXd::Identity(count, count));

  // aabb integrals rotate their (pq) and (rs) pairs separately.
  const Eigen::MatrixXd second = semicanonical * basis;
  const auto mixed = sw::rotate_two_body(integrals, basis, second, count);
  double mixed_error = 0.0;
  for (int p = 0; p < count; ++p)
    for (int q = 0; q < count; ++q)
      for (int r = 0; r < count; ++r)
        for (int s = 0; s < count; ++s) {
          double expected = 0.0;
          for (int a = 0; a < count; ++a)
            for (int b = 0; b < count; ++b)
              for (int c = 0; c < count; ++c)
                for (int d = 0; d < count; ++d)
                  expected += basis(a, p) * basis(b, q) * second(c, r) *
                              second(d, s) *
                              integrals(sw::idx4(a, b, c, d, count));
          mixed_error =
              std::max(mixed_error,
                       std::abs(mixed(sw::idx4(p, q, r, s, count)) - expected));
        }
  EXPECT_LT(mixed_error, 1e-14);
}

}  // namespace
