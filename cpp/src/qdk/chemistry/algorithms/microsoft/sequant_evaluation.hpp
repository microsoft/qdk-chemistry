// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <SeQuant/core/expr_fwd.hpp>
#include <array>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <span>
#include <string_view>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft::sequant_evaluation {

/// Number of elements of a tensor; throws std::length_error if unaddressable.
std::size_t tensor_size(std::span<const std::size_t> extents);

inline std::size_t tensor_size(std::initializer_list<std::size_t> extents) {
  return tensor_size(std::span(extents.begin(), extents.size()));
}

/// Row-major offset of element (p, q, r, s) of an n x n x n x n tensor.
inline std::size_t index4(std::size_t p, std::size_t q, std::size_t r,
                          std::size_t s, std::size_t n) {
  return ((p * n + q) * n + r) * n + s;
}

/// Spin orbitals of a single-determinant reference, by spin (0 = alpha).
struct SpinOrbitalSpace {
  std::array<std::vector<std::size_t>, 2> occupied, virtuals;
  /// Spin orbitals allowed on residual-operator indices, in output order.
  std::vector<std::size_t> target;
};

/// Element of leaf tensor `label` at the spin orbitals of its slots, bra first.
using LeafElement = std::function<double(std::wstring_view label,
                                         std::span<const std::size_t> slots)>;

/// Coefficients over the n target spin orbitals: one_body[p * n + q] multiplies
/// {a+_p a_q} and two_body[index4(p, q, r, s, n)] multiplies
/// {a+_p a+_q a_s a_r}.
struct NormalOrderedOperator {
  double scalar = 0.0;
  std::vector<double> one_body, two_body;
};

/**
 * @brief Numerically evaluate a reference-normal-ordered operator expression.
 *
 * @param expression Spin-orbital sum of products of tensors and at most one
 *        number-conserving normal operator of rank one or two, over the
 *        occupied (i), virtual (a) and general (p) spaces of
 *        sequant::mbpt::make_min_sr_spaces.
 * @param space Reference occupations and target spin orbitals.
 * @param leaf Elements of the expression's tensors.
 * @return The scalar, one-body and two-body coefficients over the target.
 */
NormalOrderedOperator evaluate(const sequant::ExprPtr& expression,
                               const SpinOrbitalSpace& space,
                               const LeafElement& leaf);

}  // namespace qdk::chemistry::algorithms::microsoft::sequant_evaluation
