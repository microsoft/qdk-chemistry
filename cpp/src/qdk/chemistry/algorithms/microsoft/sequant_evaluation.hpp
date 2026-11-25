// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <SeQuant/core/expr_fwd.hpp>
#include <array>
#include <cstddef>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft::sequant_evaluation {

/// Spin orbitals of a single-determinant reference, by spin (0 = alpha).
struct SpinOrbitalSpace {
  std::array<std::vector<std::size_t>, 2> occupied, virtuals;
  /// Spin orbitals allowed on residual-operator indices, in output order.
  std::vector<std::size_t> target;
};

/// Element of leaf tensor `label` at the spin orbitals of its slots, bra first;
/// may be called concurrently.
using LeafElement = std::function<double(std::wstring_view label,
                                         std::span<const std::size_t> slots)>;

/// Coefficients over the n target spin orbitals: one_body[p * n + q] multiplies
/// {a+_p a_q} and two_body[((p * n + q) * n + r) * n + s] multiplies
/// {a+_p a+_q a_s a_r}, as accumulated (not antisymmetrized).
struct NormalOrderedOperator {
  double scalar = 0.0;
  std::vector<double> one_body, two_body;
};

/**
 * @brief Numerically evaluate a reference-normal-ordered operator expression.
 *
 * @param expression Spin-orbital sum of terms over the occupied (i), virtual
 *        (a) and general (p) spaces of sequant::mbpt::make_min_sr_spaces. Each
 *        term is a product of tensors and constants with at most one
 *        number-conserving normal operator of rank one or two, and each of its
 *        indices occurs once in each of two tensors, or once in a tensor and
 *        once in the operator.
 * @param space Reference occupations and target spin orbitals.
 * @param leaf Elements of the expression's tensors; exceptions it throws
 *        propagate.
 * @return The scalar, one-body and two-body coefficients over the target.
 * @throws std::invalid_argument If the expression has another form, a spin
 *         orbital appears more than once in the occupied and virtual lists, or
 *         a target spin orbital is repeated or not listed.
 */
NormalOrderedOperator evaluate(const sequant::ExprPtr& expression,
                               const SpinOrbitalSpace& space,
                               const LeafElement& leaf);

}  // namespace qdk::chemistry::algorithms::microsoft::sequant_evaluation
