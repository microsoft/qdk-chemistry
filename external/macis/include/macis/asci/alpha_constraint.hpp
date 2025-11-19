/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <macis/wfn/raw_bitset.hpp>
namespace macis {

/**
 * @brief Template class representing alpha spin constraints for wavefunction
 * states
 *
 * This class manages constraints on alpha spin orbitals, providing
 * functionality to check if states satisfy specific constraint conditions and
 * to compute overlaps and symmetric differences.
 *
 * @tparam WfnTraits Wavefunction traits type defining the wavefunction
 * characteristics
 */
template <typename WfnTraits>
class alpha_constraint {
 public:
  /// @brief Alias for the wavefunction traits template parameter
  using wfn_traits = WfnTraits;
  /// @brief Wavefunction type extracted from the traits
  using wfn_type = typename WfnTraits::wfn_type;
  /// @brief Spin-specific wavefunction type derived from wfn_type
  using spin_wfn_type = spin_wfn_t<wfn_type>;

  /// @brief Type alias for constraint patterns, same as spin_wfn_type
  using constraint_type = spin_wfn_type;
  /// @brief Traits type for constraint operations and bit counting
  using constraint_traits = wavefunction_traits<spin_wfn_type>;

 private:
  /// @brief Constraint pattern defining the required orbital occupations
  constraint_type C_;
  /// @brief Boundary mask pattern for constraint validation
  constraint_type B_;
  /// @brief Lowest orbital index set in the constraint mask
  uint32_t C_min_;
  /// @brief Precomputed count of set bits in the constraint pattern C_
  uint32_t count_;

 public:
  /**
   * @brief Constructor for alpha_constraint
   *
   * @param C Constraint pattern for alpha spin orbitals
   * @param B Boundary or mask pattern
   * @param C_min Lowest index set in the constraint mask
   */
  alpha_constraint(constraint_type C, constraint_type B, uint32_t C_min)
      : C_(C), B_(B), C_min_(C_min), count_(constraint_traits::count(C)) {}

  /// @brief Default copy constructor
  alpha_constraint(const alpha_constraint&) = default;
  /// @brief Default copy assignment operator
  alpha_constraint& operator=(const alpha_constraint&) = default;

  /// @brief Default move constructor
  alpha_constraint(alpha_constraint&& other) noexcept = default;
  /// @brief Default move assignment operator
  alpha_constraint& operator=(alpha_constraint&&) noexcept = default;

  /**
   * @brief Get the constraint pattern C
   * @return The constraint pattern
   */
  inline auto C() const { return C_; }

  /**
   * @brief Get the boundary/mask pattern B
   * @return The boundary pattern
   */
  inline auto B() const { return B_; }

  /**
   * @brief Get the minimum constraint index
   * @return The lowest index set in the constraint mask
   */
  inline auto C_min() const { return C_min_; }

  /**
   * @brief Get the count of set bits in the constraint pattern
   * @return The count value
   */
  inline auto count() const { return count_; }

  /**
   * @brief Compute the intersection of a state with the constraint mask C
   * @param state The input spin wavefunction state
   * @return The result of bitwise AND between state and C
   */
  inline spin_wfn_type c_mask_union(spin_wfn_type state) const {
    return state & C_;
  }

  /**
   * @brief Compute the intersection of a state with the boundary mask B
   * @param state The input spin wavefunction state
   * @return The result of bitwise AND between state and B
   */
  inline spin_wfn_type b_mask_union(spin_wfn_type state) const {
    return state & B_;
  }

  /**
   * @brief Compute the symmetric difference between a state and constraint C
   * @param state The input spin wavefunction state
   * @return The result of bitwise XOR between state and C
   */
  inline spin_wfn_type symmetric_difference(spin_wfn_type state) const {
    return state ^ C_;
  }

  /**
   * @brief Compute the overlap between a state and the constraint
   *
   * Calculates the number of bits that overlap between the given state
   * and the constraint mask C.
   *
   * @tparam WfnType Type of the wavefunction state
   * @param state The input wavefunction state
   * @return The count of overlapping bits
   */
  template <typename WfnType>
  inline auto overlap(WfnType state) const {
    return constraint_traits::count(c_mask_union(state));
  }

  /**
   * @brief Check if a state satisfies the constraint conditions
   *
   * A state satisfies the constraint if:
   * 1. The overlap equals the total count of constraint bits
   * 2. The symmetric difference shifted by C_min has no set bits
   *
   * @tparam WfnType Type of the wavefunction state
   * @param state The input wavefunction state
   * @return True if the state satisfies the constraint, false otherwise
   */
  template <typename WfnType>
  inline bool satisfies_constraint(WfnType state) const {
    return overlap(state) == count_ and
           constraint_traits::count(symmetric_difference(state) >> C_min_) == 0;
  }

  /**
   * @brief Factory method to create a triplet constraint
   *
   * Creates an alpha constraint for a triplet configuration with three
   * specific orbital indices.
   *
   * @param i First orbital index
   * @param j Second orbital index
   * @param k Third orbital index (also used as boundary)
   * @return A new alpha_constraint instance configured for the triplet
   */
  static alpha_constraint make_triplet(unsigned i, unsigned j, unsigned k) {
    constraint_type C = 0;
    C.flip(i).flip(j).flip(k);
    constraint_type B = full_mask<B.size()>(k);
    return alpha_constraint(C, B, k);
  }
};

}  // namespace macis
