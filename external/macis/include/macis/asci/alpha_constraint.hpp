/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <macis/wfn/raw_bitset.hpp>
namespace macis {

template <typename WfnTraits>
class alpha_constraint {
 public:
  using wfn_traits = WfnTraits;
  using wfn_type = typename WfnTraits::wfn_type;
  using spin_wfn_type = spin_wfn_t<wfn_type>;

  using constraint_type = spin_wfn_type;
  using constraint_traits = wavefunction_traits<spin_wfn_type>;

 private:
  constraint_type C_;
  constraint_type B_;
  uint32_t C_min_;
  uint32_t count_;

 public:
  alpha_constraint(constraint_type C, constraint_type B, uint32_t C_min)
      : C_(C), B_(B), C_min_(C_min), count_(constraint_traits::count(C)) {}

  alpha_constraint(const alpha_constraint&) = default;
  alpha_constraint& operator=(const alpha_constraint&) = default;

  alpha_constraint(alpha_constraint&& other) noexcept = default;
  alpha_constraint& operator=(alpha_constraint&&) noexcept = default;

  inline auto C() const { return C_; }
  inline auto B() const { return B_; }
  inline auto C_min() const { return C_min_; }
  inline auto count() const { return count_; }

  inline spin_wfn_type c_mask_union(spin_wfn_type state) const {
    return state & C_;
  }
  inline spin_wfn_type b_mask_union(spin_wfn_type state) const {
    return state & B_;
  }

  inline spin_wfn_type symmetric_difference(spin_wfn_type state) const {
    return state ^ C_;
  }
  // inline spin_wfn_type symmetric_difference(wfn_type state) const {
  //   return symmetric_difference(wfn_traits::alpha_string(state));
  // }

  template <typename WfnType>
  inline auto overlap(WfnType state) const {
    return constraint_traits::count(c_mask_union(state));
  }

  template <typename WfnType>
  inline bool satisfies_constraint(WfnType state) const {
    return overlap(state) == count_ and
           constraint_traits::count(symmetric_difference(state) >> C_min_) == 0;
  }

  static alpha_constraint make_triplet(unsigned i, unsigned j, unsigned k) {
    constraint_type C = 0;
    C.flip(i).flip(j).flip(k);
    // constraint_type B = 1;
    // static_assert(B.size() <= 64, "ULLONG NOT POSSIBLE HERE");
    // B <<= k;
    // B = B.to_ullong() - 1;
    constraint_type B = full_mask<B.size()>(k);
    return alpha_constraint(C, B, k);
  }
};

}  // namespace macis
