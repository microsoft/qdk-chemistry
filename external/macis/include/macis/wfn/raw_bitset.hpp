/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <macis/bitset_operations.hpp>
#include <macis/types.hpp>

namespace macis {

enum class Spin { Alpha, Beta };

template <typename WfnType>
struct wavefunction_traits;

template <typename WfnType>
using spin_wfn_t = typename wavefunction_traits<WfnType>::spin_wfn_type;

template <size_t N>
struct wavefunction_traits<std::bitset<N>> {
  using wfn_type = std::bitset<N>;
  using spin_wfn_type = std::bitset<N / 2>;
  using orbidx_type = uint32_t;
  using orbidx_container = std::vector<orbidx_type>;

  inline static constexpr size_t bit_size = N;

  static constexpr auto size() { return bit_size; }

  static inline auto count(wfn_type state) { return state.count(); }

  static inline spin_wfn_type alpha_string(wfn_type state) {
    return bitset_lo_word(state);
  }

  static inline spin_wfn_type beta_string(wfn_type state) {
    return bitset_hi_word(state);
  }

  using wfn_comparator = bitset_less_comparator<N>;

  struct spin_comparator {
    using spin_wfn_comparator = bitset_less_comparator<N / 2>;
    bool operator()(wfn_type x, wfn_type y) const {
      auto s_comp = spin_wfn_comparator{};
      const auto x_a = alpha_string(x);
      const auto y_a = alpha_string(y);
      if (x_a == y_a) {
        const auto x_b = beta_string(x);
        const auto y_b = beta_string(y);
        return s_comp(x_b, y_b);
      } else
        return s_comp(x_a, y_a);
    }
  };

  template <Spin Sigma = Spin::Alpha>
  static inline wfn_type from_spin(spin_wfn_type alpha, spin_wfn_type beta) {
    if constexpr (Sigma == Spin::Alpha) {
      auto alpha_expand = expand_bitset<N>(alpha);
      auto beta_expand = expand_bitset<N>(beta) << N / 2;
      return alpha_expand | beta_expand;
    } else {
      auto alpha_expand = expand_bitset<N>(alpha) << N / 2;
      auto beta_expand = expand_bitset<N>(beta);
      return alpha_expand | beta_expand;
    }
  }

  static inline wfn_type canonical_hf_determinant(uint32_t nalpha,
                                                  uint32_t nbeta) {
    spin_wfn_type alpha = full_mask<N / 2>(nalpha);
    spin_wfn_type beta = full_mask<N / 2>(nbeta);
    return from_spin(alpha, beta);
  }

  template <Spin Sigma, typename... Inds>
  static inline wfn_type& flip_bits(wfn_type& state, Inds&&...);

  template <Spin Sigma>
  static inline wfn_type& flip_bits(wfn_type& state) {
    return state;
  }

  template <Spin Sigma, typename... Inds>
  static inline wfn_type& flip_bits(wfn_type& state, unsigned p,
                                    Inds&&... inds) {
    return flip_bits<Sigma>(state.flip(p + (Sigma == Spin::Alpha ? 0 : N / 2)),
                            std::forward<Inds>(inds)...);
  }

  template <Spin Sigma = Spin::Alpha>
  static inline wfn_type create_no_check(wfn_type state, unsigned p) {
    flip_bits<Sigma>(state, p);
    return state;
  }

  template <Spin Sigma = Spin::Alpha>
  static inline wfn_type single_excitation_no_check(wfn_type state, unsigned p,
                                                    unsigned q) {
    flip_bits<Sigma>(state, p, q);
    return state;
  }

  template <Spin Sigma = Spin::Alpha>
  static inline wfn_type double_excitation_no_check(wfn_type state, unsigned p,
                                                    unsigned q, unsigned r,
                                                    unsigned s) {
    flip_bits<Sigma>(state, p, q, r, s);
    return state;
  }

  static inline void state_to_occ(wfn_type state, orbidx_container& occ) {
    occ = bits_to_indices(state);
  }

  static inline orbidx_container state_to_occ(wfn_type state) {
    return bits_to_indices(state);
  }

  static inline void state_to_occ_vir(size_t norb, wfn_type state,
                                      orbidx_container& occ,
                                      orbidx_container& vir) {
    state_to_occ(state, occ);
    const auto num_occupied_orbitals = occ.size();
    assert(num_occupied_orbitals <= norb);

    const auto num_virtual_orbitals = norb - num_occupied_orbitals;
    vir.resize(num_virtual_orbitals);
    state = ~state;
    for (int i = 0; i < num_virtual_orbitals; ++i) {
      auto a = ffs(state) - 1;
      vir[i] = a;
      state.flip(a);
    }
  }
};

}  // namespace macis
