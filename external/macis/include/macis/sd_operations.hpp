/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <algorithm>
#include <cassert>
#include <macis/bitset_operations.hpp>
#include <macis/wfn/raw_bitset.hpp>
#include <numeric>
#include <tuple>

namespace macis {

/**
 *  @brief Generate canonical HF determinant.
 *
 *  Generates a string representation of the canonical HF determinant
 *  consisting of a specifed number of alpha and beta orbitals.
 *
 *  @tparam N Number of bits for the total bit string of the state
 *  @param[in] nalpha Number of occupied alpha orbitals in the HF state
 *  @param[in] nbeta  Number of occupied beta orbitals in the HF state
 *
 *  @returns The bitstring HF state consisting of the specified number of
 *    occupied orbitals.
 */
// template <size_t N>
// std::bitset<N> canonical_hf_determinant(uint32_t nalpha, uint32_t nbeta) {
//   static_assert((N % 2) == 0, "N Must Be Even");
//   std::bitset<N> alpha = full_mask<N>(nalpha);
//   std::bitset<N> beta = full_mask<N>(nbeta) << (N / 2);
//   return alpha | beta;
// }

/**
 *  @brief Generate the list of (un)occupied orbitals for a paricular state.
 *
 *  @tparam N Number of bits for the total bit string of the state
 *  @param[in]  norb   Number of orbitals used to describe the state (<= `N`)
 *  @param[in]  state  The state from which to determine orbital occupations.
 *  @param[out] occ    List of occupied orbitals in `state`
 *  @param[out] vir    List of unoccupied orbitals in `state`
 */
// template <size_t N>
// void bitset_to_occ_vir(size_t norb, std::bitset<N> state,
//                        std::vector<uint32_t>& occ, std::vector<uint32_t>&
//                        vir) {
//   occ = bits_to_indices(state);
//   const auto nocc = occ.size();
//   assert(nocc < norb);
//
//   const auto nvir = norb - nocc;
//   vir.resize(nvir);
//   state = ~state;
//   for(int i = 0; i < nvir; ++i) {
//     auto a = ffs(state) - 1;
//     vir[i] = a;
//     state.flip(a);
//   }
// }

// template <size_t N>
// auto single_excitation(std::bitset<N> state, unsigned p, unsigned q) {
//   return state.flip(p).flip(q);
// }

// template <Spin Sigma, size_t N>
// auto single_excitation_spin(std::bitset<N> state, unsigned p, unsigned q) {
//   static_assert(N%2 == 0, "Num Bits Must Be Even");
//   if constexpr (Sigma == Spin::Alpha)
//     return single_excitation(state,p,q);
//   else
//     return single_excitation(state,p+N/2,q+N/2);
// }

// template <size_t N>
// auto double_excitation(std::bitset<N> state, unsigned p, unsigned q, unsigned
// r, unsigned s) {
//   return state.flip(p).flip(q).flip(r).flip(s);
// }

// template <Spin Sigma, size_t N>
// auto double_excitation_spin(std::bitset<N> state, unsigned p, unsigned q,
// unsigned r, unsigned s) {
//   static_assert(N%2 == 0, "Num Bits Must Be Even");
//   if constexpr (Sigma == Spin::Alpha)
//     return double_excitation(state,p,q,r,s);
//   else
//     return double_excitation(state,p+N/2,q+N/2,r+N/2,s+N/2);
// }

// TODO: Test this function
template <size_t N>
uint32_t first_occupied_flipped(std::bitset<N> state, std::bitset<N> ex) {
  return ffs(state & ex) - 1u;
}

// TODO: Test this function
template <size_t N>
double single_excitation_sign(std::bitset<N> state, unsigned p, unsigned q) {
  std::bitset<N> mask = 0ul;

  if (p > q) {
    mask = state & (full_mask<N>(p) ^ full_mask<N>(q + 1));
  } else {
    mask = state & (full_mask<N>(q) ^ full_mask<N>(p + 1));
  }
  return (mask.count() % 2) ? -1. : 1.;
}

template <typename WfnType, typename WfnContainer>
void append_singles(WfnType state, const std::vector<uint32_t>& occ,
                    const std::vector<uint32_t>& vir, WfnContainer& singles) {
  using wfn_traits = wavefunction_traits<WfnType>;
  const size_t nocc = occ.size();
  const size_t nvir = vir.size();

  singles.clear();
  singles.reserve(nocc * nvir);

  for (size_t a = 0; a < nvir; ++a)
    for (size_t i = 0; i < nocc; ++i) {
      singles.emplace_back(
          wfn_traits::single_excitation_no_check(state, occ[i], vir[a]));
    }
}

template <typename WfnType, typename WfnContainer>
void append_doubles(WfnType state, const std::vector<uint32_t>& occ,
                    const std::vector<uint32_t>& vir, WfnContainer& doubles) {
  using wfn_traits = wavefunction_traits<WfnType>;
  const size_t nocc = occ.size();
  const size_t nvir = vir.size();

  doubles.clear();
  const size_t nv2 = (nvir * (nvir - 1)) / 2;
  const size_t no2 = (nocc * (nocc - 1)) / 2;
  doubles.reserve(nv2 * no2);

  for (size_t a = 0; a < nvir; ++a)
    for (size_t i = 0; i < nocc; ++i)
      for (size_t b = a + 1; b < nvir; ++b)
        for (size_t j = i + 1; j < nocc; ++j) {
          doubles.emplace_back(wfn_traits::double_excitation_no_check(
              state, occ[i], occ[j], vir[a], vir[b]));
        }
}

template <typename WfnType, typename WfnContainer>
void generate_singles(size_t norb, WfnType state, WfnContainer& singles) {
  using wfn_traits = wavefunction_traits<WfnType>;
  std::vector<uint32_t> occ_orbs, vir_orbs;
  wfn_traits::state_to_occ_vir(norb, state, occ_orbs, vir_orbs);

  singles.clear();
  append_singles(state, occ_orbs, vir_orbs, singles);
}

template <typename WfnType, typename WfnContainer>
void generate_doubles(size_t norb, WfnType state, WfnContainer& doubles) {
  using wfn_traits = wavefunction_traits<WfnType>;
  std::vector<uint32_t> occ_orbs, vir_orbs;
  wfn_traits::state_to_occ_vir(norb, state, occ_orbs, vir_orbs);

  doubles.clear();
  append_doubles(state, occ_orbs, vir_orbs, doubles);
}

template <typename WfnType, typename WfnContainer>
void generate_singles_doubles(size_t norb, WfnType state, WfnContainer& singles,
                              WfnContainer& doubles) {
  using wfn_traits = wavefunction_traits<WfnType>;
  std::vector<uint32_t> occ_orbs, vir_orbs;
  wfn_traits::state_to_occ_vir(norb, state, occ_orbs, vir_orbs);

  singles.clear();
  doubles.clear();
  append_singles(state, occ_orbs, vir_orbs, singles);
  append_doubles(state, occ_orbs, vir_orbs, doubles);
}

template <typename WfnType, typename WfnContainer>
void generate_singles_spin(size_t norb, WfnType state, WfnContainer& singles) {
  using wfn_traits = wavefunction_traits<WfnType>;

  auto state_alpha = wfn_traits::alpha_string(state);
  auto state_beta = wfn_traits::beta_string(state);

  using spin_wfn_type = spin_wfn_t<WfnType>;
  std::vector<spin_wfn_type> singles_alpha, singles_beta;

  // Generate Spin-Specific singles / doubles
  generate_singles(norb, state_alpha, singles_alpha);
  generate_singles(norb, state_beta, singles_beta);

  // Generate Singles in full space
  singles.clear();

  // Single Alpha + No Beta
  for (auto s_alpha : singles_alpha) {
    singles.emplace_back(wfn_traits::from_spin(s_alpha, state_beta));
  }

  // No Alpha + Single Beta
  for (auto s_beta : singles_beta) {
    singles.emplace_back(wfn_traits::from_spin(state_alpha, s_beta));
  }
}

template <typename WfnType, typename WfnContainer>
void generate_singles_doubles_spin(size_t norb, WfnType state,
                                   WfnContainer& singles,
                                   WfnContainer& doubles) {
  using wfn_traits = wavefunction_traits<WfnType>;

  auto state_alpha = wfn_traits::alpha_string(state);
  auto state_beta = wfn_traits::beta_string(state);

  using spin_wfn_type = spin_wfn_t<WfnType>;
  std::vector<spin_wfn_type> singles_alpha, singles_beta;
  std::vector<spin_wfn_type> doubles_alpha, doubles_beta;

  // Generate Spin-Specific singles / doubles
  generate_singles_doubles(norb, state_alpha, singles_alpha, doubles_alpha);
  generate_singles_doubles(norb, state_beta, singles_beta, doubles_beta);

  // Generate Singles in full space
  singles.clear();

  // Single Alpha + No Beta
  for (auto s_alpha : singles_alpha) {
    singles.emplace_back(wfn_traits::from_spin(s_alpha, state_beta));
  }

  // No Alpha + Single Beta
  for (auto s_beta : singles_beta) {
    singles.emplace_back(wfn_traits::from_spin(state_alpha, s_beta));
  }

  // Generate Doubles in full space
  doubles.clear();

  // Double Alpha + No Beta
  for (auto d_alpha : doubles_alpha) {
    doubles.emplace_back(wfn_traits::from_spin(d_alpha, state_beta));
  }

  // No Alpha + Double Beta
  for (auto d_beta : doubles_beta) {
    doubles.emplace_back(wfn_traits::from_spin(state_alpha, d_beta));
  }

  // Single Alpha + Single Beta
  for (auto s_alpha : singles_alpha)
    for (auto s_beta : singles_beta) {
      doubles.emplace_back(wfn_traits::from_spin(s_alpha, s_beta));
    }
}

template <typename WfnType, typename WfnContainer>
void generate_cisd_hilbert_space(size_t norb, WfnType state,
                                 WfnContainer& dets) {
  dets.clear();
  dets.emplace_back(state);
  std::vector<WfnType> singles, doubles;
  generate_singles_doubles_spin(norb, state, singles, doubles);
  dets.insert(dets.end(), singles.begin(), singles.end());
  dets.insert(dets.end(), doubles.begin(), doubles.end());
}

template <typename WfnType>
auto generate_cisd_hilbert_space(size_t norb, WfnType state) {
  std::vector<WfnType> dets;
  generate_cisd_hilbert_space(norb, state, dets);
  return dets;
}

template <typename WfnType>
std::vector<WfnType> generate_combs(uint64_t nbits, uint64_t nset) {
  using wfn_traits = wavefunction_traits<WfnType>;
  std::vector<bool> v(nbits, false);
  std::fill_n(v.begin(), nset, true);
  std::vector<WfnType> store;

  do {
    WfnType temp(0ul);
    for (uint64_t i = 0; i < nbits; ++i)
      if (v[i]) {
        temp = wfn_traits::create_no_check(temp, i);
      }
    store.emplace_back(temp);

  } while (std::prev_permutation(v.begin(), v.end()));

  return store;
}

template <typename WfnType>
std::vector<WfnType> generate_hilbert_space(size_t norbs, size_t nalpha,
                                            size_t nbeta) {
  using spin_wfn_type = spin_wfn_t<WfnType>;
  using wfn_traits = wavefunction_traits<WfnType>;

  // Get all alpha and beta combs
  auto alpha_dets = generate_combs<spin_wfn_type>(norbs, nalpha);
  auto beta_dets = generate_combs<spin_wfn_type>(norbs, nbeta);

  std::vector<WfnType> states;
  states.reserve(alpha_dets.size() * beta_dets.size());
  for (auto alpha_det : alpha_dets)
    for (auto beta_det : beta_dets) {
      states.emplace_back(wfn_traits::from_spin(alpha_det, beta_det));
    }

  return states;
}

template <typename WfnType, typename WfnContainer>
void generate_cis_hilbert_space(size_t norb, WfnType state,
                                WfnContainer& dets) {
  dets.clear();
  dets.emplace_back(state);
  std::vector<WfnType> singles;
  generate_singles_spin(norb, state, singles);
  dets.insert(dets.end(), singles.begin(), singles.end());
}

template <typename WfnType>
std::vector<WfnType> generate_cis_hilbert_space(size_t norb, WfnType state) {
  std::vector<WfnType> dets;
  generate_cis_hilbert_space(norb, state, dets);
  return dets;
}

// TODO: Test this function
template <typename WfnType>
inline auto single_excitation_sign_indices(WfnType bra, WfnType ket,
                                           WfnType ex) {
  auto o1 = first_occupied_flipped(ket, ex);
  auto v1 = first_occupied_flipped(bra, ex);
  auto sign = single_excitation_sign(ket, v1, o1);

  return std::make_tuple(o1, v1, sign);
}

// TODO: Test this function
template <typename WfnType>
inline auto doubles_sign_indices(WfnType bra, WfnType ket, WfnType ex) {
  using wfn_traits = wavefunction_traits<WfnType>;
  auto [o1, v1, sign1] = single_excitation_sign_indices(bra, ket, ex);

  ket = wfn_traits::single_excitation_no_check(ket, o1, v1);
  ex = wfn_traits::single_excitation_no_check(ex, o1, v1);

  auto [o2, v2, sign2] = single_excitation_sign_indices(bra, ket, ex);
  auto sign = sign1 * sign2;

  return std::make_tuple(o1, v1, o2, v2, sign);
}

// TODO: Test this function
template <typename WfnType>
inline auto doubles_sign(WfnType bra, WfnType ket, WfnType ex) {
  auto [p, q, r, s, sign] = doubles_sign_indices(bra, ket, ex);
  return sign;
}

template <typename WfnIterator>
auto get_unique_alpha(WfnIterator begin, WfnIterator end) {
  using wfn_type = typename std::iterator_traits<WfnIterator>::value_type;
  using wfn_traits = wavefunction_traits<wfn_type>;
  using spin_wfn_type = typename wfn_traits::spin_wfn_type;

  std::vector<std::pair<spin_wfn_type, size_t>> unique_alpha;
  unique_alpha.push_back({wfn_traits::alpha_string(*begin), 1});
  for (auto it = begin + 1; it != end; ++it) {
    auto& [cur_alpha, cur_count] = unique_alpha.back();
    auto alpha_i = wfn_traits::alpha_string(*it);
    if (alpha_i == cur_alpha) {
      cur_count++;
    } else {
      unique_alpha.push_back({alpha_i, 1});
    }
  }

  return unique_alpha;
}

template <typename WfnType>
std::string to_canonical_string(WfnType state) {
  using wfn_traits = wavefunction_traits<WfnType>;
  using spin_wfn_type = spin_wfn_t<WfnType>;
  using spin_wfn_traits = wavefunction_traits<spin_wfn_type>;

  auto state_alpha = wfn_traits::alpha_string(state);
  auto state_beta = wfn_traits::beta_string(state);
  std::string str;

  for (size_t i = 0; i < spin_wfn_traits::size(); ++i) {
    if (state_alpha[i] and state_beta[i])
      str.push_back('2');
    else if (state_alpha[i])
      str.push_back('u');
    else if (state_beta[i])
      str.push_back('d');
    else
      str.push_back('0');
  }
  return str;
}

template <typename WfnType>
WfnType from_canonical_string(std::string str) {
  using spin_wfn_type = spin_wfn_t<WfnType>;
  using wfn_traits = wavefunction_traits<WfnType>;
  using spin_wfn_traits = wavefunction_traits<spin_wfn_type>;
  spin_wfn_type state_alpha(0), state_beta(0);
  for (auto i = 0ul; i < std::min(str.length(), spin_wfn_traits::size()); ++i) {
    if (str[i] == '2') {
      state_alpha = spin_wfn_traits::create_no_check(state_alpha, i);
      state_beta = spin_wfn_traits::create_no_check(state_beta, i);
    } else if (str[i] == 'u') {
      state_alpha = spin_wfn_traits::create_no_check(state_alpha, i);
    } else if (str[i] == 'd') {
      state_beta = spin_wfn_traits::create_no_check(state_beta, i);
    }
  }
  auto state = wfn_traits::from_spin(state_alpha, state_beta);
  return state;
}

}  // namespace macis
