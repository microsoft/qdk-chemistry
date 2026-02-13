/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
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
 * @brief Find the first occupied orbital that is flipped in an excitation.
 * @tparam N Size of the bitset
 * @param[in] state The reference state bitset
 * @param[in] ex The excitation pattern bitset
 * @return Index of the first occupied orbital that is flipped (0-based)
 */
template <size_t N>
uint32_t first_occupied_flipped(std::bitset<N> state, std::bitset<N> ex) {
  return ffs(state & ex) - 1u;
}

/**
 * @brief Calculate the sign of a single excitation.
 * @tparam N Size of the bitset
 * @param[in] state The reference state bitset
 * @param[in] p First orbital index
 * @param[in] q Second orbital index
 * @return Sign factor (+1.0 or -1.0) for the single excitation
 */
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

/**
 * @brief Generate all single excitations from a reference state and append to
 * container.
 * @tparam WfnType Type of the wavefunction state
 * @tparam WfnContainer Container type for storing excitations
 * @param[in] state The reference state
 * @param[in] occ Vector of occupied orbital indices
 * @param[in] vir Vector of virtual orbital indices
 * @param[out] singles Container to store generated single excitations
 */
template <typename WfnType, typename WfnContainer>
void append_singles(WfnType state, const std::vector<uint32_t>& occ,
                    const std::vector<uint32_t>& vir, WfnContainer& singles) {
  using wfn_traits = wavefunction_traits<WfnType>;
  const size_t num_occupied_orbitals = occ.size();
  const size_t num_virtual_orbitals = vir.size();

  singles.clear();
  singles.reserve(num_occupied_orbitals * num_virtual_orbitals);

  for (size_t a = 0; a < num_virtual_orbitals; ++a)
    for (size_t i = 0; i < num_occupied_orbitals; ++i) {
      singles.emplace_back(
          wfn_traits::single_excitation_no_check(state, occ[i], vir[a]));
    }
}

/**
 * @brief Generate all double excitations from a reference state and append to
 * container.
 * @tparam WfnType Type of the wavefunction state
 * @tparam WfnContainer Container type for storing excitations
 * @param[in] state The reference state
 * @param[in] occ Vector of occupied orbital indices
 * @param[in] vir Vector of virtual orbital indices
 * @param[out] doubles Container to store generated double excitations
 */
template <typename WfnType, typename WfnContainer>
void append_doubles(WfnType state, const std::vector<uint32_t>& occ,
                    const std::vector<uint32_t>& vir, WfnContainer& doubles) {
  using wfn_traits = wavefunction_traits<WfnType>;
  const size_t num_occupied_orbitals = occ.size();
  const size_t num_virtual_orbitals = vir.size();

  doubles.clear();
  const size_t nv2 = (num_virtual_orbitals * (num_virtual_orbitals - 1)) / 2;
  const size_t no2 = (num_occupied_orbitals * (num_occupied_orbitals - 1)) / 2;
  doubles.reserve(nv2 * no2);

  for (size_t a = 0; a < num_virtual_orbitals; ++a)
    for (size_t i = 0; i < num_occupied_orbitals; ++i)
      for (size_t b = a + 1; b < num_virtual_orbitals; ++b)
        for (size_t j = i + 1; j < num_occupied_orbitals; ++j) {
          doubles.emplace_back(wfn_traits::double_excitation_no_check(
              state, occ[i], occ[j], vir[a], vir[b]));
        }
}

/**
 * @brief Generate all single excitations from a reference state.
 * @tparam WfnType Type of the wavefunction state
 * @tparam WfnContainer Container type for storing excitations
 * @param[in] norb Total number of orbitals
 * @param[in] state The reference state
 * @param[out] singles Container to store generated single excitations
 */
template <typename WfnType, typename WfnContainer>
void generate_singles(size_t norb, WfnType state, WfnContainer& singles) {
  using wfn_traits = wavefunction_traits<WfnType>;
  std::vector<uint32_t> occ_orbs, vir_orbs;
  wfn_traits::state_to_occ_vir(norb, state, occ_orbs, vir_orbs);

  singles.clear();
  append_singles(state, occ_orbs, vir_orbs, singles);
}

/**
 * @brief Generate all double excitations from a reference state.
 * @tparam WfnType Type of the wavefunction state
 * @tparam WfnContainer Container type for storing excitations
 * @param[in] norb Total number of orbitals
 * @param[in] state The reference state
 * @param[out] doubles Container to store generated double excitations
 */
template <typename WfnType, typename WfnContainer>
void generate_doubles(size_t norb, WfnType state, WfnContainer& doubles) {
  using wfn_traits = wavefunction_traits<WfnType>;
  std::vector<uint32_t> occ_orbs, vir_orbs;
  wfn_traits::state_to_occ_vir(norb, state, occ_orbs, vir_orbs);

  doubles.clear();
  append_doubles(state, occ_orbs, vir_orbs, doubles);
}

/**
 * @brief Generate all single and double excitations from a reference state.
 * @tparam WfnType Type of the wavefunction state
 * @tparam WfnContainer Container type for storing excitations
 * @param[in] norb Total number of orbitals
 * @param[in] state The reference state
 * @param[out] singles Container to store generated single excitations
 * @param[out] doubles Container to store generated double excitations
 */
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

/**
 * @brief Generate all single excitations with explicit spin handling.
 * @tparam WfnType Type of the wavefunction state
 * @tparam WfnContainer Container type for storing excitations
 * @param[in] norb Total number of orbitals
 * @param[in] state The reference state
 * @param[out] singles Container to store generated single excitations
 */
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

/**
 * @brief Generate all single and double excitations with explicit spin
 * handling.
 * @tparam WfnType Type of the wavefunction state
 * @tparam WfnContainer Container type for storing excitations
 * @param[in] norb Total number of orbitals
 * @param[in] state The reference state
 * @param[out] singles Container to store generated single excitations
 * @param[out] doubles Container to store generated double excitations
 */
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

/**
 * @brief Generate CISD (Configuration Interaction Singles and Doubles) Hilbert
 * space.
 * @tparam WfnType Type of the wavefunction state
 * @tparam WfnContainer Container type for storing determinants
 * @param[in] norb Total number of orbitals
 * @param[in] state The reference state
 * @param[out] dets Container to store all determinants in the CISD space
 */
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

/**
 * @brief Generate CISD Hilbert space and return as a vector.
 * @tparam WfnType Type of the wavefunction state
 * @param[in] norb Total number of orbitals
 * @param[in] state The reference state
 * @return Vector containing all determinants in the CISD space
 */
template <typename WfnType>
auto generate_cisd_hilbert_space(size_t norb, WfnType state) {
  std::vector<WfnType> dets;
  generate_cisd_hilbert_space(norb, state, dets);
  return dets;
}

/**
 * @brief Generate all combinations of setting nset bits in nbits positions.
 * @tparam WfnType Type of the wavefunction state
 * @param[in] nbits Total number of bits
 * @param[in] nset Number of bits to set
 * @return Vector of all possible combinations
 */
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

/**
 * @brief Generate the full Hilbert space for given orbital and electron counts.
 * @tparam WfnType Type of the wavefunction state
 * @param[in] norbs Total number of orbitals
 * @param[in] nalpha Number of alpha electrons
 * @param[in] nbeta Number of beta electrons
 * @return Vector containing all possible determinants
 */
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

/**
 * @brief Generate CIS (Configuration Interaction Singles) Hilbert space.
 * @tparam WfnType Type of the wavefunction state
 * @tparam WfnContainer Container type for storing determinants
 * @param[in] norb Total number of orbitals
 * @param[in] state The reference state
 * @param[out] dets Container to store all determinants in the CIS space
 */
template <typename WfnType, typename WfnContainer>
void generate_cis_hilbert_space(size_t norb, WfnType state,
                                WfnContainer& dets) {
  dets.clear();
  dets.emplace_back(state);
  std::vector<WfnType> singles;
  generate_singles_spin(norb, state, singles);
  dets.insert(dets.end(), singles.begin(), singles.end());
}

/**
 * @brief Generate CIS Hilbert space and return as a vector.
 * @tparam WfnType Type of the wavefunction state
 * @param[in] norb Total number of orbitals
 * @param[in] state The reference state
 * @return Vector containing all determinants in the CIS space
 */
template <typename WfnType>
std::vector<WfnType> generate_cis_hilbert_space(size_t norb, WfnType state) {
  std::vector<WfnType> dets;
  generate_cis_hilbert_space(norb, state, dets);
  return dets;
}

/**
 * @brief Calculate sign and indices for a single excitation between bra and ket
 * states.
 * @tparam WfnType Type of the wavefunction state
 * @param[in] bra The bra state
 * @param[in] ket The ket state
 * @param[in] ex The excitation pattern
 * @return Tuple containing (occupied index, virtual index, sign)
 */
template <typename WfnType>
inline auto single_excitation_sign_indices(WfnType bra, WfnType ket,
                                           WfnType ex) {
  auto o1 = first_occupied_flipped(ket, ex);
  auto v1 = first_occupied_flipped(bra, ex);
  auto sign = single_excitation_sign(ket, v1, o1);

  return std::make_tuple(o1, v1, sign);
}

/**
 * @brief Calculate sign and indices for a double excitation between bra and ket
 * states.
 * @tparam WfnType Type of the wavefunction state
 * @param[in] bra The bra state
 * @param[in] ket The ket state
 * @param[in] ex The excitation pattern
 * @return Tuple containing (occupied1, virtual1, occupied2, virtual2, sign)
 */
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

/**
 * @brief Calculate the sign factor for a double excitation.
 * @tparam WfnType Type of the wavefunction state
 * @param[in] bra The bra state
 * @param[in] ket The ket state
 * @param[in] ex The excitation pattern
 * @return Sign factor for the double excitation
 */
template <typename WfnType>
inline auto doubles_sign(WfnType bra, WfnType ket, WfnType ex) {
  auto [p, q, r, s, sign] = doubles_sign_indices(bra, ket, ex);
  return sign;
}

/**
 * @brief Get unique alpha strings and their counts from a range of
 * wavefunctions.
 * @tparam WfnIterator Iterator type for wavefunction states
 * @param[in] begin Iterator to the beginning of the range
 * @param[in] end Iterator to the end of the range
 * @return Vector of pairs containing (unique alpha string, count)
 */
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

/**
 * @brief Convert a wavefunction state to a canonical string representation.
 * @tparam WfnType Type of the wavefunction state
 * @param[in] state The wavefunction state to convert
 * @return String representation where '2'=doubly occupied, 'u'=alpha only,
 * 'd'=beta only, '0'=empty
 */
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

/**
 * @brief Convert a canonical string representation to a wavefunction state.
 * @tparam WfnType Type of the wavefunction state
 * @param[in] str String representation where '2'=doubly occupied, 'u'=alpha
 * only, 'd'=beta only, '0'=empty
 * @return The corresponding wavefunction state
 */
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
