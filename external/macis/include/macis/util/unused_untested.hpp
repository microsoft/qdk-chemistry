/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#if 0
/**
 *  @brief Generate canonical HF determinant.
 *
 *  Generates a string representation of the canonical HF determinant
 *  consisting of a specifed number of alpha and beta orbitals. This variant
 *  does not assume energetic ordering of the HF orbitals.
 *
 *  TODO: This assumes restricted orbitals
 *
 *  @tparam N Number of bits for the total bit string of the state
 *  @param[in] nalpha  Number of occupied alpha orbitals in the HF state
 *  @param[in] nbeta   Number of occupied beta orbitals in the HF state
 *  @param[in] orb_ens Orbital eigenenergies.
 *
 *  @returns The bitstring HF state consisting of the specified number of
 *    occupied orbitals populated according to the ordering of `orb_ens`.
 */
template <size_t N>
std::bitset<N> canonical_hf_determinant(uint32_t nalpha, uint32_t nbeta,
                                        const std::vector<double>& orb_ens) {
  static_assert((N % 2) == 0, "N Must Be Even");
  // First, find the sorted indices for the orbital energies
  std::vector<size_t> idx(orb_ens.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [&orb_ens](size_t i1, size_t i2) {
    return orb_ens[i1] < orb_ens[i2];
  });
  // Next, fill the electrons by energy
  std::bitset<N> alpha(0), beta(0);
  for(uint32_t i = 0; i < nalpha; i++) alpha.flip(idx[i]);
  for(uint32_t i = 0; i < nbeta; i++) beta.flip(idx[i] + N / 2);
  return alpha | beta;
}

/**
 *  @brief Generate the list of (un)occupied orbitals for a paricular state.
 *
 *  TODO: Test this function
 *
 *  @tparam N Number of bits for the total bit string of the state
 *  @param[in]  norb   Number of orbitals used to describe the state (<= `N`)
 *  @param[in]  state  The state from which to determine orbital occupations.
 *  @param[out] occ    List of occupied orbitals in `state`
 *  @param[out] vir    List of unoccupied orbitals in `state`
 *  @param[in]  as_orbs TODO:????
 */
template <size_t N>
void bitset_to_occ_vir_as(size_t norb, std::bitset<N> state,
                          std::vector<uint32_t>& occ,
                          std::vector<uint32_t>& vir,
                          const std::vector<uint32_t>& as_orbs) {
  occ.clear();
  for(const auto i : as_orbs)
    if(state[i]) occ.emplace_back(i);
  const auto nocc = occ.size();
  assert(nocc <= norb);

  const auto nvir = as_orbs.size() - nocc;
  vir.resize(nvir);
  auto it = vir.begin();
  for(const auto i : as_orbs)
    if(!state[i]) *(it++) = i;
}



// TODO: Test this function
template <size_t N>
void generate_singles_as(size_t norb, std::bitset<N> state,
                         std::vector<std::bitset<N>>& singles,
                         const std::vector<uint32_t>& as_orbs) {
  std::vector<uint32_t> occ_orbs, vir_orbs;
  bitset_to_occ_vir_as<N>(norb, state, occ_orbs, vir_orbs, as_orbs);

  singles.clear();
  append_singles(state, occ_orbs, vir_orbs, singles);
}

// TODO: Test this function
template <size_t N>
void generate_singles_doubles_as(size_t norb, std::bitset<N> state,
                                 std::vector<std::bitset<N>>& singles,
                                 std::vector<std::bitset<N>>& doubles,
                                 const std::vector<uint32_t>& as_orbs) {
  std::vector<uint32_t> occ_orbs, vir_orbs;
  bitset_to_occ_vir_as<N>(norb, state, occ_orbs, vir_orbs, as_orbs);

  singles.clear();
  doubles.clear();
  append_singles(state, occ_orbs, vir_orbs, singles);
  append_doubles(state, occ_orbs, vir_orbs, doubles);
}

// TODO: Test this function
template <size_t N>
void generate_singles_spin_as(size_t norb, std::bitset<N> state,
                              std::vector<std::bitset<N>>& singles,
                              const std::vector<uint32_t> as_orbs) {
  auto state_alpha = bitset_lo_word(state);
  auto state_beta = bitset_hi_word(state);

  std::vector<std::bitset<N / 2>> singles_alpha, singles_beta;

  // Generate Spin-Specific singles
  generate_singles_as(norb, state_alpha, singles_alpha, as_orbs);
  generate_singles_as(norb, state_beta, singles_beta, as_orbs);

  auto state_alpha_expand = expand_bitset<N>(state_alpha);
  auto state_beta_expand = expand_bitset<N>(state_beta) << (N / 2);

  // Generate Singles in full space
  singles.clear();

  // Single Alpha + No Beta
  for(auto s_alpha : singles_alpha) {
    auto s_state = expand_bitset<N>(s_alpha);
    s_state = s_state | state_beta_expand;
    singles.emplace_back(s_state);
  }

  // No Alpha + Single Beta
  for(auto s_beta : singles_beta) {
    auto s_state = expand_bitset<N>(s_beta) << (N / 2);
    s_state = s_state | state_alpha_expand;
    singles.emplace_back(s_state);
  }
}

// TODO: Test this function
template <size_t N>
void generate_singles_doubles_spin_as(size_t norb, std::bitset<N> state,
                                      std::vector<std::bitset<N>>& singles,
                                      std::vector<std::bitset<N>>& doubles,
                                      const std::vector<uint32_t>& as_orbs) {
  auto state_alpha = bitset_lo_word(state);
  auto state_beta = bitset_hi_word(state);

  std::vector<std::bitset<N / 2>> singles_alpha, singles_beta;
  std::vector<std::bitset<N / 2>> doubles_alpha, doubles_beta;

  // Generate Spin-Specific singles / doubles
  generate_singles_doubles_as(norb, state_alpha, singles_alpha, doubles_alpha,
                              as_orbs);
  generate_singles_doubles_as(norb, state_beta, singles_beta, doubles_beta,
                              as_orbs);

  auto state_alpha_expand = expand_bitset<N>(state_alpha);
  auto state_beta_expand = expand_bitset<N>(state_beta) << (N / 2);

  // Generate Singles in full space
  singles.clear();

  // Single Alpha + No Beta
  for(auto s_alpha : singles_alpha) {
    auto s_state = expand_bitset<N>(s_alpha);
    s_state = s_state | state_beta_expand;
    singles.emplace_back(s_state);
  }

  // No Alpha + Single Beta
  for(auto s_beta : singles_beta) {
    auto s_state = expand_bitset<N>(s_beta) << (N / 2);
    s_state = s_state | state_alpha_expand;
    singles.emplace_back(s_state);
  }

  // Generate Doubles in full space
  doubles.clear();

  // Double Alpha + No Beta
  for(auto d_alpha : doubles_alpha) {
    auto d_state = expand_bitset<N>(d_alpha);
    d_state = d_state | state_beta_expand;
    doubles.emplace_back(d_state);
  }

  // No Alpha + Double Beta
  for(auto d_beta : doubles_beta) {
    auto d_state = expand_bitset<N>(d_beta) << (N / 2);
    d_state = d_state | state_alpha_expand;
    doubles.emplace_back(d_state);
  }

  // Single Alpha + Single Beta
  for(auto s_alpha : singles_alpha)
    for(auto s_beta : singles_beta) {
      auto d_state_alpha = expand_bitset<N>(s_alpha);
      auto d_state_beta = expand_bitset<N>(s_beta) << (N / 2);
      doubles.emplace_back(d_state_alpha | d_state_beta);
    }
}
#if 0
// TODO: Test this function
template <size_t N>
void generate_residues(std::bitset<N> state, std::vector<std::bitset<N>>& res) {
  auto state_alpha = bitset_lo_word(state);
  auto state_beta = bitset_hi_word(state);

  auto occ_alpha = bits_to_indices(state_alpha);
  const int nalpha = occ_alpha.size();

  auto occ_beta = bits_to_indices(state_beta);
  const int nbeta = occ_beta.size();

  std::bitset<N> state_alpha_full = expand_bitset<N>(state_alpha);
  std::bitset<N> state_beta_full = expand_bitset<N>(state_beta);
  state_beta_full = state_beta_full << (N / 2);

  std::bitset<N / 2> one = 1ul;

  // Double alpha
  for(auto i = 0; i < nalpha; ++i)
    for(auto j = i + 1; j < nalpha; ++j) {
      auto mask = (one << occ_alpha[i]) | (one << occ_alpha[j]);
      std::bitset<N> _r = expand_bitset<N>(state_alpha & ~mask);
      res.emplace_back(_r | state_beta_full);
    }

  // Double beta
  for(auto i = 0; i < nbeta; ++i)
    for(auto j = i + 1; j < nbeta; ++j) {
      auto mask = (one << occ_beta[i]) | (one << occ_beta[j]);
      std::bitset<N> _r = expand_bitset<N>(state_beta & ~mask) << (N / 2);
      res.emplace_back(_r | state_alpha_full);
    }

  // Mixed
  for(auto i = 0; i < nalpha; ++i)
    for(auto j = 0; j < nbeta; ++j) {
      std::bitset<N> mask = expand_bitset<N>(one << occ_alpha[i]);
      mask = mask | (expand_bitset<N>(one << occ_beta[j]) << (N / 2));
      res.emplace_back(state & ~mask);
    }
}
#endif /* 0 */
#endif /* 0 */
