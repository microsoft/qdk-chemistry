/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <macis/bitset_operations.hpp>
#include <macis/types.hpp>

namespace macis {

/**
 * @brief Enumeration representing electron spin states
 *
 * This enum class defines the two possible spin states for electrons:
 * - Alpha: Spin-up electrons
 * - Beta: Spin-down electrons
 */
enum class Spin { Alpha, Beta };

/**
 * @brief Trait structure for extracting wavefunction type information
 *
 * This template struct provides a generic interface for accessing type
 * information and operations specific to different wavefunction
 * representations. It serves as a type trait mechanism for the wavefunction
 * system.
 *
 * @tparam WfnType The wavefunction type to extract traits for
 */
template <typename WfnType>
struct wavefunction_traits;

/**
 * @brief Type alias for extracting the spin wavefunction type from a
 * wavefunction type
 *
 * This alias provides convenient access to the spin-specific wavefunction type
 * associated with a given wavefunction type through the wavefunction_traits
 * mechanism.
 *
 * @tparam WfnType The wavefunction type to extract the spin type from
 */
template <typename WfnType>
using spin_wfn_t = typename wavefunction_traits<WfnType>::spin_wfn_type;

/**
 * @brief Specialized wavefunction traits for std::bitset<N> representation
 *
 * This specialization provides type information and operations for
 * wavefunctions represented as fixed-size bitsets. The bitset is divided into
 * two halves: the lower N/2 bits represent alpha (spin-up) electrons, and the
 * upper N/2 bits represent beta (spin-down) electrons.
 *
 * @tparam N The total number of bits in the bitset (must be even)
 */
template <size_t N>
struct wavefunction_traits<std::bitset<N>> {
  using wfn_type = std::bitset<N>;  ///< Full wavefunction type (N bits total)
  using spin_wfn_type =
      std::bitset<N / 2>;        ///< Spin-specific wavefunction type (N/2 bits)
  using orbidx_type = uint32_t;  ///< Type for orbital indices
  using orbidx_container =
      std::vector<orbidx_type>;  ///< Container for storing orbital indices

  inline static constexpr size_t bit_size =
      N;  ///< Total number of bits in the wavefunction

  /**
   * @brief Get the total number of bits in the wavefunction
   * @return The total bit size (N)
   */
  static constexpr auto size() { return bit_size; }

  /**
   * @brief Count the total number of set bits (occupied orbitals) in the
   * wavefunction
   * @param state The wavefunction state to count
   * @return The number of set bits (total electron count)
   */
  static inline auto count(wfn_type state) { return state.count(); }

  /**
   * @brief Extract the alpha (spin-up) electron configuration from the
   * wavefunction
   * @param state The full wavefunction state
   * @return The alpha spin configuration (lower N/2 bits)
   */
  static inline spin_wfn_type alpha_string(wfn_type state) {
    return bitset_lo_word(state);
  }

  /**
   * @brief Extract the beta (spin-down) electron configuration from the
   * wavefunction
   * @param state The full wavefunction state
   * @return The beta spin configuration (upper N/2 bits)
   */
  static inline spin_wfn_type beta_string(wfn_type state) {
    return bitset_hi_word(state);
  }

  using wfn_comparator =
      bitset_less_comparator<N>;  ///< Comparator for ordering full
                                  ///< wavefunctions

  /**
   * @brief Comparator for ordering wavefunctions by spin configuration
   *
   * This comparator first compares alpha spin strings, and if they are equal,
   * then compares beta spin strings. This provides a consistent ordering
   * that groups wavefunctions by their alpha spin configuration.
   */
  struct spin_comparator {
    using spin_wfn_comparator =
        bitset_less_comparator<N / 2>;  ///< Comparator for spin-specific
                                        ///< configurations

    /**
     * @brief Compare two wavefunctions by their spin configurations
     * @param x First wavefunction to compare
     * @param y Second wavefunction to compare
     * @return true if x < y in spin-ordered comparison
     */
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

  /**
   * @brief Construct a full wavefunction from separate alpha and beta spin
   * configurations
   *
   * This function combines alpha and beta spin configurations into a single
   * wavefunction. The template parameter determines the bit ordering: Alpha
   * ordering places alpha electrons in the lower bits, while Beta ordering
   * places beta electrons in the lower bits.
   *
   * @tparam Sigma The spin ordering convention (Alpha or Beta)
   * @param alpha The alpha (spin-up) electron configuration
   * @param beta The beta (spin-down) electron configuration
   * @return The combined wavefunction state
   */
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

  /**
   * @brief Create the canonical Hartree-Fock determinant with specified
   * electron counts
   *
   * This function generates a wavefunction representing the canonical
   * Hartree-Fock ground state with the lowest energy orbitals filled. Alpha
   * electrons fill orbitals 0 through nalpha-1, and beta electrons fill
   * orbitals 0 through nbeta-1.
   *
   * @param nalpha Number of alpha (spin-up) electrons
   * @param nbeta Number of beta (spin-down) electrons
   * @return The canonical HF determinant wavefunction
   */
  static inline wfn_type canonical_hf_determinant(uint32_t nalpha,
                                                  uint32_t nbeta) {
    spin_wfn_type alpha = full_mask<N / 2>(nalpha);
    spin_wfn_type beta = full_mask<N / 2>(nbeta);
    return from_spin(alpha, beta);
  }

  /**
   * @brief Flip multiple bits in the wavefunction for the specified spin
   *
   * This is a variadic template function declaration that flips bits
   * corresponding to orbital indices for either alpha or beta electrons. The
   * actual implementation is provided by the recursive template specializations
   * below.
   *
   * @tparam Sigma The spin type (Alpha or Beta) to flip bits for
   * @tparam Inds Variadic template parameter pack for orbital indices
   * @param state The wavefunction state to modify (passed by reference)
   * @param ... Orbital indices to flip
   * @return Reference to the modified wavefunction state
   */
  template <Spin Sigma, typename... Inds>
  static inline wfn_type& flip_bits(wfn_type& state, Inds&&...);

  /**
   * @brief Base case for recursive bit flipping (no more indices to flip)
   *
   * This template specialization handles the base case of the recursive
   * flip_bits function when no more orbital indices remain to be flipped.
   *
   * @tparam Sigma The spin type (Alpha or Beta)
   * @param state The wavefunction state
   * @return Reference to the unmodified wavefunction state
   */
  template <Spin Sigma>
  static inline wfn_type& flip_bits(wfn_type& state) {
    return state;
  }

  /**
   * @brief Recursive case for flipping bits in the wavefunction
   *
   * This function flips the bit at orbital index p for the specified spin type
   * and then recursively processes the remaining orbital indices. The bit
   * position is calculated by adding an offset for beta electrons (N/2).
   *
   * @tparam Sigma The spin type (Alpha or Beta)
   * @tparam Inds Remaining orbital indices to process
   * @param state The wavefunction state to modify
   * @param p The current orbital index to flip
   * @param inds Remaining orbital indices to flip
   * @return Reference to the modified wavefunction state
   */
  template <Spin Sigma, typename... Inds>
  static inline wfn_type& flip_bits(wfn_type& state, unsigned p,
                                    Inds&&... inds) {
    return flip_bits<Sigma>(state.flip(p + (Sigma == Spin::Alpha ? 0 : N / 2)),
                            std::forward<Inds>(inds)...);
  }

  /**
   * @brief Create (add) an electron in orbital p without validity checks
   *
   * This function flips the bit at orbital index p for the specified spin,
   * effectively adding an electron to that orbital. No checks are performed
   * to ensure the orbital was previously unoccupied.
   *
   * @tparam Sigma The spin type (Alpha or Beta) of the electron to create
   * @param state The initial wavefunction state
   * @param p The orbital index where the electron is created
   * @return The modified wavefunction state with the electron added
   */
  template <Spin Sigma = Spin::Alpha>
  static inline wfn_type create_no_check(wfn_type state, unsigned p) {
    flip_bits<Sigma>(state, p);
    return state;
  }

  /**
   * @brief Perform a single excitation from orbital p to orbital q without
   * validity checks
   *
   * This function performs a single electron excitation by flipping bits at
   * both orbital indices p and q for the specified spin. This moves an electron
   * from orbital p to orbital q. No checks are performed to ensure p is
   * occupied and q is empty.
   *
   * @tparam Sigma The spin type (Alpha or Beta) of the electron to excite
   * @param state The initial wavefunction state
   * @param p The source orbital index (electron removed from here)
   * @param q The target orbital index (electron added here)
   * @return The modified wavefunction state after the excitation
   */
  template <Spin Sigma = Spin::Alpha>
  static inline wfn_type single_excitation_no_check(wfn_type state, unsigned p,
                                                    unsigned q) {
    flip_bits<Sigma>(state, p, q);
    return state;
  }

  /**
   * @brief Perform a double excitation without validity checks
   *
   * This function performs a double electron excitation by flipping bits at
   * orbital indices p, q, r, and s for the specified spin. This effectively
   * moves electrons from orbitals p and q to orbitals r and s. No checks are
   * performed to ensure the initial and final orbital occupations are valid.
   *
   * @tparam Sigma The spin type (Alpha or Beta) of the electrons to excite
   * @param state The initial wavefunction state
   * @param p The first source orbital index
   * @param q The second source orbital index
   * @param r The first target orbital index
   * @param s The second target orbital index
   * @return The modified wavefunction state after the double excitation
   */
  template <Spin Sigma = Spin::Alpha>
  static inline wfn_type double_excitation_no_check(wfn_type state, unsigned p,
                                                    unsigned q, unsigned r,
                                                    unsigned s) {
    flip_bits<Sigma>(state, p, q, r, s);
    return state;
  }

  /**
   * @brief Convert wavefunction state to occupied orbital indices
   *
   * This function extracts all occupied orbital indices from the wavefunction
   * state and stores them in the provided container. The container is populated
   * with the indices of all set bits in the wavefunction.
   *
   * @param state The wavefunction state to analyze
   * @param occ Reference to container that will be filled with occupied orbital
   * indices
   */
  static inline void state_to_occ(wfn_type state, orbidx_container& occ) {
    occ = bits_to_indices(state);
  }

  /**
   * @brief Convert wavefunction state to occupied orbital indices
   *
   * This function extracts all occupied orbital indices from the wavefunction
   * state and returns them as a new container. This is a convenience overload
   * that creates and returns a new container instead of modifying an existing
   * one.
   *
   * @param state The wavefunction state to analyze
   * @return Container with occupied orbital indices
   */
  static inline orbidx_container state_to_occ(wfn_type state) {
    return bits_to_indices(state);
  }

  /**
   * @brief Convert wavefunction state to occupied and virtual orbital indices
   *
   * This function analyzes a wavefunction state and separates orbital indices
   * into occupied and virtual (unoccupied) orbitals. It first extracts all
   * occupied orbitals, then determines virtual orbitals by finding unset bits
   * within the specified orbital space.
   *
   * @param norb Total number of orbitals in the system
   * @param state The wavefunction state to analyze
   * @param occ Reference to container that will be filled with occupied orbital
   * indices
   * @param vir Reference to container that will be filled with virtual orbital
   * indices
   *
   * @note The function assumes that all orbital indices are within [0, norb)
   * @note Uses ffs (find first set) to efficiently locate unoccupied orbitals
   */
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
