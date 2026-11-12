/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace macis {

/**
 * @brief Packed index of the unordered orbital pair {p, q}.
 *
 * Lower-triangular row-major packing: max(p,q)*(max(p,q)+1)/2 + min(p,q).
 */
inline constexpr size_t fci_pair_index(size_t p, size_t q) {
  return p >= q ? p * (p + 1) / 2 + q : q * (q + 1) / 2 + p;
}

/**
 * @brief Occupation strings of a single spin for a fixed electron count.
 *
 * Strings are addressed in lexicographic order of their sorted occupied
 * orbital lists, which is the order produced by generate_combs (and hence the
 * alpha-major determinant order of generate_hilbert_space). Every string
 * carries its off-diagonal single excitations, which drive the direct FCI
 * contractions.
 *
 * Orbitals are limited to 65535 and the number of strings to 2^32 - 1.
 */
class FCIStringSpace {
 public:
  /// E_{create,destroy}|I> = sign |target> for occupied destroy, empty create
  struct Excitation {
    uint32_t target;  ///< Address of the excited string
    uint32_t pair;    ///< fci_pair_index(create, destroy)
    uint16_t create;
    uint16_t destroy;
    int32_t sign;  ///< +1 or -1
  };

  /**
   * @param[in] norb  Number of orbitals
   * @param[in] nelec Number of electrons of this spin (<= norb)
   * @throws std::invalid_argument if the space exceeds the limits above
   */
  FCIStringSpace(size_t norb, size_t nelec);

  size_t norb() const { return norb_; }
  size_t nelec() const { return nelec_; }
  size_t size() const { return nstrings_; }

  /// Number of off-diagonal single excitations of every string,
  /// nelec * (norb - nelec)
  size_t num_excitations() const { return nexc_; }

  /// Sorted occupied orbitals of string `i` (nelec entries)
  const uint32_t* occupied(size_t i) const {
    return occupied_.data() + i * nelec_;
  }

  /// Single excitations of string `i` (num_excitations() entries, ordered by
  /// destroyed then created orbital)
  const Excitation* excitations(size_t i) const {
    return excitations_.data() + i * nexc_;
  }

  /// Address of the string with sorted occupied orbitals `occ`
  size_t address(const uint32_t* occ) const;

 private:
  size_t norb_;
  size_t nelec_;
  size_t nstrings_;
  size_t nexc_;
  std::vector<size_t> rank_weight_;  ///< [i*norb + o] = C(norb-1-o, nelec-i)
  std::vector<uint32_t> occupied_;
  std::vector<Excitation> excitations_;
};

}  // namespace macis
