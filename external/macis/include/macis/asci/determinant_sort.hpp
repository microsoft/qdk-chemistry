/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/asci/determinant_contributions.hpp>
#if __has_include(<boost/sort/pdqsort/pdqsort.hpp>)
#define MACIS_USE_BOOST_SORT
#include <boost/sort/pdqsort/pdqsort.hpp>
#include <boost/sort/sort.hpp>
#endif /* __has_include(<boost/sort/pdqsort/pdqsort.hpp>) */

namespace macis {

/**
 * @brief Reorder determinants and coefficients by decreasing absolute
 * coefficient magnitude
 *
 * This function sorts both the determinant list and coefficient array based on
 * the absolute values of the coefficients in descending order. This is useful
 * for ordering CI wavefunctions by importance.
 *
 * @tparam WfnT Wavefunction type representing quantum determinants
 * @param[in,out] dets Vector of determinants to be reordered
 * @param[in,out] C Vector of CI coefficients to be reordered
 */
template <typename WfnT>
void reorder_ci_on_coeff(std::vector<WfnT>& dets, std::vector<double>& C) {
  size_t nlocal = C.size();
  size_t ndets = dets.size();
  std::vector<uint64_t> idx(nlocal);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(),
            [&](auto i, auto j) { return std::abs(C[i]) > std::abs(C[j]); });

  std::vector<double> reorder_C(nlocal);
  std::vector<WfnT> reorder_dets(ndets);
  assert(nlocal == ndets);
  for (auto i = 0ul; i < ndets; ++i) {
    reorder_C[i] = C[idx[i]];
    reorder_dets[i] = dets[idx[i]];
  }

  C = std::move(reorder_C);
  dets = std::move(reorder_dets);
}

/**
 * @brief Reorder determinants and coefficients based on alpha spin ordering
 *
 * This function sorts determinants according to their alpha spin configuration
 * using the wavefunction's natural spin comparator. The corresponding
 * coefficients are reordered to maintain consistency.
 *
 * @tparam WfnIterator Iterator type for wavefunction container
 * @param[in,out] begin Iterator to the beginning of the determinant range
 * @param[in,out] end Iterator to the end of the determinant range
 * @param[in,out] C Array of CI coefficients to be reordered
 */
template <typename WfnIterator>
void reorder_ci_on_alpha(WfnIterator begin, WfnIterator end, double* C) {
  using wfn_type = typename WfnIterator::value_type;
  using wfn_traits = wavefunction_traits<wfn_type>;
  using cmp_type = typename wfn_traits::spin_comparator;
  const size_t ndets = std::distance(begin, end);

  cmp_type comparator{};
  std::vector<size_t> idx(ndets);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](auto i, auto j) {
    return comparator(*(begin + i), *(begin + j));
  });

  std::vector<double> reorder_C(ndets);
  std::vector<wfn_type> reorder_dets(ndets);
  for (auto i = 0ul; i < ndets; ++i) {
    reorder_C[i] = C[idx[i]];
    reorder_dets[i] = *(begin + idx[i]);
  }

  std::copy(reorder_dets.begin(), reorder_dets.end(), begin);
  std::copy(reorder_C.begin(), reorder_C.end(), C);
}

/**
 * @brief Accumulate ASCI contributions for duplicate candidates states
 *
 * This function processes a sorted range of ASCI contributions and accumulates
 * the matrix elements for candidate states. Duplicate entries are marked with
 * NaN and can be removed later with std::unique.
 *
 * @tparam PairIterator Iterator type for ASCI contribution container
 * @param[in,out] pairs_begin Iterator to the beginning of the pairs range
 * @param[in,out] pairs_end Iterator to the end of the pairs range
 * @return Iterator pointing to the new end after removing duplicates
 */
template <typename PairIterator>
PairIterator accumulate_asci_pairs(PairIterator pairs_begin,
                                   PairIterator pairs_end) {
  // Accumulate the ASCI scores into first instance of unique bitstrings
  auto cur_it = pairs_begin;
  for (auto it = cur_it + 1; it != pairs_end; ++it) {
    // If iterate is not the one being tracked, update the iterator
    if (it->state != cur_it->state) {
      cur_it = it;
    }

    // Accumulate
    else {
      cur_it->c_times_matel += it->c_times_matel;
      it->c_times_matel = NAN;  // Zero out to expose potential bugs
    }
  }

  // Remote duplicate bitstrings
  return std::unique(pairs_begin, pairs_end,
                     [](auto x, auto y) { return x.state == y.state; });
}

/**
 * @brief Sort ASCI pairs by bitstring and accumulate duplicate contributions
 *
 * This function sorts ASCI contributions by their quantum state bitstrings
 * and then accumulates matrix elements for duplicate states. Uses either
 * Boost's pdqsort or std::sort depending on availability.
 *
 * @tparam PairIterator Iterator type for ASCI contribution container
 * @param[in,out] pairs_begin Iterator to the beginning of the pairs range
 * @param[in,out] pairs_end Iterator to the end of the pairs range
 * @return Iterator pointing to the new end after removing duplicates
 */
template <typename PairIterator>
PairIterator sort_and_accumulate_asci_pairs(PairIterator pairs_begin,
                                            PairIterator pairs_end) {
  const size_t npairs = std::distance(pairs_begin, pairs_end);

  if (!npairs) return pairs_end;

  auto comparator = [](const auto& x, const auto& y) {
    return bitset_less(x.state, y.state);
  };

// Sort by bitstring
#ifdef MACIS_USE_BOOST_SORT
  boost::sort::pdqsort_branchless
#else
  std::sort
#endif /* MACIS_USE_BOOST_SORT */
      (pairs_begin, pairs_end, comparator);

  return accumulate_asci_pairs(pairs_begin, pairs_end);
}

/**
 * @brief Stable sort ASCI pairs by bitstring and accumulate duplicate
 * contributions
 *
 * This function performs a stable sort of ASCI contributions by their quantum
 * state bitstrings and then accumulates matrix elements for duplicate states.
 * The stable sort preserves the relative order of equal elements.
 *
 * @tparam PairIterator Iterator type for ASCI contribution container
 * @param[in,out] pairs_begin Iterator to the beginning of the pairs range
 * @param[in,out] pairs_end Iterator to the end of the pairs range
 * @return Iterator pointing to the new end after removing duplicates
 */
template <typename PairIterator>
PairIterator stable_sort_and_accumulate_asci_pairs(PairIterator pairs_begin,
                                                   PairIterator pairs_end) {
  const size_t npairs = std::distance(pairs_begin, pairs_end);

  if (!npairs) return pairs_end;

  auto comparator = [](const auto& x, const auto& y) {
    return bitset_less(x.state, y.state);
  };

// Sort by bitstring
#ifdef MACIS_USE_BOOST_SORT
  boost::sort::flat_stable_sort
#else
  std::stable_sort
#endif /* MACIS_USE_BOOST_SORT */
      (pairs_begin, pairs_end, comparator);

  return accumulate_asci_pairs(pairs_begin, pairs_end);
}

/**
 * @brief Sort and accumulate ASCI pairs in a container
 *
 * Convenience function that sorts and accumulates ASCI contributions within
 * a container, automatically erasing the invalidated elements after processing.
 * This is a wrapper around the iterator-based version that handles container
 * management.
 *
 * @tparam WfnT Wavefunction type for the ASCI contributions
 * @param[in,out] asci_pairs Container of ASCI contributions to process
 */
template <typename WfnT>
void sort_and_accumulate_asci_pairs(asci_contrib_container<WfnT>& asci_pairs) {
  auto uit =
      sort_and_accumulate_asci_pairs(asci_pairs.begin(), asci_pairs.end());
  asci_pairs.erase(uit, asci_pairs.end());  // Erase dead space
}

/**
 * @brief Keep only the largest contribution for each unique determinant state
 *
 * This function sorts ASCI contributions by bitstring and for each unique
 * quantum state, retains only the contribution with the largest absolute
 * matrix element.
 *
 * @tparam WfnT Wavefunction type for the ASCI contributions
 * @param[in,out] asci_pairs Container of ASCI contributions to process
 */
template <typename WfnT>
void keep_only_largest_copy_asci_pairs(
    asci_contrib_container<WfnT>& asci_pairs) {
  if (!asci_pairs.size()) return;
  auto comparator = [](const auto& x, const auto& y) {
    return bitset_less(x.state, y.state);
  };

// Sort by bitstring
#ifdef MACIS_USE_BOOST_SORT
  boost::sort::pdqsort_branchless
#else
  std::sort
#endif /* MACIS_USE_BOOST_SORT */
      (asci_pairs.begin(), asci_pairs.end(), comparator);

  // Keep the largest ASCI score in the unique instance of each bit string
  auto cur_it = asci_pairs.begin();
  for (auto it = cur_it + 1; it != asci_pairs.end(); ++it) {
    // If iterate is not the one being tracked, update the iterator
    if (it->state != cur_it->state) {
      cur_it = it;
    }

    // Keep only max value
    else {
      cur_it->c_times_matel =
          std::max(cur_it->c_times_matel, it->c_times_matel);
    }
  }

  // Remote duplicate bitstrings
  auto uit = std::unique(asci_pairs.begin(), asci_pairs.end(),
                         [](auto x, auto y) { return x.state == y.state; });
  asci_pairs.erase(uit, asci_pairs.end());  // Erase dead space
}

}  // namespace macis
