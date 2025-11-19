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
#include <macis/util/mpi.hpp>
#include <random>
#include <vector>

namespace macis {

/**
 * @brief Performs a three-way partitioning of elements around a pivot value
 *
 * This function implements the LEG (Less-Equal-Greater) partitioning scheme,
 * which divides a range of elements into three partitions:
 * - L: Elements less than the pivot
 * - E: Elements equal to the pivot
 * - G: Elements greater than the pivot
 *
 * The algorithm uses two successive std::partition calls to achieve this
 * layout: [L L L | E E E | G G G]
 *
 * @tparam RandomIt Random access iterator type
 * @tparam ValueType Type of the pivot value (must be comparable to iterator
 * value type)
 * @tparam OrderCompare Binary predicate for ordering comparison (typically <)
 * @tparam EqualCompare Binary predicate for equality comparison (typically ==)
 *
 * @param begin Iterator to the beginning of the range
 * @param end Iterator to the end of the range
 * @param pivot The pivot value around which to partition
 * @param ord_comp Comparison function for ordering (x < pivot)
 * @param eq_comp Comparison function for equality (x == pivot)
 *
 * @return std::tuple<RandomIt, RandomIt, RandomIt, RandomIt> containing:
 *         - begin: Iterator to start of L partition
 *         - e_begin: Iterator to start of E partition (end of L)
 *         - g_begin: Iterator to start of G partition (end of E)
 *         - end: Iterator to end of G partition (unchanged from input)
 */
template <typename RandomIt, typename ValueType, class OrderCompare,
          class EqualCompare>
auto leg_partition(RandomIt begin, RandomIt end, ValueType pivot,
                   OrderCompare ord_comp, EqualCompare eq_comp) {
  auto less_lambda = [&](const auto& x) { return ord_comp(x, pivot); };
  auto eq_lambda = [&](const auto& x) { return eq_comp(x, pivot); };

  auto e_begin = std::partition(begin, end, less_lambda);
  auto g_begin = std::partition(e_begin, end, eq_lambda);

  return std::make_tuple(begin, e_begin, g_begin, end);
}

#ifdef MACIS_ENABLE_MPI
/**
 * @brief Performs MPI allgather and exclusive scan operations for distributed
 * computing
 *
 * This utility function combines two common MPI collective operations:
 * 1. MPI_Allgather: Collects a single value from each process and distributes
 *    the complete array to all processes
 * 2. Exclusive scan: Computes prefix sums (cumulative totals) excluding the
 * current element
 *
 * The function is typically used in distributed algorithms to:
 * - Determine how many elements each process has
 * - Calculate starting indices for each process's data in a global array
 * - Compute the total number of elements across all processes
 *
 * @tparam Integral Integer type (e.g., int, size_t, long)
 *
 * @param val The local value to contribute from this MPI process
 * @param gather Output vector that will contain all processes' values
 * @param scan Output vector that will contain exclusive prefix sums
 * @param comm MPI communicator for the operation
 *
 * @return Total sum of all values across all processes
 *
 * @note The gather and scan vectors will be resized to match the communicator
 * size
 *
 * Example:
 * If 4 processes have values [3, 1, 4, 2]:
 * - gather will be [3, 1, 4, 2]
 * - scan will be [0, 3, 4, 8] (exclusive prefix sums)
 * - return value will be 10 (total sum)
 */
template <typename Integral>
Integral total_gather_and_exclusive_scan(Integral val,
                                         std::vector<Integral>& gather,
                                         std::vector<Integral>& scan,
                                         MPI_Comm comm) {
  auto world_size = comm_size(comm);
  gather.resize(world_size);
  scan.resize(world_size);

  auto dtype = mpi_traits<Integral>::datatype();
  MPI_Allgather(&val, 1, dtype, gather.data(), 1, dtype, comm);
  Integral total = std::accumulate(gather.begin(), gather.end(), Integral(0));
  std::exclusive_scan(gather.begin(), gather.end(), scan.begin(), 0);

  return total;
}

/**
 * @brief Distributed QuickSelect algorithm for finding the k-th smallest
 * element across MPI processes
 *
 * This function implements a distributed version of the QuickSelect algorithm
 * using MPI. It finds the k-th smallest element from data distributed across
 * multiple MPI processes without requiring all data to be gathered on a single
 * process (except in degenerate cases).
 *
 * Algorithm Overview:
 * 1. Each MPI process maintains a local portion of the data
 * 2. A pivot element is randomly selected from the global dataset
 * 3. Each process partitions its local data using 3-way partitioning (LEG)
 * 4. Global partition sizes are computed via MPI reduction
 * 5. Based on global partition sizes, the algorithm decides which partition
 * contains the k-th element
 * 6. The search continues recursively in the relevant partition
 * 7. If the remaining dataset becomes too small, it's gathered and sorted
 * locally
 *
 * The algorithm maintains the invariant that the k-th element lies within the
 * active range on each iteration, progressively narrowing the search space.
 *
 * @tparam RandomIt Random access iterator type
 * @tparam OrderCompare Binary predicate for ordering comparison (typically
 * std::less)
 * @tparam EqualCompare Binary predicate for equality comparison (typically
 * std::equal_to)
 *
 * @param begin Iterator to the beginning of local data range
 * @param end Iterator to the end of local data range
 * @param k Zero-based index of the element to find (0 = smallest element)
 * @param comm MPI communicator containing all participating processes
 * @param ord_comp Comparison function for ordering elements
 * @param eq_comp Comparison function for equality testing
 *
 * @return The k-th smallest element from the global dataset
 *
 * @note Time Complexity: O(n) expected, O(n²) worst case (like standard
 * QuickSelect)
 * @note Space Complexity: O(p) where p is the number of MPI processes
 * @note The function uses a deterministic PRNG (seed=155728) for reproducible
 * results
 * @note For single process (world_size == 1), falls back to standard
 * std::nth_element
 *
 * @warning The function assumes that the value_type is MPI-serializable
 * @warning All processes must call this function collectively (MPI collective
 * operation)
 *
 * Example Usage:
 * ```cpp
 * std::vector<int> local_data = {5, 2, 8, 1, 9};
 * int third_smallest = dist_quickselect(local_data.begin(), local_data.end(),
 *                                       2, MPI_COMM_WORLD,
 *                                       std::less<int>(),
 * std::equal_to<int>());
 * ```
 */
template <typename RandomIt, class OrderCompare, class EqualCompare>
typename RandomIt::value_type dist_quickselect(RandomIt begin, RandomIt end,
                                               int k, MPI_Comm comm,
                                               OrderCompare ord_comp,
                                               EqualCompare eq_comp) {
  using value_type = typename RandomIt::value_type;

  // Get MPI process information
  auto world_rank = comm_rank(comm);
  auto world_size = comm_size(comm);
  auto dtype = mpi_traits<value_type>::datatype();

  // Optimization: Fall back to standard algorithm for single process
  if (world_size == 1) {
    std::nth_element(begin, begin + k, end, ord_comp);
    return *std::max_element(begin, begin + k, ord_comp);
  }

  // Use deterministic PRNG for reproducible pivot selection across runs
  std::default_random_engine g(155728);
  value_type pivot;

  // Buffers for tracking data distribution across processes
  std::vector<size_t> local_sizes(world_size), local_start(world_size);

  // Initialize local working range (will be narrowed in each iteration)
  auto local_begin = begin;
  auto local_end = end;
  bool found =
      false;  // Flag indicating if k-th element is found in equal partition
  size_t total_n;

  // Main QuickSelect iteration loop
  while (true) {
    // Step 1: Compute and gather local element counts from all processes
    size_t local_n = std::distance(local_begin, local_end);
    total_n = total_gather_and_exclusive_scan(local_n, local_sizes, local_start,
                                              comm);

    // Termination condition: If remaining data is smaller than process count,
    // switch to gather-and-sort approach for efficiency
    if (total_n < world_size) break;

    // Step 2: Randomly select a pivot index from the global range
    int pivot_idx = g() % (total_n - 1);

    // Step 3: Determine which process owns the selected pivot element
    int pivot_owner;
    {
      auto it =
          std::upper_bound(local_start.begin(), local_start.end(), pivot_idx);
      pivot_owner = std::distance(local_start.begin(), it) - 1;
    }

    // Step 4: Extract pivot value and broadcast to all processes
    if (world_rank == pivot_owner) {
      pivot_idx -= local_start[world_rank];  // Convert to local index
      pivot = *(local_begin + pivot_idx);
    }
    MPI_Bcast(&pivot, 1, dtype, pivot_owner, comm);

    // Step 5: Perform local 3-way partitioning (Less-Equal-Greater)
    auto [l_begin, e_begin, g_begin, _end] =
        leg_partition(local_begin, local_end, pivot, ord_comp, eq_comp);

    // Step 6: Compute local partition sizes and aggregate globally
    const size_t local_L = std::distance(l_begin, e_begin);  // Less than pivot
    const size_t local_E = std::distance(e_begin, g_begin);  // Equal to pivot
    const size_t local_G = std::distance(g_begin, _end);  // Greater than pivot
    size_t local_partition_sizes[3] = {local_L, local_E, local_G};
    size_t total_partition_sizes[3];
    allreduce(local_partition_sizes, total_partition_sizes, 3, MPI_SUM, comm);
    const size_t total_L = total_partition_sizes[0];
    const size_t total_E = total_partition_sizes[1];
    const size_t total_G = total_partition_sizes[2];

    // Step 7: Determine which partition contains the k-th element and recurse
    if (k <= total_L) {
      // k-th element is in the "Less" partition
      local_begin = l_begin;
      local_end = e_begin;
    } else if (k <= (total_L + total_E)) {
      // k-th element is in the "Equal" partition (we found it!)
      found = true;
      break;
    } else {
      // k-th element is in the "Greater" partition
      local_begin = g_begin;
      local_end = _end;
      k -= total_L + total_E;  // Adjust k for the smaller subproblem
    }
  }

  // Handle termination cases
  if (!found) {
    // Case 1: Remaining dataset is too small for efficient distributed
    // processing Gather all remaining data to all processes and perform local
    // selection

    std::vector<int> recv_size, displ;
    int local_n = std::distance(local_begin, local_end);
    total_n = total_gather_and_exclusive_scan(local_n, recv_size, displ, comm);

    // Collect all remaining data using MPI_Allgatherv
    std::vector<value_type> gathered_data(total_n);
    MPI_Allgatherv(&(*local_begin), local_n, dtype, gathered_data.data(),
                   recv_size.data(), displ.data(), dtype, comm);

    // Perform final selection on the gathered data
    // Partial sort using nth_element (optimal O(n))
    std::nth_element(gathered_data.begin(), gathered_data.begin() + k,
                     gathered_data.end(), ord_comp);
    pivot = *std::max_element(gathered_data.begin(), gathered_data.begin() + k,
                              ord_comp);
  }
  // Case 2: found == true, meaning k-th element was found in equal partition
  // The pivot value is already the k-th smallest element

  return pivot;
}
#endif /* MACIS_ENABLE_MPI */

}  // namespace macis
