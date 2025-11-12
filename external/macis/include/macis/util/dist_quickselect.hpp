/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <algorithm>
#include <macis/util/mpi.hpp>
#include <random>
#include <vector>

namespace macis {

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

template <typename RandomIt, class OrderCompare, class EqualCompare>
typename RandomIt::value_type dist_quickselect(RandomIt begin, RandomIt end,
                                               int k, MPI_Comm comm,
                                               OrderCompare ord_comp,
                                               EqualCompare eq_comp) {
  using value_type = typename RandomIt::value_type;

  auto world_rank = comm_rank(comm);
  auto world_size = comm_size(comm);
  auto dtype = mpi_traits<value_type>::datatype();

  // Fall through for 1 MPI rank
  if (world_size == 1) {
    std::nth_element(begin, begin + k, end, ord_comp);
    return *std::max_element(begin, begin + k, ord_comp);
  }

  std::default_random_engine g(155728);  // Deterministic PRNG
  value_type pivot;

  std::vector<size_t> local_sizes(world_size), local_start(world_size);

  auto local_begin = begin;
  auto local_end = end;
  bool found = false;
  size_t total_n;
  while (true) {
    // Compute local and global element counts
    size_t local_n = std::distance(local_begin, local_end);
    total_n = total_gather_and_exclusive_scan(local_n, local_sizes, local_start,
                                              comm);
    if (total_n < world_size) break;

    // Select a pivot index
    int pivot_idx = g() % (total_n - 1);

    // Get owning rank for pivot element
    int pivot_owner;
    {
      auto it =
          std::upper_bound(local_start.begin(), local_start.end(), pivot_idx);
      pivot_owner = std::distance(local_start.begin(), it) - 1;
    }

    // Select pivot and broadcast
    if (world_rank == pivot_owner) {
      pivot_idx -= local_start[world_rank];
      pivot = *(local_begin + pivot_idx);
    }
    MPI_Bcast(&pivot, 1, dtype, pivot_owner, comm);

    // Do local partitioning
    auto [l_begin, e_begin, g_begin, _end] =
        leg_partition(local_begin, local_end, pivot, ord_comp, eq_comp);

    // Compute local and global partition sizes
    const size_t local_L = std::distance(l_begin, e_begin);
    const size_t local_E = std::distance(e_begin, g_begin);
    const size_t local_G = std::distance(g_begin, _end);
    size_t local_partition_sizes[3] = {local_L, local_E, local_G};
    size_t total_partition_sizes[3];
    allreduce(local_partition_sizes, total_partition_sizes, 3, MPI_SUM, comm);
    const size_t total_L = total_partition_sizes[0];
    const size_t total_E = total_partition_sizes[1];
    const size_t total_G = total_partition_sizes[2];

    if (k <= total_L) {
      local_begin = l_begin;
      local_end = e_begin;
    } else if (k <= (total_L + total_E)) {
      found = true;
      break;
    } else {
      local_begin = g_begin;
      local_end = _end;
      k -= total_L + total_E;
    }
  }

  if (!found) {
    // Gather local data
    std::vector<int> recv_size, displ;
    int local_n = std::distance(local_begin, local_end);
    total_n = total_gather_and_exclusive_scan(local_n, recv_size, displ, comm);

    std::vector<value_type> gathered_data(total_n);
    MPI_Allgatherv(&(*local_begin), local_n, dtype, gathered_data.data(),
                   recv_size.data(), displ.data(), dtype, comm);

    // Obtain remaining kth-element
#if 0
    std::sort(gathered_data.begin(), gathered_data.end(), ord_comp);
    pivot = gathered_data[k-1];
#else
    std::nth_element(gathered_data.begin(), gathered_data.begin() + k,
                     gathered_data.end(), ord_comp);
    pivot = *std::max_element(gathered_data.begin(), gathered_data.begin() + k,
                              ord_comp);
#endif /* 0 */
  }

  return pivot;
}
#endif /* MACIS_ENABLE_MPI */

}  // namespace macis
