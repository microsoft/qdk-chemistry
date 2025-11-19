/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <macis/util/mpi.hpp>

#ifdef MACIS_ENABLE_MPI
#include <catch2/catch_template_test_macros.hpp>
#include <macis/util/dist_quickselect.hpp>

#include "ut_common.hpp"

TEMPLATE_TEST_CASE("Distributed Quickselect", "[mpi]", std::less<int>,
                   std::greater<int>) {
  MPI_Barrier(MPI_COMM_WORLD);

  // MPI Info
  const auto world_size = macis::comm_size(MPI_COMM_WORLD);
  const auto world_rank = macis::comm_rank(MPI_COMM_WORLD);

  std::vector<int> local_data;
  switch (world_rank) {
    case 0:
      local_data = {0, 1, 3, 5, 5, 5, 7};
      break;
    case 1:
      local_data = {17, 0, 0, 16, 700};
      break;
    case 2:
      local_data = {2};
      break;
    case 3:
      local_data = {15, 16, 19, 700};
      break;
    default:
      throw std::runtime_error("This Unit Test Requires NP <= 4");
  }

  using comp_type = TestType;
  comp_type comp;

  // Gather Global Data
  std::vector<int> local_sizes, displ;
  int local_n = local_data.size();
  size_t total_n = macis::total_gather_and_exclusive_scan(
      local_n, local_sizes, displ, MPI_COMM_WORLD);

  std::vector<int> global_data(total_n);
  auto mpi_dtype = macis::mpi_traits<int>::datatype();
  MPI_Allgatherv(local_data.data(), local_data.size(), mpi_dtype,
                 global_data.data(), local_sizes.data(), displ.data(),
                 mpi_dtype, MPI_COMM_WORLD);

  // Sort global data
  std::sort(global_data.begin(), global_data.end(), comp);

  // Check all possible kth elements
  for (int k = 0; k < total_n; ++k) {
    std::vector<int> data_copy = local_data;
    auto kth_element =
        macis::dist_quickselect(data_copy.begin(), data_copy.end(), k + 1,
                                MPI_COMM_WORLD, comp, std::equal_to<int>{});
    REQUIRE(global_data[k] == kth_element);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}
#endif
