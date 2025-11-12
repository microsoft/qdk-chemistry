/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <sparsexx/io/read_mm.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/util/graph.hpp>

#include "catch2/catch.hpp"

TEST_CASE("Adjacency", "[graph]") {
  using mat_type = sparsexx::csr_matrix<double, int32_t>;
  std::string ref_fname = SPARSEXX_DATA_DIR "/mycielskian3/mycielskian3.mtx";
  auto A = sparsexx::read_mm<mat_type>(ref_fname);

  auto [adj_rowptr, adj_colind] = sparsexx::extract_adjacency(A, 1);
  REQUIRE(adj_rowptr == A.rowptr());
  REQUIRE(adj_colind == A.colind());
}
