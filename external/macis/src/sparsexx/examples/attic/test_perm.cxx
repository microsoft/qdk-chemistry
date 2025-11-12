/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <sparsexx/io/read_mm.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/matrix_types/dense_conversions.hpp>
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/util/reorder.hpp>
#include <sparsexx/util/submatrix.hpp>

template <typename SpMatType>
void output_sparse_matrix(std::ostream& out, const SpMatType& A) {
  std::vector<sparsexx::detail::value_type_t<SpMatType>> A_dense(A.m() * A.n(),
                                                                 0.);

  sparsexx::convert_to_dense(A, A_dense.data(), A.m());

  out << std::scientific << std::setprecision(4);
  for (int64_t i = 0; i < A.m(); ++i) {
    for (int64_t j = 0; j < A.n(); ++j)
      out << std::setw(12) << A_dense[i + j * A.m()];
    out << std::endl;
  }
}

int main(int argc, char** argv) {
  assert(argc == 2);
  using spmat_type = sparsexx::csr_matrix<double, int32_t>;
  auto A = sparsexx::read_mm<spmat_type>(std::string(argv[1]));
  const int N = A.m();

  std::vector<int32_t> perm = {0, 2, 3, 1};
  auto Arp = sparsexx::permute_rows(A, perm);
  auto Acp = sparsexx::permute_cols(A, perm);

  auto Arcp_1 = sparsexx::permute_rows(Acp, perm);
  auto Arcp_2 = sparsexx::permute_cols(Arp, perm);
  auto Arcp_3 = sparsexx::permute_rows_cols(A, perm, perm);

  std::cout << "PERM ";
  std::copy(perm.begin(), perm.end(),
            std::ostream_iterator<int32_t>(std::cout, ", "));
  std::cout << std::endl;

  std::cout << "A" << std::endl;
  output_sparse_matrix(std::cout, A);

  std::cout << std::endl;
  std::cout << "Arp" << std::endl;
  output_sparse_matrix(std::cout, Arp);

  std::cout << std::endl;
  std::cout << "Acp" << std::endl;
  output_sparse_matrix(std::cout, Acp);

  std::cout << std::endl;
  std::cout << "Arcp_1" << std::endl;
  output_sparse_matrix(std::cout, Arcp_1);

  std::cout << std::endl;
  std::cout << "Arcp_2" << std::endl;
  output_sparse_matrix(std::cout, Arcp_2);

  std::cout << std::endl;
  std::cout << "Arcp_3" << std::endl;
  output_sparse_matrix(std::cout, Arcp_3);

  if (Arcp_1.rowptr() != Arcp_2.rowptr())
    throw std::runtime_error("1-2 Rowptr");
  if (Arcp_1.colind() != Arcp_2.colind())
    throw std::runtime_error("1-2 Colind");
  if (Arcp_1.nzval() != Arcp_2.nzval()) throw std::runtime_error("1-2 Nzval");

  if (Arcp_1.rowptr() != Arcp_3.rowptr())
    throw std::runtime_error("1-3 Rowptr");
  if (Arcp_1.colind() != Arcp_3.colind())
    throw std::runtime_error("1-3 Colind");
  if (Arcp_1.nzval() != Arcp_3.nzval()) throw std::runtime_error("1-3 Nzval");

  return 0;
}
