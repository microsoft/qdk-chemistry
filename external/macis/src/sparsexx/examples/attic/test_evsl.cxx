/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sparsexx/io/read_mm.hpp>
#include <sparsexx/io/read_rb.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/matrix_types/dense_conversions.hpp>
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/spblas/spmbv.hpp>
#include <sparsexx/util/submatrix.hpp>
#include <sparsexx/wrappers/mkl_dss_solver.hpp>
#include <sparsexx/wrappers/mkl_sparse_matrix.hpp>

#include "mkl.h"

extern "C" {
#include "evsl.h"
}

int main(int argc, char** argv) {
  std::cout << std::scientific << std::setprecision(8);
  assert(argc == 2);

  auto A = sparsexx::read_mm<double, sparsexx::detail::mkl::int_type>(
      std::string(argv[1]));
  const MKL_INT N = A.m();

  // auto U = sparsexx::extract_upper_triangle( A );
  // A = std::move(U);

  EVSLStart();

  csrMat A_evsl;
  A_evsl.owndata = false;
  A_evsl.nrows = A.m();
  A_evsl.ncols = A.n();
  A_evsl.nnz = A.nnz();
  A_evsl.ia = A.rowptr().data();
  A_evsl.ja = A.colind().data();
  A_evsl.a = A.nzval().data();

  // Zero ordering...
  for (auto& i : A.rowptr()) i -= A.indexing();
  for (auto& i : A.colind()) i -= A.indexing();

  SetAMatrix(&A_evsl);

  const MKL_INT mdeg = 300;
  const MKL_INT nvec = 60;
  const MKL_INT nsli = 1;

  std::array<double, 4> xintv;
  xintv[0] = 25;
  xintv[1] = 25.1;
  // xintv[0] = 0.1, xintv[1] = 0.1387;

  std::vector<double> mu(300 + 1);
  std::vector<double> alleigs(N), vinit(N);
  rand_double_device(N, vinit.data());

  MKL_INT inc = 1;
  std::cout << "|V| = " << dnrm2(&N, vinit.data(), &inc) << std::endl;

  FILE* fstats = stdout;

  double lmin, lmax;

  std::cout << "Getting Lanczos Bounds..." << std::endl;
  auto lan_bnds_st = std::chrono::high_resolution_clock::now();
  {
    auto ierr =
        LanTrbounds(200, 2000, 1e-8, vinit.data(), 1, &lmin, &lmax, fstats);
    if (ierr) throw std::runtime_error("LanTrboudns failed");
  }
  auto lan_bnds_en = std::chrono::high_resolution_clock::now();
  auto lan_bnds_dr =
      std::chrono::duration<double>(lan_bnds_en - lan_bnds_st).count();
  std::cout << "Done! (" << lan_bnds_dr << " s)" << std::endl;

  std::cout << "Lanczos Bounds = [" << lmin << ", " << lmax << "]" << std::endl;

  xintv[2] = lmin, xintv[3] = lmax;

  double ecount;
  std::cout << "Getting DOS..." << std::flush;
  auto dos_st = std::chrono::high_resolution_clock::now();
  {
    auto ierr = kpmdos(mdeg, 1, nvec, xintv.data(), mu.data(), &ecount);
    if (ierr) throw std::runtime_error("DOS failed");
  }
  auto dos_en = std::chrono::high_resolution_clock::now();
  auto dos_dr = std::chrono::duration<double>(dos_en - dos_st).count();
  std::cout << "Done! (" << dos_dr << " s)" << std::endl;

  std::cout << "Estimated eigenvalue count = " << ecount << std::endl;
  if (ecount < 0 or ecount > N) throw std::runtime_error("ECOUNT IS INVALID");

  std::vector<double> counts(nsli), sli(nsli + 1);
  MKL_INT npts = 10 * ecount;

  std::cout << "Getting Slices..." << std::flush;
  auto sli_st = std::chrono::high_resolution_clock::now();
  {
    auto ierr = spslicer(sli.data(), mu.data(), mdeg, xintv.data(), nsli, npts);
    if (ierr) throw std::runtime_error("SPSLICER failed");
  }
  auto sli_en = std::chrono::high_resolution_clock::now();
  auto sli_dr = std::chrono::duration<double>(sli_en - sli_st).count();
  std::cout << "Done! (" << sli_dr << " s)" << std::endl;

  std::cout << "Slice bounds " << std::endl;
  for (auto i = 0; i < nsli; ++i)
    std::cout << "  [ " << sli[i] << ", " << sli[i + 1] << " ]" << std::endl;

  MKL_INT fac = 1.2;
  MKL_INT nev_sli = fac * (1 + ecount / nsli);
  std::cout << "EVSL will try to produce " << nev_sli
            << " eigenvalues per slice" << std::endl;

  polparams pol;
  double tol = 1e-8;

  std::cout << "Processing slices..." << std::endl;
  for (auto i = 0; i < nsli; ++i) {
    auto a = sli[i], b = sli[i + 1];
    std::cout << std::endl;
    std::cout << "  * Processing interval [" << a << ", " << b << "]"
              << std::endl;

    MKL_INT mlan = std::min(N, std::max(4 * nev_sli, 300));
    MKL_INT max_lan_it = 3 * mlan;

    std::cout << "  * LanTR Dim = " << mlan << std::endl;

    xintv[0] = a;
    xintv[1] = b;

    set_pol_def(&pol);
    pol.damping = 2;
    pol.thresh_int = 0.8;
    pol.max_deg = 1000;
    find_pol(xintv.data(), &pol);

    std::cout << "  * Polynomial Settings" << std::endl
              << "    - type    = " << pol.type << std::endl
              << "    - deg     = " << pol.deg << std::endl
              << "    - bar     = " << pol.bar << std::endl
              << "    - gam     = " << pol.gam << std::endl;

    std::cout << "  * Running ChebLanTr..." << std::flush;
    MKL_INT nev_out;
    double *lam_slice, *Y, *res;
    auto chb_st = std::chrono::high_resolution_clock::now();
    {
      auto ierr =
          ChebLanTr(mlan, nev_sli, xintv.data(), max_lan_it, tol, vinit.data(),
                    &pol, &nev_out, &lam_slice, &Y, &res, fstats);
      if (ierr) throw std::runtime_error("ChebLanTr failed");
    }
    auto chb_en = std::chrono::high_resolution_clock::now();
    auto chb_dr = std::chrono::duration<double>(chb_en - chb_st).count();
    std::cout << "Done! (" << chb_dr << " s)" << std::endl;

    std::vector<size_t> ind(nev_out);
    std::iota(ind.begin(), ind.end(), 0);
    std::sort(ind.begin(), ind.end(),
              [&](auto i, auto j) { return lam_slice[i] < lam_slice[j]; });

    std::cout << "  * Found " << nev_out << " eigenvalues in slice"
              << std::endl;
    for (auto k = 0; k < nev_out; ++k) {
      std::cout << "    " << k << ": " << lam_slice[ind[k]] << ", "
                << res[ind[k]] << std::endl;
    }

    if (lam_slice) evsl_Free(lam_slice);
    if (Y) evsl_Free(Y);
    if (res) evsl_Free(res);
    free_pol(&pol);
  }

  EVSLFinish();
  return 0;
}
