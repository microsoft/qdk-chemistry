/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <sparsexx/sparsexx_config.hpp>
#if SPARSEXX_ENABLE_MKL

#include <mkl_dss.h>

#include <chrono>
#include <iostream>
#include <sparsexx/matrix_types/type_traits.hpp>
#include <sparsexx/spsolve/bunch_kaufman.hpp>

#include "mkl_exceptions.hpp"
#include "mkl_type_traits.hpp"

namespace sparsexx::spsolve {

namespace detail::mkl {

using sparsexx::detail::mkl::int_type;
using sparsexx::detail::mkl::mkl_dss_exception;

class mkl_dss_handle {
  _MKL_DSS_HANDLE_t handle_;

 public:
  mkl_dss_handle(int_type opts = MKL_DSS_DEFAULTS) {
    auto err = dss_create(handle_, opts);
    if (err != MKL_DSS_SUCCESS) throw mkl_dss_exception(err);
  }

  virtual ~mkl_dss_handle() noexcept {
    auto opts = MKL_DSS_DEFAULTS;
    dss_delete(handle_, opts);
  }

  auto& handle() { return handle_; }
  const auto& handle() const { return handle_; }
};

class mkl_dss_solver_base {
  mkl_dss_handle handle_;
  int_type matrix_type_;

 public:
  mkl_dss_solver_base() = delete;
  virtual ~mkl_dss_solver_base() noexcept = default;

  mkl_dss_solver_base(int_type symmetry, int_type type, int_type m, int_type n,
                      int_type nnz, const int_type* rowptr,
                      const int_type* colind,
                      int_type handle_opts = MKL_DSS_DEFAULTS)
      : handle_(handle_opts), matrix_type_(type) {
    auto err = dss_define_structure(handle_.handle(), symmetry, rowptr, m, n,
                                    colind, nnz);
    if (err != MKL_DSS_SUCCESS) throw mkl_dss_exception(err);
  }

  void reorder(int_type opts = MKL_DSS_DEFAULTS) {
    auto err = dss_reorder(handle_.handle(), opts, 0);
    if (err != MKL_DSS_SUCCESS) throw mkl_dss_exception(err);

    double reorder_dur;
    err = dss_statistics(handle_.handle(), opts, "ReorderTime", &reorder_dur);
    if (err != MKL_DSS_SUCCESS) throw mkl_dss_exception(err);
    std::cout << "REORDER TIME = " << reorder_dur << std::endl;
  }

  auto& handle() { return handle_.handle(); }
  const auto& handle() const { return handle_.handle(); }

  auto matrix_type() const { return matrix_type_; }

  void factorize_real(const void* nzval) {
    auto& h = handle();
    auto err = dss_factor_real(h, matrix_type_, nzval);
    if (err != MKL_DSS_SUCCESS) throw mkl_dss_exception(err);
  }

  void solve_real(int_type NRHS, const void* B, int_type LDB, void* X,
                  int_type LDX) {
    assert(LDB == LDX);
    auto& h = handle();
    auto opts = MKL_DSS_DEFAULTS;
    auto err = dss_solve_real(h, opts, B, NRHS, X);
    if (err != MKL_DSS_SUCCESS) throw mkl_dss_exception(err);
  }

  std::tuple<int64_t, int64_t, int64_t> get_inertia() {
    std::vector<double> inertia(3);
    auto opts = MKL_DSS_DEFAULTS;
    auto err = dss_statistics(handle(), opts, "Inertia", inertia.data());
    if (err != MKL_DSS_SUCCESS) throw mkl_dss_exception(err);

    int64_t p = inertia[0];
    int64_t n = inertia[1];
    int64_t z = inertia[2];

    return {p, n, z};
  }
};

template <typename SpMatType,
          typename = sparsexx::detail::enable_if_csr_matrix_t<SpMatType> >
struct mkl_dss_bunch_kaufman_solver : public bunch_kaufman_pimpl<SpMatType>,
                                      mkl_dss_solver_base {
  using value_type = typename SpMatType::value_type;

  virtual ~mkl_dss_bunch_kaufman_solver() noexcept = default;

  mkl_dss_bunch_kaufman_solver(int_type m, int_type n, int_type nnz,
                               const int_type* rowptr, const int_type* colind,
                               int_type handle_opts = MKL_DSS_DEFAULTS)
      : mkl_dss_solver_base(MKL_DSS_SYMMETRIC, MKL_DSS_INDEFINITE, m, n, nnz,
                            rowptr, colind, handle_opts) {
    reorder();
  }

  void factorize(const value_type* nzval) { factorize_real(nzval); }

  mkl_dss_bunch_kaufman_solver(const SpMatType& A,
                               int_type handle_opts = MKL_DSS_DEFAULTS)
      : mkl_dss_bunch_kaufman_solver(A.m(), A.n(), A.nnz(), A.rowptr().data(),
                                     A.colind().data(), handle_opts) {}

  void factorize(const SpMatType& A) override { factorize(A.nzval().data()); }

  void solve(int64_t NRHS, const value_type* B, int64_t LDB, value_type* X,
             int64_t LDX) override {
    solve_real(NRHS, B, LDB, X, LDX);
  };

  void solve(int64_t NRHS, value_type* B, int64_t LDB) override {
    std::allocator<value_type> alloc;
    auto* X = alloc.allocate(LDB * NRHS);
    solve(NRHS, B, LDB, X, LDB);
    std::copy_n(X, LDB * NRHS, B);
    alloc.deallocate(X, LDB * NRHS);
  };

  std::tuple<int64_t, int64_t, int64_t> get_inertia() override {
    return mkl_dss_solver_base::get_inertia();
  }
};

}  // namespace detail::mkl

template <typename SpMatType, typename... Args>
std::unique_ptr<detail::bunch_kaufman_pimpl<SpMatType> >
create_mkl_bunch_kaufman_solver(Args&&... args) {
  return std::make_unique<
      detail::mkl::mkl_dss_bunch_kaufman_solver<SpMatType> >(
      std::forward<Args>(args)...);
}

}  // namespace sparsexx::spsolve
#endif /* SPARSEXX_ENABLE_MKL */
