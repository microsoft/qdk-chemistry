/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/hamiltonian_generator/base.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <blas.hh>
#include <lapack.hh>
#pragma GCC diagnostic pop

namespace macis {

template <typename Scalar>
HamiltonianGeneratorBase<Scalar>::HamiltonianGeneratorBase(matrix_span_t T,
                                                           rank4_span_t V)
    : norb_(T.extent(0)),
      norb2_(norb_ * norb_),
      norb3_(norb2_ * norb_),
      T_pq_(T),
      V_pqrs_(V) {
  generate_integral_intermediates_(V_pqrs_);
}

template <typename Scalar>
void HamiltonianGeneratorBase<Scalar>::generate_integral_intermediates_(
    rank4_span_t V) {
  if (V.extent(0) != norb_ or V.extent(1) != norb_ or V.extent(2) != norb_ or
      V.extent(3) != norb_)
    throw std::runtime_error("V has incorrect dimensions");

  size_t no = norb_;
  size_t no2 = no * no;
  size_t no3 = no2 * no;
  size_t no4 = no3 * no;

  // G(i,j,k,l) = V(i,j,k,l) - V(i,l,k,j)
  G_pqrs_data_ = std::vector<Scalar>(V.data_handle(), V.data_handle() + no4);
  G_pqrs_ = rank4_span_t(G_pqrs_data_.data(), no, no, no, no);
  for (auto i = 0ul; i < no; ++i)
    for (auto j = 0ul; j < no; ++j)
      for (auto k = 0ul; k < no; ++k)
        for (auto l = 0ul; l < no; ++l) {
          G_pqrs_(i, j, k, l) -= V(i, l, k, j);
        }

  // G_red(i,j,k) = G(i,j,k,k) = G(k,k,i,j)
  // V_red(i,j,k) = V(i,j,k,k) = V(k,k,i,j)
  G_red_data_.resize(no3);
  V_red_data_.resize(no3);
  G_red_ = rank3_span_t(G_red_data_.data(), no, no, no);
  V_red_ = rank3_span_t(V_red_data_.data(), no, no, no);
  for (auto j = 0ul; j < no; ++j)
    for (auto i = 0ul; i < no; ++i)
      for (auto k = 0ul; k < no; ++k) {
        G_red_(k, i, j) = G_pqrs_(k, k, i, j);
        V_red_(k, i, j) = V(k, k, i, j);
      }

  // G2_red(i,j) = 0.5 * G(i,i,j,j)
  // V2_red(i,j) = V(i,i,j,j)
  G2_red_data_.resize(no2);
  V2_red_data_.resize(no2);
  G2_red_ = matrix_span<Scalar>(G2_red_data_.data(), no, no);
  V2_red_ = matrix_span<Scalar>(V2_red_data_.data(), no, no);
  for (auto j = 0ul; j < no; ++j)
    for (auto i = 0ul; i < no; ++i) {
      G2_red_(i, j) = 0.5 * G_pqrs_(i, i, j, j);
      V2_red_(i, j) = V(i, i, j, j);
    }
}

template <typename Scalar>
void HamiltonianGeneratorBase<Scalar>::rotate_hamiltonian_ordm(
    const Scalar* ordm) {
  // SVD on ordm to get natural orbitals
  std::vector<Scalar> natural_orbitals(ordm, ordm + norb2_);
  std::vector<Scalar> S(norb_);
  lapack::gesvd(lapack::Job::OverwriteVec, lapack::Job::NoVec, norb_, norb_,
                natural_orbitals.data(), norb_, S.data(), NULL, 1, NULL, 1);

  std::vector<Scalar> tmp(norb3_ * norb_), tmp2(norb3_ * norb_);

  // Transform T
  // T <- N**H * T * N
  auto* T_pq_ptr = T_pq_.data_handle();
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             norb_, norb_, norb_, 1., T_pq_ptr, norb_, natural_orbitals.data(),
             norb_, 0., tmp.data(), norb_);
  blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, norb_,
             norb_, norb_, 1., natural_orbitals.data(), norb_, tmp.data(),
             norb_, 0., T_pq_ptr, norb_);

  // Transorm V

  // 1st Quarter
  // (pj|kl) = N(i,p) (ij|kl)
  // W(p,jkl) = N(i,p) * V(i,jkl)
  blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, norb_,
             norb3_, norb_, 1., natural_orbitals.data(), norb_,
             V_pqrs_.data_handle(), norb_, 0., tmp.data(), norb_);

  // 2nd Quarter
  // (pq|kl) = N(j,q) (pj|kl)
  // W_kl(p,q) = V_kl(p,j) N(j,q)
  for (auto kl = 0; kl < norb2_; ++kl) {
    auto* V_kl = tmp.data() + kl * norb2_;
    auto* W_kl = tmp2.data() + kl * norb2_;
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               norb_, norb_, norb_, 1., V_kl, norb_, natural_orbitals.data(),
               norb_, 0., W_kl, norb_);
  }

  // 3rd Quarter
  // (pq|rl) = N(k,r) (pq|kl)
  // W_l(pq,r) = V_l(pq,k) N(k,r)
  for (auto l = 0; l < norb_; ++l) {
    auto* V_l = tmp2.data() + l * norb3_;
    auto* W_l = tmp.data() + l * norb3_;
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               norb2_, norb_, norb_, 1., V_l, norb2_, natural_orbitals.data(),
               norb_, 0., W_l, norb2_);
  }

  // 4th Quarter
  // (pq|rs) = N(l,s) (pq|rl)
  // W(pqr,s) = V(pqr,l) N(l,s)
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             norb3_, norb_, norb_, 1., tmp.data(), norb3_,
             natural_orbitals.data(), norb_, 0., V_pqrs_.data_handle(), norb3_);

  // Regenerate intermediates
  generate_integral_intermediates_(V_pqrs_);
}

}  // namespace macis
