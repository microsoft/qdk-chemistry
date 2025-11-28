// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/core/moeri.h>
#include <qdk/chemistry/scf/core/types.h>
#ifdef QDK_CHEMISTRY_ENABLE_GPU
#include <qdk/chemistry/scf/util/gpu/cuda_helper.h>
#endif
#include <Eigen/Core>
#include <blas.hh>

namespace qdk::chemistry::scf {
MOERI::MOERI(std::shared_ptr<ERI> eri)
    : eri_(eri)
#ifdef QDK_CHEMISTRY_ENABLE_GPU
      ,
      handle_(std::make_shared<cutensor::TensorHandle>())
#endif
{
}
MOERI::~MOERI() noexcept = default;

void MOERI::compute(size_t nb, size_t nt, const double* C, double* out) {
  if (nt > nb)
    throw std::runtime_error(
        "MOERI does not support num_molecular_orbitals > num_atomic_orbitals");
  const size_t nt2 = nt * nt;
  const size_t nt3 = nt2 * nt;
  const size_t nb2 = nb * nb;
  const size_t nb3 = nb2 * nb;
  std::vector<double> tmp1(std::max(nt * nb3, nb * nt3)), tmp2(nt2 * nb2);

#ifdef QDK_CHEMISTRY_ENABLE_GPU

  // Compute first quarter transformation via some "good" technique
  eri_->quarter_trans(nt, C, tmp1.data());

  auto tmp1_d = cuda::alloc<double>(nt * nb * nb * nb);
  auto tmp2_d = cuda::alloc<double>(nt * nt * nb * nb);
  auto C_d = cuda::alloc<double>(nb * nt);
  CUDA_CHECK(cudaMemcpy(C_d->data(), C, nb * nt * sizeof(double),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemcpy(tmp1_d->data(), tmp1.data(),
                        tmp1.size() * sizeof(double), cudaMemcpyHostToDevice));

  // Compute remaining transformations via cuTensor
  std::vector<int64_t> extents_q1 = {(int64_t)nb, (int64_t)nb, (int64_t)nb,
                                     (int64_t)nt},
                       extents_q2 = {(int64_t)nb, (int64_t)nb, (int64_t)nt,
                                     (int64_t)nt},
                       extents_q3 = {(int64_t)nb, (int64_t)nt, (int64_t)nt,
                                     (int64_t)nt},
                       extents_q4 = {(int64_t)nt, (int64_t)nt, (int64_t)nt,
                                     (int64_t)nt},
                       extents_mat = {(int64_t)nb, (int64_t)nt};

  auto q1desc = std::make_shared<cutensor::TensorDesc>(
      *handle_, 4, extents_q1.data(), CUTENSOR_R_64F);
  auto q2desc = std::make_shared<cutensor::TensorDesc>(
      *handle_, 4, extents_q2.data(), CUTENSOR_R_64F);
  auto q3desc = std::make_shared<cutensor::TensorDesc>(
      *handle_, 4, extents_q3.data(), CUTENSOR_R_64F);
  auto q4desc = std::make_shared<cutensor::TensorDesc>(
      *handle_, 4, extents_q4.data(), CUTENSOR_R_64F);
  auto matdesc = std::make_shared<cutensor::TensorDesc>(
      *handle_, 2, extents_mat.data(), CUTENSOR_R_64F);

  std::vector<int> q1_ind = {'i', 'j', 'k', 's'}, q2_ind = {'i', 'j', 'r', 's'},
                   q3_ind = {'i', 'q', 'r', 's'}, q4_ind = {'p', 'q', 'r', 's'};

  std::vector<int> mat_ind = {'k', 'r'};
  auto q2cont = std::make_shared<cutensor::ContractionData>(
      handle_, q1desc, q1_ind, matdesc, mat_ind, q2desc, q2_ind);

  mat_ind = {'j', 'q'};
  auto q3cont = std::make_shared<cutensor::ContractionData>(
      handle_, q2desc, q2_ind, matdesc, mat_ind, q3desc, q3_ind);

  mat_ind = {'i', 'p'};
  auto q4cont = std::make_shared<cutensor::ContractionData>(
      handle_, q3desc, q3_ind, matdesc, mat_ind, q4desc, q4_ind);

  q2cont->contract(1.0, tmp1_d->data(), C_d->data(), 0.0, tmp2_d->data());
  q3cont->contract(1.0, tmp2_d->data(), C_d->data(), 0.0, tmp1_d->data());
  q4cont->contract(1.0, tmp1_d->data(), C_d->data(), 0.0, tmp2_d->data());

  CUDA_CHECK(cudaMemcpy(out, tmp2_d->data(), nt * nt * nt * nt * sizeof(double),
                        cudaMemcpyDeviceToHost));
#else  // QDK_CHEMISTRY_ENABLE_GPU

  Eigen::Map<const RowMajorMatrix> C_rm(C, nb, nt);
  // Compute first quarter transformation via some "good" technique
  eri_->quarter_trans(nt, C_rm.data(), tmp1.data());

  // All the following conversion use col major C
  Eigen::MatrixXd C_cm = C_rm;

  // 2nd Quarter
  // TMP2(p,q,k,l) = C(j,q) * TMP1(p,j,k,l)
  // TMP2_kl(p,q)  = TMP1_kl(p,j) * C(j,q)
  for (size_t kl = 0; kl < nb2; ++kl) {
    auto TMP1_kl = tmp1.data() + kl * nb * nt;
    auto TMP2_kl = tmp2.data() + kl * nt2;
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, nt,
               nt, nb, 1.0, TMP1_kl, nt, C_cm.data(), nb, 0.0, TMP2_kl, nt);
  }

  // 3rd Quarter
  // TMP1(p,q,r,l) = C(k,r) * TMP2(p,q,k,l)
  // TMP1_l(pq,r)  = TMP2_l(pq,k) * C(k,r)
  for (size_t l = 0; l < nb; ++l) {
    auto TMP2_l = tmp2.data() + l * nt2 * nb;
    auto TMP1_l = tmp1.data() + l * nt3;
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               nt2, nt, nb, 1.0, TMP2_l, nt2, C_cm.data(), nb, 0.0, TMP1_l,
               nt2);
  }

  // 4th Quarter
  // Y(p,q,r,s) = C(l,s) * TMP1(p,q,r,l)
  // Y(pqr,s) = V(pqr,l) * C(l,s)
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, nt3,
             nt, nb, 1.0, tmp1.data(), nt3, C_cm.data(), nb, 0.0, out, nt3);

#endif  // QDK_CHEMISTRY_ENABLE_GPU
}
}  // namespace qdk::chemistry::scf
