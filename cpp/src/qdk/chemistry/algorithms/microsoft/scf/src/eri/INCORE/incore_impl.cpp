// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "incore_impl.h"

#include <qdk/chemistry/scf/config.h>

#include <stdexcept>
#ifdef QDK_CHEMISTRY_ENABLE_GPU
#include <cuda_runtime.h>
#include <qdk/chemistry/scf/util/gpu/cuda_helper.h>
#endif
#include <qdk/chemistry/scf/util/libint2_util.h>
#include <spdlog/spdlog.h>

#include <blas.hh>

#define INCORE_ERI_GEN_DEBUG 0xF00
#define INCORE_ERI_CON_HOST 0x0F0
#ifdef QDK_CHEMISTRY_ENABLE_GPU
#define QDK_CHEMISTRY_INCORE_ERI_STRATEGY 0
#else
#define QDK_CHEMISTRY_INCORE_ERI_STRATEGY INCORE_ERI_CON_HOST
#endif

namespace qdk::chemistry::scf::incore {

ERI::ERI(bool unr, const BasisSet& basis, ParallelConfig mpi, double omega) {
  unrestricted_ = unr;
  obs_ = libint2_util::convert_to_libint_basisset(basis);
  omega_ = omega;
  basis_mode_ = basis.mode;
  mpi_ = mpi;

  // Distribute on first index
  auto ni_per_rank = obs_.nbf() / mpi_.world_size;
  loc_i_st_ = mpi_.world_rank * ni_per_rank;
  loc_i_en_ = (mpi_.world_rank == mpi_.world_size - 1)
                  ? obs_.nbf()
                  : loc_i_st_ + ni_per_rank;

  // Generate the ERIs INCORE
  generate_eri_();
}

void ERI::generate_eri_() {
  // Allocate and populate ERIs on host
  const size_t num_atomic_orbitals = obs_.nbf();
  const size_t num_atomic_orbitals2 = num_atomic_orbitals * num_atomic_orbitals;
  const size_t num_atomic_orbitals3 =
      num_atomic_orbitals2 * num_atomic_orbitals;
  const size_t eri_sz = num_atomic_orbitals3 * (loc_i_en_ - loc_i_st_);

  const bool is_rsx = std::abs(omega_) > 1e-12;

  if (!mpi_.world_rank)
    spdlog::debug("Generating ERIs via Libint2 {}",
                  is_rsx ? "omega = " + std::to_string(omega_) : "");
#if (QDK_CHEMISTRY_INCORE_ERI_STRATEGY & INCORE_ERI_GEN_DEBUG) > 0
  h_eri_ =
      libint2_util::debug_eri(basis_mode_, obs_, 0.0, loc_i_st_, loc_i_en_);
  if (is_rsx)
    h_eri_erf_ = libint2_util::debug_eri(basis_mode_, obs_, omega_, loc_i_st_,
                                         loc_i_en_);
#else
  h_eri_ = libint2_util::opt_eri(basis_mode_, obs_, 0.0, loc_i_st_, loc_i_en_);
  if (is_rsx)
    h_eri_erf_ =
        libint2_util::opt_eri(basis_mode_, obs_, omega_, loc_i_st_, loc_i_en_);
#endif

#if (QDK_CHEMISTRY_INCORE_ERI_STRATEGY & INCORE_ERI_CON_HOST) > 0 || \
    (QDK_CHEMISTRY_INCORE_ERI_STRATEGY & INCORE_ERI_CON_HOST) > 0
  if (!mpi_.world_rank) spdlog::debug("Saving ERIs on Host Memory");
#else
  // Allocate ERIs on the device and ship data
  if (!mpi_.world_rank) {
    spdlog::debug("Saving ERIs in Device Memory");
    spdlog::debug("Using cuTensor for GPU ERI Contraction");
  }
  CUDA_CHECK(cudaMallocAsync(&d_eri_, eri_sz * sizeof(double), 0));
  CUDA_CHECK(cudaStreamSynchronize(0));
  CUDA_CHECK(cudaMemcpy(d_eri_, h_eri_.get(), sizeof(double) * eri_sz,
                        cudaMemcpyHostToDevice));
  h_eri_ = nullptr;  // Clear host memory
  if (is_rsx) {
    CUDA_CHECK(cudaMallocAsync(&d_eri_erf_, eri_sz * sizeof(double), 0));
    CUDA_CHECK(cudaStreamSynchronize(0));
    CUDA_CHECK(cudaMemcpy(d_eri_erf_, h_eri_erf_.get(), sizeof(double) * eri_sz,
                          cudaMemcpyHostToDevice));
    h_eri_erf_ = nullptr;  // Clear host memory
  }
#endif

#ifdef QDK_CHEMISTRY_ENABLE_GPU
  // Setup cuTensor
  handle_ = std::make_shared<cutensor::TensorHandle>();

  std::vector<int64_t> extents4 = {loc_i_en_ - loc_i_st_, num_atomic_orbitals,
                                   num_atomic_orbitals, num_atomic_orbitals};
  std::vector<int64_t> extents2_P = {unrestricted_ ? 2 : 1, num_atomic_orbitals,
                                     num_atomic_orbitals};
  std::vector<int64_t> extents2_R = {
      unrestricted_ ? 2 : 1, loc_i_en_ - loc_i_st_, num_atomic_orbitals};
  std::vector<int64_t> strides2 = {num_atomic_orbitals * num_atomic_orbitals,
                                   num_atomic_orbitals,
                                   1};  // Output is full space
  descERI_ = std::make_shared<cutensor::TensorDesc>(
      *handle_, 4, extents4.data(), CUTENSOR_R_64F);
  descR_ = std::make_shared<cutensor::TensorDesc>(
      *handle_, 3, extents2_R.data(), strides2.data(), CUTENSOR_R_64F);
  descP_ = std::make_shared<cutensor::TensorDesc>(
      *handle_, 3, extents2_P.data(), strides2.data(), CUTENSOR_R_64F);

  std::vector<int> j_ind = {'i', 'j', 'k', 'l'}, k_ind = {'i', 'k', 'j', 'l'},
                   p_ind = {'m', 'k', 'l'}, r_ind = {'m', 'i', 'j'};
  couContraction_ = std::make_unique<cutensor::ContractionData>(
      handle_, descERI_, j_ind, descP_, p_ind, descR_, r_ind);
  exxContraction_ = std::make_unique<cutensor::ContractionData>(
      handle_, descERI_, k_ind, descP_, p_ind, descR_, r_ind);
#endif
}

void ERI::build_JK(const double* P, double* J, double* K, double alpha,
                   double beta, double omega) {
  if (std::abs(omega - omega_) > 1e-12) {
    throw std::runtime_error(fmt::format(
        "Inconsistent OMEGA passed to ERIINCORE (passed {:.2e}, stored {:.2e})",
        omega, omega_));
  }

  const bool is_rsx = std::abs(omega_) > 1e-12;

  const size_t num_atomic_orbitals = obs_.nbf();
  const size_t num_atomic_orbitals2 = num_atomic_orbitals * num_atomic_orbitals;
  const size_t mat_size = (unrestricted_ ? 2 : 1) * num_atomic_orbitals2;

  // Clear out data
  if (J) std::fill_n(J, mat_size, 0.0);
  if (K) std::fill_n(K, mat_size, 0.0);

#if (QDK_CHEMISTRY_INCORE_ERI_STRATEGY & INCORE_ERI_CON_HOST) > 0
  const size_t num_atomic_orbitals3 =
      num_atomic_orbitals2 * num_atomic_orbitals;
  const size_t num_atomic_orbitals4 =
      num_atomic_orbitals2 * num_atomic_orbitals2;
  const auto* h_eri_ptr = h_eri_.get();
  const auto* h_eri_erf_ptr = h_eri_erf_.get();

  const auto* Pa = P;
  const auto* Pb = unrestricted_ ? Pa + num_atomic_orbitals2 : nullptr;

  if (J)
    for (size_t idm = 0; idm < (unrestricted_ ? 2 : 1); ++idm) {
      blas::gemv(blas::Layout::ColMajor, blas::Op::Trans, num_atomic_orbitals2,
                 num_atomic_orbitals2, 1.0, h_eri_ptr, num_atomic_orbitals2,
                 P + idm * num_atomic_orbitals2, 1, 0.0,
                 J + idm * num_atomic_orbitals2, 1);
    }

  if (K)
    for (size_t idm = 0; idm < (unrestricted_ ? 2 : 1); ++idm)
      for (size_t i = loc_i_st_; i < loc_i_en_; ++i) {
        const double* eri_i = h_eri_ptr + i * num_atomic_orbitals3;
        double* K_i = K + i * num_atomic_orbitals + idm * num_atomic_orbitals2;
        for (size_t k = 0; k < num_atomic_orbitals; ++k)
          for (size_t l = 0; l < num_atomic_orbitals; ++l) {
            const auto val =
                P[k * num_atomic_orbitals + l + idm * num_atomic_orbitals2];
            const auto* eri_ikl = eri_i + k * num_atomic_orbitals2 + l;
            for (size_t j = 0; j < num_atomic_orbitals; ++j) {
              K_i[j] += (alpha + beta) * val * eri_ikl[j * num_atomic_orbitals];
            }
          }
      }

  if (K and is_rsx)
    for (size_t idm = 0; idm < (unrestricted_ ? 2 : 1); ++idm)
      for (size_t i = loc_i_st_; i < loc_i_en_; ++i) {
        const double* eri_i = h_eri_erf_ptr + i * num_atomic_orbitals3;
        double* K_i = K + i * num_atomic_orbitals + idm * num_atomic_orbitals2;
        for (size_t k = 0; k < num_atomic_orbitals; ++k)
          for (size_t l = 0; l < num_atomic_orbitals; ++l) {
            const auto val =
                P[k * num_atomic_orbitals + l + idm * num_atomic_orbitals2];
            const auto* eri_ikl = eri_i + k * num_atomic_orbitals2 + l;
            for (size_t j = 0; j < num_atomic_orbitals; ++j) {
              K_i[j] -= beta * val * eri_ikl[j * num_atomic_orbitals];
            }
          }
      }

#else  // Use GPU-accelerated contraction

  // Allocate data for P/J/K on the device and send P
  auto dP = cuda::alloc<double>(mat_size);
  auto dJ = cuda::alloc<double>(mat_size);
  auto dK = cuda::alloc<double>(mat_size);

  CUDA_CHECK(cudaMemcpy(dP->data(), P, mat_size * sizeof(double),
                        cudaMemcpyHostToDevice));
  if (J) CUDA_CHECK(cudaMemset(dJ->data(), 0, mat_size * sizeof(double)));
  if (K) CUDA_CHECK(cudaMemset(dK->data(), 0, mat_size * sizeof(double)));

  // Perform Contractions
  if (J)
    couContraction_->contract(1.0, d_eri_, dP->data(), 0.0,
                              dJ->data() + loc_i_st_ * num_atomic_orbitals);
  if (K) {
    exxContraction_->contract(alpha + beta, d_eri_, dP->data(), 0.0,
                              dK->data() + loc_i_st_ * num_atomic_orbitals);
    if (is_rsx) {
      exxContraction_->contract(-beta, d_eri_erf_, dP->data(), 1.0,
                                dK->data() + loc_i_st_ * num_atomic_orbitals);
    }
  }

  // Retrieve J/K
  if (J)
    CUDA_CHECK(cudaMemcpy(J, dJ->data(), mat_size * sizeof(double),
                          cudaMemcpyDeviceToHost));
  if (K)
    CUDA_CHECK(cudaMemcpy(K, dK->data(), mat_size * sizeof(double),
                          cudaMemcpyDeviceToHost));

#endif
}

void ERI::get_gradients(const double* P, double* dJ, double* dK, double alpha,
                        double beta, double omega) {
  throw std::runtime_error("INCORE GRADIENTS NYI");
}

void ERI::quarter_trans(size_t nt, const double* C, double* out) {
  const size_t num_atomic_orbitals = obs_.nbf();
  const size_t num_atomic_orbitals2 = num_atomic_orbitals * num_atomic_orbitals;
  const size_t num_atomic_orbitals3 =
      num_atomic_orbitals2 * num_atomic_orbitals;
  const size_t num_atomic_orbitals4 =
      num_atomic_orbitals2 * num_atomic_orbitals2;

#ifdef QDK_CHEMISTRY_ENABLE_GPU

  // Setup
  std::vector<int64_t> extents_out = {num_atomic_orbitals, num_atomic_orbitals,
                                      num_atomic_orbitals, nt};
  std::vector<int64_t> extents_mat = {num_atomic_orbitals, nt};

  auto descOut = std::make_shared<cutensor::TensorDesc>(
      *handle_, 4, extents_out.data(), CUTENSOR_R_64F);
  auto descMat = std::make_shared<cutensor::TensorDesc>(
      *handle_, 2, extents_mat.data(), CUTENSOR_R_64F);

  std::vector<int> eri_ind = {'i', 'j', 'k', 'l'},
                   out_ind = {'i', 'j', 'k', 'p'}, mat_ind = {'l', 'p'};

  auto quarterTrans = std::make_shared<cutensor::ContractionData>(
      handle_, descERI_, eri_ind, descMat, mat_ind, descOut, out_ind);

  // Allocation
  auto dC = cuda::alloc<double>(num_atomic_orbitals * nt);
  auto dOut = cuda::alloc<double>(num_atomic_orbitals3 * nt);

  CUDA_CHECK(cudaMemcpy(dC->data(), C,
                        num_atomic_orbitals * nt * sizeof(double),
                        cudaMemcpyHostToDevice));

  // 1st Quarter Contraction
  quarterTrans->contract(1.0, d_eri_, dC->data(), 0.0, dOut->data());

  CUDA_CHECK(cudaMemcpy(out, dOut->data(),
                        num_atomic_orbitals3 * nt * sizeof(double),
                        cudaMemcpyDeviceToHost));

#else

  if (omega_ > 1e-12) {
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, nt,
               num_atomic_orbitals3, num_atomic_orbitals, 1.0, C, nt,
               h_eri_erf_.get(), num_atomic_orbitals, 0.0, out, nt);
  } else {
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, nt,
               num_atomic_orbitals3, num_atomic_orbitals, 1.0, C, nt,
               h_eri_.get(), num_atomic_orbitals, 0.0, out, nt);
  }

#endif  // QDK_CHEMISTRY_ENABLE_GPU
}

ERI::~ERI() noexcept {
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  if (d_eri_) CUDA_CHECK(cudaFree(d_eri_));
  if (d_eri_erf_) CUDA_CHECK(cudaFree(d_eri_erf_));
#endif
};

std::unique_ptr<ERI> ERI::make_incore_eri(bool unr, const BasisSet& basis,
                                          ParallelConfig mpi, double omega) {
  return std::make_unique<ERI>(unr, basis, mpi, omega);
}

}  // namespace qdk::chemistry::scf::incore
