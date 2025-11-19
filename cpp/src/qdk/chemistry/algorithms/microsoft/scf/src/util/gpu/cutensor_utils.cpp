// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/config.h>
#ifdef QDK_CHEMISTRY_ENABLE_GPU
#include <qdk/chemistry/scf/util/gpu/cutensor_utils.h>

namespace qdk::chemistry::scf::cutensor {

/**
 * @brief Creates row-major stride array for tensor operations
 *
 * Computes the stride array for a tensor stored in row-major (C-style) order.
 * The stride determines how many elements to skip when moving along each
 * dimension of the tensor. For row-major order, the last dimension has
 * stride 1, and earlier dimensions have strides equal to the product of
 * all later dimension sizes.
 *
 * @param nrank Number of tensor dimensions
 * @param dims Array containing the size of each dimension
 * @return Vector containing the stride for each dimension
 *
 * @note The input dims array must have at least nrank elements
 */
auto make_row_major_strides(int64_t nrank, int64_t dims[]) {
  std::vector<int64_t> stride(nrank);
  stride.back() = 1;
  for (int i = 1; i < nrank; ++i)
    stride[nrank - i - 1] = stride[nrank - i] * dims[nrank - i];
  return stride;
}

TensorDesc::TensorDesc(cutensorHandle_t handle, int64_t nrank, int64_t dims[],
                       cutensorDataType_t dtype, unsigned align) {
  // Row Major Data
  auto stride = make_row_major_strides(nrank, dims);
  CUTENSOR_CHECK(cutensorCreateTensorDescriptor(handle, &desc_, nrank, dims,
                                                stride.data(), dtype, align));
}

TensorDesc::TensorDesc(cutensorHandle_t handle, int64_t nrank, int64_t dims[],
                       int64_t strides[], cutensorDataType_t dtype,
                       unsigned align) {
  CUTENSOR_CHECK(cutensorCreateTensorDescriptor(handle, &desc_, nrank, dims,
                                                strides, dtype, align));
}

TensorDesc::~TensorDesc() noexcept {
  CUTENSOR_CHECK_ABORT(cutensorDestroyTensorDescriptor(desc_));
}

TensorHandle::TensorHandle() { CUTENSOR_CHECK(cutensorCreate(&handle_)); }

TensorHandle::~TensorHandle() noexcept {
  CUTENSOR_CHECK_ABORT(cutensorDestroy(handle_));
}

void ContractionData::create_contraction_() {
  CUTENSOR_CHECK(cutensorCreateContraction(
      *handle_, &descCont_, descA_->desc(), indA_.data(), CUTENSOR_OP_IDENTITY,
      descB_->desc(), indB_.data(), CUTENSOR_OP_IDENTITY, descC_->desc(),
      indC_.data(), CUTENSOR_OP_IDENTITY, descC_->desc(), indC_.data(),
      CUTENSOR_COMPUTE_DESC_64F));
}

void ContractionData::create_planpref_() {
  CUTENSOR_CHECK(cutensorCreatePlanPreference(*handle_, &planPref_, algo_,
                                              CUTENSOR_JIT_MODE_NONE));
}

uint64_t ContractionData::get_workspace_estimate_() {
  uint64_t workspaceEst;
  const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
  CUTENSOR_CHECK(cutensorEstimateWorkspaceSize(*handle_, descCont_, planPref_,
                                               workspacePref, &workspaceEst));
  return workspaceEst;
}

void ContractionData::create_plan_(uint64_t workspace_est) {
  CUTENSOR_CHECK(cutensorCreatePlan(*handle_, &plan_, descCont_, planPref_,
                                    workspace_est));
}

void ContractionData::allocate_workspace_() {
  CUTENSOR_CHECK(cutensorPlanGetAttribute(
      *handle_, plan_, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &workspace_sz_,
      sizeof(workspace_sz_)));

  if (workspace_sz_) {
    CUDA_CHECK(cudaMallocAsync(&workspace_, workspace_sz_, 0));
    CUDA_CHECK(cudaStreamSynchronize(0));
  } else
    workspace_ = nullptr;
}

ContractionData::ContractionData(tensor_hndl_ptr handle, tensor_desc_ptr descA,
                                 std::vector<int> indA, tensor_desc_ptr descB,
                                 std::vector<int> indB, tensor_desc_ptr descC,
                                 std::vector<int> indC, cutensorAlgo_t algo)
    : handle_(handle),
      descA_(descA),
      descB_(descB),
      descC_(descC),
      indA_(indA),
      indB_(indB),
      indC_(indC),
      algo_(algo) {
  create_contraction_();
  create_planpref_();
  create_plan_(get_workspace_estimate_());
  allocate_workspace_();
}

ContractionData::~ContractionData() noexcept {
  CUTENSOR_CHECK_ABORT(cutensorDestroyPlan(plan_));
  CUTENSOR_CHECK_ABORT(cutensorDestroyPlanPreference(planPref_));
  CUTENSOR_CHECK_ABORT(cutensorDestroyOperationDescriptor(descCont_));
  if (workspace_) CUDA_CHECK(cudaFree(workspace_));
}

void ContractionData::contract(double alpha, const double* A, const double* B,
                               double beta, double* C, cudaStream_t stream) {
  CUTENSOR_CHECK(cutensorContract(*handle_, plan_, &alpha, A, B, &beta, C, C,
                                  workspace_, workspace_sz_, stream));
}

}  // namespace qdk::chemistry::scf::cutensor
#endif  // QDK_CHEMISTRY_ENABLE_GPU
