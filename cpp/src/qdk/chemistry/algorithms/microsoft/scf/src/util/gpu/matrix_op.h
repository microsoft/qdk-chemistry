// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <cublasLt.h>
#include <qdk/chemistry/scf/util/gpu/cuda_helper.h>

#include <memory>
#include <vector>

/**
 * @brief Check cuBLAS status and abort on error
 *
 * Macro that evaluates a cuBLAS function call and checks the return status.
 * If an error is detected, prints error information and terminates the program.
 *
 * @param x cuBLAS function call to check
 */
#define CUBLAS_CHECK(x)                                                        \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != CUBLAS_STATUS_SUCCESS) {                                        \
      printf("Error status %d in file %s line %d\n", err, __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

namespace qdk::chemistry::scf::matrix_op {
class CuBLASLTHelper;

/**
 * @brief Singleton helper for cuBLAS LT handle and workspace management
 *
 * Manages cuBLASLt context and workspace memory per-device.
 * Uses singleton pattern to ensure one instance per GPU device.
 */
class CuBLASLTHelper {
 public:
  /**
   * @brief Get the cuBLASLt handle for the current device
   *
   * @return cublasLtHandle_t handle for cuBLAS LT operations
   */
  static cublasLtHandle_t get_handle() { return get_instance()->handle_; }

  /**
   * @brief Get the workspace size
   *
   * @return Size of workspace buffer in bytes (512 MB)
   */
  static size_t get_work_size() { return work_size_; }

  /**
   * @brief Get the workspace buffer pointer
   *
   * @return Device pointer to workspace memory
   */
  static void* get_workspace() { return get_instance()->workspace_; }

  /**
   * @brief Destructor - cleans up workspace and handle
   */
  ~CuBLASLTHelper() {
    CUDA_CHECK(cudaFree(workspace_));
    CUBLAS_CHECK(cublasLtDestroy(handle_));
  }

 private:
  /**
   * @brief Get singleton instance for current device
   *
   * Creates one helper instance per GPU device (up to 32 devices).
   *
   * @return Shared pointer to CuBLASLTHelper for current device
   */
  static std::shared_ptr<CuBLASLTHelper> get_instance() {
    static std::vector<std::shared_ptr<CuBLASLTHelper>> helper(32);
    int did;
    cudaGetDevice(&did);
    if (helper[did] == nullptr) {
      helper[did] = std::shared_ptr<CuBLASLTHelper>(new CuBLASLTHelper());
    }
    return helper[did];
  }

  /**
   * @brief Private constructor - creates handle and allocates workspace
   */
  CuBLASLTHelper() {
    CUBLAS_CHECK(cublasLtCreate(&handle_));
    CUDA_CHECK(cudaMalloc(&workspace_, work_size_));
  }

  cublasLtHandle_t handle_;  ///< cuBLASLt handle for current device
  static const size_t work_size_ =
      (1ULL << 29);  ///< Workspace size: 512 MB (2^29 bytes)
  void* workspace_;  ///< Device memory workspace for cuBLASLt operations
};

/**
 * @brief Type traits for cuBLAS data types
 *
 * Template struct providing CUDA type and compute type mappings
 * for different scalar types.
 *
 * @tparam U Scalar type (float or double)
 */
template <typename U>
struct CuBLASTypeTraits;

/**
 * @brief Type traits specialization for double precision
 */
template <>
struct CuBLASTypeTraits<double> {
  static const cudaDataType_t cuda_type =
      CUDA_R_64F;  ///< CUDA type for 64-bit float (double)
  static const cublasComputeType_t compute_type =
      CUBLAS_COMPUTE_64F;     ///< cuBLAS compute type for double
  typedef double ScalarType;  ///< Scalar type alias
};

/**
 * @brief Type traits specialization for single precision
 */
template <>
struct CuBLASTypeTraits<float> {
  static const cudaDataType_t cuda_type =
      CUDA_R_32F;  ///< CUDA type for 32-bit float
  static const cublasComputeType_t compute_type =
      CUBLAS_COMPUTE_32F;    ///< cuBLAS compute type for float
  typedef float ScalarType;  ///< Scalar type alias
};

/**
 * @brief Matrix descriptor for cuBLASLt operations
 *
 * Encapsulates matrix layout information including dimensions, batch size,
 * and transpose status. Used to describe matrices for cuBLASLt operations.
 *
 * @tparam ComputeType Precision type (float or double)
 */
template <typename ComputeType>
class MatrixDesc {
 public:
  /**
   * @brief Construct matrix descriptor with full parameters
   *
   * @param batch Number of matrices in batch
   * @param n Number of rows
   * @param m Number of columns
   * @param transpose Whether matrix should be transposed
   */
  MatrixDesc(int batch, int n, int m, bool transpose)
      : batch_(batch), n_(n), m_(m), transpose_(transpose) {
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &desc_, CuBLASTypeTraits<ComputeType>::cuda_type, n_, m_, m_));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        desc_, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_, sizeof(batch_)));
    int64_t stride = 1LL * n_ * m_;
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        desc_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
        sizeof(stride)));
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        desc_, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
  }

  /**
   * @brief Construct matrix descriptor without transpose flag (defaults to
   * false)
   *
   * @param batch Number of matrices in batch
   * @param n Number of rows
   * @param m Number of columns
   */
  MatrixDesc(int batch, int n, int m) : MatrixDesc(batch, n, m, false) {}

  /**
   * @brief Construct single matrix descriptor (batch size = 1)
   *
   * @param n Number of rows
   * @param m Number of columns
   */
  MatrixDesc(int n, int m) : MatrixDesc(1, n, m, false) {}

  /**
   * @brief Copy constructor
   *
   * @param o Source MatrixDesc to copy from
   */
  MatrixDesc(const MatrixDesc& o)
      : MatrixDesc(o.batch_, o.n_, o.m_, o.transpose_) {}

  /**
   * @brief Destructor - destroys cuBLASLt layout descriptor
   */
  ~MatrixDesc() { CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(desc_)); }

  /**
   * @brief Get the cuBLASLt layout descriptor
   *
   * @return cublasLtMatrixLayout_t descriptor for cuBLASLt operations
   */
  cublasLtMatrixLayout_t get_cublasLt_desc() const { return desc_; }

  /**
   * @brief Get batch size
   *
   * @return Number of matrices in batch
   */
  int get_batch() const { return batch_; }

  /**
   * @brief Get number of rows
   *
   * @return Row count
   */
  int get_n() const { return n_; }

  /**
   * @brief Get number of columns
   *
   * @return Column count
   */
  int get_m() const { return m_; }

  /**
   * @brief Check if matrix is transposed
   *
   * @return true if transposed, false otherwise
   */
  bool is_transpose() const { return transpose_; }

 private:
  int batch_;                    ///< Number of matrices in batch
  int n_;                        ///< Number of rows
  int m_;                        ///< Number of columns
  bool transpose_;               ///< Whether matrix is transposed
  cublasLtMatrixLayout_t desc_;  ///< cuBLASLt matrix layout descriptor
};

/**
 * @brief Batched matrix multiplication with scaling (extended version)
 *
 * Computes: C = alpha * (A × B) + beta * C
 * where A, B, C can be batched matrices with optional transpose.
 *
 * Example usage:
 * @code
 *   bmm_ex(2.0, a, {2, 3, 4}, b, {2, 4, 3}, 3.0, c);
 *   // Computes: c = 2.0*(a×b) + 3.0*c
 * @endcode
 *
 * @tparam ComputeType Precision type (float or double)
 * @param alpha Scaling factor for A×B
 * @param a Device pointer to matrix A
 * @param a_desc Descriptor for matrix A (dimensions, batch, transpose)
 * @param b Device pointer to matrix B
 * @param b_desc Descriptor for matrix B (dimensions, batch, transpose)
 * @param beta Scaling factor for C (existing values)
 * @param c Device pointer to matrix C (input/output)
 */
template <typename ComputeType>
void bmm_ex(const ComputeType alpha, const ComputeType* a,
            const MatrixDesc<ComputeType>& a_desc, const ComputeType* b,
            const MatrixDesc<ComputeType>& b_desc, const ComputeType beta,
            ComputeType* c) {
  MatrixDesc<ComputeType> c_desc(
      a_desc.get_batch(),
      a_desc.is_transpose() ? a_desc.get_m() : a_desc.get_n(),
      b_desc.is_transpose() ? b_desc.get_n() : b_desc.get_m());

  cublasLtMatmulDesc_t op_desc = nullptr;

  CUBLAS_CHECK(cublasLtMatmulDescCreate(
      &op_desc, CuBLASTypeTraits<ComputeType>::compute_type,
      CuBLASTypeTraits<ComputeType>::cuda_type));

  auto op_a = (a_desc.is_transpose() ? CUBLAS_OP_T : CUBLAS_OP_N);
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
  auto op_b = (b_desc.is_transpose() ? CUBLAS_OP_T : CUBLAS_OP_N);
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));

  cublasLtMatmulPreference_t preference = nullptr;
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));

  const size_t work_size = CuBLASLTHelper::get_work_size();
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &work_size,
      sizeof(work_size)));

  int n_algo = 0;
  cublasLtMatmulHeuristicResult_t algo[1] = {};
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      CuBLASLTHelper::get_handle(), op_desc, a_desc.get_cublasLt_desc(),
      b_desc.get_cublasLt_desc(), c_desc.get_cublasLt_desc(),
      c_desc.get_cublasLt_desc(), preference, 1, algo, &n_algo));

  CUBLAS_CHECK(cublasLtMatmul(
      CuBLASLTHelper::get_handle(), op_desc, &alpha, a,
      a_desc.get_cublasLt_desc(), b, b_desc.get_cublasLt_desc(), &beta, c,
      c_desc.get_cublasLt_desc(), c, c_desc.get_cublasLt_desc(),
      &(algo[0].algo), CuBLASLTHelper::get_workspace(),
      CuBLASLTHelper::get_work_size(), 0));

  CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
  CUBLAS_CHECK(cublasLtMatmulDescDestroy(op_desc));
}

/**
 * @brief Batched matrix multiplication (simple version)
 *
 * Computes: C = A × B
 *
 * Usage example:
 * @code
 *   bmm(a, {2, 3, 4}, b, {2, 4, 3}, c);
 *   // Result c has shape (2, 3, 3)
 * @endcode
 *
 * @tparam ComputeType Precision type (float or double)
 * @param a Device pointer to matrix A
 * @param a_desc Descriptor for matrix A
 * @param b Device pointer to matrix B
 * @param b_desc Descriptor for matrix B
 * @param c Device pointer to output matrix C
 */
template <typename ComputeType>
void bmm(const ComputeType* a, const MatrixDesc<ComputeType>& a_desc,
         const ComputeType* b, const MatrixDesc<ComputeType>& b_desc,
         ComputeType* c) {
  bmm_ex(1.0, a, a_desc, b, b_desc, 0.0, c);
}

/**
 * @brief Matrix addition with scaling (extended version)
 *
 * Computes: C = alpha * A + beta * B
 *
 * @tparam ComputeType Precision type (float or double)
 * @param alpha Scaling factor for A
 * @param a Device pointer to matrix A
 * @param a_desc Descriptor for matrix A
 * @param beta Scaling factor for B
 * @param b Device pointer to matrix B
 * @param b_desc Descriptor for matrix B
 * @param c Device pointer to output matrix C
 */
template <typename ComputeType>
void add_ex(const ComputeType alpha, const ComputeType* a,
            const MatrixDesc<ComputeType>& a_desc, const ComputeType beta,
            const ComputeType* b, const MatrixDesc<ComputeType>& b_desc,
            ComputeType* c) {
  MatrixDesc<ComputeType> c_desc(
      a_desc.get_batch(),
      a_desc.is_transpose() ? a_desc.get_m() : a_desc.get_n(),
      a_desc.is_transpose() ? a_desc.get_n() : a_desc.get_m());

  cublasLtMatrixTransformDesc_t op_desc = nullptr;

  CUBLAS_CHECK(cublasLtMatrixTransformDescCreate(
      &op_desc, CuBLASTypeTraits<ComputeType>::cuda_type));

  auto op_a = (a_desc.is_transpose() ? CUBLAS_OP_T : CUBLAS_OP_N);
  CUBLAS_CHECK(cublasLtMatrixTransformDescSetAttribute(
      op_desc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &op_a, sizeof(op_a)));
  auto op_b = (b_desc.is_transpose() ? CUBLAS_OP_T : CUBLAS_OP_N);
  CUBLAS_CHECK(cublasLtMatrixTransformDescSetAttribute(
      op_desc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &op_b, sizeof(op_b)));

  CUBLAS_CHECK(cublasLtMatrixTransform(CuBLASLTHelper::get_handle(), op_desc,
                                       &alpha, a, a_desc.get_cublasLt_desc(),
                                       &beta, b, b_desc.get_cublasLt_desc(), c,
                                       c_desc.get_cublasLt_desc(), 0));

  CUBLAS_CHECK(cublasLtMatrixTransformDescDestroy(op_desc));
}

/**
 * @brief Matrix addition (simple version)
 *
 * Computes: C = A + B
 *
 * @note All matrices must have the same descriptor (dimensions and layout).
 *
 * @tparam ComputeType Precision type (float or double)
 * @param a Device pointer to matrix A
 * @param b Device pointer to matrix B
 * @param desc Descriptor for all matrices (shared dimensions)
 * @param c Device pointer to output matrix C
 */
template <typename ComputeType>
void add(const ComputeType* a, const ComputeType* b,
         const MatrixDesc<ComputeType>& desc, ComputeType* c) {
  add_ex(ComputeType(1.0), a, desc, ComputeType(1.0), b, desc, c);
}

/**
 * @brief In-place matrix scaling
 *
 * Computes: A = scale * A
 *
 * @note This operation modifies matrix A in-place. The operation uses
 *       cuBLASLt's matrix transform with beta=0 and b=nullptr.
 *
 * @tparam ComputeType Precision type (float or double)
 * @param a Device pointer to matrix A (modified in-place)
 * @param desc Descriptor for matrix A
 * @param scale Scaling factor to apply
 */
template <typename ComputeType>
void scale(ComputeType* a, const MatrixDesc<ComputeType>& desc,
           ComputeType scale) {
  ComputeType* b = nullptr;
  add_ex(scale, a, desc, ComputeType(0.0), b, desc, a);
}

/**
 * @brief Matrix transpose operation
 *
 * Computes: B = A^T
 *
 * @note Transpose is implemented using cuBLASLt matrix transform with
 *       the transpose flag set in the descriptor.
 *
 * @tparam ComputeType Precision type (float or double)
 * @param a Device pointer to input matrix A
 * @param desc Descriptor for matrix A (n×m)
 * @param b Device pointer to output matrix B (m×n)
 */
template <typename ComputeType>
void transpose(const ComputeType* a, const MatrixDesc<ComputeType>& desc,
               ComputeType* b) {
  ComputeType* p = nullptr;
  MatrixDesc<ComputeType> a_desc(desc.get_batch(), desc.get_n(), desc.get_m(),
                                 true);
  add_ex(ComputeType(1.0), a, a_desc, ComputeType(0.0), p, desc, b);
}

}  // namespace qdk::chemistry::scf::matrix_op
