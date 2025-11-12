// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <qdk/chemistry/scf/config.h>
#ifdef QDK_CHEMISTRY_ENABLE_GPU
#include <cuda_runtime.h>
#include <cutensor.h>
#include <qdk/chemistry/scf/util/gpu/cuda_helper.h>
#include <spdlog/spdlog.h>

#include <vector>

namespace qdk::chemistry::scf::cutensor {

/**
 * @brief Macro to check cuTENSOR API calls for errors and throw on failure
 *
 * This macro wraps cuTENSOR library calls and checks their return status.
 * If an error is detected, it logs a critical error message with the error
 * description and throws an exception.
 *
 * Example usage:
 * @code
 * CUTENSOR_CHECK(cutensorCreate(&handle));
 * CUTENSOR_CHECK(cutensorCreateTensorDescriptor(...));
 * @endcode
 *
 * @param x cuTENSOR API function call to check
 */
#define CUTENSOR_CHECK(x)                                                  \
  {                                                                        \
    const auto err = x;                                                    \
    if (err != CUTENSOR_STATUS_SUCCESS) {                                  \
      spdlog::critical("cuTENSOR error: {}", cutensorGetErrorString(err)); \
      throw std::runtime_error("cuTENSOR Exception");                      \
    }                                                                      \
  };

/**
 * @brief Macro to check cuTENSOR API calls for errors and abort on failure
 *
 * This macro wraps cuTENSOR library calls and checks their return status.
 * If an error is detected, it logs a critical error message with the error
 * description and aborts the program rather than throws an exception. This
 * is for use in destructors, which are no-throw by convention
 *
 * Example usage:
 * @code
 * CUTENSOR_CHECK(cutensorCreate(&handle));
 * CUTENSOR_CHECK(cutensorCreateTensorDescriptor(...));
 * @endcode
 *
 * @param x cuTENSOR API function call to check
 */
#define CUTENSOR_CHECK_ABORT(x)                                            \
  {                                                                        \
    const auto err = x;                                                    \
    if (err != CUTENSOR_STATUS_SUCCESS) {                                  \
      spdlog::critical("cuTENSOR error: {}", cutensorGetErrorString(err)); \
      std::abort();                                                        \
    }                                                                      \
  };

/**
 * @brief RAII wrapper for cuTENSOR tensor descriptor
 *
 * Manages the lifecycle of a cuTENSOR tensor descriptor, ensuring proper
 * creation and destruction. Tensor descriptors define the shape, data type,
 * memory layout, and alignment of tensors used in cuTENSOR operations.
 *
 * The descriptor can be created with automatic stride computation (row-major)
 * or with explicitly specified strides for custom memory layouts.
 *
 * @note This class is non-copyable and non-movable to prevent descriptor handle
 * duplication
 */
class TensorDesc {
  cutensorTensorDescriptor_t desc_;  ///< cuTENSOR tensor descriptor handle

 public:
  /**
   * @brief Create tensor descriptor with automatic row-major stride computation
   *
   * Creates a tensor descriptor with row-major memory layout. Strides are
   * automatically computed from dimensions assuming contiguous row-major
   * ordering.
   *
   * @param handle cuTENSOR library handle
   * @param nrank Number of tensor dimensions
   * @param dims Array of dimension sizes (length must be nrank)
   * @param dtype Data type of tensor elements (e.g., CUTENSOR_R_64F for double)
   * @param align Memory alignment in bytes (default: 128 bytes)
   *
   */
  TensorDesc(cutensorHandle_t handle, int64_t nrank, int64_t dims[],
             cutensorDataType_t dtype, unsigned align = 128);

  /**
   * @brief Create tensor descriptor with explicit strides for custom memory
   * layout
   *
   * Creates a tensor descriptor with user-specified strides, allowing for
   * non-contiguous or column-major memory layouts. Strides define the number
   * of elements between consecutive entries along each dimension.
   *
   * @param handle cuTENSOR library handle
   * @param nrank Number of tensor dimensions
   * @param dims Array of dimension sizes (length must be nrank)
   * @param strides Array of strides for each dimension (length must be nrank)
   * @param dtype Data type of tensor elements
   * @param align Memory alignment in bytes (default: 128 bytes)
   *
   */
  TensorDesc(cutensorHandle_t handle, int64_t nrank, int64_t dims[],
             int64_t strides[], cutensorDataType_t dtype, unsigned align = 128);

  /**
   * @brief Destroy the tensor descriptor
   *
   * Automatically cleans up the cuTENSOR descriptor when the object is
   * destroyed.
   */
  ~TensorDesc() noexcept;

  TensorDesc() = delete;
  TensorDesc(TensorDesc&&) noexcept = delete;
  TensorDesc(const TensorDesc&) = delete;

  /**
   * @brief Get the underlying cuTENSOR descriptor handle
   *
   * Provides access to the raw cuTENSOR descriptor for use in library API
   * calls.
   *
   * @return cutensorTensorDescriptor_t handle to the managed descriptor
   */
  inline auto desc() const noexcept { return desc_; }
};

/**
 * @brief RAII wrapper for cuTENSOR library handle
 *
 * Manages the lifecycle of a cuTENSOR handle, which represents the library
 * context for all cuTENSOR operations. The handle must be created before
 * any cuTENSOR operations and is automatically destroyed when no longer needed.
 *
 * @note This class is non-copyable and non-movable to prevent handle
 * duplication
 */
class TensorHandle {
  cutensorHandle_t handle_;  ///< cuTENSOR library handle

 public:
  /**
   * @brief Create a cuTENSOR library handle
   *
   * Initializes the cuTENSOR library context. This must be done before
   * creating tensor descriptors or performing tensor operations.
   */
  TensorHandle();

  /**
   * @brief Destroy the cuTENSOR handle
   *
   * Automatically cleans up the cuTENSOR context when the object is destroyed.
   */
  ~TensorHandle() noexcept;

  TensorHandle(const TensorHandle&) = delete;
  TensorHandle(TensorHandle&&) noexcept = delete;

  /**
   * @brief Implicit conversion to cutensorHandle_t for seamless API usage
   *
   * Allows TensorHandle objects to be used directly in cuTENSOR API calls
   * that expect a cutensorHandle_t parameter.
   *
   * @return cutensorHandle_t The underlying cuTENSOR handle
   */
  inline operator cutensorHandle_t() const noexcept { return handle_; }
};

/**
 * @brief Manage cuTENSOR data for generalized tensor contractions
 *
 * Encapsulates all state required to perform tensor contractions of the form:
 *   C = alpha * contraction(A, B) + beta * C
 *
 * where the contraction operation is specified using Einstein summation
 * notation.
 *
 * Example usage:
 * @code
 * auto handle = std::make_shared<TensorHandle>();
 * auto descA = std::make_shared<TensorDesc>(handle, 2, dimsA, CUTENSOR_R_64F);
 * auto descB = std::make_shared<TensorDesc>(handle, 2, dimsB, CUTENSOR_R_64F);
 * auto descC = std::make_shared<TensorDesc>(handle, 2, dimsC, CUTENSOR_R_64F);
 *
 * // Matrix multiplication C = A * B
 * ContractionData contraction(handle, descA, {'i','j'},
 *                             descB, {'j','k'}, descC, {'i','k'});
 * contraction.contract(1.0, d_A, d_B, 0.0, d_C);
 * @endcode
 *
 * @note Workspace is allocated once during construction and reused for all
 * contractions
 */
class ContractionData {
  using tensor_hndl_ptr = std::shared_ptr<TensorHandle>;
  using tensor_desc_ptr = std::shared_ptr<TensorDesc>;

  /*** INPUT DATA ***/
  tensor_hndl_ptr handle_;  ///< The cuTENSOR context handle
  tensor_desc_ptr descA_;   ///< Descriptor of tensor A
  tensor_desc_ptr descB_;   ///< Descriptor of tensor B
  tensor_desc_ptr descC_;   ///< Descriptor of tensor C
  cutensorAlgo_t algo_;     ///< cuTENSOR optimization algorithm
  std::vector<int>
      indA_;  ///< Indices of A tensor in contraction (einsum notation)
  std::vector<int>
      indB_;  ///< Indices of B tensor in contraction (einsum notation)
  std::vector<int>
      indC_;  ///< Indices of C tensor in contraction (einsum notation)

  /*** EPHEMERAL DATA ***/
  cutensorOperationDescriptor_t
      descCont_;  ///< cuTENSOR contraction operation descriptor
  cutensorPlanPreference_t planPref_;  ///< Unoptimized cuTENSOR plan context
  cutensorPlan_t plan_;                ///< cuTENSOR (stateful) execution plan

  void* workspace_;        ///< Workspace memory for tensor contraction
  uint64_t workspace_sz_;  ///< Size of workspace in bytes

  /**
   * @brief Create the descriptor for the specified tensor contraction
   *
   * Initializes the cuTENSOR operation descriptor (descCont_) based on the
   * tensor descriptors and index patterns provided.
   */
  void create_contraction_();

  /**
   * @brief Create the plan preference for the specified algorithm
   *
   * Initializes the plan preference (planPref_) with the selected algorithm.
   * The plan preference guides cuTENSOR's optimization strategy for the
   * contraction operation.
   */
  void create_planpref_();

  /**
   * @brief Compute initial workspace estimate based on plan preference
   *
   * Queries cuTENSOR for an estimate of required workspace memory based on
   * the current operation descriptor and plan preference. This estimate is
   * used to allocate sufficient workspace before creating the execution plan.
   *
   * @return Estimated workspace size in bytes
   */
  uint64_t get_workspace_estimate_();

  /**
   * @brief Generate the optimized execution plan
   *
   * Creates the cuTENSOR execution plan (plan_) optimized for the specified
   * contraction operation. The plan is generated based on the operation
   * descriptor, plan preference, and available workspace.
   *
   * @param workspace_est Estimated workspace size in bytes
   */
  void create_plan_(uint64_t workspace_est);

  /**
   * @brief Allocate workspace memory for tensor contraction
   *
   * Allocates device memory (workspace_) for cuTENSOR to use during the
   * contraction operation. The size is determined by get_workspace_estimate_().
   * Memory is freed automatically in the destructor.
   */
  void allocate_workspace_();

 public:
  /**
   * @brief Construct and initialize cuTENSOR contraction data
   *
   * Sets up all necessary cuTENSOR structures for performing a generalized
   * tensor contraction specified by Einstein summation notation. The index
   * arrays define how dimensions of input tensors A and B map to the output
   * tensor C.
   *
   * @param handle Shared pointer to cuTENSOR library handle
   * @param descA Shared pointer to descriptor for tensor A
   * @param indA Index array for tensor A in Einstein notation
   * @param descB Shared pointer to descriptor for tensor B
   * @param indB Index array for tensor B in Einstein notation
   * @param descC Shared pointer to descriptor for tensor C
   * @param indC Index array for tensor C in Einstein notation
   * @param algo cuTENSOR optimization algorithm to use
   *
   */
  ContractionData(tensor_hndl_ptr handle, tensor_desc_ptr descA,
                  std::vector<int> indA, tensor_desc_ptr descB,
                  std::vector<int> indB, tensor_desc_ptr descC,
                  std::vector<int> indC,
                  cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT);

  /**
   * @brief Cleanup all cuTENSOR resources
   */
  ~ContractionData() noexcept;

  /**
   * @brief Execute the tensor contraction on GPU
   *
   * Performs the contraction operation: C = alpha * contraction(A, B) + beta *
   * C
   *
   * All input pointers must point to device memory with shapes matching the
   * descriptors provided during construction. The operation is enqueued on
   * the specified CUDA stream for asynchronous execution.
   *
   * This method can be called multiple times with different data but the same
   * tensor shapes. The execution plan is reused for efficiency.
   *
   * @param alpha Scalar multiplier for the contraction result
   * @param A Device pointer to tensor A data
   * @param B Device pointer to tensor B data
   * @param beta Scalar multiplier for the existing values in C
   * @param C Device pointer to tensor C data (input and output)
   * @param stream CUDA stream for asynchronous execution (default: stream 0)
   *
   * @note A, B, and C must be valid device pointers with sufficient allocated
   * memory
   * @note The contraction is asynchronous; use cudaStreamSynchronize if CPU
   * synchronization needed
   */
  void contract(double alpha, const double* A, const double* B, double beta,
                double* C, cudaStream_t stream = 0);
};

}  // namespace qdk::chemistry::scf::cutensor
#endif  // QDK_CHEMISTRY_ENABLE_GPU
