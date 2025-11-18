/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <macis/macis_config.hpp>

#ifdef MACIS_ENABLE_MPI
#define MACIS_MPI_CODE(...) __VA_ARGS__
#else
#define MACIS_MPI_CODE(...)
#endif /* MACIS_ENABLE_MPI */

#ifdef MACIS_ENABLE_MPI
#include <mpi.h>

#include <bitset>
#include <iostream>
#include <limits>
#include <memory>

namespace macis {

namespace detail {

/// @brief Implementation class for default-aware lifetime-managed MPI_Datatype
struct mpi_datatype_impl {
  MPI_Datatype dtype;  ///< Underlying MPI_Datatype instance
  mpi_datatype_impl() = delete;
  mpi_datatype_impl(MPI_Datatype d) : dtype(d) {}

  virtual ~mpi_datatype_impl() noexcept = default;
};

/// @brief Impementation of lifetime-managed MPI_Datatype for non-default types
struct managed_mpi_datatype_impl : public mpi_datatype_impl {
  template <typename... Args>
  managed_mpi_datatype_impl(Args&&... args)
      : mpi_datatype_impl(std::forward<Args>(args)...) {}

  ~managed_mpi_datatype_impl() noexcept {
    // Free MPI_Datatype instance when out of scope
    MPI_Type_free(&dtype);
  }
};

}  // namespace detail

/**
 *  @brief Return MPI rank of this processing element
 *
 *  @param[in] comm MPI communicator for desired compute context
 *  @returns   Rank of current PE relative to `comm`
 */
inline int comm_rank(MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

/**
 *  @brief Return number of processing elements in a compute context
 *
 *  @param[in] comm MPI communicator for desired compute context
 *  @returns   Number of processing elements in context described by `comm`
 */
inline int comm_size(MPI_Comm comm) {
  int size;
  MPI_Comm_size(comm, &size);
  return size;
}

/**
 *  @brief Lifetime Managed MPI_Datatype wrapper.
 *
 *  Adds lifetime management to MPI_Datatype instances in a defaut-aware manner.
 *  i.e. custom datatypes will have a lifetime scope while defaults (e.g.
 * MPI_INT) will be assumed to be managed by the MPI runtime
 */
class mpi_datatype {
 public:
  using pimpl_type = detail::mpi_datatype_impl;
  using pimpl_pointer_type = std::unique_ptr<pimpl_type>;
  mpi_datatype(pimpl_pointer_type&& p) : pimpl_(std::move(p)) {}

  /// Return the underlying MPI_Datatype instance
  inline operator MPI_Datatype() const { return pimpl_->dtype; }

 private:
  pimpl_pointer_type pimpl_;
};

/// Generate a lifetime managed MPI_Datatype
template <typename... Args>
inline mpi_datatype make_managed_mpi_datatype(Args&&... args) {
  return mpi_datatype(std::make_unique<detail::managed_mpi_datatype_impl>(
      std::forward<Args>(args)...));
}

/// Generate a wrapped `mpi_datatype` instance for default types
template <typename... Args>
inline mpi_datatype make_mpi_datatype(Args&&... args) {
  return mpi_datatype(
      std::make_unique<detail::mpi_datatype_impl>(std::forward<Args>(args)...));
}

/// Traits class for C++ types mapped to MPI types
template <typename T>
struct mpi_traits;

#define REGISTER_MPI_TYPE(T, TYPE)                                            \
  template <>                                                                 \
  struct mpi_traits<T> {                                                      \
    using type = T;                                                           \
    inline static mpi_datatype datatype() { return make_mpi_datatype(TYPE); } \
  };

REGISTER_MPI_TYPE(char, MPI_CHAR);
REGISTER_MPI_TYPE(int, MPI_INT);
REGISTER_MPI_TYPE(double, MPI_DOUBLE);
REGISTER_MPI_TYPE(float, MPI_FLOAT);
REGISTER_MPI_TYPE(size_t, MPI_UINT64_T);
REGISTER_MPI_TYPE(int64_t, MPI_INT64_T);

#undef REGISTER_MPI_TYPE

/**
 *  @brief Generate a custom datatype for contiguous arrays of prmitive types
 *
 *  @tparam T Datatype of array elements
 *
 *  @param[in] n Number of contiguous elements
 *  @returns   MPI_Datatype wrapper for an `n`-element array of type `T`
 */
template <typename T>
mpi_datatype make_contiguous_mpi_datatype(int n) {
  auto dtype = mpi_traits<T>::datatype();
  MPI_Datatype contig_dtype;
  MPI_Type_contiguous(n, dtype, &contig_dtype);
  MPI_Type_commit(&contig_dtype);
  return make_managed_mpi_datatype(contig_dtype);
}

template <typename T>
void reduce(const T* send, T* recv, size_t count, MPI_Op op, int root,
            MPI_Comm comm) {
  auto dtype = mpi_traits<T>::datatype();

  size_t intmax = std::numeric_limits<int>::max();
  size_t nchunk = count / intmax;
  if (nchunk) throw std::runtime_error("Msg over INT_MAX not yet tested");

  MPI_Reduce(send, recv, count, dtype, op, root, comm);
}

/**
 * @brief Type-safe wrapper for MPI_Allreduce
 *
 * @param[in]     send Buffer of local data to participate in the reduction
 * operation
 * @param[in/out] recv Buffer of reduced data
 * @param[in]     count Number of elements in `send` / `recv`
 * @param[in]     op    Reduction operation
 * @param[in]     comm  MPI communicator for PEs to participate in the reduction
 * operation
 */
template <typename T>
void allreduce(const T* send, T* recv, size_t count, MPI_Op op, MPI_Comm comm) {
  auto dtype = mpi_traits<T>::datatype();

  size_t intmax = std::numeric_limits<int>::max();
  size_t nchunk = count / intmax;
  if (nchunk) throw std::runtime_error("Msg over INT_MAX not yet tested");
  for (int i = 0; i < nchunk; ++i) {
    MPI_Allreduce(send + i * intmax, recv + i * intmax, intmax, dtype, op,
                  comm);
  }

  int nrem = count % intmax;
  if (nrem) {
    MPI_Allreduce(send + nchunk * intmax, recv + nchunk * intmax, nrem, dtype,
                  op, comm);
  }
}

/// Inplace reduction operation
template <typename T>
void allreduce(T* recv, size_t count, MPI_Op op, MPI_Comm comm) {
  auto dtype = mpi_traits<T>::datatype();

  size_t intmax = std::numeric_limits<int>::max();
  size_t nchunk = count / intmax;
  if (nchunk) throw std::runtime_error("Msg over INT_MAX not yet tested");
  for (int i = 0; i < nchunk; ++i) {
    MPI_Allreduce(MPI_IN_PLACE, recv + i * intmax, intmax, dtype, op, comm);
  }

  int nrem = count % intmax;
  if (nrem) {
    MPI_Allreduce(MPI_IN_PLACE, recv + nchunk * intmax, nrem, dtype, op, comm);
  }
}

/// Reduction of simple types
template <typename T>
T allreduce(const T& d, MPI_Op op, MPI_Comm comm) {
  T r;
  allreduce(&d, &r, 1, op, comm);
  return r;
}

/// Type-safe wrapper around MPI_Bcast
template <typename T>
void bcast(T* buffer, size_t count, int root, MPI_Comm comm) {
  auto dtype = mpi_traits<T>::datatype();

  size_t intmax = std::numeric_limits<int>::max();
  size_t nchunk = count / intmax;
  if (nchunk) throw std::runtime_error("Msg over INT_MAX not yet tested");
  for (int i = 0; i < nchunk; ++i) {
    MPI_Bcast(buffer + i * intmax, intmax, dtype, root, comm);
  }

  int nrem = count % intmax;
  if (nrem) {
    MPI_Bcast(buffer + nchunk * intmax, nrem, dtype, root, comm);
  }
}

/// MPI wrapper for `std::bitset`
template <size_t N>
struct mpi_traits<std::bitset<N>> {
  using type = std::bitset<N>;
  inline static mpi_datatype datatype() {
    return make_contiguous_mpi_datatype<char>(sizeof(type));
  }
};

/**
 * @brief Distributed atomic variable for thread-safe operations across MPI
 * processes
 *
 * This class implements a globally accessible atomic variable that can be
 * safely accessed and modified by multiple MPI processes concurrently. It uses
 * MPI-3 one-sided communication (RMA - Remote Memory Access) to provide atomic
 * operations on shared data without explicit synchronization between processes.
 *
 * @tparam T The type of the atomic variable (must be MPI-serializable)
 *
 * @note This class requires MPI-3 support for one-sided communication
 * @note T must be a type supported by mpi_traits (e.g., int, double, size_t)
 * @note The class is non-copyable and non-movable to prevent resource conflicts
 */
template <typename T>
class global_atomic {
  MPI_Win window_;  ///< MPI window for remote memory access
  T* buffer_;       ///< Pointer to the shared atomic variable

 public:
  /// Deleted default constructor - requires MPI communicator
  global_atomic() = delete;

  /**
   * @brief Constructs a global atomic variable shared across MPI processes
   *
   * Creates an MPI window that allows all processes in the communicator to
   * perform atomic operations on the shared variable. The variable is
   * initialized to the specified value on all processes.
   *
   * @param comm MPI communicator defining which processes share this atomic
   * variable
   * @param init Initial value for the atomic variable (default: 0)
   *
   * @throws std::runtime_error if MPI window creation fails
   *
   * @note This is a collective operation - all processes in comm must call it
   * @note Memory allocation is done collectively across all processes
   * @note After construction, the variable is immediately available for atomic
   * operations
   */
  global_atomic(MPI_Comm comm, T init = 0) {
    MPI_Win_allocate(sizeof(T), sizeof(T), MPI_INFO_NULL, comm, &buffer_,
                     &window_);
    if (window_ == MPI_WIN_NULL) {
      throw std::runtime_error("Window creation failed");
    }
    *buffer_ = init;
    MPI_Barrier(comm);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, window_);
  }

  /**
   * @brief Destructor that properly cleans up MPI resources
   *
   * Unlocks the MPI window and frees associated resources. This ensures
   * proper cleanup of the distributed memory structures.
   *
   * @note This must be called collectively by all processes that created the
   * window
   */
  ~global_atomic() noexcept {
    MPI_Win_unlock_all(window_);
    MPI_Win_free(&window_);
  }

  /// Deleted copy constructor - prevents resource conflicts
  global_atomic(const global_atomic&) = delete;
  /// Deleted move constructor - prevents resource conflicts
  global_atomic(global_atomic&&) noexcept = delete;

  /**
   * @brief Performs an atomic fetch-and-operate operation
   *
   * Atomically applies the specified MPI operation to the global variable
   * and returns the previous value. This is the fundamental building block
   * for all atomic operations provided by this class.
   *
   * The operation is performed as: new_value = old_value OP val
   *
   * @param val The value to use in the operation
   * @param op The MPI operation to perform (e.g., MPI_SUM, MPI_MIN, MPI_MAX)
   *
   * @return The value of the atomic variable before the operation
   *
   * @note The operation is atomic across all processes
   * @note Results are immediately flushed to ensure global consistency
   * @note This function can be called from any process in the communicator
   *
   * Supported operations include:
   * - MPI_SUM: Addition
   * - MPI_MIN: Minimum value
   * - MPI_MAX: Maximum value
   * - MPI_PROD: Multiplication
   * - And other MPI reduction operations
   */
  T fetch_and_op(T val, MPI_Op op) {
    T next_val;
    MPI_Fetch_and_op(&val, &next_val, mpi_traits<T>::datatype(), 0, 0, op,
                     window_);
    MPI_Win_flush(0, window_);
    return next_val;
  }

  /**
   * @brief Atomically adds a value to the global variable
   *
   * Convenience function that atomically adds the given value to the
   * global atomic variable and returns the previous value.
   *
   * @param val The value to add
   * @return The value before the addition
   *
   * Example:
   * ```cpp
   * global_atomic<int> counter(comm, 0);
   * int old_count = counter.fetch_add(5);  // Adds 5, returns previous value
   * ```
   */
  T fetch_add(T val) { return fetch_and_op(val, MPI_SUM); }

  /**
   * @brief Atomically updates the global variable to the minimum value
   *
   * Convenience function that atomically compares the given value with
   * the current global value and updates it to the minimum of the two.
   * Returns the previous value.
   *
   * @param val The value to compare with current minimum
   * @return The value before the potential update
   *
   * Example:
   * ```cpp
   * global_atomic<double> global_min(comm, 1e9);
   * double old_min = global_min.fetch_and_min(local_result);
   * ```
   */
  T fetch_and_min(T val) { return fetch_and_op(val, MPI_MIN); }
};

}  // namespace macis
#endif /* MACIS_ENABLE_MPI */
