/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
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
  MPI_Datatype dtype;
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

template <typename T>
class global_atomic {
  MPI_Win window_;
  T* buffer_;

 public:
  global_atomic() = delete;

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

  ~global_atomic() noexcept {
    MPI_Win_unlock_all(window_);
    MPI_Win_free(&window_);
  }

  global_atomic(const global_atomic&) = delete;
  global_atomic(global_atomic&&) noexcept = delete;

  T fetch_and_op(T val, MPI_Op op) {
    T next_val;
    MPI_Fetch_and_op(&val, &next_val, mpi_traits<T>::datatype(), 0, 0, op,
                     window_);
    MPI_Win_flush(0, window_);
    return next_val;
  }

  T fetch_add(T val) { return fetch_and_op(val, MPI_SUM); }
  T fetch_and_min(T val) { return fetch_and_op(val, MPI_MIN); }
};

}  // namespace macis
#endif /* MACIS_ENABLE_MPI */
