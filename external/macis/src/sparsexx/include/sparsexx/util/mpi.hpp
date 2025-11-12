/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <mpi.h>

#include <cstdint>
#include <iostream>
#include <typeinfo>
#include <vector>

namespace sparsexx::detail {

static inline int64_t get_mpi_rank(MPI_Comm c) {
  int rank;
  MPI_Comm_rank(c, &rank);
  return rank;
}

static inline int64_t get_mpi_size(MPI_Comm c) {
  int size;
  MPI_Comm_size(c, &size);
  return size;
}

template <typename T>
struct mpi_data;

#define REGISTER_MPI_STATIC_TYPE(TYPE, MPI_TYPE)   \
  template <>                                      \
  struct mpi_data<TYPE> {                          \
    inline static auto type() { return MPI_TYPE; } \
  };

REGISTER_MPI_STATIC_TYPE(double, MPI_DOUBLE)
REGISTER_MPI_STATIC_TYPE(int64_t, MPI_INT64_T)
REGISTER_MPI_STATIC_TYPE(uint64_t, MPI_UINT64_T)
REGISTER_MPI_STATIC_TYPE(int, MPI_INT)
REGISTER_MPI_STATIC_TYPE(unsigned, MPI_UNSIGNED)

#undef REGISTER_MPI_STATIC_TYPE

template <typename T>
T mpi_allreduce(const T& value, MPI_Op op, MPI_Comm comm) {
  T reduced_value;
  MPI_Allreduce(&value, &reduced_value, 1, mpi_data<T>::type(), op, comm);
  return reduced_value;
}

template <typename T>
void mpi_allgather(const T* data, size_t count, T* gathered_data,
                   MPI_Comm comm) {
  MPI_Allgather(data, count, mpi_data<T>::type(), gathered_data, count,
                mpi_data<T>::type(), comm);
}

template <typename T>
std::vector<T> mpi_allgather(const std::vector<T>& data, MPI_Comm comm) {
  const size_t count = data.size();
  const auto comm_size = get_mpi_size(comm);
  std::vector<T> gathered_data(count * comm_size);

  MPI_Allgather(data.data(), count, mpi_data<T>::type(), gathered_data.data(),
                count, mpi_data<T>::type(), comm);

  return gathered_data;
}

template <typename T>
void mpi_bcast(T* data, size_t count, int root, MPI_Comm comm) {
  MPI_Bcast(data, count, mpi_data<T>::type(), root, comm);
}

template <typename T>
void mpi_bcast(std::vector<T>& data, int root, MPI_Comm comm) {
  mpi_bcast(data.data(), data.size(), root, comm);
}

template <typename T>
MPI_Request mpi_irecv(T* data, size_t count, int source_rank, int tag,
                      MPI_Comm comm) {
  MPI_Request req;
  MPI_Irecv(data, count, mpi_data<T>::type(), source_rank, tag, comm, &req);
  return req;
}

template <typename T>
MPI_Request mpi_irecv(std::vector<T>& data, int source_rank, int tag,
                      MPI_Comm comm) {
  return mpi_irecv(data.data(), data.size(), source_rank, tag, comm);
}

template <typename T>
MPI_Request mpi_isend(const T* data, size_t count, int dest_rank, int tag,
                      MPI_Comm comm) {
  MPI_Request req;
  MPI_Isend(data, count, mpi_data<T>::type(), dest_rank, tag, comm, &req);
  return req;
}

template <typename T>
MPI_Request mpi_isend(const std::vector<T>& data, int dest_rank, int tag,
                      MPI_Comm comm) {
  return mpi_isend(data.data(), data.size(), dest_rank, tag, comm);
}

inline void mpi_waitall_ignore_status(std::vector<MPI_Request>& requests) {
  MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
}

}  // namespace sparsexx::detail
