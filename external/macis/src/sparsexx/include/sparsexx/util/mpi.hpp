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

/**
 * @brief Gets the rank of the current process in an MPI communicator
 *
 * @param c The MPI communicator
 * @return The rank of the current process (0-based index)
 */
static inline int64_t get_mpi_rank(MPI_Comm c) {
  int rank;
  MPI_Comm_rank(c, &rank);
  return rank;
}

/**
 * @brief Gets the number of processes in an MPI communicator
 *
 * @param c The MPI communicator
 * @return The total number of processes in the communicator
 */
static inline int64_t get_mpi_size(MPI_Comm c) {
  int size;
  MPI_Comm_size(c, &size);
  return size;
}

/**
 * @brief Template trait for mapping C++ types to MPI data types
 *
 * This template struct provides a mapping between C++ types and their
 * corresponding MPI data types. Specializations are provided for common
 * types like double, int64_t, uint64_t, int, and unsigned.
 *
 * @tparam T The C++ type to map to an MPI data type
 */
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

/**
 * @brief Performs an MPI all-reduce operation on a single value
 *
 * Combines values from all processes using the specified operation and
 * distributes the result to all processes.
 *
 * @tparam T The type of value to reduce (must have MPI type mapping)
 * @param value The input value from this process
 * @param op The MPI reduction operation (e.g., MPI_SUM, MPI_MAX)
 * @param comm The MPI communicator
 * @return The reduced value
 */
template <typename T>
T mpi_allreduce(const T& value, MPI_Op op, MPI_Comm comm) {
  T reduced_value;
  MPI_Allreduce(&value, &reduced_value, 1, mpi_data<T>::type(), op, comm);
  return reduced_value;
}

/**
 * @brief Gathers data from all processes into a contiguous array
 *
 * Each process contributes an array of data, and all arrays are
 * concatenated into the gathered_data array on all processes.
 *
 * @tparam T The type of data elements (must have MPI type mapping)
 * @param data Pointer to the input data array
 * @param count Number of elements to gather from each process
 * @param gathered_data Pointer to output array (must be size count * comm_size)
 * @param comm The MPI communicator
 */
template <typename T>
void mpi_allgather(const T* data, size_t count, T* gathered_data,
                   MPI_Comm comm) {
  MPI_Allgather(data, count, mpi_data<T>::type(), gathered_data, count,
                mpi_data<T>::type(), comm);
}

/**
 * @brief Gathers data from all processes into a vector
 *
 * Convenience function that gathers vectors from all processes and returns
 * a single vector containing all the data concatenated together.
 *
 * @tparam T The type of vector elements (must have MPI type mapping)
 * @param data The input vector from this process
 * @param comm The MPI communicator
 * @return Vector containing data from all processes concatenated
 */
template <typename T>
std::vector<T> mpi_allgather(const std::vector<T>& data, MPI_Comm comm) {
  const size_t count = data.size();
  const auto comm_size = get_mpi_size(comm);
  std::vector<T> gathered_data(count * comm_size);

  MPI_Allgather(data.data(), count, mpi_data<T>::type(), gathered_data.data(),
                count, mpi_data<T>::type(), comm);

  return gathered_data;
}

/**
 * @brief Broadcasts data from root process to all processes
 *
 * @tparam T The type of data elements (must have MPI type mapping)
 * @param data Pointer to the data array to broadcast
 * @param count Number of elements to broadcast
 * @param root Rank of the root process that provides the data
 * @param comm The MPI communicator
 */
template <typename T>
void mpi_bcast(T* data, size_t count, int root, MPI_Comm comm) {
  MPI_Bcast(data, count, mpi_data<T>::type(), root, comm);
}

/**
 * @brief Broadcasts a vector from root process to all processes
 *
 * Convenience function for broadcasting std::vector data.
 *
 * @tparam T The type of vector elements (must have MPI type mapping)
 * @param data The vector to broadcast (modified on non-root processes)
 * @param root Rank of the root process that provides the data
 * @param comm The MPI communicator
 */
template <typename T>
void mpi_bcast(std::vector<T>& data, int root, MPI_Comm comm) {
  mpi_bcast(data.data(), data.size(), root, comm);
}

/**
 * @brief Initiates a non-blocking receive operation
 *
 * Starts a non-blocking receive that can be completed later with MPI_Wait
 * or similar functions.
 *
 * @tparam T The type of data elements (must have MPI type mapping)
 * @param data Pointer to the buffer where received data will be stored
 * @param count Number of elements to receive
 * @param source_rank Rank of the source process
 * @param tag Message tag for matching with corresponding send
 * @param comm The MPI communicator
 * @return MPI_Request handle for the non-blocking operation
 */
template <typename T>
MPI_Request mpi_irecv(T* data, size_t count, int source_rank, int tag,
                      MPI_Comm comm) {
  MPI_Request req;
  MPI_Irecv(data, count, mpi_data<T>::type(), source_rank, tag, comm, &req);
  return req;
}

/**
 * @brief Initiates a non-blocking receive operation for a vector
 *
 * Convenience function for non-blocking receive into a std::vector.
 *
 * @tparam T The type of vector elements (must have MPI type mapping)
 * @param data The vector where received data will be stored
 * @param source_rank Rank of the source process
 * @param tag Message tag for matching with corresponding send
 * @param comm The MPI communicator
 * @return MPI_Request handle for the non-blocking operation
 */
template <typename T>
MPI_Request mpi_irecv(std::vector<T>& data, int source_rank, int tag,
                      MPI_Comm comm) {
  return mpi_irecv(data.data(), data.size(), source_rank, tag, comm);
}

/**
 * @brief Initiates a non-blocking send operation
 *
 * Starts a non-blocking send that can be completed later with MPI_Wait
 * or similar functions.
 *
 * @tparam T The type of data elements (must have MPI type mapping)
 * @param data Pointer to the data to send
 * @param count Number of elements to send
 * @param dest_rank Rank of the destination process
 * @param tag Message tag for matching with corresponding receive
 * @param comm The MPI communicator
 * @return MPI_Request handle for the non-blocking operation
 */
template <typename T>
MPI_Request mpi_isend(const T* data, size_t count, int dest_rank, int tag,
                      MPI_Comm comm) {
  MPI_Request req;
  MPI_Isend(data, count, mpi_data<T>::type(), dest_rank, tag, comm, &req);
  return req;
}

/**
 * @brief Initiates a non-blocking send operation for a vector
 *
 * Convenience function for non-blocking send of std::vector data.
 *
 * @tparam T The type of vector elements (must have MPI type mapping)
 * @param data The vector containing data to send
 * @param dest_rank Rank of the destination process
 * @param tag Message tag for matching with corresponding receive
 * @param comm The MPI communicator
 * @return MPI_Request handle for the non-blocking operation
 */
template <typename T>
MPI_Request mpi_isend(const std::vector<T>& data, int dest_rank, int tag,
                      MPI_Comm comm) {
  return mpi_isend(data.data(), data.size(), dest_rank, tag, comm);
}

/**
 * @brief Waits for completion of all non-blocking operations
 *
 * Blocks until all MPI operations represented by the request handles
 * have completed. Status information is ignored.
 *
 * @param requests Vector of MPI_Request handles to wait for
 */
inline void mpi_waitall_ignore_status(std::vector<MPI_Request>& requests) {
  MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
}

}  // namespace sparsexx::detail
