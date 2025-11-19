// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#ifndef MPI_VARS_H_
#define MPI_VARS_H_
#include <qdk/chemistry/scf/config.h>

#ifdef QDK_CHEMISTRY_ENABLE_MPI
#include <mpi.h>
#endif

#include <qdk/chemistry/scf/util/env_helper.h>

namespace qdk::chemistry::scf::mpi {

/**
 * @brief Get total number of MPI ranks
 * @return Number of MPI processes (1 if MPI disabled)
 */
static int get_world_size() {
#ifdef QDK_CHEMISTRY_ENABLE_MPI
  static int _world_size = -1;
  if (_world_size == -1) {
    MPI_Comm_size(MPI_COMM_WORLD, &_world_size);
  }
  return _world_size;
#else
  return 1;
#endif
}

/**
 * @brief Get number of MPI ranks on local node
 * @return Number of processes on this node (1 if MPI disabled)
 */
static int get_local_size() {
#ifdef QDK_CHEMISTRY_ENABLE_MPI
  static int _local_size = -1;
  if (_local_size == -1) {
    MPI_Comm shared_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &shared_comm);
    MPI_Comm_size(shared_comm, &_local_size);
    MPI_Comm_free(&shared_comm);
  }
  return _local_size;
#else
  return 1;
#endif
}

/**
 * @brief Get global MPI rank of this process
 * @return Global rank ID (0 if MPI disabled)
 */
static int get_world_rank() {
#ifdef QDK_CHEMISTRY_ENABLE_MPI
  static int _world_rank = -1;
  if (_world_rank == -1) {
    MPI_Comm_rank(MPI_COMM_WORLD, &_world_rank);
  }
  return _world_rank;
#else
  return 0;
#endif
}

/**
 * @brief Get local MPI rank on this node
 * @return Local rank ID (0 if MPI disabled)
 */
static int get_local_rank() {
#ifdef QDK_CHEMISTRY_ENABLE_MPI
  static int _local_rank = -1;
  if (_local_rank == -1) {
    MPI_Comm shared_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &shared_comm);
    MPI_Comm_rank(shared_comm, &_local_rank);
    MPI_Comm_free(&shared_comm);
  }
  return _local_rank;
#else
  return 0;
#endif
}

}  // namespace qdk::chemistry::scf::mpi

#endif  // MPI_VARS_H_
