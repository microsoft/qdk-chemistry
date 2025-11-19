// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/core/scf.h>

#include <cstring>
#ifdef QDK_CHEMISTRY_ENABLE_MPI
#include <mpi.h>
#endif

namespace qdk::chemistry::scf {

ParallelConfig mpi_default_input() {
  int world_size = 1;
  int world_rank = 0;
  int local_size = 1;
  int local_rank = 0;
#ifdef QDK_CHEMISTRY_ENABLE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  MPI_Comm shared_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &shared_comm);
  MPI_Comm_size(shared_comm, &local_size);
  MPI_Comm_rank(shared_comm, &local_rank);
  MPI_Comm_free(&shared_comm);
#endif
  return ParallelConfig{world_size, world_rank, local_size, local_rank};
}

}  // namespace qdk::chemistry::scf
