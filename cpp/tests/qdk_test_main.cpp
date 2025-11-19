// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>
#include <qdk/chemistry/scf/config.h>
#ifdef QDK_CHEMISTRY_ENABLE_MPI
#include <mpi.h>
#endif

#include <libint2.hpp>

using namespace qdk::chemistry::scf;

int main(int argc, char** argv) {
#ifdef QDK_CHEMISTRY_ENABLE_MPI
  int req = MPI_THREAD_SERIALIZED, prov;
  MPI_Init_thread(nullptr, nullptr, req, &prov);
  if (req != prov)
    throw std::runtime_error("QDK-Chemistry Requires MPI_THREAD_MULTIPLE");
#endif
  libint2::initialize();
  testing::InitGoogleTest(&argc, argv);
  QDKChemistryConfig::set_resources_dir(
      std::filesystem::path(TEST_RESOURCES_DIR));
  auto ret = RUN_ALL_TESTS();
#ifdef QDK_CHEMISTRY_ENABLE_MPI
  MPI_Finalize();
#endif
  libint2::finalize();
  return ret;
}
