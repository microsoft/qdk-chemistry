// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>
#include <qdk/chemistry/scf/config.h>
#ifdef QDK_CHEMISTRY_ENABLE_MPI
#include <mpi.h>
#endif
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include <cstdlib>
#include <filesystem>
#include <libint2.hpp>
#include <stdexcept>
#include <string>

using namespace qdk::chemistry::scf;

int main(int argc, char** argv) {
  std::filesystem::path sandbox;
  std::filesystem::path original_directory;
  if (const char* enabled = std::getenv("QDK_CTEST_SANDBOX");
      enabled != nullptr && std::string(enabled) == "1") {
    const auto temp_root = std::filesystem::temp_directory_path();
#ifdef _WIN32
    const auto pid = _getpid();
#else
    const auto pid = getpid();
#endif
    for (int attempt = 0; attempt < 100; ++attempt) {
      auto candidate = temp_root / ("qdk-ctest-" + std::to_string(pid) + "-" +
                                    std::to_string(attempt));
      if (std::filesystem::create_directory(candidate)) {
        sandbox = candidate;
        break;
      }
    }
    if (sandbox.empty()) {
      throw std::runtime_error("Cannot create a unique CTest sandbox");
    }

    const auto path = sandbox.string();
#ifdef _WIN32
    if (_putenv_s("TMPDIR", path.c_str()) != 0 ||
        _putenv_s("TMP", path.c_str()) != 0 ||
        _putenv_s("TEMP", path.c_str()) != 0) {
#else
    if (setenv("TMPDIR", path.c_str(), 1) != 0 ||
        setenv("TMP", path.c_str(), 1) != 0 ||
        setenv("TEMP", path.c_str(), 1) != 0) {
#endif
      throw std::runtime_error("Cannot set CTest sandbox temporary directory");
    }
    if (!std::filesystem::equivalent(std::filesystem::temp_directory_path(),
                                     sandbox)) {
      throw std::runtime_error(
          "CTest sandbox temporary directory is not in use");
    }
    original_directory = std::filesystem::current_path();
    std::filesystem::current_path(sandbox);
  }

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
  if (!sandbox.empty()) {
    std::filesystem::current_path(original_directory);
    std::filesystem::remove_all(sandbox);
  }
  return ret;
}
