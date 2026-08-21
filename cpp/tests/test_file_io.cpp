// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <qdk/chemistry/utils/file_io.hpp>
#include <stdexcept>
#include <string>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

class FileIoTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ = std::filesystem::temp_directory_path() /
            ("qdk_file_io_test_" +
             std::to_string(
                 std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root_);
  }

  void TearDown() override {
    std::error_code ignored;
    std::filesystem::remove_all(root_, ignored);
  }

  std::filesystem::path root_;
};

TEST_F(FileIoTest, WritesReadsAndReplacesText) {
  const auto path = root_ / "data.txt";

  qdk::chemistry::utils::write_text_file_atomically(path, "first");
  EXPECT_EQ(qdk::chemistry::utils::read_text_file(path), "first");

  qdk::chemistry::utils::write_text_file_atomically(path, "second");
  EXPECT_EQ(qdk::chemistry::utils::read_text_file(path), "second");
}

TEST_F(FileIoTest, CreatesParentDirectoriesWhenRequested) {
  const auto path = root_ / "nested" / "directory" / "data.txt";

  qdk::chemistry::utils::write_text_file_atomically(path, "contents", true);

  EXPECT_EQ(qdk::chemistry::utils::read_text_file(path), "contents");
}

TEST_F(FileIoTest, RejectsMissingParentDirectoryByDefault) {
  const auto path = root_ / "missing" / "data.txt";

  EXPECT_THROW(
      qdk::chemistry::utils::write_text_file_atomically(path, "contents"),
      std::runtime_error);
  EXPECT_FALSE(std::filesystem::exists(path));
}

TEST_F(FileIoTest, PreservesDestinationWhenWriterFails) {
  const auto path = root_ / "data.txt";
  qdk::chemistry::utils::write_text_file_atomically(path, "original");

  EXPECT_THROW(qdk::chemistry::utils::write_file_atomically(
                   path,
                   [](const std::filesystem::path& temporary_path) {
                     qdk::chemistry::utils::write_text_file_atomically(
                         temporary_path, "incomplete");
                     throw std::runtime_error("writer failed");
                   }),
               std::runtime_error);

  EXPECT_EQ(qdk::chemistry::utils::read_text_file(path), "original");
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(root_),
                          std::filesystem::directory_iterator()),
            1);
}

TEST_F(FileIoTest, RejectsDirectoryReads) {
  EXPECT_THROW(qdk::chemistry::utils::read_text_file(root_),
               std::runtime_error);
}

TEST_F(FileIoTest, PreservesDestinationPermissions) {
#ifndef _WIN32
  const auto path = root_ / "data.txt";
  qdk::chemistry::utils::write_text_file_atomically(path, "original");
  const auto private_permissions = std::filesystem::perms::owner_read |
                                   std::filesystem::perms::owner_write |
                                   std::filesystem::perms::group_read;
  std::filesystem::permissions(path, private_permissions,
                               std::filesystem::perm_options::replace);

  qdk::chemistry::utils::write_text_file_atomically(path, "replacement");

  EXPECT_EQ(std::filesystem::status(path).permissions(), private_permissions);
#endif
}

TEST_F(FileIoTest, CreatesNewFilesWithOwnerOnlyPermissions) {
#ifndef _WIN32
  const auto path = root_ / "data.txt";

  qdk::chemistry::utils::write_file_atomically(
      path, [](const std::filesystem::path& temporary_path) {
        std::ofstream output(temporary_path);
        output << "contents";
        output.close();
        std::filesystem::permissions(temporary_path,
                                     std::filesystem::perms::all,
                                     std::filesystem::perm_options::replace);
      });

  const auto owner_only_permissions =
      std::filesystem::perms::owner_read | std::filesystem::perms::owner_write;
  EXPECT_EQ(std::filesystem::status(path).permissions(),
            owner_only_permissions);
#endif
}

#ifndef _WIN32
TEST_F(FileIoTest, DoesNotInheritTemporaryDescriptorAcrossExec) {
  const auto path = root_ / "data.txt";

  qdk::chemistry::utils::write_file_atomically(
      path, [](const std::filesystem::path& temporary_path) {
        std::ofstream output(temporary_path);
        output << "contents";
        output.close();

        struct stat temporary_status{};
        ASSERT_EQ(::stat(temporary_path.c_str(), &temporary_status), 0);
        const long system_limit = ::sysconf(_SC_OPEN_MAX);
        const long descriptor_limit =
            system_limit < 0 ? 1024 : std::min<long>(system_limit, 1024);
        for (int descriptor = 0; descriptor < descriptor_limit; ++descriptor) {
          struct stat descriptor_status{};
          if (::fstat(descriptor, &descriptor_status) == 0 &&
              descriptor_status.st_dev == temporary_status.st_dev &&
              descriptor_status.st_ino == temporary_status.st_ino) {
            EXPECT_NE(::fcntl(descriptor, F_GETFD) & FD_CLOEXEC, 0);
            return;
          }
        }
        FAIL() << "Reserved temporary descriptor not found";
      });
}
#endif

#ifdef _WIN32
TEST_F(FileIoTest, ReplacesReadOnlyDestinationOnWindows) {
  const auto path = root_ / "data.txt";
  qdk::chemistry::utils::write_text_file_atomically(path, "original");
  ASSERT_NE(SetFileAttributesW(path.c_str(), FILE_ATTRIBUTE_READONLY), 0);

  qdk::chemistry::utils::write_text_file_atomically(path, "replacement");

  EXPECT_EQ(qdk::chemistry::utils::read_text_file(path), "replacement");
  EXPECT_EQ(std::filesystem::status(path).permissions() &
                std::filesystem::perms::owner_write,
            std::filesystem::perms::none);
}

TEST_F(FileIoTest, CleansUpReadOnlyTemporaryFileWhenWriterFails) {
  const auto path = root_ / "data.txt";
  qdk::chemistry::utils::write_text_file_atomically(path, "original");

  EXPECT_THROW(qdk::chemistry::utils::write_file_atomically(
                   path,
                   [](const std::filesystem::path& temporary_path) {
                     std::ofstream output(temporary_path);
                     output << "incomplete";
                     output.close();
                     ASSERT_NE(SetFileAttributesW(temporary_path.c_str(),
                                                  FILE_ATTRIBUTE_READONLY),
                               0);
                     throw std::runtime_error("writer failed");
                   }),
               std::runtime_error);

  EXPECT_EQ(qdk::chemistry::utils::read_text_file(path), "original");
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(root_),
                          std::filesystem::directory_iterator()),
            1);
}
#endif

TEST_F(FileIoTest, PreservesDestinationSuffixesForWriter) {
  const auto path = root_ / "data.structure.json";
  std::filesystem::path observed_temporary_path;

  qdk::chemistry::utils::write_file_atomically(
      path,
      [&observed_temporary_path](const std::filesystem::path& temporary_path) {
        observed_temporary_path = temporary_path;
        std::ofstream output(temporary_path,
                             std::ios::binary | std::ios::trunc);
        output << "contents";
      });

  EXPECT_EQ(observed_temporary_path.extension(), ".json");
  EXPECT_EQ(observed_temporary_path.stem().extension(), ".structure");
}

#ifndef _WIN32
TEST_F(FileIoTest, PreservesLongDestinationSuffix) {
  const auto path = root_ / ("x." + std::string(249, 'a'));
  std::filesystem::path observed_temporary_path;

  qdk::chemistry::utils::write_file_atomically(
      path,
      [&observed_temporary_path](const std::filesystem::path& temporary_path) {
        observed_temporary_path = temporary_path;
        std::ofstream output(temporary_path);
        output << "contents";
      });

  EXPECT_EQ(observed_temporary_path.extension(), path.extension());
  EXPECT_EQ(qdk::chemistry::utils::read_text_file(path), "contents");
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(root_),
                          std::filesystem::directory_iterator()),
            1);
}
#endif

TEST_F(FileIoTest, RejectsReplacedTemporaryFile) {
  const auto path = root_ / "data.txt";

  EXPECT_THROW(qdk::chemistry::utils::write_file_atomically(
                   path,
                   [](const std::filesystem::path& temporary_path) {
                     std::filesystem::remove(temporary_path);
                     std::ofstream output(temporary_path);
                     output << "replacement";
                   }),
               std::runtime_error);
  EXPECT_FALSE(std::filesystem::exists(path));
}

TEST_F(FileIoTest, FreezesRelativeDestinationBeforeWriterRuns) {
  const auto original_directory = std::filesystem::current_path();
  const auto first_directory = root_ / "first";
  const auto second_directory = root_ / "second";
  std::filesystem::create_directories(first_directory);
  std::filesystem::create_directories(second_directory);
  std::filesystem::current_path(first_directory);

  try {
    qdk::chemistry::utils::write_file_atomically(
        "data.txt",
        [&second_directory](const std::filesystem::path& temporary_path) {
          std::ofstream output(temporary_path);
          output << "contents";
          output.close();
          std::filesystem::current_path(second_directory);
        });
  } catch (...) {
    std::filesystem::current_path(original_directory);
    throw;
  }
  std::filesystem::current_path(original_directory);

  EXPECT_EQ(qdk::chemistry::utils::read_text_file(first_directory / "data.txt"),
            "contents");
  EXPECT_FALSE(std::filesystem::exists(second_directory / "data.txt"));
}

#ifndef _WIN32
TEST_F(FileIoTest, PreservesSymlinkParentTraversalWhenFreezingPath) {
  const auto original_directory = std::filesystem::current_path();
  const auto working_directory = root_ / "working";
  const auto target_parent = root_ / "target-parent";
  const auto target_directory = target_parent / "target";
  std::filesystem::create_directories(working_directory);
  std::filesystem::create_directories(target_directory);
  std::filesystem::create_directory_symlink(target_directory,
                                            working_directory / "link");
  std::filesystem::current_path(working_directory);

  try {
    qdk::chemistry::utils::write_text_file_atomically("link/../data.txt",
                                                      "contents");
  } catch (...) {
    std::filesystem::current_path(original_directory);
    throw;
  }
  std::filesystem::current_path(original_directory);

  EXPECT_EQ(qdk::chemistry::utils::read_text_file(target_parent / "data.txt"),
            "contents");
  EXPECT_FALSE(std::filesystem::exists(working_directory / "data.txt"));
}
#endif

TEST_F(FileIoTest, SupportsUnicodePaths) {
#ifdef _WIN32
  const std::filesystem::path filename = L"\u6570\u636e.txt";
#else
  const std::filesystem::path filename = "\xE6\x95\xB0\xE6\x8D\xAE.txt";
#endif
  const auto path = root_ / filename;

  qdk::chemistry::utils::write_text_file_atomically(path, "contents");

  EXPECT_EQ(qdk::chemistry::utils::read_text_file(path), "contents");
}

}  // namespace
