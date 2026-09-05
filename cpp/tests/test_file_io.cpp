// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <qdk/chemistry/utils/file_io.hpp>
#include <stdexcept>
#include <string>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
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
    const auto timestamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
#ifdef _WIN32
    const auto process_id = GetCurrentProcessId();
#else
    const auto process_id = ::getpid();
#endif
    for (int attempt = 0; attempt < 64; ++attempt) {
      const auto candidate =
          std::filesystem::temp_directory_path() /
          ("qdk_file_io_test_" + std::to_string(process_id) + "_" +
           std::to_string(timestamp) + "_" + std::to_string(attempt));
      std::error_code error;
      if (std::filesystem::create_directory(candidate, error)) {
        root_ = candidate;
        return;
      }
      ASSERT_FALSE(error) << "Could not create test directory '" << candidate
                          << "': " << error.message();
    }
    FAIL() << "Could not create a unique FileIO test directory";
  }

  void TearDown() override {
    if (root_.empty()) {
      return;
    }
#ifdef _WIN32
    if (std::filesystem::exists(root_)) {
      for (const auto& entry :
           std::filesystem::recursive_directory_iterator(root_)) {
        const DWORD attributes = GetFileAttributesW(entry.path().c_str());
        ASSERT_NE(attributes, INVALID_FILE_ATTRIBUTES);
        if ((attributes & FILE_ATTRIBUTE_READONLY) != 0) {
          const DWORD writable_attributes =
              (attributes & ~FILE_ATTRIBUTE_READONLY) == 0
                  ? FILE_ATTRIBUTE_NORMAL
                  : attributes & ~FILE_ATTRIBUTE_READONLY;
          ASSERT_NE(
              SetFileAttributesW(entry.path().c_str(), writable_attributes), 0);
        }
      }
    }
#endif
    std::error_code error;
    std::filesystem::remove_all(root_, error);
    EXPECT_FALSE(error) << "Could not remove test directory '" << root_
                        << "': " << error.message();
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
#ifndef _WIN32
  EXPECT_EQ(std::filesystem::status(path.parent_path()).permissions(),
            std::filesystem::perms::owner_all);
  EXPECT_EQ(
      std::filesystem::status(path.parent_path().parent_path()).permissions(),
      std::filesystem::perms::owner_all);
#endif
}

#ifndef _WIN32
TEST_F(FileIoTest, CreatesPrivateParentDirectoriesUnderRestrictiveUmask) {
  const auto path = root_ / "private" / "nested" / "data.txt";
  const mode_t original_umask = ::umask(0777);
  try {
    qdk::chemistry::utils::write_text_file_atomically(path, "contents", true);
  } catch (...) {
    ::umask(original_umask);
    throw;
  }
  ::umask(original_umask);

  EXPECT_EQ(std::filesystem::status(path.parent_path()).permissions(),
            std::filesystem::perms::owner_all);
  EXPECT_EQ(qdk::chemistry::utils::read_text_file(path), "contents");
}
#endif

TEST_F(FileIoTest, RejectsMissingParentDirectoryByDefault) {
  const auto path = root_ / "missing" / "data.txt";

  EXPECT_THROW(
      qdk::chemistry::utils::write_text_file_atomically(path, "contents"),
      std::runtime_error);
  EXPECT_FALSE(std::filesystem::exists(path));
}

TEST_F(FileIoTest, RejectsTrailingSeparatorBeforeWriterRuns) {
  const auto directory = root_ / "directory";
  const auto trailing_path = directory / "";
  std::filesystem::create_directory(directory);
  bool writer_ran = false;

  EXPECT_THROW(
      qdk::chemistry::utils::write_file_atomically(
          trailing_path,
          [&writer_ran](const std::filesystem::path&) { writer_ran = true; }),
      std::invalid_argument);
  EXPECT_THROW(qdk::chemistry::utils::ensure_parent_directory(trailing_path),
               std::invalid_argument);

  EXPECT_FALSE(writer_ran);
}

TEST_F(FileIoTest, RejectsDotComponentsBeforeWriterRuns) {
  bool writer_ran = false;

  EXPECT_THROW(
      qdk::chemistry::utils::write_file_atomically(
          root_ / ".",
          [&writer_ran](const std::filesystem::path&) { writer_ran = true; }),
      std::invalid_argument);
  EXPECT_THROW(
      qdk::chemistry::utils::write_file_atomically(
          root_ / "..",
          [&writer_ran](const std::filesystem::path&) { writer_ran = true; }),
      std::invalid_argument);

  EXPECT_FALSE(writer_ran);
}

TEST_F(FileIoTest, RejectsEmbeddedNulPaths) {
  const auto prefix = root_ / "data.txt";
  std::string path = prefix.string();
  path.append("\0ignored", 8);
  const std::filesystem::path nul_path(path);

  EXPECT_THROW(qdk::chemistry::utils::ensure_parent_directory(nul_path),
               std::invalid_argument);
  EXPECT_THROW(qdk::chemistry::utils::read_text_file(nul_path),
               std::invalid_argument);
  EXPECT_THROW(
      qdk::chemistry::utils::write_text_file_atomically(nul_path, "contents"),
      std::invalid_argument);
  EXPECT_FALSE(std::filesystem::exists(prefix));
}

TEST_F(FileIoTest, PreservesDestinationWhenWriterFails) {
  const auto path = root_ / "data.txt";
  qdk::chemistry::utils::write_text_file_atomically(path, "original");

  EXPECT_THROW(qdk::chemistry::utils::write_file_atomically(
                   path,
                   [](const std::filesystem::path& temporary_path) {
                     std::ofstream output(temporary_path);
                     output << "incomplete";
                     output.close();
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

#ifndef _WIN32
TEST_F(FileIoTest, DoesNotAcquireControllingTerminalWhenRejectingTerminal) {
  const int master_descriptor = ::posix_openpt(O_RDWR | O_NOCTTY | O_CLOEXEC);
  ASSERT_NE(master_descriptor, -1);
  ASSERT_EQ(::grantpt(master_descriptor), 0);
  ASSERT_EQ(::unlockpt(master_descriptor), 0);
  const char* slave_name = ::ptsname(master_descriptor);
  ASSERT_NE(slave_name, nullptr);
  const std::string slave_path(slave_name);

  const pid_t child = ::fork();
  ASSERT_NE(child, -1);
  if (child == 0) {
    ::close(master_descriptor);
    if (::setsid() == -1) {
      _exit(2);
    }
    try {
      static_cast<void>(qdk::chemistry::utils::read_text_file(slave_path));
      _exit(3);
    } catch (const std::runtime_error& error) {
      if (std::string(error.what()).find("not a regular file") ==
          std::string::npos) {
        _exit(5);
      }
    }
    const int terminal_descriptor =
        ::open("/dev/tty", O_RDONLY | O_NOCTTY | O_CLOEXEC);
    if (terminal_descriptor != -1) {
      ::close(terminal_descriptor);
      _exit(4);
    }
    _exit(errno == ENXIO ? 0 : 6);
  }

  int status = 0;
  ASSERT_EQ(::waitpid(child, &status, 0), child);
  ::close(master_descriptor);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}
#endif

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

TEST_F(FileIoTest, ClearsSpecialPermissionBitsOnReplacement) {
#ifndef _WIN32
  const auto path = root_ / "data.txt";
  qdk::chemistry::utils::write_text_file_atomically(path, "original");
  const auto permissions =
      std::filesystem::perms::owner_all | std::filesystem::perms::group_read |
      std::filesystem::perms::group_exec | std::filesystem::perms::others_read |
      std::filesystem::perms::others_exec | std::filesystem::perms::set_uid |
      std::filesystem::perms::set_gid | std::filesystem::perms::sticky_bit;
  std::filesystem::permissions(path, permissions,
                               std::filesystem::perm_options::replace);

  qdk::chemistry::utils::write_text_file_atomically(path, "replacement");

  EXPECT_EQ(std::filesystem::status(path).permissions(),
            permissions & std::filesystem::perms::all);
#endif
}

TEST_F(FileIoTest, RestrictiveUmaskDoesNotPreventWriting) {
#ifndef _WIN32
  const auto path = root_ / "data.txt";
  const mode_t original_umask = ::umask(0777);
  try {
    qdk::chemistry::utils::write_text_file_atomically(path, "contents");
  } catch (...) {
    ::umask(original_umask);
    throw;
  }
  ::umask(original_umask);

  EXPECT_EQ(qdk::chemistry::utils::read_text_file(path), "contents");
  EXPECT_EQ(
      std::filesystem::status(path).permissions(),
      std::filesystem::perms::owner_read | std::filesystem::perms::owner_write);
#endif
}

#ifndef _WIN32
TEST_F(FileIoTest, RejectsSymlinkDestinationsWithoutCopyingReferentMode) {
  const auto target = root_ / "target.txt";
  const auto link = root_ / "link.txt";
  qdk::chemistry::utils::write_text_file_atomically(target, "target");
  std::filesystem::permissions(
      target,
      std::filesystem::perms::owner_all | std::filesystem::perms::group_read |
          std::filesystem::perms::set_uid | std::filesystem::perms::set_gid,
      std::filesystem::perm_options::replace);
  std::filesystem::create_symlink(target, link);

  EXPECT_THROW(
      qdk::chemistry::utils::write_text_file_atomically(link, "replacement"),
      std::runtime_error);

  EXPECT_TRUE(std::filesystem::is_symlink(link));
  EXPECT_EQ(qdk::chemistry::utils::read_text_file(target), "target");
}

TEST_F(FileIoTest, RejectsNonRegularDestinations) {
  const auto path = root_ / "data.fifo";
  ASSERT_EQ(::mkfifo(path.c_str(), 0666), 0);

  EXPECT_THROW(
      qdk::chemistry::utils::write_text_file_atomically(path, "replacement"),
      std::runtime_error);

  struct stat status{};
  ASSERT_EQ(::lstat(path.c_str(), &status), 0);
  EXPECT_TRUE(S_ISFIFO(status.st_mode));
}
#endif

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

TEST_F(FileIoTest, RejectsReadOnlyDestinationWithSurvivingHardLink) {
  const auto path = root_ / "data.txt";
  const auto alias = root_ / "alias.txt";
  qdk::chemistry::utils::write_text_file_atomically(path, "original");
  std::filesystem::create_hard_link(path, alias);
  ASSERT_NE(SetFileAttributesW(path.c_str(), FILE_ATTRIBUTE_READONLY), 0);

  EXPECT_THROW(
      qdk::chemistry::utils::write_text_file_atomically(path, "replacement"),
      std::runtime_error);

  EXPECT_EQ(qdk::chemistry::utils::read_text_file(path), "original");
  EXPECT_EQ(qdk::chemistry::utils::read_text_file(alias), "original");
  EXPECT_NE(GetFileAttributesW(path.c_str()) & FILE_ATTRIBUTE_READONLY, 0);
  EXPECT_NE(GetFileAttributesW(alias.c_str()) & FILE_ATTRIBUTE_READONLY, 0);
}

TEST_F(FileIoTest, DoesNotCopyTemporaryStorageAttributeToReplacement) {
  const auto path = root_ / "data.txt";
  qdk::chemistry::utils::write_text_file_atomically(path, "original");
  ASSERT_NE(SetFileAttributesW(path.c_str(), FILE_ATTRIBUTE_READONLY |
                                                 FILE_ATTRIBUTE_TEMPORARY),
            0);

  qdk::chemistry::utils::write_text_file_atomically(path, "replacement");

  const DWORD attributes = GetFileAttributesW(path.c_str());
  ASSERT_NE(attributes, INVALID_FILE_ATTRIBUTES);
  EXPECT_NE(attributes & FILE_ATTRIBUTE_READONLY, 0);
  EXPECT_EQ(attributes & FILE_ATTRIBUTE_TEMPORARY, 0);
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

TEST_F(FileIoTest, PreservesWritableDestinationOnWindows) {
  const auto path = root_ / "data.txt";
  qdk::chemistry::utils::write_text_file_atomically(path, "original");

  qdk::chemistry::utils::write_file_atomically(
      path, [](const std::filesystem::path& temporary_path) {
        std::ofstream output(temporary_path);
        output << "replacement";
        output.close();
        ASSERT_NE(
            SetFileAttributesW(temporary_path.c_str(), FILE_ATTRIBUTE_READONLY),
            0);
      });

  EXPECT_EQ(qdk::chemistry::utils::read_text_file(path), "replacement");
  EXPECT_NE(std::filesystem::status(path).permissions() &
                std::filesystem::perms::owner_write,
            std::filesystem::perms::none);
}

TEST_F(FileIoTest, AllowsExclusiveWriterOnWindows) {
  const auto path = root_ / "data.txt";

  qdk::chemistry::utils::write_file_atomically(
      path, [](const std::filesystem::path& temporary_path) {
        HANDLE handle =
            CreateFileW(temporary_path.c_str(), GENERIC_WRITE, 0, nullptr,
                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        ASSERT_NE(handle, INVALID_HANDLE_VALUE);
        constexpr char contents[] = "contents";
        DWORD written = 0;
        EXPECT_NE(WriteFile(handle, contents, sizeof(contents) - 1, &written,
                            nullptr),
                  0);
        EXPECT_EQ(written, sizeof(contents) - 1);
        CloseHandle(handle);
      });

  EXPECT_EQ(qdk::chemistry::utils::read_text_file(path), "contents");
}

TEST_F(FileIoTest, RejectsAlternateDataStreamsOnWindows) {
  const auto path = root_ / "data.txt:stream";
  bool writer_ran = false;

  EXPECT_THROW(
      qdk::chemistry::utils::write_file_atomically(
          path,
          [&writer_ran](const std::filesystem::path&) { writer_ran = true; }),
      std::invalid_argument);

  EXPECT_FALSE(writer_ran);
}

TEST_F(FileIoTest, FallsBackForNearMaxPathDestinationOnWindows) {
  auto parent = root_;
  while ((parent / "d.txt").native().size() < 220) {
    parent /= "segment123";
  }
  const auto current_length = (parent / "d.txt").native().size();
  if (current_length < 244) {
    parent /= std::wstring(243 - current_length, L'p');
  }
  std::filesystem::create_directories(parent);
  const auto path = parent / "d.txt";

  qdk::chemistry::utils::write_text_file_atomically(path, "contents");

  EXPECT_EQ(qdk::chemistry::utils::read_text_file(path), "contents");
}

TEST_F(FileIoTest, SupportsExtendedLengthPathsOnWindows) {
  const std::filesystem::path path =
      L"\\\\?\\" + root_.native() + L"\\extended.txt";

  qdk::chemistry::utils::write_text_file_atomically(path, "contents");

  EXPECT_EQ(qdk::chemistry::utils::read_text_file(path), "contents");
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

TEST_F(FileIoTest, CompactTemporaryPathNeverAliasesDestination) {
  const auto path = root_ / ("qqqq0." + std::string(249, 'a'));
  std::filesystem::path observed_temporary_path;
  bool destination_visible = false;

  qdk::chemistry::utils::write_file_atomically(
      path, [&](const std::filesystem::path& temporary_path) {
        observed_temporary_path = temporary_path;
        destination_visible = std::filesystem::exists(path);
        std::ofstream output(temporary_path);
        output << "contents";
      });

  EXPECT_NE(observed_temporary_path, path);
  EXPECT_FALSE(destination_visible);
  EXPECT_EQ(qdk::chemistry::utils::read_text_file(path), "contents");
}
#endif

TEST_F(FileIoTest, CompactTemporaryPathUsesDistinctFilesystemIdentity) {
  const auto case_probe = root_ / "QdkCaseProbe";
  {
    std::ofstream output(case_probe);
    output << "probe";
  }
  if (!std::filesystem::exists(root_ / "qdkcaseprobe")) {
    GTEST_SKIP() << "Filesystem is case-sensitive";
  }
  std::filesystem::remove(case_probe);

  const auto path =
      root_ / ("Q" + std::string(14, 'q') + "0." + std::string(230, 'a'));
  {
    std::ofstream output(path);
    if (!output.is_open()) {
      GTEST_SKIP() << "Filesystem does not support the long test path";
    }
  }
  std::filesystem::remove(path);
  bool destination_visible = false;

  qdk::chemistry::utils::write_file_atomically(
      path, [&](const std::filesystem::path& temporary_path) {
        destination_visible = std::filesystem::exists(path);
        std::ofstream output(temporary_path);
        output << "contents";
      });

  EXPECT_FALSE(destination_visible);
  EXPECT_EQ(qdk::chemistry::utils::read_text_file(path), "contents");
}

TEST_F(FileIoTest, RejectsReplacedTemporaryFile) {
  const auto path = root_ / "data.txt";
  std::filesystem::path replacement_path;

  EXPECT_THROW(
      qdk::chemistry::utils::write_file_atomically(
          path,
          [&replacement_path](const std::filesystem::path& temporary_path) {
            replacement_path = temporary_path;
            std::filesystem::remove(temporary_path);
            std::ofstream output(temporary_path);
            output << "replacement";
          }),
      std::runtime_error);
  EXPECT_FALSE(std::filesystem::exists(path));
#ifdef _WIN32
  EXPECT_FALSE(std::filesystem::exists(replacement_path));
#else
  EXPECT_TRUE(std::filesystem::exists(replacement_path));
#endif
}

#ifndef _WIN32
TEST_F(FileIoTest, CleansReservedPathAfterWriterAddsHardLink) {
  const auto path = root_ / "data.txt";
  const auto extra_link = root_ / "extra.txt";
  std::filesystem::path temporary_path;

  EXPECT_THROW(qdk::chemistry::utils::write_file_atomically(
                   path,
                   [&](const std::filesystem::path& reserved_path) {
                     temporary_path = reserved_path;
                     std::ofstream output(reserved_path);
                     output << "sensitive";
                     output.close();
                     std::filesystem::create_hard_link(reserved_path,
                                                       extra_link);
                     throw std::runtime_error("writer failed");
                   }),
               std::runtime_error);

  EXPECT_FALSE(std::filesystem::exists(temporary_path));
  EXPECT_EQ(qdk::chemistry::utils::read_text_file(extra_link), "sensitive");
  EXPECT_FALSE(std::filesystem::exists(path));
}
#endif

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
