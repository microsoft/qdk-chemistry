// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "archive.hpp"

#include <cerrno>
#include <cstdlib>
#include <memory>
#include <qdk/chemistry/utils/logger.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <process.h>
// clang-format off
#include <windows.h>
#include <sddl.h>
// clang-format on
#else
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#ifdef __APPLE__
#include <crt_externs.h>
#else
extern char** environ;
#endif
#endif

namespace qdk::chemistry::utils {

ArchiveDirectory::ArchiveDirectory() {
  const auto root = std::filesystem::temp_directory_path();
  std::random_device device;
#ifdef _WIN32
  PSECURITY_DESCRIPTOR descriptor = nullptr;
  // The protected DACL grants access only to the new directory's owner.
  if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(
          L"D:P(A;OICI;FA;;;OW)", SDDL_REVISION_1, &descriptor, nullptr)) {
    throw std::system_error(GetLastError(), std::system_category(),
                            "Cannot create archive directory permissions");
  }
  const std::unique_ptr<void, decltype(&LocalFree)> security(descriptor,
                                                             &LocalFree);
  SECURITY_ATTRIBUTES attributes{sizeof(SECURITY_ATTRIBUTES), descriptor,
                                 FALSE};
#endif
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto candidate = root / ("qdk-basis-" + std::to_string(device()) + "-" +
                             std::to_string(device()));
#ifdef _WIN32
    if (CreateDirectoryW(candidate.c_str(), &attributes)) {
      _path = std::move(candidate);
      return;
    }
    const auto error = GetLastError();
    if (error != ERROR_ALREADY_EXISTS) {
      throw std::system_error(error, std::system_category(),
                              "Cannot create private archive directory");
    }
#else
    if (::mkdir(candidate.c_str(), S_IRWXU) == 0) {
      _path = std::move(candidate);
      return;
    }
    if (errno != EEXIST) {
      throw std::system_error(errno, std::generic_category(),
                              "Cannot create private archive directory");
    }
#endif
  }
  throw std::runtime_error("Cannot create a unique archive directory");
}

ArchiveDirectory::~ArchiveDirectory() {
  std::error_code error;
  std::filesystem::remove_all(_path, error);
  if (error) {
    QDK_LOGGER().warn("Cannot remove archive directory {}: {}", _path.string(),
                      error.message());
  }
}

void ArchiveDirectory::extract(const std::filesystem::path& archive) const {
  if (!std::filesystem::is_regular_file(archive)) {
    throw std::invalid_argument("Basis set file does not exist: " +
                                archive.string());
  }
  // Archive directory entries may change extraction-root permissions, so keep
  // that root underneath the private directory rather than extracting into it.
  const auto contents = _path / "contents";
  std::filesystem::create_directory(contents);
  // This fixed capability probe contains no archive or caller-supplied text.
#ifdef _WIN32
  static const bool force_local =
      std::system("tar --force-local --version > nul 2>&1") == 0;
#else
  static const bool force_local =
      std::system("tar --force-local --version > /dev/null 2>&1") == 0;
#endif
#ifdef _WIN32
  std::vector<std::wstring> arguments{L"tar"};
  if (force_local) {
    arguments.emplace_back(L"--force-local");
  }
  arguments.insert(arguments.end(), {L"-xzf", archive.generic_wstring(), L"-C",
                                     contents.generic_wstring()});
  for (auto& argument : arguments) {
    // The Windows CRT joins argv into a command line; quote for its parser,
    // not for cmd.exe (which is never used for the extraction).
    std::wstring quoted = L"\"";
    std::size_t backslashes = 0;
    for (const auto character : argument) {
      if (character == L'\\') {
        ++backslashes;
      } else {
        quoted.append(character == L'"' ? 2 * backslashes + 1 : backslashes,
                      L'\\');
        quoted += character;
        backslashes = 0;
      }
    }
    quoted.append(2 * backslashes, L'\\');
    quoted += L'"';
    argument = std::move(quoted);
  }
  std::vector<const wchar_t*> argv;
  for (const auto& argument : arguments) {
    argv.push_back(argument.c_str());
  }
  argv.push_back(nullptr);
  const auto result = _wspawnvp(_P_WAIT, L"tar", argv.data());
  if (result == -1) {
    throw std::system_error(errno, std::generic_category(),
                            "Cannot launch basis archive extraction");
  }
  if (result != 0) {
    throw std::runtime_error("Basis archive extraction failed: " +
                             archive.string());
  }
#else
  std::vector<std::string> arguments{"tar"};
  if (force_local) {
    arguments.emplace_back("--force-local");
  }
  arguments.insert(arguments.end(),
                   {"-xzf", archive.string(), "-C", contents.string()});
  std::vector<char*> argv;
  for (auto& argument : arguments) {
    argv.push_back(argument.data());
  }
  argv.push_back(nullptr);
  pid_t pid;
#ifdef __APPLE__
  auto environment = *_NSGetEnviron();
#else
  auto environment = environ;
#endif
  const int error =
      posix_spawnp(&pid, "tar", nullptr, nullptr, argv.data(), environment);
  if (error != 0) {
    throw std::system_error(error, std::generic_category(),
                            "Cannot launch basis archive extraction");
  }
  int status = 0;
  while (waitpid(pid, &status, 0) == -1) {
    if (errno != EINTR) {
      throw std::system_error(errno, std::generic_category(),
                              "Cannot wait for basis archive extraction");
    }
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    throw std::runtime_error("Basis archive extraction failed: " +
                             archive.string());
  }
#endif
}

std::filesystem::path ArchiveDirectory::file(
    const std::filesystem::path& relative) const {
  const auto root = std::filesystem::weakly_canonical(_path / "contents");
  const auto path = std::filesystem::weakly_canonical(root / relative);
  const auto within = path.lexically_relative(root);
  if (within.empty() || within.is_absolute() || *within.begin() == "..") {
    throw std::invalid_argument(
        "Extracted basis file is outside the extraction directory: " +
        relative.string());
  }
  if (!std::filesystem::is_regular_file(path)) {
    throw std::invalid_argument("Basis set JSON file does not exist: " +
                                path.string());
  }
  return path;
}

}  // namespace qdk::chemistry::utils
