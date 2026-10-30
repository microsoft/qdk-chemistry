// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "basis_archive.hpp"

#include <qdk/chemistry/scf/config.h>

#include <cerrno>
#include <cstdlib>
#include <memory>
#include <qdk/chemistry/utils/logger.hpp>
#include <random>
#include <stdexcept>
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

namespace {

// Shipped archive for a stem, or an empty path when the stem is not a single
// file name and so cannot name one.
std::filesystem::path archive_path(const std::string& stem) {
  const std::filesystem::path name = stem + ".tar.gz";
  if (stem.empty() || stem.find('\0') != std::string::npos ||
      name.filename() != name) {
    return {};
  }
  return scf::QDKChemistryConfig::get_resources_dir() / "compressed" / name;
}

std::filesystem::path create_private_directory() {
  const auto root = std::filesystem::temp_directory_path();
  std::random_device device;
#ifdef _WIN32
  PSECURITY_DESCRIPTOR descriptor = nullptr;
  // The protected DACL grants access only to the new directory's owner.
  if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(
          L"D:P(A;OICI;FA;;;OW)", SDDL_REVISION_1, &descriptor, nullptr)) {
    throw std::system_error(GetLastError(), std::system_category(),
                            "Cannot create basis archive permissions");
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
      return candidate;
    }
    const auto error = GetLastError();
    if (error != ERROR_ALREADY_EXISTS) {
      throw std::system_error(error, std::system_category(),
                              "Cannot create basis archive directory");
    }
#else
    if (::mkdir(candidate.c_str(), S_IRWXU) == 0) {
      return candidate;
    }
    if (errno != EEXIST) {
      throw std::system_error(errno, std::generic_category(),
                              "Cannot create basis archive directory");
    }
#endif
  }
  throw std::runtime_error("Cannot create a unique basis archive directory");
}

void remove_directory(const std::filesystem::path& path) {
  std::error_code error;
  std::filesystem::remove_all(path, error);
  if (error) {
    QDK_LOGGER().warn("Cannot remove basis archive directory {}: {}",
                      path.string(), error.message());
  }
}

// Runs tar with an argument vector, so no shell parses the paths.
void run_tar(const std::filesystem::path& archive,
             const std::filesystem::path& destination) {
#ifdef _WIN32
  // GNU tar from Git for Windows reads "C:/..." as host:path unless given
  // --force-local, which the System32 BSD tar rejects. This fixed probe
  // contains no archive or caller-supplied text.
  static const bool force_local =
      std::system("tar --force-local --version > nul 2>&1") == 0;
  std::vector<std::wstring> arguments{L"tar"};
  if (force_local) {
    arguments.emplace_back(L"--force-local");
  }
  arguments.insert(arguments.end(), {L"-xzf", archive.generic_wstring(), L"-C",
                                     destination.generic_wstring()});
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
  // GNU tar reads an archive name with a colon before its first slash as
  // host:path; an absolute path never has one.
  std::vector<std::string> arguments{
      "tar", "-xzf", std::filesystem::absolute(archive).string(), "-C",
      destination.string()};
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

}  // namespace

std::string BasisArchive::stem(const std::string& basis_name) {
  std::string stem;
  for (const char character : basis_name) {
    if (character == '*') {
      stem += "_st_";
    } else if (character == '/') {
      stem += "_sl_";
    } else {
      stem += character;
    }
  }
  return stem;
}

bool BasisArchive::exists(const std::string& basis_name) {
  const auto archive = archive_path(stem(basis_name));
  std::error_code error;
  return !archive.empty() && std::filesystem::is_regular_file(archive, error);
}

BasisArchive::BasisArchive(const std::string& basis_name)
    : _stem(stem(basis_name)) {
  const auto archive = archive_path(_stem);
  std::error_code error;
  if (archive.empty() || !std::filesystem::is_regular_file(archive, error)) {
    throw std::invalid_argument(
        "Basis set file does not exist: " +
        (archive.empty() ? basis_name : archive.string()));
  }
  _path = create_private_directory();
  try {
    run_tar(archive, _path);
  } catch (...) {
    remove_directory(_path);
    throw;
  }
}

BasisArchive::~BasisArchive() { remove_directory(_path); }

std::filesystem::path BasisArchive::json() const {
  auto path = _path / "basis" / (_stem + ".json");
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error)) {
    throw std::invalid_argument("Basis set JSON file does not exist: " +
                                path.string());
  }
  return path;
}

std::optional<std::filesystem::path> BasisArchive::guess(bool pure) const {
  auto path = _path / "guess" / (_stem + (pure ? "_pure" : "_cart"));
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error)) {
    return std::nullopt;
  }
  return path;
}

}  // namespace qdk::chemistry::utils
