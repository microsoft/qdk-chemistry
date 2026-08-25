// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <fstream>
#include <qdk/chemistry/utils/file_io.hpp>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace qdk::chemistry::utils {
namespace {

#ifndef _WIN32
template <typename Operation>
auto retry_on_eintr(Operation&& operation) -> decltype(operation()) {
  decltype(operation()) result;
  do {
    result = operation();
  } while (result == -1 && errno == EINTR);
  return result;
}
#endif

class ScopedReadHandle {
 public:
#ifdef _WIN32
  using NativeHandle = HANDLE;
#else
  using NativeHandle = int;
#endif

  static NativeHandle invalid_handle() {
#ifdef _WIN32
    return INVALID_HANDLE_VALUE;
#else
    return -1;
#endif
  }

  explicit ScopedReadHandle(NativeHandle handle) : handle_(handle) {}
  ScopedReadHandle(const ScopedReadHandle&) = delete;
  ScopedReadHandle& operator=(const ScopedReadHandle&) = delete;

  ~ScopedReadHandle() {
    if (handle_ == invalid_handle()) {
      return;
    }
#ifdef _WIN32
    CloseHandle(handle_);
#else
    ::close(handle_);
#endif
  }

  NativeHandle get() const { return handle_; }

 private:
  NativeHandle handle_;
};

std::string display_path(const std::filesystem::path& path) {
  const auto value = path.u8string();
  return {value.begin(), value.end()};
}

#ifndef _WIN32
class TransientPermissionError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
#endif

[[noreturn]] void throw_directory_error(const std::string& action,
                                        const std::filesystem::path& path,
                                        const std::error_code& error) {
  const auto message =
      action + " '" + display_path(path) + "': " + error.message();
#ifndef _WIN32
  if (error == std::errc::permission_denied) {
    throw TransientPermissionError(message);
  }
#endif
  throw std::runtime_error(message);
}

#ifndef _WIN32
bool has_initializing_directory(const std::filesystem::path& directory) {
  auto current = directory;
  while (!current.empty()) {
    struct stat status{};
    if (retry_on_eintr([&] { return ::stat(current.c_str(), &status); }) == 0) {
      if (S_ISDIR(status.st_mode) && status.st_uid == geteuid() &&
          (status.st_mode & 0777) == 0) {
        return true;
      }
    } else if (errno != EACCES && errno != ENOENT) {
      return false;
    }
    const auto parent = current.parent_path();
    if (parent == current) {
      return false;
    }
    current = parent;
  }
  return false;
}
#endif

void validate_path(const std::filesystem::path& path) {
  const auto& native_path = path.native();
  if (native_path.find(static_cast<std::filesystem::path::value_type>('\0')) !=
      std::filesystem::path::string_type::npos) {
    throw std::invalid_argument("Path contains an embedded NUL character");
  }
  const auto filename = path.filename();
  if (filename.empty() || filename == "." || filename == "..") {
    throw std::invalid_argument("Path must name a file");
  }
#ifdef _WIN32
  if (path.filename().native().find(
          static_cast<std::filesystem::path::value_type>(':')) !=
      std::filesystem::path::string_type::npos) {
    throw std::invalid_argument(
        "Windows alternate data streams are not supported");
  }
#endif
}

std::filesystem::path freeze_path(const std::filesystem::path& path) {
  std::error_code error;
  const auto frozen_path = std::filesystem::absolute(path, error);
  if (error) {
    throw std::runtime_error("Could not resolve absolute path for '" +
                             display_path(path) + "': " + error.message());
  }
  return frozen_path;
}

#ifdef _WIN32
enum class IdentityMatch { match, different, unknown };

IdentityMatch compare_handle_to_path_identity(
    HANDLE handle, const std::filesystem::path& path,
    bool require_single_link) noexcept {
  if (handle == INVALID_HANDLE_VALUE) {
    return IdentityMatch::unknown;
  }
  BY_HANDLE_FILE_INFORMATION reserved_info{};
  if (!GetFileInformationByHandle(handle, &reserved_info)) {
    return IdentityMatch::unknown;
  }
  HANDLE current_handle = CreateFileW(
      path.c_str(), FILE_READ_ATTRIBUTES,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
      OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  if (current_handle == INVALID_HANDLE_VALUE) {
    return IdentityMatch::unknown;
  }
  BY_HANDLE_FILE_INFORMATION current_info{};
  const bool inspected =
      GetFileInformationByHandle(current_handle, &current_info) != 0;
  CloseHandle(current_handle);
  if (!inspected) {
    return IdentityMatch::unknown;
  }
  const bool matches =
      reserved_info.dwVolumeSerialNumber == current_info.dwVolumeSerialNumber &&
      reserved_info.nFileIndexHigh == current_info.nFileIndexHigh &&
      reserved_info.nFileIndexLow == current_info.nFileIndexLow &&
      (current_info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) == 0 &&
      (!require_single_link || current_info.nNumberOfLinks == 1);
  return matches ? IdentityMatch::match : IdentityMatch::different;
}
#endif

std::filesystem::path make_temporary_path(
    const std::filesystem::path& destination) {
  static std::atomic<unsigned long long> counter{0};
  const auto timestamp =
      std::chrono::steady_clock::now().time_since_epoch().count();

  std::filesystem::path temporary_name = ".qdk-tmp-";
  temporary_name += std::to_string(timestamp);
  temporary_name += "-";
  temporary_name += std::to_string(counter.fetch_add(1));

  const auto filename = destination.filename().native();
  const auto dot = filename.find(
      static_cast<std::filesystem::path::value_type>('.'),
      !filename.empty() &&
              filename.front() ==
                  static_cast<std::filesystem::path::value_type>('.')
          ? 1
          : 0);
  if (dot != decltype(filename)::npos) {
    temporary_name += filename.substr(dot);
  }
  return destination.parent_path() / temporary_name;
}

std::filesystem::path make_compact_temporary_path(
    const std::filesystem::path& destination, int attempt, int stem_length) {
  constexpr std::string_view alphabet =
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_-";
  std::filesystem::path temporary_name =
      std::string(static_cast<std::size_t>(stem_length - 1), 'q') +
      alphabet[static_cast<std::size_t>(attempt)];

  const auto filename = destination.filename().native();
  const auto dot = filename.find(
      static_cast<std::filesystem::path::value_type>('.'),
      !filename.empty() &&
              filename.front() ==
                  static_cast<std::filesystem::path::value_type>('.')
          ? 1
          : 0);
  if (dot != decltype(filename)::npos) {
    temporary_name += filename.substr(dot);
  }
  return destination.parent_path() / temporary_name;
}

#ifdef _WIN32
DWORD normalized_file_attributes(DWORD attributes) {
  constexpr DWORD supported_attributes =
      FILE_ATTRIBUTE_ARCHIVE | FILE_ATTRIBUTE_HIDDEN |
      FILE_ATTRIBUTE_NOT_CONTENT_INDEXED | FILE_ATTRIBUTE_OFFLINE |
      FILE_ATTRIBUTE_READONLY | FILE_ATTRIBUTE_SYSTEM |
      FILE_ATTRIBUTE_TEMPORARY;
  const DWORD result = attributes & supported_attributes;
  return result == 0 ? FILE_ATTRIBUTE_NORMAL : result;
}

DWORD replacement_file_attributes(DWORD attributes) {
  const DWORD result = attributes & FILE_ATTRIBUTE_READONLY;
  return result == 0 ? FILE_ATTRIBUTE_NORMAL : result;
}

bool set_handle_file_attributes(HANDLE handle, DWORD attributes) noexcept {
  if (handle == INVALID_HANDLE_VALUE) {
    return false;
  }
  FILE_BASIC_INFO info{};
  if (!GetFileInformationByHandleEx(handle, FileBasicInfo, &info,
                                    sizeof(info))) {
    return false;
  }
  info.FileAttributes = normalized_file_attributes(attributes);
  return SetFileInformationByHandle(handle, FileBasicInfo, &info,
                                    sizeof(info)) != 0;
}
#endif

class ReservedTemporaryFile {
 public:
#ifdef _WIN32
  using NativeHandle = HANDLE;
#else
  using NativeHandle = int;
#endif

  static NativeHandle invalid_handle() {
#ifdef _WIN32
    return INVALID_HANDLE_VALUE;
#else
    return -1;
#endif
  }

  ReservedTemporaryFile(std::filesystem::path path, NativeHandle handle)
      : path_(std::move(path)),
        handle_(handle),
        cleanup_(handle != invalid_handle()) {}

  ReservedTemporaryFile(const ReservedTemporaryFile&) = delete;
  ReservedTemporaryFile& operator=(const ReservedTemporaryFile&) = delete;

  ReservedTemporaryFile(ReservedTemporaryFile&& other) noexcept
      : path_(std::move(other.path_)),
        handle_(other.handle_),
        cleanup_(other.cleanup_) {
    other.handle_ = invalid_handle();
    other.cleanup_ = false;
  }

  ~ReservedTemporaryFile() {
    if (cleanup_ && path_matches_identity(path_, false)) {
#ifdef _WIN32
      const DWORD attributes = GetFileAttributesW(path_.c_str());
      if (attributes != INVALID_FILE_ATTRIBUTES &&
          (attributes & FILE_ATTRIBUTE_READONLY) != 0) {
        SetFileAttributesW(
            path_.c_str(),
            normalized_file_attributes(attributes & ~FILE_ATTRIBUTE_READONLY));
      }
#endif
      std::error_code ignored;
      std::filesystem::remove(path_, ignored);
    }
    close();
  }

  const std::filesystem::path& path() const { return path_; }
  bool has_same_identity(const std::filesystem::path& path) const noexcept {
    return path_matches_identity(path, false);
  }

  void verify_identity() const {
#ifdef _WIN32
    BY_HANDLE_FILE_INFORMATION reserved_info;
    if (!GetFileInformationByHandle(handle_, &reserved_info)) {
      throw_last_error("Could not inspect reserved temporary file");
    }

    HANDLE current_handle = CreateFileW(
        path_.c_str(), FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
        OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (current_handle == INVALID_HANDLE_VALUE) {
      throw_last_error("Could not inspect temporary path");
    }

    BY_HANDLE_FILE_INFORMATION current_info;
    const bool inspected =
        GetFileInformationByHandle(current_handle, &current_info);
    const DWORD inspection_error = inspected ? ERROR_SUCCESS : GetLastError();
    CloseHandle(current_handle);
    if (!inspected) {
      throw_windows_error("Could not inspect temporary path", inspection_error);
    }

    const bool same_file =
        reserved_info.dwVolumeSerialNumber ==
            current_info.dwVolumeSerialNumber &&
        reserved_info.nFileIndexHigh == current_info.nFileIndexHigh &&
        reserved_info.nFileIndexLow == current_info.nFileIndexLow;
    if (!same_file ||
        (current_info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0 ||
        current_info.nNumberOfLinks != 1) {
      throw std::runtime_error("Temporary file identity changed: '" +
                               display_path(path_) + "'");
    }
#else
    struct stat reserved_status{};
    struct stat current_status{};
    if (::fstat(handle_, &reserved_status) != 0 ||
        ::lstat(path_.c_str(), &current_status) != 0) {
      throw std::runtime_error("Could not inspect temporary file: '" +
                               display_path(path_) + "'");
    }
    if (reserved_status.st_dev != current_status.st_dev ||
        reserved_status.st_ino != current_status.st_ino ||
        !S_ISREG(current_status.st_mode) || current_status.st_nlink != 1) {
      throw std::runtime_error("Temporary file identity changed: '" +
                               display_path(path_) + "'");
    }
#endif
  }

  void set_permissions(std::filesystem::perms permissions) {
#ifdef _WIN32
    static_cast<void>(permissions);
#else
    const auto requested = static_cast<mode_t>(permissions) & 0777;
    struct stat status{};
    if (retry_on_eintr([&] { return ::fchmod(handle_, requested); }) != 0 ||
        retry_on_eintr([&] { return ::fstat(handle_, &status); }) != 0 ||
        (status.st_mode & 0777) != requested) {
      throw std::runtime_error("Could not set temporary file permissions: '" +
                               display_path(path_) + "'");
    }
#endif
  }

  void release() {
    close();
    cleanup_ = false;
  }

 private:
#ifdef _WIN32
  [[noreturn]] static void throw_windows_error(const std::string& message,
                                               DWORD windows_error) {
    const std::error_code error(static_cast<int>(windows_error),
                                std::system_category());
    throw std::runtime_error(message + ": " + error.message());
  }

  [[noreturn]] static void throw_last_error(const std::string& message) {
    throw_windows_error(message, GetLastError());
  }
#endif

  bool path_matches_identity(const std::filesystem::path& path,
                             bool require_single_link) const noexcept {
    if (handle_ == invalid_handle()) {
      return false;
    }
#ifdef _WIN32
    return compare_handle_to_path_identity(
               handle_, path, require_single_link) == IdentityMatch::match;
#else
    struct stat reserved_status{};
    struct stat current_status{};
    return ::fstat(handle_, &reserved_status) == 0 &&
           ::lstat(path.c_str(), &current_status) == 0 &&
           reserved_status.st_dev == current_status.st_dev &&
           reserved_status.st_ino == current_status.st_ino &&
           S_ISREG(current_status.st_mode) &&
           (!require_single_link || current_status.st_nlink == 1);
#endif
  }

  void close() {
    if (handle_ == invalid_handle()) {
      return;
    }
#ifdef _WIN32
    CloseHandle(handle_);
#else
    ::close(handle_);
#endif
    handle_ = invalid_handle();
  }

  std::filesystem::path path_;
  NativeHandle handle_ = invalid_handle();
  bool cleanup_ = true;
};

ReservedTemporaryFile create_exclusive_file(const std::filesystem::path& path,
                                            std::error_code& error) {
  std::filesystem::path owned_path(path);
#ifdef _WIN32
  HANDLE handle =
      CreateFileW(owned_path.c_str(), 0,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    error = std::error_code(static_cast<int>(GetLastError()),
                            std::system_category());
    return {std::move(owned_path), ReservedTemporaryFile::invalid_handle()};
  }
#else
  const int descriptor = retry_on_eintr([&] {
    return ::open(owned_path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC,
                  0600);
  });
  if (descriptor == -1) {
    error = std::error_code(errno, std::generic_category());
    return {std::move(owned_path), ReservedTemporaryFile::invalid_handle()};
  }
  if (retry_on_eintr([&] { return ::fchmod(descriptor, S_IRUSR | S_IWUSR); }) !=
      0) {
    const int permission_error = errno;
    error = std::error_code(permission_error, std::generic_category());
    return {std::move(owned_path), descriptor};
  }
  struct stat status{};
  if (retry_on_eintr([&] { return ::fstat(descriptor, &status); }) != 0 ||
      (status.st_mode & 0777) != (S_IRUSR | S_IWUSR)) {
    error = std::make_error_code(std::errc::permission_denied);
    return {std::move(owned_path), descriptor};
  }
#endif
  return {std::move(owned_path),
#ifdef _WIN32
          handle
#else
          descriptor
#endif
  };
}

bool is_name_too_long(const std::error_code& error,
                      const std::filesystem::path& destination) {
  if (error == std::errc::filename_too_long) {
    return true;
  }
#ifdef _WIN32
  if (error.value() == ERROR_FILENAME_EXCED_RANGE ||
      error.value() == ERROR_BUFFER_OVERFLOW) {
    return true;
  }
  if (error.value() == ERROR_PATH_NOT_FOUND) {
    std::error_code parent_error;
    return std::filesystem::is_directory(destination.parent_path(),
                                         parent_error) &&
           !parent_error;
  }
#else
  static_cast<void>(destination);
#endif
  return false;
}

ReservedTemporaryFile reserve_temporary_file(
    const std::filesystem::path& destination) {
  constexpr int max_attempts = 64;
  for (int attempt = 0; attempt < max_attempts; ++attempt) {
    const auto temporary_path = make_temporary_path(destination);
#ifdef _WIN32
    if (temporary_path.filename().native().size() > 255) {
      break;
    }
#endif
    std::error_code error;
    auto temporary_file = create_exclusive_file(temporary_path, error);
    if (!error) {
      if (temporary_file.has_same_identity(destination)) {
        continue;
      }
      return temporary_file;
    }
    if (error != std::errc::file_exists) {
      if (is_name_too_long(error, destination)) {
        break;
      }
      throw_directory_error("Could not create temporary file beside",
                            destination, error);
    }
  }

  for (int stem_length = 16; stem_length > 0; --stem_length) {
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
      const auto temporary_path =
          make_compact_temporary_path(destination, attempt, stem_length);
      if (temporary_path == destination) {
        continue;
      }
#ifdef _WIN32
      if (temporary_path.filename().native().size() > 255) {
        break;
      }
#endif
      std::error_code error;
      auto temporary_file = create_exclusive_file(temporary_path, error);
      if (!error) {
        if (temporary_file.has_same_identity(destination)) {
          continue;
        }
        return temporary_file;
      }
      if (error == std::errc::file_exists) {
        continue;
      }
      if (is_name_too_long(error, destination)) {
        break;
      }
      throw_directory_error("Could not create temporary file beside",
                            destination, error);
    }
  }

  throw std::runtime_error("Could not create a unique temporary file beside '" +
                           display_path(destination) + "'");
}

void replace_file(const std::filesystem::path& source,
                  const std::filesystem::path& destination) {
#ifdef _WIN32
  bool destination_exists = true;
  HANDLE original_handle_value = INVALID_HANDLE_VALUE;
  BY_HANDLE_FILE_INFORMATION original_info{};
  DWORD original_attributes = FILE_ATTRIBUTE_NORMAL;
  bool can_write_original_attributes = false;
  original_handle_value = CreateFileW(
      destination.c_str(), FILE_READ_ATTRIBUTES | FILE_WRITE_ATTRIBUTES,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
      OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  can_write_original_attributes = original_handle_value != INVALID_HANDLE_VALUE;
  DWORD inspection_error =
      can_write_original_attributes ? ERROR_SUCCESS : GetLastError();
  if (original_handle_value == INVALID_HANDLE_VALUE &&
      inspection_error == ERROR_ACCESS_DENIED) {
    original_handle_value = CreateFileW(
        destination.c_str(), FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
        OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    inspection_error = original_handle_value == INVALID_HANDLE_VALUE
                           ? GetLastError()
                           : ERROR_SUCCESS;
  }
  if (original_handle_value == INVALID_HANDLE_VALUE) {
    if (inspection_error == ERROR_FILE_NOT_FOUND ||
        inspection_error == ERROR_PATH_NOT_FOUND) {
      destination_exists = false;
    } else {
      const std::error_code error(static_cast<int>(inspection_error),
                                  std::system_category());
      throw std::runtime_error("Could not inspect file attributes for '" +
                               display_path(destination) +
                               "': " + error.message());
    }
  } else {
    if (!GetFileInformationByHandle(original_handle_value, &original_info)) {
      const std::error_code error(static_cast<int>(GetLastError()),
                                  std::system_category());
      CloseHandle(original_handle_value);
      throw std::runtime_error("Could not inspect file attributes for '" +
                               display_path(destination) +
                               "': " + error.message());
    }
    original_attributes = original_info.dwFileAttributes;
  }
  const ScopedReadHandle original_handle(original_handle_value);
  if (destination_exists &&
      !SetFileAttributesW(source.c_str(),
                          replacement_file_attributes(original_attributes))) {
    const std::error_code error(static_cast<int>(GetLastError()),
                                std::system_category());
    throw std::runtime_error("Could not prepare file attributes for '" +
                             display_path(destination) +
                             "': " + error.message());
  }

  auto move = [&]() {
    return MoveFileExW(source.c_str(), destination.c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
  };
  if (move()) {
    return;
  }

  const DWORD first_error = GetLastError();
  const bool read_only = destination_exists &&
                         (original_attributes & FILE_ATTRIBUTE_READONLY) != 0;
  if (first_error != ERROR_ACCESS_DENIED || !read_only) {
    const std::error_code error(static_cast<int>(first_error),
                                std::system_category());
    throw std::runtime_error("Could not replace file '" +
                             display_path(destination) +
                             "': " + error.message());
  }
  if (original_handle.get() == INVALID_HANDLE_VALUE) {
    const std::error_code error(static_cast<int>(GetLastError()),
                                std::system_category());
    throw std::runtime_error("Could not inspect read-only destination '" +
                             display_path(destination) +
                             "': " + error.message());
  }
  if (!can_write_original_attributes) {
    throw std::runtime_error(
        "Could not replace read-only file without permission to change its "
        "attributes: '" +
        display_path(destination) + "'");
  }
  if (original_info.nNumberOfLinks != 1) {
    throw std::runtime_error(
        "Read-only Windows destinations with multiple hard links are not "
        "supported: '" +
        display_path(destination) + "'");
  }
  if (!set_handle_file_attributes(
          original_handle.get(),
          original_attributes & ~FILE_ATTRIBUTE_READONLY)) {
    const std::error_code error(static_cast<int>(GetLastError()),
                                std::system_category());
    throw std::runtime_error("Could not prepare read-only destination '" +
                             display_path(destination) +
                             "': " + error.message());
  }

  const bool replaced = move();
  const DWORD retry_error = replaced ? ERROR_SUCCESS : GetLastError();
  if (replaced) {
    BY_HANDLE_FILE_INFORMATION displaced_info{};
    if (GetFileInformationByHandle(original_handle.get(), &displaced_info) &&
        displaced_info.nNumberOfLinks > 0) {
      if (!set_handle_file_attributes(original_handle.get(),
                                      original_attributes)) {
        const std::error_code error(static_cast<int>(GetLastError()),
                                    std::system_category());
        throw std::runtime_error(
            "Could not restore attributes on the displaced file for '" +
            display_path(destination) + "': " + error.message());
      }
    }
    return;
  }
  if (!set_handle_file_attributes(original_handle.get(), original_attributes)) {
    const std::error_code rollback_error(static_cast<int>(GetLastError()),
                                         std::system_category());
    const std::error_code error(static_cast<int>(retry_error),
                                std::system_category());
    throw std::runtime_error(
        "Could not replace file '" + display_path(destination) +
        "': " + error.message() +
        "; could not restore original attributes: " + rollback_error.message());
  }
  const std::error_code error(static_cast<int>(retry_error),
                              std::system_category());
  throw std::runtime_error("Could not replace file '" +
                           display_path(destination) + "': " + error.message());
#else
  std::error_code error;
  std::filesystem::rename(source, destination, error);
  if (error) {
    throw std::runtime_error("Could not replace file '" +
                             display_path(destination) +
                             "': " + error.message());
  }
#endif
}

void preserve_permissions(ReservedTemporaryFile& temporary_file,
                          const std::filesystem::path& destination) {
  std::error_code status_error;
  const auto status =
      std::filesystem::symlink_status(destination, status_error);
  if (status_error) {
    if (status_error == std::errc::no_such_file_or_directory) {
#ifndef _WIN32
      temporary_file.set_permissions(std::filesystem::perms::owner_read |
                                     std::filesystem::perms::owner_write);
#endif
      return;
    }
    throw std::runtime_error("Could not inspect permissions for '" +
                             display_path(destination) +
                             "': " + status_error.message());
  }
  if (!std::filesystem::exists(status)) {
#ifndef _WIN32
    temporary_file.set_permissions(std::filesystem::perms::owner_read |
                                   std::filesystem::perms::owner_write);
#endif
    return;
  }
  if (std::filesystem::is_symlink(status)) {
    throw std::runtime_error("Symlink destinations are not supported: '" +
                             display_path(destination) + "'");
  }
  if (!std::filesystem::is_regular_file(status)) {
    throw std::runtime_error("Destination is not a regular file: '" +
                             display_path(destination) + "'");
  }

  temporary_file.set_permissions(status.permissions() &
                                 std::filesystem::perms::all);
}

void create_private_directories(const std::filesystem::path& directory) {
#ifdef _WIN32
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error) {
    throw std::runtime_error("Could not create directory '" +
                             display_path(directory) + "': " + error.message());
  }
#else
  constexpr auto retry_delay = std::chrono::milliseconds(1);
  constexpr auto retry_timeout = std::chrono::seconds(1);
  const auto deadline = std::chrono::steady_clock::now() + retry_timeout;

  auto create_once = [&]() {
    std::vector<std::filesystem::path> missing;
    auto current = directory;
    std::error_code status_error;
    while (!current.empty() &&
           !std::filesystem::is_directory(current, status_error)) {
      if (status_error &&
          status_error != std::errc::no_such_file_or_directory) {
        throw_directory_error("Could not inspect directory", current,
                              status_error);
      }
      status_error.clear();
      missing.push_back(current);
      const auto parent = current.parent_path();
      if (parent == current) {
        break;
      }
      current = parent;
    }

    for (auto iterator = missing.rbegin(); iterator != missing.rend();
         ++iterator) {
      if (retry_on_eintr([&] { return ::mkdir(iterator->c_str(), S_IRWXU); }) !=
          0) {
        const int mkdir_error = errno;
        if (mkdir_error == EEXIST) {
          std::error_code existing_error;
          if (std::filesystem::is_directory(*iterator, existing_error) &&
              !existing_error) {
            continue;
          }
          if (existing_error) {
            throw_directory_error("Could not inspect directory", *iterator,
                                  existing_error);
          }
        }
        throw_directory_error(
            "Could not create directory", *iterator,
            std::error_code(mkdir_error, std::generic_category()));
      }
      if (retry_on_eintr([&] { return ::chmod(iterator->c_str(), S_IRWXU); }) !=
          0) {
        const std::error_code error(errno, std::generic_category());
        std::error_code ignored;
        std::filesystem::remove(*iterator, ignored);
        throw_directory_error("Could not secure directory", *iterator, error);
      }
      const int descriptor = retry_on_eintr([&] {
        return ::open(iterator->c_str(),
                      O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
      });
      struct stat status{};
      int permission_error = 0;
      if (descriptor == -1) {
        permission_error = errno;
      } else if (retry_on_eintr(
                     [&] { return ::fchmod(descriptor, S_IRWXU); }) != 0) {
        permission_error = errno;
      } else if (retry_on_eintr([&] { return ::fstat(descriptor, &status); }) !=
                 0) {
        permission_error = errno;
      } else if ((status.st_mode & 0777) != S_IRWXU) {
        permission_error = EPERM;
      }
      if (permission_error != 0) {
        const std::error_code error(permission_error, std::generic_category());
        if (descriptor != -1) {
          ::close(descriptor);
        }
        std::error_code ignored;
        std::filesystem::remove(*iterator, ignored);
        throw_directory_error("Could not secure directory", *iterator, error);
      }
      ::close(descriptor);
    }
  };

  while (true) {
    try {
      create_once();
      return;
    } catch (const TransientPermissionError&) {
      if (!has_initializing_directory(directory) ||
          std::chrono::steady_clock::now() >= deadline) {
        throw;
      }
      std::this_thread::sleep_for(retry_delay);
    }
  }
#endif
}

}  // namespace

void ensure_parent_directory(const std::filesystem::path& path) {
  validate_path(path);
  const auto parent = path.parent_path();
  if (parent.empty()) {
    return;
  }

  create_private_directories(freeze_path(path).parent_path());
}

std::string read_text_file(const std::filesystem::path& path) {
  validate_path(path);
  std::string contents;
#ifdef _WIN32
  HANDLE handle =
      CreateFileW(path.c_str(), GENERIC_READ,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    throw std::runtime_error("Could not open file for reading: '" +
                             display_path(path) + "'");
  }
  const ScopedReadHandle scoped_handle(handle);

  BY_HANDLE_FILE_INFORMATION info;
  if (!GetFileInformationByHandle(handle, &info) ||
      (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0 ||
      GetFileType(handle) != FILE_TYPE_DISK) {
    throw std::runtime_error("Path is not a regular file: '" +
                             display_path(path) + "'");
  }

  std::array<char, 8192> buffer{};
  while (true) {
    DWORD bytes_read = 0;
    if (!ReadFile(handle, buffer.data(), static_cast<DWORD>(buffer.size()),
                  &bytes_read, nullptr)) {
      const std::error_code error(static_cast<int>(GetLastError()),
                                  std::system_category());
      throw std::runtime_error("Could not read file: '" + display_path(path) +
                               "': " + error.message());
    }
    if (bytes_read == 0) {
      break;
    }
    contents.append(buffer.data(), bytes_read);
  }
#else
  int open_flags = O_RDONLY | O_NONBLOCK | O_CLOEXEC;
#ifdef O_NOCTTY
  open_flags |= O_NOCTTY;
#endif
  const int descriptor =
      retry_on_eintr([&] { return ::open(path.c_str(), open_flags); });
  if (descriptor == -1) {
    throw std::runtime_error("Could not open file for reading: '" +
                             display_path(path) + "'");
  }
  const ScopedReadHandle scoped_descriptor(descriptor);

  struct stat status{};
  if (::fstat(descriptor, &status) != 0 || !S_ISREG(status.st_mode)) {
    throw std::runtime_error("Path is not a regular file: '" +
                             display_path(path) + "'");
  }

  std::array<char, 8192> buffer{};
  while (true) {
    const ssize_t bytes_read = ::read(descriptor, buffer.data(), buffer.size());
    if (bytes_read > 0) {
      contents.append(buffer.data(), static_cast<std::size_t>(bytes_read));
      continue;
    }
    if (bytes_read < 0 && errno == EINTR) {
      continue;
    }
    if (bytes_read < 0) {
      const int read_error = errno;
      throw std::runtime_error(
          "Could not read file: '" + display_path(path) + "': " +
          std::error_code(read_error, std::generic_category()).message());
    }
    break;
  }
#endif
  return contents;
}

void write_file_atomically(const std::filesystem::path& path,
                           const AtomicFileWriter& writer,
                           bool create_parent_directories) {
  validate_path(path);
  const auto destination = freeze_path(path);

  if (create_parent_directories) {
    ensure_parent_directory(destination);
  }

  const auto parent = destination.parent_path();
  if (!parent.empty()) {
    std::error_code error;
    const bool parent_is_directory =
        std::filesystem::is_directory(parent, error);
    if (error || !parent_is_directory) {
      throw std::runtime_error("Parent directory does not exist for '" +
                               display_path(path) + "'");
    }
  }

  auto temporary_file = [&]() {
#ifdef _WIN32
    return reserve_temporary_file(destination);
#else
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (true) {
      try {
        return reserve_temporary_file(destination);
      } catch (const TransientPermissionError&) {
        if (!create_parent_directories ||
            !has_initializing_directory(destination.parent_path()) ||
            std::chrono::steady_clock::now() >= deadline) {
          throw;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
#endif
  }();
  writer(temporary_file.path());
  temporary_file.verify_identity();
  preserve_permissions(temporary_file, destination);
  temporary_file.verify_identity();
  replace_file(temporary_file.path(), destination);
  temporary_file.release();
}

void write_text_file_atomically(const std::filesystem::path& path,
                                std::string_view contents,
                                bool create_parent_directories) {
  write_file_atomically(
      path,
      [contents, &path](const std::filesystem::path& temporary_path) {
        std::ofstream output(temporary_path,
                             std::ios::binary | std::ios::trunc);
        if (!output.is_open()) {
          throw std::runtime_error(
              "Could not open temporary file for writing "
              "destination '" +
              display_path(path) + "'");
        }
        output.write(contents.data(),
                     static_cast<std::streamsize>(contents.size()));
        output.close();
        if (!output) {
          throw std::runtime_error("Could not write file: '" +
                                   display_path(path) + "'");
        }
      },
      create_parent_directories);
}

}  // namespace qdk::chemistry::utils
