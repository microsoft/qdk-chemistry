// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <qdk/chemistry/utils/file_io.hpp>
#include <stdexcept>
#include <system_error>

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
DWORD settable_file_attributes(DWORD attributes) {
  constexpr DWORD supported_attributes =
      FILE_ATTRIBUTE_ARCHIVE | FILE_ATTRIBUTE_HIDDEN |
      FILE_ATTRIBUTE_NOT_CONTENT_INDEXED | FILE_ATTRIBUTE_OFFLINE |
      FILE_ATTRIBUTE_READONLY | FILE_ATTRIBUTE_SYSTEM |
      FILE_ATTRIBUTE_TEMPORARY;
  const DWORD result = attributes & supported_attributes;
  return result == 0 ? FILE_ATTRIBUTE_NORMAL : result;
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
    close();
    if (cleanup_) {
#ifdef _WIN32
      const DWORD attributes = GetFileAttributesW(path_.c_str());
      if (attributes != INVALID_FILE_ATTRIBUTES &&
          (attributes & FILE_ATTRIBUTE_READONLY) != 0) {
        SetFileAttributesW(
            path_.c_str(),
            settable_file_attributes(attributes & ~FILE_ATTRIBUTE_READONLY));
      }
#endif
      std::error_code ignored;
      std::filesystem::remove(path_, ignored);
    }
  }

  const std::filesystem::path& path() const { return path_; }

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
    if (::fchmod(handle_, static_cast<mode_t>(permissions)) != 0) {
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
#ifdef _WIN32
  HANDLE handle =
      CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    error = std::error_code(static_cast<int>(GetLastError()),
                            std::system_category());
    return {path, ReservedTemporaryFile::invalid_handle()};
  }
#else
  const int descriptor =
      ::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
  if (descriptor == -1) {
    error = std::error_code(errno, std::generic_category());
    return {path, ReservedTemporaryFile::invalid_handle()};
  }
#endif
  return {path,
#ifdef _WIN32
          handle
#else
          descriptor
#endif
  };
}

ReservedTemporaryFile reserve_temporary_file(
    const std::filesystem::path& destination) {
  constexpr int max_attempts = 64;
  for (int attempt = 0; attempt < max_attempts; ++attempt) {
    const auto temporary_path = make_temporary_path(destination);
    std::error_code error;
    auto temporary_file = create_exclusive_file(temporary_path, error);
    if (!error) {
      return temporary_file;
    }
    if (error != std::errc::file_exists) {
      if (error == std::errc::filename_too_long) {
        break;
      }
      throw std::runtime_error("Could not create temporary file beside '" +
                               display_path(destination) +
                               "': " + error.message());
    }
  }

  for (int stem_length = 16; stem_length > 0; --stem_length) {
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
      const auto temporary_path =
          make_compact_temporary_path(destination, attempt, stem_length);
      std::error_code error;
      auto temporary_file = create_exclusive_file(temporary_path, error);
      if (!error) {
        return temporary_file;
      }
      if (error == std::errc::file_exists) {
        continue;
      }
      if (error == std::errc::filename_too_long) {
        break;
      }
      throw std::runtime_error("Could not create temporary file beside '" +
                               display_path(destination) +
                               "': " + error.message());
    }
  }

  throw std::runtime_error("Could not create a unique temporary file beside '" +
                           display_path(destination) + "'");
}

void replace_file(const std::filesystem::path& source,
                  const std::filesystem::path& destination) {
#ifdef _WIN32
  const DWORD original_attributes = GetFileAttributesW(destination.c_str());
  const bool destination_exists =
      original_attributes != INVALID_FILE_ATTRIBUTES;

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
  if (first_error != ERROR_ACCESS_DENIED || !read_only ||
      !SetFileAttributesW(destination.c_str(),
                          settable_file_attributes(original_attributes &
                                                   ~FILE_ATTRIBUTE_READONLY))) {
    const std::error_code error(static_cast<int>(first_error),
                                std::system_category());
    throw std::runtime_error("Could not replace file '" +
                             display_path(destination) +
                             "': " + error.message());
  }

  if (!move()) {
    const DWORD retry_error = GetLastError();
    SetFileAttributesW(destination.c_str(),
                       settable_file_attributes(original_attributes));
    const std::error_code error(static_cast<int>(retry_error),
                                std::system_category());
    throw std::runtime_error("Could not replace file '" +
                             display_path(destination) +
                             "': " + error.message());
  }
  if (!SetFileAttributesW(destination.c_str(),
                          settable_file_attributes(original_attributes))) {
    const std::error_code error(static_cast<int>(GetLastError()),
                                std::system_category());
    throw std::runtime_error("Could not restore file attributes for '" +
                             display_path(destination) +
                             "': " + error.message());
  }
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
  const auto status = std::filesystem::status(destination, status_error);
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

  temporary_file.set_permissions(status.permissions());
}

}  // namespace

void ensure_parent_directory(const std::filesystem::path& path) {
  const auto parent = path.parent_path();
  if (parent.empty()) {
    return;
  }

  std::error_code error;
  std::filesystem::create_directories(parent, error);
  if (error) {
    throw std::runtime_error("Could not create parent directory for '" +
                             display_path(path) + "': " + error.message());
  }
}

std::string read_text_file(const std::filesystem::path& path) {
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
  const int descriptor =
      ::open(path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
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
  std::error_code absolute_error;
  const auto destination = std::filesystem::absolute(path, absolute_error);
  if (absolute_error) {
    throw std::runtime_error("Could not resolve absolute path for '" +
                             display_path(path) +
                             "': " + absolute_error.message());
  }

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

  auto temporary_file = reserve_temporary_file(destination);
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
