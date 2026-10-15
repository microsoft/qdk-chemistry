// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <filesystem>

namespace qdk::chemistry::utils {

// Owns a private extraction directory until its contents have been consumed.
class ArchiveDirectory {
 public:
  ArchiveDirectory();
  ~ArchiveDirectory();
  ArchiveDirectory(const ArchiveDirectory&) = delete;
  ArchiveDirectory& operator=(const ArchiveDirectory&) = delete;

  void extract(const std::filesystem::path& archive) const;
  std::filesystem::path file(const std::filesystem::path& relative) const;

 private:
  std::filesystem::path _path;
};

}  // namespace qdk::chemistry::utils
