// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <string_view>

namespace qdk::chemistry::utils {

using AtomicFileWriter =
    std::function<void(const std::filesystem::path& temporary_path)>;

/**
 * @brief Create the parent directory of a path when it does not exist.
 *
 * A path without a parent component refers to the current directory and needs
 * no action.
 */
void ensure_parent_directory(const std::filesystem::path& path);

/**
 * @brief Read an entire file as binary-safe text.
 */
std::string read_text_file(const std::filesystem::path& path);

/**
 * @brief Write a file through a temporary sibling and atomically replace the
 * destination.
 *
 * The writer receives the path of an existing empty temporary file in the
 * destination directory. It must open and write that file in place, close all
 * writes before returning, and must not unlink, rename, replace, or hard-link
 * the file. Cleanup is guaranteed only while the reserved file remains at the
 * temporary path. Keeping the temporary file beside the destination allows
 * replacement to remain atomic. The path preserves the destination's suffixes
 * for format-sensitive writers.
 *
 * On POSIX, replacing an existing file preserves its ordinary read, write, and
 * execute permission bits. Existing destination ACLs and extended attributes
 * are not preserved; the replacement uses metadata inherited when its
 * temporary file is created and may therefore grant broader access than the
 * file it replaced. Callers that rely on explicit ACLs or extended attributes
 * must reapply them after the write. New files are created with owner-only
 * permissions. The filesystem must enforce POSIX permission bits; the write
 * fails rather than publishing a file with broader mode bits. Platform ACLs
 * are not inspected and may grant access beyond those bits.
 * On Windows, replacement preserves the read-only attribute and new files use
 * the filesystem's standard access controls. Existing Windows security
 * descriptors and DACLs are not preserved; the replacement uses access
 * controls inherited when its temporary file is created and may therefore
 * grant broader access than the file it replaced. Callers that rely on
 * explicit access-control entries must reapply them after the write. Read-only
 * Windows destinations with multiple hard links are rejected. Other
 * file-object metadata and hard-link identity are not preserved. The named
 * temporary file also inherits the parent directory's access controls and may
 * therefore be readable while the writer runs or after cleanup fails. Atomic
 * replacement prevents partial visibility at the destination path but does not
 * guarantee durability after power loss.
 * Windows alternate data streams are not supported.
 *
 * The destination's parent directory and mutable ancestors must not be
 * writable by principals less privileged than the process performing the
 * write. Missing POSIX parent directories are created with owner-only
 * permissions. Windows parent directories use inherited filesystem access
 * controls.
 *
 * On POSIX, relative destinations are frozen to an absolute path before the
 * writer runs. A relative path may therefore be rejected when its expanded
 * absolute form exceeds the platform pathname limit.
 *
 * @param path Destination path.
 * @param writer Function that writes the complete temporary file.
 * @param create_parent_directories Create missing parent directories when true.
 */
void write_file_atomically(const std::filesystem::path& path,
                           const AtomicFileWriter& writer,
                           bool create_parent_directories = false);

/**
 * @brief Write binary-safe text through an atomic file replacement.
 */
void write_text_file_atomically(const std::filesystem::path& path,
                                std::string_view contents,
                                bool create_parent_directories = false);

}  // namespace qdk::chemistry::utils
