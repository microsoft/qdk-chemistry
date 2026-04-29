// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

/*
On Windows, spdlog's wincolor_sink and stdout_sink both cache a Windows HANDLE
(via GetStdHandle / _get_osfhandle) at construction time. When pytest's capfd
later redirects fd 1 via dup2(), the cached HANDLE still points to the original
stdout — so all Logger output bypasses capture.

See spdlog 1.17.0 sources:
  https://github.com/gabime/spdlog/blob/v1.17.0/include/spdlog/sinks/wincolor_sink-inl.h#L164-L165
    GetStdHandle(STD_OUTPUT_HANDLE) called once in constructor, stored as
    out_handle_
  https://github.com/gabime/spdlog/blob/v1.17.0/include/spdlog/sinks/wincolor_sink-inl.h#L157
    WriteFile(out_handle_, ...) on every write
  https://github.com/gabime/spdlog/blob/v1.17.0/include/spdlog/sinks/wincolor_sink.h#L42
    void *out_handle_ member (never refreshed)

On Linux this problem does not occur because stdout_color_sink_mt is aliased to
ansicolor_stdout_sink, which writes via fwrite(stdout):
  https://github.com/gabime/spdlog/blob/v1.17.0/include/spdlog/sinks/ansicolor_sink-inl.h#L123
  https://github.com/gabime/spdlog/blob/v1.17.0/include/spdlog/sinks/stdout_color_sinks.h#L16-L26

This sink writes via fwrite(stdout), which goes through the C runtime's fd layer
(fd 1) and respects dup2() redirections.

Note: unlike ansicolor_sink, this sink does not emit ANSI color codes, so log
output on Windows is uncolored.  Colored output could be added by wrapping the
formatted message's color range (msg.color_range_start / color_range_end) in
ANSI escape sequences, as ansicolor_sink does.
See: https://github.com/gabime/spdlog/issues/3138
*/

#ifdef _WIN32

#include <spdlog/sinks/base_sink.h>

#include <cstdio>
#include <mutex>

namespace qdk::chemistry::utils {

class stdout_fd_sink final : public spdlog::sinks::base_sink<std::mutex> {
 protected:
  void sink_it_(const spdlog::details::log_msg& msg) override {
    spdlog::memory_buf_t formatted;
    formatter_->format(msg, formatted);
    std::fwrite(formatted.data(), 1, formatted.size(), stdout);
    std::fflush(stdout);
  }
  void flush_() override { std::fflush(stdout); }
};

}  // namespace qdk::chemistry::utils

#endif  // _WIN32
