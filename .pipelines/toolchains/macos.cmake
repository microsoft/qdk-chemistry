# Example toolchain: AppleClang flag setup for QDK/Chemistry on macOS.
#
# Usage:
#   cmake -S cpp -B build -DCMAKE_TOOLCHAIN_FILE=<repo>/.pipelines/toolchains/macos.cmake
#
# AppleClang uses the same GCC/Clang-style flags as the Linux toolchain.

include("${CMAKE_CURRENT_LIST_DIR}/linux.cmake")
