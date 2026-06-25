# Example toolchain: GCC/Clang flag setup for QDK/Chemistry on Linux.
#
# Usage:
#   cmake -S cpp -B build -DCMAKE_TOOLCHAIN_FILE=<repo>/.pipelines/toolchains/linux.cmake

foreach(_lang C CXX)
  set(CMAKE_${_lang}_FLAGS_DEBUG_INIT          "-g -O0 -Wall -Wextra")
  set(CMAKE_${_lang}_FLAGS_RELWITHDEBINFO_INIT "-g -O3 -DNDEBUG")
  set(CMAKE_${_lang}_FLAGS_RELEASE_INIT        "-O3 -DNDEBUG")
endforeach()
