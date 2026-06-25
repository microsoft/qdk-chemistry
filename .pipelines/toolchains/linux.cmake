# Example toolchain: GCC/Clang flag setup for QDK/Chemistry on Linux.
#
# Usage:
#   cmake -S cpp -B build -DCMAKE_TOOLCHAIN_FILE=<repo>/.pipelines/toolchains/linux.cmake

foreach(_lang C CXX)
  set(CMAKE_${_lang}_FLAGS_DEBUG          "-g -O0 -Wall -Wextra" CACHE STRING "" FORCE)
  set(CMAKE_${_lang}_FLAGS_RELWITHDEBINFO "-g -O3 -DNDEBUG"      CACHE STRING "" FORCE)
  set(CMAKE_${_lang}_FLAGS_RELEASE        "-O3 -DNDEBUG"         CACHE STRING "" FORCE)
endforeach()
