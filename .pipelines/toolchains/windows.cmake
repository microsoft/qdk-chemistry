# Example toolchain: Windows (cl.exe and clang-cl) flag setup for QDK/Chemistry.
#
# clang-cl targets the MSVC ABI and accepts MSVC-style flags, so both native
# cl.exe and clang-cl use the same per-config flag setup.
#
# Chainload after the vcpkg toolchain via VCPKG_CHAINLOAD_TOOLCHAIN_FILE:
#   -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
#   -DVCPKG_CHAINLOAD_TOOLCHAIN_FILE=<repo>/.pipelines/toolchains/windows.cmake

foreach(_lang C CXX)
  set(CMAKE_${_lang}_FLAGS_DEBUG          "/Zi /Od /RTC1"    CACHE STRING "" FORCE)
  set(CMAKE_${_lang}_FLAGS_RELWITHDEBINFO "/Zi /O2 /DNDEBUG" CACHE STRING "" FORCE)
  set(CMAKE_${_lang}_FLAGS_RELEASE        "/O2 /DNDEBUG"     CACHE STRING "" FORCE)
endforeach()
