# Example toolchain: Windows (cl.exe and clang-cl) flag setup for QDK/Chemistry.
#
# clang-cl targets the MSVC ABI and accepts MSVC-style flags, so both native
# cl.exe and clang-cl use the same per-config flag setup.
#
# Chainload after the vcpkg toolchain via VCPKG_CHAINLOAD_TOOLCHAIN_FILE:
#   -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
#   -DVCPKG_CHAINLOAD_TOOLCHAIN_FILE=<repo>/.pipelines/toolchains/windows.cmake

# MS 1CS warning set. NOTE: kept here for now; these project warnings will be
# moved to per-target options so they don't propagate to dependencies.
set(_qdk_msvc_warnings
    "/W3 /w14018 /w14055 /w14100 /w14102 /w14127 /w14146 /w14242 /w14244 /w14245"
    "/w14254 /w14267 /w14302 /w14306 /w14308 /w14310 /w14389 /w14509 /w14510"
    "/w14512 /w14532 /w14533 /w14610 /w14611 /w14700 /w14701 /w14703 /w14789"
    "/w14995 /w14996")
string(JOIN " " _qdk_msvc_warnings ${_qdk_msvc_warnings})

foreach(_lang C CXX)
  set(CMAKE_${_lang}_FLAGS_DEBUG          "/Zi /Od /RTC1 ${_qdk_msvc_warnings}"   CACHE STRING "" FORCE)
  set(CMAKE_${_lang}_FLAGS_RELWITHDEBINFO "/Zi /O2 /DNDEBUG ${_qdk_msvc_warnings}" CACHE STRING "" FORCE)
  set(CMAKE_${_lang}_FLAGS_RELEASE        "/O2 /DNDEBUG ${_qdk_msvc_warnings}"    CACHE STRING "" FORCE)
endforeach()

unset(_qdk_msvc_warnings)
