# fix-ga-linalg-preference.cmake — run as a GlobalArrays_External PATCH_COMMAND (see
# cmsb-fix-dependency-reuse.patch fix #6) to fix packaging issues in GlobalArrays' own build that resurface the
# same "LAPACK resolves to ReferenceLAPACK, not FLAME" problem downstream:
#
# 1. cmake/ga-linalg.cmake unconditionally overwrites LAPACK_PREFERENCE_LIST -- TWICE -- for
#    LINALG_VENDOR=BLIS/OpenBLAS/IBMESSL: first to `${LINALG_VENDOR}` (e.g. "BLIS"), then to
#    "ReferenceLAPACK", clobbering whatever a caller passes in (e.g. install-exachem.sh's
#    "FLAME;ReferenceLAPACK" for the BLIS+libFLAME stack) either way. Snapshots the caller-supplied value
#    (if any) before the first overwrite, and uses that snapshot to guard both assignments -- preserving the
#    caller's value end-to-end when given, and reproducing the exact original default chain (${LINALG_VENDOR}
#    then "ReferenceLAPACK") when it isn't.
#
# 2 & 3. cmake/linalg-modules/LinAlgModulesMacros.cmake's install_linalg_modules() installs a fixed subset of
#    Find<Vendor>.cmake modules alongside GlobalArrays' exported globalarrays-config.cmake, for downstream
#    consumers (e.g. TAMM/ExaChem re-importing GlobalArrays::ga) to re-resolve BLAS/LAPACK without GA's own
#    build-time source tree. That subset omits FindFLAME.cmake even though GA's own build supports FLAME as a
#    LAPACK vendor -- so a downstream find_package(FLAME) against GA's installed config always fails to find
#    even the finder module, before ever falling through to ReferenceLAPACK. Fix 2 adds FindFLAME.cmake to that
#    installed list. Fix 3 additionally mirrors GA's own CMakeLists.txt workaround for two other files
#    (FindILP64.cmake/FindStandardFortran.cmake, installed a second time via an explicit install(FILES ...),
#    with a "#FIXME: Not sure why this file is not installed via ExternalProject build of GA" comment) -- since
#    that comment indicates install_linalg_modules()'s own install() calls are not reliably reached under an
#    ExternalProject-style build (exactly how CMSB builds GA), FindFLAME.cmake gets the same explicit,
#    redundant install() rather than relying on fix 2 alone.
#
# Usage: cmake -DGA_LINALG_FILE=<path-to-ga>/cmake/ga-linalg.cmake -P fix-ga-linalg-preference.cmake
# (GA_LINALG_FILE's siblings cmake/linalg-modules/LinAlgModulesMacros.cmake and ../CMakeLists.txt are derived
# automatically.)

if(NOT DEFINED GA_LINALG_FILE)
  message(FATAL_ERROR "GA_LINALG_FILE must be set to the path of GlobalArrays' cmake/ga-linalg.cmake")
endif()

file(READ "${GA_LINALG_FILE}" _contents)

set(_snapshot_line "set(BLAS_PREFERENCE_LIST      \${LINALG_VENDOR})")
set(_first_old "set(LAPACK_PREFERENCE_LIST    \${LINALG_VENDOR})")
set(_second_old "set(LAPACK_PREFERENCE_LIST ReferenceLAPACK)")
foreach(_needle _snapshot_line _first_old _second_old)
  string(FIND "${_contents}" "${${_needle}}" _idx)
  if(_idx EQUAL -1)
    message(FATAL_ERROR "fix-ga-linalg-preference.cmake: expected literal text (${_needle}) not found in "
                         "${GA_LINALG_FILE} -- GlobalArrays' ga-linalg.cmake may have changed upstream; update "
                         "this patch script.")
  endif()
endforeach()

# Snapshot the caller-supplied value right after the first thing ga-linalg.cmake itself sets in this block, so
# both overwrites below can tell "caller passed a value" apart from "this script's own first default fired".
string(REPLACE "${_snapshot_line}" "${_snapshot_line}\n    set(_qdk_caller_lapack_preference_list \"\${LAPACK_PREFERENCE_LIST}\")" _contents "${_contents}")

string(REPLACE "${_first_old}"
  "if(_qdk_caller_lapack_preference_list)\n      set(LAPACK_PREFERENCE_LIST \"\${_qdk_caller_lapack_preference_list}\")\n    else()\n      set(LAPACK_PREFERENCE_LIST \${LINALG_VENDOR})\n    endif()"
  _contents "${_contents}")

string(REPLACE "${_second_old}"
  "if(NOT _qdk_caller_lapack_preference_list)\n        set(LAPACK_PREFERENCE_LIST ReferenceLAPACK)\n      endif()"
  _contents "${_contents}")

file(WRITE "${GA_LINALG_FILE}" "${_contents}")

message(STATUS "Patched ${GA_LINALG_FILE}: LAPACK_PREFERENCE_LIST default is now overridable")

# --- Fix 2: install_linalg_modules() never installs FindFLAME.cmake (see file header) ---

get_filename_component(_ga_cmake_dir "${GA_LINALG_FILE}" DIRECTORY)
set(_ga_macros_file "${_ga_cmake_dir}/linalg-modules/LinAlgModulesMacros.cmake")

if(NOT EXISTS "${_ga_macros_file}")
  message(FATAL_ERROR "fix-ga-linalg-preference.cmake: expected file not found: ${_ga_macros_file} -- "
                       "GlobalArrays' layout may have changed upstream; update this patch script.")
endif()

file(READ "${_ga_macros_file}" _macros_contents)

set(_modules_old "     FindBLIS.cmake\n     FindIBMESSL.cmake")
string(FIND "${_macros_contents}" "${_modules_old}" _idx)
if(_idx EQUAL -1)
  message(FATAL_ERROR "fix-ga-linalg-preference.cmake: expected literal text (LINALG_FIND_MODULES entries) not "
                       "found in ${_ga_macros_file} -- GlobalArrays' install_linalg_modules() may have changed "
                       "upstream; update this patch script.")
endif()

string(REPLACE "${_modules_old}" "     FindBLIS.cmake\n     FindFLAME.cmake\n     FindIBMESSL.cmake" _macros_contents "${_macros_contents}")

file(WRITE "${_ga_macros_file}" "${_macros_contents}")

message(STATUS "Patched ${_ga_macros_file}: install_linalg_modules() now also installs FindFLAME.cmake")

# --- Fix 3: belt-and-suspenders for fix 2 ---
#
# GA's own top-level CMakeLists.txt already carries a redundant, manual install(FILES ...) for
# FindILP64.cmake/FindStandardFortran.cmake, right after the install_linalg_modules() call, with a "#FIXME: Not
# sure why this file is not installed via ExternalProject build of GA" comment -- i.e. GA's own maintainers hit
# install_linalg_modules()'s install() calls not reliably running under an ExternalProject-style build (exactly
# how CMSB builds GA) and worked around it for those two files specifically. Since fix 2 above adds FindFLAME.cmake
# to the very same install_linalg_modules() mechanism that comment distrusts, mirror that same workaround here so
# we don't depend on a path GA's own authors flagged as unreliable in this exact scenario.

get_filename_component(_ga_source_dir "${_ga_cmake_dir}/.." ABSOLUTE)
set(_ga_cmakelists_file "${_ga_source_dir}/CMakeLists.txt")

if(NOT EXISTS "${_ga_cmakelists_file}")
  message(FATAL_ERROR "fix-ga-linalg-preference.cmake: expected file not found: ${_ga_cmakelists_file} -- "
                       "GlobalArrays' layout may have changed upstream; update this patch script.")
endif()

file(READ "${_ga_cmakelists_file}" _cmakelists_contents)

set(_extra_install_old "    \${CMAKE_CURRENT_LIST_DIR}/cmake/linalg-modules/FindILP64.cmake\n    \${CMAKE_CURRENT_LIST_DIR}/cmake/linalg-modules/FindStandardFortran.cmake")
string(FIND "${_cmakelists_contents}" "${_extra_install_old}" _idx)
if(_idx EQUAL -1)
  message(FATAL_ERROR "fix-ga-linalg-preference.cmake: expected literal text (extra install(FILES ...) block) "
                       "not found in ${_ga_cmakelists_file} -- GlobalArrays' CMakeLists.txt may have changed "
                       "upstream; update this patch script.")
endif()

string(REPLACE "${_extra_install_old}"
  "${_extra_install_old}\n    \${CMAKE_CURRENT_LIST_DIR}/cmake/linalg-modules/FindFLAME.cmake"
  _cmakelists_contents "${_cmakelists_contents}")

file(WRITE "${_ga_cmakelists_file}" "${_cmakelists_contents}")

message(STATUS "Patched ${_ga_cmakelists_file}: FindFLAME.cmake now also explicitly installed")
