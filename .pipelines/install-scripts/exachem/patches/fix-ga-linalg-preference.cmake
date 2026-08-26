# fix-ga-linalg-preference.cmake — run as a GlobalArrays_External PATCH_COMMAND (see
# cmsb-fix-dependency-reuse.patch fix #6) to fix two independent GlobalArrays packaging issues that both
# resurface the same "LAPACK resolves to ReferenceLAPACK, not FLAME" problem downstream:
#
# 1. cmake/ga-linalg.cmake unconditionally overwrites LAPACK_PREFERENCE_LIST -- TWICE -- for
#    LINALG_VENDOR=BLIS/OpenBLAS/IBMESSL: first to `${LINALG_VENDOR}` (e.g. "BLIS"), then to
#    "ReferenceLAPACK", clobbering whatever a caller passes in (e.g. install-exachem.sh's
#    "FLAME;ReferenceLAPACK" for the BLIS+libFLAME stack) either way. Snapshots the caller-supplied value
#    (if any) before the first overwrite, and uses that snapshot to guard both assignments -- preserving the
#    caller's value end-to-end when given, and reproducing the exact original default chain (${LINALG_VENDOR}
#    then "ReferenceLAPACK") when it isn't.
#
# 2. cmake/linalg-modules/LinAlgModulesMacros.cmake's install_linalg_modules() installs a fixed subset of
#    Find<Vendor>.cmake modules alongside GlobalArrays' exported globalarrays-config.cmake, for downstream
#    consumers (e.g. TAMM/ExaChem re-importing GlobalArrays::ga) to re-resolve BLAS/LAPACK without GA's own
#    build-time source tree. That subset omits FindFLAME.cmake even though GA's own build supports FLAME as a
#    LAPACK vendor -- so a downstream find_package(FLAME) against GA's installed config always fails to find
#    even the finder module, before ever falling through to ReferenceLAPACK. Adds FindFLAME.cmake to the
#    installed list so it's available wherever the other vendor finders are.
#
# Usage: cmake -DGA_LINALG_FILE=<path-to-ga>/cmake/ga-linalg.cmake -P fix-ga-linalg-preference.cmake
# (GA_LINALG_FILE's sibling cmake/linalg-modules/LinAlgModulesMacros.cmake is derived automatically.)

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
