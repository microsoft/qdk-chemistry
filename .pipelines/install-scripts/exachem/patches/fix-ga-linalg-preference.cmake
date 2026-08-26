# fix-ga-linalg-preference.cmake — run as a GlobalArrays_External PATCH_COMMAND (see
# cmsb-fix-dependency-reuse.patch fix #6) to fix GlobalArrays' own cmake/ga-linalg.cmake, which unconditionally
# overwrites LAPACK_PREFERENCE_LIST -- TWICE -- for LINALG_VENDOR=BLIS/OpenBLAS/IBMESSL: first to
# `${LINALG_VENDOR}` (e.g. "BLIS"), then to "ReferenceLAPACK", clobbering whatever a caller passes in (e.g.
# install-exachem.sh's "FLAME;ReferenceLAPACK" for the BLIS+libFLAME stack) either way. Snapshots the
# caller-supplied value (if any) before the first overwrite, and uses that snapshot to guard both assignments --
# preserving the caller's value end-to-end when given, and reproducing the exact original default chain
# (${LINALG_VENDOR} then "ReferenceLAPACK") when it isn't.
#
# Usage: cmake -DGA_LINALG_FILE=<path-to-ga>/cmake/ga-linalg.cmake -P fix-ga-linalg-preference.cmake

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
