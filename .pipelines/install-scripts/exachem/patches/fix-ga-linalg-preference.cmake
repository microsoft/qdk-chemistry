# fix-ga-linalg-preference.cmake — run as a GlobalArrays_External PATCH_COMMAND (see
# cmsb-fix-dependency-reuse.patch fix #6) to fix GlobalArrays' own cmake/ga-linalg.cmake, which unconditionally
# overwrites LAPACK_PREFERENCE_LIST to "ReferenceLAPACK" for LINALG_VENDOR=BLIS/OpenBLAS/IBMESSL, clobbering
# whatever a caller passes in (e.g. install-exachem.sh's "FLAME;ReferenceLAPACK" for the BLIS+libFLAME stack).
# Makes that assignment a default instead: only applies if LAPACK_PREFERENCE_LIST isn't already set.
#
# Usage: cmake -DGA_LINALG_FILE=<path-to-ga>/cmake/ga-linalg.cmake -P fix-ga-linalg-preference.cmake

if(NOT DEFINED GA_LINALG_FILE)
  message(FATAL_ERROR "GA_LINALG_FILE must be set to the path of GlobalArrays' cmake/ga-linalg.cmake")
endif()

file(READ "${GA_LINALG_FILE}" _contents)

set(_old "set(LAPACK_PREFERENCE_LIST ReferenceLAPACK)")
string(FIND "${_contents}" "${_old}" _idx)
if(_idx EQUAL -1)
  message(FATAL_ERROR "fix-ga-linalg-preference.cmake: expected literal text not found in ${GA_LINALG_FILE} -- "
                       "GlobalArrays' ga-linalg.cmake may have changed upstream; update this patch script.")
endif()

set(_new "if(NOT LAPACK_PREFERENCE_LIST)\n        set(LAPACK_PREFERENCE_LIST ReferenceLAPACK)\n      endif()")
string(REPLACE "${_old}" "${_new}" _contents "${_contents}")
file(WRITE "${GA_LINALG_FILE}" "${_contents}")

message(STATUS "Patched ${GA_LINALG_FILE}: LAPACK_PREFERENCE_LIST default is now overridable")
