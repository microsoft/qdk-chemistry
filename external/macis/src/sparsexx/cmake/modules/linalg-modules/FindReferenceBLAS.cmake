# MACIS Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
#
# See LICENSE.txt for details

if( referenceblas_PREFERS_STATIC )
  set( referenceblas_LIBRARY_NAME "libblas.a" )
else()
  set( referenceblas_LIBRARY_NAME "blas" )
endif()

find_library( referenceblas_LIBRARY
  NAMES ${referenceblas_LIBRARY_NAME}
  HINTS ${referenceblas_PREFIX}
  PATHS ${referenceblas_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
  PATH_SUFFIXES lib lib64 lib32
  DOC "ReferenceBLAS Library"
)

#if( referenceblas_INCLUDE_DIR )
#  set( ReferenceBLAS_INCLUDE_DIR ${referenceblas_INCLUDE_DIR} )
#endif()

if( referenceblas_LIBRARY )
  set( ReferenceBLAS_LIBRARIES ${referenceblas_LIBRARY} )
endif()

# Reference BLAS is always LP64
set( ReferenceBLAS_ilp64_FOUND FALSE )
set( ReferenceBLAS_lp64_FOUND  TRUE  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args( ReferenceBLAS
#  REQUIRED_VARS ReferenceBLAS_LIBRARIES ReferenceBLAS_INCLUDE_DIR
  REQUIRED_VARS ReferenceBLAS_LIBRARIES
#  VERSION_VAR ReferenceBLAS_VERSION_STRING
  HANDLE_COMPONENTS
)

if( ReferenceBLAS_FOUND AND NOT TARGET ReferenceBLAS::blas )

  add_library( ReferenceBLAS::blas INTERFACE IMPORTED )
  set_target_properties( ReferenceBLAS::blas PROPERTIES
#    INTERFACE_INCLUDE_DIRECTORIES "${ReferenceBLAS_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES      "${ReferenceBLAS_LIBRARIES}"
  )

endif()
