# MACIS Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
#
# See LICENSE.txt for details

if( referencelapack_PREFERS_STATIC )
  set( referencelapack_LIBRARY_NAME "liblapack.a" )
else()
  set( referencelapack_LIBRARY_NAME "lapack" )
endif()

find_library( referencelapack_LIBRARY
  NAMES ${referencelapack_LIBRARY_NAME}
  HINTS ${referencelapack_PREFIX}
  PATHS ${referencelapack_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
  PATH_SUFFIXES lib lib64 lib32
  DOC "ReferenceLAPACK Library"
)

find_library( gfortran_LIBRARY
  NAMES gfortran
  PATHS ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
  DOC "GFortran Library"
)

#if( referencelapack_INCLUDE_DIR )
#  set( ReferenceLAPACK_INCLUDE_DIR ${referencelapack_INCLUDE_DIR} )
#endif()

if( referencelapack_LIBRARY AND gfortran_LIBRARY )
  set( ReferenceLAPACK_LIBRARIES ${referencelapack_LIBRARY} ${gfortran_LIBRARY} )
endif()


# Reference LAPACK is always LP64
set( ReferenceLAPACK_ilp64_FOUND FALSE )
set( ReferenceLAPACK_lp64_FOUND  TRUE  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args( ReferenceLAPACK
#  REQUIRED_VARS ReferenceLAPACK_LIBRARIES ReferenceLAPACK_INCLUDE_DIR
  REQUIRED_VARS ReferenceLAPACK_LIBRARIES
#  VERSION_VAR ReferenceLAPACK_VERSION_STRING
  HANDLE_COMPONENTS
)

if( ReferenceLAPACK_FOUND AND NOT TARGET ReferenceLAPACK::lapack )

  add_library( ReferenceLAPACK::lapack INTERFACE IMPORTED )
  set_target_properties( ReferenceLAPACK::lapack PROPERTIES
#    INTERFACE_INCLUDE_DIRECTORIES "${ReferenceLAPACK_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES      "${ReferenceLAPACK_LIBRARIES}"
  )

endif()
