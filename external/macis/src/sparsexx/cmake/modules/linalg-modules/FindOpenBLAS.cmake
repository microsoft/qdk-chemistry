# MACIS Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
#
# See LICENSE.txt for details

if( openblas_PREFERS_STATIC )
  set( openblas_LIBRARY_NAME "libopenblas.a" )
else()
  set( openblas_LIBRARY_NAME "openblas" )
endif()

find_library( openblas_LIBRARY
  NAMES ${openblas_LIBRARY_NAME}
  HINTS ${openblas_PREFIX}
  PATHS ${openblas_LIBRARY_DIR} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
  PATH_SUFFIXES lib lib64 lib32
  DOC "OpenBLAS Library"
)

#if( openblas_INCLUDE_DIR )
#  set( OpenBLAS_INCLUDE_DIR ${openblas_INCLUDE_DIR} )
#endif()

if( openblas_LIBRARY )
  find_package( OpenMP QUIET )
  if( NOT gfortran_LIBRARY )
    find_library( gfortran_LIBRARY
      NAMES gfortran
      PATHS ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
      DOC "GFortran Library"
    )
  endif()
  set( OpenBLAS_LIBRARIES ${openblas_LIBRARY} OpenMP::OpenMP_C "m" ${gfortran_LIBRARY})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args( OpenBLAS
#  REQUIRED_VARS OpenBLAS_LIBRARIES OpenBLAS_INCLUDE_DIR
  REQUIRED_VARS OpenBLAS_LIBRARIES
#  VERSION_VAR OpenBLAS_VERSION_STRING
  HANDLE_COMPONENTS
)

if( OpenBLAS_FOUND AND NOT TARGET OpenBLAS::openblas )

  add_library( OpenBLAS::openblas INTERFACE IMPORTED )
  set_target_properties( OpenBLAS::openblas PROPERTIES
#    INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES      "${OpenBLAS_LIBRARIES}"
  )

endif()
