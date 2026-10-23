find_path(cuTENSOR_INCLUDE_DIR cutensor.h
    HINTS "${CUTENSOR_ROOT}" ENV CUTENSOR_ROOT
    PATH_SUFFIXES include)
find_library(cuTENSOR_LIBRARY NAMES cutensor
    HINTS "${CUTENSOR_ROOT}" ENV CUTENSOR_ROOT
    PATH_SUFFIXES "lib/${CUDAToolkit_VERSION_MAJOR}" lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuTENSOR
    REQUIRED_VARS cuTENSOR_LIBRARY cuTENSOR_INCLUDE_DIR)

if(cuTENSOR_FOUND AND NOT TARGET cutensor)
    add_library(cutensor SHARED IMPORTED)
    set_target_properties(cutensor PROPERTIES
        IMPORTED_LOCATION "${cuTENSOR_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${cuTENSOR_INCLUDE_DIR}")
endif()
mark_as_advanced(cuTENSOR_INCLUDE_DIR cuTENSOR_LIBRARY)
