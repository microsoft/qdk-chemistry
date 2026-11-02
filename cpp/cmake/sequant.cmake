# SeQuant with its BTAS numerical backend, shared with the dependency installers (sequant/CMakeLists.txt).
block()
    # Makes SeQuant and BTAS re-find this Boost instead of downloading another.
    find_package(Boost CONFIG REQUIRED)
    set(Boost_FETCH_IF_MISSING OFF)

    # These first declarations win over SeQuant's, which exclude them from install.
    include(FetchContent)
    FetchContent_Declare(RangeV3
        GIT_REPOSITORY https://github.com/ericniebler/range-v3.git
        GIT_TAG        0.12.0
        SYSTEM
        FIND_PACKAGE_ARGS NAMES range-v3
    )
    FetchContent_Declare(BTAS
        GIT_REPOSITORY https://github.com/BTAS/btas.git
        GIT_TAG        5b435cfdd3cb423f5a40229572c38f6700ba0456
        SYSTEM
    )
    # kit-cmake's BLAS++/LAPACK++ module otherwise writes BUILD_TESTING into the cache.
    FetchContent_Declare(vg_cmake_kit
        GIT_REPOSITORY https://github.com/ValeevGroup/kit-cmake.git
        GIT_TAG        e820ffd200aebf6b97f2cf223d7b14134755d82e
        PATCH_COMMAND "${CMAKE_COMMAND}"
            "-DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/patches/kit-cmake-build-testing.patch"
            -P "${CMAKE_CURRENT_LIST_DIR}/patches/apply-patch.cmake"
    )
    # SeQuant would otherwise take an installed BTAS over the pinned one; a preinstalled SeQuant needs its own.
    if(QDK_ALLOW_DEPENDENCY_FETCH)
        set(CMAKE_DISABLE_FIND_PACKAGE_BTAS ON)
    endif()

    # Concurrent evaluation otherwise races on SeQuant's trace-label map, even with tracing off.
    set(_sequant_patch PATCH_COMMAND "${CMAKE_COMMAND}"
        "-DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/patches/sequant-eval-trace-labels.patch"
        -P "${CMAKE_CURRENT_LIST_DIR}/patches/apply-patch.cmake")
    # MSVC 19.44 (VS 2022 17.14) otherwise rejects SeQuant's evaluation byte counter.
    list(APPEND _sequant_patch COMMAND "${CMAKE_COMMAND}"
        "-DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/patches/sequant-msvc-eval-bytes.patch"
        -P "${CMAKE_CURRENT_LIST_DIR}/patches/apply-patch.cmake")
    # clang-cl ignores [[no_unique_address]] like MSVC, which fails a SeQuant size check.
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
        list(APPEND _sequant_patch COMMAND "${CMAKE_COMMAND}" -P
            "${CMAKE_CURRENT_LIST_DIR}/patches/sequant-clang-cl-no-unique-address.cmake")
    endif()
    set(SEQUANT_BTAS ON)
    # SeQuant otherwise leaves a fetched backend out of its install and exported targets.
    set(SEQUANT_INSTALL_FETCHED_BACKENDS BTAS)
    # libPerm would otherwise fetch googletest and build its tests as part of `all`.
    set(LIBPERM_TESTS OFF)
    # SeQuant and its dependencies would otherwise add their tests to ctest.
    set(BUILD_TESTING OFF)

    # SeQuant, BTAS, and libPerm seed these global defaults (e.g. strict ISO C++,
    # ccache) for standalone builds; drop any they create to keep them local.
    set(_qdk_seeded "")
    foreach(_var IN ITEMS CMAKE_BUILD_TYPE CMAKE_CXX_STANDARD CMAKE_CXX_EXTENSIONS
                          CMAKE_FIND_NO_INSTALL_PREFIX CMAKE_C_COMPILER_LAUNCHER CMAKE_CXX_COMPILER_LAUNCHER)
        if(NOT DEFINED CACHE{${_var}})
            list(APPEND _qdk_seeded ${_var})
        endif()
    endforeach()

    # Expose Catch2's CMake modules to SeQuant's tests if MACIS already fetched Catch2.
    FetchContent_GetProperties(Catch2)
    if(catch2_POPULATED)
        list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
    endif()

    # Pinned to an unreleased, patched commit, so an installed SeQuant is used only without fetching.
    handle_dependency(SeQuant
        GIT_REPOSITORY https://github.com/ValeevGroup/SeQuant.git
        GIT_TAG        be931075ebb4d15d252045ab4ba4b3065845355d
        BUILD_TARGET   SeQuant::SeQuant
        INSTALL_TARGET SeQuant::SeQuant
        ${DEPENDENCY_BUILD_FLAGS}
        FETCHCONTENT_ARGS ${_sequant_patch}
        FORCE_FETCH
        REQUIRED
    )
    foreach(_var IN LISTS _qdk_seeded)
        unset(${_var} CACHE)
    endforeach()

    # SeQuant's unit-test objects are otherwise part of `all`.
    FetchContent_GetProperties(SeQuant)
    if(sequant_POPULATED)
        set_property(DIRECTORY "${sequant_SOURCE_DIR}/tests" PROPERTY EXCLUDE_FROM_ALL ON)
    endif()
endblock()
