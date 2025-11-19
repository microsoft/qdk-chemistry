function(handle_dependency NAME)
  # This function handles the discovery and fetching of dependencies.
  # It supports two modes of fetching dependencies:
  # 1. Git repository mode: Using GIT_REPOSITORY and GIT_TAG
  # 2. URL/Tarball mode: Using URL and related parameters
  #
  # Parameters:
  # NAME - The name of the dependency
  #
  # Options:
  # REQUIRED - Indicates that the dependency is required
  #
  # Single value arguments:
  # GIT_REPOSITORY - URL of the git repository (for git mode)
  # GIT_TAG - Git tag/branch/commit to checkout (for git mode)
  # URL - URL to download the dependency from (for URL/tarball mode)
  # URL_HASH - Hash of the file at URL (for URL/tarball mode)
  # SOURCE_DIR - Directory to place the downloaded source (for URL/tarball mode)
  # DOWNLOAD_EXTRACT_TIMESTAMP - Control timestamp extraction (for URL/tarball mode)
  # BUILD_TARGET - The CMake target to build
  # INSTALL_TARGET - The CMake target to install
  #
  # Multi-value arguments:
  # EXPORTED_VARIABLES - Variables to export
  # FIND_PACKAGE_ARGS - Arguments to pass to find_package

  set(options
    REQUIRED
  ) # Flags with no values
  set(oneValueArgs
    GIT_REPOSITORY
    GIT_TAG
    URL
    URL_HASH
    SOURCE_DIR
    DOWNLOAD_EXTRACT_TIMESTAMP
    SOURCE_SUBDIR
    BUILD_TARGET
    INSTALL_TARGET
    BUILD_ARGS
  )
  set(multiValueArgs
    EXPORTED_VARIABLES
    FIND_PACKAGE_ARGS
    FETCHCONTENT_ARGS
  )

  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  message(STATUS "Handling Dependency: ${NAME}")



  message(STATUS "  Attempting to Discover ${NAME} -")
  find_package(${NAME} ${ARG_FIND_PACKAGE_ARGS} QUIET)
  if(${NAME}_FOUND)
    message(STATUS "  Attempting to Discover ${NAME} - Found ${NAME}: ${${NAME}_DIR}")
  else()
    message(STATUS "  Attempting to Discover ${NAME} - WARNING: ${NAME} not found")
  endif()

  if(NOT ${NAME}_FOUND)
    include(FetchContent)

    # Check if FETCHCONTENT_SOURCE_DIR is set for this dependency
    string(TOUPPER ${NAME} NAME_UPPER)
    if(DEFINED FETCHCONTENT_SOURCE_DIR_${NAME_UPPER})
      message(STATUS "  Using local source for ${NAME} from: ${FETCHCONTENT_SOURCE_DIR_${NAME_UPPER}}")
    endif()

    if(QDK_ALLOW_DEPENDENCY_FETCH)
      if(DEFINED ARG_GIT_REPOSITORY)
        message(STATUS "  Cloning ${NAME} from Git repository")
        message(STATUS "    -- GIT_REPOSITORY: ${ARG_GIT_REPOSITORY}")
        message(STATUS "    -- GIT_TAG: ${ARG_GIT_TAG}")
        # Git repository mode
        FetchContent_Declare(
          ${NAME}
          GIT_REPOSITORY ${ARG_GIT_REPOSITORY}
          GIT_TAG ${ARG_GIT_TAG}
          ${ARG_FETCHCONTENT_ARGS}
        )
      elseif(DEFINED ARG_URL)
        # URL/Tarball mode
        set(FETCH_ARGS
          ${NAME}
          URL ${ARG_URL}
          ${ARG_FETCHCONTENT_ARGS}
        )
        message(STATUS "  Downloading ${NAME} from URL")
        message(STATUS "    -- URL: ${ARG_URL}")

        # Check again for local source override
        if(DEFINED FETCHCONTENT_SOURCE_DIR_${NAME_UPPER})
          message(STATUS "    -- Using local source instead: ${FETCHCONTENT_SOURCE_DIR_${NAME_UPPER}}")
        endif()

        # Add optional arguments if provided
        if(DEFINED ARG_URL_HASH)
          list(APPEND FETCH_ARGS URL_HASH ${ARG_URL_HASH})
          message(STATUS "    -- URL_HASH: ${ARG_URL_HASH}")
        endif()

        if(DEFINED ARG_SOURCE_DIR)
          list(APPEND FETCH_ARGS SOURCE_DIR ${ARG_SOURCE_DIR})
          message(STATUS "    -- SOURCE_DIR: ${ARG_SOURCE_DIR}")
        endif()

        if(DEFINED ARG_SOURCE_SUBDIR)
          list(APPEND FETCH_ARGS SOURCE_SUBDIR ${ARG_SOURCE_SUBDIR})
          message(STATUS "    -- SOURCE_SUBDIR: ${ARG_SOURCE_SUBDIR}")
        endif()

        if(DEFINED ARG_DOWNLOAD_EXTRACT_TIMESTAMP)
          list(APPEND FETCH_ARGS DOWNLOAD_EXTRACT_TIMESTAMP ${ARG_DOWNLOAD_EXTRACT_TIMESTAMP})
          message(STATUS "    -- DOWNLOAD_EXTRACT_TIMESTAMP: ${ARG_DOWNLOAD_EXTRACT_TIMESTAMP}")
        endif()
        FetchContent_Declare(${FETCH_ARGS})
      else()
        message(FATAL_ERROR "Either GIT_REPOSITORY or URL must be specified for dependency ${NAME}")
      endif()

      # Handle build args
      message(STATUS "    -- BUILD_ARGS: ${ARG_BUILD_ARGS}")
      foreach( _flag IN LISTS ARG_BUILD_ARGS)
        set(CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS}     ${_flag}")
        set(CMAKE_C_FLAGS       "${CMAKE_C_FLAGS}       ${_flag}")
        set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${_flag}")
      endforeach()

      message(STATUS "")
      message(STATUS "=== ${NAME} CMake Output ===")
      FetchContent_MakeAvailable(${NAME})
      message(STATUS "=== End ${NAME} CMake Output ===")
      message(STATUS "")

      # Alias the target if needed
      if(NOT ("${ARG_BUILD_TARGET}" STREQUAL "${ARG_INSTALL_TARGET}"))
        add_library(${ARG_INSTALL_TARGET} ALIAS ${ARG_BUILD_TARGET})
      endif()
    elseif(ARG_REQUIRED)
      message(FATAL_ERROR "Required dependency ${NAME} not found and dependency fetch is disabled")
    else()
      message(STATUS "  Dependency fetch is disabled, skipping ${NAME}")
    endif()
  endif()

  # Make sure that the INSTALL target exists
  if(NOT TARGET ${ARG_INSTALL_TARGET})
    message(FATAL_ERROR "  ${NAME} Install target ${ARG_INSTALL_TARGET} does not exist")
  endif()

endfunction(handle_dependency)
