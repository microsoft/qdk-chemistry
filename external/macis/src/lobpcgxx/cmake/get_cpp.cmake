
include_guard()

#[[
# This function encapsulates the process of getting CMakePP using CMake's
# FetchContent module. We have encapsulated it in a function so we can set the
# options for it's configure step without affecting the options for the parent
# project's configure step.
#]]
function(get_cpp)
    include(FetchContent)
    set(build_testing_old "${BUILD_TESTING}")
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
            cpp
            GIT_REPOSITORY https://github.com/wavefunction91/CMakePackagingProject
            GIT_TAG zip-files
    )
    FetchContent_MakeAvailable(cpp)
    set(BUILD_TESTING "${build_testing_old}" CACHE BOOL "" FORCE)
endfunction()

# Call the function we just wrote to get CMakePP
get_cpp()

# Include CMakePP
include(cpp/cpp)
