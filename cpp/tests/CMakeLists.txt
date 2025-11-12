# Enable testing with CTest
include(CTest)

# Try to find Google Test, if not found, fetch it
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    message(STATUS "Google Test not found locally, fetching from GitHub...")
    include(FetchContent)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )

    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest)

    # Create aliases to match find_package targets
    add_library(GTest::gtest ALIAS gtest)
    add_library(GTest::gtest_main ALIAS gtest_main)
    add_library(GTest::gmock ALIAS gmock)
    add_library(GTest::gmock_main ALIAS gmock_main)

    message(STATUS "Google Test fetched successfully")
else()
    message(STATUS "Using system Google Test")
endif()

include(GoogleTest)

set(QDK_SCF_TEST_DIR ${PROJECT_SOURCE_DIR}/src/qdk/chemistry/algorithms/microsoft/scf/tests)

# Create custom main for in-source resource discovery
add_library(test_main qdk_test_main.cpp)
target_link_libraries(test_main PUBLIC GTest::gtest GTest::gmock chemistry)
target_compile_definitions(test_main PUBLIC TEST_RESOURCES_DIR="${PROJECT_SOURCE_DIR}/src/qdk/chemistry/algorithms/microsoft/scf/resources")
target_include_directories(test_main PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/tests>
  $<BUILD_INTERFACE:${QDK_SCF_TEST_DIR}>
)

# Manage reference data for SCF tests
# SCF Specific test logic
option(QDK_RUN_LONG_TESTS "Enable running of long tests" OFF)
set(TEST_DATA_DIR ${QDK_SCF_TEST_DIR}/data)
set(TEST_XYZ_DIR ${QDK_SCF_TEST_DIR}/sample)
configure_file(${QDK_SCF_TEST_DIR}/test_config.h.in ${PROJECT_BINARY_DIR}/tests/test_config.h)


# Collect all test source files
file(GLOB_RECURSE TEST_SOURCES "test_*.cpp")
list(APPEND TEST_SOURCES
    ${QDK_SCF_TEST_DIR}/int1e_test.cpp
    ${QDK_SCF_TEST_DIR}/basis_test.cpp
    ${QDK_SCF_TEST_DIR}/scf_test.cpp
    ${QDK_SCF_TEST_DIR}/util_tests.cpp
)

# Function to create test target from source file
function(add_gtest_executable test_source)
    # Get the filename without extension for the target name
    get_filename_component(test_name ${test_source} NAME_WE)

    # Create the executable
    add_executable(${test_name} ${test_source})

    # Link libraries
    target_link_libraries(${test_name} PRIVATE test_main)

    if(QDK_CHEMISTRY_ENABLE_COVERAGE)
        # Handle coverage of inline functions
        target_compile_options(${test_name} PRIVATE --coverage -fprofile-arcs -ftest-coverage)
        target_link_libraries(${test_name} PRIVATE --coverage)
    endif()

    # Discover tests automatically
    gtest_discover_tests(${test_name})

    # Set properties for better organization
    set_target_properties(${test_name} PROPERTIES
        FOLDER "Tests"
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
endfunction()

# Create test executables for all test source files
foreach(test_source ${TEST_SOURCES})
    add_gtest_executable(${test_source})
endforeach()

# Print information about discovered tests for debugging
message(STATUS "Found ${CMAKE_CURRENT_LIST_LENGTH} test files:")
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME)
    message(STATUS "  - ${test_name}")
endforeach()

# Add a custom target to run all tests with verbose output
add_custom_target(run_tests_verbose
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose --output-on-failure
    DEPENDS ${TEST_SOURCES} ${SCF_TEST_SOURCES}
    COMMENT "Running all tests with verbose output"
    VERBATIM
)

# Output test results to the console
set(CMAKE_CTEST_ARGUMENTS "--output-on-failure")
