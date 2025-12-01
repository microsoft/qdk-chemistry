cmake_minimum_required(VERSION 3.15)
project(qdk_doc_examples VERSION 1.0.0 LANGUAGES CXX)

# This CMakeLists.txt builds documentation examples to validate they compile correctly
# These are code snippets used in documentation, not full runnable programs

if(NOT TARGET qdk::chemistry)
    # If building standalone, find the qdk package
    find_package(qdk REQUIRED COMPONENTS chemistry)
endif()

# Collect all .cpp files in the current directory
file(GLOB DOC_EXAMPLES "*.cpp")

# Create object library targets for each example to validate compilation
# Using OBJECT libraries allows us to compile-check without creating executables
foreach(example_file ${DOC_EXAMPLES})
    # Get the name without extension
    get_filename_component(example_name ${example_file} NAME_WE)

    # Create target name
    set(target_name "doc_example_${example_name}")

    # Add as object library (compiles but doesn't link into executable)
    add_library(${target_name} OBJECT ${example_file})

    # Link against qdk::chemistry
    target_link_libraries(${target_name} PRIVATE qdk::chemistry)

    # Set C++20 standard
    target_compile_features(${target_name} PRIVATE cxx_std_20)

    # Suppress warnings for documentation snippets that may have unused variables
    target_compile_options(${target_name} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wno-unused-variable -Wno-unused-parameter>
        $<$<CXX_COMPILER_ID:MSVC>:/wd4101 /wd4189>
    )
endforeach()

message(STATUS "Documentation examples will be compiled for validation")
