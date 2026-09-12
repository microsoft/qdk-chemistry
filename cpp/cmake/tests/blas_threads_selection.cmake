# Regression test for qdk-blas-threads.cmake: the backend must be selected by
# link probe, not by a vendor label. Both fixture projects and the two fake BLAS
# libraries they use are generated here, so this depends on nothing but a
# compiler -- in particular not on which BLAS the machine happens to have.
#
# cmake -DQDK_SOURCE_DIR=<cpp> -DQDK_WORK_DIR=<scratch> -P run_selection_test.cmake

set(util "${QDK_SOURCE_DIR}/src/qdk/chemistry/algorithms/microsoft/scf/src/util")

# CMAKE_CONFIGURATION_TYPES pins multi-config generators (the default on
# Windows) to the one configuration CMAKE_BUILD_TYPE selects for single-config
# ones, so both kinds of generator build and probe exactly Release here.
function(run_cmake dir)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -S "${QDK_WORK_DIR}/${dir}" -B "${QDK_WORK_DIR}/${dir}/build"
            -DCMAKE_BUILD_TYPE=Release -DCMAKE_CONFIGURATION_TYPES=Release ${ARGN}
    OUTPUT_VARIABLE out ERROR_VARIABLE out RESULT_VARIABLE code)
  if(NOT code EQUAL 0)
    message(FATAL_ERROR "configure of ${dir} failed:\n${out}")
  endif()
  set(output "${out}" PARENT_SCOPE)
endfunction()

# A BLAS exporting the OpenBLAS thread-control extensions, and one exporting
# none -- as a libblas.so.3 facade over OpenBLAS does. Only symbols matter.
file(WRITE "${QDK_WORK_DIR}/fake/with_symbols.cpp" [==[
extern "C" {
static int count = 4;
void openblas_set_num_threads(int n) { count = n; }
int openblas_get_num_threads(void) { return count; }
}
]==])
file(WRITE "${QDK_WORK_DIR}/fake/without_symbols.cpp" [==[
extern "C" double qdk_fake_blas_ddot(void) { return 0.0; }
]==])
file(WRITE "${QDK_WORK_DIR}/fake/CMakeLists.txt" [==[
cmake_minimum_required(VERSION 3.15)
project(qdk_fake_blas CXX)
foreach(lib with_symbols without_symbols)
  add_library(${lib} STATIC ${lib}.cpp)
  # Per-config output name: $<TARGET_FILE:> differs between configurations, and
  # a single output would then have to be written more than once.
  file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/${lib}-$<CONFIG>.path
       CONTENT $<TARGET_FILE:${lib}>)
endforeach()
]==])

# Static libraries, so no runtime loader is involved and this behaves the same
# everywhere.
run_cmake(fake)
execute_process(
  COMMAND ${CMAKE_COMMAND} --build "${QDK_WORK_DIR}/fake/build" --config Release
  OUTPUT_VARIABLE out ERROR_VARIABLE out RESULT_VARIABLE code)
if(NOT code EQUAL 0)
  message(FATAL_ERROR "build of fake BLAS failed:\n${out}")
endif()
file(READ "${QDK_WORK_DIR}/fake/build/with_symbols-Release.path" with_symbols)
file(READ "${QDK_WORK_DIR}/fake/build/without_symbols-Release.path" without_symbols)

file(WRITE "${QDK_WORK_DIR}/select/CMakeLists.txt" [==[
cmake_minimum_required(VERSION 3.15)
project(qdk_blas_select CXX)
set(CMAKE_CXX_STANDARD 20)
include(${QDK_BLAS_MODULE})
qdk_select_blas_thread_backend(${QDK_BLAS_UTIL_DIR} backend_src)
get_filename_component(backend_name ${backend_src} NAME)
message(STATUS "QDK_SELECTED=${backend_name}")
]==])

# One build directory for all three cases, so the last also covers re-probing
# after the BLAS changes underneath an existing build.
macro(select_with expected)
  run_cmake(select
    "-DQDK_BLAS_MODULE=${QDK_SOURCE_DIR}/cmake/qdk-blas-threads.cmake"
    "-DQDK_BLAS_UTIL_DIR=${util}" ${ARGN})
  if(NOT output MATCHES "QDK_SELECTED=${expected}")
    message(FATAL_ERROR "expected ${expected}, got:\n${output}")
  endif()
endmacro()

# Symbols present, but no vendor label: BLAS_LIBRARIES was passed directly, so
# FindBLAS never ran and set none.
select_with("blas_threads_openblas.cpp" "-DBLAS_LIBRARIES=${with_symbols}")

# Vendor label correct, symbols absent: must decline rather than pick a backend
# that cannot link.
select_with("blas_threads_none.cpp" "-DBLAS_LIBRARIES=${without_symbols}"
            -DBLAS_VENDOR=OpenBLAS)

# Back to the first BLAS: a probe result cached from the case above would
# wrongly leave thread control disabled.
select_with("blas_threads_openblas.cpp" "-DBLAS_LIBRARIES=${with_symbols}")

message(STATUS "BLAS thread-control backend selection: all cases passed")
