# SeQuant asserts that [[no_unique_address]] elides a tag member on every
# compiler except cl.exe, but clang-cl ignores the attribute too (Microsoft ABI).
# Applied from the SeQuant source root before configuring the dependency.

set(_header "SeQuant/core/utility/aggregate.hpp")
if(NOT EXISTS "${_header}")
  message(FATAL_ERROR
          "Cannot patch SeQuant for clang-cl: header not found at ${_header}")
endif()

file(READ "${_header}" _content)

set(_guard "#if !defined(_MSC_VER) || defined(__clang__)")
string(FIND "${_content}" "${_guard}" _guard_pos)
if(NOT _guard_pos EQUAL -1)
  string(REPLACE "${_guard}" "#if !defined(_MSC_VER)" _content "${_content}")
  file(WRITE "${_header}" "${_content}")
  message(STATUS "Patched SeQuant [[no_unique_address]] size check for clang-cl")
elseif(NOT _content MATCHES "#if !defined\\(_MSC_VER\\)[\r\n]")
  message(FATAL_ERROR
          "Cannot patch SeQuant for clang-cl: size-check guard not found")
endif()
