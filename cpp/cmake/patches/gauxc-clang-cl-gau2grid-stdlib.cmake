# Gau2Grid's generated helper uses exit() but omits stdlib.h for clang-cl.
# Applied from the GauXC source root before configuring the dependency.

set(_helper "external/gau2grid/generated_source/gau2grid_helper.c")
if(NOT EXISTS "${_helper}")
  message(FATAL_ERROR
          "Cannot patch Gau2Grid for clang-cl: helper not found at ${_helper}")
endif()

file(READ "${_helper}" _content)

if(NOT _content MATCHES "#[ \t]*include[ \t]*<stdlib\\.h>")
  if(NOT _content MATCHES "#[ \t]*include[ \t]*<math\\.h>")
    message(FATAL_ERROR
            "Cannot patch Gau2Grid for clang-cl: math.h include not found")
  endif()
  string(REGEX REPLACE "(#[ \t]*include[ \t]*<math\\.h>)"
                       "#include <stdlib.h>\n\\1"
                       _content "${_content}")
  file(WRITE "${_helper}" "${_content}")
  message(STATUS "Patched Gau2Grid helper for clang-cl stdlib declarations")
endif()
