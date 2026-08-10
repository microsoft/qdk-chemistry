# Gau2Grid's generated helper uses exit() but omits stdlib.h for clang-cl.
# Applied from the GauXC source root before configuring the dependency.

set(_helper "external/gau2grid/generated_source/gau2grid_helper.c")
file(READ "${_helper}" _content)

if(NOT _content MATCHES "#include <stdlib.h>\n#include <math.h>")
  string(REPLACE "#include <math.h>"
                 "#include <stdlib.h>\n#include <math.h>"
                 _content "${_content}")
  file(WRITE "${_helper}" "${_content}")
  message(STATUS "Patched Gau2Grid helper for clang-cl stdlib declarations")
endif()
