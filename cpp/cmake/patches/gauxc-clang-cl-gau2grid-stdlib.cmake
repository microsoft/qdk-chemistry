# Gau2Grid's generated helper uses exit() but omits stdlib.h for clang-cl.
# Applied from the GauXC source root before configuring the dependency.

set(_helper "external/gau2grid/generated_source/gau2grid_helper.c")
if(NOT EXISTS "${_helper}")
  message(FATAL_ERROR
          "Cannot patch Gau2Grid for clang-cl: helper not found at ${_helper}")
endif()

file(READ "${_helper}" _content)

string(FIND "${_content}" "#include <stdlib.h>\n#include <math.h>" _patched_lf)
string(FIND "${_content}" "#include <stdlib.h>\r\n#include <math.h>"
       _patched_crlf)

if(_patched_lf EQUAL -1 AND _patched_crlf EQUAL -1)
  string(FIND "${_content}" "#include <math.h>" _math_include)
  if(_math_include EQUAL -1)
    message(FATAL_ERROR
            "Cannot patch Gau2Grid for clang-cl: math.h include not found")
  endif()
  if(_content MATCHES "\r\n")
    set(_newline "\r\n")
  else()
    set(_newline "\n")
  endif()
  string(REPLACE "#include <math.h>"
                 "#include <stdlib.h>${_newline}#include <math.h>"
                 _content "${_content}")
  file(WRITE "${_helper}" "${_content}")
  message(STATUS "Patched Gau2Grid helper for clang-cl stdlib declarations")
endif()
