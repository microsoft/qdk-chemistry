# Builds an add-on into the chemistry library and runs its plugins.
#
# An add-on is a directory with a CMakeLists.txt, given by the QDK_ADDONS_DIR
# cache variable and added with add_subdirectory once the chemistry target
# exists. It adds its sources to chemistry with target_sources and declares
# each plugin with qdk_add_plugin(<function>): a function `void <function>()`,
# optionally namespace-qualified, that registers the add-on's implementations
# through the algorithm factories. For example,
#
#   target_sources(chemistry PRIVATE plugin.cpp custom_scf_solver.cpp)
#   qdk_add_plugin(custom::register_plugin)
#
# The plugins are compiled into plugins.cpp, generated from
# src/qdk/chemistry/algorithms/plugins.cpp.in, which runs them once, in the
# order declared, on the first use of an algorithm factory, after the built-in
# implementations are registered. A plugin that throws is reported as a
# warning. tests/addon is an add-on built in CI.

function(qdk_add_plugin function)
  set_property(GLOBAL APPEND PROPERTY QDK_PLUGINS ${function})
endfunction()

if(QDK_ADDONS_DIR)
  add_subdirectory(${QDK_ADDONS_DIR} addons)
endif()

get_property(_qdk_plugins GLOBAL PROPERTY QDK_PLUGINS)
set(QDK_PLUGIN_DECLARATIONS "")
set(QDK_PLUGIN_ENTRIES "")
foreach(_plugin IN LISTS _qdk_plugins)
  if(_plugin MATCHES "^(.+)::([^:]+)$")
    string(APPEND QDK_PLUGIN_DECLARATIONS
      "namespace ${CMAKE_MATCH_1} {\nvoid ${CMAKE_MATCH_2}();\n}\n")
  else()
    string(APPEND QDK_PLUGIN_DECLARATIONS "void ${_plugin}();\n")
  endif()
  string(APPEND QDK_PLUGIN_ENTRIES "{\"${_plugin}\", &::${_plugin}},")
endforeach()
configure_file(src/qdk/chemistry/algorithms/plugins.cpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/plugins.cpp @ONLY)
target_sources(chemistry PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/plugins.cpp)
