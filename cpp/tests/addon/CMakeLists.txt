# Test add-on for C++ plugins, built into the library through QDK_ADDONS_DIR
# and tested by test_plugins.cpp. The plugins run in the order declared.
target_sources(chemistry PRIVATE plugin.cpp)
qdk_add_plugin(qdk_test_addon::register_duplicate)
qdk_add_plugin(qdk_test_addon::throw_non_standard)
qdk_add_plugin(qdk_test_addon::register_plugin)
