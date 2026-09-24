# Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
<https://cla.microsoft.com>.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## C++ Header Dependencies

Keep reusable QDK forward declarations in the owning module's `fwd.hpp` or `fwd.h`,
and include that header from both consumers and definitions. Prefer upstream
forward headers such as `H5Classes.h` and `nlohmann/json_fwd.hpp`; declarations
for dependencies without one belong in a single internal adapter.

Register header contracts in [the header checks](cpp/tests/headers/CMakeLists.txt).
They compile headers independently, verify that forward-declared types remain
incomplete, and check compatibility with definitions in both include orders.
The checks are part of native builds with `BUILD_TESTING=ON` and can be built
separately with `cmake --build cpp/build --target qdk_header_checks`.

## C++ Compilation Groups

On CMake 3.18 or newer, compatible QDK sources compile in explicit unity groups.
Groups are declared with `qdk_unity_group` beside their source lists; ungrouped
files compile independently. This does not change compiler optimization settings
or enable unity builds for dependencies.

Use `-DQDK_CHEMISTRY_UNITY_BUILD=OFF` for separate translation units, or
`-C cmake.define.QDK_CHEMISTRY_UNITY_BUILD=OFF` when building through pip.
The Linux Python 3.10 CI job keeps this mode covered. Header-contract checks
always compile independently, regardless of the selected mode.
