#include <pybind11/pybind11.h>

namespace py = pybind11;

static const char* build_platform() {
#if defined(__APPLE__)
    return "macos";
#elif defined(_WIN32)
    return "windows";
#elif defined(__linux__)
    return "linux";
#else
    return "unknown";
#endif
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Temporary native module for qdk-chemistry name reservation.";

#ifdef VERSION_INFO
    m.def("version", []() { return std::string(VERSION_INFO); });
#else
    m.def("version", []() { return std::string("0.0.0"); });
#endif

    m.def("platform", []() { return std::string(build_platform()); });
}
