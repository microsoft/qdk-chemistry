target_link_libraries(chemistry PUBLIC gauxc::gauxc)

# gauxc_impl.cpp uses dlsym() to look up optional OpenBLAS thread-control
# symbols at runtime; on glibc, dlsym lives in libdl (folded into libc on
# newer glibc, but linking it explicitly keeps older toolchains working).
if(NOT (MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  target_link_libraries(chemistry PRIVATE ${CMAKE_DL_LIBS})
endif()

target_sources(chemistry PRIVATE gauxc.cpp gauxc_impl.cpp)
