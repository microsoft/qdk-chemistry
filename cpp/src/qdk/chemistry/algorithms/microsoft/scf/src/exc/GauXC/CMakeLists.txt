target_link_libraries(chemistry PUBLIC gauxc::gauxc)

target_sources(chemistry PRIVATE gauxc.cpp gauxc_impl.cpp)
