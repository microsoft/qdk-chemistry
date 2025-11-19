add_subdirectory(INCORE)
add_subdirectory(SNK)
add_subdirectory(LIBINT2_DIRECT)

target_sources(chemistry PRIVATE schwarz.cpp eri.cpp eri_multiplexer.cpp moeri.cpp eri_df_base.cpp)
