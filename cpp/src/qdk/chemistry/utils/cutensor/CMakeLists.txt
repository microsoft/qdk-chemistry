# cuTENSOR evaluation backend for SeQuant expression trees. Its own target keeps
# SeQuant out of the public interface of qdk::chemistry.
add_library(sequant_cutensor
    engine.cpp
    sequant_result.cpp
)
add_library(qdk::sequant_cutensor ALIAS sequant_cutensor)
set_target_properties(sequant_cutensor PROPERTIES
    OUTPUT_NAME qdk_chemistry_sequant_cutensor
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(sequant_cutensor PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(sequant_cutensor PUBLIC
    SeQuant::SeQuant
    CUDA::cudart
    CUDA::cublas
    cutensor
)
target_compile_features(sequant_cutensor PUBLIC cxx_std_20)
