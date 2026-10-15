target_sources(chemistry PRIVATE
    archive.cpp
    cube_generator.cpp
    hash_context.cpp
    valence_space.cpp
    orbital_rotation.cpp
    golden_section.cpp
    logger.cpp
)

if(WIN32)
    target_link_libraries(chemistry PRIVATE advapi32)
endif()
