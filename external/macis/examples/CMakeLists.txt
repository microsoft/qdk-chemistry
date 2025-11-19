add_executable(standalone_driver src/standalone_driver.cxx src/ini_input.cxx)
target_link_libraries(standalone_driver PUBLIC macis)

add_executable(auto_sci src/auto_sci.cxx src/ini_input.cxx)
target_link_libraries(auto_sci PUBLIC macis)
