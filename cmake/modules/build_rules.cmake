# Build rule for problems
function(build_problem TARGET DIR OUTPUT_NAME)
  # Defines SOURCES and LIBRARIES
  include(${DIR}/dependencies.cmake)

  add_executable(${TARGET} ${SOURCES})
  set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME ${OUTPUT_NAME})
  target_compile_definitions(${TARGET} PRIVATE CURRENT_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/${DIR}\")
  target_compile_definitions(${TARGET} PRIVATE CURRENT_BINARY_DIR=\"${CMAKE_CURRENT_BINARY_DIR}\")
  # For including all symbols in the executable: Does not work on Max OS X
  # target_link_libraries(${TARGET} PUBLIC   "-Wl,--whole-archive" ${LIBRARIES} "-Wl,--no-whole-archive")
  target_link_libraries(${TARGET} PUBLIC ${LIBRARIES})
  
  add_library(${TARGET}.static STATIC ${SOURCES})
  set_target_properties(${TARGET}.static PROPERTIES OUTPUT_NAME ${OUTPUT_NAME}.static)
  target_compile_definitions(${TARGET}.static PRIVATE CURRENT_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/${DIR}\")
  target_compile_definitions(${TARGET}.static PRIVATE CURRENT_BINARY_DIR=\"${CMAKE_CURRENT_BINARY_DIR}\")
  target_link_libraries(${TARGET}.static PUBLIC ${LIBRARIES})
endfunction(build_problem)

# Build rule for tests
function(build_test TARGET TARGET_TO_TEST DIR OUTPUT_NAME)
  # Defines SOURCES and LIBRARIES
  include(${DIR}/test/dependencies.cmake)
  include(GoogleTest)

  add_executable(${TARGET} ${SOURCES})
  set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME ${OUTPUT_NAME})
  target_compile_definitions(${TARGET} PRIVATE CURRENT_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/${DIR}/test\")
  target_compile_definitions(${TARGET} PRIVATE CURRENT_BINARY_DIR=\"${CMAKE_CURRENT_BINARY_DIR}\")
  target_link_libraries(${TARGET} PUBLIC ${LIBRARIES} ${TARGET_TO_TEST}.static)

  # gtest_discover_tests(${TARGET}) Not necessary given that the CI pipeline runs the tests
endfunction(build_test)

# Helper function to create relative symbolic links from the current binary directory to the source directory
function(create_relative_symlink_from_bin_dir target link_name)
  # compute relative path from current binary directory to target
  file(RELATIVE_PATH target_rel ${CMAKE_CURRENT_BINARY_DIR} ${target})
  # create symbolic links
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${target_rel} ${CMAKE_CURRENT_BINARY_DIR}/${link_name})
endfunction()