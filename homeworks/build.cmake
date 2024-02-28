# Provides variable PROBLEM_NAME
include(${CMAKE_SOURCE_DIR}/cmake/modules/build_variables.cmake)

# Provides functions build_problem and build_test
include(${CMAKE_SOURCE_DIR}/cmake/modules/build_rules.cmake)

# Helper function to create relative symbolic links from the current binary directory to the source directory
function(create_relative_symlink_from_bin_dir target link_name)
  # compute relative path from current binary directory to target
  file(RELATIVE_PATH target_rel ${CMAKE_CURRENT_BINARY_DIR} ${target})
  # create symbolic links
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${target_rel} ${CMAKE_CURRENT_BINARY_DIR}/${link_name})
endfunction()

# pass correct arguemnts to build rules
function(build PROBLEM_NAME DIR)
  set(PROBLEM_TARGET ${PROBLEM_NAME}_${DIR})
  set(TEST_TARGET ${PROBLEM_NAME}_test_${DIR})

  # create relative symbolic link to mesh files
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/meshes)
    create_relative_symlink_from_bin_dir(${CMAKE_CURRENT_SOURCE_DIR}/meshes meshes)
  endif()

  # create relative symbolic link to master solution scirpts
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mastersolution/scripts)
    create_relative_symlink_from_bin_dir(${CMAKE_CURRENT_SOURCE_DIR}/mastersolution/scripts mastersolution_scripts)
  endif()

  # create relative symbolic link to general scripts
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/scripts)
    create_relative_symlink_from_bin_dir(${CMAKE_CURRENT_SOURCE_DIR}/scripts scripts)
  endif()
  
  # problem
  build_problem(${PROBLEM_TARGET} ${DIR} ${PROBLEM_TARGET})

  # tests
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${DIR}/test)
    build_test(${TEST_TARGET} ${PROBLEM_TARGET} ${DIR} ${TEST_TARGET})
  endif()
endfunction(build)

if(${MASTERSOLUTION})
  build(${PROBLEM_NAME} mastersolution)
endif()

if(${MYSOLUTION})
  build(${PROBLEM_NAME} mysolution)
endif()
