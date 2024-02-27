# Provides variable PROBLEM_NAME
include(${CMAKE_SOURCE_DIR}/cmake/modules/build_variables.cmake)

# Provides functions build_problem and build_test
include(${CMAKE_SOURCE_DIR}/cmake/modules/build_rules.cmake)

# pass correct arguemnts to build rules
function(build PROBLEM_NAME DIR)
  set(PROBLEM_TARGET ${PROBLEM_NAME}_${DIR})
  set(TEST_TARGET ${PROBLEM_NAME}_test_${DIR})

  # copy mesh files to build directory
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/meshes)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/meshes DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  endif()

  # copy master solution scripts to build directory
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mastersolution/scripts)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/mastersolution/scripts DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  endif()

  # copy general scripts to build directory
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/scripts)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/scripts DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
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
