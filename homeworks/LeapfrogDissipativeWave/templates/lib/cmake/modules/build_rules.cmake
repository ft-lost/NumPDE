# Build rule for problems
function(build_problem TARGET DIR OUTPUT_NAME)
  # Defines SOURCES and LIBRARIES
  include(${DIR}/dependencies.cmake)

  add_executable(${TARGET} ${SOURCES})
  set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME ${OUTPUT_NAME})
