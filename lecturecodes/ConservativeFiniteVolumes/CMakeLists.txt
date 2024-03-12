# define executable
set(sources
  consformevl.h
  ode.h
  consformevldriver.cc
  numexp_runner.h
)

add_executable(lecturecodes.consformevl ${sources})
target_link_libraries(lecturecodes.consformevl
  PUBLIC Eigen3::Eigen
  PUBLIC Boost::boost
)

set(highressrc
  highresevl.h
  highresevldriver.cc
  numexp_runner.h
)

add_executable(lecturecodes.highresevl ${highressrc})

target_link_libraries(lecturecodes.highresevl
  PUBLIC Eigen3::Eigen
  PUBLIC Boost::boost
)

create_relative_symlink_from_bin_dir(${CMAKE_CURRENT_SOURCE_DIR}/scripts scripts)