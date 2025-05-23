# define executable
set(sources
  Mesh.cc
  solve_triangular_FEM.cc
  SimpleLinearFEM2D.h
  matrix_assembler.cc
  vector_assembler.cc
  local_assembler.h
  local_assembler.cc
  fe_solver.cc
)

set(comparison
  compare_assembly_speed.cc
  Timer.h
  Mesh.cc
  SimpleLinearFEM2D.h
  matrix_assembler.cc
  vector_assembler.cc
  local_assembler.h
  local_assembler.cc
  fe_solver.cc
)


add_executable(lecturecodes.solve_triangular_FEM solve_triangular_FEM.cc ${sources})

add_executable(lecturecodes.compare_implementations compare_assembly_speed.cc ${comparison})

target_link_libraries(lecturecodes.solve_triangular_FEM
  PUBLIC Eigen3::Eigen
)

target_link_libraries(lecturecodes.compare_implementations
  PUBLIC Eigen3::Eigen
)

create_relative_symlink_from_bin_dir(${CMAKE_CURRENT_SOURCE_DIR}/meshes meshes)