/**
 * @ file taylorhoodnonmonolithic_main.cc
 * @ brief NPDE homework TEMPLATE MAIN FILE
 * @ author Ralf Hiptmair
 * @ date June 2024
 * @ copyright Developed at SAM, ETH Zurich
 */

#include <lf/mesh/test_utils/test_meshes.h>

#include <iostream>

#include "taylorhoodnonmonolithic.h"

int main(int /*argc*/, char** /*argv*/) {
  std::cout
      << "== NumPDE homework problem TaylorHoodNonMonolithic by R.H. ==\n";
#if SOLUTION
  // Simple mesh of the unit square
  std::shared_ptr<lf::mesh::Mesh> mesh_p =
      lf::mesh::test_utils::GenerateHybrid2DTestMesh(3, 1.0 / 3.0);
  unsigned int refsteps = 4;

  std::cout << "Generate fine mesh by regular refinement\n";
  const std::shared_ptr<lf::refinement::MeshHierarchy> multi_mesh_p =
      lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(mesh_p, refsteps);
  lf::refinement::MeshHierarchy& multi_mesh{*multi_mesh_p};
  // Ouput summary information about hierarchy of nested meshes
  std::cout << "\t Sequence of nested meshes created\n";
  multi_mesh.PrintInfo(std::cout);

  // Number of levels
  const int L = multi_mesh.NumLevels();
  const std::shared_ptr<const lf::mesh::Mesh> fine_mesh_p =
      multi_mesh.getMesh(L - 1);

  Eigen::Matrix<double, Eigen::Dynamic, 6> it_err_norms{
      TaylorHoodNonMonolithic::monitorUzawaConvergence(fine_mesh_p, true)};
  std::cout << it_err_norms << std::endl;
#endif
}
