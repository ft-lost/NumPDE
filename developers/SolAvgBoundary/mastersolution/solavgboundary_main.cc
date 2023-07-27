/**
 * @ file solavgboundary_main.cc
 * @ brief NPDE homework Solavgboundary code
 * @ author Bob Schreiner
 * @ date June 2023
 * @ copyright Developed at SAM, ETH Zurich
 */

#include <Eigen/Core>
#include <iostream>

#include "solavgboundary.h"

int main(int /*argc*/, char** /*argv*/) {
  // read in mesh and set up finite element space
  auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
  lf::io::GmshReader reader(std::move(mesh_factory),
                            CURRENT_SOURCE_DIR "/../meshes/square.msh");
  auto mesh = reader.mesh();
  // obtain dofh for lagrangian finite element space
  auto fe_space =
      std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
  const lf::assemble::DofHandler& dofh{fe_space->LocGlobMap()};
  // Solve test problem
  Eigen::VectorXd mu = solavgboundary::solveTestProblem(dofh);
  solavgboundary::visSolution(fe_space, mu, "square_solution.vtk");
  return 0;
}
