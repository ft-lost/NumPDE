/**
 * @ file imexrkssm_main.cc
 * @ brief NPDE homework TEMPLATE MAIN FILE
 * @ author Bob Schreiner
 * @ date June 2023
 * @ copyright Developed at SAM, ETH Zurich
 */

#include <Eigen/Core>
#include <iomanip>
#include <iostream>

#include "imexrkssm.h"

void TestConvergence() {
  std::vector<int> M(8);
  M[0] = 8;
  std::vector<double> rk_err(8);
  std::cout << "Test IMEXRK for f(x)=-x^2 g(x)=x y(0) = 0.25" << std::endl;
  std::cout << "M" << std::setw(20) << "Error" << std::endl;
  for (unsigned int i = 0; i < 7; ++i) {
    rk_err[i] = IMEX::IMEXError(M[i]);
    std::cout << M[i] << std::setw(20) << rk_err[i] << std::endl;
    M[i + 1] = 2 * M[i];
  }
  rk_err[7] = IMEX::IMEXError(M[7]);
  std::cout << M[7] << std::setw(20) << rk_err[7] << std::endl;
}

void TestSquareMesh() {
  // read in mesh and set up finite element space
  auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
  lf::io::GmshReader reader(std::move(mesh_factory), "meshes/square.msh");
  auto initial_mesh_p = reader.mesh();
  /* Debugging with finer meshes
  // Generate a sequence of meshes by regular refinement.
  std::cout<< "Genereatring Mesh sequence" << std::endl;
  const int reflevels = 2;
  std::shared_ptr<lf::refinement::MeshHierarchy> multi_mesh_p =
      lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(initial_mesh_p,
                                                              reflevels);
  lf::refinement::MeshHierarchy& multi_mesh{*multi_mesh_p};
  std::shared_ptr<lf::mesh::Mesh> mesh_p = multi_mesh.getMesh(2);
  */

  // obtain dofh for lagrangian finite element space
  auto fe_space =
      std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(initial_mesh_p);
  const lf::assemble::DofHandler& dofh{fe_space->LocGlobMap()};

  std::cout << "Computing reference solution" << std::endl;
  Eigen::VectorXd mu_exact = IMEX::solveTestProblem(fe_space, std::pow(2, 11));
  std::vector<double> err;
  // Convergence analysis by halving step size
  std::cout << "Test IMEXRK on square Mesh" << std::endl;
  std::cout << "M" << std::setw(20) << "Error" << std::endl;
  for (int M = 2; M < std::pow(2, 11); M *= 2) {
    const Eigen::VectorXd mu = IMEX::solveTestProblem(fe_space, M);
    const double error = (mu_exact - mu).norm();
    err.push_back(error);
    std::cout << M << std::setw(20) << error << std::endl;
  }
  IMEX::visSolution(fe_space, mu_exact, "square_solution.vtk");
}
int main(int /*argc*/, char** /*argv*/) {
  // TestConvergence();
  TestSquareMesh();
  return 0;
}
