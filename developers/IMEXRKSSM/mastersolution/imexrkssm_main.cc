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
  lf::io::GmshReader reader(std::move(mesh_factory),
                            CURRENT_SOURCE_DIR "/../meshes/square.msh");
  auto mesh = reader.mesh();
  // obtain dofh for lagrangian finite element space
  auto fe_space =
      std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
  const lf::assemble::DofHandler& dofh{fe_space->LocGlobMap()};
  Eigen::VectorXd mu_exact = IMEX::solveTestProblem(fe_space, 2048);
  std::vector<double> err;
  // Solve test problem
  std::cout << "Test IMEXRK on square Mesh" << std::endl;
  std::cout << "M" << std::setw(20) << "Error" << std::endl;
  for (int M = 2; M < 2048; M *= 2) {
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
