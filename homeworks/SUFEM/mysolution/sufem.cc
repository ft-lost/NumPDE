/**
 * @file sufem.cc
 * @brief NPDE homework SUFEM code
 * @author R. Hiptmair
 * @date July 2023
 * @copyright Developed at SAM, ETH Zurich
 */

#include "sufem.h"

#include <lf/geometry/geometry_interface.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/mesh_function_global.h>

#include <cstdlib>
#include <iomanip>
#include <memory>
#include <string>

namespace SUFEM {
void testSUFEMConvergence(unsigned int reflevels, const char* filename) {
  std::cout << "Convergence test for SU FEM on unit square, pure advection"
            << std::endl;
  // Velocity field: rigid body rotation
  auto velo = [](Eigen::Vector2d x) -> Eigen::Vector2d {
    return Eigen::Vector2d(-x[1], x[0]);
  };
  // Exact solution
  auto u = [](Eigen::Vector2d x) -> double {
    const double xn = x.norm();
    return ((xn <= 1.0) ? (1.0 - xn) * xn : 0.0);
  };
  // Initialize mesh function
  lf::mesh::utils::MeshFunctionGlobal mf_velo(velo);
  lf::mesh::utils::MeshFunctionGlobal mf_u(u);

  // Triangular mesh of the unit square
  auto cmesh_p = lf::mesh::test_utils::GenerateHybrid2DTestMesh(3, 1.0 / 3.0);
  // Obtain a pointer to a hierarchy of nested meshes
  std::shared_ptr<lf::refinement::MeshHierarchy> multi_mesh_p =
      lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(cmesh_p,
                                                              reflevels);
  lf::refinement::MeshHierarchy& multi_mesh{*multi_mesh_p};
  // Ouput information about hierarchy of nested meshes
  std::cout << "\t Sequence of nested meshes used in test\n";
  multi_mesh.PrintInfo(std::cout);
  // Number of levels
  lf::base::size_type L = multi_mesh.NumLevels();
  // LEVEL LOOP: Do computations on all levels
  // Vector for storing error norms
  std::vector<std::pair<lf::base::size_type, double>> errs{};
  for (lf::base::size_type level = 0; level < L; ++level) {
    std::shared_ptr<lf::mesh::Mesh> mesh_p = multi_mesh.getMesh(level);
    // Set up global FE space; lowest order Lagrangian finite elements
    auto fe_space =
        std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_p);
    const lf::assemble::DofHandler& dofh{fe_space->LocGlobMap()};
    const lf::base::size_type N_dofs = dofh.NumDofs();
    // Solve Dirichlet boundary value problem
    Eigen::VectorXd uh_coeffs =
        solveAdvectionDirichlet(fe_space, mf_velo, mf_u);
    // Mesh function for FE solution
    const lf::fe::MeshFunctionFE mf_uh(fe_space, uh_coeffs);
    // Compute error
    double L2err =  // NOLINT
        std::sqrt(lf::fe::IntegrateMeshFunction(
            *mesh_p, lf::mesh::utils::squaredNorm(mf_uh - mf_u), 2));
    errs.emplace_back(N_dofs, L2err);
    // Output result to file
    if (filename) {
      lf::io::VtkWriter vtk_writer(
          mesh_p, std::string(filename) + std::to_string(level) + ".vtk");
      auto nodal_data =
          lf::mesh::utils::make_CodimMeshDataSet<double>(mesh_p, 2);
      for (int global_idx = 0; global_idx < N_dofs; global_idx++) {
        nodal_data->operator()(dofh.Entity(global_idx)) = uh_coeffs[global_idx];
      };
      vtk_writer.WritePointData("solution_advection_problem", *nodal_data);
    }
  }
  std::cout << std::left << std::setw(10) << "N" << std::right << std::setw(16)
            << "L2 error" << std::endl;
  for (const auto& err : errs) {
    auto [N, l2err] = err;
    std::cout << std::left << std::setw(10) << N << std::left << std::setw(16)
              << l2err << std::endl;
  }
}
}  // namespace SUFEM
