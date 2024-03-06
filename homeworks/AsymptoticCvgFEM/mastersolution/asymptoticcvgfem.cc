/**
 * @file asymptoticcvgfem.cc
 * @brief NPDE homework AsymptoticCvgFEM code
 * @author R. Hiptmair
 * @date June 2023
 * @copyright Developed at SAM, ETH Zurich
 */

#include "asymptoticcvgfem.h"

#include <lf/base/lf_assert.h>

#include <Eigen/Core>

namespace asymptoticcvgfem {

/* SAM_LISTING_BEGIN_1 */
Eigen::MatrixXd studyAsymptoticCvg(unsigned int reflev) {
  // Exact solution u
  auto u = [](Eigen::Vector2d x) -> double {
    return std::exp(x.squaredNorm());
  };
  // Its gradient
  auto grad_u = [&u](Eigen::Vector2d x) -> Eigen::Vector2d {
    return 2.0 * x * u(x);
  };
  // Right-hand side source function = -Laplacian of u
  auto rhs_f = [&u](Eigen::Vector2d x) -> double {
    return -4.0 * u(x) * (1.0 + x.squaredNorm());
  };
  // Impedance source function for unit triangle domain
  auto imp_h = [&u, &grad_u](Eigen::Vector2d x) -> double {
    Eigen::Vector2d ext_normal;
    // Geometric selection of boundary part
    if ((x[0] > x[1]) && (x[1] < 0.5 * (1.0 - x[0]))) {
      // Bottom edge
      ext_normal << 0.0, -1.0;
    } else if ((x[1] >= x[0]) && (x[0] < 0.5 * (1.0 - x[1]))) {
      ext_normal << -1.0, 0.0;
    } else {
      const double ir2 = 1.0 / std::sqrt(2.0);
      ext_normal << ir2, ir2;
    }
    return grad_u(x).dot(ext_normal) + u(x);
  };

  // Read mesh of unit triangle from file triangle.msh in directory ../meshes
  std::string mesh_file = "meshes/triangle.msh";
  auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
  lf::io::GmshReader reader(std::move(mesh_factory), mesh_file);
  std::shared_ptr<lf::mesh::Mesh> cmesh_p = reader.mesh();
  const lf::mesh::Mesh& cmesh{*cmesh_p};
  std::cout << "Read mesh from meshes/triangle.msh: " << cmesh.NumEntities(0)
            << " cells, " << cmesh.NumEntities(1) << " edges, "
            << cmesh.NumEntities(2) << " nodes " << std::endl;
  // Make sure that the mesh is purel triangular
  LF_VERIFY_MSG(cmesh.NumEntities(lf::base::RefEl::kQuad()) == 0,
                "Only triangular meshes admitted");
  // Create hierarchy of meshes by regular refinement
  std::shared_ptr<lf::refinement::MeshHierarchy> multi_mesh_p =
      lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(cmesh_p, reflev);
  lf::refinement::MeshHierarchy& multi_mesh{*multi_mesh_p};
  // Ouput information about hierarchy of nested meshes
  std::cout << "\t Sequence of nested meshes used in experiment\n";
  multi_mesh.PrintInfo(std::cout);
  // Number of levels
  const int L = multi_mesh.NumLevels();

  // Matrix for collecting error numbers
  Eigen::MatrixXd errs(L, 6);
  // LEVEL LOOP: Do computations on all levels
  for (int level = 0; level < L; ++level) {
    std::shared_ptr<lf::mesh::Mesh> mesh_p = multi_mesh.getMesh(level);
    // Set up global FE space; quadratic Lagrangian finite elements
    auto fe_space =
        std::make_shared<const lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_p);
    // Compute finite element solution of impedance problem for Laplacian
    const Eigen::VectorXd mu_h =
        asymptoticcvgfem::solveLaplImpBVP(fe_space, rhs_f, imp_h);
    // L2 error of solution
    std::cout << "Level " << (errs(level, 0) = level) << ": L2 error = "
              << (errs(level, 5) = error_V(fe_space, mu_h, u))
              << ", L2 error bd = "
              << (errs(level, 3) = error_III(fe_space, mu_h, u))
              << ", delta L2^2 = "
              << (errs(level, 4) = error_IV(fe_space, mu_h, u))
              << ", bd avg = " << (errs(level, 1) = error_I(fe_space, mu_h, u))
              << ", H1 error = "
              << (errs(level, 2) = error_II(fe_space, mu_h, grad_u))
              << std::endl;
  }
  return errs;
}
/* SAM_LISTING_END_1 */

}  // namespace asymptoticcvgfem
