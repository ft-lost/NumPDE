/**
 * @ file blendedparameterization_main.cc
 * @ brief NPDE homework TEMPLATE MAIN FILE
 * @ author R. Hiptmair
 * @ date January 2018
 * @ copyright Developed at SAM, ETH Zurich
 */

#include "blendedparameterization.h"
#include "MeshTriangleUnitSquareEigen.hpp"

#include <lf/assemble/assemble.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/uscalfe/uscalfe.h>
#include <lf/io/io.h>
#include <lf/fe/fe.h>
#include <lf/fe/fe_tools.h>
#include <lf/mesh/utils/utils.h>
#include <lf/mesh/utils/mesh_function_global.h>

/**
 *
 * @param fe_space finite elemt space
 * @param A LHS Finite element disc. Matrix in Triplet format
 * @param b RHS vector from Galerkin disc.
 * this function enforces the south boundary of the unit square to have a constant value of 1
 * and the east boundary  of the unit square to have a constant value of 2
 */
void enforce_boundary_conditions(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
    lf::assemble::COOMatrix<double>& A, Eigen::VectorXd& b) {
  const lf::mesh::utils::MeshFunctionGlobal mf_g_Gamma_in{
      //[](const Eigen::Vector2d& x) { return std::pow(sin(M_PI * x(0)), 2); }};
      [](const Eigen::Vector2d& x) { return x[0] == 0 ? 1. : 2.; }};
  lf::mesh::utils::AllCodimMeshDataSet<bool> bd_flags(fe_space->Mesh(), false);
  // Loop over all edges
  for (const auto& edge : fe_space->Mesh()->Entities(1)) {
    LF_ASSERT_MSG(edge->RefEl() == lf::base::RefEl::kSegment(),
                  "Entity should be an edge");
    const lf::geometry::Geometry* geo_ptr = edge->Geometry();
    const Eigen::MatrixXd corners = lf::geometry::Corners(*geo_ptr);
    // Check if the edge lies on $\Gamma_{\mathrm{in}}$  (geometric test)
    if ((corners(0, 0) + corners(0, 1)) / 2. > 1. - 1e-5 ||
        (corners(1, 0) + corners(1, 1)) / 2. < 1e-5) {
      // Add the edge to the flagged entities
      bd_flags(*edge) = true;
    }
  }
  // Loop over all Points
  for (const auto& point : fe_space->Mesh()->Entities(2)) {
    LF_ASSERT_MSG(point->RefEl() == lf::base::RefEl::kPoint(),
                  "Entity should be an edge");
    const lf::geometry::Geometry* geo_ptr = point->Geometry();
    const Eigen::VectorXd coords = lf::geometry::Corners(*geo_ptr);
    // Check if the node lies on  $\Gamma_{\mathrm{in}}$ (geometric test)
    if (coords(0) > 1. - 1e-5 || coords(1) < 1e-5) {
      // Add the edge to the flagged entities
      bd_flags(*point) = true;
    }
  }
  auto flag_values{lf::fe::InitEssentialConditionFromFunction(
      *fe_space, bd_flags, mf_g_Gamma_in)};

  lf::assemble::FixFlaggedSolutionCompAlt<double>(
      [&flag_values](lf::assemble::glb_idx_t dof_idx) {
        return flag_values[dof_idx];
      },
      A, b);
};

class BlendedParametrizationElementMatrixProvider {
 public:
  BlendedParametrizationElementMatrixProvider() = default;
  bool isActive(const lf::mesh::Entity &) { return true; }
  Eigen::MatrixXd Eval(const lf::mesh::Entity &cell){
    LF_ASSERT_MSG(cell.RefEl() == lf::base::RefEl::kTria(), "Only implemented for Triangles");
    auto geo_p = cell.Geometry();
    const Eigen::MatrixXd coords = lf::geometry::Corners(*geo_p);
    const BlendedParameterization::coord_t a0 = coords.col(0);
    const BlendedParameterization::coord_t a1 = coords.col(1);
    const BlendedParameterization::coord_t a2 = coords.col(2);

    const BlendedParameterization::Segment gamma01(a0, a1);
    const BlendedParameterization::Segment gamma12(a1, a2);
    const BlendedParameterization::Segment gamma20(a2, a0);

    return BlendedParameterization::evalBlendLocMat(gamma01, gamma12, gamma20);
  }
};

int main(int /*argc*/, char** /*argv*/) {
  // create a triangular grid on unit square
  int N = 100;
  BlendedParameterization::matrix_t elements = generateMesh(N);

  // iterate over all triangles
  for (int i = 0; i < elements.rows(); ++i) {
    BlendedParameterization::coord_t a0 = elements.row(i).head(2);
    BlendedParameterization::coord_t a1 = elements.row(i).segment(2, 2);
    BlendedParameterization::coord_t a2 = elements.row(i).tail(2);

    BlendedParameterization::Segment gamma01(a0, a1);
    BlendedParameterization::Segment gamma12(a1, a2);
    BlendedParameterization::Segment gamma20(a2, a0);

    BlendedParameterization::matrix_t lclMat =
        BlendedParameterization::evalBlendLocMat(gamma01, gamma12, gamma20);
  }

  // For testing purposes
  // read in mesh and set up finite element space
  auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
  lf::io::GmshReader reader(std::move(mesh_factory),
                            CURRENT_SOURCE_DIR "/../meshes/square.msh");
  auto mesh = reader.mesh();
  // obtain dofh for lagrangian finite element space
  auto fe_space =
      std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh);
  const lf::assemble::DofHandler &dofh{fe_space->LocGlobMap()};
  const int N_dofs = dofh.NumDofs();

  lf::assemble::COOMatrix<double> A(N_dofs, N_dofs);

  BlendedParametrizationElementMatrixProvider elmat_builder;
  lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, A);


  lf::assemble::COOMatrix<double> B(N_dofs, N_dofs);
  auto alpha = [](Eigen::Vector2d x) -> Eigen::Matrix2d { return Eigen::MatrixXd::Identity(2,2); };
  lf::mesh::utils::MeshFunctionGlobal mf_alpha{alpha};
  auto zero = [](Eigen::Vector2d x) -> double { return 0.; };
  lf::mesh::utils::MeshFunctionGlobal mf_zero{zero};
  // set up quadrature rule to be able to compare
  std::map<lf::base::RefEl, lf::quad::QuadRule> quad_rules{
      {lf::base::RefEl::kTria(), lf::quad::make_TriaQR_EdgeMidpointRule()},
      {lf::base::RefEl::kQuad(), lf::quad::make_QuadQR_EdgeMidpointRule()}};

  lf::uscalfe::ReactionDiffusionElementMatrixProvider<
      double, decltype(mf_alpha), decltype(mf_zero)>
      elmat_builder_org(fe_space, mf_alpha, mf_zero, quad_rules);
  lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder_org, B);


  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(N_dofs);
  enforce_boundary_conditions(fe_space , A , rhs);
  const Eigen::SparseMatrix<double> A_crs = A.makeSparse();
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(A_crs);
  Eigen::VectorXd u = solver.solve(rhs);
  const lf::fe::MeshFunctionFE mf_sol(fe_space, u);
  lf::io::VtkWriter vtk_writer(fe_space->Mesh(), "./param_solution.vtk");
  vtk_writer.WritePointData("solution", mf_sol);

  Eigen::VectorXd rhs_lf = Eigen::VectorXd::Zero(N_dofs);
  enforce_boundary_conditions(fe_space , B, rhs_lf);
  Eigen::SparseMatrix<double> B_crs = B.makeSparse();
  solver.compute(B_crs);
  Eigen::VectorXd u_lf = solver.solve(rhs_lf);
  const lf::fe::MeshFunctionFE mf_sol_lf(fe_space, u_lf);
  lf::io::VtkWriter vtk_writer_lf(fe_space->Mesh(), "./lf_solution.vtk");
  vtk_writer_lf.WritePointData("solution", mf_sol_lf);
  return 0;
}
