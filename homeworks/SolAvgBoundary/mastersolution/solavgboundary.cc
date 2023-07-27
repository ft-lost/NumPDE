/**
 * @file solavgboundary.cc
 * @brief NPDE homework Solavgboundary code
 * @author Bob Schreiner
 * @date June 2023
 * @copyright Developed at SAM, ETH Zurich
 */

#include "solavgboundary.h"

#include <lf/assemble/assemble.h>
#include <lf/assemble/dofhandler.h>
#include <lf/base/base.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <string>
namespace solavgboundary {

/**
 *
 * @param fe_spc piecewise linear Lagrangian finite element space
 * @param c of type Eigenm::VectorXd of dim(N_dofs)
 * @return This function returns the augemented finite element Galerkin matrix
 * in sparse format.
 */
/* SAM_LISTING_BEGIN_1 */
Eigen::SparseMatrix<double> augmentMatrix(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_spc,
    const Eigen::VectorXd& c) {
  const lf::assemble::DofHandler* dofh = &fe_spc->LocGlobMap();
  int N_dofs = dofh->NumDofs();
  lf::assemble::COOMatrix<double> A =
      lf::assemble::COOMatrix<double>(N_dofs + 1, N_dofs + 1);
  auto const_one = [](const Eigen::Vector2d& /*x*/) -> double { return 1.0; };
  auto const_zero = [](const Eigen::Vector2d& /*x*/) -> double { return 0.0; };
  A = compGalerkinMatrix(*dofh, const_one, const_zero, const_zero);
  for (int i = 0; i < N_dofs; ++i) {
    A.AddToEntry(N_dofs, i, c[i]);
    A.AddToEntry(i, N_dofs, c[i]);
  }

  return A.makeSparse();
}
/* SAM_LISTING_END_1 */

Eigen::VectorXd solveTestProblem(const lf::assemble::DofHandler& dofh) {
  // Set up load vector
  auto fe_space =
      std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(dofh.Mesh());
  const Eigen::VectorXd c = computeCVector(fe_space);
  const Eigen::SparseMatrix<double> A = augmentMatrix(fe_space, c);
  lf::mesh::utils::MeshFunctionConstant mf_identity{1.0};
  lf::uscalfe::ScalarLoadElementVectorProvider elvec_builder(fe_space,
                                                             mf_identity);
  // since we compute the augmented Problem, we add one entry to phi
  // representing eta
  Eigen::VectorXd phi(dofh.NumDofs() + 1);
  phi.setZero();
  AssembleVectorLocally(0, dofh, elvec_builder, phi);

  // Solve system of linear equations
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(A);
  // Truncate solution to only return mu
  Eigen::VectorXd mu = solver.solve(phi).head(dofh.NumDofs());

  return mu;
}

void visSolution(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fe_space,
    Eigen::VectorXd& u, const std::string&& filename) {
  const lf::fe::MeshFunctionFE mf_sol(fe_space, u);
  lf::io::VtkWriter vtk_writer(fe_space->Mesh(), filename);
  vtk_writer.WritePointData("solution", mf_sol);
}
}  // namespace solavgboundary
